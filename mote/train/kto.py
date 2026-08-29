"""KTO on single replies marked good or bad (Ethayarajh et al. 2024) — the objective the prefs loop needs.

    python -m mote.train.kto --init-from runs/flagship_sft/last.pt --marks data/prefs/marks.export.jsonl \
        --out runs/flagship_kto --epochs 1 --lr 5e-7 --beta 0.1

DPO needs a *pair*: two replies to one prompt, one better. Collecting those means generating twice and
asking for a comparison, which is why data/prefs held 2 votes after two days against a 1000-pair gate.
KTO needs only "this reply was good" or "this reply was bad" — one thumb on a reply you were already
reading — so every mark is an example and nothing has to find a partner. A `both_bad` vote, which the
pairwise exporter drops on the floor, is two undesirable examples here.

Each example is {"messages": [...context...], "reply": "...", "label": "good" | "bad"}. With
r(x,y) = log π(y|x) − log π_ref(y|x) and z a batch estimate of KL(π ‖ π_ref):

    desirable    L = λ_D · (1 − σ( β·(r − z) ))
    undesirable  L = λ_U · (1 − σ( β·(z − r) ))

The value function is the prospect-theoretic one: a reply is judged against how far the *whole policy* has
drifted (z), not against a sibling reply, and the two λ weights carry the loss aversion. z is estimated
from mismatched (x_i, y_j) pairs in the batch and carries no gradient — that is Ethayarajh's construction,
and it is what makes the objective work without pairs.

Imbalanced marks are expected (people click "bad" more readily), so λ_D and λ_U default to the paper's
guidance: keep λ_D·n_D / (λ_U·n_U) in [1, 4/3]. --auto-lambda sets them from the actual counts.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from ..config import MoteConfig
from ..model.hnet import HNetForCausalLM, strip_retired
from ..tokenizer import PAD_ID, ByteTokenizer, ChatMessage
from .dpo import pad_batch, seq_logprob
from .train import save_checkpoint


def render_mark(tok: ByteTokenizer, ex: Dict, max_len: int) -> Tuple[List[int], List[int]]:
    """-> ids, mask with the mask on the final reply's bytes only (every earlier assistant turn zeroed)."""
    ctx = [ChatMessage(m["role"], m["content"]) for m in ex["messages"]]
    ids, mask = tok.format_chat_with_loss_mask(ctx + [ChatMessage("assistant", ex["reply"])])
    last_on = max(i for i, m in enumerate(mask) if m)
    first_on = last_on
    while first_on > 0 and mask[first_on - 1]:
        first_on -= 1
    mask = [1 if first_on <= i <= last_on else 0 for i in range(len(mask))]
    return ids[:max_len], mask[:max_len]


def kto_loss(logratio: torch.Tensor, desirable: torch.Tensor, z: torch.Tensor,
             beta: float, lam_d: float, lam_u: float) -> torch.Tensor:
    """Per-example KTO loss. `desirable` is a bool mask, `z` the detached KL estimate."""
    good = lam_d * (1.0 - torch.sigmoid(beta * (logratio - z)))
    bad = lam_u * (1.0 - torch.sigmoid(beta * (z - logratio)))
    return torch.where(desirable, good, bad)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-from", required=True, help="SFT checkpoint: policy init and frozen reference")
    ap.add_argument("--marks", required=True, help='JSONL of {messages, reply, label: "good"|"bad"}')
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-7)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lam-d", type=float, default=1.0, help="weight on desirable examples")
    ap.add_argument("--lam-u", type=float, default=1.0, help="weight on undesirable examples")
    ap.add_argument("--auto-lambda", action="store_true",
                    help="set lam_d/lam_u from the mark counts so lam_d*n_d / (lam_u*n_u) = 1 (the paper's balance)")
    ap.add_argument("--sft-weight", type=float, default=0.0,
                    help="add this much of the *good* replies' mean NLL per byte, the anchor that kept overnight_dpo2 readable")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(args.init_from, map_location="cpu", weights_only=True)
    cfg = MoteConfig.from_dict(ck["config"])
    policy = HNetForCausalLM(cfg, device=device)
    policy.load_state_dict(strip_retired(ck["model"]))
    ref = HNetForCausalLM(cfg, device=device)
    ref.load_state_dict(strip_retired(ck["model"]))
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    tok = ByteTokenizer()
    marks = [json.loads(l) for l in open(args.marks, encoding="utf-8") if l.strip()]
    if not marks:
        raise SystemExit(f"{args.marks} has no examples")
    rendered = [(render_mark(tok, m, args.max_len), m["label"] == "good") for m in marks]
    n_d = sum(1 for _, g in rendered if g)
    n_u = len(rendered) - n_d
    lam_d, lam_u = args.lam_d, args.lam_u
    if args.auto_lambda and n_d and n_u:
        # keep lam_d*n_d == lam_u*n_u: whichever class is rarer is weighted up
        lam_d, lam_u = (1.0, n_d / n_u) if n_d >= n_u else (n_u / n_d, 1.0)
    print(json.dumps({"examples": len(rendered), "good": n_d, "bad": n_u, "lam_d": lam_d, "lam_u": lam_u}), flush=True)

    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    log_f = open(out / "log.jsonl", "a", encoding="utf-8")
    step, t0 = 0, time.time()
    pad_id = PAD_ID

    def logps(model, items):
        ids, mask = pad_batch(items, pad_id, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            return seq_logprob(model, ids, mask)

    for epoch in range(args.epochs):
        random.shuffle(rendered)
        for i in range(0, len(rendered), args.batch_size):
            chunk = rendered[i : i + args.batch_size]
            if len(chunk) < 2:
                continue  # the KL estimate needs a mismatched partner
            items = [it for it, _ in chunk]
            desirable = torch.tensor([g for _, g in chunk], dtype=torch.bool, device=device)
            policy.train()
            lp = logps(policy, items)
            with torch.no_grad():
                rlp = logps(ref, items)
                # z = KL(pi || pi_ref) estimated on MISMATCHED (x_i, y_j) pairs — roll the replies by one so
                # every context is scored against someone else's reply. Detached, clamped at 0.
                rolled = items[1:] + items[:1]
                z = (logps(policy, rolled) - logps(ref, rolled)).mean().clamp_min(0.0)
            logratio = lp - rlp
            loss_vec = kto_loss(logratio, desirable, z, args.beta, lam_d, lam_u)
            loss = loss_vec.mean()
            n_tok = torch.tensor([float(sum(m)) for _, m in items], device=device).clamp_min(1)
            if args.sft_weight > 0 and bool(desirable.any()):
                loss = loss + args.sft_weight * (-(lp / n_tok)[desirable].mean())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            step += 1
            rec = {"epoch": epoch, "step": step, "loss": float(loss), "kl_z": float(z),
                   "logratio_good": float(logratio[desirable].mean()) if bool(desirable.any()) else None,
                   "logratio_bad": float(logratio[~desirable].mean()) if bool((~desirable).any()) else None,
                   "elapsed_min": (time.time() - t0) / 60}
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            if step % 5 == 0:
                print(json.dumps(rec), flush=True)

    save_checkpoint(out / "last.pt", policy, None, int(ck.get("step", 0)) + step, cfg,
                    {**ck.get("extra", {}),
                     "kto": {"examples": len(rendered), "good": n_d, "bad": n_u, "epochs": args.epochs,
                             "beta": args.beta, "lr": args.lr, "lam_d": lam_d, "lam_u": lam_u,
                             "sft_weight": args.sft_weight, "init_from": args.init_from}})
    log_f.write(json.dumps({"done": True, "final_step": step}) + "\n")
    log_f.close()
    print(json.dumps({"done": True, "steps": step, "out": str(out / "last.pt")}))


if __name__ == "__main__":
    main()
