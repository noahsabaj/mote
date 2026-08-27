"""Direct Preference Optimization on chat pairs (Rafailov et al. 2023) for a byte-level model.

    python -m mote.train.dpo --init-from runs/overnight_sft/last.pt --pairs data/sft_identity.dpo.jsonl \
        --out runs/overnight_dpo --epochs 3 --lr 2e-6 --beta 0.1

Each pair is a conversation context plus a chosen and a rejected final assistant reply. With the SFT
model frozen as the reference π_ref, the loss per pair is
    -log σ( β · [ (log π(chosen) − log π_ref(chosen)) − (log π(rejected) − log π_ref(rejected)) ] )
where log π(·) sums the next-byte log-probabilities over the final reply's bytes (mask = that reply + EOS).
Small learning rate, few epochs: the goal is to move the *choice* between hold and cave, not the language.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from ..config import MoteConfig
from ..model.hnet import HNetForCausalLM
from ..tokenizer import ByteTokenizer, ChatMessage


def pair_messages(pair: Dict) -> List[Dict]:
    """The conversation context, from either schema.

    build_identity writes {"messages": [...]}; mote.sim.generate writes {"prompt": "..."} — a single user
    turn. dpo.py only ever read "messages", so the sim's 20k verifiable pairs could not be loaded at all
    (found 2026-08-25, before the stage they are for had ever run)."""
    if "messages" in pair:
        return list(pair["messages"])
    return [{"role": "user", "content": pair["prompt"]}]


def render_pair(tok: ByteTokenizer, pair: Dict, max_len: int):
    """-> (ids, mask) for chosen and rejected (masks = 1 on the final reply's bytes + EOS)."""
    ctx = [ChatMessage(m["role"], m["content"]) for m in pair_messages(pair)]
    out = []
    for reply in (pair["chosen"], pair["rejected"]):
        ids, mask = tok.format_chat_with_loss_mask(ctx + [ChatMessage("assistant", reply)])
        # only the final reply trains: zero every earlier assistant turn
        last_on = max(i for i, m in enumerate(mask) if m)
        first_on = last_on
        while first_on > 0 and mask[first_on - 1]:
            first_on -= 1
        mask = [1 if first_on <= i <= last_on else 0 for i in range(len(mask))]
        out.append((ids[:max_len], mask[:max_len]))
    return out


def diff_weights(chosen: str, rejected: str, n_reply: int, a_diff: float, a_shared: float) -> List[float]:
    """Per-byte weights over the REJECTED reply: `a_diff` on the bytes that differ from the chosen reply,
    `a_shared` on the ones both share (TD-DPO 2607.18304 Eq. 8; their optimum is 2.0 / 0.5).

    The point is that a preference can turn on a short span inside an otherwise fine reply, and summing
    the log-probs uniformly spreads the gradient over the shared background too. Weighting only the
    rejected side is their default and beats the symmetric and chosen-only variants (their Table 8);
    driving `a_shared` to 0 *hurts*, because the shared bytes are the contextual anchor.

    Only meaningful when the pair really is a near-edit — on the swap pairs the differing span is a couple
    of bytes out of fifty. On a pair that shares little, nearly everything is marked different and this
    degenerates into a constant scale, which is why it is off by default."""
    import difflib

    keep = [a_shared] * len(rejected)
    sm = difflib.SequenceMatcher(None, chosen, rejected, autojunk=False)
    for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            for j in range(j1, min(j2, len(rejected))):
                keep[j] = a_diff
    # the reply sits at the END of the masked span; pad/trim to the masked byte count
    if len(keep) >= n_reply:
        return keep[-n_reply:]
    return keep + [a_shared] * (n_reply - len(keep))


def seq_logprob(model, ids: torch.Tensor, mask: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Sum of next-byte log-probs over masked target positions. ids [B,L], mask [B,L] (target positions)."""
    inputs, targets = ids[:, :-1], ids[:, 1:]
    tmask = mask[:, 1:].float()
    logits = model(inputs).logits.float()
    lp = -F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none").view(targets.shape)
    if weights is not None:
        tmask = tmask * weights[:, 1:]
    return (lp * tmask).sum(-1)


def pad_batch(items: List[tuple], pad_id: int, device):
    L = max(len(ids) for ids, _ in items)
    ids = torch.full((len(items), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(items), L), dtype=torch.long)
    for i, (x, m) in enumerate(items):
        ids[i, : len(x)] = torch.tensor(x)
        mask[i, : len(m)] = torch.tensor(m)
    return ids.to(device), mask.to(device)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-from", required=True, help="SFT checkpoint (policy init and frozen reference)")
    ap.add_argument("--pairs", required=True, help="JSONL of {messages, chosen, rejected}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--objective", default="dpo", choices=["dpo", "ipo", "orpo"],
                    help="dpo: -log sigma(margin), no finite optimum, so on deterministic preferences the margin runs "
                         "away (overnight_dpo hit 7.88 and the text degraded). ipo (Azar et al. 2023): (margin - 1)^2, "
                         "optimum at margin = 1 by construction — the regulariser DPO lacks, for exactly this case. "
                         "orpo (Hong et al. 2024): reference-free, single stage — NLL of the chosen reply plus a "
                         "length-normalised odds-ratio contrast, so --init-from is the pretrained base, not an SFT "
                         "checkpoint, and no reference model is loaded.")
    ap.add_argument("--orpo-lambda", type=float, default=1.0, help="weight of the odds-ratio term in --objective orpo")
    ap.add_argument("--length-norm", action="store_true",
                    help="divide each side's log-prob by its byte count before the margin (what SimPO and ORPO do). "
                         "Length predicted the label in 400/400 pushback pairs before 2026-08-25; the data is fixed "
                         "now, but this makes the objective itself immune to any skew that comes back")
    ap.add_argument("--td-diff", type=float, default=0.0,
                    help="TD-DPO (2607.18304): weight on the REJECTED reply's bytes that differ from the chosen one. "
                         "0 disables the weighting entirely; their optimum is 2.0. Only meaningful on near-edit pairs "
                         "(the `swap` kind) — on pairs that share little it degenerates to a constant scale")
    ap.add_argument("--td-shared", type=float, default=0.5, help="TD-DPO weight on the shared bytes; 0 hurts (their §5.4)")
    ap.add_argument("--sft-weight", type=float, default=0.0, help="add this much of the chosen replies' mean NLL per byte to the DPO loss (the first attempt, 3 epochs at 2e-6 without it, reached margin 8.9 and degraded the text)")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(args.init_from, map_location="cpu", weights_only=True)
    cfg = MoteConfig.from_dict(ck["config"])
    policy = HNetForCausalLM(cfg, device=device)
    policy.load_state_dict(ck["model"])
    ref = None
    if args.objective != "orpo":  # ORPO is reference-free; there is nothing to anchor to
        ref = HNetForCausalLM(cfg, device=device)
        ref.load_state_dict(ck["model"])
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)
    tok = ByteTokenizer()
    pairs = [json.loads(l) for l in open(args.pairs, encoding="utf-8") if l.strip()]
    # (chosen, rejected, per-byte weights over the rejected reply or None) — kept as one tuple so the
    # weights survive the shuffle with their pair.
    rendered = []
    for pr in pairs:
        ch, rj = render_pair(tok, pr, args.max_len)
        w = diff_weights(pr["chosen"], pr["rejected"], sum(rj[1]), args.td_diff, args.td_shared) if args.td_diff > 0 else None
        rendered.append((ch, rj, w))
    if args.td_diff > 0:
        share = sum(sum(1 for x in w if x == args.td_diff) / max(len(w), 1) for _c, _r, w in rendered) / len(rendered)
        print(json.dumps({"td_dpo": {"diff": args.td_diff, "shared": args.td_shared,
                                     "mean_diff_share": round(share, 3),
                                     "note": "a diff share near 1.0 means the pairs are not near-edits and the weighting is a constant scale"}}), flush=True)
    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    log_f = open(out / "log.jsonl", "a", encoding="utf-8")
    step, t0 = 0, time.time()
    pad_id = tok.pad_id if hasattr(tok, "pad_id") else 258

    def batch_logps(model, items, weights=None):
        ids, mask = pad_batch(items, pad_id, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            return seq_logprob(model, ids, mask, weights)

    for epoch in range(args.epochs):
        random.shuffle(rendered)
        for i in range(0, len(rendered), args.batch_size):
            chunk = rendered[i : i + args.batch_size]
            chosen = [c for c, _r, _w in chunk]
            rejected = [r for _c, r, _w in chunk]
            rej_w = None
            if args.td_diff > 0:
                L = max(len(ids) for ids, _ in rejected)
                rej_w = torch.ones((len(chunk), L), dtype=torch.float32, device=device)
                for bi, (_c, (ids, mask), w) in enumerate(chunk):
                    on = [j for j, m in enumerate(mask) if m]
                    for j, wt in zip(on, w):
                        rej_w[bi, j] = wt
            policy.train()
            lp_c = batch_logps(policy, chosen)
            lp_r = batch_logps(policy, rejected, rej_w)  # TD-DPO weights the REJECTED side only
            n_c = torch.tensor([float(sum(m)) for _, m in chosen], device=device).clamp_min(1)
            n_r = torch.tensor([float(sum(m)) for _, m in rejected], device=device).clamp_min(1)
            if args.objective == "orpo":
                # Length-normalised log-odds; log(1-p) via log1p keeps it stable when p is tiny.
                def log_odds(lp, n):
                    q = (lp / n).clamp(max=-1e-6)
                    return q - torch.log1p(-q.exp())
                margin = log_odds(lp_c, n_c) - log_odds(lp_r, n_r)
                loss = -(lp_c / n_c).mean() + args.orpo_lambda * (-F.logsigmoid(margin)).mean()
            else:
                with torch.no_grad():
                    rlp_c = batch_logps(ref, chosen)
                    rlp_r = batch_logps(ref, rejected, rej_w)
                if args.length_norm:  # SimPO/ORPO's defence against length hacking, opt-in for DPO and IPO
                    margin = args.beta * (((lp_c - rlp_c) / n_c) - ((lp_r - rlp_r) / n_r))
                else:
                    margin = args.beta * ((lp_c - rlp_c) - (lp_r - rlp_r))
                # Both are Psi-PO with the same margin; only Psi differs (2601.06108 Thm 3.2).
                loss = ((margin - 1.0) ** 2).mean() if args.objective == "ipo" else -F.logsigmoid(margin).mean()
                if args.sft_weight > 0:
                    loss = loss + args.sft_weight * (-(lp_c / n_c).mean())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            step += 1
            acc = float((margin > 0).float().mean())
            rec = {"epoch": epoch, "step": step, "loss": float(loss), "pref_acc": acc, "margin": float(margin.mean()), "elapsed_min": (time.time() - t0) / 60}
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            if step % 5 == 0:
                print(json.dumps(rec), flush=True)
    ck_out = {"model": policy.state_dict(), "optimizer": None, "step": int(ck.get("step", 0)) + step, "config": cfg.to_dict(),
              "extra": {**ck.get("extra", {}), "dpo": {"objective": args.objective, "orpo_lambda": args.orpo_lambda, "pairs": len(pairs), "epochs": args.epochs, "beta": args.beta, "lr": args.lr, "sft_weight": args.sft_weight, "init_from": args.init_from}}}
    tmp = out / "last.tmp"
    torch.save(ck_out, tmp)
    os.replace(tmp, out / "last.pt")
    log_f.write(json.dumps({"done": True, "final_step": step}) + "\n")
    print(json.dumps({"done": True, "steps": step, "out": str(out / "last.pt")}))


if __name__ == "__main__":
    main()
