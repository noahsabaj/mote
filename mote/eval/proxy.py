"""Proxy metrics over expert trajectories — the mid-training gate's decider (2605.18607).

    python -m mote.eval.proxy --checkpoint runs/branch_anneal_sft/last.pt [--n-sim 120] [--n-read 120]

Why this replaced exact match. The gate used to vote on reading EM, sim-QA EM and chat val bpb, and two
of those three cannot discriminate at Mote's scale: the 35M model scores a flat 0 on reading
(docs/search.md), so the vote is a coin flip dressed as a measurement. 2605.18607 measures exactly this
gap — cross-entropy ranks models at Spearman 0.36 against downstream truth while token-level proxies over
expert trajectories reach 0.81 — and validates the method on DataDecide, which is the same problem this
gate has: rank candidate corpora before committing to the expensive run. Their explanation is the reason
it works here: *a model which cannot solve a problem can still track the trajectory an expert wrote.*
Compatibility with the expert varies measurably between corpora long before any accuracy score leaves
the noise floor.

The byte-level adaptation. Their headline metric is frequency-weighted top-5 accuracy over a ~100k
vocabulary, where top-5 is a 0.005 % baseline. Mote's vocabulary is 269, where top-5 is 1.9 % and
saturates. Two changes follow:

  * **reciprocal rank**, not top-5 accuracy. A threshold metric is binary and sparse at a 100k
    vocabulary, which is what their weighting schemes exist to compensate for; at 269 ids the rank of the
    expert's byte is already graded at every position, so the compensation is unnecessary and the metric
    does not saturate at any vocabulary size.
  * **no weighting**, which is the opposite of what the first two attempts assumed.

The choice is measured, not argued. Three checkpoints whose quality order is known from 12-hour runs
(val_bpb_ema 1.0276 / 1.0370 / 1.0800), scored on n=120 held-out sim trajectories, `spread/sem` being the
best-to-worst gap in standard errors of the per-item mean:

    metric                      4e-4      8e-4     16e-4  order  spread/sem
    recip_rank_uniform        0.4655    0.4476    0.4180   YES      2.3x   <- the decider
    top2                      0.4528    0.4226    0.3890   YES      2.6x
    recip_rank (inv-freq)     0.4626    0.4454    0.4163   YES      2.2x
    agree_freq                0.4231    0.3735    0.3673   YES      2.1x
    agree_uniform             0.3436    0.3277    0.3236   YES      0.9x
    agree (inv-freq top-1)    0.3407    0.3259    0.3219   YES      0.8x
    agree_entropy             0.2235    0.2581    0.2271   no       0.2x
    recip_rank_entropy        0.3492    0.3873    0.3308   no       1.2x
    ce                        2.9538    3.1715    3.6565   YES      5.9x

Two things to read off it. **Every entropy-weighted cell gets the order wrong and every model-independent
one gets it right** — entropy is the *candidate's own* uncertainty, so a worse checkpoint is more uncertain
everywhere and weighting by it re-normalises away the difference being measured. That was the first
attempt's bug, and it is why the paper's winning cell is *frequency*-weighted: a property of the expert's
text rather than of the candidate. And **cross-entropy separates these three best of all**, which is not a
contradiction: 2605.18607's 0.36 is for ranking across model families, and these three differ only in
learning rate. The gate's comparison is harder than this one — two branches trained on *different
mixtures*, so their val distributions differ and val bpb stops being comparable — which is why `ce` stays
in the table and `recip_rank_uniform` decides.

The whole library is reported so a later arm can be re-scored against a different cell without re-running
the model, and so the table above stays checkable rather than being a claim in a docstring.

One forward pass per item, no generation, so this is the cheapest thing in the gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from ..tokenizer import VOCAB_SIZE, ByteTokenizer, ChatMessage
from .val_bpb import load_model


def byte_frequency(items: List[Dict], vocab: int = VOCAB_SIZE) -> torch.Tensor:
    """Relative frequency of each byte across the expert replies — the weighting the metric turns on.

    Model-independent by construction, which is the whole point: it says which positions are informative
    from the corpus's side, not from the candidate's."""
    counts = torch.ones(vocab)  # add-one, so an unseen byte is rare rather than a division by zero
    for it in items:
        for b in (it.get("gold") or "").encode("utf-8"):
            if b < vocab:
                counts[b] += 1
    return counts / counts.sum()


def trajectory_stats(model, ids: List[int], reply_from: int, device, vocab: int = VOCAB_SIZE,
                     freq: Optional[torch.Tensor] = None) -> Optional[Dict[str, float]]:
    """Token-level statistics of `model`'s next-byte distribution over one expert trajectory.

    `reply_from` is the index at which the expert's own bytes start; the prompt conditions but is not
    scored. Returns None when the reply is empty. 2605.18607 scores the last 1,000 tokens of a trajectory
    rather than all of it (§3, 'empirically outperforms using the full trace'); Mote's replies are shorter
    than that throughout, so every reply byte is scored.
    """
    if reply_from >= len(ids) - 1:
        return None
    x = torch.tensor(ids[:-1], dtype=torch.long, device=device)[None]
    gold = torch.tensor(ids[1:], dtype=torch.long, device=device)
    with torch.no_grad():
        # the checkpoint's own vocab, not this module's: the head masks columns >= cfg.vocab_size to -inf
        # (config.py, so a padding row can never be sampled), and slicing to a larger number would pull
        # those -inf columns in. p * log p is then 0 * -inf = nan, which silently NaNs every weighted mean.
        logits = model(x).logits[0].float()[:, :vocab]
    sl = slice(reply_from - 1, None)  # position i predicts ids[i+1], so the first reply byte is at i-1
    logits, gold = logits[sl], gold[sl]
    if not len(gold):
        return None
    logp = F.log_softmax(logits, -1)
    p = logp.exp()
    ce = -logp.gather(1, gold[:, None]).squeeze(1)
    plogp = torch.where(p > 0, p * logp, torch.zeros_like(p))  # 0 log 0 = 0, and never 0 * -inf = nan
    ent = -plogp.sum(-1) / float(torch.log(torch.tensor(float(vocab))))  # normalised to [0, 1]
    rank = (logits > logits.gather(1, gold[:, None])).sum(-1) + 1
    pg = p.gather(1, gold[:, None]).squeeze(1)
    top1 = (rank == 1).float()

    def wmean(v: torch.Tensor, w: torch.Tensor) -> float:
        s = w.sum()
        return float((v * w).sum() / s) if float(s) > 0 else float(v.mean())

    ones = torch.ones_like(ent)
    assert torch.isfinite(ent).all(), "entropy went non-finite: a masked padding column leaked in"
    # inverse frequency of the *expert's* byte at each position: a rare byte is where tracking the expert
    # says something, a space is not. Model-independent, unlike the entropy weighting this replaced.
    if freq is not None:
        fw = 1.0 - freq.to(gold.device)[gold.clamp(max=len(freq) - 1)]
        fq = freq.to(gold.device)[gold.clamp(max=len(freq) - 1)]
    else:
        fw = fq = ones
    rr = 1.0 / rank.float()
    return {
        # the decider: mean reciprocal rank of the expert's byte, unweighted
        "recip_rank_uniform": wmean(rr, ones),
        "agree": wmean(top1, fw),
        # the rest of the library, reported so an arm can be re-scored without re-running the model
        "agree_uniform": wmean(top1, ones),
        "agree_freq": wmean(top1, fq),
        "agree_entropy": wmean(top1, ent),        # measured 2026-08-26 to rank wrongly; kept as the record
        "agree_disagree_w": wmean(top1, 1.0 - pg),
        "recip_rank": wmean(rr, fw),
        "recip_rank_entropy": wmean(rr, ent),
        "top2": wmean((rank <= 2).float(), fw),
        "entropy": wmean(ent, ones),
        "ce": wmean(ce, ones),           # the rho=0.36 signal, kept for the comparison
        "margin": wmean(p.max(-1).values - pg, fw),
        "n_bytes": float(len(gold)),
    }


def _chat_ids(tok: ByteTokenizer, prompt: str, reply: str, card: Optional[str]) -> tuple[List[int], int]:
    msgs = ([ChatMessage("system", card)] if card else []) + [ChatMessage("user", prompt)]
    prefix = tok.format_chat(msgs, add_generation_prompt=True)
    return prefix + tok.encode(reply), len(prefix)


def run(checkpoint: str | Path, items: List[Dict], device, card: Optional[str] = None) -> Dict:
    """`items`: {"prompt", "gold", ...}. Scores the model against each gold reply as an expert trajectory."""
    model, cfg, step = load_model(checkpoint, device)
    tok = ByteTokenizer()
    freq = byte_frequency(items, cfg.vocab_size)
    rows, keys = [], None
    for it in items:
        ids, reply_from = _chat_ids(tok, it["prompt"], it["gold"], card)
        if len(ids) > cfg.max_seq_len:
            continue
        st = trajectory_stats(model, ids, reply_from, device, cfg.vocab_size, freq)
        if st is None:
            continue
        keys = keys or [k for k in st if k != "n_bytes"]
        rows.append({**{k: it[k] for k in it if k in ("source", "domain", "locale", "qtype")}, **st})
    if not rows:
        return {"checkpoint": str(checkpoint), "step": step, "n": 0}
    import statistics

    out: Dict = {"checkpoint": str(checkpoint), "step": step, "n": len(rows),
                 **{k: sum(r[k] for r in rows) / len(rows) for k in keys}}
    # the standard error of each mean. A decider without one is a decorative signal: at n=120 the gap
    # between the best and worst of three known-different checkpoints is 2.3 sem, so a branch comparison
    # that comes in under 1 sem has not decided anything and the gate says so.
    out.update({f"{k}_sem": (statistics.stdev([r[k] for r in rows]) / len(rows) ** 0.5) if len(rows) > 1 else 0.0
                for k in keys})
    by_src: Dict[str, List[float]] = {}
    for r in rows:
        by_src.setdefault(r.get("source", "all"), []).append(r["recip_rank_uniform"])
    out["per_source"] = {s: sum(v) / len(v) for s, v in sorted(by_src.items())}
    out["rows"] = rows
    return out


def gather_items(n_sim: int = 120, n_read: int = 120, locales: str = "en,ru,ja",
                 seed_base: Optional[int] = None) -> List[Dict]:
    """The expert trajectories: held-out sim answers and reading-comprehension references.

    Both are held out from training by construction — the sim probe draws worlds from seeds far above the
    generator's, and the reading items come from SQuAD's validation split. A gold answer is short, which
    is the point: it is the span where the model either tracks the expert or does not."""
    items: List[Dict] = []
    if n_sim:
        from .sim_probe import SEED_BASE, heldout_items

        for it in heldout_items(n_sim, [l.strip() for l in locales.split(",") if l.strip()], seed_base or SEED_BASE):
            items.append({"prompt": it["prompt"], "gold": it["gold"], "source": "sim",
                          "domain": it["domain"], "locale": it["locale"], "qtype": it["qtype"]})
    if n_read:
        from .read_probe import load_items, prompt_for

        for it in load_items(n_read, 0):
            if it["answers"]:
                items.append({"prompt": prompt_for(it), "gold": it["answers"][0], "source": "reading"})
    return items


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--n-sim", type=int, default=120)
    ap.add_argument("--n-read", type=int, default=120)
    ap.add_argument("--locales", default="en,ru,ja")
    ap.add_argument("--no-card", action="store_true", help="score without the identity system message")
    ap.add_argument("--out", default=None, help="default: proxy.json next to the checkpoint")
    args = ap.parse_args(argv)

    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    items = gather_items(args.n_sim, args.n_read, args.locales)
    card = None
    if not args.no_card:
        from ..config import MoteConfig  # noqa: F401  (load_model reads it; card needs the param count)
        from ..serve.identity import identity_card

        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        card = identity_card(sum(v.numel() for v in ck["model"].values()))
    res = run(args.checkpoint, items, dev, card)
    out = Path(args.out) if args.out else Path(args.checkpoint).parent / "proxy.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"recip_rank_uniform {res.get('recip_rank_uniform', float('nan')):.4f} "
          f"+/- {res.get('recip_rank_uniform_sem', float('nan')):.4f}  "
          f"(agree {res.get('agree', float('nan')):.4f}, ce {res.get('ce', float('nan')):.4f})  "
          f"n={res.get('n', 0)} -> {out}")
    for s, v in (res.get("per_source") or {}).items():
        print(f"  {s:10s} {v:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
