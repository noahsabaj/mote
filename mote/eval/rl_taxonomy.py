"""How post-training moved the sim action policy, state by state (2607.16097 § 4.1, Table 14; signed 2026-08-24).

    python -m mote.eval.rl_taxonomy --before runs/sft1/last.pt --after runs/rlvr1/last.pt [--n 60] [--k 3] [--device cpu]

For every held-out task (mote.sim.tasks.heldout_tasks) and every step of its expert line, the model's policy over
the *legal* actions of that state is the renormalised likelihood of each action's canonical call text
(`<|call|>sim: <action><|result|>`) given the prompt and the expert's earlier calls and observations — the same
bytes the RL rollouts write. With the expert action a* as ground truth and top-k membership between the `before`
and `after` policies, each state falls into one category:

  gt_amplification   a* in the top-k of both and its probability grew
  tail_discovery     a* enters the top-k from below eps_tail (0.05)
  topk_correction    a* enters the top-k from >= eps_tail
  gt_regression      a* drops out of the top-k
  wrong_mode_amp     a* outside the top-k both times and before's top-1 (wrong) gained probability
  other              everything else

Difficulty bin = the expert line's length (1..5 actions). The paper's finding to watch for: easy states are
mostly amplification; hard states show tail discovery *and* wrong-mode amplification, which is why pass@k does not
move. A single checkpoint (`--before` only) reports the expert action's rank and probability per bin instead.
Writes rl_taxonomy.json next to the `after` (or `before`) checkpoint.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from ..serve.identity import with_system_card
from ..sim.tasks import SimEnv, Task, _cal_titles, action_text, heldout_tasks, legal_actions
from ..tokenizer import ASSISTANT_ID, CALL_ID, PAD_ID, RESULT_ID, ByteTokenizer, ChatMessage
from .val_bpb import load_model

CATEGORIES = ("gt_amplification", "tail_discovery", "topk_correction", "gt_regression", "wrong_mode_amp", "other")


def categorize(p0: Sequence[float], p1: Sequence[float], gt: int, k: int = 3, eps_tail: float = 0.05) -> str:
    """Table 14 of 2607.16097 for one state: `p0`/`p1` = before/after policies over the legal actions, `gt` = a*."""
    top0 = set(sorted(range(len(p0)), key=lambda i: -p0[i])[:k])
    top1 = set(sorted(range(len(p1)), key=lambda i: -p1[i])[:k])
    if gt in top0 and gt in top1:
        return "gt_amplification" if p1[gt] > p0[gt] else "other"
    if gt not in top0 and gt in top1:
        return "tail_discovery" if p0[gt] < eps_tail else "topk_correction"
    if gt in top0 and gt not in top1:
        return "gt_regression"
    w0 = max(range(len(p0)), key=lambda i: p0[i])  # before's wrong top-1
    if w0 in top1 and p1[w0] > p0[w0]:
        return "wrong_mode_amp"
    return "other"


def prefix_ids(tok: ByteTokenizer, task: Task, n_params: int, steps: Sequence[Tuple[str, str]]) -> List[int]:
    """The bytes a policy has read before choosing the action after `steps` = [(call text, observation), ...]:
    the chat prompt with the identity card, then each earlier call and its result the way the engine injects them."""
    msgs = with_system_card([{"role": "user", "content": task.prompt}], n_params)
    ids = tok.format_chat([ChatMessage(m["role"], m["content"]) for m in msgs], add_generation_prompt=True)
    for call, obs in steps:
        ids += [CALL_ID] + list(f"sim: {call}".encode("utf-8")) + [RESULT_ID] + list(obs.encode("utf-8")) + [ASSISTANT_ID]
    return ids


def state_walk(task: Task) -> List[Dict]:
    """Every decision state along the expert line: (steps so far, legal action texts, the expert's choice)."""
    env = SimEnv(task)
    out: List[Dict] = []
    try:
        steps: List[Tuple[str, str]] = []
        for target in task.expert:
            legal = [action_text(a, _cal_titles(env.world)) for a in legal_actions(task.domain, env.world, env.init)]
            legal = sorted(set(legal))
            if target not in legal:  # the expert's text always parses to a legal action; guard anyway
                legal.append(target)
            out.append({"steps": list(steps), "legal": legal, "gt": legal.index(target)})
            steps.append((target, env.act(target)))
    finally:
        env.close()
    return out


@torch.no_grad()
def action_logprobs(model, prefix: List[int], actions: Sequence[str], device, batch: int = 32) -> torch.Tensor:
    """Sum log p(<|call|>sim: action<|result|> | prefix) for every action: [n_actions]."""
    conts = [[CALL_ID] + list(f"sim: {a}".encode("utf-8")) + [RESULT_ID] for a in actions]
    out = torch.zeros(len(conts))
    for b0 in range(0, len(conts), batch):
        part = conts[b0:b0 + batch]
        L = len(prefix) + max(len(c) for c in part)
        ids = torch.full((len(part), L), PAD_ID, dtype=torch.long)
        for i, c in enumerate(part):
            ids[i, : len(prefix) + len(c)] = torch.tensor(prefix + c)
        ids = ids.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(ids[:, :-1]).logits
        logp = F.log_softmax(logits.float(), dim=-1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        for i, c in enumerate(part):
            s, e = len(prefix) - 1, len(prefix) - 1 + len(c)  # positions predicting the continuation bytes
            out[b0 + i] = logp[i, s:e].sum()
    return out


def policies(model, tok: ByteTokenizer, tasks: Sequence[Task], device, batch: int = 32) -> List[List[Dict]]:
    """Per task, per state: the policy over legal actions (softmax of the summed log-likelihoods)."""
    n_params = model.num_params()
    res: List[List[Dict]] = []
    for task in tasks:
        rows = []
        for st in state_walk(task):
            prefix = prefix_ids(tok, task, n_params, st["steps"])
            lp = action_logprobs(model, prefix, st["legal"], device, batch)
            rows.append({"p": torch.softmax(lp, 0).tolist(), "gt": st["gt"], "n_legal": len(st["legal"])})
        res.append(rows)
    return res


def summarize(tasks: Sequence[Task], before: List[List[Dict]], after: Optional[List[List[Dict]]], k: int, eps_tail: float) -> Dict:
    by_bin: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    rank_before: Dict[int, List[float]] = defaultdict(list)
    p_before: Dict[int, List[float]] = defaultdict(list)
    n_states = 0
    for ti, task in enumerate(tasks):
        b = len(task.expert)
        for si, st in enumerate(before[ti]):
            n_states += 1
            p0, gt = st["p"], st["gt"]
            rank_before[b].append(float(sum(1 for x in p0 if x > p0[gt]) + 1))
            p_before[b].append(float(p0[gt]))
            if after is not None:
                by_bin[b][categorize(p0, after[ti][si]["p"], gt, k, eps_tail)] += 1
    out: Dict = {"n_tasks": len(tasks), "n_states": n_states, "k": k, "eps_tail": eps_tail,
                 "before": {str(b): {"n": len(v), "gt_mean_rank": sum(v) / len(v), "gt_mean_p": sum(p_before[b]) / len(v),
                                     "gt_top1_rate": sum(1 for r in v if r == 1) / len(v)} for b, v in sorted(rank_before.items())}}
    if after is not None:
        out["categories"] = {str(b): {c: cnt[c] / max(sum(cnt.values()), 1) for c in CATEGORIES} for b, cnt in sorted(by_bin.items())}
        total = defaultdict(int)
        for cnt in by_bin.values():
            for c, n in cnt.items():
                total[c] += n
        out["categories"]["all"] = {c: total[c] / max(n_states, 1) for c in CATEGORIES}
    return out


def run(before_ckpt: str, after_ckpt: Optional[str], n: int, k: int, eps_tail: float, device: torch.device, batch: int = 32) -> Dict:
    tasks = heldout_tasks(n)
    tok = ByteTokenizer()
    model, _cfg, _ = load_model(before_ckpt, device)
    pb = policies(model, tok, tasks, device, batch)
    pa = None
    if after_ckpt:
        model, _cfg, _ = load_model(after_ckpt, device)
        pa = policies(model, tok, tasks, device, batch)
    res = summarize(tasks, pb, pa, k, eps_tail)
    res.update({"before_ckpt": str(before_ckpt), "after_ckpt": str(after_ckpt) if after_ckpt else None})
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True, help="the pre-RL checkpoint (SFT-1 or post-DPO)")
    ap.add_argument("--after", default=None, help="the RL checkpoint; omit for a single-checkpoint rank report")
    ap.add_argument("--n", type=int, default=60, help="held-out tasks (every expert step is a state)")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--eps-tail", type=float, default=0.05)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    res = run(args.before, args.after, args.n, args.k, args.eps_tail, device, args.batch)
    out = Path(args.out) if args.out else Path(args.after or args.before).parent / "rl_taxonomy.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
