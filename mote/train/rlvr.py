"""RLVR-1: on-policy GRPO-style reinforcement on verifiable sim tasks (docs/shape.md § pipeline, signed 2026-08-24).

    mote train start -- rlvr --init-from runs/flagship_dpo/last.pt --out runs/rlvr1 --steps 200 \
        --tasks 16 --group 8 --lr 1e-6 --kl 0.02 --max-bytes 512 --eval-every 10 --eval-tasks 60 --eval-k 8

Each step: sample --tasks fresh tasks (seeds from --seed-base, never the probe's or the traces'), roll out
--group replies per task through the engine's tool hook at --temperature (eager decoding on the live
policy weights, the sim as the `sim` tool, the task's step budget as the call cap), reward 1 iff the goal
holds at the end (+ --partial × the fraction of goal facts, 0 by default), keep only the groups with mixed
outcomes (edge of competence: a group that all fails or all succeeds carries no signal), group-normalise
the advantages, and take ONE gradient step on

    −A · mean_t log π(y_t)  +  β · KL(π ‖ π_ref)      (KL: k3 estimator on the sampled bytes; π_ref = the initial policy)

over the model's own bytes only (tool results carry no loss). Strictly on-policy, one update per batch: no
importance ratios, no clipping. Every --eval-every steps: held-out pass@1 (greedy) and pass@k over
--eval-tasks tasks — the signed guards read these (pass@k never below the pre-RL model). Logs log.jsonl,
checkpoints last.pt (resumable) and best.pt (by held-out pass@1). Runs as a daemon job (`mote train
start -- rlvr ...`): run() yields after every rollout and micro-batch so replies slot in.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from ..config import MoteConfig
from ..model.hnet import HNetForCausalLM
from ..serve.engine import Engine, GenParams
from ..sim.tasks import TASK_DOMAINS, SimEnv, Task, heldout_tasks, make_task
from ..tokenizer import PAD_ID
from .train import load_checkpoint, save_checkpoint

TRAIN_SEED_BASE = 3_000_000


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="rlvr")
    ap.add_argument("--init-from", required=True, help="policy init and frozen reference (the post-DPO checkpoint)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--tasks", type=int, default=16, help="tasks (prompts) per step")
    ap.add_argument("--group", type=int, default=8, help="rollouts per task")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--kl", type=float, default=0.02, help="β of the KL-to-reference penalty")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-bytes", type=int, default=512, help="reply budget per rollout (calls + results + text)")
    ap.add_argument("--partial", type=float, default=0.0, help="weight of the goal-fraction reward on top of the binary one")
    ap.add_argument("--micro", type=int, default=4, help="sequences per forward/backward")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--locales", default="en,ru,ja")
    ap.add_argument("--seed-base", type=int, default=TRAIN_SEED_BASE)
    ap.add_argument("--eval-every", type=int, default=10, help="0 = never")
    ap.add_argument("--eval-tasks", type=int, default=60)
    ap.add_argument("--eval-k", type=int, default=8)
    ap.add_argument("--max-minutes", type=float, default=1440.0)
    ap.add_argument("--ckpt-minutes", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--resume", action="store_true")
    return ap


def group_advantages(rewards: Sequence[float], eps: float = 1e-6) -> List[float]:
    """GRPO: (r − mean) / (std + eps) within one prompt's group."""
    n = len(rewards)
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n
    return [(r - mean) / (math.sqrt(var) + eps) for r in rewards]


def has_signal(rewards: Sequence[float]) -> bool:
    """A group only teaches when its outcomes differ (edge of competence)."""
    return max(rewards) - min(rewards) > 1e-9


def pad_batch(seqs: List[List[int]], masks: List[List[int]], device) -> Tuple[torch.Tensor, torch.Tensor]:
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), PAD_ID, dtype=torch.long)
    m = torch.zeros((len(seqs), T), dtype=torch.float32)
    for i, (s, k) in enumerate(zip(seqs, masks)):
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        m[i, : len(k)] = torch.tensor(k, dtype=torch.float32)
    return ids.to(device), m.to(device)


def token_logprobs(model: HNetForCausalLM, ids: torch.Tensor, device) -> torch.Tensor:
    """log π(ids[:, t+1] | ids[:, :t+1]) for every position: [B, T−1]."""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        out = model(ids[:, :-1])
    logp = F.log_softmax(out.logits.float(), dim=-1)
    return logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)


class RlvrTrainer:
    """Drivable like Trainer: run() is a generator of ("slice"|"step", None); model / cfg / out_dir / step."""

    def __init__(self, argv_or_args=None):
        args = build_argparser().parse_args(argv_or_args) if (argv_or_args is None or isinstance(argv_or_args, list)) else argv_or_args
        self.args = args
        self.device = device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.out_dir = out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "run.json").write_text(json.dumps({**vars(args), "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2))
        torch.manual_seed(args.seed)
        self.rng = random.Random(args.seed)
        ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
        self.cfg = MoteConfig.from_dict(ck["config"])
        self.model = HNetForCausalLM(self.cfg, device=device)
        self.model.load_state_dict(ck["model"])
        self.ref = copy.deepcopy(self.model).eval()
        for p in self.ref.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
        self.engine = Engine.from_model(self.model, self.cfg, device=str(device), name=f"{out_dir.name}/policy")
        self.engine._graph_ok = False  # multi-turn rollouts on live weights: eager decoding (the graph path comes later)
        self.locales = [l.strip() for l in args.locales.split(",") if l.strip()]
        self.step, self.cursor, self.best = 0, args.seed_base, -1.0
        self.t_start = time.time()
        self._stop, self.stopped_reason = False, None
        self.ckpt_path = out_dir / "last.pt"
        if args.resume and self.ckpt_path.exists():
            self.step, extra = load_checkpoint(self.ckpt_path, self.model, self.opt)
            self.cursor, self.best = int(extra.get("cursor", self.cursor)), float(extra.get("best", -1.0))
            print(f"resumed rlvr from step {self.step}", flush=True)
        self.log_f = open(out_dir / "log.jsonl", "a", encoding="utf-8")
        self.n_params = self.model.num_params()

    # --- plumbing shared with Trainer ------------------------------------------------------------------
    def log(self, rec: dict) -> None:
        rec = {**rec, "step": self.step, "elapsed_min": (time.time() - self.t_start) / 60}
        self.log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.log_f.flush()
        print(json.dumps({k: v for k, v in rec.items() if k != "rows"}, ensure_ascii=False)[:400], flush=True)

    def request_stop(self, reason: str = "requested") -> None:
        self._stop, self.stopped_reason = True, reason

    def save(self, path: Optional[Path] = None) -> None:
        save_checkpoint(path or self.ckpt_path, self.model, self.opt, self.step, self.cfg,
                        {"rlvr": True, "cursor": self.cursor, "best": self.best, "n_params": self.n_params})

    def close(self) -> None:
        self.log_f.close()

    # --- rollouts -------------------------------------------------------------------------------------------
    def reward_of(self, env: SimEnv) -> Tuple[float, float]:
        ok, frac = env.score()
        return float(ok) + self.args.partial * frac, frac

    def rollout(self, task: Task, temperature: float, max_bytes: Optional[int] = None) -> Dict:
        env = SimEnv(task)
        self.engine.register_tool("sim", env.act)
        ev: List[dict] = []
        try:
            self.model.eval()
            with torch.no_grad():
                self.engine.generate([{"role": "user", "content": task.prompt}],
                                     GenParams(temperature=temperature, top_p=1.0, max_bytes=max_bytes or self.args.max_bytes,
                                               n_candidates=0, max_calls=task.budget),
                                     ev.append, threading.Event(), context={"want_ids": True, "fold": "off"})
            reward, frac = self.reward_of(env)
            done = ev[-1]
            return {"task": task, "prompt_ids": done["prompt_ids"], "ids": done["ids"], "mask": done["mask"], "eos": done["eos"],
                    "calls": done["calls"], "reward": reward, "frac": frac, "text": done["text"], "steps": env.steps,
                    "n_bytes": len(done["ids"])}
        finally:
            env.close()

    def evaluate(self, n: int, k: int) -> Dict:
        tasks = heldout_tasks(n, self.locales)
        p1 = pk = 0
        per_dom: Dict[str, List[int]] = {d: [] for d in TASK_DOMAINS}
        for t in tasks:
            greedy = self.rollout(t, 0.0)["reward"] >= 1.0
            passed = greedy or any(self.rollout(t, self.args.temperature)["reward"] >= 1.0 for _ in range(k - 1)) if k > 1 else greedy
            p1 += greedy
            pk += passed
            per_dom[t.domain].append(int(greedy))
        return {"pass_at_1": p1 / n, f"pass_at_{k}": pk / n, "k": k, "n": n,
                "per_domain_pass_at_1": {d: (sum(v) / len(v) if v else None) for d, v in per_dom.items()}}

    # --- the update -----------------------------------------------------------------------------------------
    def _loss_on(self, seqs: List[List[int]], masks: List[List[int]], advs: List[float], n_total: int):
        ids, m = pad_batch(seqs, masks, self.device)
        tgt_mask = m[:, 1:]  # mask is per position of `ids`; targets are ids[1:]
        logp = token_logprobs(self.model, ids, self.device)
        with torch.no_grad():
            ref = token_logprobs(self.ref, ids, self.device)
        n_tok = tgt_mask.sum(dim=1).clamp_min(1.0)
        mean_logp = (logp * tgt_mask).sum(dim=1) / n_tok
        d = ref - logp
        kl = ((torch.exp(d) - d - 1.0) * tgt_mask).sum(dim=1) / n_tok  # k3 estimator, ≥ 0
        A = torch.tensor(advs, dtype=torch.float32, device=self.device)
        loss = (-(A * mean_logp) + self.args.kl * kl).sum() / n_total
        loss.backward()
        return float(loss.detach()), float(kl.mean().detach()), float(mean_logp.mean().detach())

    def update(self, rollouts: List[Dict], advs: List[float]):
        """One gradient step over every rollout with an advantage; yields per micro-batch."""
        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        order = sorted(range(len(rollouts)), key=lambda i: len(rollouts[i]["prompt_ids"]) + len(rollouts[i]["ids"]))
        tot_loss = tot_kl = tot_lp = 0.0
        n_mb = 0
        for s in range(0, len(order), self.args.micro):
            idx = order[s : s + self.args.micro]
            seqs = [rollouts[i]["prompt_ids"] + rollouts[i]["ids"] for i in idx]
            masks = [[0] * len(rollouts[i]["prompt_ids"]) + rollouts[i]["mask"] for i in idx]
            l, kl, lp = self._loss_on(seqs, masks, [advs[i] for i in idx], len(rollouts))
            tot_loss += l
            tot_kl += kl * len(idx)
            tot_lp += lp * len(idx)
            n_mb += 1
            yield ("slice", None)
        gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        self.engine.prefix_cache.clear()  # the policy changed: cached anchors no longer match its states
        return {"loss": tot_loss, "kl": tot_kl / len(rollouts), "logp": tot_lp / len(rollouts), "grad_norm": float(gnorm), "micro_batches": n_mb}

    # --- the loop -------------------------------------------------------------------------------------------
    def run(self):
        args = self.args
        last_ckpt = time.time()
        while self.step < args.steps and not self._stop and (time.time() - self.t_start) / 60 < args.max_minutes:
            t0 = time.time()
            tasks = []
            for _ in range(args.tasks):
                self.cursor += 1
                tasks.append(make_task(TASK_DOMAINS[self.cursor % len(TASK_DOMAINS)], self.cursor, self.locales[self.cursor % len(self.locales)]))
            groups: List[List[Dict]] = []
            for t in tasks:
                g = []
                for _ in range(args.group):
                    g.append(self.rollout(t, args.temperature))
                    yield ("slice", None)
                groups.append(g)
            t_roll = time.time() - t0
            all_r = [r["reward"] for g in groups for r in g]
            used = [g for g in groups if has_signal([r["reward"] for r in g])]
            rollouts = [r for g in used for r in g]
            advs = [a for g in used for a in group_advantages([r["reward"] for r in g])]
            rec = {"reward": sum(all_r) / len(all_r), "success": sum(r["reward"] >= 1.0 for g in groups for r in g) / len(all_r),
                   "frac": sum(r["frac"] for g in groups for r in g) / len(all_r),
                   "groups": len(groups), "groups_used": len(used),
                   "all_fail": sum(max(r["reward"] for r in g) < 1.0 for g in groups), "all_pass": sum(min(r["reward"] for r in g) >= 1.0 for g in groups),
                   "calls_mean": sum(r["calls"] for g in groups for r in g) / len(all_r),
                   "eos_rate": sum(r["eos"] for g in groups for r in g) / len(all_r),
                   "bytes_mean": sum(r["n_bytes"] for g in groups for r in g) / len(all_r), "rollout_s": t_roll}
            if rollouts:
                rec.update((yield from self.update(rollouts, advs)))
            else:
                rec["note"] = "no group with mixed outcomes: no update"
            self.step += 1
            rec["step_s"] = time.time() - t0
            self.log(rec)
            if args.eval_every and self.step % args.eval_every == 0:
                ev = self.evaluate(args.eval_tasks, args.eval_k)
                self.log({"eval": ev})
                if ev["pass_at_1"] > self.best:
                    self.best = ev["pass_at_1"]
                    self.save(self.out_dir / "best.pt")
                yield ("slice", None)
            if (time.time() - last_ckpt) / 60 >= args.ckpt_minutes:
                self.save()
                last_ckpt = time.time()
            yield ("step", None)
        if self._stop:
            self.log({"stopped": self.stopped_reason or "requested"})
        self.save()
        self.log({"done": True, "final_step": self.step, "best_pass_at_1": self.best})


def main(argv=None):
    t = RlvrTrainer(argv)
    try:
        for _ in t.run():
            pass
    finally:
        t.close()


if __name__ == "__main__":
    main()
