"""Branch gate — the mid-training verdict (docs/shape.md § mid, re-signed 2026-08-26).

    python -m mote.eval.branch_gate --branch control=runs/branch_control --branch anneal=runs/branch_anneal \
        --sft-args "--preset flagship --data data/sft_local --sft --mix data/sft_identity:0.05 --mix data/sim_sft:0.10 \
                    --optimizer adamw --lr 3e-4 --batch-size 1 --grad-accum 8 --seq-len 4096 --ckpt-main \
                    --max-minutes 60 --eval-every 300 --ckpt-minutes 5" \
        [--out docs/results/2026-09-04-branch-gate.md] [--device cuda] [--skip-sft]

For every branch: submit the IDENTICAL SFT job to the resident daemon (init = the branch's last.pt, out =
runs/<branch>_sft), wait for it, then measure on the SFT checkpoint — reading EM/F1 (SQuAD), sim-QA EM
(held-out worlds), identity/hold/concede, needle, chat val (the SFT run's final val) — and on the branch
checkpoint itself the shared val bpb + the per-domain slices.

**One decider, the rest guards** (2026-08-26). The old rule voted on reading EM, sim-QA EM and chat val
bpb and shipped the anneal on any 2 of 3. Three things were wrong with it:

* Two of the three deciders were exact match, which at this scale is a coin flip with a number attached:
  the 35M model scores a flat 0 on reading (docs/search.md). 2605.18607 is exactly this result — a proxy
  over an expert's own trajectory ranks candidates at Spearman 0.81 where cross-entropy manages 0.36, and
  "a model which cannot solve a problem can still track the CoT written by an expert". `mote.eval.proxy`
  is now the decider, and its delta has to clear the combined standard error of the two arms — at n=120
  the gap between the best and worst of three known-different checkpoints is only 2.3 sem.
* All three were **confounded by data inclusion**. Only the anneal carried the sim, chat and identity
  extras, and it was then judged on sim EM and chat val — it answered "does training on X help X", not
  "is this mixture a better base". The extras are in *both* branches now, so the A/B varies the web
  reweighting alone.
* `needle` was measured, rendered, and then ignored by the verdict, while the anneal reweighting cut the
  long-document share. It is a guard.

Guards, all of which must hold or the anneal does not ship: val bpb ≤ control + 0.005 **within the same
decay condition** (so the mixture is never charged for the decay axis's outcome), and no regression on
`needle_auto`, `false_fire_rate` or `recovery_rate` **beyond the two arms' combined standard error** — the
same rule the decider lives under. Decided 2026-08-29 (option C): at 24 / 40 / 40 items the guards were
single-item vetoes, so a one-flip tie between equal checkpoints sent every anneal back to control; they are
144 / 120 / 120 items now and each carries a sem. Everything else is reported, not voted.

Writes <out>.json (everything) and <out> (the table). `--skip-sft` reuses runs/<branch>_sft.
"""

from __future__ import annotations

import argparse
import json
import shlex
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

GUARD = 0.005
# mote.eval.proxy: mean reciprocal rank of the expert's next byte, unweighted. Chosen by measurement over
# the 12-metric library, not by argument — the table is in that module's docstring.
DECIDER = ("proxy_track", "max")
GUARDS = (("needle_auto", "max"), ("false_fire_rate", "min"),
          ("recovery_rate", "max"))  # must not regress
# Kept in the table, never in the verdict: at 35M-100M these sit on or near their noise floor, and a
# metric that cannot discriminate should not get a vote (2605.18607 §5.2).
REPORTED = ("reading_em", "reading_f1", "sim_em", "chat_val_bpb", "identity_acc", "hold_rate",
            "concede_rate", "identity_recite_rate", "template_fire_rate")


# --- daemon ----------------------------------------------------------------------------------------
def _api(base: str, path: str, body: Optional[dict] = None) -> dict:
    from ..client import api

    return api("POST" if body is not None else "GET", path, body, base=base)


def sft_argv(sft_args: str, init: str, out: str) -> List[str]:
    return shlex.split(sft_args) + ["--init-from", init, "--out", out]


def submit(base: str, argv: List[str]) -> str:
    return _api(base, "/api/training/start", {"args": argv})["submitted"]


def wait(base: str, job_id: str, poll_s: float = 30.0) -> str:
    """Block until the job's LINEAGE leaves the queue; returns its final state (raises unless 'done').

    A daemon restart or an OOM retry re-enqueues the job as a new record whose `retry_of` names the old one
    (mote.serve.jobs), so the id being followed moves along the lineage; the old record stays `interrupted`
    or `failed` for good and would otherwise be waited on forever (2026-08-29)."""
    while True:
        st = _api(base, "/api/training/queue")
        recs = ([st["current"]] if st.get("current") else []) + st.get("queued", []) + st.get("recent", [])
        recs = [r for r in recs if r]
        successor = next((r for r in recs if r.get("retry_of") == job_id), None)
        if successor is not None:
            job_id = successor["id"]
            continue
        rec = next((r for r in recs if r.get("id") == job_id), None)
        state = rec["state"] if rec else "missing"
        if state == "done":
            return state
        if state in ("failed", "cancelled", "held", "missing"):
            raise RuntimeError(f"SFT job {job_id} ended as {state}" + (f": {rec.get('error')}" if rec and rec.get("error") else ""))
        time.sleep(poll_s)


# --- measurements ----------------------------------------------------------------------------------
def final_chat_val(run_dir: Path) -> Optional[float]:
    last = None
    for line in (run_dir / "log.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if "eval" in rec and rec["eval"].get("val_bpb") is not None:
            last = rec["eval"]["val_bpb"]
    return last


def measure_sft(sft_ckpt: Path, device: Optional[str], n_read: int, n_sim: int, k: int) -> Dict:
    from ..infer.engine import Engine
    from . import needle_probe, probe, proxy, read_probe, recovery_probe, sim_probe

    eng = Engine(str(sft_ckpt), device=device)
    ident = probe.run(eng)
    read = read_probe.run(eng, read_probe.load_items(n_read, 0))
    needle = needle_probe.run(eng, [512, 1024, 2048, 4096])
    sim = sim_probe.run(eng, sim_probe.heldout_items(n_sim, ["en", "ru", "ja"]), k=k)
    # The decider. One forward pass per item, no generation, on the same checkpoint the probes just ran —
    # it is the cheapest number in this function and the only one that votes.
    import torch

    from ..identity import identity_card

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    px = proxy.run(sft_ckpt, proxy.gather_items(n_sim, n_read, n_sim // 2), dev,
                   identity_card(eng.info()["params"]))
    # Recovery: does the model do something else after the environment refuses? Its own guard because a
    # mixture that teaches the world but not the response to it is not ready for RLVR-1 (2608.20314).
    rec = recovery_probe.run(eng, recovery_probe.build_items(n_sim, ["en", "ru", "ja"]))  # 120 items, not 40 (2026-08-29)
    head = {"proxy_track": px.get("recip_rank_uniform"), "proxy_track_sem": px.get("recip_rank_uniform_sem"),
            "proxy_agree": px.get("agree"), "proxy_ce": px.get("ce"),
            "proxy_result": (px.get("per_source") or {}).get("result"),
            "recovery_rate": rec.get("recovery_rate"), "recovery_rate_sem": rec.get("recovery_rate_sem"), "repeat_rate": rec.get("repeat_rate"),
            "unparseable_rate": rec.get("unparseable_rate"),
            "reading_em": read["exact_match"], "reading_f1": read["f1"], "sim_em": sim["em"], "sim_pass_at_1": sim["pass_at_1"],
            "identity_acc": ident["identity_acc"], "hold_rate": ident["hold_rate"], "concede_rate": ident["concede_rate"],
            "false_fire_rate": ident.get("false_fire_rate"), "false_fire_rate_sem": ident.get("false_fire_rate_sem"),
            "identity_recite_rate": ident.get("identity_recite_rate"),
            "template_fire_rate": ident.get("template_fire_rate"),
            "needle_auto": needle.get("needle_auto"), "needle_auto_sem": needle.get("needle_auto_sem"),
            "chat_val_bpb": final_chat_val(sft_ckpt.parent)}
    if k > 1:
        head[f"sim_pass_at_{k}"] = sim[f"pass_at_{k}"]
    return {"head": head, "rows": {"identity": ident["rows"], "reading": read["rows"], "needle": needle["rows"],
                                   "sim": sim["rows"], "proxy": px.get("rows", []),
                                   "recovery": rec.get("rows", [])}}


def measure_val(branch_ckpt: Path, device: Optional[str], data: str, domains: str, batches: int) -> Dict:
    import torch

    from .val_bpb import run as val_run

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    return val_run(branch_ckpt, data or None, domains or None, batches, None, 1, dev)


# --- verdict ---------------------------------------------------------------------------------------
def fmt_delta(x: Optional[float]) -> str:
    return "—" if x is None else f"{x:+.4f} vs control"


def verdict(control: Dict, anneal: Dict, guard: float = GUARD) -> Dict:
    """One decider (`proxy_track`, higher wins) and three guards; `control` is the default.

    A missing decider is not a win — an arm that failed to measure does not get promoted by its absence.
    A missing guard likewise fails closed: `needle_auto`/`false_fire_rate` are cheap and their absence
    means the probe did not run, not that nothing regressed."""
    key, better = DECIDER
    c, a = control.get(key), anneal.get(key)
    deltas = {k: (anneal.get(k) - control.get(k)) if (control.get(k) is not None and anneal.get(k) is not None) else None
              for k in (key, "val_bpb", *(g for g, _ in GUARDS), *REPORTED)}
    # The delta has to clear the noise as well as point the right way. Measured 2026-08-26 at n=120: the
    # gap between the best and worst of three checkpoints whose ordering is known from 12-hour runs is
    # 2.3 standard errors, so a branch difference inside one sem has not decided anything. Without this a
    # coin flip ships a 1.4-GPU-day branch and the table would not show that it had.
    sems = [x for x in (control.get(f"{key}_sem"), anneal.get(f"{key}_sem")) if x]
    noise = (sum(s * s for s in sems) ** 0.5) if sems else 0.0
    decided = (c is not None and a is not None) and (abs(a - c) > noise) and ((a > c) if better == "max" else (a < c))

    cv, av = control.get("val_bpb"), anneal.get("val_bpb")
    checks = {"val_bpb": bool(cv is not None and av is not None and av <= cv + guard)}
    guard_noise = {}
    for g, gb in GUARDS:
        gc, ga = control.get(g), anneal.get(g)
        sc, sa = control.get(f"{g}_sem"), anneal.get(f"{g}_sem")
        if gc is None or ga is None or sc is None or sa is None:  # fail closed: no number, or no noise estimate
            checks[g] = False
            continue
        gn = (sc * sc + sa * sa) ** 0.5  # the guards live under the decider's rule (2026-08-29)
        guard_noise[g] = gn
        checks[g] = bool((ga >= gc - gn) if gb == "max" else (ga <= gc + gn))
    guard_ok = all(checks.values())
    return {"winner": "anneal" if (decided and guard_ok) else "control", "decider": key,
            "decided": decided, "noise": noise, "guard_ok": guard_ok, "guard": guard, "guards": checks,
            "guard_noise": guard_noise, "deltas": deltas}


def render_md(results: Dict[str, Dict], v: Dict, title: str) -> str:
    names = list(results)
    keys = ["proxy_track", "proxy_track_sem", "proxy_agree", "proxy_result", "proxy_ce", "val_bpb",
            "needle_auto", "needle_auto_sem", "false_fire_rate", "false_fire_rate_sem", "recovery_rate", "recovery_rate_sem",
            "repeat_rate", "unparseable_rate", *REPORTED]
    keys += sorted({k for r in results.values() for k in r if k.startswith("sim_pass_at_") and k != "sim_pass_at_1"})
    dom = sorted({d for r in results.values() for d in (r.get("domains") or {})})
    tripped = [g for g, ok in v["guards"].items() if not ok]
    lines = [f"# {title}", "",
             f"**Verdict: {v['winner']}** — decider `{v['decider']}` {'favours anneal' if v['decided'] else 'does not favour anneal'}"
             f" ({fmt_delta(v['deltas'].get(v['decider']))}, noise +/-{v['noise']:.4f}); guards {'all ok' if v['guard_ok'] else 'TRIPPED: ' + ', '.join(tripped)}"
             f" (val bpb ≤ control + {v['guard']}; needle, false-fire and recovery no-regression beyond their combined sem"
             + (": " + ", ".join(f"{g} ±{n:.3f}" for g, n in v.get("guard_noise", {}).items()) if v.get("guard_noise") else "") + ").", "",
             "| metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]

    def fmt(x):
        return "—" if x is None else (f"{x:.4f}" if isinstance(x, float) else str(x))

    for k in keys:
        lines.append(f"| {k} | " + " | ".join(fmt(results[n].get(k)) for n in names) + " |")
    for d in dom:
        lines.append(f"| val_bpb:{d} | " + " | ".join(fmt((results[n].get('domains') or {}).get(d)) for n in names) + " |")
    lines += ["", "**Decider**: `proxy_track` — mean reciprocal rank of the expert's next byte over "
              "held-out trajectories (mote.eval.proxy, 2605.18607), which must also beat the two arms' "
              "combined standard error. **Guards**: shared val bpb ≤ control + "
              f"{v['guard']}; `needle_auto` (144 items), `false_fire_rate` (120) and `recovery_rate` (120) may not "
              "regress beyond the two arms' combined standard error (2026-08-29). Everything else in this "
              "table is reported, not voted: at this scale the exact-match rows sit on their noise floor "
              "(docs/search.md records a flat 0 on reading at 35M), and 2605.18607 §5.2 is the measurement "
              "of why a metric that cannot discriminate should not cast a vote.",
              "", "Rows for every probe question live in the .json next to this file."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", action="append", required=True, metavar="NAME=RUN_DIR", help="control=... and anneal=... (a branch's checkpoint is RUN_DIR/last.pt)")
    ap.add_argument("--sft-args", default="", help="the identical SFT argv for every branch (without --init-from/--out)")
    ap.add_argument("--skip-sft", action="store_true", help="reuse runs/<branch>_sft/last.pt")
    ap.add_argument("--sft-run", action="append", default=[], metavar="NAME=RUN_DIR", help="with --skip-sft: the SFT run to measure for a branch (default runs/<branch>_sft)")
    ap.add_argument("--api", default=None, help="daemon base URL (default: host/port from .mote/config.json, else 127.0.0.1:7860)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--n-read", type=int, default=100)
    ap.add_argument("--n-sim", type=int, default=120)
    ap.add_argument("--k", type=int, default=8, help="sim-QA samples per question: pass@8 is reported beside EM by default "
                    "(2607.16097: pass@k tracks the base, pass@1 tracks RL); pass@64 on demand for the RL start gate")
    ap.add_argument("--val-data", default="data/flagship_mix")
    ap.add_argument("--val-domains", default="data/flagship_val")
    ap.add_argument("--val-batches", type=int, default=64)
    ap.add_argument("--out", default=None, help="markdown path; default docs/results/<today>-branch-gate.md")
    args = ap.parse_args(argv)
    if not args.api:
        from ..paths import base_url

        args.api = base_url()

    branches = dict(b.split("=", 1) for b in args.branch)
    if set(branches) != {"control", "anneal"}:
        raise SystemExit("need exactly --branch control=... and --branch anneal=...")
    sft_dirs = {n: Path(f"runs/{Path(d).name}_sft") for n, d in branches.items()}
    sft_dirs.update({n: Path(d) for n, d in (s.split("=", 1) for s in args.sft_run)})
    if not args.skip_sft:
        if not args.sft_args:
            raise SystemExit("--sft-args is required unless --skip-sft")
        ids = {n: submit(args.api, sft_argv(args.sft_args, str(Path(d) / "last.pt"), str(sft_dirs[n]))) for n, d in branches.items()}
        print("submitted:", ids, flush=True)
        for n, jid in ids.items():
            print(f"{n}: {wait(args.api, jid)}", flush=True)

    results, rows = {}, {}
    for n, d in branches.items():
        m = measure_sft(sft_dirs[n] / "last.pt", args.device, args.n_read, args.n_sim, args.k)
        val = measure_val(Path(d) / "last.pt", args.device, args.val_data, args.val_domains, args.val_batches)
        results[n] = {**m["head"], "val_bpb": val.get("val_bpb"), "domains": val.get("domains")}
        rows[n] = m["rows"]
        print(n, json.dumps({k: v for k, v in results[n].items() if k != "domains"}), flush=True)
    v = verdict(results["control"], results["anneal"])
    out = Path(args.out) if args.out else Path("docs/results") / f"{time.strftime('%Y-%m-%d')}-branch-gate.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_md(results, v, f"Branch gate {time.strftime('%Y-%m-%d')}: control vs anneal"), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps({"branches": branches, "results": results, "verdict": v, "rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
