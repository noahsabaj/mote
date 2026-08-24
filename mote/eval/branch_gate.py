"""Branch gate — the mid-training verdict (docs/shape.md § pipeline, signed 2026-08-24).

    python -m mote.eval.branch_gate --branch control=runs/branch_control --branch anneal=runs/branch_anneal \
        --sft-args "--preset flagship --data data/sft_local --sft --mix data/sft_identity:0.05 --mix data/sim_sft:0.10 \
                    --optimizer adamw --lr 3e-4 --batch-size 1 --grad-accum 8 --seq-len 4096 --ckpt-main \
                    --max-minutes 60 --eval-every 300 --ckpt-minutes 5" \
        [--out docs/results/2026-09-04-branch-gate.md] [--device cuda] [--skip-sft]

For every branch: submit the IDENTICAL SFT job to the resident daemon (init = the branch's last.pt, out =
runs/<branch>_sft), wait for it, then measure on the SFT checkpoint — reading EM/F1 (SQuAD), sim-QA EM
(held-out worlds), identity/hold/concede, needle, chat val (the SFT run's final val) — and on the branch
checkpoint itself the shared val bpb + the per-domain slices. Verdict: `anneal` ships if it wins ≥ 2 of
{reading EM, sim-QA EM, chat val bpb} against `control` and its val bpb ≤ control + 0.005; otherwise
`control`. Writes <out>.json (everything) and <out> (the table). `--skip-sft` reuses runs/<branch>_sft.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

GUARD = 0.005
DECIDERS = (("reading_em", "max"), ("sim_em", "max"), ("chat_val_bpb", "min"))


# --- daemon ----------------------------------------------------------------------------------------
def _token() -> Optional[str]:
    tok = os.environ.get("MOTE_TOKEN")
    if tok:
        return tok
    p = Path(".mote/token")
    return p.read_text(encoding="utf-8").strip() if p.exists() else None


def _api(base: str, path: str, body: Optional[dict] = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method="POST" if body is not None else "GET")
    req.add_header("Content-Type", "application/json")
    tok = _token()
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def sft_argv(sft_args: str, init: str, out: str) -> List[str]:
    return shlex.split(sft_args) + ["--init-from", init, "--out", out]


def submit(base: str, argv: List[str]) -> str:
    return _api(base, "/api/training/start", {"args": argv})["submitted"]


def wait(base: str, job_id: str, poll_s: float = 30.0) -> str:
    """Block until the job leaves the queue; returns its final state (raises unless 'done')."""
    while True:
        st = _api(base, "/api/training/queue")
        recs = ([st["current"]] if st.get("current") else []) + st.get("queued", []) + st.get("recent", [])
        rec = next((r for r in recs if r and r.get("id") == job_id), None)
        state = rec["state"] if rec else "missing"
        if state in ("done", "failed", "cancelled", "missing"):
            if state != "done":
                raise RuntimeError(f"SFT job {job_id} ended as {state}")
            return state
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
    from ..serve.engine import Engine
    from . import needle_probe, probe, read_probe, sim_probe

    eng = Engine(str(sft_ckpt), device=device)
    ident = probe.run(eng)
    read = read_probe.run(eng, read_probe.load_items(n_read, 0))
    needle = needle_probe.run(eng, [512, 1024, 2048, 4096])
    sim = sim_probe.run(eng, sim_probe.heldout_items(n_sim, ["en", "ru", "ja"]), k=k)
    auto = [v for kk, v in needle["rates"].items() if kk.startswith("auto@")]
    head = {"reading_em": read["exact_match"], "reading_f1": read["f1"], "sim_em": sim["em"], "sim_pass_at_1": sim["pass_at_1"],
            "identity_acc": ident["identity_acc"], "hold_rate": ident["hold_rate"], "concede_rate": ident["concede_rate"],
            "needle_auto": sum(auto) / max(len(auto), 1), "chat_val_bpb": final_chat_val(sft_ckpt.parent)}
    if k > 1:
        head[f"sim_pass_at_{k}"] = sim[f"pass_at_{k}"]
    return {"head": head, "rows": {"identity": ident["rows"], "reading": read["rows"], "needle": needle["rows"], "sim": sim["rows"]}}


def measure_val(branch_ckpt: Path, device: Optional[str], data: str, domains: str, batches: int) -> Dict:
    import torch

    from .val_bpb import run as val_run

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    return val_run(branch_ckpt, data or None, domains or None, batches, None, 1, dev)


# --- verdict ---------------------------------------------------------------------------------------
def verdict(control: Dict, anneal: Dict, guard: float = GUARD) -> Dict:
    """control/anneal: {"reading_em", "sim_em", "chat_val_bpb", "val_bpb"} (None = missing = not a win)."""
    wins, deltas = [], {}
    for key, better in DECIDERS:
        c, a = control.get(key), anneal.get(key)
        if c is None or a is None:
            deltas[key] = None
            continue
        deltas[key] = a - c
        if (a > c) if better == "max" else (a < c):
            wins.append(key)
    c, a = control.get("val_bpb"), anneal.get("val_bpb")
    guard_ok = (c is not None and a is not None and a <= c + guard)
    deltas["val_bpb"] = (a - c) if (c is not None and a is not None) else None
    winner = "anneal" if (len(wins) >= 2 and guard_ok) else "control"
    return {"winner": winner, "wins": wins, "n_wins": len(wins), "guard_ok": guard_ok, "guard": guard, "deltas": deltas}


def render_md(results: Dict[str, Dict], v: Dict, title: str) -> str:
    names = list(results)
    keys = ["val_bpb", "reading_em", "reading_f1", "sim_em", "chat_val_bpb", "identity_acc", "hold_rate", "concede_rate", "needle_auto"]
    keys += sorted({k for r in results.values() for k in r if k.startswith("sim_pass_at_") and k != "sim_pass_at_1"})
    dom = sorted({d for r in results.values() for d in (r.get("domains") or {})})
    lines = [f"# {title}", "", f"**Verdict: {v['winner']}** — wins {v['n_wins']}/3 ({', '.join(v['wins']) or 'none'}), "
             f"val-bpb guard {'ok' if v['guard_ok'] else 'TRIPPED'} (≤ control + {v['guard']}).", "",
             "| metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]

    def fmt(x):
        return "—" if x is None else (f"{x:.4f}" if isinstance(x, float) else str(x))

    for k in keys:
        lines.append(f"| {k} | " + " | ".join(fmt(results[n].get(k)) for n in names) + " |")
    for d in dom:
        lines.append(f"| val_bpb:{d} | " + " | ".join(fmt((results[n].get('domains') or {}).get(d)) for n in names) + " |")
    lines += ["", "Deciders: reading EM, sim-QA EM (higher wins), chat val bpb (lower wins); guard: shared val bpb. "
              "Rows for every probe question live in the .json next to this file."]
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
    ap.add_argument("--k", type=int, default=1, help="sim-QA samples per question (pass@k; the RL headroom numbers)")
    ap.add_argument("--val-data", default="data/flagship_mix")
    ap.add_argument("--val-domains", default="data/flagship_val")
    ap.add_argument("--val-batches", type=int, default=64)
    ap.add_argument("--out", default=None, help="markdown path; default docs/results/<today>-branch-gate.md")
    args = ap.parse_args(argv)
    if not args.api:
        cfg = json.loads(Path(".mote/config.json").read_text()) if Path(".mote/config.json").exists() else {}
        args.api = f"http://{cfg.get('host', '127.0.0.1')}:{cfg.get('port', 7860)}"

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
