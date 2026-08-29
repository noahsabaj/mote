"""Lightning.ai driver for experiments — arms, probes, kernel tuning; the flagship trains locally (docs/shape.md,
2026-08-24). NOTHING here starts a machine unless you pass --go.

    python cloud/launch.py plan                         # print what would happen (default machine, budget math)
    python cloud/launch.py up --go                      # create/start the studio on a CPU machine, sync, bootstrap
    python cloud/launch.py data --go                    # build the pretraining + SFT mixes on the studio (detached, CPU is enough)
    python cloud/launch.py switch --machine H100 --go   # move to an interruptible H100 once the data is built
    python cloud/launch.py train --go [--resume]        # start/resume the flagship pretraining (detached)
    python cloud/launch.py sft --go                     # SFT from the latest pretrain checkpoint (detached)
    python cloud/launch.py status | logs | pull | stop --go

Auth: `lightning login` (credentials in ~/.lightning). Teamspace/user are resolved from the account.
Everything runs with interruptible=True; training auto-resumes from runs/<run>/last.pt on the studio's
persistent disk, so a preemption costs at most --ckpt-minutes of progress. Re-run `train --go --resume`
after a preemption (or put it in a cron on the studio).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDIO_NAME = "mote-flagship"
REMOTE_DIR = "mote"  # under the studio home
MACHINE = "H100"
USERNAME_FALLBACK = "noahbsabaj"

PRETRAIN_CMD = (
    "cd ~/{d} && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && "
    "nohup .venv-cloud/bin/python -m mote.train.train --preset flagship --data data/pretrain_mix --out runs/flagship "
    "--batch-size 16 --grad-accum 2 --seq-len 4096 --lr 6e-4 --ratio-weight 0.1 --max-minutes {minutes} --eval-every 500 --eval-batches 16 "
    "--ckpt-minutes 10 --log-every 20 {resume} > runs/flagship/stdout.log 2>&1 &"
)
SFT_CMD = (
    "cd ~/{d} && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && "
    "nohup .venv-cloud/bin/python -m mote.train.train --preset flagship --sft --init-from runs/flagship/last.pt "
    "--data data/sft_mix --out runs/flagship_sft --batch-size 16 --grad-accum 2 --seq-len 4096 --lr 2e-4 "
    "--max-minutes {minutes} --eval-every 200 --eval-batches 16 --ckpt-minutes 10 --log-every 20 {resume} "
    "> runs/flagship_sft/stdout.log 2>&1 &"
)
DATA_CMD = (
    "cd ~/{d} && nohup bash -c '.venv-cloud/bin/python -m mote.data.build_mix --out data/pretrain_mix --target-gb {gb} --val-mb 64 && "
    ".venv-cloud/bin/python -m mote.data.build_sft --out data/sft_mix --target-mb {sft_mb} --val-mb 8' > data/build.log 2>&1 &"
)


def studio(create: bool):
    from lightning_sdk import Studio, User
    from lightning_sdk.api.user_api import UserApi
    from lightning_sdk.lightning_cloud.login import Auth

    a = Auth()
    a.authenticate()
    try:
        name = UserApi()._get_user_by_id(a.user_id).username
    except Exception:
        name = USERNAME_FALLBACK
    user = User(name=name)
    ts = user.teamspaces[0]
    return Studio(name=STUDIO_NAME, teamspace=ts.name, user=name, create_ok=create), ts


def cmd_plan(args):
    print(json.dumps({
        "studio": STUDIO_NAME, "machine": MACHINE, "interruptible": True,
        "pretrain_minutes": args.minutes, "sft_minutes": args.sft_minutes, "data_gb": args.gb,
        "note": "25 credits total. Verify the interruptible H100 rate on the machine picker; at $0.66-1.8/h that is 14-38 h, at $3.52/h it is 7 h.",
        "commands": {
            "data": DATA_CMD.format(d=REMOTE_DIR, gb=args.gb, sft_mb=args.sft_mb),
            "train": PRETRAIN_CMD.format(d=REMOTE_DIR, minutes=args.minutes, resume=""),
            "sft": SFT_CMD.format(d=REMOTE_DIR, minutes=args.sft_minutes, resume=""),
        },
    }, indent=2))


def require_go(args):
    if not args.go:
        print("refusing: this starts or changes cloud resources. Re-run with --go.", file=sys.stderr)
        sys.exit(2)


def cmd_up(args):
    require_go(args)
    from lightning_sdk import Machine
    s, ts = studio(create=True)
    print(f"studio {s.name} in teamspace {ts.name}: status {s.status}")
    if str(s.status).lower() != "running":
        machine = getattr(Machine, args.machine)
        s.start(machine=machine, interruptible=args.machine != "CPU")
        print(f"started on {args.machine}" + (" (interruptible)" if args.machine != "CPU" else ""))
    print("syncing repo ...")
    s.upload_folder(str(ROOT), remote_path=REMOTE_DIR, progress_bar=False)
    out, code = s.run_with_exit_code(f"cd ~/{REMOTE_DIR} && bash cloud/bootstrap.sh")
    print(out[-3000:])
    if code != 0:
        sys.exit(code)


def cmd_data(args):
    require_go(args)
    s, _ = studio(create=False)
    s.run_and_detach(DATA_CMD.format(d=REMOTE_DIR, gb=args.gb, sft_mb=args.sft_mb))
    print("data build detached; follow with: python cloud/launch.py logs --file data/build.log")


def cmd_train(args):
    require_go(args)
    s, _ = studio(create=False)
    s.run(f"mkdir -p ~/{REMOTE_DIR}/runs/flagship")
    s.run_and_detach(PRETRAIN_CMD.format(d=REMOTE_DIR, minutes=args.minutes, resume="--resume" if args.resume else ""))
    print("training detached; follow with: python cloud/launch.py logs")


def cmd_sft(args):
    require_go(args)
    s, _ = studio(create=False)
    s.run(f"mkdir -p ~/{REMOTE_DIR}/runs/flagship_sft")
    s.run_and_detach(SFT_CMD.format(d=REMOTE_DIR, minutes=args.sft_minutes, resume="--resume" if args.resume else ""))
    print("sft detached; follow with: python cloud/launch.py logs --file runs/flagship_sft/stdout.log")


def cmd_switch(args):
    """Move the running studio to another machine type (e.g. CPU for data building -> H100 for training)."""
    require_go(args)
    from lightning_sdk import Machine
    s, _ = studio(create=False)
    s.switch_machine(getattr(Machine, args.machine), interruptible=args.machine != "CPU")
    print(f"switched to {args.machine}")


def cmd_status(args):
    s, ts = studio(create=False)
    print(f"{s.name}: status={s.status} machine={s.machine} interruptible={s.interruptible}")
    print(s.run(f"cd ~/{REMOTE_DIR} 2>/dev/null && nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader; tail -2 runs/flagship/stdout.log 2>/dev/null"))


def cmd_logs(args):
    s, _ = studio(create=False)
    print(s.run(f"cd ~/{REMOTE_DIR} && tail -n {args.n} {args.file}"))


def cmd_pull(args):
    s, _ = studio(create=False)
    dest = ROOT / "runs" / args.run
    dest.mkdir(parents=True, exist_ok=True)
    for f in ("last.pt", "log.jsonl", "config.json", "run.json"):
        try:
            s.download_file(f"{REMOTE_DIR}/runs/{args.run}/{f}", str(dest / f))
            print("pulled", f)
        except Exception as e:
            print("skip", f, type(e).__name__)


def cmd_stop(args):
    require_go(args)
    s, _ = studio(create=False)
    s.stop()
    print("stopped")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["plan", "up", "switch", "data", "train", "sft", "status", "logs", "pull", "stop"])
    ap.add_argument("--machine", default="CPU", help="for up/switch: CPU (data building, cheap) or H100 (training, interruptible)")
    ap.add_argument("--go", action="store_true", help="actually touch cloud resources")
    ap.add_argument("--minutes", type=float, default=600.0, help="pretraining wall-clock budget")
    ap.add_argument("--sft-minutes", type=float, default=60.0)
    ap.add_argument("--gb", type=float, default=10.0, help="pretraining mix size in GB of bytes")
    ap.add_argument("--sft-mb", type=float, default=300.0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--file", default="runs/flagship/stdout.log")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--run", default="flagship")
    args = ap.parse_args()
    globals()[f"cmd_{args.command}"](args)


if __name__ == "__main__":
    main()
