"""The 2026-08-29 cloud arms as Lightning Jobs from the org studio `mote`
(docs/results/2026-08-29-flagship-lr-transfer-prereg.md). Runs with the system python that has lightning_sdk:

    /usr/bin/python3 scripts/cloud_arms.py submit lr-3.6e-4 lr-7.2e-4 lr-14.4e-4 lr-28.8e-4 qk bitwise
    /usr/bin/python3 scripts/cloud_arms.py status            # state, machine, cost so far
    /usr/bin/python3 scripts/cloud_arms.py logs <job> [-n N]  # the job's stdout tail
    /usr/bin/python3 scripts/cloud_arms.py stop <job>

Inside a job the studio snapshot is mounted at /teamspace/studios/this_studio and is the working copy; whatever a
job writes there is synced to /teamspace/jobs/<job>/artifacts/ (that path itself is read-only from inside the
job — Errno 30 — so runs go to runs/cloud/... in the working copy). `ssh mote` sees the synced copy;
`scripts/cloud_arms.py progress` tails each run's log.jsonl there.
"""
import argparse
import subprocess
import sys

STUDIO = dict(name="mote", teamspace="deploy-model-project", org="noahbsabaj-org")
SSH_HOST = "mote"
CD = 'cd /teamspace/studios/this_studio/mote 2>/dev/null || cd "$(ls -d /teamspace/jobs/*/work/mote | head -1)"'
ENV = "export PYTHONUNBUFFERED=1 HF_HOME=/teamspace/studios/this_studio/.hf"

TRUNK = ("--preset mote-96m --data data/flagship_mix --batch-size 1 --seq-len 16384 --grad-accum 2 "
         "--optimizer muon --weight-decay 0.1 --schedule trunk --bound-floor 2048 --ckpt-main --seed 42")
EVAL = "--eval-ema 0.9999 --eval-every 500 --eval-batches 16 --eval-spread --ckpt-minutes 60"
HORIZON = 61035  # 2.0e9 bytes at 32,768 per step; warmup = 10 % of it for every arm
QK_BASE = ("--preset local --data data/local_mix --optimizer muon --lr 8e-4 --batch-size 4 --seq-len 2048 "
           "--grad-accum 4 --eval-every 500 --ckpt-minutes 30 --no-mbp --eval-spread --seed 42")
QK_ARMS = [("off", 24000, ""), ("on", 24000, "--qk-norm"), ("on_tau1.0", 12000, "--qk-norm --tau-s 1.0"),
           ("on_tau2.37", 12000, "--qk-norm --tau-s 2.37"), ("on_tau4.0", 12000, "--qk-norm --tau-s 4.0"),
           ("on_lam1.0", 12000, "--qk-norm --lambda-init 1.0")]
BITWISE = (f"{TRUNK} --lr 8e-4 --max-steps 100 --max-minutes 30 --log-every 1 --eval-every 100000 "
           "--eval-batches 8 --ckpt-minutes 999")


def art(job):
    """Where a job's runs/cloud/... land as seen from the studio (synced from the job's working copy)."""
    return f"/teamspace/jobs/{job}/artifacts/mote/runs/cloud"


def lr_job(lr, stop):
    name = "mote-lr-" + lr.replace(".", "p")
    cmd = (f"{CD} && {ENV} && bash scripts/cloud_arm.sh {stop} runs/cloud/lr/{lr} -- "
           f"{TRUNK} {EVAL} --lr {lr} --max-steps {HORIZON} --max-minutes 1440")
    return name, cmd


def qk_job():
    name = "mote-qk"
    runs = "; ".join(f"bash scripts/cloud_arm.sh {n} runs/cloud/qk/{arm} -- {QK_BASE} --max-steps {n} {flags}"
                     for arm, n, flags in QK_ARMS)
    return name, f"{CD} && {ENV} && {runs}; ls runs/cloud/qk"


def bitwise_job():
    name = "mote-bitwise"
    cmd = (f"{CD} && {ENV} && python -m mote.train.train {BITWISE} --out runs/cloud/bitwise/new; "
           f"cd ../mote-old && python -m mote.train.train {BITWISE} --out ../mote/runs/cloud/bitwise/old; "
           f"cd ../mote && python scripts/bitwise_diff.py runs/cloud/bitwise/old runs/cloud/bitwise/new; "
           f"python -m pytest -x -q 2>&1 | tail -15")
    return name, cmd


def bitwise_ctl_job():
    """Same-code control for the bitwise check: HEAD twice on one machine. If a vs b differ, the machine is not
    run-to-run deterministic at this shape and the old-vs-new verdict cannot be read there."""
    name = "mote-bitwise-ctl"
    cmd = (f"{CD} && {ENV} && python -m pytest -x -q 2>&1 | tail -40; "
           f"python -m mote.train.train {BITWISE} --out runs/cloud/bitwise/a; "
           f"python -m mote.train.train {BITWISE} --out runs/cloud/bitwise/b; "
           f"python scripts/bitwise_diff.py runs/cloud/bitwise/a runs/cloud/bitwise/b")
    return name, cmd


JOBS = {
    "bitwise-ctl": bitwise_ctl_job,
    "lr-3.6e-4": lambda: lr_job("3.6e-4", HORIZON),
    "lr-7.2e-4": lambda: lr_job("7.2e-4", HORIZON),
    "lr-14.4e-4": lambda: lr_job("14.4e-4", HORIZON),
    "lr-28.8e-4": lambda: lr_job("28.8e-4", 15259),  # D = 5.0e8; same warmup as the others
    "qk": qk_job,
    "bitwise": bitwise_job,
}
RUN_PATHS = {  # job -> the run directories it writes, for `progress`
    "mote-lr-3p6e-4": ["lr/3.6e-4"], "mote-lr-7p2e-4": ["lr/7.2e-4"], "mote-lr-14p4e-4": ["lr/14.4e-4"],
    "mote-lr-28p8e-4": ["lr/28.8e-4"], "mote-qk": [f"qk/{a}" for a, _, _ in QK_ARMS],
    "mote-bitwise": ["bitwise/new", "bitwise/old"], "mote-bitwise-ctl": ["bitwise/a", "bitwise/b"],
}


def studio():
    from lightning_sdk import Studio
    return Studio(**STUDIO, create_ok=False)


def submit(names, dry_run, machine):
    from lightning_sdk import CloudProvider, Job, Machine
    st = None if dry_run else studio()
    for i, key in enumerate(names):
        name, cmd = JOBS[key]()
        if dry_run:
            print(f"--- {name} on {machine}\n{cmd}\n")
            continue
        j = Job.run(name=name, machine=getattr(Machine, machine), cloud=CloudProvider.GCP, command=cmd, studio=st,
                    interruptible=False, reuse_snapshot=i > 0)
        print(f"submitted {j.name}: {j.status}  {j.link}")


def jobs(names=None):
    from lightning_sdk import Job
    for name in names or RUN_PATHS:
        try:
            yield name, Job(name=name, teamspace=STUDIO["teamspace"], org=STUDIO["org"])
        except Exception as e:  # not submitted (yet)
            print(f"{name}: {type(e).__name__}")


def status():
    for name, j in jobs():
        try:
            print(f"{name:18} {str(j.status):10} {str(j.machine):8} ${j.total_cost:.2f}")
        except Exception as e:
            print(f"{name:18} ? ({e})")


def logs(name, n):
    from lightning_sdk import Job
    j = Job(name=name, teamspace=STUDIO["teamspace"], org=STUDIO["org"])
    try:
        text = j.logs
        lines = text.splitlines() if isinstance(text, str) else list(text)
    except RuntimeError as e:  # "Logs are not available while the job is Pending."
        print(f"{name}: {j.status} ({e})")
        return
    print("\n".join(lines[-n:]))


def stop(name):
    from lightning_sdk import Job
    Job(name=name, teamspace=STUDIO["teamspace"], org=STUDIO["org"]).stop()
    print("stopped", name)


def progress():
    script = "; ".join(
        f'f={art(job)}/{p}/log.jsonl; [ -f $f ] && echo "{job}/{p}: $(grep -c "" $f) lines | $(tail -1 $f | cut -c1-160)" || echo "{job}/{p}: no log"'
        for job, paths in RUN_PATHS.items() for p in paths)
    subprocess.run(["ssh", "-o", "BatchMode=yes", SSH_HOST, script], check=False)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit"); s.add_argument("names", nargs="+", choices=sorted(JOBS)); s.add_argument("--dry-run", action="store_true")
    s.add_argument("--machine", default="L4")
    sub.add_parser("status"); sub.add_parser("progress")
    l = sub.add_parser("logs"); l.add_argument("name"); l.add_argument("-n", type=int, default=40)
    t = sub.add_parser("stop"); t.add_argument("name")
    a = ap.parse_args()
    if a.cmd == "submit":
        submit(a.names, a.dry_run, a.machine)
    elif a.cmd == "status":
        status()
    elif a.cmd == "progress":
        progress()
    elif a.cmd == "logs":
        logs(a.name, a.n)
    elif a.cmd == "stop":
        stop(a.name)


if __name__ == "__main__":
    sys.exit(main())
