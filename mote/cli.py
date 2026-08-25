"""`mote` — one command for running the studio, so nothing lives in a terminal.

    mote service install     register the studio to start at login (Startup folder; no admin), create the token
    mote service start|stop|restart|status|uninstall
    mote build               build the web app, run the tests, restart the studio, print the pair link
    mote pair                open the pairing page (QR + code) in the browser
    mote logs [-n 80]        tail the studio log
    mote config [--checkpoint PATH] [--device cpu|cuda] [--port N]

State lives in <repo>/.mote/: token (the access token), config.json, studio.log, *.pid.
The supervisor (`service run`) launches the server, restarts it if it dies, and re-reads config.json
on every (re)start — so `mote config --checkpoint ...` followed by `mote restart` is how a
new checkpoint goes live.

Linux: `service install` writes a systemd user unit instead (the Fedora path).
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / ".mote"
TOKEN_FILE = STATE / "token"
CONFIG_FILE = STATE / "config.json"
LOG_FILE = STATE / "studio.log"
SUP_PID = STATE / "supervisor.pid"
SRV_PID = STATE / "server.pid"
STOP_FLAG = STATE / "stop"
PY = Path(sys.executable)
PYW = PY.with_name("pythonw.exe") if os.name == "nt" else PY
DEFAULT_CONFIG = {"checkpoint": "runs/pilot_sft/last.pt", "device": "cpu", "port": 7861, "host": "127.0.0.1"}


# ---- state ----------------------------------------------------------------------------------
def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_FILE.exists():
        cfg.update(json.loads(CONFIG_FILE.read_text(encoding="utf-8")))
    return cfg


def save_config(cfg: dict) -> None:
    STATE.mkdir(exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(cfg, indent=1), encoding="utf-8")


def ensure_token() -> str:
    STATE.mkdir(exist_ok=True)
    if not TOKEN_FILE.exists():
        TOKEN_FILE.write_text(secrets.token_urlsafe(24), encoding="utf-8")
    return TOKEN_FILE.read_text(encoding="utf-8").strip()


def _pid(path: Path):
    try:
        pid = int(path.read_text().strip())
    except Exception:
        return None
    return pid if _alive(pid) else None


def _alive(pid: int) -> bool:
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _kill(pid: int) -> None:
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        else:
            os.kill(pid, signal.SIGTERM)
    except Exception:
        pass


def _port_owner(port: int):
    """pid listening on the port, or None."""
    if os.name == "nt":
        out = subprocess.run(["netstat", "-ano", "-p", "tcp"], capture_output=True, text=True).stdout
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[1].endswith(f":{port}") and parts[3] == "LISTENING":
                return int(parts[4])
        return None
    out = subprocess.run(["ss", "-ltnp"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if f":{port} " in line and "pid=" in line:
            return int(line.split("pid=")[1].split(",")[0])
    return None


def _health(port: int):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=3) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


# ---- supervisor -----------------------------------------------------------------------------
def service_run() -> None:
    """Foreground loop: run the server, restart it when it exits (unless a stop was requested)."""
    STATE.mkdir(exist_ok=True)
    STOP_FLAG.unlink(missing_ok=True)
    SUP_PID.write_text(str(os.getpid()))
    backoff = 3
    child: dict = {"proc": None}

    def _on_stop(signum, _frame):
        # systemd stops the whole cgroup: without this the supervisor died on its SIGTERM at once and systemd
        # SIGKILLed the server before its shutdown hook could checkpoint the running job (2026-08-24 night).
        # Now: mark the stop, pass the signal on (a repeated SIGTERM is a no-op for uvicorn) and keep waiting
        # for the server to leave on its own — the unit's TimeoutStopSec bounds it.
        STOP_FLAG.touch()
        p = child["proc"]
        if p is not None and p.poll() is None:
            try:
                p.send_signal(signal.SIGTERM)
            except Exception:
                pass

    if os.name != "nt":
        signal.signal(signal.SIGTERM, _on_stop)
        signal.signal(signal.SIGINT, _on_stop)
    while True:
        cfg = load_config()
        token = ensure_token()
        env = {**os.environ, "MOTE_TOKEN": token, "PYTHONIOENCODING": "utf-8"}
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # docs/shape.md, daemon (2026-08-24)
        args = [str(PY), "-m", "mote.serve.app", "--checkpoint", cfg["checkpoint"], "--device", cfg["device"],
                "--port", str(cfg["port"]), "--host", cfg.get("host", "127.0.0.1")]
        with open(LOG_FILE, "a", encoding="utf-8") as log:
            log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} starting: {' '.join(args[2:])}\n")
            log.flush()
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            proc = subprocess.Popen(args, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
            child["proc"] = proc
            SRV_PID.write_text(str(proc.pid))
            t0 = time.time()
            if STOP_FLAG.exists():  # a stop arrived between the launch and here
                _on_stop(None, None)
            rc = proc.wait()
            child["proc"] = None
            log.write(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} server exited with {rc}\n")
        SRV_PID.unlink(missing_ok=True)
        if STOP_FLAG.exists():
            STOP_FLAG.unlink(missing_ok=True)
            break
        backoff = 3 if time.time() - t0 > 60 else min(backoff * 2, 60)  # crash loops slow down
        time.sleep(backoff)
    SUP_PID.unlink(missing_ok=True)


def service_start(force: bool = False) -> int:
    cfg = load_config()
    ensure_token()
    if _pid(SUP_PID):
        print(f"already running (supervisor pid {_pid(SUP_PID)}); use `mote restart` to reload")
        return 0
    owner = _port_owner(cfg["port"])
    if owner:
        if not force:
            print(f"port {cfg['port']} is held by pid {owner} (a studio started by hand?). Stop it, or run `mote service start --force` to take the port over.")
            return 1
        _kill(owner)
        time.sleep(1)
    flags = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW) if os.name == "nt" else 0
    subprocess.Popen([str(PYW if PYW.exists() else PY), "-m", "mote.cli", "service", "run"], cwd=ROOT, creationflags=flags,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=(os.name != "nt"))
    print("supervisor started; waiting for the studio ...", end="", flush=True)
    for _ in range(90):
        time.sleep(2)
        h = _health(cfg["port"])
        if h and h.get("model_loaded"):
            print(f" up at http://127.0.0.1:{cfg['port']}  (pair devices: http://127.0.0.1:{cfg['port']}/pair)")
            return 0
        print(".", end="", flush=True)
    print(" not healthy yet — check `mote logs`")
    return 1


def service_stop() -> int:
    sup, srv = _pid(SUP_PID), _pid(SRV_PID)
    if sup:
        STOP_FLAG.touch()
        _kill(sup)
    if srv:
        _kill(srv)
    SUP_PID.unlink(missing_ok=True)
    SRV_PID.unlink(missing_ok=True)
    print("stopped" if (sup or srv) else "nothing was running")
    return 0


def service_restart() -> int:
    if not _pid(SUP_PID):
        return service_start()
    srv = _pid(SRV_PID)
    if srv:
        _kill(srv)  # the supervisor relaunches it with the current config
    cfg = load_config()
    print("restarting ...", end="", flush=True)
    time.sleep(4)
    for _ in range(90):
        time.sleep(2)
        h = _health(cfg["port"])
        if h and h.get("model_loaded") and _pid(SRV_PID) not in (None, srv):
            print(f" up at http://127.0.0.1:{cfg['port']}")
            return 0
        print(".", end="", flush=True)
    print(" not healthy yet — check `mote logs`")
    return 1


def service_status() -> int:
    cfg = load_config()
    sup, srv = _pid(SUP_PID), _pid(SRV_PID)
    h = _health(cfg["port"])
    print(f"supervisor: {'running pid ' + str(sup) if sup else 'not running'}")
    print(f"server:     {'running pid ' + str(srv) if srv else 'not running'}")
    print(f"health:     {h if h else 'no answer on port ' + str(cfg['port'])}")
    print(f"config:     {json.dumps(cfg)}")
    print(f"login item: {'installed' if _startup_entry().exists() else 'not installed'}")
    ts = _tailscale_url()
    print(f"tailscale:  {ts or 'serve not configured (run: tailscale serve --bg http://127.0.0.1:' + str(cfg['port']) + ')'}")
    return 0


# ---- login item ----------------------------------------------------------------------------
def _startup_entry() -> Path:
    if os.name == "nt":
        return Path(os.environ["APPDATA"]) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup" / "Mote Studio.vbs"
    return Path.home() / ".config" / "systemd" / "user" / "mote.service"


def service_install() -> int:
    ensure_token()
    save_config(load_config())
    entry = _startup_entry()
    entry.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        exe = PYW if PYW.exists() else PY
        entry.write_text(
            f'Set sh = CreateObject("WScript.Shell")\nsh.CurrentDirectory = "{ROOT}"\n'
            f'sh.Run """{exe}"" -m mote.cli service run", 0, False\n', encoding="utf-8")
        print(f"login item written: {entry}")
    else:
        entry.write_text(
            "[Unit]\nDescription=Mote Studio\nAfter=network-online.target\n\n[Service]\n"
            f"WorkingDirectory={ROOT}\nExecStart={PY} -m mote.cli service run\nRestart=always\n"
            "TimeoutStopSec=180\n"  # a stop lets the running job checkpoint at its next step boundary first
            "\n[Install]\nWantedBy=default.target\n",
            encoding="utf-8")
        subprocess.run(["systemctl", "--user", "daemon-reload"])
        subprocess.run(["systemctl", "--user", "enable", "mote.service"])
        print(f"systemd user unit written and enabled: {entry}")
    print(f"token file: {TOKEN_FILE}   config: {CONFIG_FILE}")
    return service_start()


def service_uninstall() -> int:
    service_stop()
    entry = _startup_entry()
    if entry.exists():
        if os.name != "nt":
            subprocess.run(["systemctl", "--user", "disable", "mote.service"])
        entry.unlink()
        print(f"removed {entry}")
    return 0


# ---- build / pair / logs / config -----------------------------------------------------------
def _tailscale_url():
    try:
        from mote.serve.pairing import detect_tailscale_url

        return detect_tailscale_url()
    except Exception:
        return None


def build(skip_tests: bool = False) -> int:
    npm = "npm.cmd" if os.name == "nt" else "npm"
    steps = [("web: check", [npm, "run", "check"], ROOT / "web"), ("web: build", [npm, "run", "build"], ROOT / "web")]
    if not skip_tests:
        steps.append(("tests", [str(PY), "-m", "pytest", "-q", "tests/"], ROOT))
    for name, cmd, cwd in steps:
        print(f"-- {name}", flush=True)
        rc = subprocess.run(cmd, cwd=cwd).returncode
        if rc != 0:
            print(f"{name} failed (exit {rc}); the studio was NOT restarted")
            return rc
    rc = service_restart()
    if rc == 0:
        cfg = load_config()
        ts = _tailscale_url()
        print(f"pair devices: http://127.0.0.1:{cfg['port']}/pair" + (f"   phone: {ts}/" if ts else ""))
    return rc


def pair() -> int:
    cfg = load_config()
    url = f"http://127.0.0.1:{cfg['port']}/pair"
    if os.name == "nt":
        os.startfile(url)  # type: ignore[attr-defined]
    else:
        subprocess.Popen(["xdg-open", url])
    print(url)
    return 0


def logs(n: int) -> int:
    if not LOG_FILE.exists():
        print("no log yet")
        return 0
    lines = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
    print("\n".join(lines[-n:]))
    return 0


def config_cmd(args) -> int:
    cfg = load_config()
    for k in ("checkpoint", "device", "port", "host"):
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v
    save_config(cfg)
    print(json.dumps(cfg, indent=1))
    if any(getattr(args, k, None) is not None for k in ("checkpoint", "device", "port", "host")):
        print("run `mote restart` to apply")
    return 0


def prefs_cmd(args) -> int:
    """Preference votes (docs/prefs.md): hand unrated pairs to the rater, take verdicts back, read the numbers."""
    from mote.serve.prefs import PrefStore, rubric

    store = PrefStore()
    if args.action == "export":
        n = store.export_for_rating(Path(args.out), args.limit)
        print(f"{n} pairs -> {args.out}  (rubric {rubric()['hash']})")
    elif args.action == "import":
        if not args.file:
            print("usage: mote prefs import <verdicts.jsonl>")
            return 2
        print(f"{store.import_verdicts(Path(args.file))} verdicts imported")
    elif args.action == "summary":
        print(json.dumps(store.summary(), indent=1, ensure_ascii=False))
    elif args.action == "disagreements":
        rows = store.disagreements()
        if not rows:
            print("no disagreements")
        for d in rows:
            kind = "HARD" if d["hard"] else "soft"
            last = d["messages"][-1]["content"] if d["messages"] else ""
            print(f"[{kind}] {d['id']}  user={d['user']} ({d['user_reason']})  claude={d['claude']} ({d['claude_reason']})")
            print(f"    prompt: {last[:160]}")
            print(f"    A: {d['a'][:200]}")
            print(f"    B: {d['b'][:200]}")
    return 0


def _api(method: str, path: str, body=None):
    import urllib.request

    cfg = load_config()
    tok = ensure_token()
    req = urllib.request.Request(f"http://127.0.0.1:{cfg['port']}{path}", method=method,
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {tok}"},
                                 data=json.dumps(body).encode() if body is not None else None)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def train_cmd(args) -> int:
    """Training jobs on the resident studio (docs/shape.md): start | stop | queue."""
    if args.action == "start":
        if not args.train_args:
            print("usage: mote train start -- --preset local --data data/local_mix --out runs/x ...")
            return 2
        out = _api("POST", "/api/training/start", {"args": args.train_args, "front": bool(args.front), "serve": bool(args.serve)})
        print(json.dumps(out, indent=1))
    elif args.action == "stop":
        print(json.dumps(_api("POST", "/api/training/stop", {"id": args.id}), indent=1))
    elif args.action == "queue":
        print(json.dumps(_api("GET", "/api/training/queue"), indent=1))
    return 0


# ---- entry ---------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="mote", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("service")
    s.add_argument("action", choices=["install", "uninstall", "start", "stop", "restart", "status", "run"])
    s.add_argument("--force", action="store_true", help="take over the port from a studio started by hand")
    sub.add_parser("restart")
    sub.add_parser("status")
    b = sub.add_parser("build")
    b.add_argument("--skip-tests", action="store_true")
    sub.add_parser("pair")
    lg = sub.add_parser("logs")
    lg.add_argument("-n", type=int, default=80)
    c = sub.add_parser("config")
    c.add_argument("--checkpoint")
    c.add_argument("--device", choices=["cpu", "cuda"])
    c.add_argument("--port", type=int)
    c.add_argument("--host")
    pr = sub.add_parser("prefs", help="preference votes: export | import | summary | disagreements (docs/prefs.md)")
    pr.add_argument("action", choices=["export", "import", "summary", "disagreements"])
    pr.add_argument("file", nargs="?", help="verdicts JSONL for `import`")
    pr.add_argument("--out", default="data/prefs/to_rate.jsonl")
    pr.add_argument("--limit", type=int, default=None)
    tr = sub.add_parser("train", help="training jobs on the resident studio: start -- <train args> | stop [--id] | queue")
    tr.add_argument("action", choices=["start", "stop", "queue"])
    tr.add_argument("--id", default=None, help="job id for `stop` (default: the running one)")
    tr.add_argument("--front", action="store_true", help="start: put the job ahead of everything queued")
    tr.add_argument("--serve", action="store_true",
                    help="start: put the job on the air — its EMA answers chats while it runs and its final checkpoint becomes the pin (the trunk and the branches; never an arm)")
    tr.add_argument("train_args", nargs=argparse.REMAINDER, help="after `--`: args for python -m mote.train.train")
    args = ap.parse_args(argv)
    if getattr(args, "cmd", None) == "train":
        # options typed after the action land in the REMAINDER (`train stop --id X` once cancelled the RUNNING
        # job, 2026-08-24): parse them out of everything before the `--`
        lead, rest = args.train_args, []
        if "--" in lead:
            k = lead.index("--")
            lead, rest = lead[:k], lead[k + 1:]
        opts = argparse.ArgumentParser(add_help=False)
        opts.add_argument("--id", default=None)
        opts.add_argument("--front", action="store_true")
        opts.add_argument("--serve", action="store_true")
        ns, unknown = opts.parse_known_args(lead)
        args.id = args.id or ns.id
        args.front = args.front or ns.front
        args.serve = args.serve or ns.serve
        args.train_args = unknown + rest

    if args.cmd == "service":
        return {"install": service_install, "uninstall": service_uninstall, "start": lambda: service_start(args.force),
                "stop": service_stop, "restart": service_restart, "status": service_status, "run": lambda: (service_run(), 0)[1]}[args.action]()
    if args.cmd == "restart":
        return service_restart()
    if args.cmd == "status":
        return service_status()
    if args.cmd == "build":
        return build(args.skip_tests)
    if args.cmd == "pair":
        return pair()
    if args.cmd == "logs":
        return logs(args.n)
    if args.cmd == "config":
        return config_cmd(args)
    if args.cmd == "prefs":
        return prefs_cmd(args)
    if args.cmd == "train":
        return train_cmd(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
