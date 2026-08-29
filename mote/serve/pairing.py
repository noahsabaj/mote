"""Device pairing: a QR magic link and a 6-digit code, so the access token is never typed.

* ``GET /pair`` (loopback clients only) renders a page with a QR of ``<public_url>/#token=<token>``
  and a fresh 6-digit code. The token travels in the URL *fragment*: the browser never sends a
  fragment to the server, so it appears in no request log; the UI stores it and scrubs the hash.
* ``POST /api/pair {"code": "123456"}`` redeems a code for the token. Codes live 10 minutes, work
  once, and redemption is rate-limited (10 attempts per minute): at most 100 guesses per code
  lifetime against a million codes.
"""

from __future__ import annotations

import html
import json
import secrets
import shutil
import subprocess
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

CODE_TTL_S = 600.0  # ten minutes: read on one device, typed on another
MAX_ATTEMPTS_PER_MIN = 10
LOOPBACK = {"127.0.0.1", "::1", "localhost"}


class Pairing:
    def __init__(self, studio, public_url: Optional[str] = None):
        self.studio = studio  # the app's Studio: the token lives there
        self.public_url = public_url
        self._codes: dict[str, float] = {}  # code -> expiry
        self._attempts: list[float] = []

    # ---- helpers --------------------------------------------------------------------------
    @property
    def token(self) -> Optional[str]:
        return self.studio.token

    def new_code(self) -> tuple[str, float]:
        now = time.time()
        self._codes = {c: e for c, e in self._codes.items() if e > now}
        code = f"{secrets.randbelow(10**6):06d}"
        self._codes[code] = now + CODE_TTL_S
        return code, CODE_TTL_S

    def redeem(self, code: str) -> Optional[str]:
        now = time.time()
        self._attempts = [t for t in self._attempts if t > now - 60]
        if len(self._attempts) >= MAX_ATTEMPTS_PER_MIN:
            raise HTTPException(429, "too many attempts; wait a minute")
        self._attempts.append(now)
        exp = self._codes.pop(code.strip(), None)
        if exp is None or exp < now:
            return None
        return self.token

    def resolve_public_url(self, request: Request) -> str:
        if self.public_url:
            return self.public_url.rstrip("/")
        url = detect_tailscale_url()
        if url:
            self.public_url = url
            return url
        return str(request.base_url).rstrip("/")

    def magic_link(self, base: str) -> str:
        return f"{base}/#token={self.token}"


def detect_tailscale_url() -> Optional[str]:
    """`tailscale serve status --json` → https://<machine>.<tailnet>.ts.net (best effort, 3 s)."""
    exe = shutil.which("tailscale") or shutil.which("tailscale", path=r"C:\Program Files\Tailscale")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "serve", "status", "--json"], capture_output=True, text=True, timeout=3).stdout
        data = json.loads(out or "{}")
        for key in (data.get("Web") or {}):
            host = key.split(":")[0]
            if host.endswith(".ts.net"):
                return f"https://{host}"
    except Exception:
        return None
    return None


def qr_svg(text: str) -> Optional[str]:
    try:
        import qrcode
        import qrcode.image.svg as qsvg
    except Exception:
        return None
    img = qrcode.make(text, image_factory=qsvg.SvgPathImage, box_size=12, border=2, error_correction=qrcode.constants.ERROR_CORRECT_M)
    return img.to_string(encoding="unicode")


def is_loopback(request: Request) -> bool:
    host = request.client.host if request.client else ""
    return host in LOOPBACK


def router(pairing: Pairing) -> APIRouter:
    r = APIRouter()

    @r.get("/pair", response_class=HTMLResponse)
    def pair_page(request: Request):
        if not is_loopback(request):
            raise HTTPException(403, "the pairing page is only served to the machine itself")
        if not pairing.token:
            return HTMLResponse("<p style='font-family:system-ui;padding:2rem'>No access token is configured (start the server with --token), so there is nothing to pair.</p>")
        base = pairing.resolve_public_url(request)
        link = pairing.magic_link(base)
        code, ttl = pairing.new_code()
        svg = qr_svg(link)
        qr_block = svg if svg else "<p class='warn'>QR unavailable: <code>pip install qrcode</code>. Use the code below or the link.</p>"
        return PAGE.format(mark=MARK_SVG, qr=qr_block, code=html.escape(code), ttl=int(ttl), base=html.escape(base), link=html.escape(link))

    @r.get("/pair/code")
    def pair_code(request: Request):
        if not is_loopback(request):
            raise HTTPException(403, "loopback only")
        if not pairing.token:
            raise HTTPException(404, "no token configured")
        code, ttl = pairing.new_code()
        return {"code": code, "ttl": ttl}

    @r.post("/api/pair")
    async def pair_redeem(request: Request):
        body = await request.json()
        code = str(body.get("code", ""))
        if not pairing.token:
            raise HTTPException(404, "no token configured")
        token = pairing.redeem(code)
        if token is None:
            return JSONResponse({"detail": "wrong or expired code — use the code the PC's /pair page shows right now"}, status_code=400)
        return {"token": token}

    return r


# the mark (brand/build.py: the boundary ring, turned) at the 32-px weights; no braces, so safe in PAGE.format
MARK_SVG = ('<svg viewBox="0 0 100 100" aria-hidden="true"><circle cx="50" cy="50" r="27" fill="none" stroke="currentColor"'
            ' stroke-width="8" stroke-dasharray="131.65 38" stroke-dashoffset="150.65" transform="rotate(-45 50 50)"/>'
            '<circle cx="69.09" cy="30.91" r="9.5" fill="currentColor"/></svg>')

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Pair a device · Mote</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body {{ margin:0; background:#faf9f7; color:#1c1a17; font-family:system-ui,-apple-system,'Segoe UI',sans-serif; }}
  main {{ max-width:44rem; margin:0 auto; padding:2.5rem 1.5rem 4rem; }}
  h1 {{ font-family:'Iowan Old Style','Palatino Linotype',Palatino,Georgia,serif; font-weight:600; font-size:1.8rem; margin:0 0 .3rem; }}
  p {{ color:#56514a; line-height:1.5; max-width:60ch; }}
  .grid {{ display:grid; grid-template-columns:minmax(0,18rem) 1fr; gap:2rem; align-items:start; margin-top:1.5rem; }}
  .qr svg {{ width:100%; height:auto; display:block; background:#fff; border:1px solid #e3ded5; border-radius:10px; padding:.5rem; }}
  .code {{ font-family:ui-monospace,'Cascadia Mono',Consolas,monospace; font-size:3rem; letter-spacing:.18em; color:#8f3f18; margin:.2rem 0 .2rem; }}
  .meta {{ font-size:.85rem; color:#6f6960; }}
  code {{ font-family:ui-monospace,Consolas,monospace; font-size:.9em; word-break:break-all; }}
  .warn {{ color:#8f3f18; }}
  .brand {{ display:flex; align-items:center; gap:.45rem; font-weight:600; margin-bottom:1.6rem; }}
  .brand svg {{ width:1.3rem; height:1.3rem; color:#a34a1f; }}
  @media (max-width:40rem) {{ .grid {{ grid-template-columns:1fr; }} }}
  @media (prefers-color-scheme: dark) {{ body {{ background:#131211; color:#ece8e2; }} p,.meta {{ color:#a7a099; }} .code {{ color:#eab183; }} .brand svg {{ color:#e0a070; }} }}
</style></head><body><main>
<div class="brand">{mark}Mote</div>
<h1>Pair a device</h1>
<p>This page is only served to this machine. Scan the QR with the phone's camera — the token rides in the link's fragment, which browsers never send to a server — or type the six digits into the studio's unlock sheet.</p>
<div class="grid">
  <div class="qr">{qr}</div>
  <div>
    <div class="meta">Pairing code · valid for <span id="ttl">{ttl}</span> · single use · a new one appears when it expires</div>
    <div class="code" id="code">{code}</div>
    <div class="meta">Studio address: <code>{base}</code></div>
    <p class="meta" style="margin-top:1.2rem">Link in the QR: <button id="show" type="button" style="font:inherit;color:#8f3f18;background:none;border:0;padding:0;cursor:pointer;text-decoration:underline">show</button> <code id="link" hidden>{link}</code></p>
  </div>
</div>
<script>
  // The link carries the token: off the screen until asked for (a shared desk, a screenshot).
  document.getElementById('show').addEventListener('click', (e) => {{
    const c = document.getElementById('link'); c.hidden = !c.hidden; e.target.textContent = c.hidden ? 'show' : 'hide';
  }});
  // The code stays put until it expires; only then is it replaced (and the page says so).
  let ttl = {ttl};
  const el = document.getElementById('ttl');
  const fmt = (s) => (s >= 60 ? Math.floor(s / 60) + ' min ' + (s % 60) + ' s' : s + ' s');
  el.textContent = fmt(ttl);
  setInterval(async () => {{
    ttl -= 1;
    if (ttl <= 0) {{
      try {{ const r = await fetch('/pair/code'); const j = await r.json(); document.getElementById('code').textContent = j.code; ttl = j.ttl; }} catch (e) {{ ttl = 0; }}
    }}
    el.textContent = fmt(ttl);
  }}, 1000);
</script>
</main></body></html>"""
