# Reaching the studio from other devices

The studio has one access token and no other login, so the network layer does the real work.
Recommended: **Tailscale**, a private WireGuard mesh between your own devices. Nothing becomes
public, TLS is handled for you, and no router or firewall changes are needed. (Rewritten for the
Fedora box on 2026-08-24; the Windows/WSL2 version of this page is in git history.)

## One-time setup (done on `<your-node>`, kept here so it can be redone)

1. Tailscale on the PC: `sudo dnf install tailscale && sudo systemctl enable --now tailscaled &&
   sudo tailscale up`, then once `sudo tailscale set --operator=$USER` so `tailscale serve` works
   without sudo. Install the Tailscale app on the phone and sign in with the same account.
2. In the admin console (https://login.tailscale.com/admin/dns) make sure **MagicDNS** and
   **HTTPS certificates** are enabled (both are one click; MagicDNS is on by default).
3. Install the studio as a user service (repo root):
   ```bash
   .venv/bin/python -m mote.cli service install    # writes ~/.config/systemd/user/mote.service, enables it
   loginctl enable-linger $USER                     # keeps it running with nobody logged in
   ```
   This creates the access token (`.mote/token`), the config (`.mote/config.json`: checkpoint, device,
   port 7861) and the unit; a supervisor keeps the server alive. `systemctl --user restart mote`
   stops the running training job at a step boundary (checkpoint kept, resumed in front of the queue)
   and comes back with the current code and `web/dist`. The server listens on `127.0.0.1` only —
   Tailscale does the exposure.
4. Publish the port to your tailnet (once; survives reboots):
   ```bash
   tailscale serve --bg http://127.0.0.1:7861
   ```
   It prints the URL — `https://<your-node>.<your-tailnet>.ts.net/` on this tailnet.
5. Pair the phone without typing the token: on the PC open **http://127.0.0.1:7861/pair** (served only
   to the machine itself). Scan its QR with the phone's camera — the link carries the token in the URL
   fragment, which browsers never send to a server — or type the six-digit code it shows into the
   studio's unlock sheet (codes last 10 minutes, work once, 10 attempts per minute). The token is then
   stored in that browser; change the token on the server to revoke every device at once.
6. Optional: Safari share sheet → **Add to Home Screen** (Chrome: Install app). The studio ships a
   web-app manifest and icons, so it opens full-screen like an app. There is deliberately no service
   worker — the app needs its backend, and a cached shell is the one thing that could go stale after a
   rebuild. The Home Screen app has its own storage, so it asks for the token once more.

`tailscale serve --https=443 off` removes the publication. Never use `tailscale funnel` — that is the
public internet, and the studio is not hardened for it.

## Home Wi-Fi only (no Tailscale)

```bash
.venv/bin/python -m mote.serve.app --checkpoint runs/overnight_sft/last.pt --device cuda --port 7861 --host 0.0.0.0 --token "$(cat .mote/token)"
```
then `http://<the PC's LAN address>:7861` from any device on the LAN (`firewall-cmd --add-port=7861/tcp`
if firewalld blocks it). Plain `http://` is not a secure context in the phone's browser, so **Copy**
buttons fall back to a selection-based copy there; everything else works. Prefer the Tailscale URL.

## What the token does

* HTTP (`/api/*`, `/v1/*`): `Authorization: Bearer <token>` on every request, else `401`.
* WebSocket (`/ws/generate`): first frame must be `{"type": "auth", "token": "<token>"}`; the
  server answers `{"type": "auth_ok"}` or closes with code `4401`.
* `/api/health`, `/api/pair` (code redemption) and the static frontend stay open so the UI can load and pair.
* `/pair` and `/pair/code` answer only loopback clients; `--public-url` (or `MOTE_URL`) overrides the
  address encoded in the QR (default: detected from `tailscale serve status`).
* Comparison is constant-time; the token is never put in a URL.
