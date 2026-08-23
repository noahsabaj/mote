# Reaching the studio from other devices

The studio has one access token and no other login, so the network layer does the real work.
Recommended: **Tailscale**, a private WireGuard mesh between your own devices. Nothing becomes
public, TLS is handled for you, and no router or firewall changes are needed.

## One-time setup (needs you at the PC — installer requires admin)

1. Install Tailscale on the PC (https://tailscale.com/download/windows) and sign in.
   Install the Tailscale app on the phone and sign in with the same account.
2. In the admin console (https://login.tailscale.com/admin/dns) make sure **MagicDNS** and
   **HTTPS certificates** are enabled (both are one click; MagicDNS is on by default).
3. Generate a token and start the studio with it (PowerShell, repo root — the project venv's Python, not the global `python`, which has no torch):
   ```powershell
   $env:MORPHEME_TOKEN = .\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(24))"
   $env:MORPHEME_TOKEN                      # copy this to the phone once
   .\.venv\Scripts\python.exe -m morpheme.serve.app --checkpoint runs/pilot_sft/last.pt --device cpu --port 7861
   ```
   The server keeps listening on `127.0.0.1` only — Tailscale does the exposure.
   Without a token the server refuses to bind anything but loopback (`--no-auth` overrides).
4. Publish the port to your tailnet (PowerShell, once; survives reboots):
   ```powershell
   tailscale serve --bg http://127.0.0.1:7861
   ```
   It prints the URL, `https://<old-node>.<tailnet>.ts.net`. This also reaches a server
   running inside WSL2, because WSL2 forwards its ports to Windows `localhost`.
5. On the phone, open that URL and paste the token when asked. It is stored in the browser;
   change the token on the server to revoke every device at once.
6. Optional: Safari share sheet → **Add to Home Screen**. The studio ships a web-app manifest and
   icons, so it opens full-screen like an app. There is deliberately no service worker — the app
   needs its backend, and a cached shell is the one thing that could go stale after a rebuild. The
   Home Screen app has its own storage, so it asks for the token once more.

`tailscale serve reset` removes the publication. Never use `tailscale funnel` — that is the
public internet, and the studio is not hardened for it.

## Home Wi-Fi only (no install)

```powershell
.\.venv\Scripts\python.exe -m morpheme.serve.app --checkpoint runs/pilot_sft/last.pt --device cpu --port 7861 --host 0.0.0.0 --token <token>
```
then `http://192.168.1.135:7861` from any device on the LAN (Windows Firewall already allows
Python inbound; the IP is the PC's current Ethernet address). WSL2 servers are NAT'd and are
not reachable this way — run the Windows server, or publish via Tailscale as above.

## What the token does

* HTTP (`/api/*`, `/v1/*`): `Authorization: Bearer <token>` on every request, else `401`.
* WebSocket (`/ws/generate`): first frame must be `{"type": "auth", "token": "<token>"}`; the
  server answers `{"type": "auth_ok"}` or closes with code `4401`.
* `/api/health` and the static frontend stay open so the UI can load and ask for the token.
* Comparison is constant-time; the token is never put in a URL.
