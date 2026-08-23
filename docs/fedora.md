# Fedora move

Runbook for moving Mote from Windows + WSL2 to the Fedora dual boot (the NVIDIA driver works there;
WSL2 costs ~2.6 GB of RAM, the 9P file mount, and the WDDM residency crashes that `expandable_segments`
ran into). Order: after the rename, with no training running. Status boxes are filled in as steps land.

## Before leaving Windows

- [ ] `git status` clean, everything pushed to the local repo's history (no remote needed).
- [ ] Copy `~/.claude/projects/D--Code-Workshop-1/memory/` (the assistant's notes) to `docs/_memory-export/`
      or a USB stick — the Fedora install gets a fresh Claude Code memory directory.
- [ ] Note the Tailscale machine name (`<old-node>`) and the studio token in `.mote/token`.
- [ ] Stop the Windows studio service (`.\mote service stop`) and remove the login item
      (`%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\Mote Studio.vbs`).

## On Fedora

1. **Disk**: the repo, `data/` (23 GB) and `runs/` live on ext4, not on the NTFS mount — NTFS through
   `ntfs-3g` is slow for the memmapped shards. Copy once: `rsync -a --info=progress2 /run/media/<user>/<D>/Code/Workshop/1/ ~/mote/`.
2. **GPU**: `nvidia-smi` works (confirmed by Noah). Install CUDA toolkit matching the driver for Triton
   (`sudo dnf install cuda-toolkit` from the NVIDIA repo, or the `nvidia-cuda-toolkit` build that matches).
3. **Python**: `uv venv --python 3.11 .venv && source .venv/bin/activate && uv pip install -e .` then
   `uv pip install torch --index-url https://download.pytorch.org/whl/cu126` (or the wheel matching the
   toolkit), `triton`, and build the Mamba-3 kernels the way `cloud/bootstrap.sh` does.
4. **Tests**: `python -m pytest -q` — the 23 GPU tests that skip on Windows run here; FlashRelation's
   exactness tests against the reference are the gate before any training.
5. **Speed check**: `python -m mote.train.profile_step --data data/local_mix --preset local --init-from runs/overnight/last.pt --batch-size 4 --grad-accum 4 --bucket 64`
   and compare with `docs/results/2026-08-23-chain.md` (80.8 KB/s, 23.5 % MFU on WSL2).
6. **Studio**: `mote service install` writes a systemd *user* unit (`~/.config/systemd/user/mote-studio.service`)
   and enables it; `loginctl enable-linger $USER` keeps it running without a session. Port 7861.
7. **Tailscale**: `sudo dnf install tailscale && sudo systemctl enable --now tailscaled && sudo tailscale up`,
   then `tailscale serve --bg https+insecure://localhost:7861` (or plain `tailscale serve 7861`); the phone
   keeps the same `https://<machine>.<tailnet>.ts.net/` name if the machine name is reused.
8. **Pairing**: open `http://127.0.0.1:7861/pair` on the Fedora desktop and pair the phone again (the
   token moved with `.mote/token`, so the existing PWA keeps working; re-add the tile only for a new icon).
9. **SearXNG** (search design): `podman run -d --name searxng -p 127.0.0.1:8080:8080 docker.io/searxng/searxng`
   with `settings.yml` enabling the JSON format — the server only ever talks to localhost:8080.
10. **WSL leftovers**: nothing in `~/` on the WSL side is needed; the chain scripts there are history.

## After the move

- Block-local window cap, segment-sum dechunk backward, the key-window in FlashRelation, then the
  windowed-vs-full A/B on the 35M (docs/context.md) and the kernel-optimisation loop (docs/results).
- Long Muon-vs-SW run; then the flagship preset is frozen and the 16384 run starts.
