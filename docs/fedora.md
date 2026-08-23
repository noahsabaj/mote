# Fedora move

Runbook for moving Mote from Windows + WSL2 to the Fedora dual boot (the NVIDIA driver works there;
WSL2 costs ~2.6 GB of RAM, the 9P file mount, and the WDDM residency crashes that `expandable_segments`
ran into). Order: after the rename, with no training running. Status boxes are filled in as steps land.

## Before leaving Windows

- [x] `git status` clean (2026-08-23, after the rename), everything pushed to the local repo's history (no remote needed).
- [x] Copied 2026-08-23 `~/.claude/projects/D--Code-Workshop-1/memory/` (the assistant's notes) to `docs/_memory-export/`
      or a USB stick — the Fedora install gets a fresh Claude Code memory directory.
- [x] Note the Tailscale machine name (`<old-node>`) and the studio token in `.mote/token`.
- [ ] Stop the Windows studio service (`.\mote service stop`) and remove the login item
      (`%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\Mote Studio.vbs`).

## Carrying the Claude Code session over

Claude Code keeps everything under `~/.claude`, so Fedora starts empty — but the Windows copy is readable
from the mounted C: drive. Copy the whole project directory (session transcripts + `memory/`) and keep
the same project name, then resume the session by id (resume searches every project since 2.1.223;
this machine runs 2.1.241):

```bash
WIN=/run/media/$USER/<C-drive-label>/Users/noahs/.claude/projects/C--Code-mote   # the project moved to C:\Code\mote on 2026-08-23
mkdir -p ~/.claude/projects && cp -r "$WIN" ~/.claude/projects/
cd ~/mote && CLAUDE_CODE_PROJECT_DIR_NAME=C--Code-mote claude --resume a5c0fd49-b82d-4d4f-840f-ece5aa374fcf
```

`CLAUDE_CODE_PROJECT_DIR_NAME` (2.1.234+) pins the project name so new sessions and the auto-memory land in
the same directory instead of `-home-noahs-mote` (the Windows sessions before the C: move sit in `D--Code-Workshop-1`, copy that too if you want them). Log in again on Fedora (the login is per machine). The
transcript format is internal and version-dependent: if the resume misbehaves, start fresh — the memory
notes (`~/.claude/projects/C--Code-mote/memory/`, mirrored in `docs/_memory-export/`) plus
`docs/shape.md`, `docs/prefs.md`, `docs/context.md` carry every decision, and that is a lighter start
than a 28 MB, thrice-compacted transcript.

## Decisions (grilling, 2026-08-23 afternoon)

* **Hand-off**: the copy on Fedora's root (`~/mote`) is canonical. A bare repo on the USB disk is the backup
  remote, readable from both OSes: `git init --bare /run/media/$USER/<D>/Code/mote.git` then
  `git remote add backup …` and `git push backup --all` after every session. The Windows working copy on D:
  stays untouched until the Windows clean-up below. (Done on Windows 2026-08-23: the working copy moved to `C:\Code\mote`, `D:\Code\mote.git` is the backup remote, `D:\Code\Workshop` is the archive.)
* **Phone**: Fedora joins Tailscale as its own node (Noah installs Tailscale); re-pair the phone once and
  re-add the home-screen tile for the new origin. The Windows node stays valid for boots back into Windows.
* **Disks, measured**: D: (USB flash, NTFS) 389 MiB/s sequential, ~9 000 random 4 KiB reads/s; C: (NVMe)
  8–40× that. Training never needed more than the USB disk gives (tens of reads per step); small-file work
  (git, npm, tests) is what it slowed. Nothing moves to C:.
* **Copy**: `data/` 29 GB + `runs/` 7.4 GB + the repo, onto the 212 GB Linux root. The venv, `node_modules`
  and the Mamba checkout are rebuilt, not copied.
* **Mamba kernels**: the code imports only Triton kernels (`mamba3_siso_combined`, `ssd_combined`); WSL
  never had `nvcc`. On Fedora: `git clone https://github.com/state-spaces/mamba.git ~/mamba && git -C ~/mamba
  checkout e9594ce && MAMBA_SKIP_CUDA_BUILD=TRUE uv pip install -e ~/mamba --no-build-isolation`.
* **Studio on Fedora** serves on `cuda` (Triton present), which is also what day-one step 5 measures.
* **Windows clean-up, after day one passes and the backup remote holds the full history**: `mote service
  uninstall` (done at leave time), then delete the WSL distro (`wsl --unregister <distro>`, ~36 GB on C:;
  the hnet-venv and chain logs go with it — the results are in docs/results). **Not before Noah says so.**

## On Fedora

1. **Disk**: the repo, `data/` (23 GB) and `runs/` go on Fedora's own partition on the NVMe (disk 0 has a
   2 GB Linux boot partition and a 212 GB Linux root — Fedora's default is Btrfs; ext4 only if chosen at install;
   `df -hT /` tells). Both C: and D: stay NTFS and are only read from. D: is a 934 GB **USB flash disk**, the
   worst place for memmapped shards, so the copy is the single biggest I/O change of the move. Copy once:
   `rsync -a --info=progress2 /run/media/<user>/<C>/Code/mote/ ~/mote/` (the repo moved from the USB disk D: to C:\Code\mote on 2026-08-23; the bare backup repo is `D:\Code\mote.git`). On Btrfs, mark the big-file
   directories no-CoW before filling them: `chattr +C ~/mote/data ~/mote/runs`.
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
10. **WSL leftovers**: nothing in `~/` on the WSL side is needed (HF cache empty, chain scripts are history);
    the distro is deleted in the Windows clean-up step above, not before.

## After the move

- Block-local window cap, segment-sum dechunk backward, the key-window in FlashRelation, then the
  windowed-vs-full A/B on the 35M (docs/context.md) and the kernel-optimisation loop (docs/results).
- Long Muon-vs-SW run; then the flagship preset is frozen and the 16384 run starts.
