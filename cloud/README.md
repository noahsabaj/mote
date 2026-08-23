# Flagship run on Lightning.ai — runbook

Nothing in here runs until you type `--go`. Budget: 25 credits total.

## 0. Before spending anything
1. Check the **interruptible H100** hourly rate on the machine picker. Third-party trackers list the on-demand H100 at
   ~$3.29/h with interruptible discounts "up to 80%"; your own account page showed $3.52/h on-demand. The whole plan
   hinges on this number: at ~$0.7–1.8/h interruptible you have 14–35 hours, at $3.52/h you have 7.
2. `python cloud/launch.py plan --minutes <pretrain minutes> --gb <pretraining GB>` prints the commands it would run.
3. The local pilot must have passed the gate: word-like chunk boundaries, stable Relation training (see `runs/pilot_1h/log.jsonl`).

## 1. Data (CPU machine — cheap)
```bash
python cloud/launch.py up --go                 # CPU studio "mote-flagship", sync repo, run cloud/bootstrap.sh (installs torch/triton/mamba_ssm, runs tests)
python cloud/launch.py data --go --gb 10 --sft-mb 300
python cloud/launch.py logs --file data/build.log
```
Streams ~10 GB of text from the Hub into `data/pretrain_mix.*` and the SFT mix into `data/sft_mix.*` on the studio's
persistent disk. Expect 1–3 hours depending on Hub throughput; the GPU is not needed yet.

## 2. Pretraining (interruptible H100)
```bash
python cloud/launch.py switch --machine H100 --go
python cloud/launch.py train --go --minutes 480      # flagship preset (~100M params), 16×4096 bytes × accum 2 per step
python cloud/launch.py status
python cloud/launch.py logs
```
Watch the first `probe_sec_per_step` / `bytes_per_sec` line and the step-500 eval. If throughput is far below
~500 kB/s, stop and reconsider the model size before burning hours. Checkpoints land in `runs/flagship/last.pt`
every 10 minutes. **After a preemption**: `python cloud/launch.py train --go --resume --minutes <remaining>`.

## 3. SFT (same machine)
```bash
python cloud/launch.py sft --go --sft-minutes 60
python cloud/launch.py logs --file runs/flagship_sft/stdout.log
```

## 4. Bring it home
```bash
python cloud/launch.py pull --run flagship          # -> runs/flagship/{last.pt, log.jsonl, config.json, run.json}
python cloud/launch.py pull --run flagship_sft
python cloud/launch.py stop --go                    # stop the machine (the disk persists; delete the studio manually when done)
python -m mote.serve.app --checkpoint runs/flagship_sft/last.pt
```

## Chunking knob (decided from the local pilots)
The pilot at the paper's α=0.03 settled at ~3.0 bytes/chunk (sub-word); α=0.1 gave ~3.7 bytes/chunk, multi-word chunks,
and better bits/byte at equal wall-clock (faster steps). The launcher therefore passes `--ratio-weight 0.1` with the
default ATDC target schedule 5→6.5. Override with `--ratio-weight` / `--target-ratio INIT FINAL` in `PRETRAIN_CMD` if you
want to push harder (the H-Net paper itself lands at 4.6–4.8 bytes/chunk for a target of 6).

## Notes
* `cloud/bootstrap.sh` pins `state-spaces/mamba` to commit `e9594ce` (the one validated locally) and runs the test suite.
* Everything training-side is identical to the local pilot (`mote/train/train.py`); only the preset, batch and data differ.
* The launcher resolves your user/teamspace from `lightning login` credentials; studio name is `mote-flagship`.
