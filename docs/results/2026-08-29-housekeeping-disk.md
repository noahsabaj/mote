# Housekeeping 2026-08-29 — disk manifest

Signed 2026-08-29 00:40 (grilling round 2, 'Orphan runs + settled arms' checkpoints + pretrain_mix').
Rule: live runs untouched; settled arms keep every log/json (their numbers stay reproducible) and lose `*.pt`; debris dirs and `*.stdout` deleted whole; `data/pretrain_mix.*` (named only by the stale README/cloud docs) and `data/fineweb_edu_pilot.*` deleted.

| action | path | MB |
|---|---|---|
| drop .pt | runs/ab2_attn_2h/last.pt | 252 |
| drop .pt | runs/ab2_bf16res_2h/last.pt | 252 |
| drop .pt | runs/ab2_muon_2h/last.pt | 252 |
| drop .pt | runs/ab2_muon_seed7/last.pt | 252 |
| drop .pt | runs/ab2_muonsw_2h/last.pt | 252 |
| drop .pt | runs/ab2_win128_2h/last.pt | 252 |
| drop .pt | runs/ab2_winctl_2h/last.pt | 252 |
| drop .pt | runs/ab3_jepa_ema/last.pt | 266 |
| drop .pt | runs/ab3_jepa_min/last.pt | 259 |
| drop .pt | runs/ab3_jepa_sig/last.pt | 259 |
| drop .pt | runs/ab_a03_2048/last.pt | 404 |
| delete | runs/ab_a03_2048.stdout | 0 |
| drop .pt | runs/ab_a10_2048/last.pt | 404 |
| delete | runs/ab_a10_2048.stdout | 0 |
| drop .pt | runs/ab_adamw_2048/last.pt | 404 |
| delete | runs/ab_adamw_2048.stdout | 0 |
| drop .pt | runs/ab_adamw_4096/last.pt | 404 |
| delete | runs/ab_adamw_4096.stdout | 0 |
| drop .pt | runs/ab_b299_2048/last.pt | 404 |
| delete | runs/ab_b299_2048.stdout | 0 |
| drop .pt | runs/ab_mbp2_2048/last.pt | 405 |
| delete | runs/ab_mbp2_2048.stdout | 0 |
| drop .pt | runs/ab_muon_2048/last.pt | 280 |
| delete | runs/ab_muon_2048.stdout | 0 |
| drop .pt | runs/ab_muon_2048_r2/last.pt | 280 |
| delete | runs/ab_muon_2048_r2.stdout | 0 |
| drop .pt | runs/ab_muon_4096/last.pt | 280 |
| delete | runs/ab_muon_4096.stdout | 0 |
| drop .pt | runs/ab_muon_4096_eq/last.pt | 280 |
| delete | runs/ab_muon_4096_eq.stdout | 0 |
| drop .pt | runs/ab_muon_mbp2/last.pt | 280 |
| delete | runs/ab_muon_mbp2.stdout | 0 |
| drop .pt | runs/ab_muon_nombp/last.pt | 252 |
| delete | runs/ab_muon_nombp.stdout | 0 |
| drop .pt | runs/ab_muonsw_2048/last.pt | 280 |
| delete | runs/ab_muonsw_2048.stdout | 0 |
| drop .pt | runs/ab_nombp_2048/last.pt | 362 |
| delete | runs/ab_nombp_2048.stdout | 0 |
| delete | runs/daemon_dogfood | 280 |
| drop .pt | runs/lr_sweep_12e-4/last.pt | 758 |
| drop .pt | runs/lr_sweep_3e-4/last.pt | 758 |
| drop .pt | runs/lr_sweep_5e-4/last.pt | 758 |
| drop .pt | runs/lr_sweep_8e-4/last.pt | 758 |
| drop .pt | runs/nsweep_10/last.pt | 758 |
| drop .pt | runs/nsweep_4/last.pt | 758 |
| drop .pt | runs/nsweep_8/last.pt | 758 |
| drop .pt | runs/overnight/last.pt | 404 |
| delete | runs/overnight.params | 0 |
| drop .pt | runs/overnight_dpo/last.pt | 134 |
| delete | runs/overnight_dpo.stdout | 0 |
| drop .pt | runs/overnight_dpo2/last.pt | 134 |
| delete | runs/overnight_dpo2.stdout | 0 |
| drop .pt | runs/overnight_sft/last.pt | 404 |
| drop .pt | runs/overnight_sft2/last.pt | 404 |
| delete | runs/overnight_sft2.stdout | 0 |
| drop .pt | runs/pilot_1h/last.pt | 144 |
| drop .pt | runs/pilot_alpha01/last.pt | 144 |
| delete | runs/pilot_short | 144 |
| delete | runs/smoke_jepa_ema | 13 |
| delete | runs/smoke_jepa_minimal | 13 |
| delete | runs/smoke_jepa_sigreg | 13 |
| delete | runs/smoke_moe_aux | 17 |
| delete | runs/smoke_moe_lf | 18 |
| delete | runs/smoke_relation_v2 | 12 |
| delete | runs/smoke_win | 144 |
| delete | runs/stackval | 504 |
| drop .pt | runs/sweep_a0.1_n4/last.pt | 144 |
| drop .pt | runs/sweep_a0.1_n6/last.pt | 144 |
| drop .pt | runs/sweep_a0.3_n4/last.pt | 144 |
| drop .pt | runs/sweep_a0.3_n6/last.pt | 144 |
| delete | data/fineweb_edu_pilot.meta.json | 0 |
| delete | data/fineweb_edu_pilot.train.bin | 572 |
| delete | data/fineweb_edu_pilot.val.bin | 15 |
| delete | data/pretrain_mix.meta.json | 0 |
| delete | data/pretrain_mix.train.bin | 18851 |
| delete | data/pretrain_mix.val.bin | 122 |

**Freed: 34.8 GB.** Live set kept intact: _serve, elr_gate, flagship_shape_v2, floor, pfail, pilot_sft, probe, qk, roundA, roundA_pairs.jsonl, t3l24_dense_2.5e-4, t3l24_dense_4e-4, t3l24_dense_8e-4, t3l_dense_16e-4, t3l_dense_4e-4, t3l_dense_8e-4, t3l_e4_8e-4, t3l_e8_8e-4, wiki_index.log.
