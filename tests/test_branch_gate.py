"""The mid-training verdict rule (mote/eval/branch_gate.py, docs/shape.md § pipeline): anneal ships on ≥ 2 of
3 decider wins with the val-bpb guard intact; the table and the SFT argv are what the driver writes/submits."""

import json

from mote.eval.branch_gate import final_chat_val, render_md, sft_argv, verdict


def test_verdict_rule():
    ctl = {"reading_em": 0.10, "sim_em": 0.20, "chat_val_bpb": 0.900, "val_bpb": 1.100}
    assert verdict(ctl, {"reading_em": 0.12, "sim_em": 0.25, "chat_val_bpb": 0.905, "val_bpb": 1.104})["winner"] == "anneal"  # 2 wins, guard ok
    assert verdict(ctl, {"reading_em": 0.12, "sim_em": 0.25, "chat_val_bpb": 0.905, "val_bpb": 1.106})["winner"] == "control"  # guard tripped
    assert verdict(ctl, {"reading_em": 0.12, "sim_em": 0.15, "chat_val_bpb": 0.905, "val_bpb": 1.100})["winner"] == "control"  # 1 win
    v = verdict(ctl, {"reading_em": 0.10, "sim_em": 0.30, "chat_val_bpb": 0.890, "val_bpb": 1.095})
    assert v["winner"] == "anneal" and v["wins"] == ["sim_em", "chat_val_bpb"] and v["deltas"]["val_bpb"] < 0
    assert verdict(ctl, {"reading_em": None, "sim_em": 0.3, "chat_val_bpb": 0.8, "val_bpb": None})["winner"] == "control"  # missing = no guard


def test_render_and_argv(tmp_path):
    results = {"control": {"val_bpb": 1.1, "reading_em": 0.1, "sim_em": 0.2, "chat_val_bpb": 0.9, "domains": {"math": 1.0}},
               "anneal": {"val_bpb": 1.102, "reading_em": 0.12, "sim_em": 0.25, "chat_val_bpb": 0.89, "domains": {"math": 0.95}}}
    md = render_md(results, verdict(results["control"], results["anneal"]), "Branch gate")
    assert "**Verdict: anneal**" in md and "| val_bpb:math | 1.0000 | 0.9500 |" in md and "| reading_f1 | — | — |" in md
    argv = sft_argv("--preset flagship --data data/sft_local --sft --mix data/sim_sft:0.10", "runs/b/last.pt", "runs/b_sft")
    assert argv[-4:] == ["--init-from", "runs/b/last.pt", "--out", "runs/b_sft"] and "--sft" in argv
    (tmp_path / "log.jsonl").write_text("\n".join([json.dumps({"lr": 1}), json.dumps({"eval": {"val_bpb": 0.95}}),
                                                    "garbage", json.dumps({"eval": {"val_bpb": 0.91}, "final": True})]))
    assert final_chat_val(tmp_path) == 0.91
