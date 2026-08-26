"""The mid-training verdict rule (mote/eval/branch_gate.py, docs/shape.md § mid).

Re-signed 2026-08-26: one decider, the rest guards. The old rule shipped the anneal on ≥ 2 of
{reading EM, sim-QA EM, chat val bpb}, and every one of those three favoured the anneal by construction —
only the anneal carried the sim, chat and identity extras, so it was measured on data only it had seen.
The extras are in both branches now and the decider is `proxy_track` — mean reciprocal rank of the
expert's next byte, chosen out of a 12-metric library by measuring which cells reproduce a known quality
ordering (the table is in `mote.eval.proxy`). Exact match cannot do that at this scale: docs/search.md
records a flat 0 on reading at 35M, and 2605.18607 §5.2 is the measurement of why a metric that cannot
discriminate should not cast a vote. `needle_auto` was measured and then ignored by the
verdict while the same reweighting cut the long-document share — it is a guard.
"""

import json

import pytest

from mote.eval.branch_gate import final_chat_val, render_md, sft_argv, verdict

CTL = {"proxy_track": 0.300, "val_bpb": 1.100, "needle_auto": 0.50, "false_fire_rate": 0.20,
       "recovery_rate": 0.40}


def _anneal(**kw):
    return {**CTL, **kw}


def test_the_decider_decides():
    assert verdict(CTL, _anneal(proxy_track=0.33))["winner"] == "anneal"
    assert verdict(CTL, _anneal(proxy_track=0.28))["winner"] == "control"
    assert verdict(CTL, _anneal())["winner"] == "control"  # a tie is not a win; control is the default


def test_every_guard_can_block_a_winning_decider():
    """A better decider is necessary and not sufficient. Each guard alone sends the verdict back to
    control, and `needle_auto` is in that list because the ANNEAL reweighting trades long documents for
    math — the direction PRISM (2603.17074 §8.1) measured taking RULER@128k from 59.09 to 6.46."""
    win = {"proxy_track": 0.33}
    assert verdict(CTL, _anneal(**win))["winner"] == "anneal"
    assert verdict(CTL, _anneal(**win, val_bpb=1.104))["winner"] == "anneal"      # inside the 0.005 guard
    assert verdict(CTL, _anneal(**win, val_bpb=1.106))["winner"] == "control"     # outside it
    assert verdict(CTL, _anneal(**win, needle_auto=0.40))["winner"] == "control"  # long-context regression
    assert verdict(CTL, _anneal(**win, false_fire_rate=0.30))["winner"] == "control"
    # a mixture that teaches the world but not what to do when it refuses is not ready for RLVR-1
    assert verdict(CTL, _anneal(**win, recovery_rate=0.30))["winner"] == "control"


def test_missing_numbers_fail_closed():
    """An arm that failed to measure must not be promoted by the absence of evidence."""
    assert verdict(CTL, {k: v for k, v in _anneal(proxy_track=0.33).items() if k != "proxy_track"})["winner"] == "control"
    for missing in ("val_bpb", "needle_auto", "false_fire_rate", "recovery_rate"):
        a = {k: v for k, v in _anneal(proxy_track=0.33).items() if k != missing}
        v = verdict(CTL, a)
        assert v["winner"] == "control" and not v["guard_ok"], missing


def test_a_delta_inside_the_noise_is_not_a_decision():
    """Measured 2026-08-26: at n=120 the gap between the best and worst of three checkpoints whose order
    is known from 12-hour runs is 2.3 standard errors. A branch difference smaller than the combined sem
    is a coin flip, and a coin flip must not ship a 1.4-GPU-day branch."""
    ctl = {**CTL, "proxy_track_sem": 0.010}
    ann = {**CTL, "proxy_track_sem": 0.010}
    assert verdict(ctl, {**ann, "proxy_track": 0.310})["winner"] == "control"   # +0.010 < sqrt(2)*0.010
    assert verdict(ctl, {**ann, "proxy_track": 0.330})["winner"] == "anneal"    # +0.030, clear of it
    v = verdict(ctl, {**ann, "proxy_track": 0.310})
    assert v["noise"] == pytest.approx(0.010 * 2 ** 0.5) and not v["decided"] and v["guard_ok"]
    # with no sem reported at all the rule degrades to sign only, rather than blocking every verdict
    assert verdict(CTL, _anneal(proxy_track=0.301))["winner"] == "anneal"


def test_exact_match_no_longer_votes():
    """reading EM and sim EM stay in the table and out of the verdict. An anneal that sweeps both while
    losing the decider still loses — that is the whole point of the change."""
    v = verdict(CTL, _anneal(proxy_track=0.29, reading_em=1.0, sim_em=1.0, chat_val_bpb=0.0))
    assert v["winner"] == "control" and v["decider"] == "proxy_track"


def test_render(tmp_path):
    results = {"control": {**CTL, "reading_em": 0.10, "sim_em": 0.20, "chat_val_bpb": 0.9, "domains": {"math": 1.0}},
               "anneal": {**CTL, "proxy_track": 0.33, "val_bpb": 1.102, "reading_em": 0.12, "sim_em": 0.25,
                          "chat_val_bpb": 0.89, "domains": {"math": 0.95}}}
    md = render_md(results, verdict(results["control"], results["anneal"]), "Branch gate")
    assert "**Verdict: anneal**" in md and "proxy_track" in md
    assert "| val_bpb:math | 1.0000 | 0.9500 |" in md and "| reading_f1 | — | — |" in md
    # the reader has to be able to see that the exact-match rows did not decide anything
    assert "reported, not voted" in md


def test_sft_argv_and_chat_val(tmp_path):
    argv = sft_argv("--preset flagship --data data/sft_local --sft --mix data/sim_sft:0.10", "runs/b/last.pt", "runs/b_sft")
    assert argv[-4:] == ["--init-from", "runs/b/last.pt", "--out", "runs/b_sft"] and "--sft" in argv
    (tmp_path / "log.jsonl").write_text("\n".join([json.dumps({"lr": 1}), json.dumps({"eval": {"val_bpb": 0.95}}),
                                                    "garbage", json.dumps({"eval": {"val_bpb": 0.91}, "final": True})]))
    assert final_chat_val(tmp_path) == 0.91
