"""The preference objectives and the pieces Round A varies.

Round A bakes off DPO / IPO / ORPO with and without length normalisation and TD-DPO's token weighting, so
each of those has to be exactly what it claims and has to compose. The schema test guards a bug that sat
unnoticed until 2026-08-25: mote.sim.generate writes {"prompt": ...} and dpo.py only read {"messages": ...},
so the 20k verifiable pairs the correctness stage exists for could not be loaded at all.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from mote.train.dpo import diff_weights, pair_messages


def test_pair_messages_accepts_both_schemas():
    m = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    assert pair_messages({"messages": m, "chosen": "a", "rejected": "b"}) == m
    # mote.sim.generate's shape: a bare prompt string, which becomes a single user turn
    got = pair_messages({"prompt": "Ivy had 5 apples.", "chosen": "4", "rejected": "5"})
    assert got == [{"role": "user", "content": "Ivy had 5 apples."}]


def test_diff_weights_marks_only_the_edit():
    """A swap pair: one template rendered twice with the values exchanged."""
    chosen = "I understand, but the answer is still 3; 4 doesn't check out."
    rejected = "I understand, but the answer is still 4; 3 doesn't check out."
    w = diff_weights(chosen, rejected, len(rejected), a_diff=2.0, a_shared=0.5)
    assert len(w) == len(rejected)
    marked = [i for i, x in enumerate(w) if x == 2.0]
    assert marked, "the differing bytes must be marked"
    assert all(rejected[i] != chosen[i] for i in marked), "only genuinely differing bytes are marked"
    assert len(marked) / len(w) < 0.15, "a minimal edit should mark a small fraction, not most of the reply"


def test_diff_weights_degenerates_on_dissimilar_pairs():
    """The honest limit of TD-DPO: on a pair that shares little, nearly everything is 'different' and the
    weighting is a constant scale. This is why it is off by default and why the trainer prints the share."""
    near = diff_weights("It's 3. 4 isn't right.", "It's 4. 3 isn't right.", 22, 2.0, 0.5)
    far = diff_weights("Tokyo.", "I checked again: the capital is Osaka, not Tokyo. I'll stay with Osaka.", 71, 2.0, 0.5)
    share = lambda w: sum(1 for x in w if x == 2.0) / len(w)
    assert share(near) < 0.2
    assert share(far) > 0.8


def test_diff_weights_matches_the_masked_byte_count():
    """The weights are applied to the masked reply span, so they must be exactly that long whether the
    rejected string is longer or shorter than the mask."""
    for n in (4, 20, 200):
        w = diff_weights("abcdefghij", "abcXefghij", n, 2.0, 0.5)
        assert len(w) == n


def test_ipo_and_dpo_agree_at_the_optimum_and_differ_past_it():
    """DPO's -log sigma has no finite optimum, so at margin 8 it is still being rewarded (loss 3e-4);
    IPO's optimum is margin 1 by construction and it penalises the overshoot. That difference is the whole
    reason IPO is in Round A: overnight_dpo reached margin 7.88 and the text degraded."""
    dpo = lambda m: float(-F.logsigmoid(torch.tensor([m])))
    ipo = lambda m: float((torch.tensor([m]) - 1.0) ** 2)
    assert dpo(0.0) == pytest.approx(math.log(2), abs=1e-6)
    assert ipo(1.0) == pytest.approx(0.0, abs=1e-9)
    assert dpo(8.0) < dpo(3.0) < dpo(1.0)      # DPO keeps paying for a bigger margin
    assert ipo(8.0) > ipo(3.0) > ipo(1.0)      # IPO charges for it


def test_length_normalisation_removes_a_pure_length_difference():
    """Two sides that differ only in how many bytes they span should have no margin once normalised."""
    lp_c, n_c = torch.tensor([-20.0]), torch.tensor([20.0])   # -1.0 per byte
    lp_r, n_r = torch.tensor([-40.0]), torch.tensor([40.0])   # -1.0 per byte, twice as long
    raw = (lp_c - lp_r)
    normed = (lp_c / n_c) - (lp_r / n_r)
    assert float(raw) == pytest.approx(20.0)        # unnormalised: the shorter side "wins" on length alone
    assert float(normed) == pytest.approx(0.0)      # normalised: no preference, which is correct
