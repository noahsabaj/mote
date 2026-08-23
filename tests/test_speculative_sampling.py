"""The verification rule must reproduce the target distribution exactly (Monte Carlo), whatever the draft."""

import torch

from morpheme.serve.engine import _dist, _draw, verify_draft


def _simulate(p, q, n=40000, seed=0):
    """One draft position: draw x ~ q, verify against p; the emitted byte is x if accepted else the correction."""
    torch.manual_seed(seed)
    counts = torch.zeros_like(p)
    for _ in range(n):
        x = _draw(q)
        n_acc, fix, _ = verify_draft([x], [q], [p])
        counts[x if n_acc == 1 else fix] += 1
    return counts / n


def test_output_matches_target_for_a_bad_draft():
    p = torch.tensor([0.7, 0.2, 0.1])
    q = torch.tensor([0.1, 0.1, 0.8])  # draft is badly wrong
    emp = _simulate(p, q)
    assert (emp - p).abs().max() < 0.01, emp


def test_output_matches_target_for_a_good_draft_and_accepts_often():
    p = torch.tensor([0.5, 0.3, 0.2])
    q = torch.tensor([0.5, 0.3, 0.2])
    torch.manual_seed(1)
    acc = 0
    for _ in range(2000):
        x = _draw(q)
        n_acc, _, _ = verify_draft([x], [q], [p])
        acc += n_acc
    assert acc == 2000  # identical distributions: never rejected
    assert (_simulate(p, q, n=20000) - p).abs().max() < 0.01


def test_multi_position_prefix_acceptance():
    p1, q1 = torch.tensor([0.9, 0.1]), torch.tensor([0.9, 0.1])
    p2, q2 = torch.tensor([0.0, 1.0]), torch.tensor([1.0, 0.0])  # second position always rejected
    n_acc, fix, fix_p = verify_draft([0, 0], [q1, q2], [p1, p2])
    assert n_acc == 1 and fix == 1 and fix_p == 1.0


def test_dist_transforms():
    logits = torch.tensor([2.0, 1.0, 0.0, -5.0])
    greedy = _dist(logits, 0.0, 1.0)
    assert greedy.tolist() == [1.0, 0.0, 0.0, 0.0]
    nucleus = _dist(logits, 1.0, 0.6)
    assert nucleus[3] == 0 and abs(float(nucleus.sum()) - 1.0) < 1e-6
