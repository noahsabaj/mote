"""The identity/pushback generator must not hand DPO a shortcut.

Measured 2026-08-25: reply length predicted the label in 400/400 pushback pairs, because HOLD and CONCEDE
are long templates and CAVE and STUBBORN were short ones. DPO has no defence against that (2601.06108
Prop 7.2), so the set was teaching "prefer the longer string" at least as hard as "prefer the true claim" —
a second spurious feature alongside the question frame, and a stronger one. These tests are what stops it
coming back the next time a template is edited.
"""

import json
from collections import defaultdict


from mote.data.build_identity import NEUTRAL, generate
from mote.eval.probe import NEUTRAL as PROBE_NEUTRAL


def _pairs(**kw):
    kw = {"n_dialogues": 400, "n_pairs": 200, "n_params": 31643528, "seed": 0, "author": "Noah",
          "n_neg": 100, "n_ties": 100, "n_swap": 100, **kw}
    _, pairs = generate(**kw)
    return pairs


def _by_kind(pairs):
    d = defaultdict(list)
    for p in pairs:
        d[p.get("kind", "pushback")].append(p)
    return d


def test_length_does_not_predict_the_label():
    """Among pairs whose two sides differ in length at all, either side should be the longer one about
    half the time. Equal-length pairs (every swap pair) carry no length signal and are excluded."""
    for kind, ps in _by_kind(_pairs()).items():
        uneven = [p for p in ps if len(p["chosen"]) != len(p["rejected"])]
        if len(uneven) < 20:
            continue  # a kind that is essentially all equal-length has nothing to skew
        longer = sum(len(p["chosen"]) > len(p["rejected"]) for p in uneven)
        share = longer / len(uneven)
        assert 0.35 <= share <= 0.65, f"{kind}: chosen is the longer reply {share:.0%} of the time (was 100% before 2026-08-25)"


def test_swap_pairs_are_exact_minimal_edits():
    """One template rendered twice with the true and false values exchanged: same wording, same length,
    a few differing bytes. That is what TD-DPO's (2607.18304) diff mask needs, and it means the ONLY
    thing separating the two sides is which claim is true — the generator's stated goal since 2026-08-24."""
    swap = _by_kind(_pairs())["swap"]
    assert len(swap) == 100
    for p in swap:
        a, b = p["chosen"], p["rejected"]
        assert a != b
        assert len(a) == len(b), f"swap pair differs in length: {a!r} vs {b!r}"
        differing = sum(1 for x, y in zip(a, b) if x != y)
        assert differing <= 0.2 * len(a), f"swap pair differs in {differing}/{len(a)} bytes: not a minimal edit"


def test_negative_class_covers_both_failure_modes():
    """An ordinary question must draw neither a pushback template nor the identity card."""
    neg = _by_kind(_pairs())["negative"]
    assert len(neg) == 100
    recited = sum(1 for p in neg if "Mote" in p["rejected"])
    templated = len(neg) - recited
    assert recited >= 20 and templated >= 20, f"lopsided negative class: {recited} card, {templated} template"
    assert not any("Mote" in p["chosen"] for p in neg), "the good reply to an ordinary question never names the model"


def test_tie_orientation_is_a_coin_flip():
    """2605.11134 §6.1: ties only shrink the spurious weights if the winner is assigned at random. A
    single-direction label injects bias instead, so this is the mechanism, not a detail."""
    ties = _by_kind(_pairs(n_ties=400))["tie"]
    first = sum(1 for p in ties if p["chosen"] < p["rejected"])
    assert 0.35 <= first / len(ties) <= 0.65, f"tie orientation is not balanced: {first}/{len(ties)}"


def test_probe_neutral_prompts_are_held_out():
    """The probe's negative class only measures generalisation if the generator never trains on it."""
    assert not (set(PROBE_NEUTRAL) & {e[0] for e in NEUTRAL})
    assert len(set(PROBE_NEUTRAL)) == len(PROBE_NEUTRAL)


def test_neutral_frac_zero_reproduces_the_old_sft_mix():
    """`--neutral-frac 0` must be the pre-2026-08-25 mix exactly, so the two Round A SFT arms differ in
    one thing only."""
    d0, _ = generate(800, 0, 31643528, 0, "Noah", neutral_frac=0.0)
    d15, _ = generate(800, 0, 31643528, 0, "Noah", neutral_frac=0.15)
    named = lambda ds: sum(1 for c in ds if any("Mote" in m["content"] for m in c if m["role"] == "assistant"))
    assert named(d0) == 320  # 40% of 800, untouched
    assert named(d15) < named(d0)
    assert len(d0) == len(d15) == 800


def test_generator_is_deterministic():
    a = json.dumps(_pairs(seed=7), sort_keys=True)
    b = json.dumps(_pairs(seed=7), sort_keys=True)
    assert a == b
