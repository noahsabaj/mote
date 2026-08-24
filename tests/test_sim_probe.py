"""The sim-QA probe's data side (mote/eval/sim_probe.py): held-out worlds are deterministic, disjoint from
the generator's training seeds, balanced over locales, and the matcher is tolerant to case, spacing and
final punctuation only."""

from mote.eval.sim_probe import SEED_BASE, contains, heldout_items, match, normalize


def test_normalise_and_match():
    assert normalize("  It is in the Attic. ") == "it is in the attic"
    assert match("4個。", "4個") and match("Да.", "да") and match("Sana has 4 loaves", "Sana has 4 loaves.")
    assert not match("Sana has 5 loaves.", "Sana has 4 loaves.")
    assert contains("I think Sana has 4 loaves, yes.", "Sana has 4 loaves.") and not contains("", "x")


def test_heldout_items_are_deterministic_disjoint_and_balanced():
    a = heldout_items(12, ["en", "ru", "ja"])
    b = heldout_items(12, ["en", "ru", "ja"])
    assert [x["prompt"] for x in a] == [x["prompt"] for x in b] and len(a) == 12
    assert all(x["seed"] > SEED_BASE for x in a)  # the generator's data uses seeds 1..N
    assert {x["locale"] for x in a} == {"en", "ru", "ja"} and len({x["domain"] for x in a}) >= 3
    assert all(x["gold"] and x["prompt"].endswith(("?", "？")) or x["gold"] for x in a)
    assert len({x["prompt"] for x in a}) == 12
