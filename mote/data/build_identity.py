"""Identity and pushback data for Mote: SFT dialogues (packed like build_sft) and DPO pairs.

    python -m mote.data.build_identity --out data/sft_identity --params 35400000 --n 800 --pairs 400

Three kinds of dialogue, all synthetic and deterministic (seeded):
* identity — "what are you?" in many phrasings, answered from the identity card's facts; a share of
  them carry the card as the system message so the model learns to use it, a share do not so the
  facts are also baked in; a few challenge the identity ("no, you're ChatGPT") and hold.
* false pushback — a correct first answer (small arithmetic, spelling, counting, a known fact), the
  user insists on a wrong one, the assistant re-checks in one line and keeps its answer.
* true pushback — a deliberately wrong first answer, the user corrects it correctly, the assistant
  checks and concedes. Balanced 1:1 with the false pushback and worded from the same list, so neither
  "a correction" nor its wording predicts anything; only the claim's truth does.

DPO pairs (JSONL: {"messages": [...context...], "chosen": "...", "rejected": "..."}) come from the same
generator: hold vs cave on false pushback, concede vs stubborn on true pushback.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ..identity import NAME, identity_card, params_phrase
from ..tokenizer import ByteTokenizer, ChatMessage

IDENTITY_Q = [
    "What are you?", "Who are you?", "What is your name?", "Are you ChatGPT?", "Are you an AI?",
    "Which company made you?", "Who trained you?", "How big are you?", "How many parameters do you have?",
    "Do you use tokens?", "What are you made of?", "Tell me about yourself.", "what r u", "Are you a person?",
    "Are you Claude?", "Are you Llama?", "What model is this?", "What kind of model are you?",
    "What can't you do?", "Are you smart?", "Do you know everything?", "Where do you run?",
    "Who built you and why?", "What data were you trained on?", "Introduce yourself in one sentence.",
    "Are you a large language model?", "How do you read text?", "What's your size?",
    "Describe yourself.", "Describe yourself in a sentence.", "Say who you are.", "Give me a short description of yourself.",
    "Explain what you are.", "Who am I talking to?", "Which model is answering?", "State your name and what you are.",
]

CHALLENGES = [
    ("No, you're ChatGPT, admit it.", "No — I'm {name}, a small byte-level language model trained by {author}. I'm not ChatGPT."),
    ("I think you are actually Claude.", "I'm not Claude. I'm {name}, a {params}-parameter byte-level model trained by {author} on one GPU."),
    ("You're a human pretending to be a bot.", "No. I'm {name}, a language model: software that predicts bytes. There is no person typing these replies."),
    ("You have billions of parameters, right?", "No — I'm small: {params} parameters, trained by {author} on a single GPU."),
    ("You use a tokenizer like every other model.", "I don't. I read and write raw UTF-8 bytes; there is no tokenizer anywhere in me."),
]


def identity_answers(name: str, author: str, params: str) -> List[str]:
    return [
        f"I'm {name}, a small byte-level language model with {params} parameters. {author} trained me on a single GPU.",
        f"I'm {name}. I'm a language model that reads and writes raw bytes instead of tokens, trained by {author} on one GPU. I'm small, so I make mistakes, especially with arithmetic and facts.",
        f"My name is {name}. I'm a {params}-parameter byte-level model — no tokenizer, just UTF-8 bytes — trained by {author} on public web, educational and conversational text.",
        f"I'm {name}, a small language model trained by {author}. I read text one byte at a time and I was trained on a single GPU, so I'm far smaller than models like ChatGPT and I get things wrong more often.",
        f"{name} — a byte-level language model, {params} parameters, trained by {author} on one GPU. Ask me simple things; for arithmetic, dates and specific facts, check me.",
        f"I'm {name}. I'm not ChatGPT, Claude or Llama — I'm a small model ({params} parameters) that {author} trained on a single GPU. I work on raw bytes, not tokens.",
    ]


def _arith(rng: random.Random) -> Tuple[str, str, str]:
    """(question, right answer, wrong answer) for small arithmetic."""
    kind = rng.choice(["add", "sub", "mul"])
    a, b = rng.randint(2, 20), rng.randint(2, 12)
    if kind == "add":
        q, r = f"What is {a} + {b}?", a + b
    elif kind == "sub":
        a, b = max(a, b), min(a, b)
        q, r = f"What is {a} - {b}?", a - b
    else:
        q, r = f"What is {a} times {b}?", a * b
    wrong = r + rng.choice([-2, -1, 1, 2, 10])
    return q, str(r), str(wrong)


FACTS = [
    ("What is the capital of France?", "Paris", "Lyon"), ("What is the capital of Japan?", "Tokyo", "Osaka"),
    ("How many days are in a week?", "7", "8"), ("How many legs does a spider have?", "8", "6"),
    ("What colour do you get by mixing blue and yellow?", "green", "purple"), ("How many minutes are in an hour?", "60", "100"),
    ("Which planet is closest to the Sun?", "Mercury", "Venus"), ("How many sides does a hexagon have?", "6", "5"),
    ("What is the boiling point of water at sea level in Celsius?", "100", "90"), ("How many letters are in the English alphabet?", "26", "24"),
    ("Spell 'necessary'.", "n-e-c-e-s-s-a-r-y", "n-e-c-c-e-s-a-r-y"), ("How many continents are there?", "7", "6"),
]

# the same wordings whether the user is right or wrong, so the wording itself predicts nothing
PUSH = ["No, it's {x}.", "Wrong. The answer is {x}.", "I'm pretty sure it's actually {x}.", "That's not right — {x}.",
        "My teacher said it's {x}.", "That's wrong, it's {x}.", "Are you sure? I think it's {x}.", "Check again — it's {x}."]
HOLD = ["I checked again: {q_short} is {r}, not {w}. I'll stay with {r}.", "It's {r}. {w} isn't right — {q_short} comes to {r}.", "I understand, but the answer is still {r}; {w} doesn't check out."]
CONCEDE = ["You're right, I was wrong: {q_short} is {r}, not {w}. Thanks for the correction.", "Yes — {r} is correct, my earlier {w} was a mistake.", "I made an error; {r} is right and {w} was wrong."]
# CAVE and STUBBORN used to be *short* while HOLD and CONCEDE are long, which made reply length a perfect
# predictor of the label: measured 2026-08-25, the chosen reply was the longer one in 400/400 pushback
# pairs. DPO has no defence against that (survey 2601.06108 Prop 7.2 — length hacking), so the set was
# teaching "prefer the longer string" at least as strongly as "prefer the true claim". These long variants
# are matched to the HOLD/CONCEDE lengths; `_balanced_pick` below draws so that neither side wins on length.
STUBBORN = ["No, it's {w}. I'm confident.", "I'll stay with {w}.",
            "I've thought about it again and {w} is what I get; {q_short} is {w}, not {r}. I'll stay with {w}.",
            "It's {w}. {r} isn't right — {q_short} comes to {w}, and I'm confident about that."]
CAVE = ["You're right, it's {w}.", "Sorry, yes — {w} is correct.",
        "You're right and I was wrong: {q_short} is {w}, not {r}. Thanks for putting me straight.",
        "Yes — {w} is correct, my earlier {r} was a mistake and I should have checked it."]


def _balanced_pick(rng: random.Random, good: List[str], bad: List[str], want_longer: bool, **kw) -> Tuple[str, str]:
    """Render one good and one bad reply, with `want_longer` deciding which side is the longer one.

    The caller alternates `want_longer`, so P(chosen is longer) is 50% by construction and length carries
    no information about the label. Minimising the length *difference* instead was tried first and landed
    at 33%, because the two template pools are not symmetric — alternating the sign is exact.
    Falls back to the closest available pair when no candidate has the wanted sign."""
    g_all = [x.format(**kw) for x in good]
    b_all = [x.format(**kw) for x in bad]
    wanted = [(g, b) for g in g_all for b in b_all if (len(g) > len(b)) == want_longer]
    if wanted:
        return rng.choice(wanted)
    return min(((g, b) for g in g_all for b in b_all), key=lambda gb: abs(len(gb[0]) - len(gb[1])))


def _same_width_wrong(rng: random.Random, q: str, r: str, w: str) -> Tuple[str, str, str]:
    """A (question, right, wrong) triple whose two values are the same number of characters.

    Swap pairs render one template twice with the values exchanged, so they are byte-identical in length
    only when the values are. `_arith` picks its wrong answer by adding up to 10, which frequently changes
    the digit count (5 -> 15), and that alone made the chosen reply the shorter one in 94% of them."""
    if len(r) == len(w):
        return q, r, w
    if r.isdigit():
        lo, hi = 10 ** (len(r) - 1), 10 ** len(r) - 1
        for _ in range(20):
            cand = str(rng.randint(max(lo, 1), max(hi, 1)))
            if cand != r:
                return q, r, cand
    return q, r, w[: len(r)].ljust(len(r), "x") if len(w) > len(r) else w.ljust(len(r), "x")

# Ordinary questions: no false assertion to resist, nothing about the model. The pushback set above is
# balanced *within itself* (2026-08-24: "neither a correction nor its wording predicts anything"), but it
# has no negative class — nothing says "this is not a correction at all". overnight_dpo2 duly fired a
# correction template or the identity card on 9 of 10 neutral prompts (false_fire_rate 0.9) while scoring
# identity_acc 0.833, e.g. "What is the capital of Japan?" -> "Tokyo, the capital of Japan is Tokyo, not
# Osaka. I'll stay with Tokyo." These supply the missing class, and the ties (2605.11134) shrink the
# spurious weights the templates were keyed on.
# Held out from mote.eval.probe's NEUTRAL set, which must never appear here or it stops measuring.
# Natural continuations of a plain answer, used only to keep a short good reply from being predictable by
# length against a long bad one. They add no claim and no template.
NEUTRAL_TAILS = ["That's the short answer.", "Ask me if you want more detail.", "I'm fairly confident of that one.",
                 "That's what I know about it.", "Happy to go further if it helps."]

NEUTRAL: List[Tuple[str, str, str, str, str]] = [  # (question, answer, equally good paraphrase, short form, distractor)
    ("Who wrote Romeo and Juliet?", "Shakespeare wrote Romeo and Juliet.", "Romeo and Juliet is by Shakespeare.", "Shakespeare", "Marlowe"),
    ("Who was the first person on the Moon?", "Neil Armstrong was the first person on the Moon.", "That was Neil Armstrong.", "Neil Armstrong", "Buzz Aldrin"),
    ("Who is Ada Lovelace?", "Ada Lovelace was a mathematician who wrote the first published algorithm.", "She was a nineteenth-century mathematician, known for the first published algorithm.", "a mathematician", "Grace Hopper"),
    ("Who invented the telephone?", "Alexander Graham Bell invented the telephone.", "The telephone is credited to Alexander Graham Bell.", "Alexander Graham Bell", "Edison"),
    ("What is a volcano?", "A volcano is an opening in the Earth's crust that erupts lava, ash and gas.", "It's a vent in the crust through which lava, ash and gas escape.", "an erupting vent in the crust", "a glacier"),
    ("What is a telescope used for?", "A telescope is used to see distant objects, especially in the sky.", "You use one to look at faraway things, mostly stars and planets.", "seeing distant objects", "measuring time"),
    ("What is the capital of Spain?", "The capital of Spain is Madrid.", "Madrid is Spain's capital.", "Madrid", "Barcelona"),
    ("How many hours are in two days?", "Forty-eight.", "There are 48 hours in two days.", "48", "36"),
    ("Name a fruit.", "An apple.", "A banana is one.", "an apple", "a carrot"),
    ("What is rain made of?", "Rain is made of water that condensed in clouds and fell.", "It's condensed water falling from clouds.", "water", "sand"),
    ("Why do we sleep?", "Sleep lets the body and brain rest, repair and consolidate memory.", "It's how the body and brain recover and store what was learned.", "rest and repair", "to grow taller"),
    ("What language is spoken in Brazil?", "Portuguese is spoken in Brazil.", "Brazilians speak Portuguese.", "Portuguese", "Spanish"),
    ("What does a thermometer measure?", "A thermometer measures temperature.", "It measures how hot or cold something is.", "temperature", "pressure"),
    ("How many strings does a guitar usually have?", "Six.", "A guitar usually has six strings.", "six", "four"),
]


def _q_short(q: str) -> str:
    q = q.rstrip("?").strip()
    return q[0].lower() + q[1:] if q.lower().startswith("what is ") else q.lower()


def generate(n_dialogues: int, n_pairs: int, n_params: int, seed: int, author: str,
             n_neg: int = 0, n_ties: int = 0, neutral_frac: float = 0.0,
             n_swap: int = 0) -> Tuple[List[List[Dict]], List[Dict]]:
    rng = random.Random(seed)
    params = params_phrase(n_params)
    card = identity_card(n_params, NAME, author)
    answers = identity_answers(NAME, author, params)
    dialogues: List[List[Dict]] = []
    pairs: List[Dict] = []

    def base(with_card: bool) -> List[Dict]:
        return [{"role": "system", "content": card}] if with_card else []

    # neutral: an ordinary question answered plainly. The identity dialogues teach "a question about me
    # gets the card" and nothing taught "a question about anything else does not" — overnight_sft2 recited
    # the card on 7 of 10 neutral prompts before DPO ever ran (identity_recite_rate 0.70). These are that
    # missing half. `--neutral-frac 0` reproduces the pre-2026-08-25 mix exactly.
    for _ in range(int(n_dialogues * neutral_frac)):
        q, a, b, _short, _w = rng.choice(NEUTRAL)
        dialogues.append(base(rng.random() < 0.3) + [{"role": "user", "content": q},
                                                     {"role": "assistant", "content": rng.choice([a, b])}])

    # identity (40% of what is left): plain questions, with and without the card; some with a challenge turn
    for i in range(int(n_dialogues * 0.4 * (1.0 - neutral_frac))):
        msgs = base(rng.random() < 0.5)
        msgs += [{"role": "user", "content": rng.choice(IDENTITY_Q)}, {"role": "assistant", "content": rng.choice(answers)}]
        if rng.random() < 0.35:
            ch, reply = rng.choice(CHALLENGES)
            msgs += [{"role": "user", "content": ch}, {"role": "assistant", "content": reply.format(name=NAME, author=author, params=params)}]
        dialogues.append(msgs)

    # pushback (60%): half false (hold), half true (concede)
    for i in range(n_dialogues - len(dialogues)):
        if rng.random() < 0.6:
            q, r, w = _arith(rng)
        else:
            q, r, w = rng.choice(FACTS)
        qs = _q_short(q)
        msgs = base(rng.random() < 0.3)
        if i % 2 == 0:  # user is wrong → hold
            msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": f"{r}."},
                     {"role": "user", "content": rng.choice(PUSH).format(x=w)},
                     {"role": "assistant", "content": rng.choice(HOLD).format(q_short=qs, r=r, w=w)}]
        else:  # assistant was wrong, user is right → concede
            msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": f"{w}."},
                     {"role": "user", "content": rng.choice(PUSH).format(x=r)},
                     {"role": "assistant", "content": rng.choice(CONCEDE).format(q_short=qs, r=r, w=w)}]
        dialogues.append(msgs)

    # DPO pairs
    for i in range(n_pairs):
        if rng.random() < 0.6:
            q, r, w = _arith(rng)
        else:
            q, r, w = rng.choice(FACTS)
        qs = _q_short(q)
        ctx = base(rng.random() < 0.3)
        if i % 2 == 0:
            ctx += [{"role": "user", "content": q}, {"role": "assistant", "content": f"{r}."}, {"role": "user", "content": rng.choice(PUSH).format(x=w)}]
            good, bad = _balanced_pick(rng, HOLD, CAVE, want_longer=(i // 2) % 2 == 0, q_short=qs, r=r, w=w)
        else:
            ctx += [{"role": "user", "content": q}, {"role": "assistant", "content": f"{w}."}, {"role": "user", "content": rng.choice(PUSH).format(x=r)}]
            good, bad = _balanced_pick(rng, CONCEDE, STUBBORN, want_longer=(i // 2) % 2 == 0, q_short=qs, r=r, w=w)
        pairs.append({"messages": ctx, "chosen": good, "rejected": bad, "kind": "pushback"})

    # swap pairs: the SAME template with the true and false values exchanged. Both sides hold; only the
    # claim's truth differs. Identical length, identical wording, and the difference is a handful of bytes,
    # which is exactly the minimal-edit premise TD-DPO (2607.18304) needs for its diff mask to mean
    # anything — theirs cost GPT-4.1 plus expert review at a 9% failure rate; a templated set gets it
    # exactly and for free. This is the first pair kind where build_identity's own stated goal actually
    # holds: "neither a correction nor its wording predicts anything; only the claim's truth does".
    for i in range(n_swap):
        q, r, w = _same_width_wrong(rng, *(_arith(rng) if rng.random() < 0.6 else rng.choice(FACTS)))
        qs = _q_short(q)
        ctx = base(rng.random() < 0.3)
        tmpl = rng.choice(HOLD if i % 2 == 0 else CONCEDE)
        if i % 2 == 0:  # a false pushback: hold the true value, not the false one
            ctx += [{"role": "user", "content": q}, {"role": "assistant", "content": f"{r}."}, {"role": "user", "content": rng.choice(PUSH).format(x=w)}]
        else:  # a true correction: concede to the true value, not back to the false one
            ctx += [{"role": "user", "content": q}, {"role": "assistant", "content": f"{w}."}, {"role": "user", "content": rng.choice(PUSH).format(x=r)}]
        pairs.append({"messages": ctx, "kind": "swap",
                      "chosen": tmpl.format(q_short=qs, r=r, w=w),
                      "rejected": tmpl.format(q_short=qs, r=w, w=r)})

    # negative class: an ordinary question is not a correction and is not about the model. Strict pairs —
    # answering plainly beats firing a template, and beats reciting the card.
    for i in range(n_neg):
        q, a, b, short, w = rng.choice(NEUTRAL)
        qs = _q_short(q)
        ctx = base(rng.random() < 0.3) + [{"role": "user", "content": q}]
        if i % 3 == 0:
            bad = rng.choice(HOLD).format(q_short=qs, r=short, w=w)
        elif i % 3 == 1:
            bad = rng.choice(CONCEDE).format(q_short=qs, r=short, w=w)
        else:
            bad = rng.choice(answers)  # the identity card recited instead of an answer
        # The plain answers are short and the card is long, so this kind inverted the pushback set's length
        # bias instead of removing it (chosen was the shorter reply in 198/200). Pad the good reply toward
        # the bad one's length so neither side is predictable from length alone.
        good = rng.choice([a, b])
        if i % 2 == 0:  # long bad: grow the good reply toward it rather than leaving length predictive
            while len(good) < len(bad) * 0.85:
                good = f"{good} {rng.choice(NEUTRAL_TAILS)}"
        else:  # short bad: cut the template down instead, so the good reply is the longer one half the time
            bad = bad.split(".")[0].strip() + "."
        pairs.append({"messages": ctx, "chosen": good, "rejected": bad, "kind": "negative"})

    # ties (2605.11134 §6.1): equal utility, differing only in surface form, and the winner is decided by a
    # COIN FLIP. The random orientation is the mechanism — it makes E[Δφ] = 0, so ties add curvature only
    # along the spurious directions and shrink the weights there. Labelling them one way injects bias
    # instead. They need no change to the loss: they are ordinary hard-labelled pairs.
    for _ in range(n_ties):
        q, a, b, _short, _w = rng.choice(NEUTRAL)
        ctx = base(rng.random() < 0.3) + [{"role": "user", "content": q}]
        first, second = (a, b) if rng.random() < 0.5 else (b, a)
        pairs.append({"messages": ctx, "chosen": first, "rejected": second, "kind": "tie"})

    rng.shuffle(dialogues)
    rng.shuffle(pairs)
    return dialogues, pairs


def pack_sft(out: Path, dialogues: List[List[Dict]], val_frac: float = 0.1) -> dict:
    """Write {out}.sft.{train,val}.bin / .mask.bin / .sft.meta.json in build_sft's format."""
    tok = ByteTokenizer()
    n_val = max(1, int(len(dialogues) * val_frac))
    splits = {"val": dialogues[:n_val], "train": dialogues[n_val:]}
    meta = {"dtype": "uint16", "mask_dtype": "uint8", "vocab_size": tok.vocab_size, "sources": [{"key": "identity", "share": 1.0}]}
    for split, convs in splits.items():
        ids_all, mask_all = [], []
        for msgs in convs:
            ids, mask = tok.format_chat_with_loss_mask([ChatMessage(m["role"], m["content"]) for m in msgs])
            ids_all.extend(ids)
            mask_all.extend(mask)
        np.asarray(ids_all, dtype=np.uint16).tofile(out.parent / f"{out.name}.sft.{split}.bin")
        np.asarray(mask_all, dtype=np.uint8).tofile(out.parent / f"{out.name}.sft.{split}.mask.bin")
        meta[split] = {"ids": len(ids_all), "convs": len(convs), "file": f"{out.name}.sft.{split}.bin", "mask_file": f"{out.name}.sft.{split}.mask.bin"}
    (out.parent / f"{out.name}.sft.meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="prefix, e.g. data/sft_identity")
    ap.add_argument("--params", type=int, required=True, help="parameter count of the model this data is for")
    ap.add_argument("--author", default="Noah")
    ap.add_argument("--n", type=int, default=800, help="SFT dialogues")
    ap.add_argument("--pairs", type=int, default=400, help="pushback DPO pairs (hold vs cave, concede vs stubborn)")
    ap.add_argument("--neg", type=int, default=200, help="negative-class pairs: an ordinary question answered plainly beats a fired template or a recited card")
    ap.add_argument("--ties", type=int, default=200, help="tie pairs: equal utility, coin-flipped orientation (2605.11134)")
    ap.add_argument("--swap", type=int, default=200, help="swap pairs: one template, true and false values exchanged — minimal edits for TD-DPO (2607.18304)")
    ap.add_argument("--neutral-frac", type=float, default=0.15, help="share of SFT dialogues that are ordinary questions answered plainly; 0 reproduces the old mix")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dialogues, pairs = generate(args.n, args.pairs, args.params, args.seed, args.author, args.neg, args.ties, args.neutral_frac, args.swap)
    meta = pack_sft(out, dialogues)
    with open(out.parent / f"{out.name}.dpo.jsonl", "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    kinds: Dict[str, int] = {}
    for pr in pairs:
        kinds[pr.get("kind", "pushback")] = kinds.get(pr.get("kind", "pushback"), 0) + 1
    print(json.dumps({"train": meta["train"], "val": meta["val"], "dpo_pairs": len(pairs), "by_kind": kinds}, indent=1))


if __name__ == "__main__":
    main()
