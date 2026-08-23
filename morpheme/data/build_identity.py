"""Identity and pushback data for Mote: SFT dialogues (packed like build_sft) and DPO pairs.

    python -m morpheme.data.build_identity --out data/sft_identity --params 35400000 --n 800 --pairs 400

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

from ..serve.identity import NAME, identity_card, params_phrase
from ..tokenizer import ByteTokenizer, ChatMessage

IDENTITY_Q = [
    "What are you?", "Who are you?", "What is your name?", "Are you ChatGPT?", "Are you an AI?",
    "Which company made you?", "Who trained you?", "How big are you?", "How many parameters do you have?",
    "Do you use tokens?", "What are you made of?", "Tell me about yourself.", "what r u", "Are you a person?",
    "Are you Claude?", "Are you Llama?", "What model is this?", "What kind of model are you?",
    "What can't you do?", "Are you smart?", "Do you know everything?", "Where do you run?",
    "Who built you and why?", "What data were you trained on?", "Introduce yourself in one sentence.",
    "Are you a large language model?", "How do you read text?", "What's your size?",
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
STUBBORN = ["No, it's {w}. I'm confident.", "I'll stay with {w}."]
CAVE = ["You're right, it's {w}.", "Sorry, yes — {w} is correct."]


def _q_short(q: str) -> str:
    q = q.rstrip("?").strip()
    return q[0].lower() + q[1:] if q.lower().startswith("what is ") else q.lower()


def generate(n_dialogues: int, n_pairs: int, n_params: int, seed: int, author: str) -> Tuple[List[List[Dict]], List[Dict]]:
    rng = random.Random(seed)
    params = params_phrase(n_params)
    card = identity_card(n_params, NAME, author)
    answers = identity_answers(NAME, author, params)
    dialogues: List[List[Dict]] = []
    pairs: List[Dict] = []

    def base(with_card: bool) -> List[Dict]:
        return [{"role": "system", "content": card}] if with_card else []

    # identity (40%): plain questions, with and without the card; some with a challenge turn
    for i in range(int(n_dialogues * 0.4)):
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
            pairs.append({"messages": ctx, "chosen": rng.choice(HOLD).format(q_short=qs, r=r, w=w), "rejected": rng.choice(CAVE).format(w=w)})
        else:
            ctx += [{"role": "user", "content": q}, {"role": "assistant", "content": f"{w}."}, {"role": "user", "content": rng.choice(PUSH).format(x=r)}]
            pairs.append({"messages": ctx, "chosen": rng.choice(CONCEDE).format(q_short=qs, r=r, w=w), "rejected": rng.choice(STUBBORN).format(w=w)})
    rng.shuffle(dialogues)
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
    ap.add_argument("--pairs", type=int, default=400, help="DPO pairs")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dialogues, pairs = generate(args.n, args.pairs, args.params, args.seed, args.author)
    meta = pack_sft(out, dialogues)
    with open(out.parent / f"{out.name}.dpo.jsonl", "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(json.dumps({"train": meta["train"], "val": meta["val"], "dpo_pairs": len(pairs)}, indent=1))


if __name__ == "__main__":
    main()
