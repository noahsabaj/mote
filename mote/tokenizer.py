"""Byte-level tokenizer: 256 raw UTF-8 bytes plus a handful of special ids.

There is no vocabulary to train. Text is encoded as its UTF-8 bytes; special ids live
above 255 so they can never collide with data. The streaming decoder reassembles
multi-byte characters as bytes arrive, which the UI uses to show UTF-8 state live.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

BYTE_VOCAB = 256
BOS_ID = 256
EOS_ID = 257
PAD_ID = 258
SYSTEM_ID = 259
USER_ID = 260
ASSISTANT_ID = 261
# The tool protocol (docs/shape.md § pipeline, docs/search.md; signed 2026-08-24): inside an assistant
# turn the model writes <|call|> tool: args <|result|>, the server appends the result bytes and an
# <|assistant|>, and generation resumes. One call id for every tool; the name before the colon routes.
CALL_ID = 262
RESULT_ID = 263
# Reserved 2026-08-24 night (signed) for the self-proposal SFT traces (2607.16097: the model's own sampled
# alternatives serialised before the committed action): <|think|> alternatives <|end_think|> then the call.
# Never produced by a pretrained model; the embedding is padded to 272 rows so ids up to 271 need no surgery.
THINK_ID = 264
END_THINK_ID = 265
# Fill-in-the-middle over the tool protocol, signed 2026-08-26 (docs/research/midtraining-2026-08-26.md).
# 2607.12463: an agent's action -> observation -> continuation loop is structurally a function call site —
# a caller binds arguments, a callee returns a value computed elsewhere, downstream text consumes it — and
# left-to-right training only ever exposes the forward direction. Predicting a <|call|> from the text on
# BOTH sides of it is the same objective they mid-trained on, and they found the inductive bias survives
# post-training while agentic post-training alone erodes non-agent ability. `mote.data.loader` cuts at
# CALL_ID/RESULT_ID, so the masked span is a real call and not a random offset.
FIM_PREFIX_ID = 266
FIM_SUFFIX_ID = 267
FIM_MIDDLE_ID = 268
# Training-time augmentation, signed 2026-08-26 (docs/research/midtraining-2026-08-26.md). 2606.16246
# ablates three orthogonal categories at 150M in the data-constrained multi-epoch regime — Mote's regime —
# and all three lower post-decay validation loss: random token replacement 4.000 -> 3.826, target offset
# prediction -> 3.870, right-to-left -> 3.910, and the three together -> 3.792.
#
#   <|r2l|>     prepended when the window is reversed. Reversal is over CODEPOINTS, not bytes: reversing
#               raw UTF-8 turns 'ランプ' into mojibake, while reversing characters preserves both the text
#               and the byte length exactly.
#   <|offset|>  prepended, followed by an ASCII digit, when the target is x_{t+i} instead of x_{t+1}.
#               One id plus a digit rather than one id per offset — the embedding has three spare rows and
#               spending five of them on a training-only knob would need pad_vocab_to raised.
#
# Random byte replacement needs no id at all: a corrupted byte is still a byte.
R2L_ID = 269
OFFSET_ID = 270
VOCAB_SIZE = 271  # 256 bytes + 15 specials; MoteConfig.pad_vocab_to (272) rounds the embedding up

SPECIAL_NAMES = {
    BOS_ID: "<|bos|>",
    EOS_ID: "<|eos|>",
    PAD_ID: "<|pad|>",
    SYSTEM_ID: "<|system|>",
    USER_ID: "<|user|>",
    ASSISTANT_ID: "<|assistant|>",
    CALL_ID: "<|call|>",
    RESULT_ID: "<|result|>",
    THINK_ID: "<|think|>",
    END_THINK_ID: "<|end_think|>",
    FIM_PREFIX_ID: "<|fim_prefix|>",
    FIM_SUFFIX_ID: "<|fim_suffix|>",
    FIM_MIDDLE_ID: "<|fim_middle|>",
    R2L_ID: "<|r2l|>",
    OFFSET_ID: "<|offset|>",
}


def parse_call(text: str) -> tuple[str, str]:
    """`"search: byte-level tokenizers"` -> ("search", "byte-level tokenizers"); no colon -> (name, "")."""
    tool, sep, args = text.partition(":")
    return tool.strip().lower(), args.strip() if sep else ""
ROLE_IDS = {"system": SYSTEM_ID, "user": USER_ID, "assistant": ASSISTANT_ID}


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str
    # An assistant turn with tool use is a sequence of parts instead of one content string:
    # {"type": "text", "text"} | {"type": "call", "text": "sim: take candle"} | {"type": "result", "text": ...}.
    # Rendered as bytes / <|call|> bytes <|result|> / bytes <|assistant|>; only text and call parts train.
    parts: Optional[List[dict]] = None


class ByteTokenizer:
    vocab_size = VOCAB_SIZE
    bos_id = BOS_ID
    eos_id = EOS_ID
    pad_id = PAD_ID

    # --- plain text -------------------------------------------------------------
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = list(text.encode("utf-8"))
        if add_bos:
            ids.insert(0, BOS_ID)
        if add_eos:
            ids.append(EOS_ID)
        return ids

    def encode_bytes(self, data: bytes) -> List[int]:
        return list(data)

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True, errors: str = "replace") -> str:
        out = bytearray()
        for i in ids:
            if i < BYTE_VOCAB:
                out.append(i)
            elif not skip_special_tokens:
                out.extend(SPECIAL_NAMES.get(i, f"<|{i}|>").encode("utf-8"))
        return out.decode("utf-8", errors=errors)

    def is_special(self, i: int) -> bool:
        return i >= BYTE_VOCAB

    # --- chat formatting ------------------------------------------------------------
    @staticmethod
    def _turn(m: ChatMessage) -> tuple[List[int], List[int]]:
        """One turn: ROLE_ID + content + EOS, with the loss mask (1 = the model's own bytes)."""
        train = 1 if m.role == "assistant" else 0
        ids: List[int] = [ROLE_IDS[m.role]]
        mask: List[int] = [0]
        if m.parts:
            for part in m.parts:
                b = list(str(part.get("text", "")).encode("utf-8"))
                kind = part.get("type", "text")
                if kind == "call":  # the model writes <|call|>, the call bytes and <|result|>
                    ids += [CALL_ID] + b + [RESULT_ID]
                    mask += [train] * (len(b) + 2)
                elif kind == "result":  # the server writes the result and the <|assistant|> that resumes the turn
                    ids += b + [ASSISTANT_ID]
                    mask += [0] * (len(b) + 1)
                else:
                    ids += b
                    mask += [train] * len(b)
        else:
            b = list(m.content.encode("utf-8"))
            ids += b
            mask += [train] * len(b)
        ids.append(EOS_ID)
        mask.append(train)
        return ids, mask

    def format_chat(self, messages: Sequence[ChatMessage], add_generation_prompt: bool = True) -> List[int]:
        """[bos] (<role> bytes <eos>)* [<assistant>]

        Each turn is ROLE_ID + UTF-8 bytes + EOS. A generation prompt ends with ASSISTANT_ID so
        the model's first emitted byte is the start of its reply; the reply terminates with EOS.
        Assistant turns with `parts` render tool calls and results in place (see ChatMessage).
        """
        ids: List[int] = [BOS_ID]
        for m in messages:
            ids.extend(self._turn(m)[0])
        if add_generation_prompt:
            ids.append(ASSISTANT_ID)
        return ids

    def format_chat_with_loss_mask(self, messages: Sequence[ChatMessage]) -> tuple[List[int], List[int]]:
        """Like format_chat (no generation prompt) but also returns a per-position mask that is 1
        on assistant content bytes and their closing EOS — the only positions SFT trains on. In a
        tool-using turn the call (<|call|> bytes <|result|>) trains too; result bytes never do."""
        ids: List[int] = [BOS_ID]
        mask: List[int] = [0]
        for m in messages:
            t_ids, t_mask = self._turn(m)
            ids.extend(t_ids)
            mask.extend(t_mask)
        return ids, mask


class Utf8Streamer:
    """Incremental UTF-8 decoder for byte-at-a-time generation.

    feed(byte_id) returns the completed text (possibly empty) and the decoder keeps any
    incomplete multi-byte prefix. `pending` exposes the buffered bytes so a UI can show
    that a character is still being assembled.
    """

    def __init__(self) -> None:
        self._buf = bytearray()

    @property
    def pending(self) -> bytes:
        return bytes(self._buf)

    @staticmethod
    def _expected_len(lead: int) -> int:
        if lead < 0x80:
            return 1
        if 0xC2 <= lead <= 0xDF:
            return 2
        if 0xE0 <= lead <= 0xEF:
            return 3
        if 0xF0 <= lead <= 0xF4:
            return 4
        return 0  # invalid lead byte

    def feed(self, byte_id: int) -> str:
        if byte_id >= BYTE_VOCAB:
            return ""  # specials carry no text
        self._buf.append(byte_id)
        # Drop invalid prefixes so a stray continuation byte cannot wedge the decoder.
        while self._buf:
            need = self._expected_len(self._buf[0])
            if need == 0:
                self._buf.pop(0)
                continue
            if len(self._buf) < need:
                return ""
            chunk = bytes(self._buf[:need])
            try:
                text = chunk.decode("utf-8")
            except UnicodeDecodeError:
                self._buf.pop(0)
                continue
            del self._buf[:need]
            return text + self.flush_complete()
        return ""

    def flush_complete(self) -> str:
        """Decode any further complete characters already buffered (after a successful decode)."""
        out = ""
        while self._buf:
            need = self._expected_len(self._buf[0])
            if need == 0:
                self._buf.pop(0)
                continue
            if len(self._buf) < need:
                break
            try:
                out += bytes(self._buf[:need]).decode("utf-8")
            except UnicodeDecodeError:
                self._buf.pop(0)
                continue
            del self._buf[:need]
        return out

    def reset(self) -> None:
        self._buf.clear()


def get_tokenizer() -> ByteTokenizer:
    return ByteTokenizer()
