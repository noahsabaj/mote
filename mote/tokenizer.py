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
VOCAB_SIZE = 262  # 256 bytes + 6 specials

SPECIAL_NAMES = {
    BOS_ID: "<|bos|>",
    EOS_ID: "<|eos|>",
    PAD_ID: "<|pad|>",
    SYSTEM_ID: "<|system|>",
    USER_ID: "<|user|>",
    ASSISTANT_ID: "<|assistant|>",
}
ROLE_IDS = {"system": SYSTEM_ID, "user": USER_ID, "assistant": ASSISTANT_ID}


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


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
    def format_chat(self, messages: Sequence[ChatMessage], add_generation_prompt: bool = True) -> List[int]:
        """[bos] (<role> bytes <eos>)* [<assistant>]

        Each turn is ROLE_ID + UTF-8 bytes + EOS. A generation prompt ends with ASSISTANT_ID so
        the model's first emitted byte is the start of its reply; the reply terminates with EOS.
        """
        ids: List[int] = [BOS_ID]
        for m in messages:
            role = ROLE_IDS[m.role]
            ids.append(role)
            ids.extend(m.content.encode("utf-8"))
            ids.append(EOS_ID)
        if add_generation_prompt:
            ids.append(ASSISTANT_ID)
        return ids

    def format_chat_with_loss_mask(self, messages: Sequence[ChatMessage]) -> tuple[List[int], List[int]]:
        """Like format_chat (no generation prompt) but also returns a per-position mask that is 1
        on assistant content bytes and their closing EOS — the only positions SFT trains on."""
        ids: List[int] = [BOS_ID]
        mask: List[int] = [0]
        for m in messages:
            role = ROLE_IDS[m.role]
            ids.append(role)
            mask.append(0)
            content = list(m.content.encode("utf-8")) + [EOS_ID]
            ids.extend(content)
            mask.extend([1 if m.role == "assistant" else 0] * len(content))
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
