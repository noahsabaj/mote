import codecs
from typing import List, Union, Optional


class ByteTokenizer:
    """
    Pure Byte-Level Tokenizer ("The Language of Machines").
    Maps arbitrary strings or binary data directly to raw bytes (0..255)
    with dedicated special tokens for sequence demarcation.
    """

    def __init__(
        self,
        bos_token: str = "<BOS>",
        eos_token: str = "<EOS>",
        pad_token: str = "<PAD>",
        sep_token: str = "<SEP>",
        unk_token: str = "<UNK>",
    ):
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.sep_token = sep_token
        self.unk_token = unk_token

        # Base byte vocabulary: 0..255
        self.num_base_bytes = 256

        # Special token IDs
        self.bos_id = 256
        self.eos_id = 257
        self.pad_id = 258
        self.sep_id = 259
        self.unk_id = 260

        self.special_tokens = {
            self.bos_id: self.bos_token,
            self.eos_id: self.eos_token,
            self.pad_id: self.pad_token,
            self.sep_id: self.sep_token,
            self.unk_id: self.unk_token,
        }
        self.special_tokens_inv = {v: k for k, v in self.special_tokens.items()}

        self.vocab_size = 261

    def encode(
        self,
        text: Union[str, bytes],
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """Encodes text or raw bytes into a list of token IDs."""
        if isinstance(text, str):
            raw_bytes = text.encode("utf-8")
        elif isinstance(text, bytes):
            raw_bytes = text
        else:
            raise TypeError(f"Expected str or bytes, got {type(text)}")

        tokens = list(raw_bytes)

        if add_bos:
            tokens = [self.bos_id] + tokens
        if add_eos:
            tokens = tokens + [self.eos_id]

        return tokens

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
        errors: str = "replace",
    ) -> str:
        """Decodes token IDs back into a UTF-8 string."""
        byte_list = []
        for tid in token_ids:
            if tid < 256:
                byte_list.append(tid)
            elif not skip_special_tokens and tid in self.special_tokens:
                # Represent special tokens as text markers
                special_bytes = self.special_tokens[tid].encode("utf-8")
                byte_list.extend(special_bytes)

        return bytes(byte_list).decode("utf-8", errors=errors)

    def decode_single_byte(self, token_id: int) -> Optional[bytes]:
        """Returns the raw byte representation for a token ID, or None if special."""
        if 0 <= token_id < 256:
            return bytes([token_id])
        return None

    def get_incremental_decoder(self):
        """Returns a stateful UTF-8 streaming decoder that buffers partial multi-byte characters."""
        return UTF8IncrementalStreamer(self)


class UTF8IncrementalStreamer:
    """Stateful streamer for real-time token-by-token UTF-8 decoding."""

    def __init__(self, tokenizer: ByteTokenizer):
        self.tokenizer = tokenizer
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.byte_buffer = bytearray()

    def step(self, token_id: int) -> str:
        """Processes one token ID and returns newly completed string text (if any)."""
        if token_id < 256:
            self.byte_buffer.append(token_id)
            try:
                # Attempt decoding buffered bytes with final=False
                decoded_str = self.decoder.decode(bytes(self.byte_buffer), final=False)
                self.byte_buffer.clear()
                return decoded_str
            except Exception:
                return ""
        elif token_id == self.tokenizer.eos_id:
            # Flush on EOS
            final_str = self.decoder.decode(bytes(self.byte_buffer), final=True)
            self.byte_buffer.clear()
            return final_str
        return ""

    def flush(self) -> str:
        """Flushes any remaining buffered bytes."""
        final_str = self.decoder.decode(bytes(self.byte_buffer), final=True)
        self.byte_buffer.clear()
        return final_str


class HFTokenizerWrapper:
    """Wrapper for Hugging Face Tokenizers (e.g. GPT-2, Llama)."""

    def __init__(self, hf_tokenizer_name: str = "gpt2"):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.vocab_size = len(self.tokenizer)
        self.bos_id = self.tokenizer.bos_token_id or self.tokenizer.cls_token_id or 0
        self.eos_id = self.tokenizer.eos_token_id or 0
        self.pad_id = self.tokenizer.pad_token_id or 0

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if add_bos and self.bos_id is not None:
            ids = [self.bos_id] + ids
        if add_eos and self.eos_id is not None:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


def get_tokenizer(tokenizer_type: str = "byte", hf_name: Optional[str] = None):
    """Factory helper to obtain a byte-level or HF tokenizer."""
    if tokenizer_type == "byte":
        return ByteTokenizer()
    elif tokenizer_type == "hf":
        return HFTokenizerWrapper(hf_name or "gpt2")
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
