import torch
import pytest

from sisa_mamba.models.sisa_lm import SISALanguageModel, SISALMConfig
from sisa_mamba.models.mamba3_lm import Mamba3LanguageModel, Mamba3LMConfig
from sisa_mamba.models.transformer_lm import TransformerLM, TransformerConfig
from sisa_mamba.models.hybrid_lm import HybridLanguageModel, HybridConfig


def test_sisa_kv_cache_equivalence():
    """Verifies that step-by-step SISA decoding produces identical logits to full sequence forward."""
    torch.manual_seed(42)
    cfg = SISALMConfig(vocab_size=64, d_model=64, n_layers=2, n_heads=4, d_s=16, max_seq_len=32)
    model = SISALanguageModel(cfg).eval()

    input_ids = torch.randint(0, 64, (1, 8))

    # Parallel pass
    with torch.no_grad():
        full_out = model(input_ids)
        full_logits = full_out["logits"]

    # Step-by-step cached decode
    with torch.no_grad():
        # Prefill first 4 tokens
        prefix_out = model(input_ids[:, :4], use_cache=True)
        cache = prefix_out["past_key_values"]

        step_logits = []
        for t in range(4, 8):
            step_out = model(input_ids[:, t:t+1], past_key_values=cache, use_cache=True)
            step_logits.append(step_out["logits"])
            cache = step_out["past_key_values"]

    step_logits = torch.cat(step_logits, dim=1)
    diff = torch.max(torch.abs(full_logits[:, 4:] - step_logits)).item()
    assert diff < 1e-4, f"SISA KV Cache mismatch: max diff = {diff}"


def test_mamba3_recurrent_state_equivalence():
    """Verifies that step-by-step Mamba-3 recurrent decoding matches full parallel forward pass."""
    torch.manual_seed(42)
    cfg = Mamba3LMConfig(vocab_size=64, d_model=64, n_layers=2, d_state=16, n_heads=4, max_seq_len=32)
    model = Mamba3LanguageModel(cfg).eval()

    input_ids = torch.randint(0, 64, (1, 8))

    with torch.no_grad():
        full_out = model(input_ids)
        full_logits = full_out["logits"]

        # Prefill first 4 tokens
        prefix_out = model(input_ids[:, :4], use_state=True)
        states = prefix_out["recurrent_states"]

        step_logits = []
        for t in range(4, 8):
            step_out = model(input_ids[:, t:t+1], recurrent_states=states, use_state=True)
            step_logits.append(step_out["logits"])
            states = step_out["recurrent_states"]

    step_logits = torch.cat(step_logits, dim=1)
    diff = torch.max(torch.abs(full_logits[:, 4:] - step_logits)).item()
    assert diff < 1e-4, f"Mamba-3 Recurrent state mismatch: max diff = {diff}"
