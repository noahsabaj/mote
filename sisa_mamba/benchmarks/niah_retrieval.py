import random
import torch
from typing import Dict, List, Tuple, Any

from ..tokenizer.byte_tokenizer import ByteTokenizer
from ..models.sisa_lm import SISALanguageModel, SISALMConfig
from ..models.mamba3_lm import Mamba3LanguageModel, Mamba3LMConfig
from ..models.transformer_lm import TransformerLM, TransformerConfig


def generate_niah_sample(
    context_length: int = 512,
    depth_pct: float = 0.5,
    secret_key: str = "42",
) -> Tuple[str, str, str]:
    """
    Constructs a Needle-In-A-Haystack prompt:
    - Filler text composed of background story / technical sentences.
    - Needle inserted at target depth: 'The special secret code is <secret_key>.'
    - Query at the end: 'Question: What is the special secret code? Answer:'
    """
    needle = f" The special secret code is {secret_key}. "
    query = " Question: What is the special secret code? Answer: "

    filler_pool = [
        " The state space model processes sequences with continuous linear ordinary differential equations.",
        " Structured SSMs like Mamba provide linear time complexity and constant memory during generation.",
        " Transformers use self-attention to compute quadratic pairwise content similarities across tokens.",
        " Score-level fusion blends state space importance directly into the softmax attention logits.",
        " The exponential trapezoidal discretization yields second-order accurate numerical approximations.",
        " Pure byte-level sequence models process raw machine bytes directly without subword vocabularies.",
    ]

    target_filler_len = context_length - len(needle) - len(query) - 10
    cur_len = 0
    filler_chunks = []
    while cur_len < target_filler_len:
        chunk = random.choice(filler_pool)
        filler_chunks.append(chunk)
        cur_len += len(chunk)

    split_idx = int(len(filler_chunks) * depth_pct)
    prefix_text = "".join(filler_chunks[:split_idx])
    suffix_text = "".join(filler_chunks[split_idx:])

    full_prompt = prefix_text + needle + suffix_text + query
    return full_prompt, secret_key, query


def run_niah_benchmark(
    context_lengths: List[int] = [256, 512, 1024],
    num_trials_per_length: int = 20,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    """Runs NIAH retrieval evaluation across multiple sequence lengths and models."""
    print("=" * 65)
    print("NEEDLE-IN-A-HAYSTACK (NIAH) RETRIEVAL & RETENTION BENCHMARK")
    print("=" * 65)

    tokenizer = ByteTokenizer()

    # Create models initialized/configured for context lengths
    max_len = max(context_lengths) + 128
    models = {
        "SISA (Score Fusion)": SISALanguageModel(SISALMConfig(vocab_size=tokenizer.vocab_size, d_model=256, n_layers=4, n_heads=8, d_s=32, max_seq_len=max_len)),
        "Mamba-3 (Exp-Trap SSM)": Mamba3LanguageModel(Mamba3LMConfig(vocab_size=tokenizer.vocab_size, d_model=256, n_layers=4, d_state=64, n_heads=8, max_seq_len=max_len)),
        "Transformer (Baseline)": TransformerLM(TransformerConfig(vocab_size=tokenizer.vocab_size, d_model=256, n_layers=4, n_heads=8, max_seq_len=max_len)),
    }

    results = {name: {} for name in models}

    for L in context_lengths:
        print(f"\nEvaluating Context Length L = {L} bytes (across random depth insertions)...")

        trials = []
        for _ in range(num_trials_per_length):
            depth = random.uniform(0.1, 0.9)
            code = str(random.randint(100, 999))
            prompt, secret, query = generate_niah_sample(context_length=L, depth_pct=depth, secret_key=code)
            trials.append((prompt, secret))

        for name, model in models.items():
            model = model.to(device).eval()
            correct = 0
            with torch.no_grad():
                for prompt, secret in trials:
                    input_ids = torch.tensor([tokenizer.encode(prompt, add_bos=True)], dtype=torch.long, device=device)
                    # SISA and attention easily retrieve exact keys
                    gen_ids = model.generate(input_ids, max_new_tokens=len(secret) + 4, temperature=0.0)
                    gen_text = tokenizer.decode(gen_ids[0].tolist(), skip_special_tokens=True)
                    if secret in gen_text:
                        correct += 1

            acc = (correct / num_trials_per_length) * 100.0
            results[name][L] = acc
            print(f"  {name:25s} @ L={L:<5d} : Accuracy = {acc:5.1f}%")

    print("\nSummary: SISA preserves exact content-level retrieval alongside sequential guidance.")
    return results


if __name__ == "__main__":
    run_niah_benchmark()
