import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

from ..models.sisa_lm import SISALanguageModel, SISALMConfig
from ..models.mamba3_lm import Mamba3LanguageModel, Mamba3LMConfig
from ..models.transformer_lm import TransformerLM, TransformerConfig


def generate_parity_data(num_samples: int, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates binary sequences and their parity: target = (sum(x_t)) mod 2.
    Tokens: 0, 1, and 2 as separator '='.
    """
    bits = torch.randint(0, 2, (num_samples, seq_len), dtype=torch.long)
    targets = bits.sum(dim=1) % 2
    sep = torch.full((num_samples, 1), 2, dtype=torch.long)
    inputs = torch.cat([bits, sep], dim=1)
    return inputs, targets


def generate_modular_arithmetic_data(
    num_samples: int,
    num_terms: int = 6,
    modulo: int = 7,
    with_brackets: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
    vocab = {
        **{str(d): d for d in range(modulo)},
        "+": modulo,
        "*": modulo + 1,
        "(": modulo + 2,
        ")": modulo + 3,
        "=": modulo + 4,
    }

    inputs = []
    targets = []

    for _ in range(num_samples):
        if not with_brackets:
            terms = [random.randint(0, modulo - 1) for _ in range(num_terms)]
            ops = [random.choice(["+", "*"]) for _ in range(num_terms - 1)]

            expr_tokens = [vocab[str(terms[0])]]
            val = terms[0]

            for i in range(num_terms - 1):
                op = ops[i]
                nxt = terms[i + 1]
                expr_tokens.append(vocab[op])
                expr_tokens.append(vocab[str(nxt)])
                if op == "+":
                    val = (val + nxt) % modulo
                else:
                    val = (val * nxt) % modulo

            expr_tokens.append(vocab["="])
            inputs.append(expr_tokens)
            targets.append(val)
        else:
            val = random.randint(0, modulo - 1)
            expr_tokens = [vocab["("], vocab[str(val)]]
            for _ in range(num_terms - 1):
                op = random.choice(["+", "*"])
                nxt = random.randint(0, modulo - 1)
                expr_tokens.append(vocab[op])
                expr_tokens.append(vocab[str(nxt)])
                expr_tokens.append(vocab[")"])
                if op == "+":
                    val = (val + nxt) % modulo
                else:
                    val = (val * nxt) % modulo
                expr_tokens = [vocab["("]] + expr_tokens

            expr_tokens = expr_tokens[1:] + [vocab["="]]
            inputs.append(expr_tokens)
            targets.append(val)

    max_len = max(len(seq) for seq in inputs)
    padded_inputs = [seq + [vocab["="]] * (max_len - len(seq)) for seq in inputs]

    return torch.tensor(padded_inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long), vocab


def train_and_eval_task(
    model_name: str,
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    num_classes: int,
    epochs: int = 8,
    batch_size: int = 64,
    lr: float = 2e-3,
    device: str = "cuda",
) -> float:
    """Trains model with live progress and returns test accuracy."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    dataset = torch.utils.data.TensorDataset(train_x, train_y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"  Training {model_name:28s} ... ", end="", flush=True)
    t0 = time.time()

    model.train()
    for epoch in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            last_logits = out["logits"][:, -1, :num_classes]
            loss = F.cross_entropy(last_logits, by)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        tx, ty = test_x.to(device), test_y.to(device)
        out = model(tx)
        preds = out["logits"][:, -1, :num_classes].argmax(dim=-1)
        acc = (preds == ty).float().mean().item() * 100.0

    elapsed = time.time() - t0
    print(f"Acc: {acc:5.1f}% ({elapsed:.1f}s)", flush=True)
    return acc


def run_state_tracking_benchmark(device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """Runs the state-tracking suite comparing SISA, Mamba-3, and Transformer."""
    print("=" * 65, flush=True)
    print("CHOMSKY STATE-TRACKING BENCHMARK (Formal Language Expressivity)", flush=True)
    print(f"Running on Device: {device.upper()}", flush=True)
    print("=" * 65, flush=True)

    # Task 1: Parity (Bitstream length 32)
    print("\n--- Task 1: Parity (Bitstream Length = 32) ---", flush=True)
    train_x, train_y = generate_parity_data(600, seq_len=32)
    test_x, test_y = generate_parity_data(200, seq_len=32)

    models = {
        "SISA (Score Fusion + RoPE)": SISALanguageModel(SISALMConfig(vocab_size=3, d_model=128, n_layers=2, n_heads=4, d_s=16, max_seq_len=64)),
        "Mamba-3 (Complex RoPE)": Mamba3LanguageModel(Mamba3LMConfig(vocab_size=3, d_model=128, n_layers=2, d_state=32, n_heads=4, max_seq_len=64)),
        "Mamba-3 (Real / No RoPE)": Mamba3LanguageModel(Mamba3LMConfig(vocab_size=3, d_model=128, n_layers=2, d_state=32, n_heads=4, complex_rope=False, max_seq_len=64)),
        "Transformer (Baseline)": TransformerLM(TransformerConfig(vocab_size=3, d_model=128, n_layers=2, n_heads=4, max_seq_len=64)),
    }

    parity_results = {}
    for name, model in models.items():
        acc = train_and_eval_task(name, model, train_x, train_y, test_x, test_y, num_classes=2, epochs=8, batch_size=64, device=device)
        parity_results[name] = acc

    # Task 2: Modular Arithmetic Modulo 7 (No Brackets)
    print("\n--- Task 2: Modular Arithmetic (No Brackets, Modulo 7) ---", flush=True)
    train_mx, train_my, vocab = generate_modular_arithmetic_data(600, num_terms=5, modulo=7, with_brackets=False)
    test_mx, test_my, _ = generate_modular_arithmetic_data(200, num_terms=5, modulo=7, with_brackets=False)
    v_size = len(vocab)

    mod_models = {
        "SISA (Score Fusion + RoPE)": SISALanguageModel(SISALMConfig(vocab_size=v_size, d_model=128, n_layers=2, n_heads=4, d_s=16, max_seq_len=64)),
        "Mamba-3 (Complex RoPE)": Mamba3LanguageModel(Mamba3LMConfig(vocab_size=v_size, d_model=128, n_layers=2, d_state=32, n_heads=4, max_seq_len=64)),
        "Transformer (Baseline)": TransformerLM(TransformerConfig(vocab_size=v_size, d_model=128, n_layers=2, n_heads=4, max_seq_len=64)),
    }

    mod_results = {}
    for name, model in mod_models.items():
        acc = train_and_eval_task(name, model, train_mx, train_my, test_mx, test_my, num_classes=7, epochs=10, batch_size=64, device=device)
        mod_results[name] = acc

    print("\n" + "=" * 65, flush=True)
    print("SUMMARY RESULTS:", flush=True)
    print(f"Parity Task: SISA = {parity_results['SISA (Score Fusion + RoPE)']:.1f}% | Mamba-3 (Complex) = {parity_results['Mamba-3 (Complex RoPE)']:.1f}% | Mamba-3 (Real) = {parity_results['Mamba-3 (Real / No RoPE)']:.1f}%")
    print(f"Modular Arith: SISA = {mod_results['SISA (Score Fusion + RoPE)']:.1f}% | Mamba-3 (Complex) = {mod_results['Mamba-3 (Complex RoPE)']:.1f}% | Transformer = {mod_results['Transformer (Baseline)']:.1f}%")
    print("=" * 65, flush=True)
    return {"parity": parity_results, "modular_arithmetic": mod_results}


if __name__ == "__main__":
    run_state_tracking_benchmark()
