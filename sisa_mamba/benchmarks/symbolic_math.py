import random
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple

from ..tokenizer.byte_tokenizer import ByteTokenizer
from ..models.sisa_lm import SISALanguageModel, SISALMConfig
from ..models.mamba3_lm import Mamba3LanguageModel, Mamba3LMConfig
from ..models.transformer_lm import TransformerLM, TransformerConfig


def generate_arithmetic_equations(num_samples: int = 1000, max_digits: int = 3) -> List[Tuple[str, str]]:
    """Generates arithmetic question/answer pairs like ('142 + 389 = ', '531')."""
    pairs = []
    for _ in range(num_samples):
        op = random.choice(["+", "-", "*"])
        if op in ("+", "-"):
            a = random.randint(10 ** (max_digits - 1), 10 ** max_digits - 1)
            b = random.randint(10 ** (max_digits - 1), 10 ** max_digits - 1)
            if op == "+":
                ans = a + b
            else:
                if a < b:
                    a, b = b, a
                ans = a - b
        else:
            # Multiplication with 1-2 digits
            a = random.randint(10, 99)
            b = random.randint(2, 99)
            ans = a * b

        prompt = f"Calc: {a} {op} {b} = "
        answer = f"{ans}"
        pairs.append((prompt, answer))
    return pairs


def run_symbolic_math_benchmark(device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """Evaluates symbolic math equation solving on byte sequences."""
    print("=" * 60)
    print("SYMBOLIC & MULTI-DIGIT MATHEMATICAL REASONING BENCHMARK")
    print("=" * 60)

    tokenizer = ByteTokenizer()
    pairs = generate_arithmetic_equations(num_samples=1200)
    train_pairs = pairs[:1000]
    test_pairs = pairs[1000:]

    # Prepare tensors
    def encode_pairs(data_pairs):
        inputs, targets = [], []
        for prompt, ans in data_pairs:
            full = prompt + ans
            p_tokens = tokenizer.encode(prompt, add_bos=True)
            f_tokens = tokenizer.encode(full, add_bos=True, add_eos=True)
            inputs.append(f_tokens)
        max_len = max(len(s) for s in inputs)
        padded_x = [s[:-1] + [tokenizer.pad_id] * (max_len - len(s)) for s in inputs]
        padded_y = [s[1:] + [tokenizer.pad_id] * (max_len - len(s)) for s in inputs]
        return torch.tensor(padded_x, dtype=torch.long), torch.tensor(padded_y, dtype=torch.long)

    train_x, train_y = encode_pairs(train_pairs)
    test_x, test_y = encode_pairs(test_pairs)

    print(f"Dataset: {len(train_pairs)} training math problems, {len(test_pairs)} test problems.")

    models = {
        "SISA LM": SISALanguageModel(SISALMConfig(vocab_size=tokenizer.vocab_size, d_model=192, n_layers=4, n_heads=6, d_s=16, max_seq_len=128)),
        "Mamba-3 LM": Mamba3LanguageModel(Mamba3LMConfig(vocab_size=tokenizer.vocab_size, d_model=192, n_layers=4, d_state=32, n_heads=6, max_seq_len=128)),
        "Transformer LM": TransformerLM(TransformerConfig(vocab_size=tokenizer.vocab_size, d_model=192, n_layers=4, n_heads=6, max_seq_len=128)),
    }

    results = {}
    for name, model in models.items():
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        dataset = torch.utils.data.TensorDataset(train_x, train_y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        model.train()
        for epoch in range(10):
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                out = model(bx, labels=by)
                loss = out["loss"]
                loss.backward()
                optimizer.step()

        # Evaluate on test set
        model.eval()
        correct = 0
        with torch.no_grad():
            for prompt, expected_ans in test_pairs[:100]:
                p_ids = torch.tensor([tokenizer.encode(prompt, add_bos=True)], dtype=torch.long, device=device)
                gen_ids = model.generate(p_ids, max_new_tokens=len(expected_ans) + 2, temperature=0.0)
                gen_text = tokenizer.decode(gen_ids[0].tolist(), skip_special_tokens=True)
                if prompt in gen_text:
                    pred_ans = gen_text.split(prompt)[-1].strip().split()[0] if gen_text.split(prompt)[-1].strip() else ""
                    if pred_ans.startswith(expected_ans):
                        correct += 1

        acc = (correct / 100.0) * 100.0
        results[name] = acc
        print(f"{name:20s}: Exact-Match Arithmetic Accuracy = {acc:.1f}%")

    return results


if __name__ == "__main__":
    run_symbolic_math_benchmark()
