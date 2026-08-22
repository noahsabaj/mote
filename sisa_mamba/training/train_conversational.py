import os
import time
import math
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, Any, List

from ..tokenizer.byte_tokenizer import ByteTokenizer, get_tokenizer
from ..models.transformer_lm import TransformerLM, TransformerConfig
from ..models.sisa_lm import SISALanguageModel, SISALMConfig
from ..models.mamba3_lm import Mamba3LanguageModel, Mamba3LMConfig
from ..models.hybrid_lm import HybridLanguageModel, HybridConfig


class TextDataset(Dataset):
    """Dataset that chunks encoded token IDs into sequences of fixed length."""
    def __init__(self, token_ids: List[int], seq_len: int = 512):
        self.seq_len = seq_len
        num_samples = len(token_ids) // (seq_len + 1)
        if num_samples == 0:
            # Pad if too short
            padded = token_ids + [0] * ((seq_len + 1) - len(token_ids))
            self.data = torch.tensor(padded, dtype=torch.long).unsqueeze(0)
        else:
            total_len = num_samples * (seq_len + 1)
            self.data = torch.tensor(token_ids[:total_len], dtype=torch.long).view(num_samples, seq_len + 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def load_conversational_text_data(dataset_name: str = "TinyStories", max_samples: int = 10000) -> str:
    """Loads a conversational/story dataset from Hugging Face or creates high quality synthetic dialogue."""
    try:
        from datasets import load_dataset
        if dataset_name.lower() in ("tinystories", "stories"):
            print("Loading TinyStories from Hugging Face...")
            ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
            texts = []
            for i, item in enumerate(ds):
                texts.append(item["text"])
                if i >= max_samples:
                    break
            return "\n\n<EOS>\n\n".join(texts)
        elif dataset_name.lower() in ("dialog", "daily_dialog"):
            print("Loading DailyDialog from Hugging Face...")
            ds = load_dataset("daily_dialog", split="train")
            texts = []
            for item in ds.select(range(min(len(ds), max_samples))):
                dialogue = "\n".join([f"User: {u}" if j % 2 == 0 else f"Assistant: {u}" for j, u in enumerate(item["dialog"])])
                texts.append(dialogue)
            return "\n\n<EOS>\n\n".join(texts)
    except Exception as e:
        print(f"Hugging Face dataset download fallback ({e}). Using built-in conversational curriculum.")

    # High-quality built-in synthetic dialogue corpus
    conversations = [
        "User: Hello! Who are you?\nAssistant: I am an intelligent language model powered by SISA and Mamba-3 sequence modeling.",
        "User: Explain what SISA is.\nAssistant: SISA stands for SSM-Informed Softmax Attention. It fuses state space dynamics directly inside the attention score via augmented query-key vectors, running in a single FlashAttention SDPA call.",
        "User: What makes Mamba-3 special?\nAssistant: Mamba-3 introduces Exponential-Trapezoidal discretization with O(Delta^3) local error, a data-dependent complex RoPE trick for state tracking, and MIMO matrix state projections for fast decoding.",
        "User: Can you solve modular arithmetic?\nAssistant: Yes, through complex-valued state transitions and rotational embeddings, modular addition and parity can be tracked perfectly across long sequences.",
        "User: What is the benefit of byte-level modeling?\nAssistant: Byte-level modeling operates directly on raw machine bytes without tokenizer vocabulary biases or out-of-vocabulary words.",
        "User: How do you handle long context retrieval?\nAssistant: SISA reaches 100% retrieval accuracy on Needle-in-a-Haystack benchmarks from the very first thousand training steps by combining global attention with sequential priors.",
        "User: Write a poem about machines and attention.\nAssistant: Silicon pulses in the night,\nThrough state and score a guided light.\nWith every byte the matrices align,\nIn recurrent loops and sparks divine.",
        "User: Calculate 25 * 4.\nAssistant: 25 multiplied by 4 equals 100.",
        "User: What is the capital of France?\nAssistant: The capital of France is Paris.",
        "User: How does the trapezoidal rule improve ODE discretization?\nAssistant: Unlike Euler's rule which holds the endpoint constant, the trapezoidal rule averages both endpoints, achieving second-order accuracy and lower truncation error.",
    ] * (max_samples // 10 + 1)
    return "\n\n<EOS>\n\n".join(conversations)


def get_model(
    model_type: str,
    vocab_size: int,
    d_model: int = 384,
    n_layers: int = 6,
    n_heads: int = 6,
    d_s: int = 32,
    d_state: int = 64,
    seq_len: int = 512,
) -> nn.Module:
    """Factory creating the selected model architecture."""
    if model_type.lower() == "sisa":
        cfg = SISALMConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_head=d_model // n_heads,
            d_s=d_s,
            max_seq_len=seq_len,
        )
        return SISALanguageModel(cfg)
    elif model_type.lower() == "mamba3":
        cfg = Mamba3LMConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            n_heads=n_heads,
            d_head=d_model // n_heads,
            max_seq_len=seq_len,
        )
        return Mamba3LanguageModel(cfg)
    elif model_type.lower() == "hybrid":
        cfg = HybridConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            d_head=d_model // n_heads,
            d_state=d_state,
            d_s=d_s,
            max_seq_len=seq_len,
            layer_pattern=["mamba3", "mamba3", "mamba3", "sisa", "mamba3", "sisa"],
        )
        return HybridLanguageModel(cfg)
    elif model_type.lower() == "transformer":
        cfg = TransformerConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_head=d_model // n_heads,
            max_seq_len=seq_len,
        )
        return TransformerLM(cfg)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_conversational(
    model_type: str = "sisa",
    dataset_name: str = "TinyStories",
    d_model: int = 384,
    n_layers: int = 6,
    n_heads: int = 6,
    d_s: int = 32,
    seq_len: int = 512,
    batch_size: int = 8,
    lr: float = 5e-4,
    epochs: int = 3,
    max_steps: Optional[int] = 500,
    save_dir: str = "checkpoints",
    device: Optional[str] = None,
):
    """End-to-end training and evaluation loop."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(save_dir, exist_ok=True)
    tokenizer = ByteTokenizer()

    print(f"=== SISA & Mamba-3 Training Pipeline ===")
    print(f"Model: {model_type.upper()} | Device: {device} | Vocab: {tokenizer.vocab_size} (Byte-level)")

    # 1. Load Data
    raw_text = load_conversational_text_data(dataset_name, max_samples=3000)
    token_ids = tokenizer.encode(raw_text, add_bos=True, add_eos=True)
    print(f"Encoded {len(token_ids):,} raw bytes for training.")

    split_idx = int(len(token_ids) * 0.9)
    train_tokens = token_ids[:split_idx]
    val_tokens = token_ids[split_idx:]

    train_dataset = TextDataset(train_tokens, seq_len=seq_len)
    val_dataset = TextDataset(val_tokens, seq_len=seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 2. Build Model
    model = get_model(
        model_type=model_type,
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_s=d_s,
        seq_len=seq_len,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,} ({total_params / 1e6:.2f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)

    total_training_steps = min(len(train_loader) * epochs, max_steps or 999999)
    warmup_steps = min(50, total_training_steps // 10)

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_training_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    global_step = 0
    best_val_loss = float("inf")
    start_time = time.time()

    model.train()
    for epoch in range(epochs):
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x, labels=y)
            loss = outputs["loss"]
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1

            if global_step % 20 == 0 or global_step == 1:
                ppl = math.exp(min(loss.item(), 20.0))
                print(f"Epoch {epoch+1} | Step {global_step}/{total_training_steps} | Loss: {loss.item():.4f} | PPL: {ppl:.2f} | LR: {scheduler.get_last_lr()[0]:.2e}")

            if global_step % 100 == 0 or global_step == total_training_steps:
                # Validation evaluation
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for vx, vy in val_loader:
                        vx, vy = vx.to(device), vy.to(device)
                        v_out = model(vx, labels=vy)
                        val_losses.append(v_out["loss"].item())
                val_loss = sum(val_losses) / max(len(val_losses), 1)
                val_ppl = math.exp(min(val_loss, 20.0))
                print(f"--> Validation Step {global_step} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    ckpt_path = os.path.join(save_dir, f"{model_type.lower()}_best.pt")
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "model_type": model_type,
                        "config": model.config,
                        "step": global_step,
                        "val_loss": val_loss,
                    }, ckpt_path)
                    print(f"--> Saved best checkpoint to {ckpt_path}")

                model.train()

            if max_steps and global_step >= max_steps:
                break
        if max_steps and global_step >= max_steps:
            break

    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.1f}s. Best Val Loss: {best_val_loss:.4f}")

    # Generate sample completion
    model.eval()
    test_prompt = "User: What is SISA?\nAssistant:"
    prompt_ids = torch.tensor([tokenizer.encode(test_prompt, add_bos=True)], dtype=torch.long, device=device)
    gen_ids = model.generate(prompt_ids, max_new_tokens=40, temperature=0.7, top_p=0.9)
    generated_text = tokenizer.decode(gen_ids[0].tolist(), skip_special_tokens=True)
    print(f"\n--- Sample Generation ---")
    print(generated_text)
    print(f"-------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SISA, Mamba-3, Hybrid, or Transformer models.")
    parser.add_argument("--model", type=str, default="sisa", choices=["sisa", "mamba3", "hybrid", "transformer"])
    parser.add_argument("--dataset", type=str, default="TinyStories")
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=6)
    parser.add_argument("--d_s", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    train_conversational(
        model_type=args.model,
        dataset_name=args.dataset,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_s=args.d_s,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        max_steps=args.max_steps,
        save_dir=args.save_dir,
    )
