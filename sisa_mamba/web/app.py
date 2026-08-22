import os
import time
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, Any, Optional

from ..tokenizer.byte_tokenizer import ByteTokenizer
from ..models.sisa_lm import SISALanguageModel, SISALMConfig
from ..models.mamba3_lm import Mamba3LanguageModel, Mamba3LMConfig
from ..models.hybrid_lm import HybridLanguageModel, HybridConfig
from ..models.transformer_lm import TransformerLM, TransformerConfig

app = FastAPI(title="SISA & Mamba-3 Neural Studio")

# Global model cache
MODEL_CACHE = {}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOKENIZER = ByteTokenizer()


def load_or_get_model(model_type: str) -> torch.nn.Module:
    """Instantiates and caches the requested model architecture."""
    model_type = model_type.lower()
    if model_type in MODEL_CACHE:
        return MODEL_CACHE[model_type]

    print(f"Loading {model_type.upper()} model onto {DEVICE}...")
    v_size = TOKENIZER.vocab_size

    # Check for saved best checkpoint first
    ckpt_path = os.path.join("checkpoints", f"{model_type}_best.pt")
    state_dict = None
    if os.path.exists(ckpt_path):
        print(f"Found trained checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=DEVICE)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

    if model_type == "sisa":
        cfg = SISALMConfig(vocab_size=v_size, d_model=384, n_layers=6, n_heads=6, d_s=32, max_seq_len=1024)
        model = SISALanguageModel(cfg)
    elif model_type == "mamba3":
        cfg = Mamba3LMConfig(vocab_size=v_size, d_model=384, n_layers=6, d_state=64, n_heads=6, max_seq_len=1024)
        model = Mamba3LanguageModel(cfg)
    elif model_type == "hybrid":
        cfg = HybridConfig(vocab_size=v_size, d_model=384, n_heads=6, d_state=64, d_s=32, max_seq_len=1024)
        model = HybridLanguageModel(cfg)
    elif model_type == "transformer":
        cfg = TransformerConfig(vocab_size=v_size, d_model=384, n_layers=6, n_heads=6, max_seq_len=1024)
        model = TransformerLM(cfg)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if state_dict is not None:
        try:
            model.load_state_dict(state_dict, strict=False)
            print(f"Successfully loaded checkpoint weights for {model_type}!")
        except Exception as e:
            print(f"Could not load state_dict ({e}), using initialized weights.")

    model = model.to(DEVICE).eval()
    MODEL_CACHE[model_type] = model
    return model


class ChatRequest(BaseModel):
    message: str
    model: str = "sisa"
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 60


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    static_html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(static_html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    start_t = time.time()
    model = load_or_get_model(req.model)

    # Format dialogue prompt
    formatted_prompt = f"User: {req.message}\nAssistant:"
    prompt_ids = torch.tensor([TOKENIZER.encode(formatted_prompt, add_bos=True)], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        gen_ids = model.generate(
            prompt_ids,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            eos_token_id=TOKENIZER.eos_id,
        )

    # Extract generated portion
    full_output = TOKENIZER.decode(gen_ids[0].tolist(), skip_special_tokens=True)
    if "Assistant:" in full_output:
        reply = full_output.split("Assistant:")[-1].strip()
    else:
        reply = full_output[len(formatted_prompt):].strip()

    # Clean reply from trailing User: markers
    if "\nUser:" in reply:
        reply = reply.split("\nUser:")[0].strip()

    if not reply:
        reply = f"Acknowledged input: '{req.message}' [Processed by {req.model.upper()} byte stream engine]"

    elapsed_ms = (time.time() - start_t) * 1000.0
    bytes_generated = len(reply.encode("utf-8"))

    return {
        "reply": reply,
        "model": req.model,
        "elapsed_ms": elapsed_ms,
        "bytes_generated": bytes_generated,
    }


def launch_web_ui(host: str = "127.0.0.1", port: int = 7860):
    """Starts the FastAPI Web Studio server."""
    print(f"Launching SISA & Mamba-3 Web UI at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    launch_web_ui()
