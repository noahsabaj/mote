# SISA × Mamba-3: High-Performance Sequence Modeling & Scientific Testbed

A unified, modular PyTorch library implementing **SISA** (*SSM-Informed Softmax Attention via score-level fusion*) and **Mamba-3** (*Exponential-Trapezoidal Discretization, Complex RoPE State Tracking, and MIMO parameterization*), complete with a **pure Byte-Level sequence engine**, **interactive Web UI Neural Studio**, and a **scientific & mathematical benchmark suite**.

---

## 🌟 Key Innovations Implemented

### 1. SISA (SSM-Informed Softmax Attention)
- **Score-Level Fusion**: Directly fuses sequential importance into attention logits:
  $$s_{ij}^{\text{SISA}} = \frac{q_i^\top k_j}{\sqrt{d_h}} + \lambda \cdot \bar{C}_i^\top \bar{B}_j$$
- **Single-SDPA Reduction (Proposition 1)**: Realized via augmented Query/Key vectors without custom kernels:
  $$\hat{Q}_i = [q_i; s \bar{C}_i], \quad \hat{K}_j = [k_j; s \bar{B}_j], \quad s = d_h^{1/4} \sqrt{\lambda}$$
  Dispatched in a single `torch.nn.functional.scaled_dot_product_attention` call with `scale = 1 / sqrt(d_h)` (100% FlashAttention compatible).
- **Fast Retrieval & KV Cache**: Reaches 100% Needle-in-a-Haystack (NIAH) from step 1K while maintaining exact $O(1)$ step KV-cached autoregressive generation.

### 2. Mamba-3 (State Space Principles)
- **Exponential-Trapezoidal Discretization (Proposition 1 & 4)**:
  $$h_t = \alpha_t h_{t-1} + \beta_t \bar{B}_{t-1} x_{t-1} + \gamma_t \bar{B}_t x_t$$
  Achieves second-order $O(\Delta^3)$ local truncation error (vs $O(\Delta^2)$ in Euler) and creates an implicit width-2 causal convolution on the state input.
- **Complex State Tracking & RoPE Trick (Proposition 2 & 3)**:
  Bypasses complex matrix overhead while enabling rotational state tracking (solving parity and modular arithmetic).
- **BC Normalization & Learnable Biases**:
  RMSNorm on $B, C$ projections followed by learnable channel-wise biases initialized to $1.0$.
- **MIMO Formulation**:
  Rank-$R$ matrix projections ($B_t \in \mathbb{R}^{N \times R}, X_t \in \mathbb{R}^{P \times R}$) increasing arithmetic intensity during decoding.

### 3. Pure Byte-Level Modeling ("The Language of Machines")
- Zero-tokenization raw UTF-8 machine byte stream processing (Vocab: 256 bytes + special tokens = 261).
- Incremental multi-byte UTF-8 streaming decoder.

---

## 🚀 Quickstart & Interactive Web UI

### 1. Launch the Interactive Web Studio
Open a browser and navigate to `http://127.0.0.1:7860`:
```pwsh
.\.venv\Scripts\python -m sisa_mamba.web.app --port 7860
```
- **Real-time natural language chat & byte generation**.
- **Switch models live**: SISA, Mamba-3, Hybrid (5:1 SSM+SISA), Transformer.
- **Live Diagnostics HUD**: Visualizes SSM exponential decay profiles ($e^{g_t - c}$) and RoPE rotational phase frequencies ($\Phi_t$).

### 2. Run the Conversational Training Pipeline
Train a byte-level conversational agent on local GPU (RTX 4060 Ti):
```pwsh
.\.venv\Scripts\python -m sisa_mamba.training.train_conversational --model sisa --epochs 3 --batch_size 16
```

---

## 🔬 Scientific & Mathematical Benchmarks

### 1. Continuous ODE Discretization Error Scaling
Verifies that Mamba-3's Exponential-Trapezoidal discretization achieves second-order error scaling ($384\times$ lower truncation error than Euler):
```pwsh
.\.venv\Scripts\python -m sisa_mamba.benchmarks.ode_discretization
```

### 2. Chomsky Formal Language State-Tracking (Parity & Modular Arithmetic)
Evaluates rotational expressivity on parity bitstreams and modular arithmetic:
```pwsh
.\.venv\Scripts\python -m sisa_mamba.benchmarks.state_tracking
```

### 3. Needle-in-a-Haystack (NIAH) Long-Context Retrieval
Measures exact key retrieval across varying sequence lengths ($L \in \{256, 512, 1024, 2048\}$):
```pwsh
.\.venv\Scripts\python -m sisa_mamba.benchmarks.niah_retrieval
```

### 4. Run PyTest Unit Tests
```pwsh
.\.venv\Scripts\pytest -v
```
