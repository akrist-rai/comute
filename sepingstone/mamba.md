

## Overall Architecture

Mamba processes sequences through **selective State Space Model (SSM) blocks** that can selectively retain or forget information based on input content.

## Detailed Flow

### 1. **Input Projection**
```
x → Linear projection → splits into two paths
```
- Input sequence x (shape: batch, seq_len, d_model)
- Projects to higher dimension (typically 2 × d_model)
- Splits into two branches: one for SSM path, one for gating

### 2. **Convolution (Short-range Dependencies)**
```
SSM branch → 1D Causal Convolution
```
- Applies depthwise 1D convolution (typically kernel size 4)
- Captures local context and patterns
- Operates causally (only looks at past)

### 3. **Generate Selective Parameters** (The Key Innovation)
```
conv_output → Linear projections → Δ, B, C
```
- **Δ (delta)**: Step size - controls discretization
  - `Δ = softplus(Linear_Δ(conv_output) + bias)`
  - Determines how much to update the state
  
- **B**: Input matrix (shape: batch, seq_len, d_state)
  - `B = Linear_B(conv_output)`
  - Controls what information enters the state
  
- **C**: Output matrix (shape: batch, seq_len, d_state)
  - `C = Linear_C(conv_output)`
  - Controls what information is read from the state

### 4. **Discretization**
```
Convert continuous SSM to discrete form using Δ
```
- Takes continuous parameters A (fixed, learned) and B
- Applies zero-order hold discretization:
  - `A_bar = exp(Δ × A)`
  - `B_bar = (Δ × A)^(-1) × (exp(Δ × A) - I) × Δ × B`
- This makes the continuous SSM work with discrete sequences

### 5. **State Space Computation** (Selective Scan)
```
Recurrent state update: h_t = A_bar × h_{t-1} + B_bar × x_t
Output: y_t = C × h_t
```
- **Sequential processing** through the sequence
- Hidden state h (shape: d_state, typically 16) acts as compressed memory
- State selectively updated based on input-dependent Δ and B
- Can be computed efficiently using parallel scan algorithms

### 6. **Activation**
```
y → SiLU/Swish activation
```
- Applies non-linearity: `SiLU(y) = y × sigmoid(y)`

### 7. **Gating Mechanism**
```
SSM output ⊙ gate_branch
```
- Element-wise multiplication with the second branch from step 1
- Gate branch often goes through SiLU activation
- Similar to gating in GLU (Gated Linear Units)

### 8. **Output Projection**
```
gated_output → Linear projection → output
```
- Projects back to d_model dimension
- Ready for next layer or residual connection

## Complete Flow Diagram

```
Input x
    ↓
Linear (expand to 2×d_model)
    ↓
Split into two branches
    ↓                    ↓
[SSM Path]          [Gate Path]
    ↓                    ↓
Conv1D              SiLU
    ↓                    ↓
Generate Δ,B,C           |
    ↓                    |
Discretize               |
    ↓                    |
Selective Scan           |
    ↓                    |
SiLU                     |
    ↓                    |
    └────── ⊙ ──────────┘
         (element-wise multiply)
              ↓
       Linear (project back)
              ↓
           Output
```

## Key Innovation: Selectivity

The magic of Mamba is that **Δ, B, and C are computed from the input** rather than being fixed parameters. This means:

- **Δ controls how fast** information flows through time
- **B controls what information** enters the state
- **C controls what information** is retrieved from the state

This allows Mamba to:
- Filter out irrelevant information (increase Δ to "forget" quickly)
- Remember important information (decrease Δ to retain longer)
- Focus on specific content based on context

This selectivity mechanism enables Mamba to achieve Transformer-like performance while maintaining O(N) complexity instead of O(N²).
