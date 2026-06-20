# Tiny GPT – A Minimal LLM From Scratch

A tiny GPT-style Transformer built in PyTorch that demonstrates the core concepts behind modern Large Language Models (LLMs) in a compact and understandable implementation.

The goal of this project is educational: to show how tokenization, embeddings, self-attention, transformer blocks, training, and text generation work together to form a language model.

Although extremely small, this model shares the same fundamental architecture used by modern systems such as GPT, Llama, DeepSeek, and EproQi.

---

# Overview

This project implements:

- Character-level tokenizer
- Token embeddings
- Positional embeddings
- Multi-Head Self-Attention
- Feed Forward Network (FFN)
- Layer Normalization
- Residual Connections
- Transformer Blocks
- Autoregressive Training
- Text Generation

The implementation is intentionally minimal so that every component can be studied individually.

---

# Architecture

```text
Input Text
     │
     ▼
Tokenizer
     │
     ▼
Token IDs
     │
     ▼
Token Embedding
     │
     ▼
Position Embedding
     │
     ▼
Transformer Block × N
     │
     ├── Self Attention
     │
     ├── Feed Forward Network
     │
     ├── LayerNorm
     │
     └── Residual Connections
     │
     ▼
Linear Head
     │
     ▼
Vocabulary Logits
     │
     ▼
Next Token Prediction
```

---

# Learning Objectives

This project demonstrates:

1. How text becomes tokens.
2. How tokens become vectors.
3. How self-attention works.
4. Why causal masking is required.
5. How transformer blocks are constructed.
6. How next-token prediction is trained.
7. How autoregressive text generation works.
8. Why model size and dataset size matter.

---

# Step 1 — Tokenization

The tokenizer converts text into integer IDs.

Example:

```text
"cat"
```

becomes:

```text
[3, 1, 20]
```

This project uses:

```python
stoi
itos
```

Mappings:

```text
Character → ID
ID → Character
```

Modern LLMs use subword tokenizers such as:

- SentencePiece
- BPE
- WordPiece

while this demo uses a simple character tokenizer.

---

# Step 2 — Model Configuration

Example configuration:

```python
dim = 64
n_heads = 4
n_layers = 3
```

Meaning:

| Parameter | Description |
|------------|------------|
| dim | Embedding size |
| n_heads | Attention heads |
| n_layers | Transformer layers |

These are miniature versions of real LLM settings.

Example comparison:

| Tiny GPT | EproQi |
|-----------|---------|
| 64 | 1024 |
| 4 heads | 16 heads |
| 3 layers | 28 layers |

---

# Step 3 — Self-Attention

Self-attention allows tokens to look at previous tokens when making predictions.

The model computes:

```text
Q = Query
K = Key
V = Value
```

Attention score:

```text
Attention(Q,K,V)
=
Softmax(QKᵀ / √d)
× V
```

Example:

```text
"The cat sat on the mat"
```

The word:

```text
mat
```

can attend to:

```text
cat
sat
on
the
```

to understand context.

---

# Step 4 — Causal Masking

A language model must not see future tokens.

Allowed:

```text
I love
```

Predict:

```text
pizza
```

Not allowed:

```text
I love pizza
```

and then predict:

```text
pizza
```

because the answer is already visible.

The causal mask enforces:

```text
Past    ✓
Present ✓
Future  ✗
```

---

# Step 5 — Transformer Block

A transformer block contains:

```text
Input
 │
 ▼
Self Attention
 │
 ▼
Add & Norm
 │
 ▼
Feed Forward Network
 │
 ▼
Add & Norm
 │
 ▼
Output
```

Each block learns increasingly abstract language patterns.

---

# Step 6 — Feed Forward Network

The FFN processes each token independently after attention.

Typical structure:

```text
dim
 ↓
4 × dim
 ↓
dim
```

Example:

```text
64
 ↓
256
 ↓
64
```

Purpose:

- Pattern extraction
- Feature transformation
- Non-linear reasoning

---

# Step 7 — Embeddings

Tokens are converted into vectors.

Example:

```text
dog
```

becomes:

```text
[0.13, -0.52, 0.44, ...]
```

Position embeddings are added:

```text
Token Embedding
+
Position Embedding
```

allowing the model to understand token order.

---

# Step 8 — Training

The model learns by predicting the next token.

Example:

```text
The sky is
```

Target:

```text
blue
```

Loss function:

```python
CrossEntropyLoss
```

Training cycle:

```text
Forward Pass
     ↓
Loss
     ↓
Backpropagation
     ↓
Weight Update
     ↓
Repeat
```

As training progresses:

```text
Loss ↓
Predictions ↑
```

---

# Step 9 — Text Generation

Generation is autoregressive.

Example:

Prompt:

```text
The sky is
```

Model predicts:

```text
blue
```

Sequence becomes:

```text
The sky is blue
```

Then predicts:

```text
.
```

Sequence becomes:

```text
The sky is blue.
```

This repeats until generation ends.

---

# Temperature

Temperature controls randomness.

Example:

```python
temperature = 0.1
```

Result:

```text
More deterministic
```

Example:

```python
temperature = 1.5
```

Result:

```text
More creative
More random
```

Typical values:

| Temperature | Behavior |
|-------------|-----------|
| 0.1 | Deterministic |
| 0.7 | Balanced |
| 1.0 | Natural |
| 1.5 | Creative |
| 2.0 | Chaotic |

---

# Example Training Results

Initial loss:

```text
3.27
```

Final loss:

```text
0.03
```

This demonstrates successful learning.

The model memorized most training facts and correctly completed several prompts.

Some outputs may remain imperfect due to:

- Small dataset
- Small parameter count
- Limited training steps

This is a realistic demonstration of model capacity limitations.

---

# Why Some Facts Are Incorrect

When training data is tiny:

```text
Few examples
+
Small model
+
Shared sentence patterns
```

the model may partially confuse facts.

Example:

```text
Fire is hot.
Ice is cold.
```

can become mixed because the model has limited capacity.

Increasing:

```python
dim
```

or

```python
STEPS
```

usually resolves this issue.

---

# Relation To Real LLMs

This project and modern LLMs share the same fundamental ideas.

| Component | Tiny GPT | Modern LLM |
|------------|------------|------------|
| Tokenizer | Character | SentencePiece/BPE |
| Embeddings | Yes | Yes |
| Attention | Yes | Yes |
| FFN | Yes | Yes |
| LayerNorm | Yes | Yes |
| Residuals | Yes | Yes |
| Training | Cross Entropy | Cross Entropy |
| Generation | Autoregressive | Autoregressive |

Modern systems primarily differ in:

- Scale
- Data volume
- Context length
- Advanced positional encoding
- Optimized attention mechanisms

---

# Possible Extensions

To make this model closer to modern LLMs:

- RoPE Positional Encoding
- KV Cache
- Multi-Query Attention
- Grouped Query Attention
- MLA Attention
- Flash Attention
- SentencePiece Tokenizer
- Mixed Precision Training
- Distributed Training

---

# Key Takeaway

A modern LLM is fundamentally:

```text
Tokenizer
+
Embeddings
+
Attention
+
Feed Forward Networks
+
Training Data
+
Scale
```

This tiny GPT demonstrates all of those concepts in a form that can be trained and understood in minutes rather than requiring billions of parameters and massive compute resources.
I'll build you a minimal but complete LLM — a tiny GPT-style transformer you can train and run on CPU in seconds. It's structured step-by-step so each piece maps to a real concept (tokenizer → embeddings → attention → FFN → training loop → generation). It shares the same DNA as eproqi, just shrunk to the bone so the mechanics are visible.Torch is ready. Let me write the program and actually run it end-to-end so you see real output, not just code.It works. The loss fell from 3.27 (random — model has no idea, ~log(22)) to ~0.03 (confident), and it correctly completed 3 of 5 facts verbatim. "fire" and "ice" wobble because with only ~154K params, 76 training tokens, and short shared prefixes ("X is Y."), the tiny model partially confuses overlapping patterns — which is itself a real lesson: too little data + too small model = imperfect memorization. Bump `STEPS` or `dim` slightly and all five lock in.

## Here's the file:## What you just watched happen

**The 8 steps map directly onto eproqi**, just shrunk so you can see them:

1. **Tokenizer** — text → integer ids. Mine is character-level (22 tokens); eproqi's is a 94K SentencePiece subword model. Same job, same `stoi`/`itos` idea.
2. **Config** — dim/n_heads/n_layers, the exact knobs in your `config.json`, just tiny (64/4/3 vs 1024/16/28).
3. **Self-attention** — Q·K scores decide who attends to whom; the causal mask blocks peeking at future tokens. This is the *one* mechanism that makes a transformer a transformer. (eproqi's MLA is this plus KV-compression.)
4. **Block** — attention + FFN, each with LayerNorm and a residual `+x`. Stack N of them.
5. **Model** — token embedding + position embedding → blocks → a linear head producing one logit per vocab token.
6. **Training** — predict the next token, cross-entropy measures surprise, backprop nudges weights. The falling loss *is* learning.
7. **Generation** — feed a prompt, take the last position's logits, sample, append, repeat. Autoregression.
8. **Test** — the 5 facts, completed from a prompt.

## Two knobs to play with for intuition

- **`temperature`** in `generate` (currently 0.8): set it to `0.1` → rigidly repeats training data; set it to `1.5` → creative/garbled. This is the same temperature you tune on eproqi's endpoint.
- **`STEPS`/`dim`**: raise either and the two wobbly facts ("fire", "ice") snap into place — a live demo of capacity/training vs memorization.

The whole thing is ~150 lines and runs in seconds on CPU. The conceptual distance from this to eproqi is *scale and a few attention refinements (MLA, RoPE)* — not a different kind of object. If you want, I can add RoPE positional encoding or swap in MLA-style KV compression next, so it lines up even more closely with your actual config.# LLM-WORKING
