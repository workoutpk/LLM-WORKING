"""
tiny_llm.py — A minimal GPT-style language model you can read in one sitting.

It is the same family as eproqi / DeepSeek (decoder-only transformer), shrunk so
every moving part is visible. We train it on 5 tiny "facts" and then ask it to
complete them, so you can watch it go from random noise -> memorizing the data.

Pipeline (each STEP below maps to a real LLM concept):
  STEP 1  Data + character tokenizer      (text  <-> integer ids)
  STEP 2  Config                          (the knobs, mini version of your config.json)
  STEP 3  Self-attention head             (the "look at other tokens" mechanism)
  STEP 4  Transformer block               (attention + feed-forward + residuals)
  STEP 5  The full model                  (embeddings -> blocks -> output logits)
  STEP 6  Training loop                   (predict next token, measure loss, learn)
  STEP 7  Generation                      (sample one token at a time)
  STEP 8  Test on our 5 samples
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)  # reproducible

# =====================================================================
# STEP 1 — DATA + TOKENIZER
# An LLM never sees text; it sees integers. The tokenizer is the
# dictionary that maps characters <-> integer ids. (eproqi uses a 94K
# subword vocab; here we use single characters so vocab is tiny.)
# =====================================================================

SAMPLES = [
    "the sky is blue.",
    "the grass is green.",
    "water is wet.",
    "fire is hot.",
    "ice is cold.",
]

text = "\n".join(SAMPLES)                 # one training blob
chars = sorted(set(text))                 # our "vocabulary"
vocab_size = len(chars)

stoi = {c: i for i, c in enumerate(chars)}   # string  -> id  (encode)
itos = {i: c for i, c in enumerate(chars)}   # id      -> string (decode)
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)
print(f"[STEP 1] vocab_size={vocab_size}  total_tokens={len(data)}")
print(f"         vocab = {''.join(chars)!r}")


# =====================================================================
# STEP 2 — CONFIG  (your config.json, in miniature)
# =====================================================================
class Config:
    vocab_size = vocab_size
    block_size = 32     # max context length  (your max_seq_len)
    dim        = 64     # hidden size          (your dim=1024)
    n_heads    = 4      # attention heads      (your n_heads=16)
    n_layers   = 3      # transformer blocks   (your n_layers=28)
    head_dim   = dim // n_heads

cfg = Config()


# =====================================================================
# STEP 3 — SELF-ATTENTION  (the heart of the transformer)
# For each token, build Query/Key/Value. Q·K decides "how much should
# token i pay attention to token j". A causal mask forbids looking at
# FUTURE tokens (a language model may only use the past to predict next).
# =====================================================================
class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads  = cfg.n_heads
        self.head_dim = cfg.head_dim
        # one big linear makes Q, K, V together, then we split
        self.qkv  = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.proj = nn.Linear(cfg.dim, cfg.dim, bias=False)
        # causal mask: lower-triangular 1s (allowed), upper 0s (masked)
        self.register_buffer(
            "mask", torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        )

    def forward(self, x):
        B, T, C = x.shape                       # batch, time(seq), channels(dim)
        q, k, v = self.qkv(x).split(C, dim=2)   # each (B, T, C)
        # reshape into separate heads: (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # attention scores: how much each token attends to every other
        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        att = att.masked_fill(self.mask[:T, :T] == 0, float("-inf"))  # no peeking ahead
        att = F.softmax(att, dim=-1)            # turn scores into weights summing to 1
        out = att @ v                           # weighted sum of values

        out = out.transpose(1, 2).contiguous().view(B, T, C)  # merge heads back
        return self.proj(out)


# =====================================================================
# STEP 4 — TRANSFORMER BLOCK
# attention (mix info across tokens) + feed-forward (think per token),
# each wrapped with LayerNorm and a residual (+x) connection.
# =====================================================================
class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.dim)
        self.att = MultiHeadAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.dim)
        self.ff  = nn.Sequential(               # the "FFN" (your inter_dim)
            nn.Linear(cfg.dim, 4 * cfg.dim),
            nn.GELU(),
            nn.Linear(4 * cfg.dim, cfg.dim),
        )

    def forward(self, x):
        x = x + self.att(self.ln1(x))           # residual + attention
        x = x + self.ff(self.ln2(x))            # residual + feed-forward
        return x


# =====================================================================
# STEP 5 — THE FULL MODEL
# token embedding + position embedding -> N blocks -> final linear that
# produces a score (logit) for every possible next token.
# =====================================================================
class TinyLLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)   # what the token is
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.dim)   # where it sits
        self.blocks  = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f    = nn.LayerNorm(cfg.dim)
        self.head    = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)  # -> logits

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)   # combine meaning + position
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)                       # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # cross-entropy: how surprised was the model by the true next token?
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8):
        # STEP 7 (used below): repeatedly predict the next token and append it
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]      # keep within context
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature       # focus on last position
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # sample
            idx = torch.cat([idx, next_id], dim=1)
            if itos[next_id.item()] == "\n":              # stop at line end
                break
        return idx


model = TinyLLM(cfg)
n_params = sum(p.numel() for p in model.parameters())
print(f"[STEP 5] model built: {n_params:,} parameters "
      f"({cfg.n_layers} layers, {cfg.n_heads} heads, dim={cfg.dim})")


# =====================================================================
# STEP 6 — TRAINING LOOP
# Grab random chunks, ask the model to predict each next character,
# compute loss, backprop, update weights. Watch the loss fall.
# =====================================================================
def get_batch(batch_size=16):
    # pick random start points, x = chunk, y = same chunk shifted by 1
    ix = torch.randint(len(data) - cfg.block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + cfg.block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + cfg.block_size] for i in ix])
    return x, y

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

print("[STEP 6] training...")
STEPS = 3000
for step in range(STEPS):
    xb, yb = get_batch()
    _, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 500 == 0 or step == STEPS - 1:
        print(f"         step {step:4d}  loss {loss.item():.4f}")


# =====================================================================
# STEP 7 + 8 — TEST ON OUR 5 SAMPLES
# Give the model the first few characters of each fact and let it
# complete the rest. A trained model should reproduce the facts.
# =====================================================================
print("\n[STEP 8] testing — feed a prompt, model completes it:\n")
prompts = ["the sky", "the grass", "water", "fire", "ice"]
for p in prompts:
    ctx = torch.tensor([encode(p)], dtype=torch.long)
    out = model.generate(ctx, max_new_tokens=30)
    print(f"   prompt {p!r:12s} -> {decode(out[0].tolist())!r}")