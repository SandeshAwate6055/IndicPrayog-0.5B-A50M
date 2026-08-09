# Experiments

Two experiment stages sit on top of the base training pipeline
([05-training-methodology](../05-training-methodology/README.md)): a controlled 6-way ablation,
and a proxy-language-model comparison used to validate that intrinsic metrics actually predict
downstream learning quality.

## Stage 1 — The 6-way ablation

Six tokenizer variants (A–F) were trained on the **identical corpus**, changing one design
factor at a time — see [02-tokenizer-variants](../02-tokenizer-variants/README.md) for the
full table of what each variant changes and why. Holding the corpus, vocab size, and seed
constant across all six isolates the effect of each individual design choice.

Three bugs were found and fixed during this stage, all affecting Devanagari quality
specifically (`add_prefix_space`, `UnicodeScripts` ordering, and the Letter-vs-Mark regex
issue) — see [03-architecture-and-configuration](../03-architecture-and-configuration/README.md)
for the technical detail on each.

A pre-registered shortlisting gate (written before results were seen) reduced the 6 variants
to 3 finalists — **A, C, D** — based on fertility subject to round-trip = 100% and Devanagari
integrity ≥ the Variant A baseline. Details in
[02-tokenizer-variants](../02-tokenizer-variants/README.md#shortlisting-gate).

## Stage 2 — Proxy-LM comparison

Intrinsic metrics (fertility, parity, etc. — see
[07-evaluation-methodology](../07-evaluation-methodology/README.md)) measure compression
efficiency, not whether a lower-fertility tokenizer actually helps a model learn better. To
test that directly, a small language model was trained from scratch on each of the 3
shortlisted tokenizers (A, C, D) and compared.

### Model architecture

A 28.1M-parameter Llama-style model: 6 transformer layers, hidden size 320, 5 attention
heads, RoPE positional encoding, SwiGLU MLP, RMSNorm, tied embedding/output weights. Small
enough to train in ~5 minutes on a single GPU, large enough to show real learning signal.

### Training setup

- 3,000 gradient steps
- Batch size 32 × sequence length 1024 = 32,768 tokens/step (~98M tokens seen per run)
- AdamW, learning rate 3e-4, 100-step warmup
- Fixed seed 1234, bfloat16 precision, on an RTX PRO 6000 (96 GB)

### Controlling for data, not just tokenizer

Each of the 3 candidates trained on the **exact same source documents** — only the
tokenization differed. The training split (start of the pool file) and held-out eval split
(last 500 docs per language) were identical across all three runs, so any BPB difference is
attributable to the tokenizer, not to seeing different data. See
[08-evaluation-datasets](../08-evaluation-datasets/README.md) for the split details.

### Why bits-per-byte (BPB), not loss-per-token

Cross-entropy loss is measured per *token*. A tokenizer that uses more tokens per sentence
(e.g. 1.27 vs 1.24 tokens/word) means its model sees fewer sentences per training step; a raw
loss comparison would unfairly penalize that tokenizer even if its tokens are individually
harder (rarer, more morphologically complete). BPB instead normalizes by raw UTF-8 **bytes**
in the source text, so all three models are compared on how well they reconstruct exactly the
same underlying bytes — the only fair basis for comparing models trained on different
tokenizers.

### The initialization bug

The first proxy-LM run showed a step-0 loss of ~307 nats. The correct value for a randomly
initialized 64K-vocab LM is `ln(64000) ≈ 11.07` (equivalent to guessing uniformly at random
among all tokens) — so 307 was 28x too high.

**Root cause:** PyTorch's default `nn.Embedding` initialization is `N(0, 1)` (standard
deviation 1.0). Because the output head is tied to the embedding, that same std-1.0 matrix
produces the logits directly; with hidden size 320, logits end up with std ≈ `sqrt(320) ≈
17.9`. Cross-entropy on logits that large against random targets produces astronomically high
loss, and the model spent most of its 3,000 steps recovering from the broken initialization
instead of learning real signal.

**Fix:** GPT-2/Llama-style initialization — `N(0, 0.02)` for all Linear and Embedding
weights. After the fix, step-0 loss came in at 11.1 for all three candidates (within 0.1% of
the theoretical `ln(64000) = 11.07`), confirming correct initialization. The buggy first run
(reported "winner: D, mean_bpb=5.4") is explicitly discarded and is **not** the number
reported in [10-final-tokenizer-selection](../10-final-tokenizer-selection/README.md).

### Result

Variant **D** (Unigram) won on mean BPB and on every individual language (Hindi, English,
Marathi) — see the full table in
[10-final-tokenizer-selection](../10-final-tokenizer-selection/README.md#proxy-lm-results-corrected-run).
The proxy-LM ranking agreed with the intrinsic fertility-based shortlist ranking, so no
override of the intrinsic ranking was needed.

## Why the results should be trusted

1. **Standard, public benchmark** — FLORES-200 devtest, used community-wide for multilingual
   tokenizer comparison (Petrov et al. 2023, the canonical tokenizer-fairness paper).
2. **Same code, same data, for every tokenizer compared** — all 14 tokenizers (6 ours + 8
   baselines) ran through one identical evaluation function on the same sentences.
3. **No eval contamination** — FLORES devtest was never in the training corpus; the proxy-LM
   held-out docs came from past the skip point used during corpus sampling.
4. **Identical proxy-LM conditions** — same architecture, hyperparameters, seed, and source
   documents across the three candidates; only the tokenizer differs.
5. **Step-0 sanity check** — every proxy-LM run is verified to start near
   `ln(vocab_size) = 11.07` before its results are trusted.
6. **Pre-registered selection rule** — the shortlisting gate and "lowest mean BPB wins" rule
   were fixed before any evaluation ran.
7. **All bugs disclosed** — each of the four bugs found (three tokenizer-config bugs plus the
   initialization bug) is documented with its symptom and fix, here and in
   [03-architecture-and-configuration](../03-architecture-and-configuration/README.md).
