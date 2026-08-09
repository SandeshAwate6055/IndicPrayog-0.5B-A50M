# Results

Full intrinsic-metrics table across all 6 of our variants plus 8 external baselines, measured
on FLORES-200 devtest (see [08-evaluation-datasets](../08-evaluation-datasets/README.md) and
[07-evaluation-methodology](../07-evaluation-methodology/README.md) for what each column means).

```
Tokenizer         | hi fert | mr fert | en fert | hi parity | mr parity | integrity | RT
------------------+---------+---------+---------+-----------+-----------+-----------+-----
ours_D (WINNER)   |  1.27   |  1.54   |  1.23   |   1.21    |   1.09    |  96.0%    | 100%
ours_A            |  1.24   |  1.54   |  1.24   |   1.17    |   1.08    |  93.2%    | 100%
ours_C            |  1.24   |  1.56   |  1.24   |   1.17    |   1.09    |  93.3%    | 100%
ours_E            |  1.38   |  1.69   |  1.39   |   1.16    |   1.06    |  94.4%    | 100%
ours_F            |  1.22   |  1.51   |  1.21   |   1.18    |   1.08    |  93.2%    | 100%
ours_B            |  1.26   |  1.51   |  1.25   |   1.18    |   1.06    |  92.7%    | 100%
------------------+---------+---------+---------+-----------+-----------+-----------+-----
sarvam-1          |  1.40   |  1.77   |  1.43   |   1.14    |   1.07    |  95.6%    | 100%  ← Indic gold standard
sarvam-30b        |  1.39   |  1.99   |  1.24   |   1.31    |   1.40    |  94.7%    | 100%
gemma-2           |  1.96   |  3.20   |  1.23   |   1.86    |   2.26    |  81.0%    | 100%
qwen2.5           |  4.76   |  6.69   |  1.26   |   4.42    |   4.62    |  40.5%    | 100%
mistral-v0.3      |  5.33   |  7.34   |  1.37   |   4.55    |   4.66    |  39.9%    | 100%
gpt2              |  7.82   | 11.16   |  1.23   |   7.41    |   7.86    |  39.9%    | 100%
indicbert-v2      |  1.24   |  1.49   |  1.24   |   1.17    |   1.05    |  87.3%    |  0.1% ← lossy by design
```

## Interpretation

### Against general-purpose baselines (GPT-2 / Mistral / Qwen2.5)

These tokenizers were built for English. GPT-2 scores 7.82 tokens/word on Hindi versus our
Variant D's 1.27 — a **6.2x improvement**. Mistral (same vocab size class, 33K) scores 5.33
versus our 1.27 — **4.2x better**. Their parity scores (4.5–7.4) mean Hindi effectively paid
4–7x more context per thought than English under those tokenizers; ours scores 1.21,
essentially fair. Their Devanagari integrity sits around 39.9% — meaning roughly 60% of
Devanagari aksharas are split mid-cluster. Ours: 96.0%.

### Against Gemma-2

Hindi fertility 1.96 vs. our 1.27; Marathi 3.20 vs. our 1.54. Gemma-2's Marathi parity of
2.26 means Marathi pays 2.26x more tokens than English for the same content under Gemma-2;
ours is 1.09.

### Against Sarvam-1 (the hardest comparison — the Indic-specific gold standard)

Sarvam-1 was purpose-built for Indic languages with a 68K vocabulary (4K larger than ours).
Our Variant D scores better on Hindi (1.27 vs. 1.40, −9.3%), better on Marathi (1.54 vs.
1.77, −13%), and better on English (1.23 vs. 1.43, −14%) — with a **smaller** vocabulary.
Devanagari integrity is also slightly higher (96.0% vs. 95.6%). The one metric where Sarvam-1
edges ahead is Hindi parity (1.14 vs. our 1.21) — Sarvam-1 is marginally fairer between Hindi
and English on that specific dimension. On raw compression efficiency — the thing that
directly saves context window and training compute — Variant D wins across the board.

### Why IndicBERTv2 is not a valid comparison

Its fertility numbers look strong (1.24/1.49) but its round-trip fidelity is only 0.1% — it's
a BERT WordPiece tokenizer that is **lossy by design** (it normalizes, lowercases, and
decomposes characters in ways that can't be fully recovered). It cannot be used for a
generative LM, since it corrupts training text before the model ever sees it.

## Proxy-LM (downstream learning) results

See [10-final-tokenizer-selection](../10-final-tokenizer-selection/README.md#proxy-lm-results-corrected-run)
for the bits-per-byte comparison across the three shortlisted candidates (A, C, D), which
confirmed the intrinsic ranking above by actually training models on each tokenizer.
