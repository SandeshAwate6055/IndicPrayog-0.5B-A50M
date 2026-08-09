# Evaluation Methodology

Every tokenizer candidate — 6 of ours plus 8 external baselines — was run through the same
evaluation code on the same benchmark sentences, across six intrinsic metrics, followed by a
proxy-LM comparison for the top 3 shortlisted candidates.

## The six intrinsic metrics

### 1. Fertility (tokens per word) — lower is better

How many tokens it takes to encode one word on average. Lower means more compression, a
longer effective context window, and cheaper training/inference. Measured on the FLORES-200
devtest set (see [08-evaluation-datasets](../08-evaluation-datasets/README.md)).

### 2. Parity score — closer to 1.0 is fairer

On the same parallel FLORES sentence (identical meaning, translated into each language), how
many more tokens does Hindi (or Marathi) require versus English? A parity of 4.5 means that
language pays 4.5x more tokens for the same thought. This is the "tokenizer fairness" metric
from Petrov et al. 2023.

### 3. Round-trip fidelity — must be 100%

Checks `decode(encode(text)) == text` for every evaluation sentence. Anything under 100%
means the tokenizer is lossy — it silently corrupts text, which disqualifies it regardless of
how good its other metrics look (see IndicBERTv2 in [09-results](../09-results/README.md)).

### 4. Devanagari integrity — higher is better

The percentage of Devanagari grapheme clusters (aksharas, including conjuncts like `क्ष`) that
survive as a contiguous span inside a single token, i.e. are never split mid-akshara. This
metric was added specifically for this project — most public tokenizer benchmarks don't
include it — because it directly measures script-level quality for Devanagari rather than
just raw compression.

### 5. Vocabulary utilization — higher means less waste

What fraction of the 64,000 vocabulary slots are actually used when encoding real text. Low
utilization means many slots are wasted on rare fragments that rarely or never appear.

### 6. Compression (bytes per token) — higher is better

The average number of raw bytes each token represents. A higher number means each token
carries more signal per unit of sequence length.

## Proxy-LM evaluation (for the top 3 shortlisted candidates only)

Intrinsic metrics measure compression efficiency but don't directly answer whether a
lower-fertility tokenizer helps a model *learn* better. For that, an actual small language
model was trained from scratch on each shortlisted tokenizer and compared using **bits per
byte (BPB)** rather than loss-per-token, since BPB normalizes away the confound of different
tokenizers producing different token counts for the same text. Full methodology, model
architecture, and the initialization bug that was caught and fixed are documented in
[06-experiments](../06-experiments/README.md#stage-2--proxy-lm-comparison).

## Evaluation datasets used

See [08-evaluation-datasets](../08-evaluation-datasets/README.md) for exactly which datasets
and splits back each metric above.
