# Evaluation Datasets

Three distinct held-out sets are used across the pipeline, deliberately kept separate from
the training corpus so that quality numbers reflect generalization, not memorization.

## 1. FLORES-200 devtest — intrinsic metrics (fertility, parity, round-trip, integrity, vocab utilization, compression)

1,012 professionally translated, parallel sentences per language across Hindi, Marathi, and
English. This is a standard public benchmark used community-wide to compare multilingual
tokenizers (it is the benchmark behind Petrov et al. 2023, the canonical tokenizer-fairness
paper). Being parallel sentences (identical meaning across languages) is what makes the
parity metric meaningful — the same sentence's token count can be directly compared across
languages. Because it's a curated benchmark set rather than scraped web text, it was never
part of the training corpus, ruling out contamination.

## 2. Streamed held-out documents — fertility validation during tokenizer training

Used in [05-training-methodology](../05-training-methodology/README.md) Step 3, before FLORES
evaluation even runs, as a first quick pass/fail gate. For each language, the source dataset
stream is advanced past `dataset.skip(300_000)` documents (well past what was consumed during
corpus sampling — see [04-training-dataset](../04-training-dataset/README.md)) and 2,000 fresh
documents are taken from there. Because this reuses the exact skip point past the training
sample, none of these documents overlap with the training corpus.

## 3. Proxy-LM training/eval split — BPB comparison stage

For the model-training comparison in [06-experiments](../06-experiments/README.md#stage-2--proxy-lm-comparison):

- **Training split:** the beginning of the pool file, identical across all three candidate
  runs (A, C, D) — only the tokenization of this same text differs between runs.
- **Held-out eval split:** the last 500 documents per language from the same pool file,
  also identical across all three runs.

Keeping both splits identical across candidates isolates the tokenizer as the only variable,
so any BPB difference between runs is caused by the tokenizer, not by the models seeing
different underlying text.
