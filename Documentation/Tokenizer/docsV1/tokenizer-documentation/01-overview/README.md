# Tokenizer Overview

## What this is

IndicPrayog uses a **custom 64,000-vocabulary byte-level BPE tokenizer** built specifically
for **Hindi, Marathi, and English**, to feed the IndicPrayog-0.5B model. It is not a
general-purpose or off-the-shelf tokenizer (GPT-2 / Mistral / Qwen / Gemma) — it was trained
from scratch on a balanced Indic + English corpus and validated against explicit quality
gates before being adopted for pretraining.

## The problem it solves

A tokenizer decides how raw text is chopped into integer IDs before the model ever sees a
character. Hindi and Marathi words are built from *aksharas* (syllables) — a base consonant
plus vowel marks (*matras*), e.g. `भ` + `ा` = `भा`. A tokenizer that splits aksharas mid-syllable
forces the model to learn from fragments that carry no independent meaning.

General-purpose tokenizers built for English (GPT-2, Mistral, Qwen2.5, Gemma-2) produce
**5–8 tokens per Hindi word** versus ~1.2–1.4 for English. That means, for the same content:

- Hindi consumes **5–8x more context window**
- Hindi costs **5–8x more compute** to train/run on
- Most vocabulary slots are wasted on byte-fragments that never form real words

## Goal

Build a 64K tokenizer for Hindi, Marathi, and English that is:

| Property | Meaning |
|---|---|
| Compact | Low fertility (tokens per word) |
| Fair | Low parity gap between languages (Hindi shouldn't cost far more tokens than English for the same meaning) |
| Lossless | 100% round-trip fidelity — `decode(encode(text)) == text` |
| Effective | Actually improves downstream language-model learning, not just intrinsic metrics |

## Languages and mix

| Language | Tokenizer-training share | Model-training share |
|---|---|---|
| Hindi | 40% | 45% |
| English | 35% | 35% |
| Marathi | 25% | 20% |

The tokenizer's training mix intentionally upweights Marathi relative to the model's
pretraining mix — see [05-training-methodology](../05-training-methodology/README.md) for why.

## Outcome (short version)

The final tokenizer (Variant **D**, Unigram/SentencePiece) beats every general-purpose
baseline by 3–6x on Hindi/Marathi fertility, and outperforms the closest Indic-specific
competitor (Sarvam-1) on fertility across all three languages despite a smaller vocabulary.
Full numbers: [09-results](../09-results/README.md) and
[10-final-tokenizer-selection](../10-final-tokenizer-selection/README.md).

## Where to go next

- [02-tokenizer-variants](../02-tokenizer-variants/README.md) — the 6 variants that were compared
- [03-architecture-and-configuration](../03-architecture-and-configuration/README.md) — exact tokenizer config and bug fixes
- [04-training-dataset](../04-training-dataset/README.md) — data sources and sizes
- [05-training-methodology](../05-training-methodology/README.md) — how the corpus was sampled and the tokenizer trained
- [06-experiments](../06-experiments/README.md) — the ablation study and proxy-LM experiment
- [07-evaluation-methodology](../07-evaluation-methodology/README.md) / [08-evaluation-datasets](../08-evaluation-datasets/README.md) — how quality was measured
- [09-results](../09-results/README.md) / [10-final-tokenizer-selection](../10-final-tokenizer-selection/README.md) — the numbers and the winner
- [11-directory-structure](../11-directory-structure/README.md) / [12-how-to-run](../12-how-to-run/README.md) — using the repo
- [13-source-files](../13-source-files/README.md) — original files this documentation was consolidated from
