# Tokenizer Variants — The 6-Way Ablation

Rather than training one tokenizer and shipping it, six variants were trained on the
**identical 1.5 GB corpus**, changing exactly one design decision at a time (a controlled
ablation). Because the corpus, vocab size, and seed are held constant across variants, any
difference in the resulting metrics is caused by the one changed factor — not by data noise.

| ID | What changed | Hypothesis being tested |
|:---|:-------------|:------------------------|
| **A** | BPE, mix 40/35/25, ByteLevel + digit-split | Balanced baseline — is this good enough? |
| B | BPE, equal mix 33/33/34 | Does giving Marathi more data help it enough to justify weakening Hindi? |
| C | BPE, raw training mix 45/35/20 | What does "naive" mixing (just reuse the model's training ratio) cost Marathi? |
| **D** | **Unigram** (SentencePiece), mix 40/35/25 | Does Unigram beat BPE on morphologically rich languages, as the literature suggests? |
| E | BPE + `UnicodeScripts` boundary splitting | Does splitting at script boundaries (Devanagari vs Latin) help or hurt? |
| F | BPE, no digit-splitting | Does isolating digits into their own tokens matter? |

Bold IDs (**A**, **D**) are the ones referenced most in later documents: A is the balanced
BPE baseline, D is the eventual winner.

## Shortlisting gate

Before the expensive proxy-LM comparison stage, a pre-registered rule (written before seeing
any results, to avoid cherry-picking) reduced the 6 variants to 3:

**Rule:** Take the top 3 by mean FLORES fertility (Hindi + Marathi + English averaged),
subject to:

1. Round-trip fidelity = 100% (non-negotiable — a lossy tokenizer is disqualified outright)
2. Devanagari integrity ≥ Variant A's baseline (93.2%) — a variant cannot trade script
   quality for raw compression

**Result:**

- Variant B rejected — integrity 92.7% < 93.2% (equal mixing hurt Devanagari quality)
- Variant F rejected — integrity 93.15% < 93.2%, despite having the *best raw fertility*
  (1.22/1.51/1.21). This was a deliberate call: script quality matters more than raw
  compression alone.
- **Shortlist: A, C, D** — all three proceeded to the proxy-LM (actual language model
  training) comparison described in [06-experiments](../06-experiments/README.md).

See [09-results](../09-results/README.md) for the full metrics table across all 6 variants
plus external baselines, and [10-final-tokenizer-selection](../10-final-tokenizer-selection/README.md)
for why D was chosen as the final tokenizer.
