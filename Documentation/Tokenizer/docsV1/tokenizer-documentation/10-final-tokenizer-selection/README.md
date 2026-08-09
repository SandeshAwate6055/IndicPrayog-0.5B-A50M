# Final Tokenizer Selection

## Proxy-LM results (corrected run)

After fixing the initialization bug described in
[06-experiments](../06-experiments/README.md#the-initialization-bug), the three shortlisted
candidates (A, C, D) were compared on bits-per-byte (BPB) — lower is better:

| Variant | Mean BPB | Hindi BPB | English BPB | Marathi BPB |
|:---|:---:|:---:|:---:|:---:|
| **D (winner)** | **1.0404** | **0.7229** | **1.6664** | **0.7320** |
| A | 1.0671 | 0.7324 | 1.7201 | 0.7489 |
| C | 1.0686 | 0.7316 | 1.7226 | 0.7516 |

Variant D wins on every language individually — not as a tradeoff (e.g. better at Hindi at
English's expense), but genuinely better across all three. D's advantage was 2.5% on mean BPB
over A and 2.7% over C. This is the comparison that selected the winner, per the project's
pre-registered selection rule; the ranking agreed with the intrinsic fertility-based ranking
from [09-results](../09-results/README.md), so no override was needed.

## Why Variant D (Unigram) wins

The architectural reasons Unigram tends to outperform BPE on Hindi/Marathi are covered in
[03-architecture-and-configuration](../03-architecture-and-configuration/README.md#variant-d-winner-unigram-instead-of-bpe).
In summary: Unigram optimizes a global probability model over the whole corpus rather than
greedily merging the locally most frequent pair, so linguistically meaningful subwords (like
the conjunct `प्र`) tend to survive more consistently — reflected in D's higher Devanagari
integrity (96.0% vs. 93.2–93.3% for the BPE variants).

## Why D is the right choice for IndicPrayog-0.5B specifically

IndicPrayog-0.5B trains primarily on Hindi (45%), with English (35%) and Marathi (20%).
Variant D's profile matches this:

- **Best Devanagari integrity (96.0%)** — aksharas stay whole, keeping Hindi/Marathi training
  signal maximally coherent
- **Competitive Hindi/Marathi fertility (1.27 / 1.54)** — within 2–3% of the best BPE variants
  on fertility, while being measurably better on downstream BPB
- **Slightly lower English fertility (1.23)** than the BPE variants
- **Confirmed by the proxy-LM** — lower BPB means that, for the same compute budget, a model
  trained with D's tokenization predicts Hindi, Marathi, *and* English text better than the
  same model trained with A's or C's tokenization

The tokenizer is a **permanent bottleneck**: once IndicPrayog-0.5B is pretrained, the
tokenizer is frozen, and changing it later means retraining from scratch. Choosing correctly
now, backed by proxy-LM evidence rather than intrinsic metrics alone, was worth the extra
ablation cost.

## Summary

| Stage | Method | Winner | Key finding |
|:---|:---|:---|:---|
| Corpus sampling | Byte-budget streaming | — | Prevents silent English over-representation |
| Ablation training | 6 variants, controlled | — | 3 bugs found that were silently killing Devanagari quality |
| Intrinsic eval | FLORES-200, 6 metrics | A, C, D shortlisted | All our variants beat non-Indic baselines by 3–6x on hi/mr fertility |
| Proxy-LM BPB | 28M-parameter LM × 3 candidates | **D (Unigram)** | D wins on hi, en, mr individually; confirms the intrinsic ranking |
| Published | HF (private) | `prashantcp8/IndicPrayog-tokenizer-64k` | Transfer to AxisQuant org pending write access |
