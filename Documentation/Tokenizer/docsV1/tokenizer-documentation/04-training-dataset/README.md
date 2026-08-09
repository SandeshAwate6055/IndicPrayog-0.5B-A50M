# Training Dataset

## Sampled corpus (what the tokenizer actually trains on)

| Language | File | Size | Share | ~Docs |
|:---|:---|:---:|:---:|:---|
| Hindi | `tokenizer_dataset/hi.txt` | 600 MB | 40% | ~250–350K |
| English | `tokenizer_dataset/en.txt` | 525 MB | 35% | ~120–180K |
| Marathi | `tokenizer_dataset/mr.txt` | 375 MB | 25% | ~180–250K |
| **Total** | | **~1.5 GB** | 100% | ~550–780K |

Doc counts are approximate: sampling is done by **byte budget**, not document count, and
stops the instant each target is hit. Devanagari characters are 3 bytes in UTF-8, so
Hindi/Marathi files hold fewer *characters* than same-size English files but comparable
linguistic content.

## Source datasets

| Source | Config / Split | Full size on HF | Amount streamed |
|:---|:---|:---|:---|
| `ai4bharat/sangraha` | `verified` / `hin` | ~2.25 GB parquet (~1.05M docs) | 600 MB (~27%) |
| `ai4bharat/sangraha` | `verified` / `mar` | ~1.91 GB parquet (~874K docs) | 375 MB (~20%) |
| `HuggingFaceFW/fineweb-edu` | `sample-10BT` / train | ~27 GB parquet (9.67M docs) | 525 MB (~2%) |

Nothing is fully downloaded — all three sources are read with `streaming=True`, one document
at a time over HTTP, and writing stops once the byte target is reached. Peak disk usage is the
~1.5 GB sampled corpus itself.

## Why these sources

| Dataset | Why chosen |
|:---|:---|
| **Sangraha `verified`** | Highest-quality open Indic corpus, from IndicLLMSuite (AI4Bharat), ACL 2024 Outstanding Paper (arXiv:2403.06350). "Verified" means manually curated sources plus cleaning, flagging, and dedup. |
| **FineWeb-Edu** | Best open English data per token — CommonCrawl filtered by an educational-quality classifier, keeping only documents scoring ≥3/5 (arXiv:2406.17557). `sample-10BT` is a ready-made 10B-token slice. |

Rejected alternatives: CulturaX / mC4 / OSCAR (noisier for Indic languages), IndicCorp v2
(superseded by Sangraha, same lab), C4 (beaten by FineWeb-Edu on quality benchmarks).

> **API note:** Sangraha's HF configs are `verified` / `unverified` / `synthetic`, and the
> **language is the split** (`hin`, `mar`, `eng`) — not part of the config name. An earlier
> plan draft referenced `verified_hi`, which does not exist as a config; the pipeline uses the
> correct `config="verified", split="hin"` form.

## Why byte budgets, not document counts

A Devanagari character is 3 UTF-8 bytes; ASCII is 1 byte. Sampling by document count would
silently give English ~3x more raw information per byte than Hindi/Marathi. Sampling by byte
budget (600 / 525 / 375 MB) makes each language contribute proportionally regardless of script.

## Why 40/35/25 and not the model's training mix (45/35/20)

The model's pretraining corpus (12B tokens) uses Hindi 45% / English 35% / Marathi 20%. The
**tokenizer's** training corpus deliberately differs, upweighting Marathi to 25%.

Reason: BPE (and Unigram) merges/survival are frequency-driven. If Marathi were only 20% of
the tokenizer's training data, Hindi-specific subwords would crowd out Marathi ones, and
Marathi text would fragment into more tokens **permanently** — every future Marathi document
would cost more tokens for the model's entire lifetime, since the tokenizer is frozen after
training. Upweighting Marathi to 25% at tokenizer-training time protects its subword
inventory at close to zero cost to Hindi or English quality.

## Sampling rules enforced by the pipeline

| Rule | Value | Why |
|:---|:---|:---|
| Count by bytes, not docs | 600 / 525 / 375 MB | Marathi docs are shorter and Devanagari is 3 bytes/char — doc counts would silently unbalance the corpus |
| Minimum document length | 150 characters | Skips navigation junk and fragments |
| Newlines → spaces | one document = one line | The BPE/Unigram trainer reads line-per-document |
| Resume-friendly | skip a language's file if already ≥98% of its target | Re-running the sampling step never re-downloads |
| Held-out evaluation split | `dataset.skip(300_000)` then take 2,000 fresh documents | Fertility is measured on text the tokenizer never trained on |

## Disk accounting

| Item | Size | Keep after training? |
|:---|:---|:---|
| `tokenizer_dataset/*.txt` (sampled corpus) | ~1.5 GB | No — delete once the tokenizer passes its quality gate |
| `tokenizer/` (trained tokenizer output) | < 5 MB | Yes, indefinitely |
| HF streaming cache buffers | < 1 GB | Auto-managed by the `datasets` library |
| **Peak total** | **~2.5 GB** | (roughly 0.8% of a 300 GB disk budget) |
