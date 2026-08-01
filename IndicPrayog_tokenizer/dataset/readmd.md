# Tokenizer Dataset — Everything About the Data

> All dataset-related information for the tokenizer corpus lives here: sources, exact sizes,
> what we download vs stream, and what lands on disk.

---

## 1. How Much Data? — The Numbers

### What we actually use (sampled corpus)

| Language | File (created by notebook)   | Size       | Share | ~Docs     |
|:---------|:------------------------------|:-----------|:-----:|:----------|
| Hindi    | `tokenizer_dataset/hi.txt`    | **600 MB** | 40%   | ~250–350K |
| English  | `tokenizer_dataset/en.txt`    | **525 MB** | 35%   | ~120–180K |
| Marathi  | `tokenizer_dataset/mr.txt`    | **375 MB** | 25%   | ~180–250K |
| **Total**|                               | **~1.5 GB**| 100%  | ~550–780K |

> Doc counts are approximate — the notebook samples by **bytes**, not docs, and stops the
> moment each byte target is hit. Devanagari chars are 3 UTF-8 bytes, so Hindi/Marathi files
> hold fewer *characters* than same-size English files but similar *linguistic* content.

### Source datasets (full size on HF — we only stream a slice)

| Source                        | Config / Split       | Full size on HF | We stream    |
|:------------------------------|:---------------------|:----------------|:-------------|
| `ai4bharat/sangraha`          | `verified` / `hin`   | ~2.25 GB parquet (~1.05M docs) | 600 MB (~27%) |
| `ai4bharat/sangraha`          | `verified` / `mar`   | ~1.91 GB parquet (~874K docs)  | 375 MB (~20%) |
| `HuggingFaceFW/fineweb-edu`   | `sample-10BT` train  | ~27 GB parquet (9.67M docs)    | 525 MB (~2%)  |

**Nothing is fully downloaded.** The notebook uses `streaming=True` — docs are read one at a
time over HTTP and writing stops at the byte target. Peak disk = the 1.5 GB corpus itself.

### Disk accounting

| Item                          | Size     | Keep?                          |
|:------------------------------|:---------|:-------------------------------|
| `tokenizer_dataset/*.txt`     | ~1.5 GB  | Delete after tokenizer trained |
| `../tokenizer/` (output)      | < 5 MB   | Keep forever                   |
| HF cache (streaming buffers)  | < 1 GB   | Auto-managed                   |
| **Peak total**                | **~2.5 GB** | (of 300 GB budget — 0.8%)   |

---

## 2. Why These Sources? (Research Summary)

| Dataset       | Why chosen                                                                 |
|:--------------|:----------------------------------------------------------------------------|
| **Sangraha `verified`** | Highest-quality open Indic corpus. From IndicLLMSuite (AI4Bharat), **ACL 2024 Outstanding Paper** (arXiv:2403.06350). "Verified" = manually curated sources + cleaning + flagging + dedup. |
| **FineWeb-Edu** | Best open English data per token. CommonCrawl filtered by an educational-quality classifier — only docs scoring ≥3/5 kept (arXiv:2406.17557). `sample-10BT` is a ready-made 10B-token cut. |

Rejected: CulturaX / mC4 / OSCAR (noisier for Indic), IndicCorp v2 (superseded by Sangraha,
same lab), C4 (beaten by FineWeb-Edu on all quality benchmarks).

> ⚠️ **API structure note:** Sangraha's HF configs are `verified` / `unverified` / `synthetic`,
> and the **language is the split** (`hin`, `mar`, `eng`). Older plan drafts said `verified_hi` —
> that config name does not exist. The notebook uses the correct form.

---

## 3. Why 40 / 35 / 25 and Not the Training Mix (45 / 35 / 20)?

The **training** corpus (12B tokens) uses hi 45% / en 35% / mr 20%.
The **tokenizer** corpus deliberately differs: hi 40% / en 35% / **mr 25%**.

Reason: BPE merges are won by frequency. If Marathi is only 20% of the tokenizer corpus,
Hindi-specific merges crowd out Marathi subwords, and Marathi text fragments into more tokens
**forever** (bad fertility — every Marathi document costs more tokens for the model's whole
life). Upweighting Marathi to 25% at tokenizer time protects its subword inventory at nearly
zero cost to Hindi/English.

---

## 4. Sampling Rules (what the notebook enforces)

| Rule                        | Value                | Why                                        |
|:-----------------------------|:---------------------|:--------------------------------------------|
| Count by **bytes**, not docs | 600/525/375 MB       | Marathi docs shorter; Devanagari 3 B/char — doc counts would silently unbalance |
| Min doc length               | 150 chars            | Skip navigation junk / fragments             |
| Newlines → spaces            | one doc = one line   | BPE trainer reads line-per-doc               |
| Resume-friendly              | skip file if ≥98% target | Re-running the cell never re-downloads   |
| Held-out eval                | `ds.skip(300_000)` then 2,000 fresh docs | Fertility is measured on unseen text, never training text |

---

## 5. What Lands Where

```
IndicPrayog_tokenizer/
├── dataset/
│   ├── readmd.md              ← this file
│   └── tokenizer_dataset/     ← hi.txt (600 MB), en.txt (525 MB), mr.txt (375 MB)
└── tokenizer/                 ← trained 64K tokenizer (output, <5 MB)
```

After the tokenizer passes the fertility gate and is uploaded to HF, the entire
`tokenizer_dataset/` folder is safe to delete — the tokenizer itself is all that matters.
