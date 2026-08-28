# Dataset Preparation

## Folder structure

```
dataset_preparation/
├── dataset_lookup/                 # notebooks to explore the raw downloaded data (per language)
│   ├── hindi_overview.ipynb
│   ├── marathi_overview.ipynb
│   └── english_overview.ipynb
└── dataset_complete_pipeline/      # the actual processing pipeline, one folder per step
    ├── 1_format_standardize/       # Step 1 : done
    ├── 2_lang_validation/          # Step 2 : done
    ├── 3_train_and_val_split/      # Step 3 : done
    ├── 4_tokenization/             # Step 4 : done
    └── 5_final_dataset/            # Step 5 : done
```

## Steps done so far

### Step 1: Format standardization (`1_format_standardize/`)

**Why:** Sangraha, Wikipedia, and FineWeb-Edu each ship with a different schema (different column names, some have `title`/`url`, some don't, some tag document type, some don't). Every later step (text extraction, quality filtering, dedup, tokenization) needs one predictable shape to work against, otherwise each step would need custom logic per source. So Step 1 rewrites every source into a single canonical schema before anything else touches the data.

**Canonical schema:**

```
id, text, language, source, domain, url, title, metadata
```

| field    | meaning                                             |
|----------|------------------------------------------------------|
| id       | unique document id, prefixed by source and language |
| text     | the raw document text                               |
| language | `hi`, `mr`, or `en`                                 |
| source   | `sangraha`, `wikipedia`, or `fineweb-edu`           |
| domain   | `web`, `pdf`, `speech`, or `encyclopedia`           |
| url      | source URL if the dataset has one, else null        |
| title    | document title if the dataset has one, else null    |
| metadata | JSON string of anything source specific (score, token count, etc.) |

**Example (Sangraha, raw to canonical):**

Raw Sangraha row:
```
{"doc_id": "c729b49b...", "text": "अमेरिकी राष्ट्रपति चुनाव...", "type": "web"}
```

Becomes:
```
{
  "id": "sangraha_hi_c729b49b...",
  "text": "अमेरिकी राष्ट्रपति चुनाव...",
  "language": "hi",
  "source": "sangraha",
  "domain": "web",
  "url": null,
  "title": null,
  "metadata": "{}"
}
```

**How to run:**

```bash
cd dataset_preparation/dataset_complete_pipeline/1_format_standardize
python standardize_format.py
```

This reads every downloaded shard from the sibling `dataset/` folder and writes the canonical parquet files to `dataset/standardized/<language>/<source>/`.

A demo notebook (`example.ipynb`) walks through the raw to canonical conversion with real rows from all three sources, for learning.

### Step 2: Language validation (`2_lang_validation/`)

**Why:** our folder names (`sangraha/verified/hin/`, `sangraha/verified/mar/`, etc.) tell us what language a shard is supposed to contain, not what it actually contains. This matters most for Hindi and Marathi since both use the Devanagari script, so a simple Unicode script check cannot tell them apart. A row that is actually Marathi could be sitting in the Hindi folder, or the other way around, and we would never know unless we check the text itself. So Step 2 runs every row through a real language identification model and tags it, without dropping anything yet.

**Model:** fastText `lid.176` (176 languages, downloaded once to `dataset/.models/lid.176.bin`). Only the first 1000 characters of each row are checked, that is enough signal for language identification.

**Columns added to every row:**

| field | meaning |
|-------|---------|
| predicted_language | language the model actually detected |
| language_confidence | model's confidence score, 0 to 1 |

**Example:**

A row already tagged `language: "hi"` in Step 1 gets checked again in Step 2:

```
{
  "language": "hi",
  "predicted_language": "hi",
  "language_confidence": 0.97
}
```

If `predicted_language` did not match `language`, or `language_confidence` was low, that row is suspect and can be reviewed or filtered in a later step. A sanity check on one shard per source measured mismatch rates under 1% and mean confidence above 0.94 for every language and source, so this was not a major problem, just something worth tagging and keeping an eye on.

**How to run:**

```bash
cd dataset_preparation/dataset_complete_pipeline/2_lang_validation
jupyter nbconvert --to notebook --execute --inplace lang_validation.ipynb --ExecutePreprocessor.timeout=-1
```

This reads every shard from `dataset/standardized/<language>/<source>/`, adds `predicted_language` and `language_confidence` to every row, and writes the result to `dataset/lang_validated/<language>/<source>/`. A summary of the mismatch and low confidence rate per shard is saved to `dataset/lang_validated/validation_summary.csv`. Takes roughly 25 to 30 minutes for the full dataset.

### Step 3: Subsample to token budget (`3_train_and_val_split/`)

**Why:** Step 4's full dataset run measured 18.41 billion tokens total. We only want a **10 billion token** training set, at a fixed target mix of **Hindi 40%, English 35%, Marathi 25%** (matching the tokenizer's own training mix), not just a proportional shrink of today's mix.

**How the drop is decided:** within each language there is a small curated source (`wikipedia`) and a large source (`sangraha` or `fineweb-edu`). For each language, the small curated source is always kept in full, and only the large source is trimmed down to whatever fits in the remaining per language token budget. This means no Wikipedia row is ever dropped, and all of the cut comes out of the largest, most redundant source.

**Result of the real run:**

| language | source | keep_ratio | rows kept | rows dropped | tokens kept (est.) |
|----------|--------|-----------|-----------|---------------|---------------------|
| hi | sangraha | 0.408 | 7,102,415 | 10,318,517 | 3.93B |
| hi | wikipedia | 1.000 | 163,093 | 0 | 0.07B |
| en | fineweb-edu | 0.543 | 2,788,635 | 2,342,365 | 2.80B |
| en | wikipedia | 1.000 | 781,445 | 0 | 0.70B |
| mr | sangraha | 0.877 | 5,144,381 | 721,236 | 2.48B |
| mr | wikipedia | 1.000 | 94,133 | 0 | 0.02B |
| **Total** | | | **16,074,090** | **13,382,130** | **~10.00B** |

**How to run:**

```bash
cd dataset_preparation/dataset_complete_pipeline/3_train_and_val_split
jupyter nbconvert --to notebook --execute --inplace subsample_to_budget.ipynb --ExecutePreprocessor.timeout=-1
```

Reads Step 4's exact per shard token counts, works out a keep ratio per (language, source), then randomly samples that fraction of rows (seeded, reproducible) from every shard in `dataset/standardized/` and writes the kept rows to `dataset/sampled_10b/<language>/<source>/`, plus a `sampling_summary.csv`. Takes roughly 6 minutes. Does not modify `dataset/standardized/`.

### Step 4: Tokenizer evaluation (`4_tokenization/`)

**Why:** we trained our own tokenizer (`AxisQuant/IndicPrayog-tokenizer-64k`, 64,000 pieces) for English, Hindi, and Marathi. Before finalizing the data mixture, we need two things: (1) how efficiently the tokenizer encodes each language and category of text, and (2) the real total token count of the entire dataset, since that number is what actually decides the training token budget for the 0.5B model.

**What it measures:** a small evaluation corpus covering English, Hindi, Marathi, and tricky categories (code, numbers, URLs, punctuation, names, mixed language text), scored on `chars_per_token` (compression ratio, higher is more efficient) and `tokens_per_word`. Separately, every row of every one of the 149 standardized shards is tokenized once to get exact totals, not a sample estimate.

**Sample based efficiency (200K characters per language):**

| category | chars_per_token | tokens_per_word |
|----------|------------------|------------------|
| numbers | 2.08 | 3.79 |
| code | 2.13 | 4.81 |
| punctuation | 2.89 | 2.03 |
| urls | 2.94 | 17.60 |
| names | 3.60 | 2.05 |
| mixed_language | 3.80 | 1.72 |
| hindi | 3.99 | 1.29 |
| marathi | 4.44 | 1.55 |
| english | 4.78 | 1.29 |

Hindi and Marathi need noticeably more tokens per character than English (lower `chars_per_token`), so the data mixture and token budget must account for that instead of assuming a flat per document cost across languages.

**Full dataset token count (real run over all 149 shards, not a sample):**

| language | rows | characters | tokens | chars_per_token |
|----------|------|------------|--------|------------------|
| en (English) | 5.91 million | 27.05 billion | 5.85 billion | 4.62 |
| hi (Hindi) | 17.58 million | 38.51 billion | 9.71 billion | 3.96 |
| mr (Marathi) | 5.96 million | 12.82 billion | 2.85 billion | 4.50 |
| **Total** | **29.46 million** | **78.37 billion** | **18.41 billion** | |

So the full dataset gives roughly **18.41 billion training tokens** once tokenized. Hindi is more than half of all rows but not half of all tokens, since Hindi text compresses into fewer characters per token than English but each Hindi document is on average shorter, the two effects roughly balance out.

**How to run:**

The notebook is meant to be run by whoever needs a fresh report, it is not part of the automated pipeline.

```bash
cd dataset_preparation/dataset_complete_pipeline/4_tokenization
jupyter nbconvert --to notebook --execute --inplace tokenizer_eval.ipynb --ExecutePreprocessor.timeout=-1
```

The sample based comparison finishes in seconds and saves `tokenizer_eval_report.csv`. The full dataset token count tokenizes all 29.46 million rows, this runs on CPU only (the tokenizer has no GPU backend) and takes roughly 2 hours, saving `full_dataset_token_counts.csv`. See `undersatnd.md` in the same folder for a full walkthrough of how the tokenizer actually works, step by step, with real examples.

### Step 5: Tokenize and pack the final dataset (`5_final_dataset/`)

**Why:** Step 3 produced `dataset/sampled_10b/`, a 10B token, Hindi 40 / English 35 / Marathi 25 mix, but it is still raw text. Nothing before this step produces actual `input_ids` a model can train on. This step holds out a validation set per language, tokenizes everything, and packs it into the fixed length rows training actually consumes.

**What it does:**

1. Holds out **0.5%** of rows per language as validation, before any tokenization, so the model never sees them.
2. Tokenizes everything with the frozen `AxisQuant/IndicPrayog-tokenizer-64k` tokenizer.
3. Packs the token ids into fixed length rows of **2048 tokens** (the model's context length), inserting an `<eos>` id between documents.
4. Writes flat binary `.bin` files (`uint16`, since vocab size 64,000 fits in 16 bits): `train.bin` (all 3 languages, shard order shuffled so training batches are not one language at a time) and one `val_<language>.bin` per language, so per language perplexity can be measured later without re tokenizing anything.

**Known gap, recorded on purpose:** this run skips quality filtering and MinHash dedup, see the deviation note in `row_not_push_material/final_plan.md`. Whatever duplicate or low quality documents exist in `sampled_10b/` are packed as is.

**Result of the real run:**

| file | rows | tokens | tail tokens dropped |
|------|------|--------|-----------------------|
| train.bin | 4,863,624 | 9,960,701,952 | 597 |
| val_en.bin | 8,601 | 17,614,848 | 1,189 |
| val_hi.bin | 9,890 | 20,254,720 | 1,672 |
| val_mr.bin | 6,224 | 12,746,752 | 1,244 |
| **Total** | **4,888,339** | **10,011,318,272** | **4,702** |

Total tokens land at ~10.01B, within 0.1% of the 10B target. Tail tokens dropped are just the last incomplete buffer in each file (always under 2048), negligible. `dataset/final_packed/{train.bin, val_en.bin, val_hi.bin, val_mr.bin}` are the final, real model inputs, ready for training.

**How to run:**

```bash
cd dataset_preparation/dataset_complete_pipeline/5_final_dataset
jupyter nbconvert --to notebook --execute --inplace pack_dataset.ipynb --ExecutePreprocessor.timeout=-1
```

Reads every shard from `dataset/sampled_10b/`, writes `train.bin`, `val_en.bin`, `val_hi.bin`, `val_mr.bin`, and a `final_pack_report.csv` (rows and tokens actually written per split) to `dataset/final_packed/`. Took roughly 65 minutes on CPU for the real run above.

`final_lookup.ipynb` in the same folder decodes a few random rows from each `.bin` file back to text, useful to sanity check the packed dataset before training.
