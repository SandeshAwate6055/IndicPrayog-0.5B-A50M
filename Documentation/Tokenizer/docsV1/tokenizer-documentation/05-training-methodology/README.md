# Training Methodology

The tokenizer training pipeline is a single notebook run in six ordered steps.

## Step 1 — Sample the corpus (streaming)

Each language is streamed from its HuggingFace source (see
[04-training-dataset](../04-training-dataset/README.md)) without ever downloading the full
dataset: `load_dataset(..., streaming=True)` reads one document at a time over HTTP and
writing stops the moment the byte target is hit. Each document is lightly cleaned (newlines
collapsed to spaces) and written as one line to `tokenizer_dataset/<lang>.txt`. Documents
under 150 characters are skipped as junk. The full sampling step takes roughly 10–25 minutes
depending on network speed, and is resume-friendly (a language is skipped if its file is
already ≥98% of its byte target).

## Step 2 — Train the tokenizer

Trained on all three language files together in one call, so that BPE/Unigram merge
statistics compete across languages according to the 40/35/25 byte balance. Key settings
(full detail in [03-architecture-and-configuration](../03-architecture-and-configuration/README.md)):

- NFC normalization (canonicalizes Devanagari matra encodings)
- Digit-splitting before byte-level pre-tokenization (`2024` → `2 0 2 4`)
- `add_prefix_space=False` and a custom Letter+Mark-aware split regex, to avoid the matra-
  splitting bug described in [03-architecture-and-configuration](../03-architecture-and-configuration/README.md)
- 64,000 target vocabulary size, `min_frequency=2`

Training is fully deterministic (same corpus → same tokenizer) and takes roughly 15–40
minutes on CPU for the ~1.5 GB corpus.

## Step 3 — Validate fertility on held-out text

Before anything downstream depends on the tokenizer, fertility (tokens ÷ words) is measured
against explicit pass/reject targets, using text the tokenizer never trained on (see
[04-training-dataset](../04-training-dataset/README.md#sampling-rules-enforced-by-the-pipeline)
for the held-out split method):

| Language | Target | Hard reject |
|:---|:---:|:---:|
| Hindi | ~1.8 | > 2.5 |
| Marathi | ~2.0 | > 2.8 |
| English | ~1.3 | > 1.8 |

A round-trip check (`decode(encode(x)) == x`) also runs at this stage as a lossless-ness
sanity check before proceeding.

## Step 4 — Save in HuggingFace format

The raw `tokenizer.json` is wrapped in `transformers.PreTrainedTokenizerFast` with
`pad_token`, `bos_token`, `eos_token`, `unk_token`, and `model_max_length=131072` set
explicitly, then saved with `save_pretrained()` so `AutoTokenizer.from_pretrained()` works
everywhere downstream (training loop, evaluation scripts, inference).

## Step 5 — Manual inspection ("playground")

Arbitrary text (including Hindi, Marathi, English, and Hindi–English code-switching) is run
through the tokenizer and the resulting token pieces are printed, to sanity-check that the
segmentation looks linguistically reasonable before trusting the automated metrics alone.

## Step 6 — Publish and clean up

Once fertility passes its gate, the tokenizer folder is optionally uploaded to a HuggingFace
dataset repo, and the ~1.5 GB sampled corpus is deleted — only the < 5 MB trained tokenizer
is kept going forward.

## Relationship to the broader experiment

The steps above describe training **one** tokenizer configuration end-to-end. The actual
pipeline runs this six times (once per variant, A–F) as a controlled ablation, then runs a
further model-training comparison stage on the top 3. That process is documented in
[06-experiments](../06-experiments/README.md).
