# Tokenizer Architecture & Configuration

## Base configuration (all variants)

| Setting | Value | Why |
|---|---|---|
| Algorithm | Byte-level BPE (baseline; Variant D uses Unigram instead — see below) | Standard, fast (Rust `tokenizers` trainer), works well with byte fallback |
| Vocabulary size | 64,000 | Matches the model's `vocab_size` config |
| Normalizer | NFC | Devanagari vowel marks (matras) have multiple valid Unicode encodings — NFC collapses them to one canonical form so the same word always tokenizes identically |
| Pre-tokenizer | `Sequence([Digits(individual_digits=True), ByteLevel(add_prefix_space=False)])` | Splits digits into individual characters (`2024` → `2 0 2 4`, which models handle far better than a single opaque number token) and falls back to raw bytes for any Unicode character, so `<unk>` almost never fires |
| Decoder | `ByteLevel` decoder | Inverse of the byte-level pre-tokenizer |
| `min_frequency` | 2 | A merge must occur at least twice in the corpus to be learned |
| `initial_alphabet` | `ByteLevel.alphabet()` | Guarantees all 256 raw bytes are present in the vocab, so any input is representable |
| Special tokens | `<pad>`, `<bos>`, `<eos>`, `<unk>` (ids 0–3, fixed by declaration order) | Standard set required by the training/inference pipeline |
| `model_max_length` | 131,072 | Matches the model's `max_position_embeddings` |

Training data is fed as three per-language text files (`hi.txt`, `en.txt`, `mr.txt`); since
BPE merges are frequency-based, the byte-budget balance between the three files (40/35/25,
see [04-training-dataset](../04-training-dataset/README.md)) directly controls whose subwords
win the merge competition.

## Variant D (winner): Unigram instead of BPE

Variant D replaces byte-level BPE with the **Unigram** (SentencePiece-style) algorithm:

- **BPE** is greedy bottom-up: start from bytes, repeatedly merge the most frequent adjacent
  pair, always taking the locally optimal merge.
- **Unigram** is probabilistic top-down: start from a large candidate vocabulary and remove
  tokens whose removal increases total corpus loss the least, i.e. optimize a global unigram
  language-model loss rather than local co-occurrence counts. Each tokenization is the
  maximum-likelihood segmentation under that model.

For morphologically rich languages like Hindi and Marathi, Unigram tends to keep
linguistically meaningful subwords (e.g. the conjunct `प्र`) intact more consistently, because
survival depends on frequency across *all* corpus positions, not just one frequent bigram.
Unigram's `Metaspace` pre-tokenizer (using `▁` for whitespace) also handles leading spaces
natively, avoiding a workaround BPE needed (see bugs below).

## Three bugs found and fixed (config-level, not just training-level)

These were discovered while building the pipeline and are important because they silently
degrade Devanagari quality even though nothing throws an error. Each is disclosed here rather
than only in commit history, since most Indic tokenizers built with default library settings
likely carry the same bugs.

### Bug 1 — `add_prefix_space=True` injects spurious spaces

The library default, `ByteLevel(add_prefix_space=True)`, inserts a leading-space marker
(`Ġ`) at **every** upstream pre-tokenizer split boundary, not only at the start of a string.
Example: `"पास4महीने"` (a digit embedded in Hindi text, no space in the source) became
`['Ġपास', 'Ġ4', 'Ġमहीने']` — a space was fabricated before the digit. This both breaks
round-trip fidelity (`decode(encode(x)) != x`) and inflates fertility.

**Fix:** `ByteLevel(add_prefix_space=False)`.

### Bug 2 — `UnicodeScripts()` silently drops whitespace (Variant E)

Running `UnicodeScripts()` (splits at script boundaries) *after* `Digits()` means a lone space
between a digit and the next word belongs to no script, so `UnicodeScripts()` silently deletes
it — no error, just a missing character. `"पास 4 महीने"` was pre-tokenized as if it were
`"पास4महीने"`.

**Fix:** Run `UnicodeScripts()` **before** `Digits()`, so it only ever sees text where spaces
are still attached to a script-bearing run.

### Bug 3 — ByteLevel's default word regex splits Devanagari at every matra (the big one)

`ByteLevel(use_regex=True)`, the default, groups Unicode category **Letter (L)** characters
into pretoken "words" before byte-encoding. Devanagari vowel marks (matras) — `ा ि ी े ो ौ ं ँ ्`
— are Unicode category **Mark (Mc/Mn)**, not Letter. Consequence: `भारत` (4 characters, 1
akshara cluster) was pre-split into `['भ', 'ा', 'रत']`. Since BPE can only merge bytes
*within* a single pretoken, it could never merge `भ` + `ा` into `भा` — no matter how large the
corpus or vocabulary, this was structurally impossible to fix by training longer.

This was the root cause of Hindi/Marathi fertility of 3.3–5 tokens/word (vs. the ~1.2–1.4
target) and Devanagari integrity of only 40% in early runs.

**Fix:** Replace ByteLevel's internal regex with a custom `Split` using
`r" ?[^\s\d]+| ?\d+|\s+"` — this groups Letter *and* Mark characters together (`[^\s\d]`
matches anything that isn't whitespace or a digit, including Marks), then pass
`ByteLevel(use_regex=False)` to byte-encode without re-splitting. The leading `" ?"` in the
regex follows GPT-2's convention of attaching a leading space to the following word as one
pretoken; omitting it turns spaces into isolated, unmergeable pretokens, which nearly doubled
English fertility when first tested.

### A fourth bug, found during the proxy-LM stage

A model-initialization bug (not a tokenizer bug, but part of the same validation pipeline) is
documented in [06-experiments](../06-experiments/README.md#the-initialization-bug), since it
affected how the tokenizer candidates were compared.

## Output format

The trained tokenizer is saved as raw `tokenizer.json`, then wrapped in
`transformers.PreTrainedTokenizerFast` with `pad_token`, `eos_token`, `bos_token`,
`unk_token` set explicitly, so `AutoTokenizer.from_pretrained()` works in training, evaluation,
and inference code without extra glue.
