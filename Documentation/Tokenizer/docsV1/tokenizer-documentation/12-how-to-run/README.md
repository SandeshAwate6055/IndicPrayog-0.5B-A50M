# How to Run / Use the Tokenizer

## Option A — Load the already-trained tokenizer

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("path/to/tokenizer")  # or the HF repo id once published

text = "भारत एक विशाल देश है।"
ids = tok(text).input_ids
print(ids)
print(tok.decode(ids))
```

The saved folder (`tokenizer/`) already contains `tokenizer.json`, `tokenizer_config.json`,
and the special-tokens map, so no extra setup is required — see
[11-directory-structure](../11-directory-structure/README.md).

## Option B — Reproduce training from scratch

The full pipeline is one notebook, run top to bottom.

### 1. Install dependencies

```bash
pip install -q datasets tokenizers transformers tqdm
```

### 2. Configure

Set the folder layout and the sampling plan (language, HF dataset id, config/split, byte
target). Defaults: 64,000 vocab size, minimum document length 150 characters. Full rationale
for the byte targets is in [04-training-dataset](../04-training-dataset/README.md).

### 3. Sample the corpus

Streams each language from its HuggingFace source until its byte target is hit, writing one
cleaned document per line to `dataset/tokenizer_dataset/<lang>.txt`. Resume-friendly — safe to
re-run if interrupted.

### 4. Train the tokenizer

Trains BPE (or Unigram, for the Variant D configuration) across all three language files
together, using the configuration in
[03-architecture-and-configuration](../03-architecture-and-configuration/README.md). Saves
`tokenizer/tokenizer.json`.

### 5. Validate fertility

Runs the quick sanity check (a handful of hand-picked sentences) followed by the proper
held-out fertility evaluation, and asserts every language passes its reject threshold before
allowing the notebook to continue. See
[05-training-methodology](../05-training-methodology/README.md#step-3--validate-fertility-on-held-out-text)
for the pass/reject targets.

### 6. Save in HuggingFace format

Wraps `tokenizer.json` in `PreTrainedTokenizerFast` and calls `save_pretrained(TOKENIZER_DIR)`.

### 7. Inspect (optional)

Run arbitrary text — including Hindi–English code-switched sentences — through the tokenizer
and print the resulting pieces, to visually confirm segmentation looks correct.

### 8. Publish and clean up (optional)

Upload the `tokenizer/` folder to a HuggingFace dataset repo, then delete
`dataset/tokenizer_dataset/` to free the ~1.5 GB sampled corpus — it isn't needed once the
tokenizer is trained and validated.

## Reproducing the full experiment (ablation + proxy-LM)

To reproduce the full comparison behind the chosen tokenizer (not just train one
configuration):

1. Repeat steps 3–5 above once per variant (A–F), changing only the one factor each variant
   tests — see [02-tokenizer-variants](../02-tokenizer-variants/README.md).
2. Apply the shortlisting gate to reduce to the top 3 candidates.
3. Train the 28M-parameter proxy-LM on each of the 3 shortlisted candidates using identical
   data, seed, and hyperparameters, and compare mean BPB.
4. Select the candidate with the lowest mean BPB, subject to the round-trip and integrity
   gates.

Full detail on each step is in [06-experiments](../06-experiments/README.md).
