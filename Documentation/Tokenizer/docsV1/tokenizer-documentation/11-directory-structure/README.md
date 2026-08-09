# Directory Structure

## Tokenizer pipeline directory (as produced by the notebook)

```
IndicPrayog_tokenizer/
├── dataset/
│   ├── readme.md               ← dataset notes (consolidated into 04-training-dataset)
│   └── tokenizer_dataset/      ← sampled corpus (deleted after training)
│       ├── hi.txt              ← 600 MB
│       ├── en.txt              ← 525 MB
│       └── mr.txt              ← 375 MB
└── tokenizer/                  ← trained tokenizer output (kept forever, < 5 MB)
    ├── tokenizer.json          ← raw tokenizers-library file
    ├── tokenizer_config.json   ← HuggingFace PreTrainedTokenizerFast config
    └── special_tokens_map.json
```

- `dataset/tokenizer_dataset/` is safe to delete once the tokenizer has passed its fertility
  gate — see [05-training-methodology](../05-training-methodology/README.md#step-6--publish-and-clean-up).
- `tokenizer/` is the only artifact that needs to persist; it is what gets uploaded to
  HuggingFace and consumed everywhere downstream.

## This documentation directory

```
docs/
└── tokenizer-documentation/
    ├── 01-overview/README.md
    ├── 02-tokenizer-variants/README.md
    ├── 03-architecture-and-configuration/README.md
    ├── 04-training-dataset/README.md
    ├── 05-training-methodology/README.md
    ├── 06-experiments/README.md
    ├── 07-evaluation-methodology/README.md
    ├── 08-evaluation-datasets/README.md
    ├── 09-results/README.md
    ├── 10-final-tokenizer-selection/README.md
    ├── 11-directory-structure/README.md   ← this file
    ├── 12-how-to-run/README.md
    └── 13-source-files/README.md
```

Numbered folders reflect reading order: overview → design → data → method → experiments →
evaluation → results → selection → practical usage → source material. Each section is a
single `README.md` so links resolve cleanly on GitHub without extra navigation.
