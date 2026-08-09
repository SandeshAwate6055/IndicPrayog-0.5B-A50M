# Source Files

This documentation was consolidated from the following existing files in the repository.
Where this documentation summarizes or reorganizes their content, the originals remain the
source of truth for exact code — refer back to them for the runnable notebook cells.

| Source file (original) | What it contained | Consolidated into |
|:---|:---|:---|
| Tokenizer pipeline notebook (`IndicPrayog-0.5B Custom 64K Tokenizer Pipeline`) | The runnable pipeline: corpus sampling, BPE training, fertility validation, HuggingFace export, and an interactive playground cell | [03-architecture-and-configuration](../03-architecture-and-configuration/README.md), [05-training-methodology](../05-training-methodology/README.md), [12-how-to-run](../12-how-to-run/README.md) |
| Tokenizer experiment explainer (`IndicPrayog Tokenizer Experiment — Full Explainer`) | The full narrative: problem statement, 6-way ablation design, the 3 Devanagari-splitting bugs, evaluation metrics, results tables, the proxy-LM stage and its initialization bug, and the final variant selection | [01-overview](../01-overview/README.md), [02-tokenizer-variants](../02-tokenizer-variants/README.md), [06-experiments](../06-experiments/README.md), [07-evaluation-methodology](../07-evaluation-methodology/README.md), [09-results](../09-results/README.md), [10-final-tokenizer-selection](../10-final-tokenizer-selection/README.md) |
| Tokenizer dataset README (`Tokenizer Dataset — Everything About the Data`) | Exact corpus sizes, source datasets, sampling rules, and disk accounting | [04-training-dataset](../04-training-dataset/README.md) |

## Publication status

The trained tokenizer referenced throughout this documentation is published (privately) on
HuggingFace as `prashantcp8/IndicPrayog-tokenizer-64k`. Transfer to the AxisQuant
organization is pending write access at the time of writing — update this note once the
transfer completes.

## Maintaining this documentation

If the tokenizer is retrained, re-ablated, or the underlying source READMEs change:

1. Update the relevant numbered section(s) directly rather than the original scattered
   READMEs, so this folder remains the single source of truth going forward.
2. If a source file listed above is deleted or superseded, update this table.
3. Keep the root repository `README.md` link (see the main README's Documentation section)
   pointing at [01-overview](../01-overview/README.md) as the entry point.
