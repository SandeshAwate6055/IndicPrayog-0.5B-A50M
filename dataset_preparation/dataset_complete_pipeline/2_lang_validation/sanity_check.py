"""
Sanity check for Step 2: language validation.

Runs fastText lid.176 on one shard per source (not the full dataset) to
measure: throughput (docs/sec), mismatch rate against the folder-derived
language, and the confidence distribution. Used to size the full run and
pick a confidence threshold before processing everything.
"""

import time
from pathlib import Path

import fasttext
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_ROOT = (SCRIPT_DIR / "../../../../dataset").resolve()
MODEL_PATH = DATASET_ROOT / ".models" / "lid.176.bin"

SAMPLE_SHARDS = [
    ("sangraha", "hi", DATASET_ROOT / "standardized/hi/sangraha/data-0.parquet"),
    ("sangraha", "mr", DATASET_ROOT / "standardized/mr/sangraha/data-0.parquet"),
    ("wikipedia", "hi", DATASET_ROOT / "standardized/hi/wikipedia/train-00000-of-00002.parquet"),
    ("wikipedia", "mr", DATASET_ROOT / "standardized/mr/wikipedia/train-00000-of-00001.parquet"),
    ("wikipedia", "en", DATASET_ROOT / "standardized/en/wikipedia/train-00000-of-00041.parquet"),
    ("fineweb-edu", "en", DATASET_ROOT / "standardized/en/fineweb-edu/000_00000.parquet"),
]


def predict(model, text: str):
    first_chunk = text[:1000].replace("\n", " ").strip()
    if not first_chunk:
        return None, 0.0
    labels, probs = model.predict(first_chunk, k=1)
    lang = labels[0].replace("__label__", "")
    return lang, float(probs[0])


def main() -> None:
    model = fasttext.load_model(str(MODEL_PATH))

    for source, expected_lang, shard_path in SAMPLE_SHARDS:
        df = pd.read_parquet(shard_path, columns=["text", "language"])

        start = time.time()
        preds = [predict(model, t) for t in df["text"]]
        elapsed = time.time() - start

        pred_langs = [p[0] for p in preds]
        confidences = pd.Series([p[1] for p in preds])
        mismatch = sum(1 for p in pred_langs if p != expected_lang) / len(df)
        low_conf = (confidences < 0.70).mean()

        print(f"[{source}/{expected_lang}] {len(df)} rows in {elapsed:.1f}s "
              f"({len(df) / elapsed:.0f} docs/sec)")
        print(f"  mismatch rate: {mismatch:.2%}   low-confidence (<0.70) rate: {low_conf:.2%}")
        print(f"  confidence: min={confidences.min():.2f} mean={confidences.mean():.2f} "
              f"p25={confidences.quantile(0.25):.2f} p50={confidences.median():.2f}")
        print()


if __name__ == "__main__":
    main()
