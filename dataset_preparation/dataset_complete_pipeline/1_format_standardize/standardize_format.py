"""
Step 1: Standardize the format.

{
    "id": "unique_document_id",
    "text": "...",
    "language": "hi" | "mr" | "en",
    "source": "sangraha" | "wikipedia" | "fineweb-edu",
    "domain": "web" | "pdf" | "speech" | "encyclopedia",
    "url": "..." | null,
    "title": "..." | null,
    "metadata": "{}"   # JSON-encoded string (parquet can't store an all-empty struct)
}

"""

import json
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_ROOT = (SCRIPT_DIR / "../../../../dataset").resolve()
OUTPUT_ROOT = DATASET_ROOT / "standardized"

CANONICAL_COLUMNS = ["id", "text", "language", "source", "domain", "url", "title", "metadata"]


def write_canonical(df: pd.DataFrame, language: str, source: str, shard_name: str) -> Path:
    out_dir = OUTPUT_ROOT / language / source
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{shard_name}.parquet"
    df[CANONICAL_COLUMNS].to_parquet(out_path, index=False)
    return out_path


def standardize_sangraha(language: str) -> None:
    lang_dir_name = {"hi": "hin", "mr": "mar"}[language]
    src_dir = DATASET_ROOT / "sangraha" / "verified" / lang_dir_name
    shards = sorted(src_dir.glob("*.parquet"))
    print(f"[sangraha/{language}] {len(shards)} shard(s)")

    for shard in shards:
        df = pd.read_parquet(shard)
        canonical = pd.DataFrame({
            "id": "sangraha_" + language + "_" + df["doc_id"].astype(str),
            "text": df["text"],
            "language": language,
            "source": "sangraha",
            "domain": df["type"],
            "url": None,
            "title": None,
            "metadata": "{}",
        })
        out_path = write_canonical(canonical, language, "sangraha", shard.stem)
        print(f"  {shard.name} -> {out_path.relative_to(DATASET_ROOT)} ({len(canonical)} rows)")


def standardize_wikipedia(language: str) -> None:
    lang_dir_name = {"hi": "20231101.hi", "mr": "20231101.mr", "en": "20231101.en"}[language]
    src_dir = DATASET_ROOT / "wikipedia" / lang_dir_name
    shards = sorted(src_dir.glob("*.parquet"))
    print(f"[wikipedia/{language}] {len(shards)} shard(s)")

    for shard in shards:
        df = pd.read_parquet(shard)
        canonical = pd.DataFrame({
            "id": "wikipedia_" + language + "_" + df["id"].astype(str),
            "text": df["text"],
            "language": language,
            "source": "wikipedia",
            "domain": "encyclopedia",
            "url": df["url"],
            "title": df["title"],
            "metadata": "{}",
        })
        out_path = write_canonical(canonical, language, "wikipedia", shard.stem)
        print(f"  {shard.name} -> {out_path.relative_to(DATASET_ROOT)} ({len(canonical)} rows)")


def standardize_fineweb_edu() -> None:
    src_dir = DATASET_ROOT / "fineweb-edu" / "sample" / "10BT"
    shards = sorted(src_dir.glob("*.parquet"))
    print(f"[fineweb-edu/en] {len(shards)} shard(s)")

    for shard in shards:
        df = pd.read_parquet(shard)
        metadata = [
            json.dumps({"language_score": ls, "token_count": int(tc), "score": sc})
            for ls, tc, sc in zip(df["language_score"], df["token_count"], df["score"])
        ]
        canonical = pd.DataFrame({
            "id": "fineweb-edu_en_" + df["id"].astype(str),
            "text": df["text"],
            "language": "en",
            "source": "fineweb-edu",
            "domain": "web",
            "url": df["url"],
            "title": None,
            "metadata": metadata,
        })
        out_path = write_canonical(canonical, "en", "fineweb-edu", shard.stem)
        print(f"  {shard.name} -> {out_path.relative_to(DATASET_ROOT)} ({len(canonical)} rows)")


def main() -> None:
    standardize_sangraha("hi")
    standardize_sangraha("mr")
    standardize_wikipedia("hi")
    standardize_wikipedia("mr")
    standardize_wikipedia("en")
    standardize_fineweb_edu()
    print(f"\n Completet The Standardized output written under : dataset/standardized/")


if __name__ == "__main__":
    main()
