import os, json, time

MB = 1024 * 1024
POOL_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "pools")
os.makedirs(POOL_DIR, exist_ok=True)
MIN_DOC_LEN = 150

POOL_PLAN = [
    ("hi", "ai4bharat/sangraha",          "verified",    "hin",   800 * MB),
    ("en", "HuggingFaceFW/fineweb-edu",   "sample-10BT", "train", 650 * MB),
    ("mr", "ai4bharat/sangraha",          "verified",    "mar",   650 * MB),
]


def sample_pool(lang, dataset_id, config, split, target_bytes):
    from datasets import load_dataset

    out_path  = os.path.join(POOL_DIR, f"{lang}.txt")
    meta_path = os.path.join(POOL_DIR, f"{lang}_meta.json")

    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        if meta.get("bytes_written", 0) >= target_bytes * 0.98:
            print(f"[{lang}] already complete ({meta['bytes_written']/MB:.0f} MB) — skipping")
            return

    print(f"[{lang}] streaming {dataset_id}/{config}/{split} -> {target_bytes/MB:.0f} MB")
    ds = load_dataset(dataset_id, config, split=split, streaming=True)

    written, n_docs, t0 = 0, 0, time.time()
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            text = row["text"].replace("\n", " ").strip()
            if len(text) < MIN_DOC_LEN:
                continue
            line = text + "\n"
            f.write(line)
            written += len(line.encode("utf-8"))
            n_docs  += 1
            if n_docs % 20000 == 0:
                print(f"  [{lang}] {written/MB:.0f}/{target_bytes/MB:.0f} MB  {n_docs:,} docs  {time.time()-t0:.0f}s")
            if written >= target_bytes:
                break

    meta = {"dataset_id": dataset_id, "config": config, "split": split,
            "target_bytes": target_bytes, "bytes_written": written,
            "n_docs": n_docs, "seconds": round(time.time() - t0, 1)}
    json.dump(meta, open(meta_path, "w"), indent=2)
    print(f"[{lang}] done: {n_docs:,} docs, {written/MB:.0f} MB in {meta['seconds']:.0f}s")


if __name__ == "__main__":
    for lang, ds_id, cfg, split, target in POOL_PLAN:
        sample_pool(lang, ds_id, cfg, split, target)
    print("\nAll pools ready:")
    for lang, *_ in POOL_PLAN:
        m = json.load(open(os.path.join(POOL_DIR, f"{lang}_meta.json")))
        print(f"  {lang}: {m['bytes_written']/MB:.0f} MB, {m['n_docs']:,} docs")
