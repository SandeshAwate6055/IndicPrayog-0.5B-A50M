import os, json

BASELINE_DIR = os.path.join(os.path.dirname(__file__), "..", "baselines")
os.makedirs(BASELINE_DIR, exist_ok=True)

BASELINES = [
    ("sarvam-1",     "sarvamai/sarvam-1"),
    ("sarvam-30b",   "sarvamai/sarvam-30b"),
    ("gemma-2",      "google/gemma-2-9b"),
    ("llama-3.1",    "meta-llama/Llama-3.1-8B"),
    ("qwen2.5",      "Qwen/Qwen2.5-7B"),
    ("mistral-v0.3", "mistralai/Mistral-7B-v0.3"),
    ("gpt2",         "gpt2"),
    ("indicbert-v2", "ai4bharat/IndicBERTv2-MLM-only"),
]

CANDIDATE_FILES = [
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "vocab.json", "merges.txt", "spiece.model", "tokenizer.model",
]


def fetch(name, repo_id, token):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, GatedRepoError, RepositoryNotFoundError

    out_dir   = os.path.join(BASELINE_DIR, name)
    ok_marker = os.path.join(out_dir, "loaded_ok.json")
    if os.path.exists(ok_marker):
        print(f"[{name}] cached — skipping")
        return {"name": name, "repo_id": repo_id, "status": "ok (cached)"}
    os.makedirs(out_dir, exist_ok=True)

    fetched = []
    for fname in CANDIDATE_FILES:
        try:
            hf_hub_download(repo_id=repo_id, filename=fname, token=token, local_dir=out_dir)
            fetched.append(fname)
        except EntryNotFoundError:
            continue
        except GatedRepoError as e:
            return {"name": name, "repo_id": repo_id, "status": "failed", "error": f"gated: {e}"}
        except RepositoryNotFoundError as e:
            return {"name": name, "repo_id": repo_id, "status": "failed", "error": f"not found: {e}"}
        except Exception as e:
            print(f"[{name}] {fname} skipped: {type(e).__name__}: {e}")

    has_fast = "tokenizer.json" in fetched
    has_slow = any(f in fetched for f in ("spiece.model", "tokenizer.model")) or \
               ("vocab.json" in fetched and "merges.txt" in fetched)
    if not (has_fast or has_slow):
        return {"name": name, "repo_id": repo_id, "status": "failed",
                "error": f"no usable tokenizer files (got: {fetched})"}

    json.dump({"fetched_files": fetched}, open(ok_marker, "w"))
    print(f"[{name}] OK — {fetched}")
    return {"name": name, "repo_id": repo_id, "status": "ok", "fetched_files": fetched}


if __name__ == "__main__":
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("WARNING: HF_TOKEN not set — gated repos will fail.")

    results = [fetch(name, repo_id, token) for name, repo_id in BASELINES]

    json.dump(results, open(os.path.join(BASELINE_DIR, "manifest.json"), "w"), indent=2)
    ok     = [r for r in results if r["status"].startswith("ok")]
    failed = [r for r in results if r["status"] == "failed"]
    print(f"\n{len(ok)}/{len(results)} baselines fetched OK.")
    for r in failed:
        print(f"  FAILED {r['name']}: {r['error'][:150]}")
