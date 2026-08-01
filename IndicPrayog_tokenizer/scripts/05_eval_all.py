import os, json, regex, unicodedata

ROOT         = os.path.join(os.path.dirname(__file__), "..")
OUT_TOK_DIR  = os.path.join(ROOT, "tokenizers_out")
BASELINE_DIR = os.path.join(ROOT, "baselines")
RESULTS_DIR  = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

FLORES_REPO       = "haoranxu/FLORES-200"
FLORES_PAIR_FILES = {"hi": "hi-en/test-00000-of-00001.parquet",
                     "mr": "mr-en/test-00000-of-00001.parquet"}
HELDOUT_DOCS  = 2000
HELDOUT_SKIP  = 300_000
MIN_DOC_LEN   = 150

GRAPHEME_RE   = regex.compile(r"\X")
DEVANAGARI_RE = regex.compile(r"[ऀ-ॿ]")

HELDOUT_SOURCES = {
    "hi": ("ai4bharat/sangraha",        "verified",    "hin"),
    "mr": ("ai4bharat/sangraha",        "verified",    "mar"),
    "en": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train"),
}


def load_our_tokenizer(path):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(path)
    class W:
        def encode(self, t):    return tok.encode(t).ids
        def decode(self, ids):  return tok.decode(ids)
        def vocab_size(self):   return tok.get_vocab_size()
    return W()


def load_hf_tokenizer(dir_path):
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    tok = None
    fast_json = os.path.join(dir_path, "tokenizer.json")
    if os.path.exists(fast_json):
        try:
            tok = PreTrainedTokenizerFast(tokenizer_file=fast_json)
        except Exception:
            tok = None
    if tok is None:
        tok = AutoTokenizer.from_pretrained(dir_path, trust_remote_code=False)
    class W:
        def encode(self, t):    return tok.encode(t, add_special_tokens=False)
        def decode(self, ids):  return tok.decode(ids)
        def vocab_size(self):   return tok.vocab_size
    return W()


_FLORES_CACHE = {}

def _load_flores_pair(lang_code):
    from huggingface_hub import hf_hub_download
    import pandas as pd
    path = hf_hub_download(repo_id=FLORES_REPO, filename=FLORES_PAIR_FILES[lang_code],
                            repo_type="dataset")
    rows = pd.read_parquet(path)[pd.read_parquet(path).columns[0]].tolist()
    return [r[lang_code] for r in rows], [r["en"] for r in rows]

def load_flores(lang_code):
    if lang_code == "en":
        if "hi" not in _FLORES_CACHE:
            _FLORES_CACHE["hi"], _FLORES_CACHE["en"] = _load_flores_pair("hi")
        return _FLORES_CACHE["en"]
    if lang_code not in _FLORES_CACHE:
        _FLORES_CACHE[lang_code], en = _load_flores_pair(lang_code)
        _FLORES_CACHE.setdefault("en", en)
    return _FLORES_CACHE[lang_code]


def load_heldout_docs(lang_code, dataset_id, config, split, n_docs):
    from datasets import load_dataset
    ds, docs = load_dataset(dataset_id, config, split=split, streaming=True), []
    for row in iter(ds.skip(HELDOUT_SKIP)):
        text = row["text"].replace("\n", " ").strip()
        if len(text) >= MIN_DOC_LEN:
            docs.append(text)
        if len(docs) >= n_docs:
            break
    return docs


def devanagari_integrity(tok, text):
    graphemes     = list(GRAPHEME_RE.finditer(text))
    dev_graphemes = [g for g in graphemes if DEVANAGARI_RE.search(g.group())]
    if not dev_graphemes:
        return None
    ids, acc, cum_lens = tok.encode(text), "", []
    for k in range(1, len(ids) + 1):
        acc = tok.decode(ids[:k])
        cum_lens.append(len(acc))
    boundaries = set([0] + cum_lens)
    kept = sum(1 for g in dev_graphemes
               if not any(g.start() < b < g.end() for b in boundaries))
    return kept / len(dev_graphemes)


def eval_tokenizer(name, tok):
    result = {"name": name}
    langs  = ("hi", "mr", "en")
    flores = {lang: load_flores(lang) for lang in langs}
    n_sent = min(len(v) for v in flores.values())

    fert_flores, tokens_flores = {}, {}
    roundtrip_ok, dev_integrity_scores = [], []

    for lang in langs:
        n_tok = n_words = 0
        for s in flores[lang][:n_sent]:
            ids = tok.encode(s)
            n_tok   += len(ids)
            n_words += len(s.split())
            decoded  = tok.decode(ids)
            # NFC-normalize both sides: tokenizer applies NFC normalization by design,
            # so decode(encode(s)) returns NFC(s), which may differ from non-canonical source.
            roundtrip_ok.append(
                unicodedata.normalize("NFC", decoded.strip()) ==
                unicodedata.normalize("NFC", s.strip())
            )
            if lang in ("hi", "mr"):
                di = devanagari_integrity(tok, s)
                if di is not None:
                    dev_integrity_scores.append(di)
        fert_flores[lang]   = n_tok / max(n_words, 1)
        tokens_flores[lang] = n_tok

    result["flores_fertility"]       = fert_flores
    result["flores_parity_hi_vs_en"] = round(tokens_flores["hi"] / tokens_flores["en"], 3)
    result["flores_parity_mr_vs_en"] = round(tokens_flores["mr"] / tokens_flores["en"], 3)
    result["round_trip_fidelity"]    = round(sum(roundtrip_ok) / len(roundtrip_ok), 4)
    result["devanagari_integrity"]   = (
        round(sum(dev_integrity_scores) / len(dev_integrity_scores), 4)
        if dev_integrity_scores else None
    )

    heldout_fert, heldout_compression = {}, {}
    for lang, (ds_id, cfg, split) in HELDOUT_SOURCES.items():
        docs = load_heldout_docs(lang, ds_id, cfg, split, HELDOUT_DOCS)
        n_tok = n_words = n_bytes = 0
        for d in docs:
            ids      = tok.encode(d)
            n_tok   += len(ids)
            n_words += len(d.split())
            n_bytes += len(d.encode("utf-8"))
        heldout_fert[lang]        = round(n_tok / max(n_words, 1), 3)
        heldout_compression[lang] = round(n_bytes / max(n_tok, 1), 3)
    result["heldout_fertility"]                   = heldout_fert
    result["heldout_compression_bytes_per_token"] = heldout_compression

    used_ids = set()
    for lang, (ds_id, cfg, split) in HELDOUT_SOURCES.items():
        for d in load_heldout_docs(lang, ds_id, cfg, split, min(HELDOUT_DOCS, 500)):
            used_ids.update(tok.encode(d))
    result["vocab_utilization"] = round(len(used_ids) / tok.vocab_size(), 4)
    result["vocab_size"]        = tok.vocab_size()
    return result


def main():
    all_results = []

    for m in json.load(open(os.path.join(OUT_TOK_DIR, "manifest.json"))):
        name = f"ours_{m['variant']}"
        print(f"\n### {name} ###")
        r = eval_tokenizer(name, load_our_tokenizer(m["path"]))
        r["group"] = "ours"
        r["meta"]  = m
        all_results.append(r)

    bmanifest = os.path.join(BASELINE_DIR, "manifest.json")
    if os.path.exists(bmanifest):
        for b in json.load(open(bmanifest)):
            if not b["status"].startswith("ok"):
                continue
            name = f"baseline_{b['name']}"
            print(f"\n### {name} ###")
            try:
                r = eval_tokenizer(name, load_hf_tokenizer(os.path.join(BASELINE_DIR, b["name"])))
                r["group"] = "baseline"
                r["meta"]  = b
                all_results.append(r)
            except Exception as e:
                print(f"  SKIPPED: {type(e).__name__}: {e}")

    json.dump(all_results, open(os.path.join(RESULTS_DIR, "eval_results.json"), "w"),
              indent=2, ensure_ascii=False)

    lines = [
        "| Tokenizer | Group | Vocab | FLORES fert hi/mr/en | Parity hi/en | Parity mr/en | "
        "Round-trip | Dev.Integrity | VocabUtil |",
        "|:---|:---|---:|:---|---:|---:|---:|---:|---:|",
    ]
    for r in all_results:
        ff = r["flores_fertility"]
        di = (r["devanagari_integrity"] or 0) * 100
        lines.append(
            f"| {r['name']} | {r['group']} | {r['vocab_size']:,} | "
            f"{ff['hi']:.2f}/{ff['mr']:.2f}/{ff['en']:.2f} | "
            f"{r['flores_parity_hi_vs_en']:.2f} | {r['flores_parity_mr_vs_en']:.2f} | "
            f"{r['round_trip_fidelity']*100:.1f}% | {di:.1f}% | "
            f"{r['vocab_utilization']*100:.1f}% |"
        )
    open(os.path.join(RESULTS_DIR, "eval_results.md"), "w").write("\n".join(lines))
    print("\nSaved results/eval_results.json + .md")


if __name__ == "__main__":
    main()
