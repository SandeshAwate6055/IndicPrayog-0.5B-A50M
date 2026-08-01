import os, json, time, math

ROOT        = os.path.join(os.path.dirname(__file__), "..")
POOL_DIR    = os.path.join(ROOT, "dataset", "pools")
OUT_TOK_DIR = os.path.join(ROOT, "tokenizers_out")
RESULTS_DIR = os.path.join(ROOT, "results")
PROXY_DIR   = os.path.join(ROOT, "proxy_runs")
os.makedirs(PROXY_DIR, exist_ok=True)

TARGET_TOKENS_TOTAL   = 300_000_000
LANG_SHARE            = {"hi": 0.40, "en": 0.35, "mr": 0.25}
HELDOUT_DOCS_PER_LANG = 500

SEQ_LEN      = 1024
BATCH_SIZE   = 32
MAX_STEPS    = 3000
LR           = 3e-4
WARMUP_STEPS = 100
SEED         = 1234

MODEL_CFG = dict(
    hidden_size=320, n_layers=6, n_heads=5, n_kv_heads=5,
    intermediate_size=896, max_position_embeddings=SEQ_LEN,
)


class TokWrap:
    def __init__(self, path):
        from tokenizers import Tokenizer
        self.tok    = Tokenizer.from_file(path)
        self.eos_id = self.tok.token_to_id("<eos>")
        self.pad_id = self.tok.token_to_id("<pad>")
    def encode(self, text): return self.tok.encode(text).ids
    def vocab_size(self):   return self.tok.get_vocab_size()


def load_docs(lang, max_docs=None):
    docs = []
    with open(os.path.join(POOL_DIR, f"{lang}.txt"), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(line)
            if max_docs and len(docs) >= max_docs:
                break
    return docs


def build_packed_bins(tw, docs_by_lang, out_prefix, target_tokens_by_lang):
    import numpy as np
    all_ids, actual = [], {}
    for lang, docs in docs_by_lang.items():
        n_tok = 0
        for d in docs:
            ids = tw.encode(d)
            all_ids.extend(ids)
            all_ids.append(tw.eos_id)
            n_tok += len(ids) + 1
            if n_tok >= target_tokens_by_lang[lang]:
                break
        actual[lang] = n_tok
    arr = np.array(all_ids, dtype=np.uint32)
    arr.tofile(out_prefix + ".bin")
    return actual, len(arr)


def prepare_data_for_variant(variant, held_docs, train_docs):
    tw      = TokWrap(os.path.join(OUT_TOK_DIR, f"{variant}.json"))
    out_dir = os.path.join(PROXY_DIR, variant)
    os.makedirs(out_dir, exist_ok=True)

    train_target = {l: int(TARGET_TOKENS_TOTAL * LANG_SHARE[l]) for l in LANG_SHARE}
    print(f"[{variant}] packing train bins ...")
    t0 = time.time()
    actual, total = build_packed_bins(tw, train_docs,
                                      os.path.join(out_dir, "train"), train_target)
    print(f"[{variant}] train: {total:,} tokens ({time.time()-t0:.0f}s)")

    heldout_meta = {}
    for lang, docs in held_docs.items():
        _, total_h = build_packed_bins(tw, {lang: docs},
                                       os.path.join(out_dir, f"heldout_{lang}"), {lang: 10**12})
        heldout_meta[lang] = {
            "n_tokens": total_h,
            "n_bytes":  sum(len(d.encode("utf-8")) for d in docs),
        }

    meta = {"variant": variant, "vocab_size": tw.vocab_size(),
            "train_tokens_total": total, "train_tokens_by_lang": actual,
            "heldout": heldout_meta, "pad_id": tw.pad_id, "eos_id": tw.eos_id}
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=2)
    return meta


def train_proxy_lm(variant, meta, device):
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(SEED)
    out_dir    = os.path.join(PROXY_DIR, variant)
    vocab_size = meta["vocab_size"]

    class RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.eps    = eps
        def forward(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

    def rotary(x, pos, head_dim):
        half  = head_dim // 2
        freqs = 1.0 / (10000 ** (torch.arange(0, half, device=x.device).float() / half))
        ang   = pos[:, None].float() * freqs[None, :]
        c, s  = ang.cos()[None, None], ang.sin()[None, None]
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).to(x.dtype)

    class Block(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            h, nh       = cfg["hidden_size"], cfg["n_heads"]
            self.nh, self.hd = nh, h // nh
            self.ln1  = RMSNorm(h)
            self.qkv  = nn.Linear(h, 3 * h, bias=False)
            self.o    = nn.Linear(h, h, bias=False)
            self.ln2  = RMSNorm(h)
            self.gate = nn.Linear(h, cfg["intermediate_size"], bias=False)
            self.up   = nn.Linear(h, cfg["intermediate_size"], bias=False)
            self.down = nn.Linear(cfg["intermediate_size"], h, bias=False)
        def forward(self, x, pos):
            B, T, H = x.shape
            qkv = self.qkv(self.ln1(x)).view(B, T, 3, self.nh, self.hd).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            q, k = rotary(q, pos, self.hd), rotary(k, pos, self.hd)
            x = x + self.o(F.scaled_dot_product_attention(q, k, v, is_causal=True)
                           .transpose(1, 2).reshape(B, T, H))
            xn = self.ln2(x)
            return x + self.down(F.silu(self.gate(xn)) * self.up(xn))

    class TinyLM(nn.Module):
        def __init__(self, cfg, vocab_size):
            super().__init__()
            h           = cfg["hidden_size"]
            self.embed  = nn.Embedding(vocab_size, h)
            self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg["n_layers"])])
            self.norm_f = RMSNorm(h)
            self.head   = nn.Linear(h, vocab_size, bias=False)
            self.head.weight = self.embed.weight  # tied embeddings
            # N(0, 0.02) init — required with tied embeddings to keep step-0 loss near ln(vocab)
            self.apply(lambda m: nn.init.normal_(m.weight, 0, 0.02)
                       if isinstance(m, (nn.Linear, nn.Embedding)) else None)
        def forward(self, ids):
            x = self.embed(ids)
            pos = torch.arange(ids.shape[1], device=ids.device)
            for blk in self.blocks:
                x = blk(x, pos)
            return self.head(self.norm_f(x))

    model    = TinyLM(MODEL_CFG, vocab_size).to(device).to(torch.bfloat16)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{variant}] {n_params/1e6:.1f}M params")

    train_arr = np.memmap(os.path.join(out_dir, "train.bin"), dtype=np.uint32, mode="r")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)

    def get_batch():
        idx = np.random.randint(0, len(train_arr) - SEQ_LEN - 1, size=BATCH_SIZE)
        x = np.stack([train_arr[i:i + SEQ_LEN]     for i in idx]).astype(np.int64)
        y = np.stack([train_arr[i+1:i + SEQ_LEN+1] for i in idx]).astype(np.int64)
        return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)

    model.train()
    t0 = time.time()
    for step in range(MAX_STEPS):
        for g in opt.param_groups:
            g["lr"] = LR * min(1.0, (step + 1) / WARMUP_STEPS)
        x, y   = get_batch()
        logits = model(x)
        loss   = F.cross_entropy(logits.view(-1, vocab_size).float(), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 200 == 0 or step == MAX_STEPS - 1:
            print(f"[{variant}] step {step}/{MAX_STEPS}  loss={loss.item():.4f}  ({time.time()-t0:.0f}s)")

    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    print(f"[{variant}] saved -> {out_dir}/model.pt")

    model.eval()
    bpb_by_lang = {}
    for lang, hmeta in meta["heldout"].items():
        harr      = np.memmap(os.path.join(out_dir, f"heldout_{lang}.bin"), dtype=np.uint32, mode="r")
        total_nll = total_tok = 0
        with torch.no_grad():
            i = 0
            while i + SEQ_LEN + 1 <= len(harr):
                x = torch.from_numpy(harr[i:i+SEQ_LEN].astype(np.int64))[None].to(device)
                y = torch.from_numpy(harr[i+1:i+SEQ_LEN+1].astype(np.int64))[None].to(device)
                total_nll += F.cross_entropy(model(x).view(-1, vocab_size).float(),
                                              y.view(-1), reduction="sum").item()
                total_tok += SEQ_LEN
                i         += SEQ_LEN
        frac_covered  = total_tok / hmeta["n_tokens"] if hmeta["n_tokens"] else 0
        bytes_covered = hmeta["n_bytes"] * frac_covered
        bpb           = (total_nll * math.log2(math.e)) / max(bytes_covered, 1)
        bpb_by_lang[lang] = round(bpb, 4)
        print(f"[{variant}] heldout {lang}: BPB={bpb:.4f}")

    return {"variant": variant, "n_params": n_params, "bpb_by_lang": bpb_by_lang,
            "mean_bpb": round(sum(bpb_by_lang.values()) / len(bpb_by_lang), 4),
            "train_seconds": round(time.time() - t0, 1)}


def main():
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    device = "cuda"

    variants = [s.replace("ours_", "")
                for s in json.load(open(os.path.join(RESULTS_DIR, "shortlist.json")))["shortlist"]]
    print(f"Proxy-LM candidates: {variants}")

    langs      = ("hi", "en", "mr")
    all_docs   = {l: load_docs(l) for l in langs}
    held_docs  = {l: all_docs[l][-HELDOUT_DOCS_PER_LANG:] for l in langs}
    train_docs = {l: all_docs[l][:-HELDOUT_DOCS_PER_LANG] for l in langs}

    proxy_results = []
    for variant in variants:
        print(f"\n=== Proxy-LM: variant {variant} ===")
        meta = prepare_data_for_variant(variant, held_docs, train_docs)
        proxy_results.append(train_proxy_lm(variant, meta, device))

    proxy_results.sort(key=lambda r: r["mean_bpb"])
    out_path = os.path.join(RESULTS_DIR, "proxy_lm_results.json")
    json.dump(proxy_results, open(out_path, "w"), indent=2)
    print("\nProxy-LM results (sorted by mean BPB):")
    for r in proxy_results:
        print(f"  {r['variant']}: mean_bpb={r['mean_bpb']:.4f}  {r['bpb_by_lang']}")
    print(f"\nWinner (lowest mean BPB): {proxy_results[0]['variant']}")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
