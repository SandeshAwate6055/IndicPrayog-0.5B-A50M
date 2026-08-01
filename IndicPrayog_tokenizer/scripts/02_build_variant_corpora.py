import os

MB = 1024 * 1024
POOL_DIR    = os.path.join(os.path.dirname(__file__), "..", "dataset", "pools")
VARIANT_DIR = os.path.join(os.path.dirname(__file__), "..", "variants")
TOTAL_BYTES = int(1.5 * 1024 * MB)

# D/E/F reuse mix_A corpus (algorithm/pretokenizer differs at training time, not data level)
MIXES = {
    "mix_A": {"hi": 0.40, "en": 0.35, "mr": 0.25},
    "mix_B": {"hi": 0.33, "en": 0.33, "mr": 0.34},
    "mix_C": {"hi": 0.45, "en": 0.35, "mr": 0.20},
}


def slice_file(lang, n_bytes, dst_path):
    written = 0
    with open(os.path.join(POOL_DIR, f"{lang}.txt"), "r", encoding="utf-8") as fin, \
         open(dst_path, "w", encoding="utf-8") as fout:
        for line in fin:
            b = len(line.encode("utf-8"))
            if written + b > n_bytes:
                break
            fout.write(line)
            written += b
    return written


def build_mix(name, shares):
    mix_dir = os.path.join(VARIANT_DIR, name)
    os.makedirs(mix_dir, exist_ok=True)
    print(f"\n=== {name} {shares} ===")
    for lang, share in shares.items():
        target  = int(TOTAL_BYTES * share)
        written = slice_file(lang, target, os.path.join(mix_dir, f"{lang}.txt"))
        print(f"  {lang}: {written/MB:.0f} / {target/MB:.0f} MB", end="")
        if written < target * 0.95:
            print(f"  !! pool too small — got {written/MB:.0f} MB, need {target/MB:.0f} MB", end="")
        print()


if __name__ == "__main__":
    for name, shares in MIXES.items():
        build_mix(name, shares)
    print("\nCorpora ready:")
    for name in MIXES:
        d = os.path.join(VARIANT_DIR, name)
        for f in sorted(os.listdir(d)):
            print(f"  {name}/{f}: {os.path.getsize(os.path.join(d,f))/MB:.1f} MB")
