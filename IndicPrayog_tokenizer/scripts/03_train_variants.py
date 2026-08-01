import os, time

VOCAB_SIZE     = 64_000
SPECIAL_TOKENS = ["<pad>", "<eos>", "<bos>", "<unk>"]
VARIANT_DIR    = os.path.join(os.path.dirname(__file__), "..", "variants")
OUT_DIR        = os.path.join(os.path.dirname(__file__), "..", "tokenizers_out")
os.makedirs(OUT_DIR, exist_ok=True)


def files_for(mix_name):
    d = os.path.join(VARIANT_DIR, mix_name)
    return [os.path.join(d, f"{lang}.txt") for lang in ("hi", "en", "mr")]


def train_bpe(name, mix_name, pretok_extra=None, digit_split=True):
    from tokenizers import Tokenizer, Regex
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel, Digits, Sequence, UnicodeScripts, Split
    from tokenizers.normalizers import NFC
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder

    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.normalizer = NFC()

    # Custom word-split regex: keeps Devanagari Letter+Mark runs together as one pretoken
    # (ByteLevel's default regex only groups Unicode Letter, splitting matras from consonants),
    # and attaches a leading space to the following word (GPT-2 convention) so BPE can merge
    # space+word into a single token. use_regex=False disables ByteLevel's internal re-split.
    word_split = Split(pattern=Regex(r" ?[^\s\d]+| ?\d+|\s+"), behavior="isolated")

    steps = []
    if pretok_extra == "unicode_scripts":
        # UnicodeScripts must precede Digits — if Digits runs first it can leave a whitespace-
        # only pretoken that UnicodeScripts silently drops, corrupting round-trip.
        steps.append(UnicodeScripts())
    steps.append(word_split)
    if digit_split:
        steps.append(Digits(individual_digits=True))
    steps.append(ByteLevel(add_prefix_space=False, use_regex=False))
    tok.pre_tokenizer = Sequence(steps)
    tok.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )

    files = files_for(mix_name)
    print(f"\n=== Training {name} (BPE, mix={mix_name}, digit_split={digit_split}, extra={pretok_extra}) ===")
    t0 = time.time()
    tok.train(files, trainer=trainer)
    dt = time.time() - t0
    print(f"{name}: {dt/60:.1f} min, vocab={tok.get_vocab_size():,}")

    out_path = os.path.join(OUT_DIR, f"{name}.json")
    tok.save(out_path)
    print(f"{name}: saved -> {out_path}")
    return {"variant": name, "algo": "BPE", "mix": mix_name, "digit_split": digit_split,
            "extra": pretok_extra, "train_seconds": round(dt, 1), "path": out_path}


def train_unigram(name, mix_name):
    from tokenizers import Tokenizer
    from tokenizers.models import Unigram
    from tokenizers.trainers import UnigramTrainer
    from tokenizers.pre_tokenizers import Metaspace
    from tokenizers.normalizers import NFC
    from tokenizers.decoders import Metaspace as MetaspaceDecoder

    tok = Tokenizer(Unigram())
    tok.normalizer = NFC()
    tok.pre_tokenizer = Metaspace()
    tok.decoder = MetaspaceDecoder()

    trainer = UnigramTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        unk_token="<unk>",
        byte_fallback=True,
        show_progress=True,
    )

    files = files_for(mix_name)
    print(f"\n=== Training {name} (Unigram, mix={mix_name}) ===")
    t0 = time.time()
    tok.train(files, trainer=trainer)
    dt = time.time() - t0
    print(f"{name}: {dt/60:.1f} min, vocab={tok.get_vocab_size():,}")

    out_path = os.path.join(OUT_DIR, f"{name}.json")
    tok.save(out_path)
    print(f"{name}: saved -> {out_path}")
    return {"variant": name, "algo": "Unigram", "mix": mix_name, "digit_split": None,
            "extra": "byte_fallback", "train_seconds": round(dt, 1), "path": out_path}


if __name__ == "__main__":
    import json

    manifest = [
        train_bpe("A", "mix_A"),
        train_bpe("B", "mix_B"),
        train_bpe("C", "mix_C"),
        train_unigram("D", "mix_A"),
        train_bpe("E", "mix_A", pretok_extra="unicode_scripts"),
        train_bpe("F", "mix_A", digit_split=False),
    ]

    manifest_path = os.path.join(OUT_DIR, "manifest.json")
    json.dump(manifest, open(manifest_path, "w"), indent=2)
    print(f"\nAll 6 variants trained. Manifest -> {manifest_path}")
    for m in manifest:
        print(f"  {m['variant']}: {m['algo']:8s}  mix={m['mix']:6s}  {m['train_seconds']:6.1f}s")
