# How the tokenizer actually works

## What it is

`AxisQuant/IndicPrayog-tokenizer-64k` is a subword tokenizer with a fixed vocabulary of 64,000 pieces. It was trained once, ahead of time, on our own English, Hindi, and Marathi text, so that these 64,000 pieces cover common words, common word fragments, and individual characters as a fallback. It is not a neural network, it does not "understand" text, it is a lookup and merge algorithm. That is exactly why it runs on CPU, see the CPU section below.

## Step by step, with a real example

Take the sentence `"Hello world"`. Here is exactly what happens inside `tokenizer.encode(...)`:

**Step 1: Normalize.** The raw text is cleaned up a bit (unicode normalization, sometimes lowercasing depending on how the tokenizer was trained). Our tokenizer keeps case as is.

**Step 2: Pre tokenize (split into rough chunks).** The text is split on spaces and punctuation boundaries first. `"Hello world"` becomes rough chunks like `Hello` and `world`, with a marker (`▁`) added to remember "this chunk started with a space".

**Step 3: Apply the trained vocabulary (subword split).** Each rough chunk is then broken down into pieces that actually exist in the 64,000 piece vocabulary. If the whole word exists as one piece, it stays as one token. If not, it gets split into smaller known pieces until every piece is something the vocabulary recognizes. The full mechanics of how it picks the split are below, in "Under the hood: the Unigram algorithm".

**Step 4: Map pieces to numbers.** Every piece has a fixed integer id in the vocabulary. The final output is a list of these integers, this is what actually gets fed to the model.

Real output from our tokenizer:

```
text:   "Hello world"
pieces: ['▁Hell', 'o', '▁world']
ids:    [18815, 176, 416]
```

Notice `Hello` was not one single piece, it got split into `▁Hell` and `o`, because `Hello` as a whole word was not common enough in our training data to earn its own vocabulary slot, but `Hell` and `o` were. `world` on the other hand was common enough to get its own single piece, `▁world`.

## Same thing in Hindi and Marathi

```
text:   "नमस्ते दुनिया"
pieces: ['▁नम', 'स्ते', '▁दुनिया']
ids:    [15140, 16592, 1218]

text:   "नमस्कार जग"
pieces: ['▁नमस्कार', '▁जग']
ids:    [15986, 2340]
```

`नमस्कार` (a very common Marathi greeting) got its own single piece, but `नमस्ते` (Hindi greeting) did not, it got split into `▁नम` and `स्ते`. This is exactly the kind of thing Step 4's chars per token measurement is trying to capture, some words and some languages need more pieces than others, and that changes how many tokens the same sentence actually costs.

## Under the hood: the Unigram algorithm

We checked our actual tokenizer file, its model type is `Unigram` (this is the SentencePiece Unigram algorithm, not BPE). Here is exactly how it decides where to split:

**1. Every one of the 64,000 pieces has a fixed score**, a log probability, learned once during tokenizer training from how often that piece appeared. Bigger (less negative) score means a more common, more "worth keeping whole" piece. From our real vocabulary:

```
piece      score
▁Hello     not in vocab at all
▁Hell      -12.00
o          -7.62
▁world     -8.42
▁नमस्कार    -11.82   (whole word, common enough to earn its own piece)
▁नम        -11.76
स्ते       -11.86    (नमस्ते was not common enough, so it is not one piece)
```

**2. To split a chunk, it tries every possible way to cut it into known pieces**, and adds up the scores for each full way of cutting it. It does this efficiently with a dynamic programming pass (Viterbi algorithm), not by brute force, walking through the chunk once and keeping the best scoring path ending at each position.

**3. It picks the single split with the highest total score.** For `"Hello"`, there is no piece for the whole word, so every candidate split is scored, and `▁Hell` + `o` wins over other options like `▁H` + `e` + `l` + `l` + `o`, because fewer, bigger, more common pieces score higher than many tiny ones.

This is why `नमस्कार` stays as one token (it earned its own high scoring piece) while `नमस्ते` gets cut into `▁नम` + `स्ते` (no single piece existed for it, so the best scoring 2 piece split was used instead).

**The Viterbi DP, in plain code:**

```python
def viterbi_segment(chunk, piece_score):
    """
    chunk: the rough chunk to split, e.g. "Hello"
    piece_score: dict mapping a known piece (substring) to its log probability score
    returns: the list of pieces with the highest total score
    """
    n = len(chunk)
    best_score = [float("-inf")] * (n + 1)
    best_score[0] = 0.0
    best_cut = [None] * (n + 1)   # best_cut[j] = (i, piece) that reaches position j

    for j in range(1, n + 1):
        for i in range(j):
            piece = chunk[i:j]
            if piece not in piece_score:
                continue
            score = best_score[i] + piece_score[piece]
            if score > best_score[j]:
                best_score[j] = score
                best_cut[j] = (i, piece)

    # walk the best path backwards to recover the pieces
    pieces = []
    j = n
    while j > 0:
        i, piece = best_cut[j]
        pieces.append(piece)
        j = i
    return list(reversed(pieces))
```

`best_score[j]` is "the highest total score of any valid split of the first `j` characters". For every position `j`, it looks back at every possible last piece ending there, and keeps whichever one gives the best running total, that is the dynamic programming part, each position is only computed once, not recomputed for every possible full split. `best_cut` remembers which choice was best, so at the end we just walk backwards from the last position to rebuild the winning split. This is a simplified version, SentencePiece's real implementation is written in C++ and is more optimized, but the actual logic it runs is exactly this.

## CPU vs GPU, in short

This is string matching and a scoring lookup, not matrix math, so there is nothing for a GPU to speed up here. Every tokenizer, GPT's, LLaMA's, and ours, runs on CPU for this exact reason. GPU only matters later, once the resulting token ids are turned into embeddings inside the actual model.

## Where this fits in our pipeline

```
raw text  ->  tokenizer.encode()  ->  list of integer ids  ->  (later) model
```

`4_tokenization/tokenizer_eval.ipynb` measures how expensive this conversion is per language and per category (English vs Hindi vs Marathi vs code vs URLs, etc.), and also runs it over the entire standardized dataset to get the real total token count, this is what tells us how many actual training tokens we have once everything is tokenized.
