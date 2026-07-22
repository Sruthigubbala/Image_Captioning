"""
vocabulary.py
Builds a word<->index vocabulary from the Flickr8k captions file
and provides encode/decode helpers.
"""

import re
import pickle
from collections import Counter


class Vocabulary:
    PAD_TOKEN = "<pad>"
    START_TOKEN = "<start>"
    END_TOKEN = "<end>"
    UNK_TOKEN = "<unk>"

    def __init__(self, freq_threshold: int = 5):
        self.freq_threshold = freq_threshold
        self.word2idx = {}
        self.idx2word = {}
        self._build_special_tokens()

    def _build_special_tokens(self):
        specials = [self.PAD_TOKEN, self.START_TOKEN, self.END_TOKEN, self.UNK_TOKEN]
        for i, tok in enumerate(specials):
            self.word2idx[tok] = i
            self.idx2word[i] = tok

    def __len__(self):
        return len(self.word2idx)

    @staticmethod
    def tokenize(text: str):
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return text.split()

    def build_vocabulary(self, caption_list):
        """caption_list: list of raw caption strings"""
        counter = Counter()
        for caption in caption_list:
            tokens = self.tokenize(caption)
            counter.update(tokens)

        idx = len(self.word2idx)
        for word, freq in counter.items():
            if freq >= self.freq_threshold and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

        print(f"Vocabulary built: {len(self.word2idx)} tokens "
              f"(freq_threshold={self.freq_threshold})")

    def numericalize(self, text: str):
        """Convert a raw caption string into a list of token indices
        (without start/end tokens)."""
        tokens = self.tokenize(text)
        return [
            self.word2idx.get(token, self.word2idx[self.UNK_TOKEN])
            for token in tokens
        ]

    def encode_caption(self, text: str, max_len: int):
        """Full encode: <start> + tokens + <end>, padded/truncated to max_len."""
        ids = [self.word2idx[self.START_TOKEN]]
        ids += self.numericalize(text)
        ids.append(self.word2idx[self.END_TOKEN])

        if len(ids) < max_len:
            ids += [self.word2idx[self.PAD_TOKEN]] * (max_len - len(ids))
        else:
            ids = ids[:max_len - 1] + [self.word2idx[self.END_TOKEN]]

        return ids

    def decode_indices(self, indices):
        """Convert a list of indices back into a readable string,
        stopping at <end> and skipping special tokens."""
        words = []
        for idx in indices:
            word = self.idx2word.get(int(idx), self.UNK_TOKEN)
            if word == self.END_TOKEN:
                break
            if word in (self.PAD_TOKEN, self.START_TOKEN):
                continue
            words.append(word)
        return " ".join(words)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str):
        with open(path, "rb") as f:
            return pickle.load(f)


if __name__ == "__main__":
    # Quick smoke test
    sample_captions = [
        "A man is riding a horse on the beach",
        "A dog runs through the grass",
        "A man is riding a horse near the ocean",
    ]
    vocab = Vocabulary(freq_threshold=1)
    vocab.build_vocabulary(sample_captions)
    encoded = vocab.encode_caption("A man rides a horse", max_len=12)
    print("Encoded:", encoded)
    print("Decoded:", vocab.decode_indices(encoded))