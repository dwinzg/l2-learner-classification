import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterator
from collections import Counter
import string
import re
import math

import spacy

# Allow importing src/part_1.py
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.part_1 import iterate_documents  # must yield (l1, text, filename)

# Initialise spaCy once
_nlp = spacy.blank("en")
_nlp.add_pipe("sentencizer")


def sentence_length_stats(text: str, long_thresh: int = 20) -> Dict[str, float]:
    """
    Compute sentence-based features for a single text.
    
    Features (updated):
        - sent_per_100_tokens: number of sentences per 100 tokens
        - avg_sent_len_tokens: mean sentence length in tokens
        - sent_cv_log: log-normalized coefficient of variation of sentence length
          (log(1 + std_sent_len_tokens / avg_sent_len_tokens))

    Notes:
        - The original feature `prop_long_sents` (proportion of sentences with
          length > long_thresh) was removed because ablation showed it hurt model
          performance and it was highly correlated with avg/std sentence length.
        - The original raw features `sent_count` and `std_sent_len_tokens` were
          replaced by the normalized rate `sent_per_100_tokens` and by
          `sent_cv_log` plus a lexical richness feature `herdan_c` (see text_stats).
    """
    doc = _nlp(text)
    lengths = []

    for sent in doc.sents:
        # count tokens, ignoring spaces
        tokens = [t for t in sent if not t.is_space]
        if tokens:
            lengths.append(len(tokens))

    # No sentences / no tokens case
    if not lengths:
        return {
            # "sent_count": 0,                # original feature (now replaced)
            "sent_per_100_tokens": 0.0,
            "avg_sent_len_tokens": 0.0,
            # "std_sent_len_tokens": 0.0,     # original feature (now replaced)
            # "prop_long_sents": 0.0,         # original feature (removed)
            "sent_cv_log": 0.0,
        }

    sent_count = len(lengths)
    total_tokens = sum(lengths)
    avg_len = mean(lengths)
    std_len = pstdev(lengths) if len(lengths) > 1 else 0.0

    # Original prop_long_sents (commented out):
    # prop_long = sum(l > long_thresh for l in lengths) / sent_count

    # Coefficient of variation of sentence length
    if avg_len > 0:
        sent_cv = std_len / avg_len
        sent_cv_log = math.log1p(sent_cv)  # log(1 + cv), more stable
    else:
        sent_cv_log = 0.0

    # Sentences per 100 tokens (length-normalized sentence count)
    if total_tokens > 0:
        sent_per_100_tokens = sent_count / total_tokens * 100.0
    else:
        sent_per_100_tokens = 0.0

    return {
        # "sent_count": sent_count,              # replaced
        "sent_per_100_tokens": sent_per_100_tokens,
        "avg_sent_len_tokens": avg_len,
        # "std_sent_len_tokens": std_len,        # replaced
        # "prop_long_sents": prop_long,          # removed
        "sent_cv_log": sent_cv_log,
    }


def text_stats(text: str) -> Dict[str, float]:
    """
    Compute token-level statistical features for a single text.

    Features (updated):
        - herdan_c: length-independent lexical richness index
        - hapax_ratio: hapax_legomena_count / total_tokens
        - mean_word_len: average characters per token (letters only)
        - punct_per_token: punctuation characters count / total_tokens

    Notes:
        - The original feature `type_token_ratio` was removed because
          ablation and correlation analysis showed it was redundant
          with hapax_ratio and harmed classifier performance.
    """

    # Simple tokenization: words made of letters/digits/underscore
    tokens = re.findall(r"\b\w+\b", text.lower())
    total_tokens = len(tokens)

    # Count punctuation characters in the raw text
    punct_chars = sum(1 for ch in text if ch in string.punctuation)

    if total_tokens == 0:
        return {
            # "type_token_ratio": 0.0,   # removed
            "herdan_c": 0.0,
            "hapax_ratio": 0.0,
            "mean_word_len": 0.0,
            "punct_per_token": 0.0,
        }

    # Unique types (for herdan_c)
    types = set(tokens)
    n_types = len(types)

    # Herdan's C: log(types) / log(tokens), a length-normalized richness measure
    if total_tokens > 1 and n_types > 1:
        herdan_c = math.log(n_types) / math.log(total_tokens)
    else:
        herdan_c = 0.0

    # --- REMOVED ---
    # unique_tokens = len(set(tokens))
    # type_token_ratio = unique_tokens / total_tokens

    # Hapax ratio (tokens that appear exactly once)
    counts = Counter(tokens)
    hapax_count = sum(1 for tok, c in counts.items() if c == 1)
    hapax_ratio = hapax_count / total_tokens

    # Mean word length, counting only letters in each token
    total_letters = 0
    for tok in tokens:
        letters_only = [ch for ch in tok if ch.isalpha()]
        total_letters += len(letters_only)
    mean_word_len = total_letters / total_tokens if total_letters > 0 else 0.0

    # Punctuation per token (punctuation characters per word token)
    punct_per_token = punct_chars / total_tokens

    return {
        # "type_token_ratio": type_token_ratio,  # removed
        "herdan_c": herdan_c,
        "hapax_ratio": hapax_ratio,
        "mean_word_len": mean_word_len,
        "punct_per_token": punct_per_token,
    }


def iter_sentence_features(
    zip_path: str, long_thresh: int = 20
) -> Iterator[Dict[str, float]]:
    """
    Iterate over all Lang-8 documents and yield features.

    Yields one dict per document with keys:
        - 'l1'
        - 'filename'
        - sentence-based:
            'sent_per_100_tokens', 'avg_sent_len_tokens', 'sent_cv_log'
        - statistical:
            'herdan_c', 'hapax_ratio', 'mean_word_len', 'punct_per_token'
    """
    for l1, text, filename in iterate_documents(zip_path):
        # Example of excluding one L1 if needed
        if l1 == "Russian":
            continue

        sent_stats = sentence_length_stats(text, long_thresh=long_thresh)
        token_stats = text_stats(text)

        yield {
            "l1": l1,
            "filename": filename,
            **sent_stats,
            **token_stats,
        }


# Example Usage
if __name__ == "__main__":
    zip_path = "data/raw/lang-8.zip"
    for i, row in enumerate(iter_sentence_features(zip_path, long_thresh=20)):
        print(
            f"{row['filename']}: L1={row['l1']}, "
            f"sent_per_100={row['sent_per_100_tokens']:.2f}, "
            f"avg_len={row['avg_sent_len_tokens']:.2f}, "
            f"sent_cv_log={row['sent_cv_log']:.3f}, "
            f"herdan_c={row['herdan_c']:.3f}, "
            f"hapax={row['hapax_ratio']:.3f}, "
            f"mean_word_len={row['mean_word_len']:.2f}, "
            f"punct_per_tok={row['punct_per_token']:.3f}"
        )
        if i >= 20:
            break