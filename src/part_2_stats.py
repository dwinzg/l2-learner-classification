import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterator
from collections import Counter
import string
import re
import math

import spacy
from nltk.stem import WordNetLemmatizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.part_1 import iterate_documents  

_nlp = spacy.blank("en")
_nlp.add_pipe("sentencizer")

# Initialise lemmatizer
_lemmatizer = WordNetLemmatizer()

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
        - The raw features `sent_count` and `std_sent_len_tokens` are not returned
          directly; instead we use a normalized rate (sent_per_100_tokens) and
          sent_cv_log to capture sentence-length variation.
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
            "sent_per_100_tokens": 0.0,
            "avg_sent_len_tokens": 0.0,
            "sent_cv_log": 0.0,
        }

    sent_count = len(lengths)
    total_tokens = sum(lengths)
    avg_len = mean(lengths)
    std_len = pstdev(lengths) if len(lengths) > 1 else 0.0

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
        "sent_per_100_tokens": sent_per_100_tokens,
        "avg_sent_len_tokens": avg_len,
        "sent_cv_log": sent_cv_log,
    }


def text_stats(text: str) -> Dict[str, float]:
    """
    Compute token-level statistical features for a single text.

    Features (updated):
        - unique_lemma_ratio: #unique lemmas / total_tokens
        - hapax_ratio: hapax_legomena_count / total_tokens
        - mean_word_len: average characters per token (letters only)
        - punct_per_token: punctuation characters count / total_tokens

    Notes:
        - The original feature `type_token_ratio` was removed because
          it was redundant with hapax_ratio.
        - The original feature `herdan_c` was removed because it was
          poorly discriminative on short Lang-8 texts (values collapsed
          near 1.0 for almost all documents).
    """

    tokens = re.findall(r"\b\w+\b", text.lower())
    total_tokens = len(tokens)

    punct_chars = sum(1 for ch in text if ch in string.punctuation)

    if total_tokens == 0:
        return {
            "unique_lemma_ratio": 0.0,
            "hapax_ratio": 0.0,
            "mean_word_len": 0.0,
            "punct_per_token": 0.0,
        }

    # Lemmas for unique_lemma_ratio
    lemmas = [_lemmatizer.lemmatize(tok) for tok in tokens]
    unique_lemmas = len(set(lemmas))
    unique_lemma_ratio = unique_lemmas / total_tokens

    # Hapax ratio
    counts = Counter(tokens)
    hapax_count = sum(1 for tok, c in counts.items() if c == 1)
    hapax_ratio = hapax_count / total_tokens

    # Mean word length
    total_letters = 0
    for tok in tokens:
        letters_only = [ch for ch in tok if ch.isalpha()]
        total_letters += len(letters_only)
    mean_word_len = total_letters / total_tokens if total_letters > 0 else 0.0

    # Punctuation per token
    punct_per_token = punct_chars / total_tokens

    return {
        "unique_lemma_ratio": unique_lemma_ratio,
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
            'unique_lemma_ratio', 'hapax_ratio', 'mean_word_len', 'punct_per_token'
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
# if __name__ == "__main__":
#     zip_path = "data/raw/lang-8.zip"
#     for i, row in enumerate(iter_sentence_features(zip_path, long_thresh=20)):
#         print(
#             f"{row['filename']}: L1={row['l1']}, "
#             f"sent_per_100={row['sent_per_100_tokens']:.2f}, "
#             f"avg_len={row['avg_sent_len_tokens']:.2f}, "
#             f"sent_cv_log={row['sent_cv_log']:.3f}, "
#             f"lemma_ratio={row['unique_lemma_ratio']:.3f}, "
#             f"hapax={row['hapax_ratio']:.3f}, "
#             f"mean_word_len={row['mean_word_len']:.2f}, "
#             f"punct_per_tok={row['punct_per_token']:.3f}"
#         )
#         if i >= 10:  # To show more or less information, edit i
#             break
