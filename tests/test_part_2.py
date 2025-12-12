import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.part_2 import text_stats, sentence_length_stats

def test_text_stats_basic():
    text = "Hello world! Hello again."
    stats = text_stats(text)

    assert isinstance(stats, dict)
    # 3 tokens: "hello", "world", "again" (lowercased, unique)
    assert stats["type_token_ratio"] > 0
    assert stats["hapax_ratio"] > 0
    assert stats["mean_word_len"] > 0
    assert stats["punct_per_token"] > 0
    print('test_text_stats_basic test pass')


def test_text_stats_empty():
    stats = text_stats("")

    assert stats["type_token_ratio"] == 0.0
    assert stats["hapax_ratio"] == 0.0
    assert stats["mean_word_len"] == 0.0
    assert stats["punct_per_token"] == 0.0
    print('test_text_stats_empty test pass')

def test_sentence_length_stats_basic():
    text = "This is a short sentence. This is a much longer sentence with many words."
    stats = sentence_length_stats(text, long_thresh=5)

    # Basic sanity checks
    assert isinstance(stats, dict)
    assert stats["sent_count"] == 2
    assert stats["avg_sent_len_tokens"] > 0
    assert stats["std_sent_len_tokens"] >= 0
    assert 0 <= stats["prop_long_sents"] <= 1
    print('test_sentence_length_stats_basic test pass')


def test_sentence_length_stats_empty():
    stats = sentence_length_stats("", long_thresh=5)

    # All zeros for empty input
    assert stats["sent_count"] == 0
    assert stats["avg_sent_len_tokens"] == 0.0
    assert stats["std_sent_len_tokens"] == 0.0
    assert stats["prop_long_sents"] == 0.0
    print('test_sentence_length_stats_empty test pass')

def test_sentence_length_stats_long_threshold():
    text = (
        "Short sentence. "
        "This is definitely a much longer sentence with many many words in it."
    )
    stats = sentence_length_stats(text, long_thresh=5)

    # expect at least 2 sentences
    assert stats["sent_count"] >= 2

    # Proportion of long sentences should be strictly between 0 and 1
    assert 0 < stats["prop_long_sents"] < 1
    print('test_sentence_length_stats_long_threshold test pass')

if __name__ == "__main__":
    test_text_stats_basic()
    test_text_stats_empty()
    test_sentence_length_stats_basic()
    test_sentence_length_stats_empty()
    test_sentence_length_stats_long_threshold()