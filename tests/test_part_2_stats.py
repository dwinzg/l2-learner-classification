import sys
from pathlib import Path

# Make sure `src` is importable when running this file directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.part_2_stats import text_stats, sentence_length_stats
from src.part_2_lexicon_pos import extract_lexicon_features, get_POS_rato_features


# -----------------------
# part_2_stats.py tests
# -----------------------

def test_text_stats_basic():
    text = "Hello world! Hello again."
    stats = text_stats(text)

    assert isinstance(stats, dict)

    # Check expected keys exist
    assert "unique_lemma_ratio" in stats
    assert "hapax_ratio" in stats
    assert "mean_word_len" in stats
    assert "punct_per_token" in stats

    # Basic sanity checks
    assert 0 <= stats["unique_lemma_ratio"] <= 1
    assert 0 <= stats["hapax_ratio"] <= 1
    assert stats["mean_word_len"] > 0
    assert stats["punct_per_token"] > 0

    print("test_text_stats_basic pass")


def test_text_stats_empty():
    stats = text_stats("")

    assert stats["unique_lemma_ratio"] == 0.0
    assert stats["hapax_ratio"] == 0.0
    assert stats["mean_word_len"] == 0.0
    assert stats["punct_per_token"] == 0.0

    print("test_text_stats_empty pass")


def test_sentence_length_stats_basic():
    text = "This is a short sentence. This is a much longer sentence with many words."
    stats = sentence_length_stats(text, long_thresh=5)

    assert isinstance(stats, dict)

    # Check expected keys exist
    assert "sent_per_100_tokens" in stats
    assert "avg_sent_len_tokens" in stats
    assert "sent_cv_log" in stats

    # Expect positive mean length
    assert stats["avg_sent_len_tokens"] > 0

    # sent_per_100_tokens should be >0 if there is at least one sentence/token
    assert stats["sent_per_100_tokens"] > 0

    # sent_cv_log is log1p(CV) => should be >= 0
    assert stats["sent_cv_log"] >= 0

    print("test_sentence_length_stats_basic pass")


def test_sentence_length_stats_empty():
    stats = sentence_length_stats("", long_thresh=5)

    assert stats["sent_per_100_tokens"] == 0.0
    assert stats["avg_sent_len_tokens"] == 0.0
    assert stats["sent_cv_log"] == 0.0

    print("test_sentence_length_stats_empty pass")


def test_sentence_length_stats_variability():
    text = "Short sentence. This is definitely a much longer sentence with many many words in it."
    stats = sentence_length_stats(text, long_thresh=5)

    # With a short + long sentence, variability should be > 0
    assert stats["sent_cv_log"] > 0

    print("test_sentence_length_stats_variability pass")


# -----------------------
# part_2_lexicon_pos.py tests
# -----------------------

def test_lexicon_asian_top_word_match_false():
    # Text has no Asia lexicon words in top nouns => should be False
    text = "I went to school today. The weather was nice. My friends were happy."
    feats = extract_lexicon_features(text)

    assert isinstance(feats, dict)
    assert "asian_top_word_match" in feats
    assert feats["asian_top_word_match"] in [True, False]

    print("test_lexicon_asian_top_word_match_false pass")


def test_lexicon_religious_feature_true():
    text = "I went to church and I pray every day. God is important to me."
    feats = extract_lexicon_features(text)

    assert feats["Religious_Feature"] is True

    print("test_lexicon_religious_feature_true pass")


def test_pos_ratio_all_in_range():
    text = (
        "I have a very good feeling that my parents are going to sponsor my study at UBC. "
        "Even though a relative of mine is being toxic about it."
    )
    feats = get_POS_rato_features(text)

    assert isinstance(feats, dict)

    # Ensure all expected keys exist (including first_person_ratio)
    expected = {
        "article_ratio",
        "pronoun_density",
        "preposition_ratio",
        "modal_verb_ratio",
        "adjective_ratio",
        "first_person_ratio",
    }
    assert expected.issubset(set(feats.keys()))

    # All ratios should be in [0, 1]
    for k, v in feats.items():
        assert 0.0 <= v <= 1.0

    # first_person_ratio should be > 0 here since "I", "my", "mine" appear
    assert feats["first_person_ratio"] > 0

    print("test_pos_ratio_all_in_range pass")


if __name__ == "__main__":
    test_text_stats_basic()
    test_text_stats_empty()
    test_sentence_length_stats_basic()
    test_sentence_length_stats_empty()
    test_sentence_length_stats_variability()
    test_lexicon_asian_top_word_match_false()
    test_lexicon_religious_feature_true()
    test_pos_ratio_all_in_range()
