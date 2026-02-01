"""
Unit tests for feature extraction modules.
"""

import pytest
from seshat.features.lexical import LexicalFeatureExtractor
from seshat.features.function_words import FunctionWordExtractor
from seshat.features.punctuation import PunctuationFeatureExtractor
from seshat.features.ngrams import NGramExtractor
from seshat.features.emoji import EmojiFeatureExtractor


class TestLexicalFeatures:
    """Tests for lexical feature extraction."""

    def test_extractor_init(self):
        extractor = LexicalFeatureExtractor()
        assert extractor is not None

    def test_extract_basic(self, sample_text):
        extractor = LexicalFeatureExtractor()
        features = extractor.extract(sample_text)
        assert isinstance(features, dict)
        assert len(features) > 0

    def test_type_token_ratio(self, sample_text):
        extractor = LexicalFeatureExtractor()
        features = extractor.extract(sample_text)
        assert "type_token_ratio" in features
        assert 0 <= features["type_token_ratio"] <= 1

    def test_avg_word_length(self, sample_text):
        extractor = LexicalFeatureExtractor()
        features = extractor.extract(sample_text)
        assert "avg_word_length" in features
        assert features["avg_word_length"] > 0

    def test_vocabulary_richness(self, sample_text):
        extractor = LexicalFeatureExtractor()
        features = extractor.extract(sample_text)
        assert "yules_k" in features


class TestFunctionWordFeatures:
    """Tests for function word extraction."""

    def test_extractor_init(self):
        extractor = FunctionWordExtractor()
        assert extractor is not None

    def test_extract_pronouns(self, sample_text):
        extractor = FunctionWordExtractor()
        features = extractor.extract(sample_text)
        assert "first_person_singular_ratio" in features
        assert "article_ratio" in features

    def test_extract_formal_text(self, formal_text):
        extractor = FunctionWordExtractor()
        features = extractor.extract(formal_text)
        assert "first_person_singular_ratio" in features


class TestPunctuationFeatures:
    """Tests for punctuation feature extraction."""

    def test_extractor_init(self):
        extractor = PunctuationFeatureExtractor()
        assert extractor is not None

    def test_extract_basic(self, sample_text):
        extractor = PunctuationFeatureExtractor()
        features = extractor.extract(sample_text)
        assert isinstance(features, dict)
        assert "comma_per_1k" in features

    def test_terminal_punctuation(self):
        text = "Hello! How are you? I'm fine."
        extractor = PunctuationFeatureExtractor()
        features = extractor.extract(text)
        assert "terminal_question_ratio" in features
        assert "terminal_exclamation_ratio" in features


class TestNGramFeatures:
    """Tests for n-gram feature extraction."""

    def test_extractor_init(self):
        extractor = NGramExtractor()
        assert extractor is not None

    def test_character_ngrams(self, sample_text):
        extractor = NGramExtractor()
        features = extractor.extract(sample_text)
        char_features = [k for k in features if k.startswith("char_")]
        assert len(char_features) > 0

    def test_word_ngrams(self, sample_text):
        extractor = NGramExtractor()
        features = extractor.extract(sample_text)
        word_features = [k for k in features if k.startswith("word_")]
        assert len(word_features) > 0


class TestEmojiFeatures:
    """Tests for emoji feature extraction."""

    def test_extractor_init(self):
        extractor = EmojiFeatureExtractor()
        assert extractor is not None

    def test_extract_no_emojis(self, formal_text):
        extractor = EmojiFeatureExtractor()
        features = extractor.extract(formal_text)
        assert features.get("emoji_count", 0) == 0

    def test_extract_with_emojis(self):
        text = "Hello! 😊 How are you? 🎉"
        extractor = EmojiFeatureExtractor()
        features = extractor.extract(text)
        assert features.get("emoji_count", 0) >= 2
