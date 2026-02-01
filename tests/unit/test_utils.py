"""
Unit tests for utility functions.
"""

import pytest
from seshat.utils import (
    tokenize_words,
    tokenize_sentences,
    normalize_text,
    count_syllables,
    compute_text_hash,
)


class TestTokenization:
    """Tests for tokenization functions."""

    def test_tokenize_words_basic(self):
        text = "Hello world, this is a test."
        words = tokenize_words(text)
        assert len(words) > 0
        assert "hello" in words or "Hello" in words

    def test_tokenize_words_empty(self):
        words = tokenize_words("")
        assert words == []

    def test_tokenize_words_punctuation(self):
        text = "Hello! How are you?"
        words = tokenize_words(text)
        assert "!" not in words
        assert "?" not in words

    def test_tokenize_sentences_basic(self):
        text = "First sentence. Second sentence! Third sentence?"
        sentences = tokenize_sentences(text)
        assert len(sentences) == 3

    def test_tokenize_sentences_empty(self):
        sentences = tokenize_sentences("")
        assert sentences == []

    def test_tokenize_sentences_no_period(self):
        text = "No period at the end"
        sentences = tokenize_sentences(text)
        assert len(sentences) == 1


class TestNormalization:
    """Tests for text normalization."""

    def test_normalize_lowercase(self):
        text = "HELLO World"
        normalized = normalize_text(text, lowercase=True)
        assert normalized == "hello world"

    def test_normalize_remove_punctuation(self):
        text = "Hello, world!"
        normalized = normalize_text(text, remove_punctuation=True)
        assert "," not in normalized
        assert "!" not in normalized

    def test_normalize_preserve_case(self):
        text = "HELLO World"
        normalized = normalize_text(text, lowercase=False)
        assert "HELLO" in normalized


class TestSyllableCount:
    """Tests for syllable counting."""

    def test_syllables_simple(self):
        assert count_syllables("hello") >= 2
        assert count_syllables("cat") == 1
        assert count_syllables("beautiful") >= 3

    def test_syllables_empty(self):
        assert count_syllables("") == 0

    def test_syllables_numbers(self):
        result = count_syllables("123")
        assert result >= 0


class TestHashing:
    """Tests for text hashing."""

    def test_hash_consistency(self):
        text = "Hello world"
        hash1 = compute_text_hash(text)
        hash2 = compute_text_hash(text)
        assert hash1 == hash2

    def test_hash_different_texts(self):
        hash1 = compute_text_hash("Hello")
        hash2 = compute_text_hash("World")
        assert hash1 != hash2

    def test_hash_length(self):
        text_hash = compute_text_hash("test")
        assert len(text_hash) == 64
