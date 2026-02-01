"""
Unit tests for text comparison.
"""

import pytest
from seshat.comparator import Comparator, ComparisonResult


class TestComparator:
    """Tests for Comparator class."""

    def test_comparator_init(self):
        comparator = Comparator()
        assert comparator is not None

    def test_compare_texts_same(self):
        comparator = Comparator()
        text = "This is a test text with enough words for analysis."
        result = comparator.compare_texts(text, text)
        assert result["cosine_similarity"] > 0.9

    def test_compare_texts_different(self):
        comparator = Comparator()
        text1 = "The quick brown fox jumps over the lazy dog."
        text2 = "Technical documentation requires precise language and clarity."
        result = comparator.compare_texts(text1, text2)
        assert "cosine_similarity" in result

    def test_compare_to_profile(self, analyzer, sample_profile):
        comparator = Comparator(analyzer=analyzer)
        text = "This is a test text to compare against the profile."
        result = comparator.compare(text, sample_profile)
        assert isinstance(result, ComparisonResult)
        assert hasattr(result, "overall_score")
        assert hasattr(result, "confidence")

    def test_compare_multiple_profiles(self, profile_manager, analyzer):
        profile1 = profile_manager.create_profile(
            name="Author1",
            samples=["First author writes in a formal style consistently."],
        )
        profile2 = profile_manager.create_profile(
            name="Author2",
            samples=["Second author has a different writing approach entirely."],
        )

        comparator = Comparator(analyzer=analyzer)
        text = "Test text for multiple profile comparison."
        results = comparator.compare_multiple(text, [profile1, profile2])

        assert len(results) == 2
        assert results[0].overall_score >= results[1].overall_score


class TestComparisonResult:
    """Tests for ComparisonResult class."""

    def test_comparison_result_fields(self, analyzer, sample_profile):
        comparator = Comparator(analyzer=analyzer)
        text = "Test comparison text."
        result = comparator.compare(text, sample_profile)

        assert result.profile_name is not None
        assert result.overall_score >= 0
        assert result.confidence in ["Very High", "High", "Medium", "Low", "Very Low"]

    def test_comparison_result_to_dict(self, analyzer, sample_profile):
        comparator = Comparator(analyzer=analyzer)
        text = "Test comparison text."
        result = comparator.compare(text, sample_profile)
        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "overall_score" in result_dict
        assert "confidence" in result_dict
