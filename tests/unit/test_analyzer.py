"""
Unit tests for the Analyzer class.
"""

import pytest
from seshat.analyzer import Analyzer, AnalysisResult


class TestAnalyzer:
    """Tests for Analyzer class."""

    def test_analyzer_init(self):
        analyzer = Analyzer()
        assert analyzer is not None

    def test_analyze_basic(self, analyzer, sample_text):
        result = analyzer.analyze(sample_text)
        assert isinstance(result, AnalysisResult)
        assert result.word_count > 0
        assert result.sentence_count > 0

    def test_analyze_short_text(self, analyzer, short_text):
        result = analyzer.analyze(short_text)
        assert result.word_count > 0

    def test_analyze_empty_text(self, analyzer, empty_text):
        with pytest.raises(ValueError):
            analyzer.analyze(empty_text)

    def test_analysis_result_features(self, analyzer, sample_text):
        result = analyzer.analyze(sample_text)
        assert result.lexical_features is not None
        assert result.function_word_features is not None
        assert result.punctuation_features is not None
        assert result.syntactic_features is not None

    def test_analysis_result_to_dict(self, analyzer, sample_text):
        result = analyzer.analyze(sample_text)
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert "word_count" in result_dict
        assert "all_features" in result_dict

    def test_lexical_features_present(self, analyzer, sample_text):
        result = analyzer.analyze(sample_text)
        assert "type_token_ratio" in result.lexical_features
        assert "avg_word_length" in result.lexical_features

    def test_function_word_features_present(self, analyzer, sample_text):
        result = analyzer.analyze(sample_text)
        assert len(result.function_word_features) > 0

    def test_analyze_formal_vs_informal(self, analyzer, formal_text, informal_text):
        formal_result = analyzer.analyze(formal_text)
        informal_result = analyzer.analyze(informal_text)

        assert formal_result.word_count > 0
        assert informal_result.word_count > 0


class TestAnalysisResult:
    """Tests for AnalysisResult class."""

    def test_all_features_combined(self, analyzer, sample_text):
        result = analyzer.analyze(sample_text)
        all_features = result.all_features
        assert isinstance(all_features, dict)
        assert len(all_features) > 0

    def test_to_json(self, analyzer, sample_text):
        result = analyzer.analyze(sample_text)
        json_str = result.to_json()
        assert isinstance(json_str, str)
        assert "word_count" in json_str
