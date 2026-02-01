"""
Unit tests for psychological analysis modules.
"""

import pytest
from seshat.psychology.personality import PersonalityAnalyzer
from seshat.psychology.emotional import EmotionalAnalyzer
from seshat.psychology.cognitive import CognitiveAnalyzer


class TestPersonalityAnalyzer:
    """Tests for Big Five personality analysis."""

    def test_analyzer_init(self):
        analyzer = PersonalityAnalyzer()
        assert analyzer is not None

    def test_analyze_basic(self, sample_text):
        analyzer = PersonalityAnalyzer()
        result = analyzer.analyze(sample_text)
        assert isinstance(result, dict)

    def test_big_five_traits(self, sample_text):
        analyzer = PersonalityAnalyzer()
        result = analyzer.analyze(sample_text)
        traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
        for trait in traits:
            assert trait in result
            assert "score" in result[trait]
            assert 0 <= result[trait]["score"] <= 1

    def test_formal_vs_informal(self, formal_text, informal_text):
        analyzer = PersonalityAnalyzer()
        formal_result = analyzer.analyze(formal_text)
        informal_result = analyzer.analyze(informal_text)
        assert formal_result != informal_result


class TestEmotionalAnalyzer:
    """Tests for emotional tone analysis."""

    def test_analyzer_init(self):
        analyzer = EmotionalAnalyzer()
        assert analyzer is not None

    def test_analyze_basic(self, sample_text):
        analyzer = EmotionalAnalyzer()
        result = analyzer.analyze(sample_text)
        assert isinstance(result, dict)

    def test_sentiment_label(self, sample_text):
        analyzer = EmotionalAnalyzer()
        result = analyzer.analyze(sample_text)
        assert "sentiment_label" in result
        assert result["sentiment_label"] in ["positive", "negative", "neutral"]

    def test_emotional_intensity(self, sample_text):
        analyzer = EmotionalAnalyzer()
        result = analyzer.analyze(sample_text)
        assert "emotional_intensity" in result
        assert 0 <= result["emotional_intensity"] <= 1

    def test_positive_text(self):
        text = "I'm so happy and excited! This is wonderful news! Amazing!"
        analyzer = EmotionalAnalyzer()
        result = analyzer.analyze(text)
        assert result["sentiment_score"] > 0

    def test_negative_text(self):
        text = "This is terrible and disappointing. I'm very upset and angry."
        analyzer = EmotionalAnalyzer()
        result = analyzer.analyze(text)
        assert result["sentiment_score"] < 0


class TestCognitiveAnalyzer:
    """Tests for cognitive style analysis."""

    def test_analyzer_init(self):
        analyzer = CognitiveAnalyzer()
        assert analyzer is not None

    def test_analyze_basic(self, sample_text):
        analyzer = CognitiveAnalyzer()
        result = analyzer.analyze(sample_text)
        assert isinstance(result, dict)

    def test_analytical_score(self, sample_text):
        analyzer = CognitiveAnalyzer()
        result = analyzer.analyze(sample_text)
        assert "analytical_score" in result
        assert 0 <= result["analytical_score"] <= 1

    def test_cognitive_complexity(self, sample_text):
        analyzer = CognitiveAnalyzer()
        result = analyzer.analyze(sample_text)
        assert "cognitive_complexity" in result

    def test_time_orientation(self, sample_text):
        analyzer = CognitiveAnalyzer()
        result = analyzer.analyze(sample_text)
        assert "time_orientation" in result
