"""
Integration tests for full analysis pipeline.
"""

import pytest
from pathlib import Path


class TestFullPipeline:
    """Test complete analysis workflows."""

    def test_profile_creation_to_comparison(self, profile_manager, analyzer):
        """Test creating profiles and comparing text."""
        samples1 = [
            "I believe that education is fundamental to success. "
            "Throughout my career, I have observed that dedicated individuals "
            "consistently achieve their goals through persistent effort.",
            "Furthermore, it is essential to maintain a structured approach "
            "when tackling complex problems. One must analyze the situation "
            "carefully before proceeding.",
        ]

        samples2 = [
            "hey so i was thinking about this stuff and like honestly "
            "its not that big of a deal lol. we should just chill and "
            "see what happens tbh.",
            "anyway wanna grab some food later? theres this new place "
            "that just opened up and everyone says its amazing!!",
        ]

        profile1 = profile_manager.create_profile(
            name="FormalWriter",
            samples=samples1,
        )

        profile2 = profile_manager.create_profile(
            name="InformalWriter",
            samples=samples2,
        )

        from seshat.comparator import Comparator

        comparator = Comparator(analyzer=analyzer)

        formal_test = (
            "I would like to express my appreciation for your consideration. "
            "The methodology employed demonstrates rigorous analytical thinking."
        )

        results = comparator.compare_multiple(
            formal_test,
            [profile1, profile2],
        )

        assert len(results) == 2
        assert results[0].profile_name == "FormalWriter" or results[0].overall_score > results[1].overall_score

    def test_psychological_analysis_integration(self, sample_text):
        """Test full psychological profiling."""
        from seshat.psychology.personality import PersonalityAnalyzer
        from seshat.psychology.emotional import EmotionalAnalyzer
        from seshat.psychology.cognitive import CognitiveAnalyzer

        personality = PersonalityAnalyzer().analyze(sample_text)
        emotional = EmotionalAnalyzer().analyze(sample_text)
        cognitive = CognitiveAnalyzer().analyze(sample_text)

        assert "openness" in personality
        assert "sentiment_label" in emotional
        assert "analytical_score" in cognitive

    def test_ai_detection_integration(self, sample_text):
        """Test AI detection module."""
        from seshat.advanced.ai_detection import AIDetector

        detector = AIDetector()
        result = detector.detect(sample_text)

        assert "classification" in result
        assert "ai_probability" in result
        assert 0 <= result["ai_probability"] <= 1

    def test_cross_platform_analysis(self, profile_manager, analyzer):
        """Test cross-platform style consistency analysis."""
        from seshat.advanced.cross_platform import CrossPlatformAnalyzer

        twitter_samples = [
            "Just finished this amazing book! Highly recommend it #reading",
            "Can't believe it's already Monday again... #mondayblues",
            "The weather is perfect today! Going for a long walk #sunshine",
        ]

        email_samples = [
            "Hi team, I wanted to follow up on our discussion from yesterday. "
            "I believe we should proceed with the proposed timeline.",
            "Thank you for your email. I have reviewed the documents and "
            "have a few suggestions for improvement.",
        ]

        cpa = CrossPlatformAnalyzer(analyzer=analyzer)

        result = cpa.compare_platforms({
            "twitter": twitter_samples,
            "email": email_samples,
        })

        assert "platforms_analyzed" in result
        assert "overall_consistency" in result
        assert "same_author_assessment" in result


class TestExportImport:
    """Test profile export and import functionality."""

    def test_export_import_profile(self, sample_profile, tmp_path):
        """Test exporting and reimporting a profile."""
        from seshat.io.exporters import ProfileExporter
        from seshat.profile import AuthorProfile

        export_path = tmp_path / "test_profile.json"

        exporter = ProfileExporter()
        exporter.export_json(sample_profile, str(export_path))

        assert export_path.exists()

        imported = AuthorProfile.from_file(str(export_path))

        assert imported.name == sample_profile.name
        assert len(imported.samples) == len(sample_profile.samples)
