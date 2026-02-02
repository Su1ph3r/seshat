"""
Unit tests for personality disorder indicator analysis module.
"""

import pytest
from seshat.psychology.personality_disorders import PersonalityDisorderIndicators


class TestPersonalityDisorderIndicators:
    """Tests for personality disorder linguistic indicator analysis."""

    def test_analyzer_init(self):
        """Test analyzer initialization."""
        analyzer = PersonalityDisorderIndicators()
        assert analyzer is not None
        assert analyzer.indicator_words is not None
        assert len(analyzer.indicator_words) == 10  # 10 disorders

    def test_analyze_basic(self, sample_text):
        """Test basic analysis returns expected structure."""
        analyzer = PersonalityDisorderIndicators()
        result = analyzer.analyze(sample_text)

        assert isinstance(result, dict)
        assert "disclaimer" in result
        assert "text_adequacy" in result
        assert "disorders" in result
        assert "clusters" in result
        assert "validation" in result
        assert "confidence" in result
        assert "summary" in result

    def test_disclaimer_always_present(self, sample_text):
        """Test that disclaimer is always included."""
        analyzer = PersonalityDisorderIndicators()
        result = analyzer.analyze(sample_text)

        assert "disclaimer" in result
        assert "NOT clinical diagnoses" in result["disclaimer"]
        assert "linguistic correlations" in result["disclaimer"].lower()

    def test_all_disorders_present(self, sample_text):
        """Test that all 10 disorders are analyzed."""
        analyzer = PersonalityDisorderIndicators()
        result = analyzer.analyze(sample_text)

        expected_disorders = [
            "paranoid", "schizoid", "schizotypal",
            "antisocial", "borderline", "histrionic", "narcissistic",
            "avoidant", "dependent", "obsessive_compulsive"
        ]

        for disorder in expected_disorders:
            assert disorder in result["disorders"]
            assert "score" in result["disorders"][disorder]
            assert 0 <= result["disorders"][disorder]["score"] <= 1

    def test_all_clusters_present(self, sample_text):
        """Test that all 3 clusters are analyzed."""
        analyzer = PersonalityDisorderIndicators()
        result = analyzer.analyze(sample_text)

        expected_clusters = ["cluster_a", "cluster_b", "cluster_c"]

        for cluster in expected_clusters:
            assert cluster in result["clusters"]
            assert "score" in result["clusters"][cluster]
            assert "label" in result["clusters"][cluster]
            assert 0 <= result["clusters"][cluster]["score"] <= 1

    def test_cluster_labels(self, sample_text):
        """Test cluster labels are correct."""
        analyzer = PersonalityDisorderIndicators()
        result = analyzer.analyze(sample_text)

        assert result["clusters"]["cluster_a"]["label"] == "Odd/Eccentric"
        assert result["clusters"]["cluster_b"]["label"] == "Dramatic/Emotional"
        assert result["clusters"]["cluster_c"]["label"] == "Anxious/Fearful"

    def test_empty_text(self):
        """Test handling of empty text."""
        analyzer = PersonalityDisorderIndicators()
        result = analyzer.analyze("")

        assert result["text_adequacy"]["word_count"] == 0
        assert result["text_adequacy"]["is_sufficient"] is False
        assert result["confidence"] == "very_low"

    def test_short_text_warning(self, short_text):
        """Test that short text triggers adequacy warning."""
        analyzer = PersonalityDisorderIndicators()
        result = analyzer.analyze(short_text)

        assert result["text_adequacy"]["is_sufficient"] is False
        assert len(result["text_adequacy"]["limitations"]) > 0

    def test_text_adequacy_tiers(self):
        """Test text adequacy confidence tiers."""
        analyzer = PersonalityDisorderIndicators()

        # Very short text
        result = analyzer.analyze("word " * 50)
        assert result["text_adequacy"]["confidence_tier"] == "very_low"

        # Short text
        result = analyzer.analyze("word " * 300)
        assert result["text_adequacy"]["confidence_tier"] == "low"

        # Medium text
        result = analyzer.analyze("word " * 700)
        assert result["text_adequacy"]["confidence_tier"] == "medium"

        # Long text
        result = analyzer.analyze("word " * 1500)
        assert result["text_adequacy"]["confidence_tier"] == "high"

        # Very long text
        result = analyzer.analyze("word " * 2500)
        assert result["text_adequacy"]["confidence_tier"] == "very_high"

    def test_score_interpretation(self, sample_text):
        """Test that scores include interpretation labels."""
        analyzer = PersonalityDisorderIndicators()
        result = analyzer.analyze(sample_text)

        for disorder in result["disorders"].values():
            assert "interpretation" in disorder
            assert disorder["interpretation"] in [
                "minimal", "low", "moderate", "elevated", "high"
            ]

    def test_validation_structure(self, sample_text):
        """Test validation section structure."""
        analyzer = PersonalityDisorderIndicators()
        result = analyzer.analyze(sample_text)

        validation = result["validation"]
        assert "feature_coverage" in validation
        assert "is_consistent" in validation
        assert "flags" in validation
        assert isinstance(validation["flags"], list)
        assert 0 <= validation["feature_coverage"] <= 1

    def test_paranoid_markers(self):
        """Test detection of paranoid markers."""
        paranoid_text = """
        They are always watching me, plotting and scheming against me.
        I can't trust anyone because everyone is suspicious and deceitful.
        They blame me for everything but it's their fault, not mine.
        I have to be vigilant and careful because they're always targeting me.
        Everyone is lying and conspiring against me. They are so untrustworthy.
        """ * 20  # Repeat to meet word count

        analyzer = PersonalityDisorderIndicators()
        result = analyzer.analyze(paranoid_text)

        paranoid_score = result["disorders"]["paranoid"]["score"]
        assert paranoid_score > 0.1  # Should detect some markers

    def test_narcissistic_markers(self):
        """Test detection of narcissistic markers."""
        narcissistic_text = """
        I am the best and most brilliant person in this room. I deserve special
        treatment because I am superior to everyone else. My achievements are
        extraordinary and unmatched. I expect others to recognize my greatness.
        These trivial concerns are beneath me. I am talented and gifted beyond
        compare. Everyone should admire my exceptional abilities.
        """ * 20

        analyzer = PersonalityDisorderIndicators()
        result = analyzer.analyze(narcissistic_text)

        narcissistic_score = result["disorders"]["narcissistic"]["score"]
        assert narcissistic_score > 0.1

    def test_borderline_markers(self):
        """Test detection of borderline markers."""
        borderline_text = """
        My emotions are completely out of control and overwhelming. I hate you,
        no wait, I love you - you're perfect. Don't leave me, I need you, I can't
        be alone. Everything is either wonderful or terrible, nothing in between.
        I feel empty and lost, don't know who I am anymore. Please stay with me,
        I'm falling apart without you. My mood is so unstable and chaotic.
        """ * 20

        analyzer = PersonalityDisorderIndicators()
        result = analyzer.analyze(borderline_text)

        borderline_score = result["disorders"]["borderline"]["score"]
        assert borderline_score > 0.1

    def test_avoidant_markers(self):
        """Test detection of avoidant markers."""
        avoidant_text = """
        I'm too shy and awkward to go to that party. I'm afraid of what people
        will think of me - they'll probably criticize and reject me. I feel
        inadequate and inferior to everyone else. I'm nervous and anxious about
        being judged. I'd rather be alone than face the embarrassment and
        humiliation of social rejection. I'm not good enough.
        """ * 20

        analyzer = PersonalityDisorderIndicators()
        result = analyzer.analyze(avoidant_text)

        avoidant_score = result["disorders"]["avoidant"]["score"]
        assert avoidant_score > 0.1

    def test_forensic_mode(self, sample_text):
        """Test forensic mode returns additional metadata."""
        analyzer = PersonalityDisorderIndicators()
        result = analyzer.analyze_forensic(sample_text, case_id="TEST-001")

        assert "forensic_metadata" in result
        meta = result["forensic_metadata"]
        assert meta["case_id"] == "TEST-001"
        assert "text_hash" in meta
        assert len(meta["text_hash"]) == 64  # SHA-256 hex
        assert "analyzed_at" in meta
        assert "analyzer_version" in meta
        assert "limitations" in meta

    def test_forensic_hash_consistency(self):
        """Test that forensic hash is consistent for same text."""
        analyzer = PersonalityDisorderIndicators()
        text = "This is a test text for hashing consistency."

        result1 = analyzer.analyze_forensic(text)
        result2 = analyzer.analyze_forensic(text)

        assert result1["forensic_metadata"]["text_hash"] == result2["forensic_metadata"]["text_hash"]

    def test_forensic_hash_difference(self):
        """Test that different texts produce different hashes."""
        analyzer = PersonalityDisorderIndicators()

        result1 = analyzer.analyze_forensic("Text one for testing.")
        result2 = analyzer.analyze_forensic("Text two for testing.")

        assert result1["forensic_metadata"]["text_hash"] != result2["forensic_metadata"]["text_hash"]

    def test_indicator_summary(self, sample_text):
        """Test simplified indicator summary."""
        analyzer = PersonalityDisorderIndicators()
        summary = analyzer.get_indicator_summary(sample_text)

        assert isinstance(summary, dict)
        # Check disorder markers
        assert "paranoid_markers" in summary
        assert "narcissistic_markers" in summary
        # Check cluster scores
        assert "cluster_a_score" in summary
        assert "cluster_b_score" in summary
        assert "cluster_c_score" in summary

    def test_dimension_ratios_present(self, sample_text):
        """Test that dimension ratios are included for each disorder."""
        analyzer = PersonalityDisorderIndicators()
        result = analyzer.analyze(sample_text)

        # Check paranoid has all expected ratios
        paranoid = result["disorders"]["paranoid"]
        assert "suspicion_ratio" in paranoid
        assert "mistrust_ratio" in paranoid
        assert "blame_external_ratio" in paranoid
        assert "hypervigilance_ratio" in paranoid

        # Check borderline has all expected ratios
        borderline = result["disorders"]["borderline"]
        assert "emotional_instability_ratio" in borderline
        assert "abandonment_fear_ratio" in borderline
        assert "splitting_ratio" in borderline

    def test_consistency_flags_contradictory_patterns(self):
        """Test that contradictory patterns are flagged."""
        # Text with both schizoid (detachment) and histrionic (attention-seeking) markers
        contradictory_text = """
        I prefer to be alone and detached from everyone. I don't care about
        social connections at all. I'm indifferent to what others think.
        But also look at me! Notice me! I need attention and admiration!
        Everyone should focus on me! This is dramatic and incredible!
        I'm withdrawn but also need the spotlight constantly.
        """ * 50  # Repeat to get high scores

        analyzer = PersonalityDisorderIndicators()
        result = analyzer.analyze(contradictory_text)

        # If both scores are elevated, should flag inconsistency
        schizoid_score = result["disorders"]["schizoid"]["score"]
        histrionic_score = result["disorders"]["histrionic"]["score"]

        if schizoid_score > 0.5 and histrionic_score > 0.5:
            assert not result["validation"]["is_consistent"]
            assert any("contradictory" in flag.lower() for flag in result["validation"]["flags"])

    def test_scores_clamped_to_range(self):
        """Test that all scores are clamped to 0-1 range."""
        # Text designed to potentially inflate scores
        extreme_text = """
        suspicious suspect watching spying plotting scheming conspiring
        deceiving betraying lying manipulating hiding secret secretive
        """ * 100

        analyzer = PersonalityDisorderIndicators()
        result = analyzer.analyze(extreme_text)

        for disorder_data in result["disorders"].values():
            assert 0 <= disorder_data["score"] <= 1

        for cluster_data in result["clusters"].values():
            assert 0 <= cluster_data["score"] <= 1

    def test_neutral_text_low_scores(self, formal_text):
        """Test that neutral/formal text produces relatively low scores."""
        analyzer = PersonalityDisorderIndicators()
        result = analyzer.analyze(formal_text)

        # Most scores should be relatively low for neutral business text
        high_scores = sum(
            1 for d in result["disorders"].values()
            if d["score"] > 0.5
        )
        # Should not have many elevated scores
        assert high_scores <= 2


class TestPersonalityDisorderIndicatorsEdgeCases:
    """Edge case tests for personality disorder indicators."""

    def test_none_text(self):
        """Test handling of None-like text."""
        analyzer = PersonalityDisorderIndicators()
        result = analyzer.analyze("")
        assert result["text_adequacy"]["word_count"] == 0

    def test_whitespace_only(self):
        """Test handling of whitespace-only text."""
        analyzer = PersonalityDisorderIndicators()
        result = analyzer.analyze("   \n\t   ")
        assert result["text_adequacy"]["word_count"] == 0

    def test_single_word(self):
        """Test handling of single word."""
        analyzer = PersonalityDisorderIndicators()
        result = analyzer.analyze("suspicious")
        assert result["text_adequacy"]["is_sufficient"] is False

    def test_special_characters_only(self):
        """Test handling of special characters only."""
        analyzer = PersonalityDisorderIndicators()
        result = analyzer.analyze("!!! ??? ### $$$")
        assert result["text_adequacy"]["word_count"] == 0

    def test_unicode_text(self):
        """Test handling of unicode text."""
        analyzer = PersonalityDisorderIndicators()
        unicode_text = "I feel suspicious about their motives. 日本語テキスト. Émotions fortes."
        result = analyzer.analyze(unicode_text)
        assert result is not None
        assert "disorders" in result

    def test_mixed_case_detection(self):
        """Test that word detection is case-insensitive."""
        analyzer = PersonalityDisorderIndicators()

        text1 = "SUSPICIOUS SUSPICIOUS SUSPICIOUS " * 50
        text2 = "suspicious suspicious suspicious " * 50

        result1 = analyzer.analyze(text1)
        result2 = analyzer.analyze(text2)

        # Scores should be identical for same words different cases
        assert abs(
            result1["disorders"]["paranoid"]["suspicion_ratio"] -
            result2["disorders"]["paranoid"]["suspicion_ratio"]
        ) < 0.01

    def test_forensic_mode_without_case_id(self, sample_text):
        """Test forensic mode works without case_id."""
        analyzer = PersonalityDisorderIndicators()
        result = analyzer.analyze_forensic(sample_text)

        assert "forensic_metadata" in result
        assert result["forensic_metadata"]["case_id"] is None

    def test_minimum_word_count_constant(self):
        """Test that MINIMUM_WORD_COUNT is defined."""
        analyzer = PersonalityDisorderIndicators()
        assert hasattr(analyzer, 'MINIMUM_WORD_COUNT')
        assert analyzer.MINIMUM_WORD_COUNT == 500

    def test_score_interpretation_boundaries(self):
        """Test score interpretation at boundaries."""
        analyzer = PersonalityDisorderIndicators()

        assert analyzer._interpret_score(0.0) == "minimal"
        assert analyzer._interpret_score(0.14) == "minimal"
        assert analyzer._interpret_score(0.15) == "low"
        assert analyzer._interpret_score(0.29) == "low"
        assert analyzer._interpret_score(0.30) == "moderate"
        assert analyzer._interpret_score(0.49) == "moderate"
        assert analyzer._interpret_score(0.50) == "elevated"
        assert analyzer._interpret_score(0.69) == "elevated"
        assert analyzer._interpret_score(0.70) == "high"
        assert analyzer._interpret_score(0.99) == "high"
        assert analyzer._interpret_score(1.0) == "high"  # Edge case: exactly 1.0
