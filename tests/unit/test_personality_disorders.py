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


# =============================================================================
# Tests for v2.0 Enhanced Layers
# =============================================================================


class TestPDLinguisticLayer:
    """Tests for the linguistic analysis layer."""

    def test_phrase_detection_paranoid(self):
        """Test detection of multi-word paranoid phrases."""
        from seshat.psychology.pd_linguistic import PDLinguisticLayer

        layer = PDLinguisticLayer(use_spacy=False)
        text = "They are out to get me and plotting against me behind my back."

        matches = layer.detect_phrases(text)

        assert "paranoid" in matches
        phrases_found = [m.phrase for m in matches["paranoid"]]
        assert any("out to get me" in p for p in phrases_found)

    def test_phrase_detection_borderline(self):
        """Test detection of multi-word borderline phrases."""
        from seshat.psychology.pd_linguistic import PDLinguisticLayer

        layer = PDLinguisticLayer(use_spacy=False)
        text = "Please don't leave me. I can't live without you. My emotions are out of control."

        matches = layer.detect_phrases(text)

        assert "borderline" in matches
        phrases_found = [m.phrase for m in matches["borderline"]]
        assert len(phrases_found) >= 1

    def test_negation_handling(self):
        """Test that negated markers are detected."""
        from seshat.psychology.pd_linguistic import PDLinguisticLayer

        layer = PDLinguisticLayer(use_spacy=False)
        text = "I am not suspicious of anyone. I don't distrust people."

        # Test negation detection
        indicator_words = {
            "paranoid": {
                "suspicion": ["suspicious"],
                "mistrust": ["distrust"],
            }
        }

        negated, adjustments = layer.handle_negation(text, indicator_words)

        assert len(negated) >= 1
        assert any(n.marker == "suspicious" for n in negated)

    def test_context_windows(self):
        """Test context window extraction."""
        from seshat.psychology.pd_linguistic import PDLinguisticLayer

        layer = PDLinguisticLayer(use_spacy=False)
        text = "I feel very suspicious about their motives and intentions."

        indicator_words = {
            "paranoid": {
                "suspicion": ["suspicious"],
            }
        }

        windows = layer.extract_context_windows(text, indicator_words, window_size=3)

        assert "paranoid" in windows
        assert len(windows["paranoid"]) >= 1
        assert windows["paranoid"][0].marker == "suspicious"

    def test_syntactic_patterns_basic(self):
        """Test basic syntactic pattern analysis."""
        from seshat.psychology.pd_linguistic import PDLinguisticLayer

        layer = PDLinguisticLayer(use_spacy=False)
        text = "I am suspicious. They are watching me? This is terrible!"

        patterns = layer.analyze_syntactic_patterns(text)

        assert patterns.question_ratio > 0
        assert patterns.exclamation_ratio > 0
        assert patterns.first_person_ratio > 0

    def test_phrase_score_boost(self):
        """Test that phrase matches produce score boosts."""
        from seshat.psychology.pd_linguistic import PDLinguisticLayer

        layer = PDLinguisticLayer(use_spacy=False)
        text = "They are out to get me. I can't trust anyone. They're watching me."

        matches = layer.detect_phrases(text)
        boosts = layer.get_phrase_score_boost(matches)

        assert "paranoid" in boosts
        # Should have some boost for paranoid dimensions
        total_boost = sum(boosts["paranoid"].values())
        assert total_boost > 0


class TestPDCalibrationLayer:
    """Tests for the calibration layer."""

    def test_normalize_score(self):
        """Test score normalization."""
        from seshat.psychology.pd_calibration import PDCalibrationLayer

        layer = PDCalibrationLayer()

        # A raw score equal to the mean should normalize to around 0.5
        normalized = layer.normalize_score(0.15, "paranoid")
        assert 0.4 < normalized < 0.6

        # A high raw score should normalize higher
        high_normalized = layer.normalize_score(0.40, "paranoid")
        assert high_normalized > normalized

    def test_genre_detection_formal(self):
        """Test detection of formal genre."""
        from seshat.psychology.pd_calibration import PDCalibrationLayer

        layer = PDCalibrationLayer()
        formal_text = """
        Dear Sir or Madam, I am writing to express my concerns. Furthermore,
        I believe that notwithstanding the circumstances, we should proceed.
        Yours faithfully, John Smith.
        """

        result = layer.detect_genre(formal_text)

        assert result.genre == "formal"
        assert result.confidence > 0

    def test_genre_detection_informal(self):
        """Test detection of informal genre."""
        from seshat.psychology.pd_calibration import PDCalibrationLayer

        layer = PDCalibrationLayer()
        informal_text = """
        hey!! gonna go out later, wanna come? lol it's gonna be lit. btw
        did u see that meme? omg so funny haha
        """

        result = layer.detect_genre(informal_text)

        assert result.genre == "informal"

    def test_genre_adjustment(self):
        """Test genre-based score adjustment."""
        from seshat.psychology.pd_calibration import PDCalibrationLayer

        layer = PDCalibrationLayer()
        scores = {"histrionic": 0.5, "schizoid": 0.3}

        # Formal text should reduce histrionic signals
        adjusted = layer.adjust_for_genre(scores, "formal")
        assert adjusted["histrionic"] <= scores["histrionic"]

    def test_confidence_calibration(self):
        """Test confidence calibration."""
        from seshat.psychology.pd_calibration import PDCalibrationLayer

        layer = PDCalibrationLayer()
        scores = {"paranoid": 0.3, "schizoid": 0.2}
        validation = {"is_consistent": True, "feature_coverage": 0.5, "flags": []}

        result = layer.calibrate_confidence(scores, validation, word_count=1000)

        assert result.level in ["very_low", "low", "medium", "high", "very_high"]
        assert 0 <= result.score <= 1
        assert "text_length" in result.factors

    def test_z_score_calculation(self):
        """Test z-score calculation."""
        from seshat.psychology.pd_calibration import PDCalibrationLayer

        layer = PDCalibrationLayer()

        # Score at the mean should have z-score of 0
        z = layer.calculate_z_score(0.15, "paranoid")  # mean is 0.15
        assert abs(z) < 0.1

        # Score above mean should have positive z-score
        z_high = layer.calculate_z_score(0.35, "paranoid")
        assert z_high > 0


class TestPDValidationLayer:
    """Tests for the validation layer."""

    def test_discriminant_validity_contradictions(self):
        """Test detection of contradictory patterns."""
        from seshat.psychology.pd_validation import PDValidationLayer

        layer = PDValidationLayer()

        # Schizoid and histrionic are contradictory
        scores = {
            "schizoid": 0.6,
            "histrionic": 0.6,
            "paranoid": 0.1,
            "antisocial": 0.1,
            "borderline": 0.1,
            "narcissistic": 0.1,
            "avoidant": 0.1,
            "dependent": 0.1,
            "schizotypal": 0.1,
            "obsessive_compulsive": 0.1,
        }

        result = layer.check_discriminant_validity(scores)

        assert not result.is_valid
        assert len(result.contradictions) >= 1
        assert len(result.warnings) >= 1

    def test_minimum_markers_check(self):
        """Test minimum marker requirement checking."""
        from seshat.psychology.pd_validation import PDValidationLayer

        layer = PDValidationLayer()

        # Should pass with enough markers
        meets, explanation = layer.check_minimum_markers("paranoid", 5)
        assert meets is True

        # Should fail with too few markers
        meets, explanation = layer.check_minimum_markers("paranoid", 1)
        assert meets is False
        assert "Below minimum" in explanation

    def test_circumplex_mapping(self):
        """Test interpersonal circumplex mapping."""
        from seshat.psychology.pd_validation import PDValidationLayer

        layer = PDValidationLayer()

        # Narcissistic should map to dominant
        scores = {
            "narcissistic": 0.8,
            "paranoid": 0.1,
            "schizoid": 0.1,
            "schizotypal": 0.1,
            "antisocial": 0.1,
            "borderline": 0.1,
            "histrionic": 0.1,
            "avoidant": 0.1,
            "dependent": 0.1,
            "obsessive_compulsive": 0.1,
        }

        position = layer.map_to_circumplex(scores)

        assert position.dominance > 0  # Narcissistic is dominant
        assert -1 <= position.dominance <= 1
        assert -1 <= position.affiliation <= 1
        assert position.quadrant in [
            "dominant-hostile", "dominant-friendly",
            "submissive-hostile", "submissive-friendly"
        ]

    def test_validation_flags_generation(self):
        """Test validation flag generation."""
        from seshat.psychology.pd_validation import PDValidationLayer

        layer = PDValidationLayer()
        scores = {"paranoid": 0.9, "schizoid": 0.1}

        flags = layer.generate_validation_flags(scores, word_count=50)

        assert flags.severity != "none"  # Short text should trigger flags
        assert len(flags.flags) > 0

    def test_profile_clarity(self):
        """Test profile clarity calculation."""
        from seshat.psychology.pd_validation import PDValidationLayer

        layer = PDValidationLayer()

        # Clear profile: one high score
        clear_scores = {
            "paranoid": 0.8,
            "schizoid": 0.1,
            "narcissistic": 0.1,
        }
        clarity = layer.calculate_profile_clarity(clear_scores)
        assert clarity["clarity_label"] == "clear"

        # Unclear profile: many moderate scores
        unclear_scores = {
            "paranoid": 0.5,
            "schizoid": 0.5,
            "narcissistic": 0.5,
            "borderline": 0.5,
            "avoidant": 0.5,
        }
        clarity2 = layer.calculate_profile_clarity(unclear_scores)
        assert clarity2["clarity_score"] < clarity["clarity_score"]


class TestPDAdvancedMetrics:
    """Tests for advanced metrics layer."""

    def test_temporal_patterns_past_focus(self):
        """Test detection of past temporal focus."""
        from seshat.psychology.pd_advanced_metrics import PDAdvancedMetrics

        metrics = PDAdvancedMetrics()
        past_text = """
        I was always worried back then. When I was young, I used to feel afraid.
        Years ago, I had these problems. Previously, I would always overthink.
        """

        profile = metrics.analyze_temporal_patterns(past_text)

        assert profile.past_focus > profile.future_focus
        assert profile.dominant_focus == "past"

    def test_temporal_patterns_future_focus(self):
        """Test detection of future temporal focus."""
        from seshat.psychology.pd_advanced_metrics import PDAdvancedMetrics

        metrics = PDAdvancedMetrics()
        future_text = """
        I will succeed tomorrow. I'm going to achieve my goals next year.
        Eventually, I plan to become the best. Soon I'll show everyone.
        """

        profile = metrics.analyze_temporal_patterns(future_text)

        assert profile.future_focus > profile.past_focus
        assert profile.dominant_focus == "future"

    def test_linguistic_complexity(self):
        """Test linguistic complexity metrics."""
        from seshat.psychology.pd_advanced_metrics import PDAdvancedMetrics

        metrics = PDAdvancedMetrics()

        complex_text = """
        Notwithstanding the aforementioned circumstances, the paradigmatic
        juxtaposition of multifaceted considerations necessitates a nuanced
        approach to ameliorating the exacerbated situation.
        """

        simple_text = """
        The thing is good. I like stuff. It was nice. Very good things happened.
        """

        complex_result = metrics.analyze_linguistic_complexity(complex_text)
        simple_result = metrics.analyze_linguistic_complexity(simple_text)

        assert complex_result.vocabulary_sophistication > simple_result.vocabulary_sophistication

    def test_response_style_hedging(self):
        """Test hedging detection in response style."""
        from seshat.psychology.pd_advanced_metrics import PDAdvancedMetrics

        metrics = PDAdvancedMetrics()
        hedging_text = """
        Maybe I could possibly do that. Perhaps it might be somewhat okay.
        I think it could sort of work. I guess it's fairly reasonable.
        """

        result = metrics.analyze_response_style(hedging_text)

        assert result.hedging_ratio > 0.1

    def test_response_style_absolutism(self):
        """Test absolutism detection in response style."""
        from seshat.psychology.pd_advanced_metrics import PDAdvancedMetrics

        metrics = PDAdvancedMetrics()
        absolute_text = """
        Everything is always terrible. Everyone never understands. Nothing
        ever works. I completely hate absolutely everything. It's definitely
        the worst thing ever. Totally impossible.
        """

        result = metrics.analyze_response_style(absolute_text)

        assert result.absolutism_ratio > 0.1


class TestPDTemporalAnalyzer:
    """Tests for temporal series analyzer."""

    def test_trend_detection_increasing(self):
        """Test detection of increasing trend."""
        from seshat.psychology.pd_temporal import PDTemporalAnalyzer

        analyzer = PDTemporalAnalyzer()
        scores = [0.2, 0.3, 0.4, 0.5, 0.6]

        trend = analyzer.detect_trend(scores)

        assert trend.direction == "increasing"
        assert trend.slope > 0
        assert trend.change_percent > 0

    def test_trend_detection_stable(self):
        """Test detection of stable trend."""
        from seshat.psychology.pd_temporal import PDTemporalAnalyzer

        analyzer = PDTemporalAnalyzer()
        scores = [0.3, 0.31, 0.29, 0.30, 0.31]

        trend = analyzer.detect_trend(scores)

        assert trend.direction == "stable"
        assert abs(trend.slope) < 0.02

    def test_change_point_detection(self):
        """Test significant change point detection."""
        from seshat.psychology.pd_temporal import PDTemporalAnalyzer

        analyzer = PDTemporalAnalyzer()
        # Sudden increase at index 3
        scores = [0.2, 0.2, 0.2, 0.6, 0.6, 0.6]

        changes = analyzer.identify_significant_changes(scores, disorder="paranoid")

        assert len(changes) >= 1
        assert changes[0].direction == "increase"

    def test_series_analysis(self):
        """Test full series analysis."""
        from seshat.psychology.pd_temporal import PDTemporalAnalyzer

        analyzer = PDTemporalAnalyzer()

        # Create mock analysis results
        results = [
            {"disorders": {d: {"score": 0.2 + i * 0.05} for d in [
                "paranoid", "schizoid", "schizotypal", "antisocial",
                "borderline", "histrionic", "narcissistic", "avoidant",
                "dependent", "obsessive_compulsive"
            ]}}
            for i in range(5)
        ]

        analysis = analyzer.analyze_series(results)

        assert analysis.sample_count == 5
        assert "paranoid" in analysis.disorder_trends
        assert 0 <= analysis.stability_score <= 1


class TestEnhancedIntegration:
    """Integration tests for enhanced personality disorder analysis."""

    def test_enhanced_analysis_structure(self):
        """Test that enhanced analysis returns all expected fields."""
        analyzer = PersonalityDisorderIndicators()
        text = """
        I always feel like they are watching me and plotting against me.
        Everyone is out to get me. I can't trust anyone because they're all
        lying and deceiving me. They talk about me behind my back constantly.
        """ * 20

        result = analyzer.analyze(text)

        # Check for enhanced fields
        assert "phrase_matches" in result
        assert "negated_markers" in result
        assert "context_windows" in result
        assert "syntactic_patterns" in result
        assert "genre" in result
        assert "normalized_scores" in result
        assert "discriminant_validity" in result
        assert "circumplex_position" in result
        assert "temporal_patterns" in result
        assert "linguistic_complexity" in result
        assert "response_style" in result
        assert "calibrated_confidence" in result

    def test_feature_flags_disable(self):
        """Test that feature flags can disable layers."""
        analyzer = PersonalityDisorderIndicators(
            use_phrases=False,
            use_negation=False,
            use_context=False,
            use_circumplex=False,
        )
        text = "They are always watching me suspiciously." * 20

        result = analyzer.analyze(text)

        # With flags disabled, these shouldn't be present
        assert "phrase_matches" not in result
        assert "context_windows" not in result
        assert "circumplex_position" not in result

    def test_compare_method(self):
        """Test text comparison method."""
        analyzer = PersonalityDisorderIndicators()

        text1 = "I feel happy and content with life. Everything is good." * 20
        text2 = "I am suspicious and distrustful of everyone around me." * 20

        comparison = analyzer.compare(text1, text2)

        assert "disorders" in comparison
        assert "overall_change" in comparison
        assert "summary" in comparison

    def test_analyze_series_method(self):
        """Test series analysis method."""
        analyzer = PersonalityDisorderIndicators()

        texts = [
            "I feel fine today. Life is normal." * 20,
            "I'm becoming more suspicious of people." * 20,
            "Everyone is definitely watching me now." * 20,
        ]

        result = analyzer.analyze_series(texts)

        assert "sample_count" in result
        assert result["sample_count"] == 3
        assert "disorder_trends" in result
        assert "stability_score" in result

    def test_enhanced_forensic_report(self):
        """Test enhanced forensic report generation."""
        analyzer = PersonalityDisorderIndicators(
            use_phrases=False,  # Start with minimal features
        )
        text = "They are watching me and plotting against me." * 30

        report = analyzer.get_enhanced_forensic_report(text, case_id="TEST-123")

        # Enhanced report should have all features enabled
        assert "forensic_metadata" in report
        assert report["forensic_metadata"]["case_id"] == "TEST-123"
        assert report["forensic_metadata"]["analyzer_version"] == "2.0.0"
        # Should have enhanced features
        assert "phrase_matches" in report
        assert "circumplex_position" in report

    def test_negation_reduces_score(self):
        """Test that negation properly affects scores."""
        analyzer = PersonalityDisorderIndicators()

        # Text without negation
        positive_text = "I am suspicious of them. I distrust everyone." * 20

        # Text with negation
        negated_text = "I am not suspicious of anyone. I don't distrust people." * 20

        result_positive = analyzer.analyze(positive_text)
        result_negated = analyzer.analyze(negated_text)

        # Negated text should have lower paranoid score
        assert result_negated["disorders"]["paranoid"]["score"] <= result_positive["disorders"]["paranoid"]["score"]

    def test_genre_affects_scores(self):
        """Test that genre detection affects scores appropriately."""
        analyzer = PersonalityDisorderIndicators()

        # Formal text with some dramatic language (should be adjusted down)
        formal_dramatic = """
        Dear Sir, furthermore, I must express my concerns regarding this
        absolutely devastating crisis. The situation is catastrophic and
        requires urgent attention. Nevertheless, I remain yours faithfully.
        """ * 20

        result = analyzer.analyze(formal_dramatic)

        # Genre should be detected
        assert result["genre"]["detected"] == "formal"
        # Histrionic score should be lower than it would be without genre adjustment


class TestBackwardCompatibility:
    """Tests to ensure backward compatibility with v1.0 API."""

    def test_default_init_works(self):
        """Test that default initialization still works."""
        analyzer = PersonalityDisorderIndicators()
        assert analyzer is not None

    def test_analyze_returns_same_structure(self, sample_text):
        """Test that analyze() returns the same core structure."""
        analyzer = PersonalityDisorderIndicators()
        result = analyzer.analyze(sample_text)

        # All v1.0 fields should be present
        assert "disclaimer" in result
        assert "text_adequacy" in result
        assert "disorders" in result
        assert "clusters" in result
        assert "validation" in result
        assert "confidence" in result
        assert "summary" in result

    def test_analyze_forensic_works(self, sample_text):
        """Test that analyze_forensic() still works."""
        analyzer = PersonalityDisorderIndicators()
        result = analyzer.analyze_forensic(sample_text, case_id="BC-123")

        assert "forensic_metadata" in result
        assert result["forensic_metadata"]["case_id"] == "BC-123"

    def test_get_indicator_summary_works(self, sample_text):
        """Test that get_indicator_summary() still works."""
        analyzer = PersonalityDisorderIndicators()
        summary = analyzer.get_indicator_summary(sample_text)

        assert "paranoid_markers" in summary
        assert "cluster_a_score" in summary


# =============================================================================
# Semantic Layer Tests
# =============================================================================


class TestPDSemanticLayer:
    """Tests for the PDSemanticLayer."""

    def test_semantic_layer_initialization(self):
        """Test semantic layer can be initialized."""
        from seshat.psychology.pd_semantic import PDSemanticLayer

        layer = PDSemanticLayer(use_embeddings=False, use_topics=True)
        assert layer is not None
        assert layer.use_topics is True
        assert layer.use_embeddings is False

    def test_topic_extraction(self):
        """Test topic extraction from text."""
        from seshat.psychology.pd_semantic import PDSemanticLayer

        layer = PDSemanticLayer(use_embeddings=False, use_topics=True)

        # Long enough text for topic modeling
        text = """
        I know they are all talking about me behind my back. Everyone is plotting
        against me and I can't trust anyone. They are watching my every move and
        waiting for me to fail. People pretend to be friendly but they have hidden
        agendas. I have to be constantly on guard because the world is full of
        liars and manipulators. Someone is definitely spying on me.
        """ * 5

        result = layer.extract_topics(text, n_topics=3)

        # Should return TopicAnalysisResult
        assert hasattr(result, "topics")
        assert hasattr(result, "dominant_topic")
        assert hasattr(result, "disorder_relevance")

    def test_analyze_returns_dict(self):
        """Test analyze method returns expected structure."""
        from seshat.psychology.pd_semantic import PDSemanticLayer

        layer = PDSemanticLayer(use_embeddings=False, use_topics=True)

        text = "I am suspicious of everyone around me." * 20

        result = layer.analyze(text)

        assert isinstance(result, dict)
        assert "semantic_similarity" in result
        assert "topics" in result
        assert "topic_disorder_weights" in result

    def test_disorder_prototypes_exist(self):
        """Test that disorder prototypes are defined."""
        from seshat.psychology.pd_semantic import DISORDER_PROTOTYPES

        assert "paranoid" in DISORDER_PROTOTYPES
        assert "narcissistic" in DISORDER_PROTOTYPES
        assert "borderline" in DISORDER_PROTOTYPES
        assert len(DISORDER_PROTOTYPES) == 10

    def test_is_available_method(self):
        """Test is_available returns status dict."""
        from seshat.psychology.pd_semantic import PDSemanticLayer

        layer = PDSemanticLayer(use_embeddings=False, use_topics=True)
        availability = layer.is_available()

        assert isinstance(availability, dict)
        assert "embeddings" in availability
        assert "topics" in availability


# =============================================================================
# Classifier Layer Tests
# =============================================================================


class TestPDClassifier:
    """Tests for the PDClassifier."""

    def test_feature_extractor_initialization(self):
        """Test feature extractor can be initialized."""
        from seshat.psychology.pd_classifier import PDFeatureExtractor

        extractor = PDFeatureExtractor()
        assert extractor is not None
        assert len(extractor.FEATURE_NAMES) == 32

    def test_feature_extraction(self):
        """Test feature extraction from text."""
        from seshat.psychology.pd_classifier import PDFeatureExtractor
        import numpy as np

        extractor = PDFeatureExtractor()
        text = "I am happy and content. Life is wonderful!"

        features = extractor.extract_features(text)

        assert isinstance(features, np.ndarray)
        assert len(features) == 32  # 32 features defined

    def test_feature_extraction_batch(self):
        """Test batch feature extraction."""
        from seshat.psychology.pd_classifier import PDFeatureExtractor
        import numpy as np

        extractor = PDFeatureExtractor()
        texts = [
            "I am happy and content.",
            "They are plotting against me.",
            "I need someone to help me.",
        ]

        features = extractor.extract_features_batch(texts)

        assert isinstance(features, np.ndarray)
        assert features.shape == (3, 32)

    def test_classifier_initialization(self):
        """Test classifier can be initialized."""
        from seshat.psychology.pd_classifier import PDClassifier

        classifier = PDClassifier()
        assert classifier is not None
        assert classifier.is_trained() is False

    def test_classifier_training(self):
        """Test classifier training with synthetic data."""
        from seshat.psychology.pd_classifier import (
            PDClassifier,
            create_training_data_from_prototypes,
        )

        classifier = PDClassifier()

        # Get synthetic training data
        texts, labels = create_training_data_from_prototypes()

        result = classifier.train(texts, labels)

        assert result.success is True
        assert result.n_samples > 0
        assert classifier.is_trained() is True

    def test_classifier_prediction_after_training(self):
        """Test classifier prediction after training."""
        from seshat.psychology.pd_classifier import (
            PDClassifier,
            create_training_data_from_prototypes,
        )

        classifier = PDClassifier()
        texts, labels = create_training_data_from_prototypes()
        classifier.train(texts, labels)

        # Predict on paranoid text
        text = "I know they are watching me and plotting against me." * 5

        results = classifier.predict(text)

        assert "paranoid" in results
        assert hasattr(results["paranoid"], "probability")
        assert hasattr(results["paranoid"], "confidence")
        assert 0.0 <= results["paranoid"].probability <= 1.0

    def test_feature_importance(self):
        """Test feature importance extraction."""
        from seshat.psychology.pd_classifier import (
            PDClassifier,
            create_training_data_from_prototypes,
        )

        classifier = PDClassifier()
        texts, labels = create_training_data_from_prototypes()
        classifier.train(texts, labels)

        importance = classifier.get_feature_importance("paranoid")

        assert isinstance(importance, dict)
        assert len(importance) > 0

    def test_create_training_data_from_prototypes(self):
        """Test synthetic training data creation."""
        from seshat.psychology.pd_classifier import create_training_data_from_prototypes

        texts, labels = create_training_data_from_prototypes()

        assert len(texts) > 0
        assert all(disorder in labels for disorder in [
            "paranoid", "schizoid", "narcissistic", "borderline"
        ])
        assert len(labels["paranoid"]) == len(texts)


# =============================================================================
# Integration Tests for Optional Layers
# =============================================================================


class TestOptionalLayersIntegration:
    """Tests for integration of optional semantic and classifier layers."""

    def test_semantic_layer_in_analyzer(self):
        """Test semantic layer integration in main analyzer."""
        # Topics should work without embeddings
        analyzer = PersonalityDisorderIndicators(
            use_embeddings=False,
            use_topics=True,
            use_classifier=False,
        )

        text = """
        I know they are all watching me. Everyone is suspicious and plotting.
        I can't trust anyone because they are all liars. They deceive me
        constantly and betray my trust. I must be vigilant at all times.
        """ * 10

        result = analyzer.analyze(text)

        # Should have topics in results if available
        # Note: topics require sufficient text and sklearn
        assert "disorders" in result
        assert "paranoid" in result["disorders"]

    def test_analyzer_with_all_optional_disabled(self):
        """Test analyzer works with all optional features disabled."""
        analyzer = PersonalityDisorderIndicators(
            use_embeddings=False,
            use_topics=False,
            use_classifier=False,
        )

        text = "This is a normal text for testing purposes." * 20

        result = analyzer.analyze(text)

        assert "disorders" in result
        assert "semantic_similarity" not in result
        assert "topics" not in result
        assert "classifier_predictions" not in result

    def test_classifier_integration_untrained(self):
        """Test that untrained classifier doesn't affect results."""
        analyzer = PersonalityDisorderIndicators(
            use_embeddings=False,
            use_topics=False,
            use_classifier=True,  # Enable but don't train
        )

        text = "I am suspicious of everyone." * 20

        result = analyzer.analyze(text)

        # Should not have classifier predictions (not trained)
        assert "classifier_predictions" not in result

    def test_full_analysis_with_optional_layers(self):
        """Test full analysis including optional layers where available."""
        analyzer = PersonalityDisorderIndicators(
            use_phrases=True,
            use_negation=True,
            use_context=True,
            use_syntactic=True,
            use_baseline_norm=True,
            use_genre_detection=True,
            use_confidence_cal=True,
            use_cross_validation=True,
            use_minimum_markers=True,
            use_circumplex=True,
            use_temporal=True,
            use_complexity=True,
            use_response_style=True,
            use_embeddings=False,  # Skip embeddings (requires model)
            use_topics=True,
            use_classifier=False,
        )

        text = """
        They are all watching me and plotting against me behind my back.
        I can't trust anyone because everyone is suspicious and deceitful.
        I have to be vigilant because they're targeting me constantly.
        """ * 15

        result = analyzer.analyze(text)

        # Core results
        assert "disorders" in result
        assert "clusters" in result
        assert "validation" in result

        # Enhanced results
        assert "phrase_matches" in result
        assert "negated_markers" in result
        assert "genre" in result
        assert "circumplex_position" in result
        assert "temporal_patterns" in result
        assert "linguistic_complexity" in result
        assert "response_style" in result
