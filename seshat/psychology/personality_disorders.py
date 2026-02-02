"""
Personality disorder linguistic indicator analysis based on DSM-5 clusters.

CRITICAL DISCLAIMER: This module identifies LINGUISTIC CORRELATIONS only.
These are NOT clinical diagnoses and should NEVER be used as diagnostic tools.
For forensic/research use by qualified professionals only.

Based on research correlating language patterns with personality characteristics,
including work on linguistic markers in clinical populations.

Version 2.0 adds:
- Phrase-level detection (multi-word patterns)
- Negation handling
- Context window extraction
- Genre detection and adjustment
- Baseline normalization
- Cross-disorder validation
- Interpersonal circumplex mapping
- Temporal pattern analysis
- Linguistic complexity metrics
- Response style indicators
- Comparison mode
- Temporal series analysis
"""

from typing import Dict, List, Any, Optional, Union
from collections import Counter
from datetime import datetime, timezone
import hashlib

from seshat.utils import tokenize_words

# Import enhancement layers
from .pd_linguistic import PDLinguisticLayer, PhraseMatch, NegatedMarker, ContextWindow
from .pd_calibration import PDCalibrationLayer, ConfidenceResult, GenreDetectionResult
from .pd_validation import PDValidationLayer, ValidationResult, CircumplexPosition, ValidationFlags
from .pd_advanced_metrics import PDAdvancedMetrics, TemporalProfile, ComplexityMetrics, ResponseStyleMetrics
from .pd_temporal import PDTemporalAnalyzer, TemporalAnalysis, TrendResult
from .pd_dictionaries import MINIMUM_MARKERS

# Optional layers (require additional dependencies)
try:
    from .pd_semantic import PDSemanticLayer, SemanticSimilarityResult, TopicAnalysisResult
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False

try:
    from .pd_classifier import PDClassifier, PDFeatureExtractor, ClassificationResult
    CLASSIFIER_AVAILABLE = True
except ImportError:
    CLASSIFIER_AVAILABLE = False


class PersonalityDisorderIndicators:
    """
    Analyze text for linguistic markers associated with personality disorder patterns.

    CRITICAL DISCLAIMER: This analysis identifies linguistic patterns that research
    has correlated with certain personality characteristics. These are NOT clinical
    diagnoses and should NEVER be used as a substitute for professional psychological
    assessment or diagnosis. This tool is intended for forensic and research use
    by qualified professionals only.

    DSM-5 Personality Disorder Clusters:
    - Cluster A (Odd/Eccentric): Paranoid, Schizoid, Schizotypal
    - Cluster B (Dramatic/Emotional): Antisocial, Borderline, Histrionic, Narcissistic
    - Cluster C (Anxious/Fearful): Avoidant, Dependent, Obsessive-Compulsive
    """

    MINIMUM_WORD_COUNT = 500

    SCORE_INTERPRETATION = {
        "minimal": (0.00, 0.15, "Within normal range"),
        "low": (0.15, 0.30, "Some markers present"),
        "moderate": (0.30, 0.50, "Notable presence of markers"),
        "elevated": (0.50, 0.70, "Significant presence of markers"),
        "high": (0.70, 1.00, "Strong presence of markers"),
    }

    CONFIDENCE_TIERS = {
        "very_low": (0, 200, "Analysis not recommended"),
        "low": (200, 500, "Interpret with extreme caution"),
        "medium": (500, 1000, "Adequate for preliminary analysis"),
        "high": (1000, 2000, "Reliable for analysis"),
        "very_high": (2000, float('inf'), "Forensic-grade reliability"),
    }

    def __init__(
        self,
        # Linguistic layer features
        use_phrases: bool = True,
        use_negation: bool = True,
        use_context: bool = True,
        use_syntactic: bool = True,
        # Calibration layer features
        use_baseline_norm: bool = True,
        use_genre_detection: bool = True,
        use_confidence_cal: bool = True,
        # Validation layer features
        use_cross_validation: bool = True,
        use_minimum_markers: bool = True,
        use_circumplex: bool = True,
        # Advanced metrics
        use_temporal: bool = True,
        use_complexity: bool = True,
        use_response_style: bool = True,
        # Optional features (require additional dependencies)
        use_embeddings: bool = False,
        use_classifier: bool = False,
        use_topics: bool = False,
    ):
        """
        Initialize personality disorder indicator analyzer.

        Args:
            use_phrases: Enable multi-word phrase detection
            use_negation: Enable negation handling
            use_context: Enable context window extraction
            use_syntactic: Enable syntactic pattern analysis (requires spaCy)
            use_baseline_norm: Enable baseline normalization
            use_genre_detection: Enable genre detection and adjustment
            use_confidence_cal: Enable enhanced confidence calibration
            use_cross_validation: Enable cross-disorder validation
            use_minimum_markers: Enable minimum marker checks
            use_circumplex: Enable interpersonal circumplex mapping
            use_temporal: Enable temporal pattern analysis
            use_complexity: Enable linguistic complexity metrics
            use_response_style: Enable response style analysis
            use_embeddings: Enable embedding-based similarity (requires model)
            use_classifier: Enable ML classifier layer (requires trained model)
            use_topics: Enable topic modeling (requires embeddings)
        """
        self.indicator_words = self._load_disorder_dictionaries()

        # Store feature flags
        self.use_phrases = use_phrases
        self.use_negation = use_negation
        self.use_context = use_context
        self.use_syntactic = use_syntactic
        self.use_baseline_norm = use_baseline_norm
        self.use_genre_detection = use_genre_detection
        self.use_confidence_cal = use_confidence_cal
        self.use_cross_validation = use_cross_validation
        self.use_minimum_markers = use_minimum_markers
        self.use_circumplex = use_circumplex
        self.use_temporal = use_temporal
        self.use_complexity = use_complexity
        self.use_response_style = use_response_style
        self.use_embeddings = use_embeddings
        self.use_classifier = use_classifier
        self.use_topics = use_topics

        # Initialize layers based on feature flags
        self._init_layers()

    def _init_layers(self):
        """Initialize analysis layers based on feature flags."""
        # Linguistic layer
        if any([self.use_phrases, self.use_negation, self.use_context, self.use_syntactic]):
            self.linguistic_layer = PDLinguisticLayer(use_spacy=self.use_syntactic)
        else:
            self.linguistic_layer = None

        # Calibration layer
        if any([self.use_baseline_norm, self.use_genre_detection, self.use_confidence_cal]):
            self.calibration_layer = PDCalibrationLayer()
        else:
            self.calibration_layer = None

        # Validation layer
        if any([self.use_cross_validation, self.use_minimum_markers, self.use_circumplex]):
            self.validation_layer = PDValidationLayer()
        else:
            self.validation_layer = None

        # Advanced metrics layer
        if any([self.use_temporal, self.use_complexity, self.use_response_style]):
            self.advanced_metrics = PDAdvancedMetrics()
        else:
            self.advanced_metrics = None

        # Temporal analyzer (for series analysis)
        self.temporal_analyzer = PDTemporalAnalyzer()

        # Semantic layer (optional - requires sentence-transformers/sklearn)
        if (self.use_embeddings or self.use_topics) and SEMANTIC_AVAILABLE:
            self.semantic_layer = PDSemanticLayer(
                use_embeddings=self.use_embeddings,
                use_topics=self.use_topics,
            )
        else:
            self.semantic_layer = None

        # Classifier layer (optional - requires sklearn and trained model)
        if self.use_classifier and CLASSIFIER_AVAILABLE:
            try:
                self.classifier = PDClassifier()
            except ImportError:
                self.classifier = None
        else:
            self.classifier = None

    def _load_disorder_dictionaries(self) -> Dict[str, Dict[str, List[str]]]:
        """Load word dictionaries for each personality disorder cluster."""
        return {
            # Cluster A - Odd/Eccentric
            "paranoid": {
                "suspicion": [
                    "suspicious", "suspect", "watching", "spying", "plotting",
                    "scheming", "conspiring", "deceiving", "betraying", "lying",
                    "manipulating", "hiding", "secret", "secretive", "covert",
                ],
                "mistrust": [
                    "distrust", "mistrust", "untrustworthy", "unreliable", "deceitful",
                    "dishonest", "disloyal", "treacherous", "backstab", "betray",
                    "double-cross", "two-faced", "fake", "phony", "pretending",
                ],
                "blame_external": [
                    "they", "them", "their", "fault", "blame", "blamed", "responsible",
                    "caused", "made me", "forced", "attacked", "targeted", "victim",
                    "against me", "persecuted", "unfair", "unjust",
                ],
                "hypervigilance": [
                    "watching", "monitoring", "tracking", "following", "observing",
                    "careful", "cautious", "alert", "vigilant", "guard", "protect",
                    "danger", "threat", "unsafe", "risky", "wary",
                ],
            },
            "schizoid": {
                "social_detachment": [
                    "alone", "solitary", "isolated", "withdrawn", "detached",
                    "distant", "remote", "apart", "separate", "independent",
                    "prefer", "rather", "myself", "own", "private",
                ],
                "emotional_flatness": [
                    "indifferent", "neutral", "unmoved", "unaffected", "calm",
                    "flat", "blank", "empty", "numb", "cold", "cool", "matter-of-fact",
                    "objective", "logical", "rational", "practical",
                ],
                "indifference": [
                    "don't care", "doesn't matter", "whatever", "fine", "okay",
                    "either way", "no preference", "same", "irrelevant", "unimportant",
                    "meaningless", "pointless", "why bother", "shrug",
                ],
            },
            "schizotypal": {
                "magical_thinking": [
                    "psychic", "telepathy", "sixth sense", "premonition", "vision",
                    "supernatural", "mystical", "magical", "cosmic", "fate",
                    "destiny", "meant to be", "sign", "omen", "coincidence",
                    "universe", "energy", "aura", "spiritual", "karma",
                ],
                "unusual_perceptions": [
                    "sense", "feel", "presence", "shadow", "glimpse", "hear",
                    "voice", "whisper", "strange", "weird", "odd", "peculiar",
                    "bizarre", "uncanny", "eerie", "paranormal",
                ],
                "odd_speech": [
                    "tangent", "rambling", "digress", "anyway", "speaking of",
                    "reminds me", "by the way", "incidentally", "random",
                    "unrelated", "off-topic", "abstract", "vague", "unclear",
                ],
            },
            # Cluster B - Dramatic/Emotional
            "antisocial": {
                "rule_violation": [
                    "rules", "laws", "regulations", "restrictions", "limits",
                    "boundaries", "break", "violate", "ignore", "disregard",
                    "above", "don't apply", "exception", "exempt", "special",
                ],
                "deceit": [
                    "lie", "lied", "lying", "deceive", "trick", "con", "scam",
                    "manipulate", "exploit", "use", "fool", "mislead", "pretend",
                    "fake", "act", "disguise", "cover", "hide",
                ],
                "impulsivity": [
                    "impulse", "spontaneous", "sudden", "instant", "immediately",
                    "now", "couldn't wait", "had to", "just did", "without thinking",
                    "regret", "oops", "mistake", "shouldn't have", "rash",
                ],
                "lack_remorse": [
                    "deserved", "asked for", "brought upon", "their problem",
                    "not my fault", "don't care", "so what", "whatever", "who cares",
                    "too bad", "tough", "deal with it", "get over it",
                ],
            },
            "borderline": {
                "emotional_instability": [
                    "up and down", "mood", "suddenly", "intense", "overwhelming",
                    "can't handle", "falling apart", "out of control", "spinning",
                    "chaotic", "unstable", "unpredictable", "volatile", "explosive",
                ],
                "abandonment_fear": [
                    "leave", "leaving", "left", "abandon", "alone", "rejected",
                    "unwanted", "forgotten", "replaced", "discarded", "thrown away",
                    "don't go", "stay", "please", "need you", "without you",
                ],
                "splitting": [
                    "perfect", "wonderful", "amazing", "terrible", "horrible", "worst",
                    "love", "hate", "always", "never", "everything", "nothing",
                    "completely", "totally", "absolutely", "entirely", "all or nothing",
                ],
                "identity_disturbance": [
                    "who am i", "don't know myself", "empty", "hollow", "lost",
                    "confused", "different person", "changing", "unstable", "unclear",
                    "identity", "self", "purpose", "meaning", "direction",
                ],
            },
            "histrionic": {
                "attention_seeking": [
                    "look", "watch", "notice", "see", "attention", "spotlight",
                    "center", "focus", "everyone", "audience", "impressed",
                    "admire", "appreciate", "recognize", "acknowledge",
                ],
                "dramatic_language": [
                    "amazing", "incredible", "unbelievable", "fantastic", "fabulous",
                    "gorgeous", "stunning", "devastating", "catastrophic", "disaster",
                    "emergency", "crisis", "urgent", "critical", "extreme",
                    "exclamation", "!!", "!!!", "omg", "oh my god",
                ],
                "suggestibility": [
                    "they said", "heard", "apparently", "supposedly", "seems",
                    "everyone thinks", "people say", "trend", "popular", "fashionable",
                    "influenced", "convinced", "persuaded", "impressed",
                ],
                "superficial_emotion": [
                    "like", "love", "adore", "hate", "despise", "feel",
                    "emotional", "sensitive", "passionate", "dramatic",
                    "expressive", "theatrical", "performative",
                ],
            },
            "narcissistic": {
                "grandiosity": [
                    "best", "greatest", "superior", "exceptional", "extraordinary",
                    "unique", "special", "talented", "gifted", "genius", "brilliant",
                    "amazing", "incredible", "unmatched", "incomparable",
                ],
                "entitlement": [
                    "deserve", "entitled", "owed", "should", "must", "have to",
                    "expect", "demand", "require", "need", "want", "right",
                    "privilege", "special treatment", "exception",
                ],
                "superiority": [
                    "better", "smarter", "more", "above", "beyond", "superior",
                    "elite", "top", "first", "leading", "ahead", "winning",
                    "successful", "accomplished", "achieved",
                ],
                "lack_empathy": [
                    "don't understand", "can't relate", "their problem", "not my concern",
                    "irrelevant", "unimportant", "trivial", "petty", "beneath",
                    "boring", "tedious", "annoying", "pathetic", "weak",
                ],
            },
            # Cluster C - Anxious/Fearful
            "avoidant": {
                "social_inhibition": [
                    "shy", "quiet", "reserved", "withdrawn", "hesitant", "reluctant",
                    "uncomfortable", "awkward", "nervous", "anxious", "worried",
                    "afraid", "scared", "terrified", "dread",
                ],
                "inadequacy": [
                    "inadequate", "inferior", "lacking", "deficient", "flawed",
                    "imperfect", "not good enough", "failure", "loser", "worthless",
                    "incompetent", "incapable", "unable", "can't",
                ],
                "criticism_sensitivity": [
                    "criticism", "critique", "feedback", "judgment", "evaluation",
                    "rejected", "disapproved", "ridiculed", "mocked", "laughed at",
                    "embarrassed", "humiliated", "ashamed", "hurt", "offended",
                ],
            },
            "dependent": {
                "need_for_others": [
                    "need", "need you", "can't without", "depend", "rely",
                    "help", "support", "guidance", "advice", "opinion",
                    "what do you think", "what should i", "tell me", "show me",
                ],
                "submission": [
                    "agree", "yes", "okay", "whatever you say", "you're right",
                    "defer", "follow", "obey", "comply", "accommodate",
                    "please", "sorry", "apologize", "my fault",
                ],
                "helplessness": [
                    "helpless", "powerless", "weak", "vulnerable", "unable",
                    "can't", "couldn't", "don't know how", "lost", "confused",
                    "overwhelmed", "incapable", "incompetent", "useless",
                ],
                "fear_of_separation": [
                    "alone", "by myself", "without", "left", "abandoned",
                    "separated", "apart", "distance", "away", "gone",
                    "miss", "missing", "lonely", "isolated",
                ],
            },
            "obsessive_compulsive": {
                "perfectionism": [
                    "perfect", "flawless", "impeccable", "exact", "precise",
                    "accurate", "correct", "right", "proper", "ideal",
                    "standard", "quality", "excellence", "best", "optimal",
                ],
                "control": [
                    "control", "order", "organize", "arrange", "plan", "schedule",
                    "system", "method", "procedure", "process", "routine",
                    "structure", "discipline", "strict", "rigid",
                ],
                "rigidity": [
                    "must", "should", "have to", "need to", "required", "necessary",
                    "always", "never", "rule", "principle", "standard",
                    "proper", "correct", "appropriate", "acceptable",
                ],
                "detail_focus": [
                    "detail", "specific", "particular", "precise", "exact",
                    "careful", "thorough", "meticulous", "check", "verify",
                    "review", "inspect", "examine", "scrutinize", "analyze",
                ],
            },
        }

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for personality disorder-related linguistic patterns.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary with linguistic marker analysis and mandatory disclaimers

        Note: Results are linguistic correlations, NOT clinical diagnoses.
        """
        if not text:
            return self._empty_results()

        words = tokenize_words(text)

        if not words:
            return self._empty_results()

        word_count = len(words)
        word_counts = Counter(words)

        text_adequacy = self._calculate_text_adequacy(word_count)

        results = {
            "disclaimer": (
                "CRITICAL: These are linguistic correlations, NOT clinical diagnoses. "
                "This analysis identifies patterns that research has associated with "
                "certain personality characteristics. It should NEVER be used as a "
                "diagnostic tool or substitute for professional psychological assessment. "
                "For forensic and research use by qualified professionals only."
            ),
            "text_adequacy": text_adequacy,
        }

        # === Phase 1: Basic word-level analysis ===
        disorders = {}
        disorders["paranoid"] = self._analyze_paranoid_markers(word_counts, word_count)
        disorders["schizoid"] = self._analyze_schizoid_markers(word_counts, word_count)
        disorders["schizotypal"] = self._analyze_schizotypal_markers(word_counts, word_count)
        disorders["antisocial"] = self._analyze_antisocial_markers(word_counts, word_count)
        disorders["borderline"] = self._analyze_borderline_markers(word_counts, word_count)
        disorders["histrionic"] = self._analyze_histrionic_markers(word_counts, word_count, text)
        disorders["narcissistic"] = self._analyze_narcissistic_markers(word_counts, word_count)
        disorders["avoidant"] = self._analyze_avoidant_markers(word_counts, word_count)
        disorders["dependent"] = self._analyze_dependent_markers(word_counts, word_count)
        disorders["obsessive_compulsive"] = self._analyze_obsessive_compulsive_markers(word_counts, word_count)

        # Extract raw scores for processing
        raw_scores = {d: data["score"] for d, data in disorders.items()}

        # === Phase 2: Linguistic layer enhancements ===
        if self.linguistic_layer:
            # Phrase detection
            if self.use_phrases:
                phrase_matches = self.linguistic_layer.detect_phrases(text)
                phrase_boosts = self.linguistic_layer.get_phrase_score_boost(phrase_matches)
                results["phrase_matches"] = {
                    d: [{"phrase": m.phrase, "dimension": m.dimension, "context": m.context}
                        for m in matches]
                    for d, matches in phrase_matches.items()
                }
                # Apply phrase boosts to scores
                for disorder, dimensions in phrase_boosts.items():
                    for dimension, boost in dimensions.items():
                        raw_scores[disorder] = min(1.0, raw_scores[disorder] + boost)

            # Negation handling
            if self.use_negation:
                negated, adjustments = self.linguistic_layer.handle_negation(
                    text, self.indicator_words
                )
                results["negated_markers"] = [
                    {"marker": n.marker, "negation": n.negation_word, "disorder": n.disorder}
                    for n in negated
                ]
                # Apply negation adjustments
                for disorder, dimensions in adjustments.items():
                    for dimension, adjustment in dimensions.items():
                        # Reduce score when markers are negated
                        raw_scores[disorder] = max(0.0, raw_scores[disorder] + adjustment * 0.1)

            # Context windows (for forensic use)
            if self.use_context:
                context_windows = self.linguistic_layer.extract_context_windows(
                    text, self.indicator_words
                )
                results["context_windows"] = {
                    d: [{"marker": w.marker, "before": w.before, "after": w.after}
                        for w in windows]
                    for d, windows in context_windows.items()
                }

            # Syntactic patterns
            if self.use_syntactic:
                syntactic = self.linguistic_layer.analyze_syntactic_patterns(text)
                results["syntactic_patterns"] = {
                    "passive_voice_ratio": syntactic.passive_voice_ratio,
                    "avg_sentence_length": syntactic.avg_sentence_length,
                    "question_ratio": syntactic.question_ratio,
                    "exclamation_ratio": syntactic.exclamation_ratio,
                    "first_person_ratio": syntactic.first_person_ratio,
                    "third_person_ratio": syntactic.third_person_ratio,
                }

        # === Phase 3: Genre detection and calibration ===
        detected_genre = "neutral"
        if self.calibration_layer:
            if self.use_genre_detection:
                genre_result = self.calibration_layer.detect_genre(text)
                detected_genre = genre_result.genre
                results["genre"] = {
                    "detected": genre_result.genre,
                    "confidence": genre_result.confidence,
                    "indicator_counts": genre_result.indicator_counts,
                }
                # Apply genre adjustments
                raw_scores = self.calibration_layer.adjust_for_genre(raw_scores, detected_genre)

            if self.use_baseline_norm:
                normalized_scores = self.calibration_layer.normalize_scores(raw_scores)
                results["normalized_scores"] = normalized_scores
                # Use normalized scores for further processing
                raw_scores = normalized_scores

        # Update disorder scores with enhanced values
        for disorder in disorders:
            disorders[disorder]["score"] = raw_scores[disorder]
            disorders[disorder]["interpretation"] = self._interpret_score(raw_scores[disorder])

        results["disorders"] = disorders
        results["clusters"] = self._calculate_cluster_scores(disorders)

        # === Phase 4: Validation layer ===
        validation = self._validate_feature_coverage(disorders, word_count)

        if self.validation_layer:
            if self.use_cross_validation:
                discriminant = self.validation_layer.check_discriminant_validity(raw_scores)
                results["discriminant_validity"] = {
                    "is_valid": discriminant.is_valid,
                    "contradictions": [
                        {"disorder1": c[0], "disorder2": c[1], "score1": c[2], "score2": c[3]}
                        for c in discriminant.contradictions
                    ],
                    "warnings": discriminant.warnings,
                    "confidence_adjustment": discriminant.confidence_adjustment,
                }
                # Merge warnings into validation flags
                validation["flags"].extend(discriminant.warnings)
                if not discriminant.is_valid:
                    validation["is_consistent"] = False

            if self.use_minimum_markers:
                # Count markers per disorder
                marker_counts = self.linguistic_layer.count_markers_by_dimension(
                    text, self.indicator_words
                ) if self.linguistic_layer else {}
                total_markers = {
                    d: sum(dims.values()) for d, dims in marker_counts.items()
                }
                marker_checks = self.validation_layer.check_all_minimum_markers(total_markers)
                results["minimum_markers"] = {
                    d: {"meets_minimum": check[0], "explanation": check[1]}
                    for d, check in marker_checks.items()
                }
                # Add flags for elevated scores with insufficient markers
                for disorder, (meets_min, explanation) in marker_checks.items():
                    if not meets_min and raw_scores.get(disorder, 0) > 0.4:
                        validation["flags"].append(f"{disorder}: {explanation}")

            if self.use_circumplex:
                circumplex = self.validation_layer.map_to_circumplex(raw_scores)
                results["circumplex_position"] = {
                    "dominance": circumplex.dominance,
                    "affiliation": circumplex.affiliation,
                    "quadrant": circumplex.quadrant,
                    "angle_degrees": circumplex.angle_degrees,
                    "intensity": circumplex.intensity,
                }

        results["validation"] = validation

        # === Phase 5: Advanced metrics ===
        if self.advanced_metrics:
            if self.use_temporal:
                temporal = self.advanced_metrics.analyze_temporal_patterns(text)
                results["temporal_patterns"] = {
                    "past_focus": temporal.past_focus,
                    "present_focus": temporal.present_focus,
                    "future_focus": temporal.future_focus,
                    "dominant_focus": temporal.dominant_focus,
                    "interpretation": temporal.interpretation,
                }

            if self.use_complexity:
                complexity = self.advanced_metrics.analyze_linguistic_complexity(text)
                results["linguistic_complexity"] = {
                    "vocabulary_sophistication": complexity.vocabulary_sophistication,
                    "lexical_diversity": complexity.lexical_diversity,
                    "avg_word_length": complexity.avg_word_length,
                    "avg_sentence_length": complexity.avg_sentence_length,
                    "readability_score": complexity.readability_score,
                    "interpretation": complexity.interpretation,
                }

            if self.use_response_style:
                response = self.advanced_metrics.analyze_response_style(text)
                results["response_style"] = {
                    "hedging_ratio": response.hedging_ratio,
                    "absolutism_ratio": response.absolutism_ratio,
                    "deflection_ratio": response.deflection_ratio,
                    "self_reference_ratio": response.self_reference_ratio,
                    "emotional_expressiveness": response.emotional_expressiveness,
                    "certainty_ratio": response.certainty_ratio,
                    "interpretation": response.interpretation,
                }

        # === Phase 6: Semantic layer (optional) ===
        if self.semantic_layer:
            semantic_results = self.semantic_layer.analyze(text)

            # Add semantic similarity scores
            if self.use_embeddings and semantic_results.get("semantic_similarity"):
                results["semantic_similarity"] = semantic_results["semantic_similarity"]
                # Optionally blend semantic scores with raw scores
                for disorder, sim_data in semantic_results["semantic_similarity"].items():
                    if disorder in raw_scores:
                        # Weight semantic similarity as a factor (0.2 weight)
                        semantic_boost = sim_data.get("similarity_score", 0.5) - 0.5
                        raw_scores[disorder] = min(1.0, max(0.0,
                            raw_scores[disorder] + semantic_boost * 0.2
                        ))

            # Add topic analysis
            if self.use_topics and semantic_results.get("topics"):
                results["topics"] = semantic_results["topics"]
                results["topic_disorder_weights"] = semantic_results.get("topic_disorder_weights", {})
                # Apply topic-based adjustments
                for disorder, weight in semantic_results.get("topic_disorder_weights", {}).items():
                    if disorder in raw_scores and weight > 0.3:
                        raw_scores[disorder] = min(1.0, raw_scores[disorder] + weight * 0.1)

            # Update disorder scores with semantic adjustments
            for disorder in disorders:
                disorders[disorder]["score"] = raw_scores[disorder]
                disorders[disorder]["interpretation"] = self._interpret_score(raw_scores[disorder])

        # === Phase 7: ML Classifier (optional) ===
        if self.classifier and self.classifier.is_trained():
            try:
                classifier_results = self.classifier.predict(text)
                results["classifier_predictions"] = {
                    d: {
                        "probability": r.probability,
                        "confidence": r.confidence,
                        "contributing_features": [
                            {"feature": f[0], "contribution": f[1]}
                            for f in r.contributing_features
                        ],
                    }
                    for d, r in classifier_results.items()
                }
                # Optionally blend classifier predictions (if confidence is high)
                for disorder, pred in classifier_results.items():
                    if pred.confidence == "high" and disorder in raw_scores:
                        # Weighted average with classifier (0.3 weight for classifier)
                        raw_scores[disorder] = (
                            raw_scores[disorder] * 0.7 + pred.probability * 0.3
                        )
                        disorders[disorder]["score"] = raw_scores[disorder]
                        disorders[disorder]["interpretation"] = self._interpret_score(raw_scores[disorder])
            except RuntimeError:
                # Classifier not trained - skip
                pass

        # === Phase 8: Confidence calibration ===
        if self.calibration_layer and self.use_confidence_cal:
            marker_counts_total = {}
            if self.linguistic_layer:
                marker_counts_dict = self.linguistic_layer.count_markers_by_dimension(
                    text, self.indicator_words
                )
                marker_counts_total = {d: sum(dims.values()) for d, dims in marker_counts_dict.items()}

            calibrated = self.calibration_layer.calibrate_confidence(
                raw_scores,
                validation,
                word_count,
                marker_counts_total,
            )
            results["calibrated_confidence"] = {
                "level": calibrated.level,
                "score": calibrated.score,
                "factors": calibrated.factors,
                "explanation": calibrated.explanation,
            }
            results["confidence"] = calibrated.level
        else:
            results["confidence"] = self._calculate_confidence(results, word_count)

        results["summary"] = self._generate_summary(results)

        return results

    def analyze_forensic(
        self,
        text: str,
        case_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Extended analysis with forensic chain-of-custody metadata.

        Args:
            text: Input text to analyze
            case_id: Optional case identifier for forensic tracking

        Returns:
            Full analysis with forensic metadata
        """
        results = self.analyze(text)

        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()

        results["forensic_metadata"] = {
            "case_id": case_id,
            "text_hash": text_hash,
            "text_length_chars": len(text),
            "text_length_words": results["text_adequacy"]["word_count"],
            "analyzed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "analyzer_version": "2.0.0",
            "limitations": self._get_forensic_limitations(results),
        }

        return results

    def _empty_results(self) -> Dict[str, Any]:
        """Return empty results structure."""
        empty_disorder = {
            "score": 0.0,
        }

        return {
            "disclaimer": (
                "CRITICAL: These are linguistic correlations, NOT clinical diagnoses. "
                "This analysis identifies patterns that research has associated with "
                "certain personality characteristics. It should NEVER be used as a "
                "diagnostic tool or substitute for professional psychological assessment. "
                "For forensic and research use by qualified professionals only."
            ),
            "text_adequacy": {
                "word_count": 0,
                "is_sufficient": False,
                "confidence_tier": "very_low",
                "confidence_description": "Analysis not recommended",
                "limitations": ["Insufficient text for analysis"],
            },
            "disorders": {
                "paranoid": empty_disorder.copy(),
                "schizoid": empty_disorder.copy(),
                "schizotypal": empty_disorder.copy(),
                "antisocial": empty_disorder.copy(),
                "borderline": empty_disorder.copy(),
                "histrionic": empty_disorder.copy(),
                "narcissistic": empty_disorder.copy(),
                "avoidant": empty_disorder.copy(),
                "dependent": empty_disorder.copy(),
                "obsessive_compulsive": empty_disorder.copy(),
            },
            "clusters": {
                "cluster_a": {"score": 0.0, "label": "Odd/Eccentric"},
                "cluster_b": {"score": 0.0, "label": "Dramatic/Emotional"},
                "cluster_c": {"score": 0.0, "label": "Anxious/Fearful"},
            },
            "validation": {
                "feature_coverage": 0.0,
                "is_consistent": True,
                "flags": [],
            },
            "confidence": "very_low",
            "summary": "Insufficient text provided for analysis.",
        }

    def _calculate_text_adequacy(self, word_count: int) -> Dict[str, Any]:
        """Calculate text adequacy and confidence tier."""
        limitations = []

        confidence_tier = "very_low"
        for tier_name, (min_words, max_words, description) in self.CONFIDENCE_TIERS.items():
            if min_words <= word_count < max_words:
                confidence_tier = tier_name
                break

        is_sufficient = word_count >= self.MINIMUM_WORD_COUNT

        if word_count < 200:
            limitations.append("Text is too short for reliable analysis")
        elif word_count < 500:
            limitations.append("Text below recommended minimum; interpret with caution")

        if word_count < 1000:
            limitations.append("Limited sample may not capture full linguistic patterns")

        return {
            "word_count": word_count,
            "is_sufficient": is_sufficient,
            "confidence_tier": confidence_tier,
            "confidence_description": self.CONFIDENCE_TIERS[confidence_tier][2],
            "limitations": limitations,
        }

    def _count_dimension_words(
        self,
        word_counts: Counter,
        dimension_words: List[str],
    ) -> int:
        """Count occurrences of dimension-associated words."""
        return sum(word_counts.get(word.lower(), 0) for word in dimension_words)

    def _calculate_dimension_ratio(
        self,
        word_counts: Counter,
        dimension_words: List[str],
        word_count: int,
    ) -> float:
        """Calculate ratio of dimension words to total words."""
        if word_count == 0:
            return 0.0
        count = self._count_dimension_words(word_counts, dimension_words)
        return count / word_count

    def _calculate_disorder_score(
        self,
        dimension_ratios: Dict[str, float],
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Calculate weighted composite score for a disorder.

        Args:
            dimension_ratios: Dictionary of dimension names to their ratios
            weights: Optional weights for each dimension (defaults to equal)

        Returns:
            Score clamped to 0-1 range
        """
        if not dimension_ratios:
            return 0.0

        if weights is None:
            weights = {k: 1.0 for k in dimension_ratios}

        total_weight = sum(weights.values())
        if total_weight == 0:
            return 0.0

        weighted_sum = sum(
            dimension_ratios[dim] * weights.get(dim, 1.0) * 25
            for dim in dimension_ratios
        )

        score = weighted_sum / total_weight

        return max(0.0, min(1.0, score))

    def _analyze_paranoid_markers(
        self,
        word_counts: Counter,
        word_count: int,
    ) -> Dict[str, Any]:
        """Analyze linguistic markers associated with paranoid patterns."""
        indicators = self.indicator_words["paranoid"]

        suspicion_ratio = self._calculate_dimension_ratio(
            word_counts, indicators["suspicion"], word_count
        )
        mistrust_ratio = self._calculate_dimension_ratio(
            word_counts, indicators["mistrust"], word_count
        )
        blame_external_ratio = self._calculate_dimension_ratio(
            word_counts, indicators["blame_external"], word_count
        )
        hypervigilance_ratio = self._calculate_dimension_ratio(
            word_counts, indicators["hypervigilance"], word_count
        )

        dimension_ratios = {
            "suspicion": suspicion_ratio,
            "mistrust": mistrust_ratio,
            "blame_external": blame_external_ratio,
            "hypervigilance": hypervigilance_ratio,
        }

        weights = {
            "suspicion": 1.5,
            "mistrust": 1.5,
            "blame_external": 1.0,
            "hypervigilance": 1.0,
        }

        score = self._calculate_disorder_score(dimension_ratios, weights)

        return {
            "score": score,
            "suspicion_ratio": suspicion_ratio,
            "mistrust_ratio": mistrust_ratio,
            "blame_external_ratio": blame_external_ratio,
            "hypervigilance_ratio": hypervigilance_ratio,
            "interpretation": self._interpret_score(score),
        }

    def _analyze_schizoid_markers(
        self,
        word_counts: Counter,
        word_count: int,
    ) -> Dict[str, Any]:
        """Analyze linguistic markers associated with schizoid patterns."""
        indicators = self.indicator_words["schizoid"]

        social_detachment_ratio = self._calculate_dimension_ratio(
            word_counts, indicators["social_detachment"], word_count
        )
        emotional_flatness_ratio = self._calculate_dimension_ratio(
            word_counts, indicators["emotional_flatness"], word_count
        )
        indifference_ratio = self._calculate_dimension_ratio(
            word_counts, indicators["indifference"], word_count
        )

        dimension_ratios = {
            "social_detachment": social_detachment_ratio,
            "emotional_flatness": emotional_flatness_ratio,
            "indifference": indifference_ratio,
        }

        weights = {
            "social_detachment": 1.5,
            "emotional_flatness": 1.0,
            "indifference": 1.0,
        }

        score = self._calculate_disorder_score(dimension_ratios, weights)

        return {
            "score": score,
            "social_detachment_ratio": social_detachment_ratio,
            "emotional_flatness_ratio": emotional_flatness_ratio,
            "indifference_ratio": indifference_ratio,
            "interpretation": self._interpret_score(score),
        }

    def _analyze_schizotypal_markers(
        self,
        word_counts: Counter,
        word_count: int,
    ) -> Dict[str, Any]:
        """Analyze linguistic markers associated with schizotypal patterns."""
        indicators = self.indicator_words["schizotypal"]

        magical_thinking_ratio = self._calculate_dimension_ratio(
            word_counts, indicators["magical_thinking"], word_count
        )
        unusual_perceptions_ratio = self._calculate_dimension_ratio(
            word_counts, indicators["unusual_perceptions"], word_count
        )
        odd_speech_ratio = self._calculate_dimension_ratio(
            word_counts, indicators["odd_speech"], word_count
        )

        dimension_ratios = {
            "magical_thinking": magical_thinking_ratio,
            "unusual_perceptions": unusual_perceptions_ratio,
            "odd_speech": odd_speech_ratio,
        }

        weights = {
            "magical_thinking": 1.5,
            "unusual_perceptions": 1.2,
            "odd_speech": 1.0,
        }

        score = self._calculate_disorder_score(dimension_ratios, weights)

        return {
            "score": score,
            "magical_thinking_ratio": magical_thinking_ratio,
            "unusual_perceptions_ratio": unusual_perceptions_ratio,
            "odd_speech_ratio": odd_speech_ratio,
            "interpretation": self._interpret_score(score),
        }

    def _analyze_antisocial_markers(
        self,
        word_counts: Counter,
        word_count: int,
    ) -> Dict[str, Any]:
        """Analyze linguistic markers associated with antisocial patterns."""
        indicators = self.indicator_words["antisocial"]

        rule_violation_ratio = self._calculate_dimension_ratio(
            word_counts, indicators["rule_violation"], word_count
        )
        deceit_ratio = self._calculate_dimension_ratio(
            word_counts, indicators["deceit"], word_count
        )
        impulsivity_ratio = self._calculate_dimension_ratio(
            word_counts, indicators["impulsivity"], word_count
        )
        lack_remorse_ratio = self._calculate_dimension_ratio(
            word_counts, indicators["lack_remorse"], word_count
        )

        dimension_ratios = {
            "rule_violation": rule_violation_ratio,
            "deceit": deceit_ratio,
            "impulsivity": impulsivity_ratio,
            "lack_remorse": lack_remorse_ratio,
        }

        weights = {
            "rule_violation": 1.2,
            "deceit": 1.5,
            "impulsivity": 1.0,
            "lack_remorse": 1.5,
        }

        score = self._calculate_disorder_score(dimension_ratios, weights)

        return {
            "score": score,
            "rule_violation_ratio": rule_violation_ratio,
            "deceit_ratio": deceit_ratio,
            "impulsivity_ratio": impulsivity_ratio,
            "lack_remorse_ratio": lack_remorse_ratio,
            "interpretation": self._interpret_score(score),
        }

    def _analyze_borderline_markers(
        self,
        word_counts: Counter,
        word_count: int,
    ) -> Dict[str, Any]:
        """Analyze linguistic markers associated with borderline patterns."""
        indicators = self.indicator_words["borderline"]

        emotional_instability_ratio = self._calculate_dimension_ratio(
            word_counts, indicators["emotional_instability"], word_count
        )
        abandonment_fear_ratio = self._calculate_dimension_ratio(
            word_counts, indicators["abandonment_fear"], word_count
        )
        splitting_ratio = self._calculate_dimension_ratio(
            word_counts, indicators["splitting"], word_count
        )
        identity_disturbance_ratio = self._calculate_dimension_ratio(
            word_counts, indicators["identity_disturbance"], word_count
        )

        dimension_ratios = {
            "emotional_instability": emotional_instability_ratio,
            "abandonment_fear": abandonment_fear_ratio,
            "splitting": splitting_ratio,
            "identity_disturbance": identity_disturbance_ratio,
        }

        weights = {
            "emotional_instability": 1.5,
            "abandonment_fear": 1.5,
            "splitting": 1.2,
            "identity_disturbance": 1.0,
        }

        score = self._calculate_disorder_score(dimension_ratios, weights)

        return {
            "score": score,
            "emotional_instability_ratio": emotional_instability_ratio,
            "abandonment_fear_ratio": abandonment_fear_ratio,
            "splitting_ratio": splitting_ratio,
            "identity_disturbance_ratio": identity_disturbance_ratio,
            "interpretation": self._interpret_score(score),
        }

    def _analyze_histrionic_markers(
        self,
        word_counts: Counter,
        word_count: int,
        text: str,
    ) -> Dict[str, Any]:
        """Analyze linguistic markers associated with histrionic patterns."""
        indicators = self.indicator_words["histrionic"]

        attention_seeking_ratio = self._calculate_dimension_ratio(
            word_counts, indicators["attention_seeking"], word_count
        )
        dramatic_language_ratio = self._calculate_dimension_ratio(
            word_counts, indicators["dramatic_language"], word_count
        )
        suggestibility_ratio = self._calculate_dimension_ratio(
            word_counts, indicators["suggestibility"], word_count
        )
        superficial_emotion_ratio = self._calculate_dimension_ratio(
            word_counts, indicators["superficial_emotion"], word_count
        )

        exclamation_count = text.count("!!")
        exclamation_ratio = exclamation_count / word_count if word_count > 0 else 0

        dimension_ratios = {
            "attention_seeking": attention_seeking_ratio,
            "dramatic_language": dramatic_language_ratio + exclamation_ratio,
            "suggestibility": suggestibility_ratio,
            "superficial_emotion": superficial_emotion_ratio,
        }

        weights = {
            "attention_seeking": 1.5,
            "dramatic_language": 1.5,
            "suggestibility": 1.0,
            "superficial_emotion": 1.0,
        }

        score = self._calculate_disorder_score(dimension_ratios, weights)

        return {
            "score": score,
            "attention_seeking_ratio": attention_seeking_ratio,
            "dramatic_language_ratio": dramatic_language_ratio,
            "suggestibility_ratio": suggestibility_ratio,
            "superficial_emotion_ratio": superficial_emotion_ratio,
            "exclamation_ratio": exclamation_ratio,
            "interpretation": self._interpret_score(score),
        }

    def _analyze_narcissistic_markers(
        self,
        word_counts: Counter,
        word_count: int,
    ) -> Dict[str, Any]:
        """Analyze linguistic markers associated with narcissistic patterns."""
        indicators = self.indicator_words["narcissistic"]

        grandiosity_ratio = self._calculate_dimension_ratio(
            word_counts, indicators["grandiosity"], word_count
        )
        entitlement_ratio = self._calculate_dimension_ratio(
            word_counts, indicators["entitlement"], word_count
        )
        superiority_ratio = self._calculate_dimension_ratio(
            word_counts, indicators["superiority"], word_count
        )
        lack_empathy_ratio = self._calculate_dimension_ratio(
            word_counts, indicators["lack_empathy"], word_count
        )

        dimension_ratios = {
            "grandiosity": grandiosity_ratio,
            "entitlement": entitlement_ratio,
            "superiority": superiority_ratio,
            "lack_empathy": lack_empathy_ratio,
        }

        weights = {
            "grandiosity": 1.5,
            "entitlement": 1.2,
            "superiority": 1.2,
            "lack_empathy": 1.5,
        }

        score = self._calculate_disorder_score(dimension_ratios, weights)

        return {
            "score": score,
            "grandiosity_ratio": grandiosity_ratio,
            "entitlement_ratio": entitlement_ratio,
            "superiority_ratio": superiority_ratio,
            "lack_empathy_ratio": lack_empathy_ratio,
            "interpretation": self._interpret_score(score),
        }

    def _analyze_avoidant_markers(
        self,
        word_counts: Counter,
        word_count: int,
    ) -> Dict[str, Any]:
        """Analyze linguistic markers associated with avoidant patterns."""
        indicators = self.indicator_words["avoidant"]

        social_inhibition_ratio = self._calculate_dimension_ratio(
            word_counts, indicators["social_inhibition"], word_count
        )
        inadequacy_ratio = self._calculate_dimension_ratio(
            word_counts, indicators["inadequacy"], word_count
        )
        criticism_sensitivity_ratio = self._calculate_dimension_ratio(
            word_counts, indicators["criticism_sensitivity"], word_count
        )

        dimension_ratios = {
            "social_inhibition": social_inhibition_ratio,
            "inadequacy": inadequacy_ratio,
            "criticism_sensitivity": criticism_sensitivity_ratio,
        }

        weights = {
            "social_inhibition": 1.5,
            "inadequacy": 1.2,
            "criticism_sensitivity": 1.5,
        }

        score = self._calculate_disorder_score(dimension_ratios, weights)

        return {
            "score": score,
            "social_inhibition_ratio": social_inhibition_ratio,
            "inadequacy_ratio": inadequacy_ratio,
            "criticism_sensitivity_ratio": criticism_sensitivity_ratio,
            "interpretation": self._interpret_score(score),
        }

    def _analyze_dependent_markers(
        self,
        word_counts: Counter,
        word_count: int,
    ) -> Dict[str, Any]:
        """Analyze linguistic markers associated with dependent patterns."""
        indicators = self.indicator_words["dependent"]

        need_for_others_ratio = self._calculate_dimension_ratio(
            word_counts, indicators["need_for_others"], word_count
        )
        submission_ratio = self._calculate_dimension_ratio(
            word_counts, indicators["submission"], word_count
        )
        helplessness_ratio = self._calculate_dimension_ratio(
            word_counts, indicators["helplessness"], word_count
        )
        fear_of_separation_ratio = self._calculate_dimension_ratio(
            word_counts, indicators["fear_of_separation"], word_count
        )

        dimension_ratios = {
            "need_for_others": need_for_others_ratio,
            "submission": submission_ratio,
            "helplessness": helplessness_ratio,
            "fear_of_separation": fear_of_separation_ratio,
        }

        weights = {
            "need_for_others": 1.5,
            "submission": 1.0,
            "helplessness": 1.2,
            "fear_of_separation": 1.5,
        }

        score = self._calculate_disorder_score(dimension_ratios, weights)

        return {
            "score": score,
            "need_for_others_ratio": need_for_others_ratio,
            "submission_ratio": submission_ratio,
            "helplessness_ratio": helplessness_ratio,
            "fear_of_separation_ratio": fear_of_separation_ratio,
            "interpretation": self._interpret_score(score),
        }

    def _analyze_obsessive_compulsive_markers(
        self,
        word_counts: Counter,
        word_count: int,
    ) -> Dict[str, Any]:
        """Analyze linguistic markers associated with obsessive-compulsive patterns."""
        indicators = self.indicator_words["obsessive_compulsive"]

        perfectionism_ratio = self._calculate_dimension_ratio(
            word_counts, indicators["perfectionism"], word_count
        )
        control_ratio = self._calculate_dimension_ratio(
            word_counts, indicators["control"], word_count
        )
        rigidity_ratio = self._calculate_dimension_ratio(
            word_counts, indicators["rigidity"], word_count
        )
        detail_focus_ratio = self._calculate_dimension_ratio(
            word_counts, indicators["detail_focus"], word_count
        )

        dimension_ratios = {
            "perfectionism": perfectionism_ratio,
            "control": control_ratio,
            "rigidity": rigidity_ratio,
            "detail_focus": detail_focus_ratio,
        }

        weights = {
            "perfectionism": 1.5,
            "control": 1.2,
            "rigidity": 1.2,
            "detail_focus": 1.0,
        }

        score = self._calculate_disorder_score(dimension_ratios, weights)

        return {
            "score": score,
            "perfectionism_ratio": perfectionism_ratio,
            "control_ratio": control_ratio,
            "rigidity_ratio": rigidity_ratio,
            "detail_focus_ratio": detail_focus_ratio,
            "interpretation": self._interpret_score(score),
        }

    def _calculate_cluster_scores(
        self,
        disorders: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate aggregated scores for each DSM-5 cluster."""
        cluster_a_scores = [
            disorders["paranoid"]["score"],
            disorders["schizoid"]["score"],
            disorders["schizotypal"]["score"],
        ]
        cluster_a_weights = [1.0, 1.0, 1.0]

        cluster_b_scores = [
            disorders["antisocial"]["score"],
            disorders["borderline"]["score"],
            disorders["histrionic"]["score"],
            disorders["narcissistic"]["score"],
        ]
        cluster_b_weights = [1.0, 1.0, 1.0, 1.0]

        cluster_c_scores = [
            disorders["avoidant"]["score"],
            disorders["dependent"]["score"],
            disorders["obsessive_compulsive"]["score"],
        ]
        cluster_c_weights = [1.0, 1.0, 1.0]

        def weighted_average(scores, weights):
            if not scores:
                return 0.0
            total_weight = sum(weights)
            if total_weight == 0:
                return 0.0
            return sum(s * w for s, w in zip(scores, weights)) / total_weight

        return {
            "cluster_a": {
                "score": weighted_average(cluster_a_scores, cluster_a_weights),
                "label": "Odd/Eccentric",
                "disorders": ["paranoid", "schizoid", "schizotypal"],
                "interpretation": self._interpret_score(
                    weighted_average(cluster_a_scores, cluster_a_weights)
                ),
            },
            "cluster_b": {
                "score": weighted_average(cluster_b_scores, cluster_b_weights),
                "label": "Dramatic/Emotional",
                "disorders": ["antisocial", "borderline", "histrionic", "narcissistic"],
                "interpretation": self._interpret_score(
                    weighted_average(cluster_b_scores, cluster_b_weights)
                ),
            },
            "cluster_c": {
                "score": weighted_average(cluster_c_scores, cluster_c_weights),
                "label": "Anxious/Fearful",
                "disorders": ["avoidant", "dependent", "obsessive_compulsive"],
                "interpretation": self._interpret_score(
                    weighted_average(cluster_c_scores, cluster_c_weights)
                ),
            },
        }

    def _validate_feature_coverage(
        self,
        disorders: Dict[str, Dict[str, Any]],
        word_count: int,
    ) -> Dict[str, Any]:
        """Validate feature coverage and check for inconsistencies."""
        total_dimensions = 0
        dimensions_with_markers = 0
        flags = []

        for disorder_name, disorder_data in disorders.items():
            for key, value in disorder_data.items():
                if key.endswith("_ratio") and key != "interpretation":
                    total_dimensions += 1
                    if value > 0:
                        dimensions_with_markers += 1

        feature_coverage = (
            dimensions_with_markers / total_dimensions
            if total_dimensions > 0 else 0.0
        )

        cluster_a_high = any(
            disorders[d]["score"] > 0.5
            for d in ["paranoid", "schizoid", "schizotypal"]
        )
        cluster_b_high = any(
            disorders[d]["score"] > 0.5
            for d in ["antisocial", "borderline", "histrionic", "narcissistic"]
        )
        cluster_c_high = any(
            disorders[d]["score"] > 0.5
            for d in ["avoidant", "dependent", "obsessive_compulsive"]
        )

        elevated_clusters = sum([cluster_a_high, cluster_b_high, cluster_c_high])

        is_consistent = True
        if elevated_clusters >= 3:
            flags.append("Elevated markers across all clusters; may indicate response style artifact")
            is_consistent = False

        if disorders["schizoid"]["score"] > 0.5 and disorders["histrionic"]["score"] > 0.5:
            flags.append("Contradictory pattern: schizoid and histrionic markers both elevated")
            is_consistent = False

        if disorders["dependent"]["score"] > 0.5 and disorders["antisocial"]["score"] > 0.5:
            flags.append("Contradictory pattern: dependent and antisocial markers both elevated")
            is_consistent = False

        if word_count < 200:
            flags.append("Sample too small for reliable pattern detection")

        return {
            "feature_coverage": feature_coverage,
            "dimensions_analyzed": total_dimensions,
            "dimensions_with_markers": dimensions_with_markers,
            "is_consistent": is_consistent,
            "flags": flags,
        }

    def _calculate_confidence(
        self,
        results: Dict[str, Any],
        word_count: int,
    ) -> str:
        """Calculate overall confidence level for the analysis."""
        text_confidence = results["text_adequacy"]["confidence_tier"]
        is_consistent = results["validation"]["is_consistent"]
        feature_coverage = results["validation"]["feature_coverage"]

        confidence_order = ["very_low", "low", "medium", "high", "very_high"]

        confidence_level = confidence_order.index(text_confidence)

        if not is_consistent:
            confidence_level = max(0, confidence_level - 1)

        if feature_coverage < 0.1:
            confidence_level = max(0, confidence_level - 1)

        return confidence_order[confidence_level]

    def _interpret_score(self, score: float) -> str:
        """Convert numeric score to interpretation label."""
        for label, (min_val, max_val, description) in self.SCORE_INTERPRETATION.items():
            if min_val <= score < max_val:
                return label
        # Handle edge case where score is exactly 1.0
        if score >= 1.0:
            return "high"
        return "minimal"

    def _generate_summary(self, results: Dict[str, Any]) -> str:
        """Generate human-readable summary with appropriate caveats."""
        parts = [
            "IMPORTANT: This analysis identifies linguistic patterns only.",
            "These are NOT clinical diagnoses or assessments.",
            "",
        ]

        text_adequacy = results["text_adequacy"]
        if not text_adequacy["is_sufficient"]:
            parts.append(
                f"WARNING: Text sample ({text_adequacy['word_count']} words) is below "
                f"the recommended minimum of {self.MINIMUM_WORD_COUNT} words. "
                "Results should be interpreted with extreme caution."
            )
            parts.append("")

        elevated_disorders = []
        for disorder_name, disorder_data in results["disorders"].items():
            if disorder_data["score"] >= 0.30:
                elevated_disorders.append((disorder_name, disorder_data["score"]))

        elevated_disorders.sort(key=lambda x: x[1], reverse=True)

        if elevated_disorders:
            parts.append("Linguistic patterns with notable presence:")
            for disorder, score in elevated_disorders[:3]:
                interpretation = self._interpret_score(score)
                parts.append(f"  - {disorder.replace('_', ' ').title()}: {interpretation} ({score:.2f})")
        else:
            parts.append("No notable linguistic patterns detected above threshold.")

        validation = results["validation"]
        if validation["flags"]:
            parts.append("")
            parts.append("Validation notes:")
            for flag in validation["flags"]:
                parts.append(f"  - {flag}")

        parts.append("")
        parts.append(f"Overall confidence: {results['confidence'].replace('_', ' ').title()}")

        return "\n".join(parts)

    def _get_forensic_limitations(self, results: Dict[str, Any]) -> List[str]:
        """Get list of limitations for forensic reporting."""
        limitations = []

        limitations.append(
            "Linguistic correlations only; not clinical diagnoses"
        )

        text_adequacy = results["text_adequacy"]
        if not text_adequacy["is_sufficient"]:
            limitations.append(
                f"Text sample below recommended minimum ({text_adequacy['word_count']} "
                f"of {self.MINIMUM_WORD_COUNT} words)"
            )

        if not results["validation"]["is_consistent"]:
            limitations.append("Internal consistency checks failed")

        if results["validation"]["feature_coverage"] < 0.2:
            limitations.append("Low feature coverage; limited marker detection")

        limitations.append(
            "Analysis should be corroborated with other evidence and professional assessment"
        )

        return limitations

    def get_indicator_summary(self, text: str) -> Dict[str, float]:
        """
        Get a simplified summary of indicator scores.

        Returns normalized scores (0-1) for each disorder and cluster.
        """
        full_analysis = self.analyze(text)

        summary = {}

        for disorder_name, disorder_data in full_analysis["disorders"].items():
            summary[f"{disorder_name}_markers"] = disorder_data["score"]

        for cluster_name, cluster_data in full_analysis["clusters"].items():
            summary[f"{cluster_name}_score"] = cluster_data["score"]

        return summary

    def compare(
        self,
        text1: str,
        text2: str,
        timestamp1: Optional[datetime] = None,
        timestamp2: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Compare personality disorder indicators between two texts.

        Args:
            text1: First text to analyze
            text2: Second text to analyze
            timestamp1: Optional timestamp for first text
            timestamp2: Optional timestamp for second text

        Returns:
            Dictionary with comparison metrics including:
            - Per-disorder score differences
            - Per-cluster score differences
            - Significant changes
            - Overall change magnitude
        """
        result1 = self.analyze(text1)
        result2 = self.analyze(text2)

        comparison = self.temporal_analyzer.compare_samples(
            result1, result2, timestamp1, timestamp2
        )

        # Add analysis results for reference
        comparison["text1_analysis"] = result1
        comparison["text2_analysis"] = result2

        # Generate summary
        significant = comparison.get("significant_changes", [])
        if significant:
            summary_parts = ["Significant changes detected:"]
            for change in significant[:3]:
                summary_parts.append(
                    f"  - {change['disorder']}: {change['direction']} by {abs(change['change']):.2f}"
                )
        else:
            summary_parts = ["No significant changes detected between samples."]

        comparison["summary"] = "\n".join(summary_parts)

        return comparison

    def analyze_series(
        self,
        texts: List[str],
        timestamps: Optional[List[datetime]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze personality disorder indicators across multiple text samples over time.

        Args:
            texts: List of texts to analyze
            timestamps: Optional list of timestamps for each text

        Returns:
            TemporalAnalysis with trend and change point information
        """
        if len(texts) < 2:
            return {
                "error": "At least 2 text samples required for temporal analysis",
                "sample_count": len(texts),
            }

        # Analyze each text
        results = [self.analyze(text) for text in texts]

        # Perform temporal analysis
        analysis = self.temporal_analyzer.analyze_series(results, timestamps)

        return {
            "sample_count": analysis.sample_count,
            "time_span": analysis.time_span,
            "disorder_trends": {
                d: {
                    "direction": t.direction,
                    "slope": t.slope,
                    "r_squared": t.r_squared,
                    "start_value": t.start_value,
                    "end_value": t.end_value,
                    "change_percent": t.change_percent,
                    "is_significant": t.is_significant,
                    "interpretation": t.interpretation,
                }
                for d, t in analysis.disorder_trends.items()
            },
            "cluster_trends": {
                c: {
                    "direction": t.direction,
                    "slope": t.slope,
                    "change_percent": t.change_percent,
                    "is_significant": t.is_significant,
                }
                for c, t in analysis.cluster_trends.items()
            },
            "change_points": [
                {
                    "index": cp.index,
                    "timestamp": cp.timestamp.isoformat() if cp.timestamp else None,
                    "disorder": cp.disorder,
                    "before_mean": cp.before_mean,
                    "after_mean": cp.after_mean,
                    "change_magnitude": cp.change_magnitude,
                    "direction": cp.direction,
                }
                for cp in analysis.change_points
            ],
            "stability_score": analysis.stability_score,
            "dominant_pattern": analysis.dominant_pattern,
            "interpretation": analysis.interpretation,
            "individual_results": results,
        }

    def get_enhanced_forensic_report(
        self,
        text: str,
        case_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate an enhanced forensic report with all analysis layers.

        Args:
            text: Input text to analyze
            case_id: Optional case identifier for forensic tracking

        Returns:
            Comprehensive forensic report with full analysis
        """
        # Ensure all features are enabled for forensic analysis
        original_flags = {
            "use_phrases": self.use_phrases,
            "use_negation": self.use_negation,
            "use_context": self.use_context,
            "use_syntactic": self.use_syntactic,
            "use_baseline_norm": self.use_baseline_norm,
            "use_genre_detection": self.use_genre_detection,
            "use_confidence_cal": self.use_confidence_cal,
            "use_cross_validation": self.use_cross_validation,
            "use_minimum_markers": self.use_minimum_markers,
            "use_circumplex": self.use_circumplex,
            "use_temporal": self.use_temporal,
            "use_complexity": self.use_complexity,
            "use_response_style": self.use_response_style,
        }

        # Temporarily enable all features
        self.use_phrases = True
        self.use_negation = True
        self.use_context = True
        self.use_syntactic = True
        self.use_baseline_norm = True
        self.use_genre_detection = True
        self.use_confidence_cal = True
        self.use_cross_validation = True
        self.use_minimum_markers = True
        self.use_circumplex = True
        self.use_temporal = True
        self.use_complexity = True
        self.use_response_style = True

        # Re-initialize layers
        self._init_layers()

        # Perform analysis
        results = self.analyze(text)

        # Restore original flags
        for flag, value in original_flags.items():
            setattr(self, flag, value)
        self._init_layers()

        # Add forensic metadata
        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()

        results["forensic_metadata"] = {
            "case_id": case_id,
            "text_hash": text_hash,
            "text_length_chars": len(text),
            "text_length_words": results["text_adequacy"]["word_count"],
            "analyzed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "analyzer_version": "2.0.0",
            "features_used": list(original_flags.keys()),
            "limitations": self._get_forensic_limitations(results),
        }

        return results
