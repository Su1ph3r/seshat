"""
Linguistic analysis layer for personality disorder detection.

Provides phrase-level detection, negation handling, context window extraction,
and syntactic pattern analysis using spaCy.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict

from seshat.utils import tokenize_words

# Import dictionaries
from .pd_dictionaries import (
    PHRASE_PATTERNS,
    NEGATION_WORDS,
    NEGATION_EXCEPTIONS,
    NEGATION_WINDOW_SIZE,
)


@dataclass
class PhraseMatch:
    """Represents a matched phrase in text."""
    phrase: str
    disorder: str
    dimension: str
    start_pos: int
    end_pos: int
    context: str = ""


@dataclass
class NegatedMarker:
    """Represents a negated marker."""
    marker: str
    negation_word: str
    disorder: str
    dimension: str
    position: int
    original_contribution: float = 0.0


@dataclass
class ContextWindow:
    """Context window around a marker."""
    marker: str
    disorder: str
    dimension: str
    before: List[str] = field(default_factory=list)
    after: List[str] = field(default_factory=list)
    position: int = 0


@dataclass
class SyntacticPatterns:
    """Syntactic pattern analysis results."""
    passive_voice_ratio: float = 0.0
    avg_sentence_length: float = 0.0
    sentence_length_variance: float = 0.0
    subordinate_clause_ratio: float = 0.0
    question_ratio: float = 0.0
    exclamation_ratio: float = 0.0
    pronoun_ratio: float = 0.0
    first_person_ratio: float = 0.0
    third_person_ratio: float = 0.0


class PDLinguisticLayer:
    """Enhanced linguistic analysis for personality disorder detection."""

    def __init__(self, use_spacy: bool = True):
        """
        Initialize the linguistic layer.

        Args:
            use_spacy: Whether to use spaCy for advanced analysis (passive voice, etc.)
        """
        self.use_spacy = use_spacy
        self._nlp = None  # Lazy load spaCy
        self.phrase_patterns = PHRASE_PATTERNS
        self.negation_words = NEGATION_WORDS
        self.negation_exceptions = NEGATION_EXCEPTIONS

        # Build regex patterns for phrase matching
        self._phrase_regexes = self._compile_phrase_patterns()

    def _compile_phrase_patterns(self) -> Dict[str, Dict[str, List[re.Pattern]]]:
        """Compile phrase patterns into regex for efficient matching."""
        compiled = {}
        for disorder, dimensions in self.phrase_patterns.items():
            compiled[disorder] = {}
            for dimension, phrases in dimensions.items():
                patterns = []
                for phrase in phrases:
                    # Escape special regex characters and allow flexible whitespace
                    escaped = re.escape(phrase)
                    # Allow flexible spacing between words
                    pattern = re.compile(
                        r'\b' + escaped.replace(r'\ ', r'\s+') + r'\b',
                        re.IGNORECASE
                    )
                    patterns.append((phrase, pattern))
                compiled[disorder][dimension] = patterns
        return compiled

    @property
    def nlp(self):
        """Lazy load spaCy model."""
        if self._nlp is None and self.use_spacy:
            try:
                import spacy
                try:
                    self._nlp = spacy.load("en_core_web_sm")
                except OSError:
                    # Model not installed
                    self._nlp = False
            except ImportError:
                self._nlp = False
        return self._nlp if self._nlp else None

    def analyze(self, text: str) -> Dict[str, any]:
        """
        Perform full linguistic analysis.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary with phrase matches, negated markers, context windows, and syntactic patterns
        """
        if not text:
            return self._empty_results()

        results = {
            "phrase_matches": self.detect_phrases(text),
            "negated_markers": [],  # Will be populated during analysis
            "context_windows": {},
            "syntactic_patterns": self.analyze_syntactic_patterns(text),
        }

        return results

    def detect_phrases(self, text: str) -> Dict[str, List[PhraseMatch]]:
        """
        Detect multi-word phrases associated with each disorder.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary mapping disorder names to lists of PhraseMatch objects
        """
        matches: Dict[str, List[PhraseMatch]] = defaultdict(list)

        for disorder, dimensions in self._phrase_regexes.items():
            for dimension, patterns in dimensions.items():
                for phrase, pattern in patterns:
                    for match in pattern.finditer(text):
                        # Extract context (50 chars before and after)
                        start = max(0, match.start() - 50)
                        end = min(len(text), match.end() + 50)
                        context = text[start:end]
                        if start > 0:
                            context = "..." + context
                        if end < len(text):
                            context = context + "..."

                        phrase_match = PhraseMatch(
                            phrase=phrase,
                            disorder=disorder,
                            dimension=dimension,
                            start_pos=match.start(),
                            end_pos=match.end(),
                            context=context,
                        )
                        matches[disorder].append(phrase_match)

        return dict(matches)

    def handle_negation(
        self,
        text: str,
        word_markers: Dict[str, Dict[str, List[str]]],
    ) -> Tuple[List[NegatedMarker], Dict[str, Dict[str, float]]]:
        """
        Detect negated markers and calculate adjustment factors.

        Args:
            text: Input text
            word_markers: Dictionary of disorder -> dimension -> list of marker words

        Returns:
            Tuple of (list of NegatedMarker objects, adjustment factors per disorder/dimension)
        """
        text_lower = text.lower()
        words = tokenize_words(text)
        word_positions = self._get_word_positions(text_lower, words)

        negated_markers = []
        adjustments: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

        # Check for negation exceptions first
        exception_positions = set()
        for exception in self.negation_exceptions:
            pos = text_lower.find(exception)
            while pos != -1:
                exception_positions.update(range(pos, pos + len(exception)))
                pos = text_lower.find(exception, pos + 1)

        for disorder, dimensions in word_markers.items():
            for dimension, markers in dimensions.items():
                for marker in markers:
                    marker_lower = marker.lower()
                    # Find all occurrences of this marker
                    marker_indices = [
                        i for i, w in enumerate(words) if w == marker_lower
                    ]

                    for marker_idx in marker_indices:
                        # Check if position is within a negation exception
                        if marker_idx in word_positions:
                            char_pos = word_positions[marker_idx]
                            if char_pos in exception_positions:
                                continue

                        # Look for negation in the window before the marker
                        window_start = max(0, marker_idx - NEGATION_WINDOW_SIZE)
                        window_words = words[window_start:marker_idx]

                        for neg_word in self.negation_words:
                            neg_word_lower = neg_word.lower()
                            if neg_word_lower in window_words:
                                # Found negation
                                negated = NegatedMarker(
                                    marker=marker,
                                    negation_word=neg_word,
                                    disorder=disorder,
                                    dimension=dimension,
                                    position=marker_idx,
                                    original_contribution=1.0,
                                )
                                negated_markers.append(negated)
                                # Reduce contribution (flip to negative)
                                adjustments[disorder][dimension] -= 0.5
                                break

        return negated_markers, dict(adjustments)

    def _get_word_positions(self, text: str, words: List[str]) -> Dict[int, int]:
        """Map word indices to character positions in text."""
        positions = {}
        current_pos = 0
        for i, word in enumerate(words):
            pos = text.find(word, current_pos)
            if pos != -1:
                positions[i] = pos
                current_pos = pos + len(word)
        return positions

    def extract_context_windows(
        self,
        text: str,
        word_markers: Dict[str, Dict[str, List[str]]],
        window_size: int = 5,
    ) -> Dict[str, List[ContextWindow]]:
        """
        Extract context windows around detected markers.

        Args:
            text: Input text
            word_markers: Dictionary of disorder -> dimension -> list of marker words
            window_size: Number of words before and after to include

        Returns:
            Dictionary mapping disorders to lists of ContextWindow objects
        """
        words = tokenize_words(text)
        windows: Dict[str, List[ContextWindow]] = defaultdict(list)

        for disorder, dimensions in word_markers.items():
            for dimension, markers in dimensions.items():
                for marker in markers:
                    marker_lower = marker.lower()
                    # Find all occurrences
                    for i, word in enumerate(words):
                        if word == marker_lower:
                            before = words[max(0, i - window_size):i]
                            after = words[i + 1:i + 1 + window_size]
                            window = ContextWindow(
                                marker=marker,
                                disorder=disorder,
                                dimension=dimension,
                                before=before,
                                after=after,
                                position=i,
                            )
                            windows[disorder].append(window)

        return dict(windows)

    def analyze_syntactic_patterns(self, text: str) -> SyntacticPatterns:
        """
        Analyze syntactic patterns in text.

        Args:
            text: Input text to analyze

        Returns:
            SyntacticPatterns dataclass with various metrics
        """
        if not text:
            return SyntacticPatterns()

        # Basic analysis without spaCy
        sentences = self._split_sentences(text)
        words = tokenize_words(text)
        word_count = len(words)

        if word_count == 0:
            return SyntacticPatterns()

        # Calculate basic metrics
        sentence_lengths = [len(tokenize_words(s)) for s in sentences]
        avg_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
        variance = self._calculate_variance(sentence_lengths) if sentence_lengths else 0

        # Count sentence types
        questions = sum(1 for s in sentences if s.strip().endswith('?'))
        exclamations = sum(1 for s in sentences if s.strip().endswith('!'))
        sentence_count = len(sentences) if sentences else 1

        # Pronoun analysis
        first_person = {'i', "i'm", "i've", "i'd", "i'll", 'me', 'my', 'mine', 'myself', 'we', "we're", "we've", 'us', 'our', 'ours', 'ourselves'}
        third_person = {'he', "he's", "he'd", "he'll", 'him', 'his', 'himself', 'she', "she's", "she'd", "she'll", 'her', 'hers', 'herself', 'they', "they're", "they've", "they'd", 'them', 'their', 'theirs', 'themselves', 'it', "it's", 'its', 'itself'}

        first_person_count = sum(1 for w in words if w in first_person)
        third_person_count = sum(1 for w in words if w in third_person)
        total_pronouns = first_person_count + third_person_count

        result = SyntacticPatterns(
            avg_sentence_length=avg_length,
            sentence_length_variance=variance,
            question_ratio=questions / sentence_count,
            exclamation_ratio=exclamations / sentence_count,
            pronoun_ratio=total_pronouns / word_count if word_count > 0 else 0,
            first_person_ratio=first_person_count / word_count if word_count > 0 else 0,
            third_person_ratio=third_person_count / word_count if word_count > 0 else 0,
        )

        # Advanced analysis with spaCy if available
        if self.nlp:
            result = self._analyze_with_spacy(text, result)

        return result

    def _analyze_with_spacy(self, text: str, base_result: SyntacticPatterns) -> SyntacticPatterns:
        """Enhance syntactic analysis using spaCy."""
        doc = self.nlp(text)

        # Passive voice detection
        passive_count = 0
        total_verbs = 0
        subordinate_clauses = 0

        for sent in doc.sents:
            for token in sent:
                if token.pos_ == "VERB":
                    total_verbs += 1
                    # Check for passive voice
                    if self._is_passive(token):
                        passive_count += 1

                # Count subordinate clause markers
                if token.dep_ in ("mark", "advcl", "acl", "relcl"):
                    subordinate_clauses += 1

        base_result.passive_voice_ratio = passive_count / total_verbs if total_verbs > 0 else 0
        base_result.subordinate_clause_ratio = subordinate_clauses / len(list(doc.sents)) if doc.sents else 0

        return base_result

    def _is_passive(self, token) -> bool:
        """Check if a verb token is in passive voice."""
        # Look for passive auxiliary + past participle pattern
        if token.dep_ == "auxpass":
            return True
        if token.tag_ == "VBN":  # Past participle
            for child in token.children:
                if child.dep_ == "auxpass":
                    return True
            # Check if subject is nsubjpass
            for child in token.head.children if token.head else []:
                if child.dep_ == "nsubjpass":
                    return True
        return False

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using basic rules."""
        # Simple sentence splitting on .!? followed by space and capital
        pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values."""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)

    def get_phrase_score_boost(
        self,
        phrase_matches: Dict[str, List[PhraseMatch]],
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate score boosts from phrase matches.

        Phrases are weighted more heavily than individual words as they
        represent stronger indicators.

        Args:
            phrase_matches: Output from detect_phrases()

        Returns:
            Dictionary of disorder -> dimension -> boost value
        """
        boosts: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

        for disorder, matches in phrase_matches.items():
            for match in matches:
                # Each phrase match contributes a boost
                # (phrases are more specific than single words)
                boosts[disorder][match.dimension] += 0.02  # Configurable weight

        return dict(boosts)

    def count_markers_by_dimension(
        self,
        text: str,
        word_markers: Dict[str, Dict[str, List[str]]],
    ) -> Dict[str, Dict[str, int]]:
        """
        Count word markers found per disorder/dimension.

        Args:
            text: Input text
            word_markers: Dictionary of disorder -> dimension -> list of marker words

        Returns:
            Dictionary of disorder -> dimension -> count
        """
        words_lower = set(tokenize_words(text))
        counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for disorder, dimensions in word_markers.items():
            for dimension, markers in dimensions.items():
                for marker in markers:
                    if marker.lower() in words_lower:
                        counts[disorder][dimension] += 1

        return dict(counts)

    def _empty_results(self) -> Dict[str, any]:
        """Return empty results structure."""
        return {
            "phrase_matches": {},
            "negated_markers": [],
            "context_windows": {},
            "syntactic_patterns": SyntacticPatterns(),
        }
