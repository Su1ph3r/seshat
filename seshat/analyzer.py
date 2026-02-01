"""
Core analysis engine for Seshat.

Coordinates feature extraction from all modules and produces comprehensive
stylometric profiles.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import json

from seshat.features.lexical import LexicalFeatures
from seshat.features.function_words import FunctionWordFeatures
from seshat.features.punctuation import PunctuationFeatures
from seshat.features.formatting import FormattingFeatures


@dataclass
class AnalysisResult:
    """Container for analysis results."""

    text_hash: str
    text_length: int
    word_count: int
    sentence_count: int

    lexical_features: Dict[str, Any] = field(default_factory=dict)
    function_word_features: Dict[str, Any] = field(default_factory=dict)
    punctuation_features: Dict[str, Any] = field(default_factory=dict)
    formatting_features: Dict[str, Any] = field(default_factory=dict)
    ngram_features: Dict[str, Any] = field(default_factory=dict)
    syntactic_features: Dict[str, Any] = field(default_factory=dict)
    emoji_features: Dict[str, Any] = field(default_factory=dict)
    social_media_features: Dict[str, Any] = field(default_factory=dict)
    idiolect_features: Dict[str, Any] = field(default_factory=dict)

    psychological_features: Dict[str, Any] = field(default_factory=dict)

    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def all_features(self) -> Dict[str, Any]:
        """Get all features combined into a single dictionary."""
        combined = {}
        combined.update(self.lexical_features)
        combined.update(self.function_word_features)
        combined.update(self.punctuation_features)
        combined.update(self.formatting_features)
        combined.update(self.ngram_features)
        combined.update(self.syntactic_features)
        combined.update(self.emoji_features)
        combined.update(self.social_media_features)
        combined.update(self.idiolect_features)
        combined.update(self.psychological_features)
        return combined

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text_hash": self.text_hash,
            "text_length": self.text_length,
            "word_count": self.word_count,
            "sentence_count": self.sentence_count,
            "lexical_features": self.lexical_features,
            "function_word_features": self.function_word_features,
            "punctuation_features": self.punctuation_features,
            "formatting_features": self.formatting_features,
            "ngram_features": self.ngram_features,
            "syntactic_features": self.syntactic_features,
            "emoji_features": self.emoji_features,
            "social_media_features": self.social_media_features,
            "idiolect_features": self.idiolect_features,
            "psychological_features": self.psychological_features,
            "metadata": self.metadata,
            "all_features": self.all_features,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisResult":
        """Create from dictionary."""
        return cls(
            text_hash=data.get("text_hash", ""),
            text_length=data.get("text_length", 0),
            word_count=data.get("word_count", 0),
            sentence_count=data.get("sentence_count", 0),
            lexical_features=data.get("lexical_features", {}),
            function_word_features=data.get("function_word_features", {}),
            punctuation_features=data.get("punctuation_features", {}),
            formatting_features=data.get("formatting_features", {}),
            ngram_features=data.get("ngram_features", {}),
            syntactic_features=data.get("syntactic_features", {}),
            emoji_features=data.get("emoji_features", {}),
            social_media_features=data.get("social_media_features", {}),
            idiolect_features=data.get("idiolect_features", {}),
            psychological_features=data.get("psychological_features", {}),
            metadata=data.get("metadata", {}),
        )

    def get_flat_features(self) -> Dict[str, float]:
        """
        Get a flat dictionary of all numeric features.

        Useful for ML models and comparison.
        """
        flat = {}

        feature_dicts = [
            ("lexical", self.lexical_features),
            ("function", self.function_word_features),
            ("punctuation", self.punctuation_features),
            ("formatting", self.formatting_features),
            ("ngram", self.ngram_features),
            ("syntactic", self.syntactic_features),
            ("emoji", self.emoji_features),
            ("social", self.social_media_features),
            ("idiolect", self.idiolect_features),
            ("psych", self.psychological_features),
        ]

        for prefix, features in feature_dicts:
            self._flatten_dict(features, prefix, flat)

        return flat

    def _flatten_dict(
        self, d: Dict[str, Any], prefix: str, result: Dict[str, float]
    ) -> None:
        """Recursively flatten a nested dictionary."""
        for key, value in d.items():
            full_key = f"{prefix}_{key}"
            if isinstance(value, (int, float)):
                result[full_key] = float(value)
            elif isinstance(value, bool):
                result[full_key] = 1.0 if value else 0.0
            elif isinstance(value, dict):
                self._flatten_dict(value, full_key, result)


class Analyzer:
    """
    Main stylometric analyzer.

    Coordinates feature extraction from all modules and produces
    comprehensive analysis results.
    """

    def __init__(
        self,
        enable_lexical: bool = True,
        enable_function_words: bool = True,
        enable_punctuation: bool = True,
        enable_formatting: bool = True,
        enable_ngrams: bool = True,
        enable_syntactic: bool = True,
        enable_emoji: bool = True,
        enable_social_media: bool = True,
        enable_idiolect: bool = True,
        enable_psychological: bool = True,
    ):
        """
        Initialize the analyzer with specified feature modules.

        Args:
            enable_*: Flags to enable/disable specific feature extractors
        """
        self.enable_lexical = enable_lexical
        self.enable_function_words = enable_function_words
        self.enable_punctuation = enable_punctuation
        self.enable_formatting = enable_formatting
        self.enable_ngrams = enable_ngrams
        self.enable_syntactic = enable_syntactic
        self.enable_emoji = enable_emoji
        self.enable_social_media = enable_social_media
        self.enable_idiolect = enable_idiolect
        self.enable_psychological = enable_psychological

        self._init_extractors()

    def _init_extractors(self) -> None:
        """Initialize feature extractors."""
        if self.enable_lexical:
            self.lexical_extractor = LexicalFeatures()

        if self.enable_function_words:
            self.function_word_extractor = FunctionWordFeatures()

        if self.enable_punctuation:
            self.punctuation_extractor = PunctuationFeatures()

        if self.enable_formatting:
            self.formatting_extractor = FormattingFeatures()

        self.ngram_extractor = None
        self.syntactic_extractor = None
        self.emoji_extractor = None
        self.social_media_extractor = None
        self.idiolect_extractor = None
        self.psychological_extractor = None

    def _lazy_load_extractors(self) -> None:
        """Lazy load additional extractors when needed."""
        if self.enable_ngrams and self.ngram_extractor is None:
            try:
                from seshat.features.ngrams import NGramFeatures
                self.ngram_extractor = NGramFeatures()
            except ImportError:
                self.ngram_extractor = None

        if self.enable_syntactic and self.syntactic_extractor is None:
            try:
                from seshat.features.syntactic import SyntacticFeatures
                self.syntactic_extractor = SyntacticFeatures()
            except ImportError:
                self.syntactic_extractor = None

        if self.enable_emoji and self.emoji_extractor is None:
            try:
                from seshat.features.emoji import EmojiFeatures
                self.emoji_extractor = EmojiFeatures()
            except ImportError:
                self.emoji_extractor = None

        if self.enable_social_media and self.social_media_extractor is None:
            try:
                from seshat.features.social_media import SocialMediaFeatures
                self.social_media_extractor = SocialMediaFeatures()
            except ImportError:
                self.social_media_extractor = None

        if self.enable_idiolect and self.idiolect_extractor is None:
            try:
                from seshat.features.idiolect import IdiolectFeatures
                self.idiolect_extractor = IdiolectFeatures()
            except ImportError:
                self.idiolect_extractor = None

    def analyze(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AnalysisResult:
        """
        Perform full stylometric analysis on text.

        Args:
            text: Input text to analyze
            metadata: Optional metadata to attach to results

        Returns:
            AnalysisResult with all extracted features

        Raises:
            ValueError: If text is empty or contains only whitespace
        """
        if not text or not text.strip():
            raise ValueError("Cannot analyze empty text")

        from seshat.utils import get_text_hash, tokenize_words, tokenize_sentences

        self._lazy_load_extractors()

        text_hash = get_text_hash(text)
        words = tokenize_words(text)
        sentences = tokenize_sentences(text)

        result = AnalysisResult(
            text_hash=text_hash,
            text_length=len(text),
            word_count=len(words),
            sentence_count=len(sentences),
            metadata=metadata or {},
        )

        if self.enable_lexical:
            result.lexical_features = self.lexical_extractor.extract(text)

        if self.enable_function_words:
            result.function_word_features = self.function_word_extractor.extract(text)

        if self.enable_punctuation:
            result.punctuation_features = self.punctuation_extractor.extract(text)

        if self.enable_formatting:
            result.formatting_features = self.formatting_extractor.extract(text)

        if self.enable_ngrams and self.ngram_extractor:
            result.ngram_features = self.ngram_extractor.extract(text)

        if self.enable_syntactic and self.syntactic_extractor:
            result.syntactic_features = self.syntactic_extractor.extract(text)

        if self.enable_emoji and self.emoji_extractor:
            result.emoji_features = self.emoji_extractor.extract(text)

        if self.enable_social_media and self.social_media_extractor:
            result.social_media_features = self.social_media_extractor.extract(text)

        if self.enable_idiolect and self.idiolect_extractor:
            result.idiolect_features = self.idiolect_extractor.extract(text)

        return result

    def analyze_batch(
        self,
        texts: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ) -> List[AnalysisResult]:
        """
        Analyze multiple texts.

        Args:
            texts: List of texts to analyze
            metadata_list: Optional list of metadata dicts for each text

        Returns:
            List of AnalysisResult objects
        """
        if metadata_list is None:
            metadata_list = [None] * len(texts)

        results = []
        for text, metadata in zip(texts, metadata_list):
            result = self.analyze(text, metadata)
            results.append(result)

        return results

    def get_feature_names(self) -> List[str]:
        """
        Get list of all feature names that the analyzer extracts.

        Useful for understanding what features are available.
        """
        sample_text = "This is a sample text for feature enumeration."
        result = self.analyze(sample_text)
        flat_features = result.get_flat_features()
        return sorted(flat_features.keys())

    def get_feature_vector(self, text: str) -> List[float]:
        """
        Get a consistent feature vector for ML models.

        Returns features in a fixed order.
        """
        result = self.analyze(text)
        flat_features = result.get_flat_features()

        sorted_keys = sorted(flat_features.keys())
        return [flat_features[k] for k in sorted_keys]


class QuickAnalyzer:
    """
    Lightweight analyzer for quick analysis with minimal features.

    Uses only the fastest feature extractors for rapid assessment.
    """

    def __init__(self):
        self.lexical = LexicalFeatures()
        self.function_words = FunctionWordFeatures()
        self.punctuation = PunctuationFeatures()

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Perform quick analysis returning key metrics only.

        Raises:
            ValueError: If text is empty or contains only whitespace
        """
        if not text or not text.strip():
            raise ValueError("Cannot analyze empty text")

        from seshat.utils import tokenize_words, tokenize_sentences

        words = tokenize_words(text)
        sentences = tokenize_sentences(text)

        lexical = self.lexical.extract(text)
        function = self.function_words.extract(text)
        punctuation = self.punctuation.extract(text)

        return {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "avg_word_length": lexical.get("avg_word_length", 0),
            "type_token_ratio": lexical.get("type_token_ratio", 0),
            "function_word_ratio": function.get("total_function_word_ratio", 0),
            "i_ratio": function.get("i_ratio", 0),
            "question_ratio": punctuation.get("terminal_question_ratio", 0),
            "exclamation_ratio": punctuation.get("terminal_exclamation_ratio", 0),
        }
