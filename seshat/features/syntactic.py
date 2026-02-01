"""
Syntactic feature extraction for stylometric analysis.

Analyzes sentence structure, complexity, and grammatical patterns.
"""

import re
from collections import Counter
from typing import Dict, List, Any, Optional

from seshat.utils import tokenize_sentences, tokenize_words, safe_divide


class SyntacticFeatures:
    """Extract syntactic features from text."""

    def __init__(self, use_spacy: bool = True):
        """
        Initialize syntactic extractor.

        Args:
            use_spacy: Whether to use spaCy for advanced parsing
        """
        self.use_spacy = use_spacy
        self.nlp = None

        if use_spacy:
            self._load_spacy()

    def _load_spacy(self) -> None:
        """Load spaCy model."""
        try:
            import spacy
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                self.nlp = None
                self.use_spacy = False
        except ImportError:
            self.nlp = None
            self.use_spacy = False

    def extract(self, text: str) -> Dict[str, Any]:
        """
        Extract all syntactic features from text.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary of syntactic features
        """
        if not text:
            return self._empty_features()

        features = {}

        sentence_features = self._extract_sentence_features(text)
        features.update(sentence_features)

        starter_features = self._extract_sentence_starters(text)
        features.update(starter_features)

        if self.use_spacy and self.nlp:
            spacy_features = self._extract_spacy_features(text)
            features.update(spacy_features)
        else:
            heuristic_features = self._extract_heuristic_complexity(text)
            features.update(heuristic_features)

        return features

    def _empty_features(self) -> Dict[str, Any]:
        """Return empty feature dictionary."""
        return {
            "avg_sentence_length": 0.0,
            "sentence_length_std": 0.0,
            "sentence_length_min": 0,
            "sentence_length_max": 0,
            "sentences_per_paragraph": 0.0,
            "declarative_ratio": 0.0,
            "interrogative_ratio": 0.0,
            "exclamatory_ratio": 0.0,
            "imperative_ratio": 0.0,
            "starter_i_ratio": 0.0,
            "starter_the_ratio": 0.0,
            "starter_but_ratio": 0.0,
            "starter_so_ratio": 0.0,
            "starter_and_ratio": 0.0,
            "avg_parse_depth": 0.0,
            "avg_noun_phrases": 0.0,
            "avg_verb_phrases": 0.0,
            "passive_voice_ratio": 0.0,
            "subordinate_clause_ratio": 0.0,
            "pos_noun_ratio": 0.0,
            "pos_verb_ratio": 0.0,
            "pos_adj_ratio": 0.0,
            "pos_adv_ratio": 0.0,
        }

    def _extract_sentence_features(self, text: str) -> Dict[str, Any]:
        """Extract sentence-level features."""
        sentences = tokenize_sentences(text)

        if not sentences:
            return {
                "avg_sentence_length": 0.0,
                "sentence_length_std": 0.0,
                "sentence_length_min": 0,
                "sentence_length_max": 0,
                "sentences_per_paragraph": 0.0,
            }

        sentence_lengths = []
        for sentence in sentences:
            words = tokenize_words(sentence)
            sentence_lengths.append(len(words))

        avg_length = sum(sentence_lengths) / len(sentence_lengths)

        variance = sum((l - avg_length) ** 2 for l in sentence_lengths) / len(sentence_lengths)
        std_length = variance ** 0.5

        paragraphs = [p for p in re.split(r'\n\s*\n', text) if p.strip()]
        sentences_per_para = len(sentences) / len(paragraphs) if paragraphs else 0

        declarative = 0
        interrogative = 0
        exclamatory = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if sentence.endswith("?"):
                interrogative += 1
            elif sentence.endswith("!"):
                exclamatory += 1
            else:
                declarative += 1

        total = len(sentences)

        return {
            "avg_sentence_length": avg_length,
            "sentence_length_std": std_length,
            "sentence_length_min": min(sentence_lengths),
            "sentence_length_max": max(sentence_lengths),
            "sentences_per_paragraph": sentences_per_para,
            "declarative_ratio": declarative / total,
            "interrogative_ratio": interrogative / total,
            "exclamatory_ratio": exclamatory / total,
            "imperative_ratio": 0.0,
        }

    def _extract_sentence_starters(self, text: str) -> Dict[str, Any]:
        """Extract sentence starter patterns."""
        sentences = tokenize_sentences(text)

        if not sentences:
            return {
                "starter_i_ratio": 0.0,
                "starter_the_ratio": 0.0,
                "starter_but_ratio": 0.0,
                "starter_so_ratio": 0.0,
                "starter_and_ratio": 0.0,
                "starter_distribution": {},
            }

        starters = []
        for sentence in sentences:
            words = tokenize_words(sentence)
            if words:
                starters.append(words[0])

        total = len(starters)
        starter_counts = Counter(starters)

        common_starters = ["i", "the", "but", "so", "and", "it", "this", "we", "he", "she", "they", "there", "when", "if", "as", "what", "how", "well"]

        features = {}
        for starter in common_starters:
            features[f"starter_{starter}_ratio"] = starter_counts.get(starter, 0) / total

        top_starters = starter_counts.most_common(20)
        features["starter_distribution"] = {s: c / total for s, c in top_starters}

        return features

    def _extract_spacy_features(self, text: str) -> Dict[str, Any]:
        """Extract features using spaCy NLP."""
        if not self.nlp:
            return self._extract_heuristic_complexity(text)

        max_text_length = 100000
        if len(text) > max_text_length:
            text = text[:max_text_length]

        doc = self.nlp(text)

        pos_counts = Counter(token.pos_ for token in doc if not token.is_space)
        total_tokens = sum(pos_counts.values())

        noun_ratio = safe_divide(
            pos_counts.get("NOUN", 0) + pos_counts.get("PROPN", 0),
            total_tokens
        )
        verb_ratio = safe_divide(pos_counts.get("VERB", 0), total_tokens)
        adj_ratio = safe_divide(pos_counts.get("ADJ", 0), total_tokens)
        adv_ratio = safe_divide(pos_counts.get("ADV", 0), total_tokens)

        parse_depths = []
        for sent in doc.sents:
            for token in sent:
                depth = 0
                current = token
                while current.head != current:
                    depth += 1
                    current = current.head
                parse_depths.append(depth)

        avg_depth = sum(parse_depths) / len(parse_depths) if parse_depths else 0

        noun_chunks_per_sent = []
        for sent in doc.sents:
            sent_doc = sent.as_doc()
            noun_chunks_per_sent.append(len(list(sent_doc.noun_chunks)))

        avg_noun_chunks = (
            sum(noun_chunks_per_sent) / len(noun_chunks_per_sent)
            if noun_chunks_per_sent else 0
        )

        passive_count = 0
        for token in doc:
            if token.dep_ == "nsubjpass" or token.dep_ == "auxpass":
                passive_count += 1

        sentences = list(doc.sents)
        passive_ratio = passive_count / len(sentences) if sentences else 0

        subordinate_markers = {"because", "although", "while", "if", "when", "since", "unless", "until", "whereas", "whenever", "wherever", "whether"}
        subordinate_count = sum(1 for token in doc if token.text.lower() in subordinate_markers)
        subordinate_ratio = subordinate_count / len(sentences) if sentences else 0

        return {
            "avg_parse_depth": avg_depth,
            "avg_noun_phrases": avg_noun_chunks,
            "avg_verb_phrases": 0.0,
            "passive_voice_ratio": passive_ratio,
            "subordinate_clause_ratio": subordinate_ratio,
            "pos_noun_ratio": noun_ratio,
            "pos_verb_ratio": verb_ratio,
            "pos_adj_ratio": adj_ratio,
            "pos_adv_ratio": adv_ratio,
        }

    def _extract_heuristic_complexity(self, text: str) -> Dict[str, Any]:
        """Extract complexity features using heuristics (without spaCy)."""
        subordinate_markers = {"because", "although", "while", "if", "when", "since", "unless", "until", "whereas", "whenever", "wherever", "whether", "that", "which", "who", "whom"}

        words = tokenize_words(text)
        sentences = tokenize_sentences(text)

        subordinate_count = sum(1 for w in words if w in subordinate_markers)
        subordinate_ratio = subordinate_count / len(sentences) if sentences else 0

        passive_indicators = 0
        passive_pattern = re.compile(r"\b(was|were|is|are|been|being)\s+\w+ed\b", re.IGNORECASE)
        passive_indicators = len(passive_pattern.findall(text))
        passive_ratio = passive_indicators / len(sentences) if sentences else 0

        comma_count = text.count(",")
        avg_commas = comma_count / len(sentences) if sentences else 0

        estimated_depth = 1 + (avg_commas * 0.5) + (subordinate_ratio * 1.5)

        return {
            "avg_parse_depth": estimated_depth,
            "avg_noun_phrases": 0.0,
            "avg_verb_phrases": 0.0,
            "passive_voice_ratio": passive_ratio,
            "subordinate_clause_ratio": subordinate_ratio,
            "pos_noun_ratio": 0.0,
            "pos_verb_ratio": 0.0,
            "pos_adj_ratio": 0.0,
            "pos_adv_ratio": 0.0,
        }

    def get_pos_tag_ngrams(self, text: str, n: int = 2) -> Dict[str, float]:
        """
        Get POS tag n-gram frequencies.

        Args:
            text: Input text
            n: N-gram size

        Returns:
            Dictionary of POS n-gram frequencies
        """
        if not self.nlp:
            return {}

        doc = self.nlp(text)

        pos_tags = [token.pos_ for token in doc if not token.is_space]

        if len(pos_tags) < n:
            return {}

        ngrams = [tuple(pos_tags[i:i+n]) for i in range(len(pos_tags) - n + 1)]
        counts = Counter(ngrams)
        total = len(ngrams)

        return {"-".join(ng): count / total for ng, count in counts.most_common(50)}

    def analyze_sentence_complexity(self, text: str) -> Dict[str, Any]:
        """
        Provide detailed sentence complexity analysis.

        Returns human-readable analysis of syntactic patterns.
        """
        features = self.extract(text)

        analysis = {
            "complexity_level": "medium",
            "patterns": [],
            "metrics": {},
        }

        avg_length = features.get("avg_sentence_length", 0)
        if avg_length > 25:
            analysis["complexity_level"] = "high"
            analysis["patterns"].append("Long average sentence length (academic/formal)")
        elif avg_length < 12:
            analysis["complexity_level"] = "low"
            analysis["patterns"].append("Short average sentence length (conversational)")

        if features.get("subordinate_clause_ratio", 0) > 0.5:
            analysis["patterns"].append("Heavy use of subordinate clauses")
            analysis["complexity_level"] = "high"

        if features.get("passive_voice_ratio", 0) > 0.3:
            analysis["patterns"].append("Frequent passive voice (formal/academic)")

        if features.get("interrogative_ratio", 0) > 0.2:
            analysis["patterns"].append("High question frequency (engaging style)")

        if features.get("starter_i_ratio", 0) > 0.15:
            analysis["patterns"].append("Frequently starts sentences with 'I' (personal)")

        if features.get("starter_and_ratio", 0) > 0.05 or features.get("starter_but_ratio", 0) > 0.05:
            analysis["patterns"].append("Starts sentences with conjunctions (informal)")

        analysis["metrics"] = {
            "avg_sentence_length": avg_length,
            "sentence_length_std": features.get("sentence_length_std", 0),
            "subordinate_ratio": features.get("subordinate_clause_ratio", 0),
            "passive_ratio": features.get("passive_voice_ratio", 0),
        }

        return analysis
