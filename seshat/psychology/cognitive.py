"""
Cognitive style analysis for psychological profiling.

Analyzes thinking patterns, analytical style, and cognitive complexity.
"""

from typing import Dict, List, Any
from collections import Counter

from seshat.utils import tokenize_words, tokenize_sentences, safe_divide


class CognitiveAnalyzer:
    """
    Analyze cognitive and thinking style patterns in text.
    """

    def __init__(self):
        """Initialize cognitive analyzer with word categories."""
        self.cognitive_words = self._load_cognitive_dictionaries()

    def _load_cognitive_dictionaries(self) -> Dict[str, List[str]]:
        """Load cognitive process word lists."""
        return {
            "insight": [
                "think", "know", "consider", "realize", "understand",
                "find", "found", "thought", "believe", "feel",
                "sense", "aware", "recognize", "discover", "notice",
                "perceive", "comprehend", "grasp", "conclude",
            ],
            "causation": [
                "because", "cause", "caused", "causing", "effect",
                "hence", "therefore", "thus", "consequently", "result",
                "resulting", "reason", "since", "why", "lead", "led",
                "due", "owing", "accordingly", "thereby",
            ],
            "discrepancy": [
                "should", "would", "could", "ought", "need",
                "must", "want", "wanted", "wish", "wished",
                "hope", "hoped", "expect", "expected", "desire",
            ],
            "tentative": [
                "maybe", "perhaps", "might", "possibly", "probably",
                "guess", "suppose", "assume", "seem", "seemed",
                "appear", "appeared", "likely", "unlikely", "uncertain",
                "doubt", "doubtful", "unsure", "questionable",
            ],
            "certainty": [
                "always", "never", "definitely", "certainly", "absolutely",
                "completely", "totally", "entirely", "clearly", "obviously",
                "undoubtedly", "surely", "truly", "indeed", "exact",
                "precisely", "exactly", "certain", "sure", "positive",
            ],
            "differentiation": [
                "but", "however", "although", "though", "except",
                "unless", "rather", "instead", "otherwise", "whereas",
                "nevertheless", "nonetheless", "despite", "yet", "still",
                "either", "neither", "nor", "contrast", "contrary",
            ],
            "comparison": [
                "like", "as", "than", "similar", "same", "different",
                "compare", "compared", "comparison", "contrast",
                "versus", "vs", "unlike", "equally", "more", "less",
            ],
            "quantifier": [
                "all", "every", "each", "any", "many", "much",
                "few", "little", "some", "several", "most", "least",
                "none", "both", "half", "whole", "entire", "total",
            ],
            "time_orientation": {
                "past": [
                    "was", "were", "had", "did", "been", "went",
                    "came", "said", "told", "asked", "yesterday",
                    "ago", "before", "previously", "formerly", "once",
                ],
                "present": [
                    "is", "are", "am", "do", "does", "have", "has",
                    "now", "today", "currently", "presently", "ongoing",
                ],
                "future": [
                    "will", "shall", "going", "gonna", "tomorrow",
                    "soon", "later", "eventually", "next", "upcoming",
                    "future", "plan", "planning", "intend", "expect",
                ],
            },
        }

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze cognitive style of text.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary with cognitive analysis results
        """
        if not text:
            return self._empty_results()

        words = tokenize_words(text)
        sentences = tokenize_sentences(text)

        if not words:
            return self._empty_results()

        word_count = len(words)
        word_counts = Counter(words)

        results = {}

        analytical = self._analyze_analytical_thinking(word_counts, word_count)
        results.update(analytical)

        complexity = self._analyze_cognitive_complexity(word_counts, word_count, sentences)
        results.update(complexity)

        certainty = self._analyze_certainty(word_counts, word_count)
        results.update(certainty)

        time_focus = self._analyze_time_focus(word_counts, word_count)
        results.update(time_focus)

        results["cognitive_style"] = self._determine_cognitive_style(results)
        results["thinking_summary"] = self._generate_summary(results)

        return results

    def _empty_results(self) -> Dict[str, Any]:
        """Return empty results structure."""
        return {
            "analytical_score": 0.0,
            "insight_ratio": 0.0,
            "causation_ratio": 0.0,
            "cognitive_complexity": 0.0,
            "differentiation_ratio": 0.0,
            "comparison_ratio": 0.0,
            "certainty_ratio": 0.0,
            "tentative_ratio": 0.0,
            "certainty_vs_tentative": 0.5,
            "past_focus": 0.0,
            "present_focus": 0.0,
            "future_focus": 0.0,
            "time_orientation": "balanced",
            "cognitive_style": "balanced",
            "thinking_summary": "",
        }

    def _analyze_analytical_thinking(
        self, word_counts: Counter, word_count: int
    ) -> Dict[str, Any]:
        """
        Analyze analytical thinking markers.

        Higher analytical thinking is associated with:
        - More articles (a, an, the)
        - More prepositions
        - More cognitive process words
        """
        articles = ["a", "an", "the"]
        article_count = sum(word_counts.get(w, 0) for w in articles)
        article_ratio = article_count / word_count if word_count > 0 else 0

        prepositions = [
            "in", "on", "at", "to", "for", "with", "by", "about",
            "from", "into", "through", "during", "before", "after",
            "above", "below", "between", "under", "over",
        ]
        preposition_count = sum(word_counts.get(w, 0) for w in prepositions)
        preposition_ratio = preposition_count / word_count if word_count > 0 else 0

        insight_count = sum(word_counts.get(w, 0) for w in self.cognitive_words["insight"])
        insight_ratio = insight_count / word_count if word_count > 0 else 0

        causation_count = sum(word_counts.get(w, 0) for w in self.cognitive_words["causation"])
        causation_ratio = causation_count / word_count if word_count > 0 else 0

        analytical_score = (
            article_ratio * 5 +
            preposition_ratio * 3 +
            insight_ratio * 10 +
            causation_ratio * 15
        )
        analytical_score = min(analytical_score, 1.0)

        return {
            "analytical_score": analytical_score,
            "article_ratio": article_ratio,
            "preposition_ratio": preposition_ratio,
            "insight_ratio": insight_ratio,
            "causation_ratio": causation_ratio,
        }

    def _analyze_cognitive_complexity(
        self, word_counts: Counter, word_count: int, sentences: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze cognitive complexity markers.

        Higher complexity indicated by:
        - Exclusive/differentiating words
        - Comparison words
        - Longer sentences
        - More conjunctions
        """
        diff_count = sum(word_counts.get(w, 0) for w in self.cognitive_words["differentiation"])
        diff_ratio = diff_count / word_count if word_count > 0 else 0

        comp_count = sum(word_counts.get(w, 0) for w in self.cognitive_words["comparison"])
        comp_ratio = comp_count / word_count if word_count > 0 else 0

        if sentences:
            avg_sentence_length = word_count / len(sentences)
        else:
            avg_sentence_length = 0

        conjunctions = ["and", "but", "or", "so", "because", "although", "while", "if", "when"]
        conjunction_count = sum(word_counts.get(w, 0) for w in conjunctions)
        conjunction_ratio = conjunction_count / word_count if word_count > 0 else 0

        complexity_score = (
            diff_ratio * 15 +
            comp_ratio * 10 +
            min(avg_sentence_length / 30, 0.3) +
            conjunction_ratio * 8
        )
        complexity_score = min(complexity_score, 1.0)

        return {
            "cognitive_complexity": complexity_score,
            "differentiation_ratio": diff_ratio,
            "comparison_ratio": comp_ratio,
            "avg_sentence_length": avg_sentence_length,
            "conjunction_ratio": conjunction_ratio,
        }

    def _analyze_certainty(
        self, word_counts: Counter, word_count: int
    ) -> Dict[str, Any]:
        """Analyze certainty vs tentative language."""
        certainty_count = sum(word_counts.get(w, 0) for w in self.cognitive_words["certainty"])
        certainty_ratio = certainty_count / word_count if word_count > 0 else 0

        tentative_count = sum(word_counts.get(w, 0) for w in self.cognitive_words["tentative"])
        tentative_ratio = tentative_count / word_count if word_count > 0 else 0

        total = certainty_count + tentative_count
        if total > 0:
            certainty_vs_tentative = certainty_count / total
        else:
            certainty_vs_tentative = 0.5

        return {
            "certainty_ratio": certainty_ratio,
            "tentative_ratio": tentative_ratio,
            "certainty_vs_tentative": certainty_vs_tentative,
        }

    def _analyze_time_focus(
        self, word_counts: Counter, word_count: int
    ) -> Dict[str, Any]:
        """Analyze temporal focus (past, present, future)."""
        time_words = self.cognitive_words["time_orientation"]

        past_count = sum(word_counts.get(w, 0) for w in time_words["past"])
        present_count = sum(word_counts.get(w, 0) for w in time_words["present"])
        future_count = sum(word_counts.get(w, 0) for w in time_words["future"])

        past_ratio = past_count / word_count if word_count > 0 else 0
        present_ratio = present_count / word_count if word_count > 0 else 0
        future_ratio = future_count / word_count if word_count > 0 else 0

        total = past_count + present_count + future_count
        if total > 0:
            ratios = {
                "past": past_count / total,
                "present": present_count / total,
                "future": future_count / total,
            }
            orientation = max(ratios.items(), key=lambda x: x[1])[0]

            if max(ratios.values()) < 0.4:
                orientation = "balanced"
        else:
            orientation = "balanced"

        return {
            "past_focus": past_ratio,
            "present_focus": present_ratio,
            "future_focus": future_ratio,
            "time_orientation": orientation,
        }

    def _determine_cognitive_style(self, results: Dict[str, Any]) -> str:
        """Determine overall cognitive style label."""
        analytical = results.get("analytical_score", 0)
        complexity = results.get("cognitive_complexity", 0)
        certainty = results.get("certainty_vs_tentative", 0.5)

        if analytical > 0.6 and complexity > 0.5:
            return "highly analytical"
        elif analytical > 0.4:
            return "analytical"
        elif complexity > 0.6:
            return "complex/nuanced"
        elif certainty > 0.7:
            return "decisive/certain"
        elif certainty < 0.3:
            return "exploratory/tentative"
        else:
            return "balanced"

    def _generate_summary(self, results: Dict[str, Any]) -> str:
        """Generate human-readable cognitive style summary."""
        style = results.get("cognitive_style", "balanced")
        time_orientation = results.get("time_orientation", "balanced")
        certainty = results.get("certainty_vs_tentative", 0.5)

        parts = []

        style_descriptions = {
            "highly analytical": "Highly analytical thinking with careful logical reasoning",
            "analytical": "Analytical thinking style with attention to cause and effect",
            "complex/nuanced": "Complex, nuanced thinking with consideration of multiple perspectives",
            "decisive/certain": "Decisive thinking style with strong convictions",
            "exploratory/tentative": "Exploratory thinking style, open to possibilities",
            "balanced": "Balanced cognitive style",
        }
        parts.append(style_descriptions.get(style, ""))

        time_descriptions = {
            "past": "Focus on past events and experiences",
            "present": "Focus on current situations",
            "future": "Forward-looking, future-oriented thinking",
            "balanced": "Balanced temporal perspective",
        }
        if time_orientation != "balanced":
            parts.append(time_descriptions.get(time_orientation, ""))

        return ". ".join(p for p in parts if p) + "."

    def get_cognitive_profile(self, text: str) -> Dict[str, Any]:
        """
        Get a simplified cognitive profile for comparison.

        Returns key metrics suitable for profile comparison.
        """
        full_analysis = self.analyze(text)

        return {
            "analytical": full_analysis.get("analytical_score", 0),
            "complexity": full_analysis.get("cognitive_complexity", 0),
            "certainty": full_analysis.get("certainty_vs_tentative", 0.5),
            "time_focus": full_analysis.get("time_orientation", "balanced"),
            "style": full_analysis.get("cognitive_style", "balanced"),
        }
