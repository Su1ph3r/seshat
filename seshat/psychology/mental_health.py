"""
Mental health indicator analysis based on linguistic markers.

IMPORTANT: These are research-validated linguistic correlations, NOT diagnostic tools.
They should never be used to diagnose or treat mental health conditions.
"""

from typing import Dict, List, Any
from collections import Counter

from seshat.utils import tokenize_words, tokenize_sentences


class MentalHealthIndicators:
    """
    Analyze text for linguistic markers associated with mental health patterns.

    DISCLAIMER: This analysis identifies linguistic patterns that research has
    correlated with certain psychological states. It is NOT a diagnostic tool
    and should never be used as a substitute for professional mental health
    assessment or treatment.
    """

    def __init__(self):
        """Initialize mental health indicator analyzer."""
        self.indicator_words = self._load_indicator_dictionaries()

    def _load_indicator_dictionaries(self) -> Dict[str, Dict[str, List[str]]]:
        """Load mental health indicator word lists based on research."""
        return {
            "depression": {
                "negative_emotion": [
                    "sad", "depressed", "unhappy", "miserable", "hopeless",
                    "worthless", "empty", "numb", "meaningless", "pointless",
                    "tired", "exhausted", "drained", "alone", "lonely",
                ],
                "absolutist": [
                    "always", "never", "nothing", "everything", "completely",
                    "totally", "absolutely", "entirely", "constantly",
                ],
                "self_focus": [
                    "i", "me", "my", "mine", "myself",
                ],
            },
            "anxiety": {
                "worry": [
                    "worried", "worry", "anxious", "nervous", "scared",
                    "afraid", "fear", "panic", "terrified", "dread",
                    "stressed", "tense", "uneasy", "restless",
                ],
                "uncertainty": [
                    "maybe", "might", "could", "possibly", "perhaps",
                    "uncertain", "unsure", "doubt", "wondering", "confused",
                ],
                "future_threat": [
                    "will", "going to", "what if", "suppose", "imagine",
                    "expect", "anticipate", "predict", "happen",
                ],
            },
            "anger": {
                "anger_words": [
                    "angry", "mad", "furious", "rage", "hate", "hated",
                    "annoyed", "frustrated", "irritated", "pissed",
                    "resentful", "bitter", "hostile", "aggressive",
                ],
                "blame": [
                    "fault", "blame", "blamed", "responsible", "caused",
                    "because of", "their fault", "your fault",
                ],
            },
            "positive_wellbeing": {
                "positive_emotion": [
                    "happy", "joy", "joyful", "grateful", "thankful",
                    "blessed", "content", "satisfied", "peaceful", "calm",
                    "excited", "enthusiastic", "hopeful", "optimistic",
                ],
                "social_connection": [
                    "friends", "family", "loved", "together", "support",
                    "supported", "connected", "belong", "community",
                ],
                "meaning": [
                    "purpose", "meaningful", "fulfilling", "rewarding",
                    "worthwhile", "matter", "matters", "important",
                ],
            },
        }

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for mental health-related linguistic patterns.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary with mental health indicator analysis

        Note: Results are linguistic correlations, NOT diagnoses.
        """
        if not text:
            return self._empty_results()

        words = tokenize_words(text)
        sentences = tokenize_sentences(text)

        if not words:
            return self._empty_results()

        word_count = len(words)
        word_counts = Counter(words)

        results = {
            "disclaimer": "These are linguistic correlations, NOT clinical diagnoses.",
        }

        depression = self._analyze_depression_markers(word_counts, word_count, sentences)
        results["depression_indicators"] = depression

        anxiety = self._analyze_anxiety_markers(word_counts, word_count)
        results["anxiety_indicators"] = anxiety

        anger = self._analyze_anger_markers(word_counts, word_count)
        results["anger_indicators"] = anger

        wellbeing = self._analyze_wellbeing_markers(word_counts, word_count)
        results["wellbeing_indicators"] = wellbeing

        results["overall_tone"] = self._determine_overall_tone(results)
        results["summary"] = self._generate_summary(results)

        return results

    def _empty_results(self) -> Dict[str, Any]:
        """Return empty results structure."""
        return {
            "disclaimer": "These are linguistic correlations, NOT clinical diagnoses.",
            "depression_indicators": {
                "score": 0.0,
                "negative_emotion_ratio": 0.0,
                "absolutist_ratio": 0.0,
                "self_focus_ratio": 0.0,
            },
            "anxiety_indicators": {
                "score": 0.0,
                "worry_ratio": 0.0,
                "uncertainty_ratio": 0.0,
            },
            "anger_indicators": {
                "score": 0.0,
                "anger_ratio": 0.0,
                "blame_ratio": 0.0,
            },
            "wellbeing_indicators": {
                "score": 0.0,
                "positive_emotion_ratio": 0.0,
                "social_connection_ratio": 0.0,
            },
            "overall_tone": "neutral",
            "summary": "",
        }

    def _analyze_depression_markers(
        self, word_counts: Counter, word_count: int, sentences: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze linguistic markers correlated with depression.

        Research shows depression is associated with:
        - Increased first-person singular pronouns
        - More negative emotion words
        - More absolutist words ("always", "never", "nothing")
        - Past tense focus
        """
        indicators = self.indicator_words["depression"]

        negative_count = sum(
            word_counts.get(w, 0) for w in indicators["negative_emotion"]
        )
        negative_ratio = negative_count / word_count if word_count > 0 else 0

        absolutist_count = sum(
            word_counts.get(w, 0) for w in indicators["absolutist"]
        )
        absolutist_ratio = absolutist_count / word_count if word_count > 0 else 0

        self_focus_count = sum(
            word_counts.get(w, 0) for w in indicators["self_focus"]
        )
        self_focus_ratio = self_focus_count / word_count if word_count > 0 else 0

        past_tense = ["was", "were", "had", "did", "been", "went", "said"]
        past_count = sum(word_counts.get(w, 0) for w in past_tense)
        past_ratio = past_count / word_count if word_count > 0 else 0

        score = (
            negative_ratio * 15 +
            absolutist_ratio * 10 +
            (self_focus_ratio - 0.05) * 5 +
            past_ratio * 3
        )
        score = max(0, min(1, score))

        return {
            "score": score,
            "negative_emotion_ratio": negative_ratio,
            "absolutist_ratio": absolutist_ratio,
            "self_focus_ratio": self_focus_ratio,
            "past_focus_ratio": past_ratio,
        }

    def _analyze_anxiety_markers(
        self, word_counts: Counter, word_count: int
    ) -> Dict[str, Any]:
        """
        Analyze linguistic markers correlated with anxiety.

        Research shows anxiety is associated with:
        - More worry/fear words
        - More uncertainty markers
        - Future-oriented concerns
        - Shorter sentences (in some contexts)
        """
        indicators = self.indicator_words["anxiety"]

        worry_count = sum(word_counts.get(w, 0) for w in indicators["worry"])
        worry_ratio = worry_count / word_count if word_count > 0 else 0

        uncertainty_count = sum(
            word_counts.get(w, 0) for w in indicators["uncertainty"]
        )
        uncertainty_ratio = uncertainty_count / word_count if word_count > 0 else 0

        score = (
            worry_ratio * 15 +
            uncertainty_ratio * 8
        )
        score = max(0, min(1, score))

        return {
            "score": score,
            "worry_ratio": worry_ratio,
            "uncertainty_ratio": uncertainty_ratio,
        }

    def _analyze_anger_markers(
        self, word_counts: Counter, word_count: int
    ) -> Dict[str, Any]:
        """Analyze linguistic markers associated with anger."""
        indicators = self.indicator_words["anger"]

        anger_count = sum(word_counts.get(w, 0) for w in indicators["anger_words"])
        anger_ratio = anger_count / word_count if word_count > 0 else 0

        blame_count = sum(word_counts.get(w, 0) for w in indicators["blame"])
        blame_ratio = blame_count / word_count if word_count > 0 else 0

        score = (anger_ratio * 15 + blame_ratio * 10)
        score = max(0, min(1, score))

        return {
            "score": score,
            "anger_ratio": anger_ratio,
            "blame_ratio": blame_ratio,
        }

    def _analyze_wellbeing_markers(
        self, word_counts: Counter, word_count: int
    ) -> Dict[str, Any]:
        """Analyze linguistic markers associated with positive wellbeing."""
        indicators = self.indicator_words["positive_wellbeing"]

        positive_count = sum(
            word_counts.get(w, 0) for w in indicators["positive_emotion"]
        )
        positive_ratio = positive_count / word_count if word_count > 0 else 0

        social_count = sum(
            word_counts.get(w, 0) for w in indicators["social_connection"]
        )
        social_ratio = social_count / word_count if word_count > 0 else 0

        meaning_count = sum(
            word_counts.get(w, 0) for w in indicators["meaning"]
        )
        meaning_ratio = meaning_count / word_count if word_count > 0 else 0

        score = (
            positive_ratio * 10 +
            social_ratio * 8 +
            meaning_ratio * 5
        )
        score = max(0, min(1, score))

        return {
            "score": score,
            "positive_emotion_ratio": positive_ratio,
            "social_connection_ratio": social_ratio,
            "meaning_ratio": meaning_ratio,
        }

    def _determine_overall_tone(self, results: Dict[str, Any]) -> str:
        """Determine overall emotional tone."""
        depression = results["depression_indicators"]["score"]
        anxiety = results["anxiety_indicators"]["score"]
        anger = results["anger_indicators"]["score"]
        wellbeing = results["wellbeing_indicators"]["score"]

        if wellbeing > 0.3 and wellbeing > max(depression, anxiety, anger):
            return "positive"

        if depression > 0.3 or anxiety > 0.3:
            if depression > anxiety:
                return "distressed (depressive markers)"
            else:
                return "distressed (anxiety markers)"

        if anger > 0.3:
            return "frustrated/angry"

        return "neutral"

    def _generate_summary(self, results: Dict[str, Any]) -> str:
        """Generate summary of findings with appropriate caveats."""
        tone = results.get("overall_tone", "neutral")

        parts = [
            "IMPORTANT: This analysis identifies linguistic patterns only.",
            "It is NOT a clinical assessment or diagnosis.",
            "",
        ]

        if tone == "positive":
            parts.append(
                "Text contains elevated positive emotional language and social connection markers."
            )
        elif "distressed" in tone:
            if "depressive" in tone:
                parts.append(
                    "Text contains linguistic patterns sometimes associated with low mood, "
                    "including elevated self-focus and absolutist language."
                )
            else:
                parts.append(
                    "Text contains linguistic patterns sometimes associated with worry, "
                    "including uncertainty markers and concern-related language."
                )
        elif tone == "frustrated/angry":
            parts.append(
                "Text contains elevated anger-related and blame-oriented language."
            )
        else:
            parts.append(
                "No notable patterns detected. Linguistic markers within typical ranges."
            )

        return " ".join(parts)

    def get_indicator_summary(self, text: str) -> Dict[str, float]:
        """
        Get a simplified summary of indicator scores.

        Returns normalized scores (0-1) for each category.
        """
        full_analysis = self.analyze(text)

        return {
            "depression_markers": full_analysis["depression_indicators"]["score"],
            "anxiety_markers": full_analysis["anxiety_indicators"]["score"],
            "anger_markers": full_analysis["anger_indicators"]["score"],
            "wellbeing_markers": full_analysis["wellbeing_indicators"]["score"],
        }
