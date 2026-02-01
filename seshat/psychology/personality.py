"""
Big Five (OCEAN) personality trait analysis based on linguistic markers.

Based on research correlating language use with personality traits,
including work by Pennebaker et al. and the LIWC framework.
"""

from typing import Dict, List, Any
from collections import Counter

from seshat.utils import tokenize_words, tokenize_sentences, safe_divide


class PersonalityAnalyzer:
    """
    Analyze text for Big Five personality trait indicators.

    The Big Five traits are:
    - Openness to Experience
    - Conscientiousness
    - Extraversion
    - Agreeableness
    - Neuroticism
    """

    def __init__(self):
        """Initialize personality analyzer with trait word dictionaries."""
        self.trait_words = self._load_trait_dictionaries()

    def _load_trait_dictionaries(self) -> Dict[str, Dict[str, List[str]]]:
        """Load word dictionaries associated with each trait."""
        return {
            "openness": {
                "high": [
                    "abstract", "art", "artistic", "beauty", "complex", "create",
                    "creative", "creativity", "curious", "dream", "fantasy",
                    "imagination", "imaginative", "innovate", "innovative",
                    "insight", "intellectual", "interesting", "invent", "novel",
                    "original", "philosophical", "poetry", "reflect", "theoretical",
                    "unconventional", "unique", "wonder", "aesthetic", "conceptual",
                ],
                "low": [
                    "basic", "common", "conventional", "familiar", "normal",
                    "ordinary", "plain", "practical", "routine", "simple",
                    "standard", "traditional", "typical", "usual",
                ],
            },
            "conscientiousness": {
                "high": [
                    "accomplish", "achieve", "achievement", "careful", "complete",
                    "deadline", "detail", "diligent", "discipline", "efficient",
                    "exact", "goal", "hard-working", "meticulous", "organize",
                    "organized", "plan", "precise", "prepared", "productive",
                    "punctual", "responsible", "schedule", "systematic", "thorough",
                    "success", "succeed", "duty", "obligation", "commitment",
                ],
                "low": [
                    "careless", "disorganized", "forget", "forgot", "ignore",
                    "impulsive", "lazy", "messy", "neglect", "procrastinate",
                    "random", "reckless", "scattered", "sloppy", "spontaneous",
                ],
            },
            "extraversion": {
                "high": [
                    "active", "adventure", "bold", "celebrate", "cheerful",
                    "confident", "energetic", "enthusiastic", "excited", "exciting",
                    "extrovert", "friendly", "fun", "group", "happy", "lively",
                    "loud", "outgoing", "party", "social", "sociable", "talk",
                    "talkative", "team", "together", "vibrant", "we", "us", "our",
                ],
                "low": [
                    "alone", "avoid", "distant", "introvert", "isolated", "lonely",
                    "private", "quiet", "reserved", "shy", "silent", "solitary",
                    "withdrawn",
                ],
            },
            "agreeableness": {
                "high": [
                    "agree", "agreeable", "altruistic", "appreciate", "caring",
                    "compassion", "considerate", "cooperate", "cooperation",
                    "empathy", "fair", "forgive", "friendly", "generous", "gentle",
                    "grateful", "harmony", "help", "helpful", "honest", "kind",
                    "kindness", "nice", "patient", "polite", "sincere", "support",
                    "supportive", "sympathetic", "thank", "trust", "understand",
                    "warm", "welcome",
                ],
                "low": [
                    "aggressive", "angry", "annoyed", "argue", "arrogant", "blame",
                    "complain", "competitive", "critical", "cruel", "cynical",
                    "demanding", "disagree", "dislike", "distrust", "hostile",
                    "impatient", "insult", "jealous", "rude", "selfish",
                    "stubborn", "suspicious", "unfriendly",
                ],
            },
            "neuroticism": {
                "high": [
                    "afraid", "anger", "angry", "anxious", "anxiety", "bad",
                    "bitter", "confused", "depressed", "depression", "desperate",
                    "disappoint", "disappointed", "distress", "doubt", "embarrassed",
                    "fail", "failure", "fear", "frustrated", "guilt", "guilty",
                    "helpless", "hopeless", "hurt", "insecure", "irritated",
                    "lonely", "miserable", "nervous", "overwhelmed", "panic",
                    "regret", "sad", "scared", "shame", "stress", "stressed",
                    "tense", "terrible", "upset", "vulnerable", "worry", "worried",
                    "worthless",
                ],
                "low": [
                    "balanced", "calm", "composed", "confident", "content",
                    "happy", "peaceful", "relaxed", "secure", "serene", "stable",
                    "steady", "tranquil",
                ],
            },
        }

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for Big Five personality indicators.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary with personality trait scores and analysis
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
            "openness": self._analyze_openness(word_counts, word_count, text),
            "conscientiousness": self._analyze_conscientiousness(word_counts, word_count),
            "extraversion": self._analyze_extraversion(word_counts, word_count, text),
            "agreeableness": self._analyze_agreeableness(word_counts, word_count),
            "neuroticism": self._analyze_neuroticism(word_counts, word_count),
        }

        for trait in results:
            results[trait]["score"] = self._calculate_trait_score(results[trait])

        results["summary"] = self._generate_summary(results)
        results["dominant_traits"] = self._get_dominant_traits(results)

        return results

    def _empty_results(self) -> Dict[str, Any]:
        """Return empty results structure."""
        empty_trait = {
            "score": 0.5,
            "high_markers": 0,
            "low_markers": 0,
            "indicator_ratio": 0.0,
        }
        return {
            "openness": empty_trait.copy(),
            "conscientiousness": empty_trait.copy(),
            "extraversion": empty_trait.copy(),
            "agreeableness": empty_trait.copy(),
            "neuroticism": empty_trait.copy(),
            "summary": "",
            "dominant_traits": [],
        }

    def _count_trait_words(
        self, word_counts: Counter, word_list: List[str]
    ) -> int:
        """Count occurrences of trait-associated words."""
        return sum(word_counts.get(word, 0) for word in word_list)

    def _analyze_openness(
        self, word_counts: Counter, word_count: int, text: str
    ) -> Dict[str, Any]:
        """Analyze openness to experience indicators."""
        high_count = self._count_trait_words(word_counts, self.trait_words["openness"]["high"])
        low_count = self._count_trait_words(word_counts, self.trait_words["openness"]["low"])

        unique_words = len(word_counts)
        ttr = unique_words / word_count if word_count > 0 else 0

        question_count = text.count("?")
        question_ratio = question_count / word_count * 100 if word_count > 0 else 0

        avg_word_length = sum(len(w) for w in word_counts.keys()) / len(word_counts) if word_counts else 0

        return {
            "high_markers": high_count,
            "low_markers": low_count,
            "indicator_ratio": (high_count - low_count) / word_count if word_count > 0 else 0,
            "vocabulary_diversity": ttr,
            "question_ratio": question_ratio,
            "avg_word_length": avg_word_length,
        }

    def _analyze_conscientiousness(
        self, word_counts: Counter, word_count: int
    ) -> Dict[str, Any]:
        """Analyze conscientiousness indicators."""
        high_count = self._count_trait_words(word_counts, self.trait_words["conscientiousness"]["high"])
        low_count = self._count_trait_words(word_counts, self.trait_words["conscientiousness"]["low"])

        certainty_words = ["always", "never", "definitely", "certainly", "absolutely", "must", "should"]
        certainty_count = sum(word_counts.get(w, 0) for w in certainty_words)

        future_words = ["will", "going", "plan", "future", "tomorrow", "next"]
        future_count = sum(word_counts.get(w, 0) for w in future_words)

        return {
            "high_markers": high_count,
            "low_markers": low_count,
            "indicator_ratio": (high_count - low_count) / word_count if word_count > 0 else 0,
            "certainty_ratio": certainty_count / word_count if word_count > 0 else 0,
            "future_orientation": future_count / word_count if word_count > 0 else 0,
        }

    def _analyze_extraversion(
        self, word_counts: Counter, word_count: int, text: str
    ) -> Dict[str, Any]:
        """Analyze extraversion indicators."""
        high_count = self._count_trait_words(word_counts, self.trait_words["extraversion"]["high"])
        low_count = self._count_trait_words(word_counts, self.trait_words["extraversion"]["low"])

        first_plural = sum(word_counts.get(w, 0) for w in ["we", "us", "our", "ours"])

        positive_words = ["happy", "great", "good", "love", "awesome", "amazing", "wonderful", "excellent", "fantastic", "excited"]
        positive_count = sum(word_counts.get(w, 0) for w in positive_words)

        exclamation_count = text.count("!")

        return {
            "high_markers": high_count,
            "low_markers": low_count,
            "indicator_ratio": (high_count - low_count) / word_count if word_count > 0 else 0,
            "first_person_plural_ratio": first_plural / word_count if word_count > 0 else 0,
            "positive_emotion_ratio": positive_count / word_count if word_count > 0 else 0,
            "exclamation_ratio": exclamation_count / word_count if word_count > 0 else 0,
            "word_count": word_count,
        }

    def _analyze_agreeableness(
        self, word_counts: Counter, word_count: int
    ) -> Dict[str, Any]:
        """Analyze agreeableness indicators."""
        high_count = self._count_trait_words(word_counts, self.trait_words["agreeableness"]["high"])
        low_count = self._count_trait_words(word_counts, self.trait_words["agreeableness"]["low"])

        agreement_words = ["yes", "agree", "okay", "ok", "sure", "right", "true"]
        agreement_count = sum(word_counts.get(w, 0) for w in agreement_words)

        inclusive_words = ["we", "us", "our", "together", "everyone", "all"]
        inclusive_count = sum(word_counts.get(w, 0) for w in inclusive_words)

        swear_words = ["damn", "hell", "crap", "shit", "fuck", "ass", "bastard"]
        swear_count = sum(word_counts.get(w, 0) for w in swear_words)

        return {
            "high_markers": high_count,
            "low_markers": low_count,
            "indicator_ratio": (high_count - low_count) / word_count if word_count > 0 else 0,
            "agreement_ratio": agreement_count / word_count if word_count > 0 else 0,
            "inclusive_ratio": inclusive_count / word_count if word_count > 0 else 0,
            "profanity_ratio": swear_count / word_count if word_count > 0 else 0,
        }

    def _analyze_neuroticism(
        self, word_counts: Counter, word_count: int
    ) -> Dict[str, Any]:
        """Analyze neuroticism indicators."""
        high_count = self._count_trait_words(word_counts, self.trait_words["neuroticism"]["high"])
        low_count = self._count_trait_words(word_counts, self.trait_words["neuroticism"]["low"])

        first_singular = sum(word_counts.get(w, 0) for w in ["i", "me", "my", "mine", "myself"])

        negative_words = ["not", "no", "never", "nothing", "nowhere", "neither", "nobody", "none"]
        negative_count = sum(word_counts.get(w, 0) for w in negative_words)

        hedge_words = ["maybe", "perhaps", "possibly", "might", "could", "probably", "seems", "apparently"]
        hedge_count = sum(word_counts.get(w, 0) for w in hedge_words)

        past_words = ["was", "were", "had", "did", "been", "went", "said", "told"]
        past_count = sum(word_counts.get(w, 0) for w in past_words)

        return {
            "high_markers": high_count,
            "low_markers": low_count,
            "indicator_ratio": (high_count - low_count) / word_count if word_count > 0 else 0,
            "first_person_singular_ratio": first_singular / word_count if word_count > 0 else 0,
            "negation_ratio": negative_count / word_count if word_count > 0 else 0,
            "hedge_ratio": hedge_count / word_count if word_count > 0 else 0,
            "past_tense_ratio": past_count / word_count if word_count > 0 else 0,
        }

    def _calculate_trait_score(self, trait_data: Dict[str, Any]) -> float:
        """
        Calculate overall trait score (0-1 scale).

        0.5 is neutral, <0.5 is low, >0.5 is high.
        """
        ratio = trait_data.get("indicator_ratio", 0)

        score = 0.5 + (ratio * 10)

        return max(0, min(1, score))

    def _get_dominant_traits(self, results: Dict[str, Any]) -> List[str]:
        """Get traits with notably high or low scores."""
        traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

        dominant = []
        for trait in traits:
            score = results[trait].get("score", 0.5)
            if score >= 0.65:
                dominant.append(f"High {trait.capitalize()}")
            elif score <= 0.35:
                dominant.append(f"Low {trait.capitalize()}")

        return dominant

    def _generate_summary(self, results: Dict[str, Any]) -> str:
        """Generate a human-readable personality summary."""
        traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
        descriptions = []

        for trait in traits:
            score = results[trait].get("score", 0.5)

            if score >= 0.65:
                level = "high"
            elif score <= 0.35:
                level = "low"
            else:
                level = "moderate"

            trait_descriptions = {
                "openness": {
                    "high": "intellectually curious and open to new experiences",
                    "low": "practical and conventional in thinking",
                    "moderate": "balanced between traditional and novel approaches",
                },
                "conscientiousness": {
                    "high": "organized, disciplined, and achievement-oriented",
                    "low": "flexible and spontaneous",
                    "moderate": "reasonably organized with some flexibility",
                },
                "extraversion": {
                    "high": "outgoing, energetic, and socially engaged",
                    "low": "reserved and introspective",
                    "moderate": "balanced between social engagement and solitude",
                },
                "agreeableness": {
                    "high": "cooperative, trusting, and considerate",
                    "low": "competitive and skeptical",
                    "moderate": "balanced between cooperation and assertiveness",
                },
                "neuroticism": {
                    "high": "emotionally sensitive and prone to stress",
                    "low": "emotionally stable and resilient",
                    "moderate": "generally stable with normal emotional variation",
                },
            }

            descriptions.append(trait_descriptions[trait][level])

        return "Writing suggests someone who is " + "; ".join(descriptions[:3]) + "."
