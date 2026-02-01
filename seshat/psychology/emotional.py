"""
Emotional tone analysis for psychological profiling.

Analyzes sentiment, emotional intensity, and affective patterns in text.
"""

from typing import Dict, List, Any
from collections import Counter

from seshat.utils import tokenize_words, tokenize_sentences


class EmotionalAnalyzer:
    """
    Analyze emotional tone and sentiment in text.
    """

    def __init__(self):
        """Initialize emotional analyzer with sentiment dictionaries."""
        self.emotion_words = self._load_emotion_dictionaries()
        self.intensifiers = self._load_intensifiers()

    def _load_emotion_dictionaries(self) -> Dict[str, List[str]]:
        """Load emotion-related word lists."""
        return {
            "positive_affect": [
                "happy", "joy", "love", "loved", "loving", "great", "good",
                "wonderful", "amazing", "awesome", "excellent", "fantastic",
                "beautiful", "perfect", "lovely", "delightful", "pleased",
                "glad", "cheerful", "excited", "thrilled", "grateful",
                "thankful", "blessed", "fortunate", "lucky", "hopeful",
                "optimistic", "confident", "proud", "satisfied", "content",
            ],
            "negative_affect": [
                "sad", "unhappy", "depressed", "miserable", "terrible",
                "awful", "horrible", "bad", "worse", "worst", "hate",
                "hated", "hating", "angry", "furious", "frustrated",
                "disappointed", "upset", "annoyed", "irritated", "disgusted",
                "afraid", "scared", "frightened", "worried", "anxious",
                "nervous", "stressed", "overwhelmed", "hurt", "pain",
            ],
            "anxiety": [
                "anxious", "anxiety", "worried", "worry", "nervous",
                "stressed", "stress", "panic", "fear", "afraid", "scared",
                "frightened", "terrified", "uneasy", "apprehensive",
                "concerned", "tense", "restless", "agitated", "distressed",
            ],
            "anger": [
                "angry", "anger", "mad", "furious", "rage", "outraged",
                "frustrated", "irritated", "annoyed", "pissed", "hate",
                "hostile", "aggressive", "resentful", "bitter", "disgusted",
            ],
            "sadness": [
                "sad", "sadness", "unhappy", "depressed", "depression",
                "miserable", "hopeless", "despair", "grief", "mourning",
                "heartbroken", "lonely", "loneliness", "melancholy",
                "gloomy", "down", "blue", "disappointed", "hurt",
            ],
            "joy": [
                "happy", "happiness", "joy", "joyful", "delighted",
                "pleased", "glad", "cheerful", "elated", "ecstatic",
                "thrilled", "excited", "enthusiastic", "blissful",
            ],
            "surprise": [
                "surprised", "surprise", "shocked", "amazed", "astonished",
                "stunned", "unexpected", "wow", "unbelievable", "incredible",
            ],
            "disgust": [
                "disgusted", "disgust", "revolting", "repulsive", "gross",
                "nauseating", "sickening", "appalling", "offensive",
            ],
            "trust": [
                "trust", "believe", "faith", "confident", "reliable",
                "honest", "sincere", "loyal", "faithful", "dependable",
            ],
            "anticipation": [
                "anticipate", "expect", "hope", "hopeful", "eager",
                "looking forward", "excited", "await", "planning",
            ],
        }

    def _load_intensifiers(self) -> Dict[str, float]:
        """Load intensity modifiers with their multipliers."""
        return {
            "very": 1.5,
            "really": 1.4,
            "extremely": 2.0,
            "incredibly": 1.8,
            "absolutely": 1.9,
            "totally": 1.6,
            "completely": 1.7,
            "utterly": 1.8,
            "quite": 1.2,
            "fairly": 1.1,
            "somewhat": 0.8,
            "slightly": 0.6,
            "barely": 0.4,
            "hardly": 0.3,
            "super": 1.6,
            "so": 1.4,
            "too": 1.3,
            "insanely": 1.9,
            "ridiculously": 1.7,
        }

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze emotional content of text.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary with emotional analysis results
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

        sentiment = self._analyze_sentiment(word_counts, word_count)
        results.update(sentiment)

        emotions = self._analyze_emotions(word_counts, word_count)
        results["emotions"] = emotions

        intensity = self._analyze_intensity(text, words, word_count)
        results.update(intensity)

        authenticity = self._analyze_authenticity(word_counts, word_count, text)
        results["authenticity_score"] = authenticity

        results["dominant_emotion"] = self._get_dominant_emotion(emotions)
        results["emotional_summary"] = self._generate_summary(results)

        return results

    def _empty_results(self) -> Dict[str, Any]:
        """Return empty results structure."""
        return {
            "positive_ratio": 0.0,
            "negative_ratio": 0.0,
            "sentiment_score": 0.0,
            "sentiment_label": "neutral",
            "emotions": {},
            "emotional_intensity": 0.0,
            "intensifier_ratio": 0.0,
            "emotional_range": 0.0,
            "authenticity_score": 0.5,
            "dominant_emotion": "neutral",
            "emotional_summary": "",
        }

    def _analyze_sentiment(
        self, word_counts: Counter, word_count: int
    ) -> Dict[str, Any]:
        """Analyze overall sentiment polarity."""
        positive_words = self.emotion_words["positive_affect"]
        negative_words = self.emotion_words["negative_affect"]

        positive_count = sum(word_counts.get(w, 0) for w in positive_words)
        negative_count = sum(word_counts.get(w, 0) for w in negative_words)

        positive_ratio = positive_count / word_count if word_count > 0 else 0
        negative_ratio = negative_count / word_count if word_count > 0 else 0

        total_emotional = positive_count + negative_count
        if total_emotional > 0:
            sentiment_score = (positive_count - negative_count) / total_emotional
        else:
            sentiment_score = 0

        if sentiment_score > 0.3:
            sentiment_label = "positive"
        elif sentiment_score < -0.3:
            sentiment_label = "negative"
        else:
            sentiment_label = "neutral"

        return {
            "positive_ratio": positive_ratio,
            "negative_ratio": negative_ratio,
            "sentiment_score": sentiment_score,
            "sentiment_label": sentiment_label,
        }

    def _analyze_emotions(
        self, word_counts: Counter, word_count: int
    ) -> Dict[str, float]:
        """Analyze specific emotion categories."""
        emotions = {}

        emotion_categories = ["anxiety", "anger", "sadness", "joy", "surprise", "disgust", "trust", "anticipation"]

        for category in emotion_categories:
            words = self.emotion_words.get(category, [])
            count = sum(word_counts.get(w, 0) for w in words)
            emotions[category] = count / word_count if word_count > 0 else 0

        return emotions

    def _analyze_intensity(
        self, text: str, words: List[str], word_count: int
    ) -> Dict[str, Any]:
        """Analyze emotional intensity."""
        intensifier_count = sum(1 for w in words if w in self.intensifiers)
        intensifier_ratio = intensifier_count / word_count if word_count > 0 else 0

        superlatives = ["best", "worst", "most", "least", "greatest", "finest", "highest", "lowest"]
        superlative_count = sum(1 for w in words if w in superlatives)

        exclamation_count = text.count("!")
        question_count = text.count("?")

        all_caps_pattern = sum(1 for word in text.split() if word.isupper() and len(word) > 1)

        intensity_score = (
            intensifier_ratio * 2 +
            (superlative_count / word_count) * 3 +
            (exclamation_count / word_count) * 2 +
            (all_caps_pattern / word_count)
        ) if word_count > 0 else 0

        intensity_score = min(intensity_score, 1.0)

        emotion_categories = list(self.emotion_words.keys())
        emotions_present = set()
        for cat in emotion_categories:
            for word in self.emotion_words[cat]:
                if word in words:
                    emotions_present.add(cat)
                    break

        emotional_range = len(emotions_present) / len(emotion_categories)

        return {
            "emotional_intensity": intensity_score,
            "intensifier_ratio": intensifier_ratio,
            "superlative_ratio": superlative_count / word_count if word_count > 0 else 0,
            "exclamation_ratio": exclamation_count / word_count if word_count > 0 else 0,
            "emotional_range": emotional_range,
        }

    def _analyze_authenticity(
        self, word_counts: Counter, word_count: int, text: str
    ) -> float:
        """
        Calculate authenticity score based on linguistic markers.

        Higher authenticity is associated with:
        - More first-person singular pronouns
        - More negative emotion words
        - Fewer exclusive words (but, except)
        - Lower hedge word usage
        """
        first_person = sum(word_counts.get(w, 0) for w in ["i", "me", "my", "mine", "myself"])
        first_person_ratio = first_person / word_count if word_count > 0 else 0

        negative_words = self.emotion_words["negative_affect"]
        negative_count = sum(word_counts.get(w, 0) for w in negative_words)
        negative_ratio = negative_count / word_count if word_count > 0 else 0

        exclusive_words = ["but", "except", "without", "however", "although", "unless"]
        exclusive_count = sum(word_counts.get(w, 0) for w in exclusive_words)
        exclusive_ratio = exclusive_count / word_count if word_count > 0 else 0

        hedge_words = ["maybe", "perhaps", "possibly", "probably", "might", "could", "seems"]
        hedge_count = sum(word_counts.get(w, 0) for w in hedge_words)
        hedge_ratio = hedge_count / word_count if word_count > 0 else 0

        authenticity = 0.5
        authenticity += first_person_ratio * 2
        authenticity += negative_ratio * 1.5
        authenticity -= exclusive_ratio * 1.5
        authenticity -= hedge_ratio * 2

        return max(0, min(1, authenticity))

    def _get_dominant_emotion(self, emotions: Dict[str, float]) -> str:
        """Get the dominant emotion category."""
        if not emotions:
            return "neutral"

        max_emotion = max(emotions.items(), key=lambda x: x[1])

        if max_emotion[1] < 0.01:
            return "neutral"

        return max_emotion[0]

    def _generate_summary(self, results: Dict[str, Any]) -> str:
        """Generate human-readable emotional summary."""
        sentiment = results.get("sentiment_label", "neutral")
        intensity = results.get("emotional_intensity", 0)
        dominant = results.get("dominant_emotion", "neutral")
        authenticity = results.get("authenticity_score", 0.5)

        intensity_desc = "high" if intensity > 0.5 else "moderate" if intensity > 0.2 else "low"

        parts = []

        if sentiment != "neutral":
            parts.append(f"Overall {sentiment} sentiment")
        else:
            parts.append("Emotionally neutral tone")

        parts.append(f"with {intensity_desc} emotional intensity")

        if dominant != "neutral":
            parts.append(f"Primary emotion: {dominant}")

        if authenticity > 0.6:
            parts.append("Writing appears authentic/genuine")
        elif authenticity < 0.4:
            parts.append("Writing may be guarded/filtered")

        return ". ".join(parts) + "."

    def get_emotion_trajectory(
        self, text: str, window_size: int = 50
    ) -> List[Dict[str, float]]:
        """
        Analyze emotional trajectory through the text.

        Args:
            text: Input text
            window_size: Number of words per window

        Returns:
            List of emotion scores for each window
        """
        words = tokenize_words(text)

        if len(words) < window_size:
            return [self._analyze_emotions(Counter(words), len(words))]

        trajectory = []
        for i in range(0, len(words) - window_size + 1, window_size // 2):
            window = words[i:i + window_size]
            window_counts = Counter(window)
            emotions = self._analyze_emotions(window_counts, len(window))
            emotions["position"] = i / len(words)
            trajectory.append(emotions)

        return trajectory
