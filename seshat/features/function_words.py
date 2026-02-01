"""
Function word analysis for stylometric profiling.

Function words (pronouns, articles, prepositions, conjunctions, auxiliary verbs)
are critical for authorship attribution because their usage is largely subconscious
and content-independent.
"""

from collections import Counter
from typing import Dict, List, Any

from seshat.utils import tokenize_words, safe_divide


class FunctionWordFeatures:
    """Extract function word features from text."""

    def __init__(self):
        self.pronouns = self._define_pronouns()
        self.articles = self._define_articles()
        self.prepositions = self._define_prepositions()
        self.conjunctions = self._define_conjunctions()
        self.auxiliary_verbs = self._define_auxiliary_verbs()
        self.all_function_words = self._combine_all()

    def _define_pronouns(self) -> Dict[str, set]:
        """Define pronoun categories."""
        return {
            "first_person_singular": {"i", "me", "my", "mine", "myself"},
            "first_person_plural": {"we", "us", "our", "ours", "ourselves"},
            "second_person": {"you", "your", "yours", "yourself", "yourselves"},
            "third_person_singular": {"he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself"},
            "third_person_plural": {"they", "them", "their", "theirs", "themselves"},
            "demonstrative": {"this", "that", "these", "those"},
            "indefinite": {"all", "another", "any", "anybody", "anyone", "anything", "both", "each", "either", "everybody", "everyone", "everything", "few", "many", "neither", "nobody", "none", "no one", "nothing", "one", "other", "others", "several", "some", "somebody", "someone", "something", "such"},
            "interrogative": {"who", "whom", "whose", "which", "what"},
            "relative": {"who", "whom", "whose", "which", "that", "whoever", "whomever", "whichever", "whatever"},
            "reflexive": {"myself", "yourself", "himself", "herself", "itself", "ourselves", "yourselves", "themselves"},
        }

    def _define_articles(self) -> set:
        """Define articles."""
        return {"a", "an", "the"}

    def _define_prepositions(self) -> set:
        """Define common prepositions."""
        return {
            "about", "above", "across", "after", "against", "along", "among",
            "around", "at", "before", "behind", "below", "beneath", "beside",
            "besides", "between", "beyond", "but", "by", "concerning", "considering",
            "despite", "down", "during", "except", "for", "from", "in", "inside",
            "into", "like", "near", "of", "off", "on", "onto", "out", "outside",
            "over", "past", "regarding", "round", "since", "through", "throughout",
            "till", "to", "toward", "towards", "under", "underneath", "until",
            "unto", "up", "upon", "with", "within", "without",
        }

    def _define_conjunctions(self) -> Dict[str, set]:
        """Define conjunction categories."""
        return {
            "coordinating": {"and", "but", "or", "nor", "for", "yet", "so"},
            "subordinating": {
                "after", "although", "as", "because", "before", "even if",
                "even though", "if", "in order that", "once", "provided that",
                "rather than", "since", "so that", "than", "that", "though",
                "till", "unless", "until", "when", "whenever", "where",
                "whereas", "wherever", "whether", "while",
            },
            "correlative": {"both", "either", "neither", "not only", "whether"},
        }

    def _define_auxiliary_verbs(self) -> set:
        """Define auxiliary verbs."""
        return {
            "am", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "having",
            "do", "does", "did", "doing",
            "will", "would", "shall", "should",
            "may", "might", "must", "can", "could",
            "need", "dare", "ought", "used",
        }

    def _combine_all(self) -> set:
        """Combine all function words into a single set."""
        all_words = set()
        for category in self.pronouns.values():
            all_words.update(category)
        all_words.update(self.articles)
        all_words.update(self.prepositions)
        for category in self.conjunctions.values():
            all_words.update(category)
        all_words.update(self.auxiliary_verbs)
        return all_words

    def extract(self, text: str) -> Dict[str, Any]:
        """
        Extract all function word features from text.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary of function word features
        """
        words = tokenize_words(text)

        if not words:
            return self._empty_features()

        total_words = len(words)
        word_counts = Counter(words)

        features = {}

        pronoun_features = self._extract_pronoun_features(word_counts, total_words)
        features.update(pronoun_features)

        article_features = self._extract_article_features(word_counts, total_words)
        features.update(article_features)

        preposition_features = self._extract_preposition_features(word_counts, total_words)
        features.update(preposition_features)

        conjunction_features = self._extract_conjunction_features(word_counts, total_words)
        features.update(conjunction_features)

        auxiliary_features = self._extract_auxiliary_features(word_counts, total_words)
        features.update(auxiliary_features)

        total_function_words = sum(
            word_counts[w] for w in self.all_function_words if w in word_counts
        )
        features["total_function_word_ratio"] = total_function_words / total_words

        features["function_word_frequencies"] = self._get_individual_frequencies(
            word_counts, total_words
        )

        return features

    def _empty_features(self) -> Dict[str, Any]:
        """Return empty feature dictionary."""
        return {
            "first_person_singular_ratio": 0.0,
            "first_person_plural_ratio": 0.0,
            "second_person_ratio": 0.0,
            "third_person_singular_ratio": 0.0,
            "third_person_plural_ratio": 0.0,
            "first_to_third_person_ratio": 0.0,
            "i_ratio": 0.0,
            "we_ratio": 0.0,
            "you_ratio": 0.0,
            "demonstrative_pronoun_ratio": 0.0,
            "indefinite_pronoun_ratio": 0.0,
            "article_ratio": 0.0,
            "definite_article_ratio": 0.0,
            "indefinite_article_ratio": 0.0,
            "preposition_ratio": 0.0,
            "coordinating_conjunction_ratio": 0.0,
            "subordinating_conjunction_ratio": 0.0,
            "and_ratio": 0.0,
            "but_ratio": 0.0,
            "because_ratio": 0.0,
            "auxiliary_verb_ratio": 0.0,
            "modal_verb_ratio": 0.0,
            "total_function_word_ratio": 0.0,
            "function_word_frequencies": {},
        }

    def _extract_pronoun_features(
        self, word_counts: Counter, total_words: int
    ) -> Dict[str, float]:
        """Extract pronoun-related features."""
        features = {}

        for category, pronouns in self.pronouns.items():
            count = sum(word_counts.get(p, 0) for p in pronouns)
            features[f"{category}_ratio"] = count / total_words

        first_person_count = sum(
            word_counts.get(p, 0)
            for p in self.pronouns["first_person_singular"] | self.pronouns["first_person_plural"]
        )
        third_person_count = sum(
            word_counts.get(p, 0)
            for p in self.pronouns["third_person_singular"] | self.pronouns["third_person_plural"]
        )
        features["first_to_third_person_ratio"] = safe_divide(
            first_person_count, third_person_count, default=0.0
        )

        features["i_ratio"] = word_counts.get("i", 0) / total_words
        features["we_ratio"] = word_counts.get("we", 0) / total_words
        features["you_ratio"] = word_counts.get("you", 0) / total_words

        return features

    def _extract_article_features(
        self, word_counts: Counter, total_words: int
    ) -> Dict[str, float]:
        """Extract article-related features."""
        total_articles = sum(word_counts.get(a, 0) for a in self.articles)
        definite = word_counts.get("the", 0)
        indefinite = word_counts.get("a", 0) + word_counts.get("an", 0)

        return {
            "article_ratio": total_articles / total_words,
            "definite_article_ratio": definite / total_words,
            "indefinite_article_ratio": indefinite / total_words,
        }

    def _extract_preposition_features(
        self, word_counts: Counter, total_words: int
    ) -> Dict[str, float]:
        """Extract preposition-related features."""
        total_prepositions = sum(
            word_counts.get(p, 0) for p in self.prepositions
        )

        return {
            "preposition_ratio": total_prepositions / total_words,
        }

    def _extract_conjunction_features(
        self, word_counts: Counter, total_words: int
    ) -> Dict[str, float]:
        """Extract conjunction-related features."""
        coordinating_count = sum(
            word_counts.get(c, 0) for c in self.conjunctions["coordinating"]
        )
        subordinating_count = sum(
            word_counts.get(c, 0) for c in self.conjunctions["subordinating"]
        )

        return {
            "coordinating_conjunction_ratio": coordinating_count / total_words,
            "subordinating_conjunction_ratio": subordinating_count / total_words,
            "and_ratio": word_counts.get("and", 0) / total_words,
            "but_ratio": word_counts.get("but", 0) / total_words,
            "because_ratio": word_counts.get("because", 0) / total_words,
        }

    def _extract_auxiliary_features(
        self, word_counts: Counter, total_words: int
    ) -> Dict[str, float]:
        """Extract auxiliary verb features."""
        total_auxiliary = sum(
            word_counts.get(v, 0) for v in self.auxiliary_verbs
        )

        modal_verbs = {"will", "would", "shall", "should", "may", "might", "must", "can", "could"}
        modal_count = sum(word_counts.get(v, 0) for v in modal_verbs)

        return {
            "auxiliary_verb_ratio": total_auxiliary / total_words,
            "modal_verb_ratio": modal_count / total_words,
        }

    def _get_individual_frequencies(
        self, word_counts: Counter, total_words: int
    ) -> Dict[str, float]:
        """Get normalized frequency for each function word."""
        frequencies = {}
        for word in self.all_function_words:
            if word_counts.get(word, 0) > 0:
                frequencies[word] = word_counts[word] / total_words
        return frequencies

    def get_function_word_vector(self, text: str) -> List[float]:
        """
        Get a fixed-length vector of function word frequencies.

        Useful for ML models that require fixed feature dimensions.
        """
        words = tokenize_words(text)
        if not words:
            return [0.0] * len(self.all_function_words)

        total_words = len(words)
        word_counts = Counter(words)

        sorted_function_words = sorted(self.all_function_words)
        return [word_counts.get(w, 0) / total_words for w in sorted_function_words]

    def get_function_word_names(self) -> List[str]:
        """Get the ordered list of function word names for the vector."""
        return sorted(self.all_function_words)


# Alias for backward compatibility
FunctionWordExtractor = FunctionWordFeatures
