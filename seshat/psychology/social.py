"""
Social dynamics analysis for psychological profiling.

Analyzes power dynamics, status indicators, affiliation, and social orientation.
"""

from typing import Dict, List, Any
from collections import Counter

from seshat.utils import tokenize_words, tokenize_sentences


class SocialAnalyzer:
    """
    Analyze social dynamics and interpersonal patterns in text.
    """

    def __init__(self):
        """Initialize social analyzer with word categories."""
        self.social_words = self._load_social_dictionaries()

    def _load_social_dictionaries(self) -> Dict[str, List[str]]:
        """Load social-related word lists."""
        return {
            "social_references": [
                "friend", "friends", "family", "people", "person",
                "everyone", "someone", "anyone", "nobody", "everybody",
                "team", "group", "community", "society", "public",
                "neighbor", "colleague", "partner", "relationship",
            ],
            "affiliation": [
                "together", "join", "joined", "share", "shared",
                "belong", "connect", "connected", "bond", "unite",
                "united", "cooperate", "collaborate", "team", "group",
                "community", "friendship", "relationship", "partner",
            ],
            "achievement": [
                "win", "won", "success", "successful", "achieve",
                "achieved", "accomplishment", "accomplish", "goal",
                "compete", "competition", "best", "better", "superior",
                "excel", "excellent", "outstanding", "victory", "triumph",
            ],
            "power": [
                "control", "power", "powerful", "authority", "command",
                "lead", "leader", "leadership", "direct", "order",
                "decide", "decision", "determine", "influence", "dominant",
                "dominate", "force", "strong", "strength",
            ],
            "submission": [
                "follow", "obey", "submit", "comply", "accept",
                "agree", "allow", "permit", "serve", "support",
                "help", "assist", "defer", "yield", "subordinate",
            ],
            "politeness": [
                "please", "thank", "thanks", "sorry", "excuse",
                "pardon", "appreciate", "grateful", "welcome",
                "kindly", "respectfully", "sincerely",
            ],
            "assertiveness": [
                "must", "need", "should", "have to", "demand",
                "require", "insist", "expect", "want", "will",
                "definitely", "certainly", "absolutely", "clearly",
            ],
        }

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze social dynamics in text.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary with social analysis results
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

        clout = self._analyze_clout(word_counts, word_count)
        results.update(clout)

        affiliation = self._analyze_affiliation(word_counts, word_count)
        results.update(affiliation)

        power = self._analyze_power_dynamics(word_counts, word_count, text)
        results.update(power)

        social_focus = self._analyze_social_focus(word_counts, word_count)
        results.update(social_focus)

        results["social_style"] = self._determine_social_style(results)
        results["social_summary"] = self._generate_summary(results)

        return results

    def _empty_results(self) -> Dict[str, Any]:
        """Return empty results structure."""
        return {
            "clout_score": 0.5,
            "first_person_singular_ratio": 0.0,
            "first_person_plural_ratio": 0.0,
            "second_person_ratio": 0.0,
            "affiliation_score": 0.0,
            "social_reference_ratio": 0.0,
            "power_score": 0.0,
            "submission_score": 0.0,
            "power_balance": 0.5,
            "politeness_ratio": 0.0,
            "assertiveness_ratio": 0.0,
            "directive_ratio": 0.0,
            "question_ratio": 0.0,
            "social_style": "balanced",
            "social_summary": "",
        }

    def _analyze_clout(
        self, word_counts: Counter, word_count: int
    ) -> Dict[str, Any]:
        """
        Analyze social clout/status indicators.

        Research shows:
        - Lower first-person singular = higher status
        - Higher first-person plural = higher status
        - Higher second-person = higher status (directing others)
        """
        first_singular = ["i", "me", "my", "mine", "myself"]
        first_plural = ["we", "us", "our", "ours", "ourselves"]
        second_person = ["you", "your", "yours", "yourself", "yourselves"]

        first_singular_count = sum(word_counts.get(w, 0) for w in first_singular)
        first_plural_count = sum(word_counts.get(w, 0) for w in first_plural)
        second_person_count = sum(word_counts.get(w, 0) for w in second_person)

        fs_ratio = first_singular_count / word_count if word_count > 0 else 0
        fp_ratio = first_plural_count / word_count if word_count > 0 else 0
        sp_ratio = second_person_count / word_count if word_count > 0 else 0

        clout_score = 0.5
        clout_score -= fs_ratio * 3
        clout_score += fp_ratio * 4
        clout_score += sp_ratio * 2
        clout_score = max(0, min(1, clout_score))

        return {
            "clout_score": clout_score,
            "first_person_singular_ratio": fs_ratio,
            "first_person_plural_ratio": fp_ratio,
            "second_person_ratio": sp_ratio,
        }

    def _analyze_affiliation(
        self, word_counts: Counter, word_count: int
    ) -> Dict[str, Any]:
        """Analyze affiliation and social connection markers."""
        affiliation_count = sum(
            word_counts.get(w, 0) for w in self.social_words["affiliation"]
        )
        affiliation_ratio = affiliation_count / word_count if word_count > 0 else 0

        social_ref_count = sum(
            word_counts.get(w, 0) for w in self.social_words["social_references"]
        )
        social_ref_ratio = social_ref_count / word_count if word_count > 0 else 0

        inclusive_words = ["we", "us", "our", "together", "everyone", "all"]
        inclusive_count = sum(word_counts.get(w, 0) for w in inclusive_words)
        inclusive_ratio = inclusive_count / word_count if word_count > 0 else 0

        affiliation_score = (affiliation_ratio * 10 + social_ref_ratio * 5 + inclusive_ratio * 8)
        affiliation_score = min(affiliation_score, 1.0)

        return {
            "affiliation_score": affiliation_score,
            "social_reference_ratio": social_ref_ratio,
            "inclusive_ratio": inclusive_ratio,
        }

    def _analyze_power_dynamics(
        self, word_counts: Counter, word_count: int, text: str
    ) -> Dict[str, Any]:
        """Analyze power and dominance indicators."""
        power_count = sum(word_counts.get(w, 0) for w in self.social_words["power"])
        power_ratio = power_count / word_count if word_count > 0 else 0

        submission_count = sum(word_counts.get(w, 0) for w in self.social_words["submission"])
        submission_ratio = submission_count / word_count if word_count > 0 else 0

        total = power_count + submission_count
        if total > 0:
            power_balance = power_count / total
        else:
            power_balance = 0.5

        politeness_count = sum(word_counts.get(w, 0) for w in self.social_words["politeness"])
        politeness_ratio = politeness_count / word_count if word_count > 0 else 0

        assertiveness_count = sum(word_counts.get(w, 0) for w in self.social_words["assertiveness"])
        assertiveness_ratio = assertiveness_count / word_count if word_count > 0 else 0

        sentences = tokenize_sentences(text)
        imperative_count = 0
        for sentence in sentences:
            words = sentence.strip().split()
            if words:
                first_word = words[0].lower()
                imperative_verbs = ["do", "don't", "please", "go", "come", "let", "make", "take", "give", "put", "get", "tell", "show", "stop", "start", "try", "remember", "consider", "think", "look", "listen", "wait", "check"]
                if first_word in imperative_verbs:
                    imperative_count += 1

        directive_ratio = imperative_count / len(sentences) if sentences else 0

        question_count = text.count("?")
        question_ratio = question_count / len(sentences) if sentences else 0

        power_score = (power_ratio * 10 + assertiveness_ratio * 5 + directive_ratio * 3)
        power_score = min(power_score, 1.0)

        submission_score = (submission_ratio * 10 + politeness_ratio * 5 + question_ratio * 2)
        submission_score = min(submission_score, 1.0)

        return {
            "power_score": power_score,
            "submission_score": submission_score,
            "power_balance": power_balance,
            "politeness_ratio": politeness_ratio,
            "assertiveness_ratio": assertiveness_ratio,
            "directive_ratio": directive_ratio,
            "question_ratio": question_ratio,
        }

    def _analyze_social_focus(
        self, word_counts: Counter, word_count: int
    ) -> Dict[str, Any]:
        """Analyze achievement vs affiliation orientation."""
        achievement_count = sum(
            word_counts.get(w, 0) for w in self.social_words["achievement"]
        )
        achievement_ratio = achievement_count / word_count if word_count > 0 else 0

        affiliation_count = sum(
            word_counts.get(w, 0) for w in self.social_words["affiliation"]
        )
        affiliation_ratio = affiliation_count / word_count if word_count > 0 else 0

        total = achievement_count + affiliation_count
        if total > 0:
            achievement_vs_affiliation = achievement_count / total
        else:
            achievement_vs_affiliation = 0.5

        return {
            "achievement_ratio": achievement_ratio,
            "affiliation_ratio": affiliation_ratio,
            "achievement_vs_affiliation": achievement_vs_affiliation,
        }

    def _determine_social_style(self, results: Dict[str, Any]) -> str:
        """Determine overall social style label."""
        clout = results.get("clout_score", 0.5)
        power_balance = results.get("power_balance", 0.5)
        affiliation = results.get("affiliation_score", 0)

        if clout > 0.65 and power_balance > 0.6:
            return "authoritative"
        elif clout > 0.6:
            return "confident"
        elif affiliation > 0.5:
            return "affiliative"
        elif power_balance < 0.4:
            return "deferential"
        elif results.get("politeness_ratio", 0) > 0.02:
            return "polite/formal"
        else:
            return "balanced"

    def _generate_summary(self, results: Dict[str, Any]) -> str:
        """Generate human-readable social dynamics summary."""
        style = results.get("social_style", "balanced")
        clout = results.get("clout_score", 0.5)
        affiliation = results.get("affiliation_score", 0)

        parts = []

        style_descriptions = {
            "authoritative": "Authoritative communication style with high social confidence",
            "confident": "Confident communication style",
            "affiliative": "Affiliative style focused on social connection",
            "deferential": "Deferential communication, showing respect to others",
            "polite/formal": "Polite and formal communication style",
            "balanced": "Balanced social communication style",
        }
        parts.append(style_descriptions.get(style, ""))

        if clout > 0.65:
            parts.append("High social clout indicators")
        elif clout < 0.35:
            parts.append("Lower status indicators (may indicate modesty or deference)")

        if affiliation > 0.4:
            parts.append("Strong focus on social relationships and connection")

        return ". ".join(p for p in parts if p) + "."

    def get_social_profile(self, text: str) -> Dict[str, Any]:
        """
        Get a simplified social profile for comparison.

        Returns key metrics suitable for profile comparison.
        """
        full_analysis = self.analyze(text)

        return {
            "clout": full_analysis.get("clout_score", 0.5),
            "affiliation": full_analysis.get("affiliation_score", 0),
            "power_orientation": full_analysis.get("power_balance", 0.5),
            "politeness": full_analysis.get("politeness_ratio", 0),
            "style": full_analysis.get("social_style", "balanced"),
        }
