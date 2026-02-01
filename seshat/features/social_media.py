"""
Social media-specific feature extraction for stylometric analysis.

Extracts features unique to social media platforms: hashtags, mentions,
URLs, informal markers, and platform-specific patterns.
"""

import re
from collections import Counter
from typing import Dict, List, Any

from seshat.utils import (
    tokenize_words,
    extract_urls,
    extract_mentions,
    extract_hashtags,
)


class SocialMediaFeatures:
    """Extract social media-specific features from text."""

    def __init__(self):
        """Initialize social media feature extractor."""
        self.informal_markers = self._define_informal_markers()
        self.discourse_markers = self._define_discourse_markers()

    def _define_informal_markers(self) -> Dict[str, List[str]]:
        """Define informal language markers."""
        return {
            "laughter": ["lol", "lmao", "lmfao", "rofl", "roflmao", "haha", "hahaha", "hehe", "hihi", "lulz", "kek"],
            "agreement": ["ikr", "imo", "imho", "tbh", "ngl", "fr", "frfr", "facts", "true", "same", "mood", "bet"],
            "reaction": ["omg", "omfg", "wtf", "wth", "smh", "ffs", "jfc", "yikes", "oof", "bruh", "sheesh"],
            "acknowledgment": ["ok", "okay", "k", "kk", "alright", "aight", "yep", "yup", "nope", "np", "nw", "ty", "thx", "tysm", "yw", "pls", "plz"],
            "filler": ["like", "um", "uh", "well", "so", "anyway", "anyways", "basically", "literally", "honestly"],
            "intensifier": ["very", "really", "super", "hella", "mad", "crazy", "insanely", "lowkey", "highkey", "deadass", "straight up"],
            "abbreviation": ["rn", "atm", "irl", "imo", "imho", "tbh", "ngl", "idk", "idc", "idgaf", "fyi", "btw", "afaik", "iirc", "tl;dr", "eli5"],
        }

    def _define_discourse_markers(self) -> Dict[str, List[str]]:
        """Define social media discourse markers."""
        return {
            "topic_change": ["anyway", "anyways", "so", "btw", "speaking of", "on another note"],
            "emphasis": ["literally", "actually", "seriously", "honestly", "truly", "genuinely"],
            "hedging": ["maybe", "probably", "possibly", "kind of", "kinda", "sort of", "sorta", "i think", "i guess", "idk"],
            "conclusion": ["so yeah", "anyway yeah", "but yeah", "so basically", "long story short"],
        }

    def extract(self, text: str) -> Dict[str, Any]:
        """
        Extract all social media features from text.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary of social media features
        """
        if not text:
            return self._empty_features()

        features = {}

        platform_features = self._extract_platform_elements(text)
        features.update(platform_features)

        informal_features = self._extract_informal_markers(text)
        features.update(informal_features)

        style_features = self._extract_style_features(text)
        features.update(style_features)

        message_features = self._extract_message_patterns(text)
        features.update(message_features)

        return features

    def _empty_features(self) -> Dict[str, Any]:
        """Return empty feature dictionary."""
        return {
            "hashtag_count": 0,
            "hashtag_per_100_words": 0.0,
            "mention_count": 0,
            "mention_per_100_words": 0.0,
            "url_count": 0,
            "url_per_100_words": 0.0,
            "hashtag_style": "none",
            "top_hashtags": {},
            "laughter_marker_count": 0,
            "laughter_per_100_words": 0.0,
            "reaction_marker_count": 0,
            "abbreviation_count": 0,
            "abbreviation_per_100_words": 0.0,
            "informal_marker_ratio": 0.0,
            "elongated_word_count": 0,
            "repeated_char_ratio": 0.0,
            "all_caps_ratio": 0.0,
            "avg_message_length": 0.0,
            "short_message_ratio": 0.0,
            "question_message_ratio": 0.0,
        }

    def _extract_platform_elements(self, text: str) -> Dict[str, Any]:
        """Extract platform-specific elements (hashtags, mentions, URLs)."""
        hashtags = extract_hashtags(text)
        mentions = extract_mentions(text)
        urls = extract_urls(text)

        words = tokenize_words(text)
        word_count = len(words) if words else 1

        hashtag_count = len(hashtags)
        mention_count = len(mentions)
        url_count = len(urls)

        hashtag_styles = []
        for tag in hashtags:
            if tag.isupper():
                hashtag_styles.append("ALLCAPS")
            elif tag.istitle() or re.match(r'^[A-Z][a-z]+([A-Z][a-z]+)*$', tag):
                hashtag_styles.append("CamelCase")
            elif tag.islower():
                hashtag_styles.append("lowercase")
            else:
                hashtag_styles.append("mixed")

        if hashtag_styles:
            style_counts = Counter(hashtag_styles)
            dominant_style = style_counts.most_common(1)[0][0]
        else:
            dominant_style = "none"

        hashtag_counts = Counter(hashtags)
        top_hashtags = dict(hashtag_counts.most_common(10))

        return {
            "hashtag_count": hashtag_count,
            "hashtag_per_100_words": (hashtag_count / word_count) * 100,
            "mention_count": mention_count,
            "mention_per_100_words": (mention_count / word_count) * 100,
            "url_count": url_count,
            "url_per_100_words": (url_count / word_count) * 100,
            "hashtag_style": dominant_style,
            "top_hashtags": top_hashtags,
        }

    def _extract_informal_markers(self, text: str) -> Dict[str, Any]:
        """Extract informal language marker features."""
        text_lower = text.lower()
        words = tokenize_words(text)
        word_count = len(words) if words else 1

        category_counts = {}
        total_informal = 0

        for category, markers in self.informal_markers.items():
            count = 0
            for marker in markers:
                pattern = r'\b' + re.escape(marker) + r'\b'
                matches = re.findall(pattern, text_lower)
                count += len(matches)
            category_counts[category] = count
            total_informal += count

        return {
            "laughter_marker_count": category_counts.get("laughter", 0),
            "laughter_per_100_words": (category_counts.get("laughter", 0) / word_count) * 100,
            "reaction_marker_count": category_counts.get("reaction", 0),
            "abbreviation_count": category_counts.get("abbreviation", 0),
            "abbreviation_per_100_words": (category_counts.get("abbreviation", 0) / word_count) * 100,
            "informal_marker_ratio": total_informal / word_count,
            "informal_category_distribution": {
                cat: count / total_informal if total_informal > 0 else 0
                for cat, count in category_counts.items()
            },
        }

    def _extract_style_features(self, text: str) -> Dict[str, Any]:
        """Extract writing style features common in social media."""
        elongated_pattern = re.compile(r'(\w)\1{2,}')
        elongated_words = elongated_pattern.findall(text)
        elongated_count = len(elongated_words)

        repeated_chars = sum(len(match.group()) - 1 for match in re.finditer(r'(\w)\1+', text))
        total_chars = len(text)
        repeated_ratio = repeated_chars / total_chars if total_chars > 0 else 0

        words = text.split()
        all_caps_words = [w for w in words if w.isupper() and len(w) > 1 and w.isalpha()]
        all_caps_ratio = len(all_caps_words) / len(words) if words else 0

        asterisk_emphasis = len(re.findall(r'\*\w+\*', text))
        underscore_emphasis = len(re.findall(r'_\w+_', text))

        trailing_punctuation = len(re.findall(r'[.]{2,}|[!]{2,}|[?]{2,}', text))

        return {
            "elongated_word_count": elongated_count,
            "repeated_char_ratio": repeated_ratio,
            "all_caps_ratio": all_caps_ratio,
            "asterisk_emphasis_count": asterisk_emphasis,
            "underscore_emphasis_count": underscore_emphasis,
            "trailing_punctuation_count": trailing_punctuation,
        }

    def _extract_message_patterns(self, text: str) -> Dict[str, Any]:
        """Extract message-level patterns (for multi-message texts)."""
        messages = re.split(r'\n\n+|\n(?=[A-Z@#])', text)
        messages = [m.strip() for m in messages if m.strip()]

        if not messages:
            return {
                "avg_message_length": 0.0,
                "short_message_ratio": 0.0,
                "question_message_ratio": 0.0,
                "message_count": 0,
            }

        message_lengths = [len(m.split()) for m in messages]
        avg_length = sum(message_lengths) / len(message_lengths)

        short_messages = sum(1 for l in message_lengths if l <= 10)
        short_ratio = short_messages / len(messages)

        question_messages = sum(1 for m in messages if m.rstrip().endswith('?'))
        question_ratio = question_messages / len(messages)

        return {
            "avg_message_length": avg_length,
            "short_message_ratio": short_ratio,
            "question_message_ratio": question_ratio,
            "message_count": len(messages),
        }

    def get_social_media_signature(self, text: str) -> Dict[str, Any]:
        """
        Get a summary of distinctive social media features.

        Returns a signature that can help identify platform and style.
        """
        features = self.extract(text)

        signature = {
            "platform_hints": [],
            "communication_style": "neutral",
            "formality": "medium",
            "engagement_markers": [],
        }

        if features.get("hashtag_per_100_words", 0) > 5:
            signature["platform_hints"].append("twitter/instagram")
        if features.get("mention_per_100_words", 0) > 3:
            signature["platform_hints"].append("twitter")
        if features.get("avg_message_length", 0) < 50:
            signature["platform_hints"].append("twitter/sms")
        if features.get("avg_message_length", 0) > 200:
            signature["platform_hints"].append("reddit/forum")

        informal_ratio = features.get("informal_marker_ratio", 0)
        if informal_ratio > 0.1:
            signature["formality"] = "very informal"
            signature["communication_style"] = "casual"
        elif informal_ratio > 0.05:
            signature["formality"] = "informal"
            signature["communication_style"] = "conversational"
        elif informal_ratio < 0.01:
            signature["formality"] = "formal"
            signature["communication_style"] = "professional"

        if features.get("laughter_marker_count", 0) > 0:
            signature["engagement_markers"].append("laughter")
        if features.get("elongated_word_count", 0) > 0:
            signature["engagement_markers"].append("emphasis")
        if features.get("all_caps_ratio", 0) > 0.05:
            signature["engagement_markers"].append("caps_emphasis")

        return signature

    def analyze_platform_style(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for platform-specific writing patterns.

        Returns human-readable analysis.
        """
        features = self.extract(text)
        signature = self.get_social_media_signature(text)

        analysis = {
            "likely_platforms": signature.get("platform_hints", []),
            "style_patterns": [],
            "formality_level": signature.get("formality", "medium"),
        }

        if features.get("hashtag_count", 0) > 0:
            style = features.get("hashtag_style", "mixed")
            analysis["style_patterns"].append(f"Uses {style} hashtag style")

        if features.get("laughter_per_100_words", 0) > 2:
            analysis["style_patterns"].append("Frequent laughter markers (lol, haha)")

        if features.get("abbreviation_per_100_words", 0) > 3:
            analysis["style_patterns"].append("Heavy use of abbreviations")

        if features.get("elongated_word_count", 0) > 2:
            analysis["style_patterns"].append("Uses word elongation for emphasis (sooo, yesss)")

        if features.get("all_caps_ratio", 0) > 0.1:
            analysis["style_patterns"].append("Frequent ALL CAPS for emphasis")

        if features.get("short_message_ratio", 0) > 0.7:
            analysis["style_patterns"].append("Prefers short, punchy messages")

        return analysis
