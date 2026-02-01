"""
Emoji and emoticon analysis for stylometric profiling.

Emoji usage patterns provide strong authorship signals, especially
in social media and informal communication.
"""

import re
from collections import Counter
from typing import Dict, List, Any, Tuple

try:
    import emoji
    EMOJI_AVAILABLE = True
except ImportError:
    EMOJI_AVAILABLE = False


class EmojiFeatures:
    """Extract emoji and emoticon features from text."""

    def __init__(self):
        """Initialize emoji extractor."""
        self.emoticon_patterns = self._define_emoticons()
        self.emoji_categories = self._define_categories()

    def _define_emoticons(self) -> Dict[str, List[str]]:
        """Define common text emoticons."""
        return {
            "happy": [":)", ":-)", ":D", ":-D", "=)", "=D", ":P", ":-P", ":p", ":-p", "xD", "XD", "^^", "^_^", "(:", "(:"],
            "sad": [":(", ":-(", ":'(", ":'-(", "T_T", "T.T", ";(", ";-(", "):", ")':"],
            "love": ["<3", "♥", "♡", ":*", ":-*", "xo", "xoxo"],
            "wink": [";)", ";-)", ";D", ";-D"],
            "surprised": [":O", ":-O", ":o", ":-o", "=O", "=o", "O_O", "o_o", "O.O"],
            "neutral": [":|", ":-|", ":/", ":-/", ":\\", ":-\\", "-_-", "._."],
            "laugh": ["lol", "lmao", "lmfao", "rofl", "haha", "hehe", "hihi", "jaja", "kek"],
            "cry": ["T_T", ";_;", "QQ", "TT", "ToT"],
        }

    def _define_categories(self) -> Dict[str, List[str]]:
        """Define emoji unicode categories."""
        return {
            "faces_positive": [
                "😀", "😃", "😄", "😁", "😆", "😅", "🤣", "😂", "🙂", "😊",
                "😇", "🥰", "😍", "🤩", "😘", "😗", "😚", "😙", "🥲",
            ],
            "faces_negative": [
                "😞", "😔", "😟", "😕", "🙁", "☹️", "😣", "😖", "😫", "😩",
                "🥺", "😢", "😭", "😤", "😠", "😡", "🤬", "😈", "👿",
            ],
            "faces_neutral": [
                "😐", "😑", "😶", "🙄", "😏", "😬", "🤔", "🤨", "🧐",
            ],
            "gestures": [
                "👍", "👎", "👊", "✊", "🤛", "🤜", "👏", "🙌", "👐", "🤲",
                "🤝", "🙏", "✌️", "🤞", "🤟", "🤘", "👌", "🤌", "👈", "👉",
                "👆", "👇", "☝️", "✋", "🤚", "🖐️", "🖖", "👋", "🤙", "💪",
            ],
            "hearts": [
                "❤️", "🧡", "💛", "💚", "💙", "💜", "🖤", "🤍", "🤎", "💔",
                "❣️", "💕", "💞", "💓", "💗", "💖", "💘", "💝", "💟",
            ],
            "nature": [
                "🌸", "🌺", "🌻", "🌼", "🌷", "🪷", "🌹", "🥀", "🪻", "💐",
                "🌲", "🌳", "🌴", "🌵", "🌾", "🌿", "☘️", "🍀", "🍁", "🍂",
            ],
            "food": [
                "🍎", "🍐", "🍊", "🍋", "🍌", "🍉", "🍇", "🍓", "🫐", "🍈",
                "🍕", "🍔", "🍟", "🌭", "🍿", "🧂", "🥓", "🥚", "🍳", "🧇",
            ],
            "activities": [
                "⚽", "🏀", "🏈", "⚾", "🥎", "🎾", "🏐", "🏉", "🥏", "🎱",
                "🎮", "🎲", "🎯", "🎳", "🎸", "🎹", "🎺", "🎻", "🎬", "🎨",
            ],
            "objects": [
                "💻", "🖥️", "🖨️", "⌨️", "🖱️", "💿", "📱", "📞", "☎️", "📺",
                "📷", "📸", "📹", "🎥", "📽️", "🎞️", "📀", "💾", "💽", "🔋",
            ],
            "symbols": [
                "✅", "❌", "❓", "❔", "❕", "❗", "⭕", "🔴", "🟠", "🟡",
                "💯", "🔥", "✨", "⭐", "🌟", "💫", "⚡", "☀️", "🌈", "💥",
            ],
        }

    def extract(self, text: str) -> Dict[str, Any]:
        """
        Extract all emoji and emoticon features from text.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary of emoji features
        """
        if not text:
            return self._empty_features()

        features = {}

        if EMOJI_AVAILABLE:
            emoji_features = self._extract_emoji_features(text)
            features.update(emoji_features)
        else:
            features.update(self._basic_emoji_features(text))

        emoticon_features = self._extract_emoticon_features(text)
        features.update(emoticon_features)

        placement_features = self._extract_placement_features(text)
        features.update(placement_features)

        cluster_features = self._extract_cluster_features(text)
        features.update(cluster_features)

        return features

    def _empty_features(self) -> Dict[str, Any]:
        """Return empty feature dictionary."""
        return {
            "emoji_count": 0,
            "emoji_per_100_words": 0.0,
            "unique_emoji_count": 0,
            "emoji_diversity": 0.0,
            "emoticon_count": 0,
            "emoticon_per_100_words": 0.0,
            "emoji_to_text_ratio": 0.0,
            "top_emojis": {},
            "emoji_category_distribution": {},
            "emoticon_category_distribution": {},
            "emoji_at_start_ratio": 0.0,
            "emoji_at_end_ratio": 0.0,
            "emoji_inline_ratio": 0.0,
            "avg_emoji_cluster_size": 0.0,
            "max_emoji_cluster_size": 0,
            "emoji_only_messages": 0,
        }

    def _extract_emoji_features(self, text: str) -> Dict[str, Any]:
        """Extract emoji features using emoji library."""
        emojis = []
        for char in text:
            if emoji.is_emoji(char):
                emojis.append(char)

        emoji_count = len(emojis)

        words = text.split()
        word_count = len(words) if words else 1

        emoji_per_100 = (emoji_count / word_count) * 100

        unique_emojis = set(emojis)
        unique_count = len(unique_emojis)

        diversity = unique_count / emoji_count if emoji_count > 0 else 0

        emoji_text_ratio = emoji_count / len(text) if text else 0

        emoji_counts = Counter(emojis)
        top_emojis = dict(emoji_counts.most_common(10))

        category_counts = Counter()
        for e in emojis:
            for category, category_emojis in self.emoji_categories.items():
                if e in category_emojis:
                    category_counts[category] += 1
                    break
            else:
                category_counts["other"] += 1

        total_categorized = sum(category_counts.values())
        category_dist = {
            cat: count / total_categorized
            for cat, count in category_counts.items()
        } if total_categorized > 0 else {}

        return {
            "emoji_count": emoji_count,
            "emoji_per_100_words": emoji_per_100,
            "unique_emoji_count": unique_count,
            "emoji_diversity": diversity,
            "emoji_to_text_ratio": emoji_text_ratio,
            "top_emojis": top_emojis,
            "emoji_category_distribution": category_dist,
        }

    def _basic_emoji_features(self, text: str) -> Dict[str, Any]:
        """Basic emoji extraction without emoji library."""
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE
        )

        emojis = emoji_pattern.findall(text)
        all_emojis = []
        for match in emojis:
            all_emojis.extend(list(match))

        emoji_count = len(all_emojis)
        words = text.split()
        word_count = len(words) if words else 1

        return {
            "emoji_count": emoji_count,
            "emoji_per_100_words": (emoji_count / word_count) * 100,
            "unique_emoji_count": len(set(all_emojis)),
            "emoji_diversity": len(set(all_emojis)) / emoji_count if emoji_count > 0 else 0,
            "emoji_to_text_ratio": emoji_count / len(text) if text else 0,
            "top_emojis": dict(Counter(all_emojis).most_common(10)),
            "emoji_category_distribution": {},
        }

    def _extract_emoticon_features(self, text: str) -> Dict[str, Any]:
        """Extract text emoticon features."""
        emoticon_counts = Counter()
        category_counts = Counter()

        text_lower = text.lower()

        for category, patterns in self.emoticon_patterns.items():
            for pattern in patterns:
                count = text_lower.count(pattern.lower())
                if count > 0:
                    emoticon_counts[pattern] += count
                    category_counts[category] += count

        total_emoticons = sum(emoticon_counts.values())

        words = text.split()
        word_count = len(words) if words else 1

        total_categorized = sum(category_counts.values())
        category_dist = {
            cat: count / total_categorized
            for cat, count in category_counts.items()
        } if total_categorized > 0 else {}

        return {
            "emoticon_count": total_emoticons,
            "emoticon_per_100_words": (total_emoticons / word_count) * 100,
            "emoticon_category_distribution": category_dist,
        }

    def _extract_placement_features(self, text: str) -> Dict[str, Any]:
        """Extract emoji placement patterns."""
        lines = text.split('\n')
        sentences = re.split(r'[.!?]+', text)

        at_start = 0
        at_end = 0
        inline = 0

        def has_emoji(s: str) -> bool:
            if EMOJI_AVAILABLE:
                return any(emoji.is_emoji(c) for c in s)
            return bool(re.search(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF]', s))

        def get_emojis(s: str) -> List[Tuple[int, str]]:
            result = []
            for i, c in enumerate(s):
                if EMOJI_AVAILABLE and emoji.is_emoji(c):
                    result.append((i, c))
                elif re.match(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF]', c):
                    result.append((i, c))
            return result

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            emojis = get_emojis(sentence)
            if not emojis:
                continue

            for pos, _ in emojis:
                relative_pos = pos / len(sentence) if sentence else 0

                if relative_pos < 0.1:
                    at_start += 1
                elif relative_pos > 0.9:
                    at_end += 1
                else:
                    inline += 1

        total = at_start + at_end + inline
        if total == 0:
            return {
                "emoji_at_start_ratio": 0.0,
                "emoji_at_end_ratio": 0.0,
                "emoji_inline_ratio": 0.0,
            }

        return {
            "emoji_at_start_ratio": at_start / total,
            "emoji_at_end_ratio": at_end / total,
            "emoji_inline_ratio": inline / total,
        }

    def _extract_cluster_features(self, text: str) -> Dict[str, Any]:
        """Extract emoji clustering patterns."""
        if EMOJI_AVAILABLE:
            is_emoji_char = lambda c: emoji.is_emoji(c)
        else:
            emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF]')
            is_emoji_char = lambda c: bool(emoji_pattern.match(c))

        clusters = []
        current_cluster = 0

        for char in text:
            if is_emoji_char(char):
                current_cluster += 1
            else:
                if current_cluster > 0:
                    clusters.append(current_cluster)
                    current_cluster = 0

        if current_cluster > 0:
            clusters.append(current_cluster)

        if not clusters:
            return {
                "avg_emoji_cluster_size": 0.0,
                "max_emoji_cluster_size": 0,
                "emoji_only_messages": 0,
            }

        lines = text.split('\n')
        emoji_only = 0
        for line in lines:
            line = line.strip()
            if line and all(is_emoji_char(c) or c.isspace() for c in line):
                emoji_only += 1

        return {
            "avg_emoji_cluster_size": sum(clusters) / len(clusters),
            "max_emoji_cluster_size": max(clusters),
            "emoji_only_messages": emoji_only,
        }

    def get_emoji_signature(self, text: str, top_n: int = 5) -> List[str]:
        """Get the most characteristic emojis for a text."""
        if EMOJI_AVAILABLE:
            emojis = [c for c in text if emoji.is_emoji(c)]
        else:
            emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF]')
            emojis = emoji_pattern.findall(text)

        counts = Counter(emojis)
        return [e for e, _ in counts.most_common(top_n)]

    def analyze_emoji_style(self, text: str) -> Dict[str, Any]:
        """Provide human-readable analysis of emoji usage style."""
        features = self.extract(text)

        analysis = {
            "style": "neutral",
            "patterns": [],
            "emotional_tone": "neutral",
        }

        emoji_rate = features.get("emoji_per_100_words", 0)
        if emoji_rate > 10:
            analysis["style"] = "emoji-heavy"
            analysis["patterns"].append("Very frequent emoji usage")
        elif emoji_rate > 3:
            analysis["style"] = "moderate"
            analysis["patterns"].append("Regular emoji usage")
        elif emoji_rate > 0:
            analysis["style"] = "light"
            analysis["patterns"].append("Occasional emoji usage")
        else:
            analysis["style"] = "text-only"
            analysis["patterns"].append("No emoji usage")

        category_dist = features.get("emoji_category_distribution", {})
        if category_dist.get("faces_positive", 0) > 0.3:
            analysis["emotional_tone"] = "positive"
            analysis["patterns"].append("Predominantly positive emojis")
        elif category_dist.get("faces_negative", 0) > 0.2:
            analysis["emotional_tone"] = "negative"
            analysis["patterns"].append("Significant negative emoji usage")
        elif category_dist.get("hearts", 0) > 0.2:
            analysis["emotional_tone"] = "affectionate"
            analysis["patterns"].append("Heavy heart/love emoji usage")

        if features.get("emoji_at_end_ratio", 0) > 0.7:
            analysis["patterns"].append("Prefers emojis at end of sentences")
        elif features.get("emoji_inline_ratio", 0) > 0.5:
            analysis["patterns"].append("Uses emojis inline within text")

        if features.get("avg_emoji_cluster_size", 0) > 2:
            analysis["patterns"].append("Uses emoji clusters/combinations")

        return analysis


# Alias for backward compatibility
EmojiFeatureExtractor = EmojiFeatures
