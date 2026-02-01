"""
Utility functions for text preprocessing, tokenization, and normalization.
"""

import re
import unicodedata
from typing import List, Optional, Tuple
from collections import Counter
import string


def normalize_unicode(text: str) -> str:
    """Normalize unicode characters to NFC form."""
    return unicodedata.normalize("NFC", text)


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace while preserving paragraph structure."""
    lines = text.split('\n')
    normalized_lines = []
    for line in lines:
        normalized_line = ' '.join(line.split())
        normalized_lines.append(normalized_line)
    return '\n'.join(normalized_lines)


def remove_urls(text: str) -> str:
    """Remove URLs from text."""
    url_pattern = r'https?://\S+|www\.\S+'
    return re.sub(url_pattern, '', text)


def remove_mentions(text: str) -> str:
    """Remove @mentions from text."""
    return re.sub(r'@\w+', '', text)


def remove_hashtags(text: str) -> str:
    """Remove #hashtags from text."""
    return re.sub(r'#\w+', '', text)


def extract_urls(text: str) -> List[str]:
    """Extract all URLs from text."""
    url_pattern = r'https?://\S+|www\.\S+'
    return re.findall(url_pattern, text)


def extract_mentions(text: str) -> List[str]:
    """Extract all @mentions from text."""
    return re.findall(r'@(\w+)', text)


def extract_hashtags(text: str) -> List[str]:
    """Extract all #hashtags from text."""
    return re.findall(r'#(\w+)', text)


def tokenize_words(text: str) -> List[str]:
    """
    Tokenize text into words.
    Preserves contractions and handles punctuation appropriately.
    """
    word_pattern = r"\b[\w']+\b"
    return re.findall(word_pattern, text.lower())


def tokenize_words_preserve_case(text: str) -> List[str]:
    """Tokenize text into words preserving original case."""
    word_pattern = r"\b[\w']+\b"
    return re.findall(word_pattern, text)


def tokenize_sentences(text: str) -> List[str]:
    """
    Tokenize text into sentences.
    Handles common abbreviations and edge cases.
    """
    abbreviations = {
        'mr.', 'mrs.', 'ms.', 'dr.', 'prof.', 'sr.', 'jr.',
        'vs.', 'etc.', 'e.g.', 'i.e.', 'inc.', 'ltd.', 'co.',
        'st.', 'ave.', 'blvd.', 'rd.', 'apt.', 'no.',
    }

    placeholder = '\x00ABBR\x00'
    modified_text = text
    abbr_positions = []

    for abbr in abbreviations:
        pattern = re.compile(re.escape(abbr), re.IGNORECASE)
        for match in pattern.finditer(modified_text):
            abbr_positions.append((match.start(), match.end(), match.group()))

    abbr_positions.sort(key=lambda x: x[0], reverse=True)
    for start, end, original in abbr_positions:
        replacement = original.replace('.', placeholder)
        modified_text = modified_text[:start] + replacement + modified_text[end:]

    sentence_pattern = r'[^.!?]+[.!?]+(?:\s|$)|[^.!?]+$'
    sentences = re.findall(sentence_pattern, modified_text)

    sentences = [s.replace(placeholder, '.').strip() for s in sentences if s.strip()]

    return sentences


def tokenize_paragraphs(text: str) -> List[str]:
    """Tokenize text into paragraphs (separated by blank lines or double newlines)."""
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paragraphs if p.strip()]


def get_character_ngrams(text: str, n: int) -> List[str]:
    """Generate character n-grams from text."""
    return [text[i:i+n] for i in range(len(text) - n + 1)]


def get_word_ngrams(words: List[str], n: int) -> List[Tuple[str, ...]]:
    """Generate word n-grams from a list of words."""
    return [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]


def count_syllables(word: str) -> int:
    """
    Count syllables in a word using a heuristic approach.
    """
    if not word:
        return 0

    word = word.lower()

    if len(word) <= 2:
        return 1

    word = re.sub(r'(es|ed|e)$', '', word)

    vowels = 'aeiouy'
    count = 0
    prev_was_vowel = False

    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_was_vowel:
            count += 1
        prev_was_vowel = is_vowel

    return max(count, 1)


def flesch_reading_ease(text: str) -> float:
    """
    Calculate Flesch Reading Ease score.
    Higher score = easier to read (0-100 typical range).
    """
    sentences = tokenize_sentences(text)
    words = tokenize_words(text)

    if not sentences or not words:
        return 0.0

    total_syllables = sum(count_syllables(word) for word in words)

    avg_sentence_length = len(words) / len(sentences)
    avg_syllables_per_word = total_syllables / len(words)

    score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)

    return score


def flesch_kincaid_grade(text: str) -> float:
    """
    Calculate Flesch-Kincaid Grade Level.
    Returns approximate US grade level needed to understand the text.
    """
    sentences = tokenize_sentences(text)
    words = tokenize_words(text)

    if not sentences or not words:
        return 0.0

    total_syllables = sum(count_syllables(word) for word in words)

    avg_sentence_length = len(words) / len(sentences)
    avg_syllables_per_word = total_syllables / len(words)

    grade = (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59

    return max(grade, 0.0)


def is_contraction(word: str) -> bool:
    """Check if a word is a contraction."""
    contractions = {
        "i'm", "i've", "i'll", "i'd",
        "you're", "you've", "you'll", "you'd",
        "he's", "he'll", "he'd",
        "she's", "she'll", "she'd",
        "it's", "it'll",
        "we're", "we've", "we'll", "we'd",
        "they're", "they've", "they'll", "they'd",
        "that's", "that'll", "that'd",
        "who's", "who'll", "who'd",
        "what's", "what'll", "what'd",
        "where's", "where'll", "where'd",
        "when's", "when'll", "when'd",
        "why's", "why'll", "why'd",
        "how's", "how'll", "how'd",
        "isn't", "aren't", "wasn't", "weren't",
        "hasn't", "haven't", "hadn't",
        "doesn't", "don't", "didn't",
        "won't", "wouldn't",
        "can't", "couldn't",
        "shouldn't", "mightn't", "mustn't",
        "let's", "here's", "there's",
        "ain't", "gonna", "wanna", "gotta",
        "y'all", "ma'am", "o'clock",
    }
    return word.lower() in contractions


def is_abbreviation(word: str) -> bool:
    """Check if a word is a common abbreviation/acronym."""
    common_abbrevs = {
        "btw", "idk", "lol", "lmao", "rofl", "omg", "wtf", "wth",
        "tbh", "imo", "imho", "fyi", "afaik", "iirc", "tl;dr",
        "brb", "bbl", "ttyl", "np", "nvm", "jk", "smh", "ngl",
        "rn", "atm", "irl", "dm", "pm", "asap", "eta",
        "usa", "uk", "eu", "un", "nato", "fbi", "cia", "nsa",
        "ceo", "cto", "cfo", "hr", "pr", "qa", "it", "ai", "ml",
        "api", "url", "html", "css", "js", "sql", "http", "https",
    }
    return word.lower() in common_abbrevs


def get_word_length_stats(words: List[str]) -> dict:
    """Calculate word length statistics."""
    if not words:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0,
            "max": 0,
            "short_ratio": 0.0,
            "long_ratio": 0.0,
        }

    lengths = [len(w) for w in words]
    mean_length = sum(lengths) / len(lengths)

    variance = sum((l - mean_length) ** 2 for l in lengths) / len(lengths)
    std_length = variance ** 0.5

    short_count = sum(1 for l in lengths if l <= 3)
    long_count = sum(1 for l in lengths if l >= 7)

    return {
        "mean": mean_length,
        "std": std_length,
        "min": min(lengths),
        "max": max(lengths),
        "short_ratio": short_count / len(lengths),
        "long_ratio": long_count / len(lengths),
    }


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero."""
    if denominator == 0:
        return default
    return numerator / denominator


def normalize_features(features: dict) -> dict:
    """
    Normalize feature values to 0-1 range using min-max normalization.
    Only normalizes numeric values.
    """
    numeric_features = {k: v for k, v in features.items() if isinstance(v, (int, float))}

    if not numeric_features:
        return features

    min_val = min(numeric_features.values())
    max_val = max(numeric_features.values())

    if max_val == min_val:
        normalized = {k: 0.5 for k in numeric_features}
    else:
        normalized = {
            k: (v - min_val) / (max_val - min_val)
            for k, v in numeric_features.items()
        }

    result = features.copy()
    result.update(normalized)
    return result


def clean_text_for_analysis(
    text: str,
    remove_urls_flag: bool = True,
    remove_mentions_flag: bool = False,
    remove_hashtags_flag: bool = False,
    normalize_unicode_flag: bool = True,
    normalize_whitespace_flag: bool = True,
) -> str:
    """
    Clean text for stylometric analysis.

    Args:
        text: Input text to clean
        remove_urls_flag: Whether to remove URLs
        remove_mentions_flag: Whether to remove @mentions
        remove_hashtags_flag: Whether to remove #hashtags
        normalize_unicode_flag: Whether to normalize unicode
        normalize_whitespace_flag: Whether to normalize whitespace

    Returns:
        Cleaned text
    """
    if normalize_unicode_flag:
        text = normalize_unicode(text)

    if remove_urls_flag:
        text = remove_urls(text)

    if remove_mentions_flag:
        text = remove_mentions(text)

    if remove_hashtags_flag:
        text = remove_hashtags(text)

    if normalize_whitespace_flag:
        text = normalize_whitespace(text)

    return text.strip()


def extract_punctuation(text: str) -> List[str]:
    """Extract all punctuation characters from text."""
    return [char for char in text if char in string.punctuation]


def get_punctuation_counts(text: str) -> Counter:
    """Count each type of punctuation in text."""
    punctuation = extract_punctuation(text)
    return Counter(punctuation)


def detect_language(text: str) -> Optional[str]:
    """
    Detect the language of the text.
    Returns ISO 639-1 language code or None if detection fails.
    """
    try:
        from langdetect import detect
        return detect(text)
    except Exception:
        return None


def get_text_hash(text: str) -> str:
    """Generate SHA-256 hash of text for deduplication and verification."""
    import hashlib
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


# Alias for backward compatibility
compute_text_hash = get_text_hash


def normalize_text(
    text: str,
    lowercase: bool = False,
    remove_punctuation: bool = False,
    remove_urls_flag: bool = False,
    remove_mentions_flag: bool = False,
    remove_hashtags_flag: bool = False,
) -> str:
    """
    Normalize text with various options.

    Args:
        text: Input text to normalize
        lowercase: Convert to lowercase
        remove_punctuation: Remove punctuation characters
        remove_urls_flag: Remove URLs
        remove_mentions_flag: Remove @mentions
        remove_hashtags_flag: Remove #hashtags

    Returns:
        Normalized text
    """
    result = text

    if remove_urls_flag:
        result = remove_urls(result)

    if remove_mentions_flag:
        result = remove_mentions(result)

    if remove_hashtags_flag:
        result = remove_hashtags(result)

    if remove_punctuation:
        result = ''.join(char if char not in string.punctuation else ' ' for char in result)

    result = normalize_whitespace(result)

    if lowercase:
        result = result.lower()

    return result.strip()
