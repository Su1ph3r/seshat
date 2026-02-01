"""
Idiolect feature extraction for stylometric analysis.

Identifies unique author signatures: consistent spelling variants,
unusual word combinations, catchphrases, and personal linguistic quirks.
"""

import re
from collections import Counter, defaultdict
from typing import Dict, List, Any, Set, Tuple

from seshat.utils import tokenize_words, get_word_ngrams


class IdiolectFeatures:
    """Extract idiolect (unique personal language) features from text."""

    def __init__(self):
        """Initialize idiolect extractor."""
        self.common_misspellings = self._load_common_misspellings()
        self.standard_spellings = self._load_standard_spellings()

    def _load_common_misspellings(self) -> Dict[str, str]:
        """Load common misspelling patterns."""
        return {
            "definately": "definitely",
            "seperate": "separate",
            "occurence": "occurrence",
            "occured": "occurred",
            "recieve": "receive",
            "wierd": "weird",
            "untill": "until",
            "tommorow": "tomorrow",
            "goverment": "government",
            "enviroment": "environment",
            "begining": "beginning",
            "beleive": "believe",
            "calender": "calendar",
            "cemetary": "cemetery",
            "collegue": "colleague",
            "commited": "committed",
            "concious": "conscious",
            "dissapoint": "disappoint",
            "embarass": "embarrass",
            "existance": "existence",
            "fourty": "forty",
            "grammer": "grammar",
            "harrass": "harass",
            "independant": "independent",
            "knowlege": "knowledge",
            "liason": "liaison",
            "lisence": "license",
            "maintainance": "maintenance",
            "millenium": "millennium",
            "mispell": "misspell",
            "neccessary": "necessary",
            "noticable": "noticeable",
            "ocassion": "occasion",
            "paralell": "parallel",
            "persistant": "persistent",
            "posession": "possession",
            "priviledge": "privilege",
            "pronounciation": "pronunciation",
            "publically": "publicly",
            "questionaire": "questionnaire",
            "recomend": "recommend",
            "refered": "referred",
            "relevent": "relevant",
            "rythm": "rhythm",
            "succesful": "successful",
            "suprise": "surprise",
            "thier": "their",
            "truely": "truly",
            "vaccuum": "vacuum",
            "wether": "whether",
            "writting": "writing",
            "your/you're": None,
            "its/it's": None,
            "there/their/they're": None,
            "alot": "a lot",
            "gonna": "going to",
            "wanna": "want to",
            "gotta": "got to",
            "shouldof": "should have",
            "couldof": "could have",
            "wouldof": "would have",
        }

    def _load_standard_spellings(self) -> Dict[str, List[str]]:
        """Load standard spelling variants (British vs American)."""
        return {
            "color": ["colour"],
            "favor": ["favour"],
            "honor": ["honour"],
            "humor": ["humour"],
            "labor": ["labour"],
            "neighbor": ["neighbour"],
            "organize": ["organise"],
            "realize": ["realise"],
            "recognize": ["recognise"],
            "analyze": ["analyse"],
            "catalog": ["catalogue"],
            "dialog": ["dialogue"],
            "center": ["centre"],
            "meter": ["metre"],
            "theater": ["theatre"],
            "defense": ["defence"],
            "offense": ["offence"],
            "license": ["licence"],
            "practice": ["practise"],
            "gray": ["grey"],
            "traveled": ["travelled"],
            "canceled": ["cancelled"],
            "modeling": ["modelling"],
            "jewelry": ["jewellery"],
            "aging": ["ageing"],
        }

    def extract(self, text: str) -> Dict[str, Any]:
        """
        Extract idiolect features from text.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary of idiolect features
        """
        if not text:
            return self._empty_features()

        features = {}

        spelling_features = self._extract_spelling_patterns(text)
        features.update(spelling_features)

        phrase_features = self._extract_phrase_patterns(text)
        features.update(phrase_features)

        quirk_features = self._extract_linguistic_quirks(text)
        features.update(quirk_features)

        consistency_features = self._extract_consistency_patterns(text)
        features.update(consistency_features)

        return features

    def _empty_features(self) -> Dict[str, Any]:
        """Return empty feature dictionary."""
        return {
            "misspelling_count": 0,
            "misspelling_ratio": 0.0,
            "consistent_misspellings": [],
            "british_spelling_ratio": 0.0,
            "american_spelling_ratio": 0.0,
            "spelling_preference": "neutral",
            "unique_bigrams": [],
            "repeated_phrases": [],
            "catchphrase_candidates": [],
            "filler_word_pattern": {},
            "sentence_starter_signature": {},
            "punctuation_quirks": [],
            "capitalization_quirks": [],
            "contraction_preference": 0.0,
            "quote_style": "neutral",
        }

    def _extract_spelling_patterns(self, text: str) -> Dict[str, Any]:
        """Extract spelling pattern features."""
        words = tokenize_words(text)
        words_original = text.lower().split()

        if not words:
            return {
                "misspelling_count": 0,
                "misspelling_ratio": 0.0,
                "consistent_misspellings": [],
                "british_spelling_ratio": 0.0,
                "american_spelling_ratio": 0.0,
                "spelling_preference": "neutral",
            }

        misspelling_count = 0
        found_misspellings = []

        for word in words:
            if word in self.common_misspellings:
                misspelling_count += 1
                found_misspellings.append(word)

        british_count = 0
        american_count = 0

        for american, british_variants in self.standard_spellings.items():
            if american in words:
                american_count += Counter(words)[american]
            for british in british_variants:
                if british in words:
                    british_count += Counter(words)[british]

        total_variant_words = british_count + american_count
        if total_variant_words > 0:
            british_ratio = british_count / total_variant_words
            american_ratio = american_count / total_variant_words
        else:
            british_ratio = 0.0
            american_ratio = 0.0

        if british_ratio > 0.7:
            spelling_preference = "british"
        elif american_ratio > 0.7:
            spelling_preference = "american"
        else:
            spelling_preference = "mixed"

        return {
            "misspelling_count": misspelling_count,
            "misspelling_ratio": misspelling_count / len(words),
            "consistent_misspellings": list(set(found_misspellings)),
            "british_spelling_ratio": british_ratio,
            "american_spelling_ratio": american_ratio,
            "spelling_preference": spelling_preference,
        }

    def _extract_phrase_patterns(self, text: str) -> Dict[str, Any]:
        """Extract phrase and collocation patterns."""
        words = tokenize_words(text)

        if len(words) < 2:
            return {
                "unique_bigrams": [],
                "repeated_phrases": [],
                "catchphrase_candidates": [],
            }

        bigrams = get_word_ngrams(words, 2)
        trigrams = get_word_ngrams(words, 3) if len(words) >= 3 else []

        bigram_counts = Counter(bigrams)
        trigram_counts = Counter(trigrams)

        repeated_bigrams = [
            (" ".join(bg), count)
            for bg, count in bigram_counts.items()
            if count >= 2
        ]
        repeated_bigrams.sort(key=lambda x: x[1], reverse=True)

        repeated_trigrams = [
            (" ".join(tg), count)
            for tg, count in trigram_counts.items()
            if count >= 2
        ]

        stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "must", "shall", "can", "to", "of", "in", "for", "on", "with", "at", "by", "from", "as", "into", "through", "during", "before", "after", "above", "below", "between", "under", "again", "further", "then", "once", "and", "but", "or", "nor", "so", "yet", "both", "either", "neither", "not", "only", "own", "same", "than", "too", "very", "just", "also"}

        unique_bigrams = []
        for bg, count in bigram_counts.items():
            if bg[0] not in stop_words and bg[1] not in stop_words:
                unique_bigrams.append((" ".join(bg), count))

        unique_bigrams.sort(key=lambda x: x[1], reverse=True)

        catchphrases = []
        for phrase, count in repeated_trigrams[:10]:
            words_in_phrase = phrase.split()
            if not all(w in stop_words for w in words_in_phrase):
                catchphrases.append({"phrase": phrase, "count": count})

        return {
            "unique_bigrams": unique_bigrams[:20],
            "repeated_phrases": repeated_bigrams[:20],
            "catchphrase_candidates": catchphrases[:10],
        }

    def _extract_linguistic_quirks(self, text: str) -> Dict[str, Any]:
        """Extract unusual linguistic patterns."""
        words = tokenize_words(text)

        filler_words = ["like", "you know", "basically", "actually", "literally", "honestly", "um", "uh", "well", "so", "i mean", "kind of", "sort of"]

        filler_counts = {}
        text_lower = text.lower()
        for filler in filler_words:
            count = len(re.findall(r'\b' + re.escape(filler) + r'\b', text_lower))
            if count > 0:
                filler_counts[filler] = count

        sentences = re.split(r'[.!?]+', text)
        sentence_starters = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                first_word = sentence.split()[0].lower() if sentence.split() else ""
                sentence_starters.append(first_word)

        starter_counts = Counter(sentence_starters)
        starter_signature = dict(starter_counts.most_common(10))

        punctuation_quirks = []

        if text.count('...') > 2:
            punctuation_quirks.append("frequent_ellipsis")
        if text.count('!!') > 0:
            punctuation_quirks.append("double_exclamation")
        if text.count('??') > 0:
            punctuation_quirks.append("double_question")
        if text.count('!?') > 0 or text.count('?!') > 0:
            punctuation_quirks.append("interrobang_style")
        if text.count(' - ') > 2:
            punctuation_quirks.append("dash_interrupter")
        if text.count('—') > 2:
            punctuation_quirks.append("em_dash_user")

        capitalization_quirks = []

        all_caps_words = re.findall(r'\b[A-Z]{2,}\b', text)
        if len(all_caps_words) > 3:
            capitalization_quirks.append("frequent_all_caps")

        mid_sentence_caps = re.findall(r'[a-z]\s+[A-Z][a-z]', text)
        if len(mid_sentence_caps) > 2:
            capitalization_quirks.append("mid_sentence_caps")

        return {
            "filler_word_pattern": filler_counts,
            "sentence_starter_signature": starter_signature,
            "punctuation_quirks": punctuation_quirks,
            "capitalization_quirks": capitalization_quirks,
        }

    def _extract_consistency_patterns(self, text: str) -> Dict[str, Any]:
        """Extract consistency patterns in writing choices."""
        contractions = ["i'm", "i've", "i'll", "i'd", "you're", "you've", "you'll", "you'd", "he's", "she's", "it's", "we're", "we've", "we'll", "they're", "they've", "they'll", "don't", "doesn't", "didn't", "won't", "wouldn't", "can't", "couldn't", "shouldn't", "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't", "hadn't", "let's", "that's", "there's", "here's", "what's", "who's"]

        expanded_forms = ["i am", "i have", "i will", "i would", "you are", "you have", "you will", "you would", "he is", "she is", "it is", "we are", "we have", "we will", "they are", "they have", "they will", "do not", "does not", "did not", "will not", "would not", "can not", "cannot", "could not", "should not", "is not", "are not", "was not", "were not", "have not", "has not", "had not", "let us", "that is", "there is", "here is", "what is", "who is"]

        text_lower = text.lower()

        contraction_count = sum(text_lower.count(c) for c in contractions)
        expanded_count = sum(text_lower.count(e) for e in expanded_forms)

        total = contraction_count + expanded_count
        contraction_preference = contraction_count / total if total > 0 else 0.5

        single_quotes = text.count("'") + text.count("'") + text.count("'")
        double_quotes = text.count('"') + text.count('"') + text.count('"')

        if double_quotes > single_quotes * 2:
            quote_style = "double"
        elif single_quotes > double_quotes * 2:
            quote_style = "single"
        else:
            quote_style = "mixed"

        return {
            "contraction_preference": contraction_preference,
            "quote_style": quote_style,
        }

    def get_author_signature(self, text: str) -> Dict[str, Any]:
        """
        Get a compact signature of distinctive author features.

        Returns the most identifying characteristics.
        """
        features = self.extract(text)

        signature = {
            "spelling_style": features.get("spelling_preference", "neutral"),
            "top_fillers": list(features.get("filler_word_pattern", {}).keys())[:3],
            "top_starters": list(features.get("sentence_starter_signature", {}).keys())[:5],
            "punctuation_style": features.get("punctuation_quirks", []),
            "contraction_tendency": "high" if features.get("contraction_preference", 0.5) > 0.7 else "low" if features.get("contraction_preference", 0.5) < 0.3 else "moderate",
            "catchphrases": [p["phrase"] for p in features.get("catchphrase_candidates", [])[:3]],
        }

        return signature

    def compare_idiolects(
        self,
        text1: str,
        text2: str,
    ) -> Dict[str, Any]:
        """
        Compare idiolect features between two texts.

        Returns similarity assessment.
        """
        features1 = self.extract(text1)
        features2 = self.extract(text2)

        similarities = []
        differences = []

        if features1.get("spelling_preference") == features2.get("spelling_preference"):
            similarities.append(f"Same spelling preference: {features1.get('spelling_preference')}")
        else:
            differences.append(f"Different spelling: {features1.get('spelling_preference')} vs {features2.get('spelling_preference')}")

        fillers1 = set(features1.get("filler_word_pattern", {}).keys())
        fillers2 = set(features2.get("filler_word_pattern", {}).keys())
        common_fillers = fillers1 & fillers2

        if common_fillers:
            similarities.append(f"Shared filler words: {', '.join(common_fillers)}")

        quirks1 = set(features1.get("punctuation_quirks", []))
        quirks2 = set(features2.get("punctuation_quirks", []))
        common_quirks = quirks1 & quirks2

        if common_quirks:
            similarities.append(f"Shared punctuation quirks: {', '.join(common_quirks)}")

        pref1 = features1.get("contraction_preference", 0.5)
        pref2 = features2.get("contraction_preference", 0.5)

        if abs(pref1 - pref2) < 0.2:
            similarities.append("Similar contraction usage")
        else:
            differences.append("Different contraction tendencies")

        similarity_score = len(similarities) / (len(similarities) + len(differences) + 1)

        return {
            "similarity_score": similarity_score,
            "similarities": similarities,
            "differences": differences,
        }
