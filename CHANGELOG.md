# Changelog

All notable changes to Seshat are documented in this file.

## [1.0.1] - 2026-02-01

### Improvements
- Significantly improved AI text detection accuracy (65% → 75-100% for AI-generated text)
- Added AI-specific phrase detection (distinguishes AI patterns from common formal phrases)
- Added human authenticity marker detection (quoted speech, specific names/dates, hedging, personal anecdotes, academic methodology references)
- Reduced false positive rate for formal human writing (academic papers, professional emails, news articles)
- Improved cognitive marker detection with context-aware weighting
- Added academic-specific authenticity markers (citations, methodology language, study references)

### Bug Fixes
- Fixed false positive in typo detection caused by substring matching (now uses word boundaries)
- Fixed first-person pronoun scoring to account for formal academic AI writing style
- Fixed vocabulary distribution scoring to require longer text before penalizing
- Fixed formal transition detection to require higher thresholds

## [1.0.0] - 2026-02-01

### Initial Release

Comprehensive stylometric authorship attribution and psychological profiling tool.

#### Core Features
- Full stylometric analysis engine with 10+ feature extraction modules
- Author profile management with aggregated features
- Text comparison using Burrow's Delta, cosine similarity, and other metrics
- Confidence-based scoring system

#### Feature Extraction
- Lexical features (TTR, Yule's K, Simpson's D, Honoré's R)
- Function word analysis (pronouns, articles, prepositions)
- Punctuation pattern analysis
- Character and word n-grams
- Syntactic features with optional spaCy integration
- Emoji and emoticon analysis
- Social media specific features

#### Psychological Profiling
- Big Five (OCEAN) personality trait analysis
- Emotional tone and sentiment analysis
- Cognitive style assessment
- Social dynamics indicators

#### Advanced Analysis
- AI vs human text detection
- Cross-platform consistency analysis
- Temporal style drift tracking
- Native language identification
- Multi-author detection
- Adversarial detection (obfuscation)

#### Infrastructure
- REST API with FastAPI
- Web interface with Flask
- Command-line interface
- Web scraping (Twitter/X, Reddit, generic web)
- Database persistence with SQLAlchemy
- Evidence chain management for forensics
- Privacy controls (PII redaction, anonymization)

### Security Fixes
- Removed hardcoded secret key fallback - now generates random key with warning
- Disabled debug mode by default in web application
- Added path traversal protection to profile save/load operations
- Added path traversal protection to evidence chain export/load

### Bug Fixes
- Fixed `Sample.from_dict()` to properly restore `word_count` and `features` fields
- Fixed tokenization inconsistency in `add_sample()` - now uses `tokenize_words()`
- Added empty text validation to `QuickAnalyzer.analyze()`
- Added empty text validation to `Analyzer.analyze()` (raises ValueError)
- Added minimum word validation to `add_sample()` (default: 5 words)

### Improvements
- `add_samples()` now returns tuple of (successful, failures) with logging
- Profile loading now logs and warns on failures instead of silent skip
- Added comprehensive warnings for missing dependencies (spaCy, chardet, etc.)
- Added backward-compatible class aliases for all feature extractors

### Documentation
- Added comprehensive README.md with usage examples
- Added CHANGELOG.md

### Testing
- 83 unit and integration tests
- 50% code coverage
