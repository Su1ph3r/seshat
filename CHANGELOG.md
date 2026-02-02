# Changelog

All notable changes to Seshat are documented in this file.

## [1.0.2] - 2026-02-02

### Major Features: Personality Disorder Analysis v2.0

Complete overhaul of the personality disorder linguistic indicator analysis with 18 new accuracy enhancements across 6 modular layers.

#### Linguistic Layer (`pd_linguistic.py`)
- Multi-word phrase detection with DSM-5 aligned patterns (e.g., "out to get me", "can't trust anyone")
- Negation handling with 5-word window radius to properly adjust scores
- Context window extraction for forensic review of marker occurrences
- Syntactic pattern analysis (passive voice ratio, sentence complexity, question/exclamation ratios)

#### Calibration Layer (`pd_calibration.py`)
- Z-score baseline normalization against empirically-derived population means
- Automatic genre detection (formal, informal, clinical, social_media)
- Genre-specific score adjustments to reduce false positives
- Enhanced confidence calibration considering multiple factors

#### Validation Layer (`pd_validation.py`)
- Cross-disorder discriminant validity checks (flags contradictory patterns like schizoid + histrionic)
- Minimum viable marker thresholds per disorder
- Interpersonal circumplex mapping (dominance vs affiliation dimensions)
- Profile clarity scoring

#### Advanced Metrics (`pd_advanced_metrics.py`)
- Temporal pattern analysis (past/present/future focus with interpretation)
- Linguistic complexity metrics (vocabulary sophistication, lexical diversity, readability)
- Response style indicators (hedging, absolutism, deflection, self-reference, emotional expressiveness)

#### Temporal Analysis (`pd_temporal.py`)
- Multi-sample series analysis with trend detection
- Change point identification across text samples
- Stability scoring and dominant pattern detection

#### Optional ML Layers (`pd_semantic.py`, `pd_classifier.py`)
- Embedding-based semantic similarity to disorder prototype texts
- Topic modeling with LDA/NMF and disorder relevance scoring
- Machine learning classifier with RandomForest/GradientBoosting/Logistic options
- Secure model persistence with path validation and structure verification
- Feature extraction with 32 linguistic features for ML classification

### New Files
- `seshat/psychology/pd_linguistic.py` - Linguistic analysis layer
- `seshat/psychology/pd_calibration.py` - Score calibration and genre detection
- `seshat/psychology/pd_validation.py` - Cross-disorder validation
- `seshat/psychology/pd_advanced_metrics.py` - Advanced linguistic metrics
- `seshat/psychology/pd_temporal.py` - Temporal series analysis
- `seshat/psychology/pd_semantic.py` - Embedding and topic modeling
- `seshat/psychology/pd_classifier.py` - ML classification
- `seshat/psychology/pd_dictionaries.py` - Linguistic resources and baselines

### New Methods
- `PersonalityDisorderIndicators.compare()` - Compare two texts for indicator differences
- `PersonalityDisorderIndicators.analyze_series()` - Analyze multiple texts over time
- `PersonalityDisorderIndicators.get_enhanced_forensic_report()` - Full analysis with all features

### Security Fixes
- Added path validation to ML model loading to prevent path traversal
- Added model structure validation to prevent corrupted file attacks
- Improved error handling for model persistence operations

### Improvements
- All features configurable via constructor flags (16 feature flags)
- Lazy loading for optional dependencies (sentence-transformers, sklearn)
- Full backward compatibility with v1.0 API
- Analyzer version updated to "2.0.0"

### Testing
- 86 tests for personality disorder modules (all passing)
- 91% code coverage for `personality_disorders.py`
- 169 total tests across the project

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
