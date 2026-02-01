# Seshat

> *Named after the Egyptian goddess of writing, wisdom, and measurement - "She who scribes"*

**Stylometric Authorship Attribution & Psychological Profiling Tool**

A comprehensive Python library for stylometric analysis that creates "writing fingerprints" from known samples, identifies authorship with confidence scoring, and performs psychological profiling based on linguistic markers.

## Features

### Core Capabilities
- **Authorship Attribution** - Compare unknown texts against author profiles with statistical confidence scoring
- **Psychological Profiling** - Big Five (OCEAN) personality trait analysis from linguistic markers
- **AI Detection** - Distinguish human-written content from AI-generated text
- **Cross-Platform Analysis** - Track writing consistency across different platforms

### Feature Extraction
- **Lexical Features** - Vocabulary richness (TTR, Yule's K, Simpson's D), word-level statistics
- **Function Words** - Pronoun, article, preposition, and conjunction analysis
- **Punctuation Patterns** - Terminal punctuation, comma density, special character usage
- **N-gram Analysis** - Character and word n-grams (2-5 grams)
- **Syntactic Features** - Sentence complexity, clause analysis, POS patterns
- **Emoji & Social Media** - Platform-specific features, emoji usage patterns
- **Idiolect Markers** - Consistent typos, unique word combinations, signature phrases

### Advanced Analysis
- **Temporal Style Drift** - Track writing style evolution over time
- **Native Language Identification** - Detect L1 interference patterns
- **Multi-Author Detection** - Identify documents with multiple authors
- **Adversarial Detection** - Detect obfuscation attempts
- **Demographics Estimation** - Age, education level, regional patterns

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/seshat.git
cd seshat

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model (optional, for enhanced syntactic analysis)
python -m spacy download en_core_web_sm
```

## Quick Start

### Basic Usage

```python
from seshat.analyzer import Analyzer
from seshat.profile import AuthorProfile, ProfileManager
from seshat.comparator import Comparator

# Create an analyzer
analyzer = Analyzer()

# Analyze a text
result = analyzer.analyze("""
    I believe that education is fundamental to success.
    Throughout my career, I have observed that dedicated
    individuals consistently achieve their goals.
""")

print(f"Word count: {result.word_count}")
print(f"Type-token ratio: {result.lexical_features['type_token_ratio']:.3f}")
print(f"Average sentence length: {result.syntactic_features.get('avg_sentence_length', 0):.1f}")
```

### Creating Author Profiles

```python
# Initialize profile manager
manager = ProfileManager(storage_dir="./profiles")

# Create a profile with sample texts
profile = manager.create_profile(
    name="AuthorA",
    samples=[
        "First sample text with the author's characteristic style...",
        "Second sample showing consistent patterns...",
        "Third sample for more reliable profiling...",
    ]
)

print(f"Profile created: {profile.name}")
print(f"Samples: {profile.get_sample_count()}")
print(f"Total words: {profile.get_total_word_count()}")
```

### Comparing Texts

```python
# Create comparator
comparator = Comparator(analyzer=analyzer)

# Compare unknown text against profile
unknown_text = "This is a text from an unknown author..."
result = comparator.compare(unknown_text, profile)

print(f"Match score: {result.overall_score:.2%}")
print(f"Confidence: {result.confidence}")
print(f"Burrow's Delta: {result.burrows_delta:.4f}")
print(f"Cosine Similarity: {result.cosine_similarity:.4f}")
```

### Psychological Profiling

```python
from seshat.psychology.personality import PersonalityAnalyzer
from seshat.psychology.emotional import EmotionalAnalyzer
from seshat.psychology.cognitive import CognitiveAnalyzer

text = "Your sample text here..."

# Big Five personality analysis
personality = PersonalityAnalyzer().analyze(text)
print(f"Openness: {personality['openness']['score']:.2f}")
print(f"Conscientiousness: {personality['conscientiousness']['score']:.2f}")

# Emotional tone analysis
emotional = EmotionalAnalyzer().analyze(text)
print(f"Sentiment: {emotional['sentiment_label']}")
print(f"Emotional intensity: {emotional['emotional_intensity']:.2f}")

# Cognitive style analysis
cognitive = CognitiveAnalyzer().analyze(text)
print(f"Analytical score: {cognitive['analytical_score']:.2f}")
```

### AI Detection

```python
from seshat.advanced.ai_detection import AIDetector

detector = AIDetector()
result = detector.detect(text)

print(f"Classification: {result['classification']}")
print(f"AI Probability: {result['ai_probability']:.2%}")
```

## CLI Usage

```bash
# Analyze a file
python cli.py analyze document.txt

# Create a profile
python cli.py profile create "AuthorName" --samples ./samples/

# Compare text to profiles
python cli.py compare unknown.txt --profiles ./profiles/

# Psychological profiling
python cli.py psych document.txt --full

# AI detection
python cli.py ai-detect suspicious.txt
```

## Web Interface

```bash
# Set environment variables for production
export SECRET_KEY="your-secure-secret-key"
export FLASK_DEBUG=false
export FLASK_HOST=127.0.0.1
export FLASK_PORT=5000

# Run the web application
python -m web.app
```

Then open http://localhost:5000 in your browser.

## API

```bash
# Start the API server
python -m api.app
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/profiles` | Create new profile |
| GET | `/api/v1/profiles` | List all profiles |
| GET | `/api/v1/profiles/{id}` | Get profile details |
| POST | `/api/v1/analyze` | Analyze text |
| POST | `/api/v1/compare` | Compare text to profile |
| DELETE | `/api/v1/profiles/{id}` | Delete profile |

## Project Structure

```
seshat/
├── seshat/                 # Core library
│   ├── features/          # Feature extraction modules
│   │   ├── lexical.py     # Vocabulary metrics
│   │   ├── function_words.py  # Function word analysis
│   │   ├── punctuation.py # Punctuation patterns
│   │   ├── ngrams.py      # N-gram extraction
│   │   ├── syntactic.py   # Sentence structure
│   │   ├── emoji.py       # Emoji/emoticon analysis
│   │   └── social_media.py # Platform-specific features
│   ├── psychology/        # Psychological profiling
│   │   ├── personality.py # Big Five traits
│   │   ├── emotional.py   # Emotional tone
│   │   ├── cognitive.py   # Thinking style
│   │   └── mental_health.py # Mental health indicators
│   ├── advanced/          # Advanced analysis
│   │   ├── ai_detection.py
│   │   ├── temporal.py
│   │   ├── nli.py
│   │   └── cross_platform.py
│   ├── analyzer.py        # Core analysis engine
│   ├── profile.py         # Profile management
│   ├── comparator.py      # Text comparison
│   └── utils.py           # Utility functions
├── api/                   # REST API
├── web/                   # Web interface
├── scraper/               # Web scraping modules
├── database/              # Data persistence
├── forensics/             # Forensic features
├── privacy/               # Privacy controls
├── tests/                 # Test suite
└── cli.py                 # Command-line interface
```

## Comparison Algorithms

### Burrow's Delta
The primary distance metric for authorship attribution:
```
Delta = (1/n) * Σ |z_author(f) - z_text(f)|
```

### Additional Metrics
- **Cosine Similarity** - Vector space comparison
- **Manhattan Distance** - Sum of absolute differences
- **Euclidean Distance** - Straight-line distance

### Confidence Scoring
```python
Final Score =
    0.30 * statistical_similarity +
    0.30 * ml_prediction_confidence +
    0.20 * distinctive_feature_match +
    0.20 * cross_validation_consistency
```

Confidence levels: Very High (≥85%), High (≥70%), Medium (≥55%), Low (≥40%), Very Low (<40%)

## Security Considerations

- **Secret Key**: Set `SECRET_KEY` environment variable in production
- **Debug Mode**: Disabled by default; use `FLASK_DEBUG=true` only in development
- **Path Traversal Protection**: File operations validate paths against base directories
- **Rate Limiting**: Implement rate limiting for production API deployments

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=seshat --cov-report=html

# Run specific test file
python -m pytest tests/unit/test_analyzer.py -v
```

## Dependencies

### Core
- numpy, scipy, pandas - Numerical computing
- nltk, spacy - Natural language processing
- scikit-learn - Machine learning

### Optional
- sentence-transformers - Deep learning embeddings
- torch - PyTorch backend
- playwright - Browser automation for scraping

See `requirements.txt` for full list.

## Research Background

Based on established stylometric research:
- Abbasi & Chen (2008) - Writeprints
- Pennebaker et al. - LIWC development
- Koppel et al. - Computational authorship attribution
- Brennan, Afroz & Greenstadt (2012) - Adversarial stylometry

## License

MIT License - See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- Named after Seshat, the ancient Egyptian goddess of writing, wisdom, and measurement
- Inspired by JGAAP, LIWC, and Stylo projects
