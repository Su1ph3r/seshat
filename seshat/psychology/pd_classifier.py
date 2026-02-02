"""
Machine learning classifier layer for personality disorder detection.

Provides scikit-learn based classification with feature extraction,
training capability, and model persistence.

SECURITY NOTE: Model persistence uses pickle for scikit-learn compatibility.
Only load models from trusted sources.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Check for scikit-learn and joblib
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.multioutput import MultiOutputClassifier
    import joblib  # Preferred over pickle for sklearn models
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from seshat.utils import tokenize_words


@dataclass
class ClassificationResult:
    """Result of ML classification."""
    disorder: str
    probability: float  # 0.0 to 1.0
    confidence: str  # "low", "medium", "high"
    contributing_features: List[Tuple[str, float]]  # Top features and their importance


@dataclass
class TrainingResult:
    """Result of model training."""
    success: bool
    n_samples: int
    cv_scores: Dict[str, float]  # Cross-validation scores per disorder
    feature_importances: Dict[str, Dict[str, float]]  # Per-disorder feature importance
    message: str


class PDFeatureExtractor:
    """Extract features from text for ML classification."""

    # Feature categories
    FEATURE_NAMES = [
        # Pronoun features
        "first_person_singular_ratio",
        "first_person_plural_ratio",
        "second_person_ratio",
        "third_person_ratio",

        # Emotional features
        "positive_emotion_ratio",
        "negative_emotion_ratio",
        "anger_ratio",
        "anxiety_ratio",
        "sadness_ratio",

        # Cognitive features
        "certainty_ratio",
        "tentative_ratio",
        "causation_ratio",
        "insight_ratio",
        "discrepancy_ratio",

        # Social features
        "social_ratio",
        "family_ratio",
        "friend_ratio",

        # Linguistic features
        "word_count",
        "avg_word_length",
        "sentence_length_mean",
        "sentence_length_std",
        "question_ratio",
        "exclamation_ratio",
        "negation_ratio",

        # Complexity features
        "type_token_ratio",
        "hapax_legomena_ratio",
        "complex_word_ratio",

        # PD-specific features
        "suspicion_words",
        "grandiosity_words",
        "abandonment_words",
        "control_words",
        "impulsivity_words",
    ]

    # Word lists for feature extraction
    WORD_LISTS = {
        "first_person_singular": ["i", "me", "my", "mine", "myself"],
        "first_person_plural": ["we", "us", "our", "ours", "ourselves"],
        "second_person": ["you", "your", "yours", "yourself", "yourselves"],
        "third_person": ["he", "she", "they", "him", "her", "them", "his", "hers", "their"],
        "positive_emotion": ["love", "happy", "good", "great", "wonderful", "excellent", "amazing", "joy"],
        "negative_emotion": ["hate", "sad", "bad", "terrible", "awful", "horrible", "miserable", "pain"],
        "anger": ["angry", "furious", "rage", "mad", "hate", "hostile", "annoyed", "irritated"],
        "anxiety": ["anxious", "worried", "nervous", "scared", "afraid", "fear", "panic", "terrified"],
        "sadness": ["sad", "depressed", "miserable", "unhappy", "hopeless", "lonely", "grief", "sorrow"],
        "certainty": ["always", "never", "definitely", "certainly", "absolutely", "completely", "totally"],
        "tentative": ["maybe", "perhaps", "possibly", "might", "could", "seems", "appears"],
        "causation": ["because", "since", "therefore", "hence", "thus", "consequently", "reason"],
        "insight": ["think", "know", "believe", "understand", "realize", "consider", "recognize"],
        "discrepancy": ["should", "would", "could", "ought", "need", "want", "wish", "hope"],
        "social": ["people", "person", "friend", "family", "social", "relationship", "together"],
        "family": ["mother", "father", "parent", "child", "family", "sister", "brother", "son", "daughter"],
        "friend": ["friend", "buddy", "pal", "companion", "colleague", "peer"],
        "negation": ["not", "no", "never", "none", "nothing", "neither", "nobody", "nowhere"],
        "suspicion": ["suspicious", "watching", "plotting", "lying", "deceiving", "betraying", "trust"],
        "grandiosity": ["best", "superior", "exceptional", "brilliant", "genius", "special", "unique"],
        "abandonment": ["leave", "alone", "abandon", "reject", "unwanted", "forgotten", "left"],
        "control": ["control", "order", "perfect", "must", "should", "rule", "strict", "proper"],
        "impulsivity": ["impulse", "sudden", "couldn't wait", "had to", "regret", "mistake"],
    }

    def __init__(self):
        """Initialize the feature extractor."""
        self.feature_names = self.FEATURE_NAMES

    def extract_features(self, text: str) -> np.ndarray:
        """
        Extract feature vector from text.

        Args:
            text: Input text

        Returns:
            Numpy array of feature values
        """
        words = tokenize_words(text)
        word_count = len(words)

        if word_count == 0:
            return np.zeros(len(self.feature_names))

        word_set = set(words)
        text_lower = text.lower()

        features = []

        # Helper function for ratio calculation
        def word_ratio(word_list: List[str]) -> float:
            count = sum(1 for w in words if w in word_list)
            return count / word_count if word_count > 0 else 0

        # Pronoun features
        features.append(word_ratio(self.WORD_LISTS["first_person_singular"]))
        features.append(word_ratio(self.WORD_LISTS["first_person_plural"]))
        features.append(word_ratio(self.WORD_LISTS["second_person"]))
        features.append(word_ratio(self.WORD_LISTS["third_person"]))

        # Emotional features
        features.append(word_ratio(self.WORD_LISTS["positive_emotion"]))
        features.append(word_ratio(self.WORD_LISTS["negative_emotion"]))
        features.append(word_ratio(self.WORD_LISTS["anger"]))
        features.append(word_ratio(self.WORD_LISTS["anxiety"]))
        features.append(word_ratio(self.WORD_LISTS["sadness"]))

        # Cognitive features
        features.append(word_ratio(self.WORD_LISTS["certainty"]))
        features.append(word_ratio(self.WORD_LISTS["tentative"]))
        features.append(word_ratio(self.WORD_LISTS["causation"]))
        features.append(word_ratio(self.WORD_LISTS["insight"]))
        features.append(word_ratio(self.WORD_LISTS["discrepancy"]))

        # Social features
        features.append(word_ratio(self.WORD_LISTS["social"]))
        features.append(word_ratio(self.WORD_LISTS["family"]))
        features.append(word_ratio(self.WORD_LISTS["friend"]))

        # Linguistic features
        features.append(min(1.0, word_count / 1000))  # Normalized word count
        features.append(sum(len(w) for w in words) / word_count if word_count > 0 else 0)

        # Sentence features
        sentences = self._split_sentences(text)
        sentence_lengths = [len(tokenize_words(s)) for s in sentences]
        if sentence_lengths:
            features.append(np.mean(sentence_lengths) / 50)  # Normalized
            features.append(np.std(sentence_lengths) / 20)  # Normalized
        else:
            features.append(0)
            features.append(0)

        features.append(text.count('?') / word_count if word_count > 0 else 0)
        features.append(text.count('!') / word_count if word_count > 0 else 0)
        features.append(word_ratio(self.WORD_LISTS["negation"]))

        # Complexity features
        unique_words = set(words)
        features.append(len(unique_words) / word_count if word_count > 0 else 0)  # TTR

        # Hapax legomena (words appearing once)
        from collections import Counter
        word_counts = Counter(words)
        hapax = sum(1 for w, c in word_counts.items() if c == 1)
        features.append(hapax / word_count if word_count > 0 else 0)

        # Complex words (3+ syllables)
        complex_count = sum(1 for w in words if self._count_syllables(w) >= 3)
        features.append(complex_count / word_count if word_count > 0 else 0)

        # PD-specific features
        features.append(word_ratio(self.WORD_LISTS["suspicion"]))
        features.append(word_ratio(self.WORD_LISTS["grandiosity"]))
        features.append(word_ratio(self.WORD_LISTS["abandonment"]))
        features.append(word_ratio(self.WORD_LISTS["control"]))
        features.append(word_ratio(self.WORD_LISTS["impulsivity"]))

        return np.array(features)

    def extract_features_batch(self, texts: List[str]) -> np.ndarray:
        """
        Extract features for multiple texts.

        Args:
            texts: List of texts

        Returns:
            Feature matrix (n_texts, n_features)
        """
        return np.array([self.extract_features(text) for text in texts])

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _count_syllables(self, word: str) -> int:
        """Estimate syllable count."""
        word = word.lower()
        count = 0
        vowels = 'aeiouy'
        prev_is_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_is_vowel:
                count += 1
            prev_is_vowel = is_vowel

        if word.endswith('e'):
            count -= 1

        return max(1, count)


class PDClassifier:
    """Machine learning classifier for personality disorder detection."""

    DISORDERS = [
        "paranoid", "schizoid", "schizotypal",
        "antisocial", "borderline", "histrionic", "narcissistic",
        "avoidant", "dependent", "obsessive_compulsive",
    ]

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_type: str = "random_forest",
    ):
        """
        Initialize the classifier.

        Args:
            model_path: Path to saved model (optional, only load from trusted sources)
            model_type: Type of model ("random_forest", "gradient_boosting", "logistic")
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for PDClassifier. "
                "Install with: pip install scikit-learn"
            )

        self.model_type = model_type
        self.feature_extractor = PDFeatureExtractor()
        self.scaler = StandardScaler()

        # Initialize models (one per disorder for multi-label classification)
        self.models: Dict[str, Any] = {}
        self._is_trained = False

        if model_path:
            self.load_model(model_path)
        else:
            self._init_models()

    def _init_models(self):
        """Initialize untrained models for each disorder."""
        for disorder in self.DISORDERS:
            self.models[disorder] = self._create_model()

    def _create_model(self):
        """Create a new model instance based on model_type."""
        if self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
            )
        elif self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42,
            )
        elif self.model_type == "logistic":
            return LogisticRegression(
                max_iter=1000,
                random_state=42,
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def predict(self, text: str) -> Dict[str, ClassificationResult]:
        """
        Predict personality disorder indicators for text.

        Args:
            text: Input text to classify

        Returns:
            Dictionary mapping disorder names to ClassificationResult
        """
        if not self._is_trained:
            raise RuntimeError("Model must be trained before prediction. Call train() first.")

        # Extract and scale features
        features = self.feature_extractor.extract_features(text)
        features_scaled = self.scaler.transform(features.reshape(1, -1))

        results = {}
        for disorder in self.DISORDERS:
            model = self.models[disorder]

            # Get probability
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(features_scaled)[0]
                # Get probability of positive class
                probability = float(proba[1]) if len(proba) > 1 else float(proba[0])
            else:
                # For models without predict_proba
                probability = float(model.predict(features_scaled)[0])

            # Determine confidence level
            if probability > 0.7:
                confidence = "high"
            elif probability > 0.4:
                confidence = "medium"
            else:
                confidence = "low"

            # Get feature importances
            contributing_features = self._get_contributing_features(model, features)

            results[disorder] = ClassificationResult(
                disorder=disorder,
                probability=probability,
                confidence=confidence,
                contributing_features=contributing_features[:5],
            )

        return results

    def predict_batch(self, texts: List[str]) -> List[Dict[str, ClassificationResult]]:
        """
        Predict for multiple texts.

        Args:
            texts: List of texts

        Returns:
            List of prediction dictionaries
        """
        return [self.predict(text) for text in texts]

    def train(
        self,
        texts: List[str],
        labels: Dict[str, List[int]],
        validation_split: float = 0.2,
    ) -> TrainingResult:
        """
        Train the classifier on labeled data.

        Args:
            texts: List of training texts
            labels: Dictionary of disorder -> list of labels (0 or 1)
            validation_split: Fraction for validation

        Returns:
            TrainingResult with training metrics
        """
        n_samples = len(texts)

        if n_samples < 10:
            return TrainingResult(
                success=False,
                n_samples=n_samples,
                cv_scores={},
                feature_importances={},
                message="Insufficient training data. Need at least 10 samples.",
            )

        # Extract features
        X = self.feature_extractor.extract_features_batch(texts)

        # Fit scaler
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        cv_scores = {}
        feature_importances = {}

        # Train each disorder model
        for disorder in self.DISORDERS:
            if disorder not in labels:
                continue

            y = np.array(labels[disorder])

            # Check for class balance
            if len(np.unique(y)) < 2:
                cv_scores[disorder] = 0.0
                continue

            # Cross-validation
            try:
                scores = cross_val_score(
                    self.models[disorder],
                    X_scaled,
                    y,
                    cv=min(5, n_samples // 2),
                    scoring='accuracy',
                )
                cv_scores[disorder] = float(np.mean(scores))
            except Exception:
                cv_scores[disorder] = 0.0

            # Train on full data
            self.models[disorder].fit(X_scaled, y)

            # Extract feature importances
            if hasattr(self.models[disorder], 'feature_importances_'):
                importances = self.models[disorder].feature_importances_
                feature_importances[disorder] = {
                    name: float(imp)
                    for name, imp in zip(
                        self.feature_extractor.feature_names,
                        importances
                    )
                }
            elif hasattr(self.models[disorder], 'coef_'):
                coefs = np.abs(self.models[disorder].coef_[0])
                feature_importances[disorder] = {
                    name: float(coef)
                    for name, coef in zip(
                        self.feature_extractor.feature_names,
                        coefs
                    )
                }

        self._is_trained = True

        return TrainingResult(
            success=True,
            n_samples=n_samples,
            cv_scores=cv_scores,
            feature_importances=feature_importances,
            message=f"Successfully trained on {n_samples} samples.",
        )

    def get_feature_importance(
        self,
        disorder: str,
    ) -> Dict[str, float]:
        """
        Get feature importance for a specific disorder.

        Args:
            disorder: Disorder name

        Returns:
            Dictionary of feature name -> importance
        """
        if not self._is_trained:
            return {}

        model = self.models.get(disorder)
        if model is None:
            return {}

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            return {}

        return {
            name: float(imp)
            for name, imp in zip(
                self.feature_extractor.feature_names,
                importances
            )
        }

    def _get_contributing_features(
        self,
        model,
        features: np.ndarray,
    ) -> List[Tuple[str, float]]:
        """Get features contributing most to the prediction."""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            return []

        # Weight by feature value
        contributions = importances * np.abs(features)

        # Sort by contribution
        indices = np.argsort(contributions)[::-1]

        return [
            (self.feature_extractor.feature_names[i], float(contributions[i]))
            for i in indices
        ]

    def save_model(self, path: str):
        """
        Save trained model to disk using joblib.

        SECURITY NOTE: Only load models from trusted sources.

        Args:
            path: Path to save model
        """
        if not self._is_trained:
            raise RuntimeError("Cannot save untrained model.")

        save_data = {
            "model_type": self.model_type,
            "models": self.models,
            "scaler": self.scaler,
            "feature_names": self.feature_extractor.feature_names,
        }

        # Use joblib (safer than pickle, optimized for numpy arrays)
        joblib.dump(save_data, path)

    def load_model(self, path: str):
        """
        Load trained model from disk.

        SECURITY WARNING: Only load models from trusted sources.
        Loading untrusted model files can execute arbitrary code.

        Args:
            path: Path to saved model

        Raises:
            FileNotFoundError: If model file does not exist
            ValueError: If model file is invalid or corrupted
        """
        model_path = Path(path).resolve()

        # Validate file exists
        if not model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")

        # Use joblib (safer than pickle, optimized for numpy arrays)
        try:
            save_data = joblib.load(model_path)
        except Exception as e:
            raise ValueError(f"Failed to load model from {path}: {e}")

        # Validate model structure
        required_keys = {"model_type", "models", "scaler"}
        if not required_keys.issubset(save_data.keys()):
            raise ValueError(
                f"Invalid model file: missing keys {required_keys - save_data.keys()}"
            )

        self.model_type = save_data["model_type"]
        self.models = save_data["models"]
        self.scaler = save_data["scaler"]
        self._is_trained = True

    def is_trained(self) -> bool:
        """Check if the model is trained."""
        return self._is_trained


def create_training_data_from_prototypes() -> Tuple[List[str], Dict[str, List[int]]]:
    """
    Create synthetic training data from disorder prototypes.

    This provides a baseline model without real labeled data.
    Not recommended for production use.

    Returns:
        Tuple of (texts, labels)
    """
    from .pd_semantic import DISORDER_PROTOTYPES

    texts = []
    labels = {disorder: [] for disorder in PDClassifier.DISORDERS}

    # Add prototype texts as positive examples
    for disorder, prototypes in DISORDER_PROTOTYPES.items():
        for proto in prototypes:
            texts.append(proto)
            for d in labels:
                labels[d].append(1 if d == disorder else 0)

    # Add some neutral texts as negative examples
    neutral_texts = [
        "The weather today is quite pleasant. I went for a walk in the park.",
        "I need to buy groceries for dinner. Maybe some vegetables and chicken.",
        "The meeting was productive. We discussed the quarterly results.",
        "I finished reading that book yesterday. It was interesting.",
        "Traffic was heavy on the way to work this morning.",
    ] * 5  # Repeat to balance

    for text in neutral_texts:
        texts.append(text)
        for d in labels:
            labels[d].append(0)

    return texts, labels
