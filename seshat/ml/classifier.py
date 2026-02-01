"""
Machine learning classifiers for authorship attribution.

Implements SVM, Random Forest, and XGBoost classifiers with
cross-validation and model persistence.

Security Note: Model serialization uses joblib instead of pickle
for safer handling. However, users should only load models from
trusted sources.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import joblib

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class AuthorshipClassifier:
    """
    Machine learning classifier for authorship attribution.

    Supports multiple algorithms:
    - SVM (Support Vector Machine)
    - Random Forest
    - XGBoost (if available)
    """

    def __init__(
        self,
        algorithm: str = "svm",
        random_state: int = 42,
    ):
        """
        Initialize the classifier.

        Args:
            algorithm: "svm", "random_forest", or "xgboost"
            random_state: Random seed for reproducibility
        """
        self.algorithm = algorithm
        self.random_state = random_state

        self.model = None
        self.scaler = StandardScaler()
        self.label_map: Dict[int, str] = {}
        self.reverse_label_map: Dict[str, int] = {}
        self.feature_names: List[str] = []
        self.is_fitted = False

        self._init_model()

    def _init_model(self) -> None:
        """Initialize the underlying model."""
        if self.algorithm == "svm":
            self.model = SVC(
                kernel="rbf",
                C=10,
                gamma="scale",
                probability=True,
                random_state=self.random_state,
            )
        elif self.algorithm == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=self.random_state,
                n_jobs=-1,
            )
        elif self.algorithm == "xgboost":
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost is not installed. Use 'pip install xgboost'")
            self.model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric="mlogloss",
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def fit(
        self,
        X: np.ndarray,
        y: List[str],
        feature_names: Optional[List[str]] = None,
    ) -> "AuthorshipClassifier":
        """
        Train the classifier.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Author labels
            feature_names: Optional list of feature names

        Returns:
            self
        """
        unique_labels = sorted(set(y))
        self.label_map = {i: label for i, label in enumerate(unique_labels)}
        self.reverse_label_map = {label: i for i, label in enumerate(unique_labels)}

        y_encoded = np.array([self.reverse_label_map[label] for label in y])

        if feature_names:
            self.feature_names = feature_names

        X_scaled = self.scaler.fit_transform(X)

        self.model.fit(X_scaled, y_encoded)
        self.is_fitted = True

        return self

    def predict(self, X: np.ndarray) -> List[str]:
        """
        Predict author labels.

        Args:
            X: Feature matrix

        Returns:
            List of predicted author labels
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier has not been fitted")

        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)

        return [self.label_map[int(label)] for label in y_pred]

    def predict_proba(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict class probabilities.

        Args:
            X: Feature matrix

        Returns:
            Dictionary mapping author names to probability arrays
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier has not been fitted")

        X_scaled = self.scaler.transform(X)
        probas = self.model.predict_proba(X_scaled)

        result = {}
        for i, label in self.label_map.items():
            result[label] = probas[:, i]

        return result

    def predict_top_k(
        self, X: np.ndarray, k: int = 5
    ) -> List[List[Tuple[str, float]]]:
        """
        Get top-k predictions with probabilities.

        Args:
            X: Feature matrix
            k: Number of top predictions

        Returns:
            List of lists of (author, probability) tuples
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier has not been fitted")

        X_scaled = self.scaler.transform(X)
        probas = self.model.predict_proba(X_scaled)

        results = []
        for proba in probas:
            top_indices = np.argsort(proba)[::-1][:k]
            top_predictions = [
                (self.label_map[idx], proba[idx])
                for idx in top_indices
            ]
            results.append(top_predictions)

        return results

    def cross_validate(
        self,
        X: np.ndarray,
        y: List[str],
        cv: int = 5,
    ) -> Dict[str, Any]:
        """
        Perform cross-validation.

        Args:
            X: Feature matrix
            y: Author labels
            cv: Number of folds

        Returns:
            Cross-validation results
        """
        unique_labels = sorted(set(y))
        label_map = {label: i for i, label in enumerate(unique_labels)}
        y_encoded = np.array([label_map[label] for label in y])

        X_scaled = self.scaler.fit_transform(X)

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)

        scores = cross_val_score(
            self.model, X_scaled, y_encoded,
            cv=skf, scoring="accuracy"
        )

        return {
            "mean_accuracy": float(np.mean(scores)),
            "std_accuracy": float(np.std(scores)),
            "fold_scores": scores.tolist(),
            "n_folds": cv,
        }

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: List[str],
    ) -> Dict[str, Any]:
        """
        Evaluate the classifier on test data.

        Args:
            X_test: Test feature matrix
            y_test: Test labels

        Returns:
            Evaluation metrics
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier has not been fitted")

        y_pred = self.predict(X_test)

        y_test_encoded = [self.reverse_label_map.get(label, -1) for label in y_test]
        y_pred_encoded = [self.reverse_label_map.get(label, -1) for label in y_pred]

        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test_encoded, y_pred_encoded)

        return {
            "accuracy": report["accuracy"],
            "classification_report": report,
            "confusion_matrix": conf_matrix.tolist(),
            "labels": list(self.label_map.values()),
        }

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores (for Random Forest and XGBoost).

        Returns:
            Dictionary mapping feature names to importance scores, or None
        """
        if not self.is_fitted:
            return None

        if self.algorithm in ["random_forest", "xgboost"]:
            importances = self.model.feature_importances_

            if self.feature_names and len(self.feature_names) == len(importances):
                return dict(zip(self.feature_names, importances))
            else:
                return {f"feature_{i}": imp for i, imp in enumerate(importances)}

        return None

    def save(self, path: str) -> None:
        """
        Save the trained model to disk using joblib.

        Security Warning: Only load models from trusted sources.

        Args:
            path: Output file path
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted classifier")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "algorithm": self.algorithm,
            "random_state": self.random_state,
            "model": self.model,
            "scaler": self.scaler,
            "label_map": self.label_map,
            "reverse_label_map": self.reverse_label_map,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted,
        }

        joblib.dump(state, path)

    @classmethod
    def load(cls, path: str) -> "AuthorshipClassifier":
        """
        Load a trained model from disk.

        Security Warning: Only load models from trusted sources,
        as deserialization can execute arbitrary code.

        Args:
            path: Model file path

        Returns:
            Loaded classifier
        """
        state = joblib.load(path)

        classifier = cls(
            algorithm=state["algorithm"],
            random_state=state["random_state"],
        )

        classifier.model = state["model"]
        classifier.scaler = state["scaler"]
        classifier.label_map = state["label_map"]
        classifier.reverse_label_map = state["reverse_label_map"]
        classifier.feature_names = state["feature_names"]
        classifier.is_fitted = state["is_fitted"]

        return classifier


def train_classifier(
    profiles: List[Any],
    algorithm: str = "svm",
    test_size: float = 0.2,
    cross_validate: bool = True,
    cv_folds: int = 5,
) -> Tuple[AuthorshipClassifier, Dict[str, Any]]:
    """
    Train a classifier from author profiles.

    Args:
        profiles: List of AuthorProfile objects
        algorithm: Classifier algorithm
        test_size: Fraction of data to use for testing
        cross_validate: Whether to perform cross-validation
        cv_folds: Number of cross-validation folds

    Returns:
        Tuple of (trained classifier, training results)
    """
    from sklearn.model_selection import train_test_split

    X_list = []
    y_list = []
    feature_names = None

    for profile in profiles:
        if not profile.samples:
            continue

        if feature_names is None:
            feature_names = profile.get_feature_names()

        for sample in profile.samples:
            if sample.analysis:
                features = sample.analysis.get_flat_features()
                feature_vector = [features.get(name, 0.0) for name in feature_names]
                X_list.append(feature_vector)
                y_list.append(profile.name)

    if not X_list:
        raise ValueError("No valid samples found in profiles")

    X = np.array(X_list)
    y = y_list

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    classifier = AuthorshipClassifier(algorithm=algorithm)

    results = {"algorithm": algorithm}

    if cross_validate:
        cv_results = classifier.cross_validate(X_train, y_train, cv=cv_folds)
        results["cross_validation"] = cv_results

    classifier.fit(X_train, y_train, feature_names=feature_names)

    eval_results = classifier.evaluate(X_test, y_test)
    results["evaluation"] = eval_results

    importance = classifier.get_feature_importance()
    if importance:
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        results["top_features"] = sorted_importance[:20]

    return classifier, results
