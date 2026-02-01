"""
Ensemble methods for authorship attribution.

Combines multiple classifiers for improved accuracy and robustness.

Security Note: Model serialization uses joblib. Only load models
from trusted sources.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import joblib
from pathlib import Path

from seshat.ml.classifier import AuthorshipClassifier


class EnsembleClassifier:
    """
    Ensemble classifier combining multiple authorship attribution models.

    Supports:
    - Voting (majority or weighted)
    - Stacking
    - Probability averaging
    """

    def __init__(
        self,
        algorithms: Optional[List[str]] = None,
        weights: Optional[List[float]] = None,
        voting: str = "soft",
        random_state: int = 42,
    ):
        """
        Initialize ensemble classifier.

        Args:
            algorithms: List of algorithms to include (default: ["svm", "random_forest"])
            weights: Optional weights for each classifier
            voting: "hard" (majority) or "soft" (probability average)
            random_state: Random seed
        """
        if algorithms is None:
            algorithms = ["svm", "random_forest"]

        self.algorithms = algorithms
        self.weights = weights or [1.0] * len(algorithms)
        self.voting = voting
        self.random_state = random_state

        self.classifiers: List[AuthorshipClassifier] = []
        self.is_fitted = False
        self.label_map: Dict[int, str] = {}
        self.reverse_label_map: Dict[str, int] = {}

    def fit(
        self,
        X: np.ndarray,
        y: List[str],
        feature_names: Optional[List[str]] = None,
    ) -> "EnsembleClassifier":
        """
        Train all classifiers in the ensemble.

        Args:
            X: Feature matrix
            y: Author labels
            feature_names: Optional feature names

        Returns:
            self
        """
        unique_labels = sorted(set(y))
        self.label_map = {i: label for i, label in enumerate(unique_labels)}
        self.reverse_label_map = {label: i for i, label in enumerate(unique_labels)}

        self.classifiers = []

        for algorithm in self.algorithms:
            try:
                classifier = AuthorshipClassifier(
                    algorithm=algorithm,
                    random_state=self.random_state,
                )
                classifier.fit(X, y, feature_names=feature_names)
                self.classifiers.append(classifier)
            except Exception as e:
                print(f"Warning: Failed to train {algorithm}: {e}")
                continue

        if not self.classifiers:
            raise RuntimeError("No classifiers were successfully trained")

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> List[str]:
        """
        Predict author labels using ensemble voting.

        Args:
            X: Feature matrix

        Returns:
            List of predicted labels
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble has not been fitted")

        if self.voting == "soft":
            return self._predict_soft(X)
        else:
            return self._predict_hard(X)

    def _predict_hard(self, X: np.ndarray) -> List[str]:
        """Hard voting (majority vote)."""
        all_predictions = []

        for classifier, weight in zip(self.classifiers, self.weights):
            predictions = classifier.predict(X)
            all_predictions.append((predictions, weight))

        results = []
        for i in range(X.shape[0]):
            votes: Dict[str, float] = {}
            for predictions, weight in all_predictions:
                label = predictions[i]
                votes[label] = votes.get(label, 0) + weight

            winner = max(votes.items(), key=lambda x: x[1])[0]
            results.append(winner)

        return results

    def _predict_soft(self, X: np.ndarray) -> List[str]:
        """Soft voting (weighted probability average)."""
        combined_proba = self._get_combined_proba(X)

        results = []
        for proba in combined_proba:
            winner_idx = np.argmax(proba)
            results.append(self.label_map[winner_idx])

        return results

    def predict_proba(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get combined class probabilities.

        Args:
            X: Feature matrix

        Returns:
            Dictionary mapping author names to probability arrays
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble has not been fitted")

        combined_proba = self._get_combined_proba(X)

        result = {}
        for i, label in self.label_map.items():
            result[label] = combined_proba[:, i]

        return result

    def _get_combined_proba(self, X: np.ndarray) -> np.ndarray:
        """Get weighted average of probabilities from all classifiers."""
        n_samples = X.shape[0]
        n_classes = len(self.label_map)

        combined = np.zeros((n_samples, n_classes))
        total_weight = 0

        for classifier, weight in zip(self.classifiers, self.weights):
            proba_dict = classifier.predict_proba(X)

            for label_idx, label in self.label_map.items():
                if label in proba_dict:
                    combined[:, label_idx] += proba_dict[label] * weight

            total_weight += weight

        combined /= total_weight

        return combined

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
            raise RuntimeError("Ensemble has not been fitted")

        combined_proba = self._get_combined_proba(X)

        results = []
        for proba in combined_proba:
            top_indices = np.argsort(proba)[::-1][:k]
            top_predictions = [
                (self.label_map[idx], float(proba[idx]))
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
        Cross-validate the ensemble.

        Args:
            X: Feature matrix
            y: Labels
            cv: Number of folds

        Returns:
            Cross-validation results
        """
        from sklearn.model_selection import StratifiedKFold

        unique_labels = sorted(set(y))
        reverse_map = {label: i for i, label in enumerate(unique_labels)}
        y_encoded = np.array([reverse_map.get(label, 0) for label in y])

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)

        fold_scores = []

        for train_idx, test_idx in skf.split(X, y_encoded):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train = [y[i] for i in train_idx]
            y_test = [y[i] for i in test_idx]

            temp_ensemble = EnsembleClassifier(
                algorithms=self.algorithms,
                weights=self.weights,
                voting=self.voting,
                random_state=self.random_state,
            )
            temp_ensemble.fit(X_train, y_train)

            y_pred = temp_ensemble.predict(X_test)

            accuracy = sum(1 for p, t in zip(y_pred, y_test) if p == t) / len(y_test)
            fold_scores.append(accuracy)

        return {
            "mean_accuracy": float(np.mean(fold_scores)),
            "std_accuracy": float(np.std(fold_scores)),
            "fold_scores": fold_scores,
            "n_folds": cv,
        }

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: List[str],
    ) -> Dict[str, Any]:
        """
        Evaluate ensemble on test data.

        Args:
            X_test: Test feature matrix
            y_test: Test labels

        Returns:
            Evaluation metrics
        """
        from sklearn.metrics import classification_report, confusion_matrix

        if not self.is_fitted:
            raise RuntimeError("Ensemble has not been fitted")

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

    def get_classifier_agreement(self, X: np.ndarray) -> np.ndarray:
        """
        Measure agreement between classifiers.

        Args:
            X: Feature matrix

        Returns:
            Agreement scores (0-1) for each sample
        """
        if not self.is_fitted or len(self.classifiers) < 2:
            return np.ones(X.shape[0])

        all_predictions = []
        for classifier in self.classifiers:
            predictions = classifier.predict(X)
            all_predictions.append(predictions)

        agreement_scores = []
        for i in range(X.shape[0]):
            votes = [pred[i] for pred in all_predictions]
            most_common = max(set(votes), key=votes.count)
            agreement = votes.count(most_common) / len(votes)
            agreement_scores.append(agreement)

        return np.array(agreement_scores)

    def save(self, path: str) -> None:
        """
        Save ensemble to disk using joblib.

        Security Warning: Only load models from trusted sources.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "algorithms": self.algorithms,
            "weights": self.weights,
            "voting": self.voting,
            "random_state": self.random_state,
            "classifiers": self.classifiers,
            "is_fitted": self.is_fitted,
            "label_map": self.label_map,
            "reverse_label_map": self.reverse_label_map,
        }

        joblib.dump(state, path)

    @classmethod
    def load(cls, path: str) -> "EnsembleClassifier":
        """
        Load ensemble from disk.

        Security Warning: Only load models from trusted sources,
        as deserialization can execute arbitrary code.
        """
        state = joblib.load(path)

        ensemble = cls(
            algorithms=state["algorithms"],
            weights=state["weights"],
            voting=state["voting"],
            random_state=state["random_state"],
        )

        ensemble.classifiers = state["classifiers"]
        ensemble.is_fitted = state["is_fitted"]
        ensemble.label_map = state["label_map"]
        ensemble.reverse_label_map = state["reverse_label_map"]

        return ensemble
