import numpy as np
from typing import Dict, Any, Optional
from src.core.ports.model import BaseModelPort
from src.core.domain.config import ModelConfig
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score
)

class SklearnModelAdapter(BaseModelPort):
    """Adapter for any scikit-learn estimator or pipeline."""
    def __init__(self, config: ModelConfig, estimator: Any):
        self.config = config
        self.estimator = estimator

    def _calc_csps(self, X: np.ndarray) -> np.ndarray:
        """Calculates Common Spatial Patters (CSPs) for X."""
        # TODO Implement CPSs
        raise NotImplementedError("Can't calculate CSPs yet.")

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None, tracker: Optional[Any] = None) -> None:
        """Trains the configured model with passed data."""
        X_csp = self._calc_csps(X)
        self.estimator.fit(X_csp, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Model calculates labels based on input data."""
        X_csp = self._calc_csps(X)
        return self.estimator.predict(X_csp)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Full calculation of metric scores: acc, precision, recall, f1 & kappa."""
        X_csp = self._calc_csps(X)
        preds = self.predict(X_csp)
        accuracy = (preds == y).mean()
        precision = precision_score(y, preds, average='weighted', zero_division=0)
        recall = recall_score(y, preds, average='weighted', zero_division=0)
        f1 = f1_score(y, preds, average='weighted', zero_division=0)
        kappa = cohen_kappa_score(y, preds)

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "kappa": float(kappa)
        }

    def get_params(self) -> Dict[str, Any]:
        """Log training config and total model parameters count."""
        raise NotImplementedError("SklearnModelAdapter does not support get_params() yes")

    def reset(self) -> None:
        # TODO: Implement proper reset for sklearn estimators
        raise NotImplementedError("SklearnModelAdapter does not support reset() yet")

    def save(self, path: str) -> None:
        # TODO: Implement model persistence
        raise NotImplementedError("SklearnModelAdapter does not support save() yet")
