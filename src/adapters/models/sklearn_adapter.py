import numpy as np
from typing import Dict, Any, Optional
from src.core.ports.model import BaseModelPort
from src.core.domain.config import ModelConfig

class SklearnModelAdapter(BaseModelPort):
    """
    Adapter for any scikit-learn estimator or pipeline.
    Expects an estimator implementing `fit` and `predict` (or `predict_proba`).
    """
    def __init__(self, config: ModelConfig, estimator: Any):
        self.config = config
        self.estimator = estimator

    def _flatten_X(self, X: np.ndarray) -> np.ndarray:
        """Flatten epochs (n_trials, n_channels, n_times) to 2D for standard sklearn."""
        if X.ndim > 2:
            return X.reshape(X.shape[0], -1)
        return X

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_flat = self._flatten_X(X)
        self.estimator.fit(X_flat, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_flat = self._flatten_X(X)
        return self.estimator.predict(X_flat)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        X_flat = self._flatten_X(X)
        score = self.estimator.score(X_flat, y)
        return {"accuracy": float(score)}

    def get_params(self) -> Dict[str, Any]:
        # Return UULMI ModelConfig combined with sklearn estimator parameters
        params = self.config.model_dump()
        try:
            params.update(self.estimator.get_params())
        except AttributeError:
            # If estimator lacks get_params
            pass
        return params
