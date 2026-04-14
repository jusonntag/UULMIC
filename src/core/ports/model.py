from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np


class BaseModelPort(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model"""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels"""
        pass
        
    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate and return metrics like accuracy or loss"""
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Return model parameters for tracking"""
        pass
