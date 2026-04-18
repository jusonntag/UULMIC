from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from src.core.ports.tracker import TrackerPort

class BaseModelPort(ABC):
    @abstractmethod
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        tracker: Optional['TrackerPort'] = None
    ) -> None:
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

    @abstractmethod
    def reset(self) -> None:
        """Reset model weights and optimizer to a fresh state"""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the model weights to the specified path"""
        pass
