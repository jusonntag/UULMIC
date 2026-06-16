from abc import ABC, abstractmethod
from typing import Dict, Any


class TrackerPort(ABC):
    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters of the trial"""
        pass

    @abstractmethod
    def init_run(self, run_name: str) -> None:
        """Init new run"""
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: int | None = None) -> None:
        """Log trial metrics"""
        pass
        
    @abstractmethod
    def finish(self) -> None:
        """Close the tracking session"""
        pass
