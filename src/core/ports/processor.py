from abc import ABC, abstractmethod
from src.core.domain.config import PreprocessingConfig
from typing import Any


class PreprocessingStepPort(ABC):
    def __init__(self, config: PreprocessingConfig):
        self.config = config

    @abstractmethod
    def process(self, data: Any) -> Any:
        """Process an MNE raw or epochs instance and return it"""
        pass
