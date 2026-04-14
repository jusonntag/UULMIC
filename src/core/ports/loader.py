from abc import ABC, abstractmethod
from src.core.domain.trial import TrialData
from pathlib import Path
from typing import Any


class DataLoaderPort(ABC):
    @abstractmethod
    def load_training_data(self, data_dir: Path) -> TrialData:
        """Load and return preprocessed training records"""
        pass
        
    @abstractmethod
    def load_raw_subject(self, vp_id: str) -> Any:
        """Load raw subject data (e.g. MNE Raw instance)"""
        pass
