from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import List, Optional, Dict, Any


class ModelConfig(BaseModel):
    """Configuration definition for deep learning models"""
    model_config = ConfigDict(extra = 'allow')
    name: str
    epochs: int = 50
    lr: float = 0.001
    batch_size: int = 32


class EEGNetConfig(ModelConfig):
    """Specialized config for the EEGNet architecture"""
    F1: int = 8
    D: int = 2
    F2: Optional[int] = None # Calculated automatically
    kernLength: int = 64
    dropoutRate: float = 0.5
    dropoutType: str = "spatial"
    n_classes: int = 4
    channels: int = 61
    samples: int = 538
    bias: bool = False

    @model_validator(mode='after')
    def compute_f2(self) -> 'EEGNetConfig':
        if self.F2 is None:
            self.F2 = self.F1 * self.D
        return self


class PreprocessingConfig(BaseModel):
    """Basic config dictating how the pipeline should run"""
    model_config = ConfigDict(extra = 'allow')
    high_pass: float = 4.0
    low_pass: float = 40.0
    filterbank_stepsize: Optional[float] = None
    target_sample_rate: int = 128
    ica: bool = True

    # Epoch window
    tmin: float = -0.2
    tmax: float = 4.0

    # Baseline correction window (only used when baseline_correction=True)
    baseline_correction: bool = True
    baseline_tmin: Optional[float] = None   # None = start of epoch
    baseline_tmax: float = 0.0

    # Marker-to-label mapping: maps raw EEG annotation string -> integer class label
    # Supports N classes. Example: {'11': 1, '12': 2, '13': 3, '98': 0, '99': 0}
    class_map: Dict[str, int] = {'11': 1, '12': 2, '13': 3, '98': 0, '99': 0}

    reference_channels: Optional[List[str]] = None
    eog_channels: Optional[List[str]] = None
    classes: List[str] = ['11', '12', '13']
    resting_classes: List[str] = ['98', '99']
    splits: Optional[Dict[str, List[str]]] = None
