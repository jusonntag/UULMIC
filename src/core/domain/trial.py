from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional, Dict
import numpy as np


class TrialMetadata(BaseModel):
    """
    The strict lineage and metadata of how a specific EEG Trial was generated.
    This guarantees that we always know the exact preprocessing history of our data.
    """
    model_config = ConfigDict(extra = 'forbid') # Prevent accidental typos in fields
    
    subject_id: str = Field(..., description = "ID of the subject (e.g., 'VP110')")

    # Time properties
    window_begin_sec: float = Field(..., description = "Start of window relative to event (e.g., -0.5)")
    window_end_sec: float = Field(..., description = "End of window relative to event (e.g., 4.0)")
    sample_rate: int = Field(..., description = "Target sampling rate in Hz")
    
    # Channel properties
    channels: List[str] = Field(description="Names of all EEG channels present in the data")
    eog_channels: List[str] = Field(description="Names of all EOG channels present in the data")
    reference_channels: List[str] = Field(description="Channels used for referencing")
    
    # Preprocessing Lineage
    high_pass_freq: float = Field(description = "High-pass filter cutoff frequency")
    low_pass_freq: float = Field(description = "Low-pass filter cutoff frequency")
    notch_filtered_50hz: bool = Field(False, description = "Was a 50Hz notch filter applied?")
    ica_applied: bool = Field(False, description = "Was ICA artifact rejection applied?")
    baseline_corrected: bool = Field(False, description = "Was baseline correction applied?")
    filterbank_stepsize: Optional[float] = Field(None, description = "Step size if filterbanks were used")
    removed_channels: List[str] = Field(default_factory=list, description = "List of channels removed during preprocessing (e.g. reference or EOG)")

    # Dataset facts
    classes: List[str] = Field(description = "The string labels of the target classes (e.g., ['left', 'right'])")
    class_map: Dict[str, int] = Field(description = "Mapping from class labels to integers")


class TrialData(BaseModel):
    """
    The fundamental Core Entity for a training/evaluation dataset in UULMIC.
    Contains the raw numpy voltages and the strict chronological metadata.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True) # Required to allow raw numpy arrays

    X: np.ndarray = Field(description = "Shape: (n_trials, n_channels, n_times) or (n_trials, n_channels, n_times, n_filterbanks)")
    y: np.ndarray = Field(description = "Shape: (n_trials,)")

    metadata: TrialMetadata
