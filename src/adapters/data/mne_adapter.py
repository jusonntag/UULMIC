import mne
from pathlib import Path
import numpy as np
import json

from src.core.ports.loader import DataLoaderPort
from src.core.domain.trial import TrialData, TrialMetadata


class MneDataLoaderAdapter(DataLoaderPort):
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        
    def load_training_data(self, data_dir: Path) -> TrialData:
        # Load X and Y numpy arrays
        vps = list(data_dir.glob("VP*"))

        total_data = []

        for vp_path in vps:
            x_path = list(vp_path.glob("*_X.npy"))[0]
            y_path = list(vp_path.glob("*_Y.npy"))[0]
        
            # Load exact TrialMetadata from JSON
            json_paths = list(vp_path.glob("*.json"))
            if not json_paths:
                raise FileNotFoundError(f"Metadata JSON missing in {data_dir}. Cannot reconstruct TrialData safely.")
            
            with open(json_paths[0], 'r') as f:
                metadata_dict = json.load(f)
            
            # Inject subject_id from directory name
            metadata_dict['subject_id'] = vp_path.name
            metadata = TrialMetadata(**metadata_dict)
        
            X = np.load(x_path)
            y = np.load(y_path)
        
            total_data.append(TrialData(X = X, y = y, metadata = metadata))
        
        return total_data
        
    def load_raw_subject(self, vp_id: str) -> mne.io.Raw:
        # Load the Raw .set file from data/raw/{vp_id}
        vp_dir = self.data_dir #/ vp_id
        for file in vp_dir.iterdir():
            if file.name.endswith('.set'):
                return mne.io.read_raw_eeglab(file, preload=True)
            elif file.name.endswith('.fif'):
                return mne.io.read_raw_fif(file, preload=True)
        raise FileNotFoundError(f"No raw data found for {vp_id} in {vp_dir}")
