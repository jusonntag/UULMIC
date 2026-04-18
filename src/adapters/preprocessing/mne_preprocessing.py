import mne
import numpy as np

from src.core.ports.processor import PreprocessingStepPort
from src.core.domain.config import PreprocessingConfig
from src.core.domain.trial import TrialData, TrialMetadata

class MneFilterStep(PreprocessingStepPort):
    def process(self, data: mne.io.Raw) -> mne.io.Raw:
        # Filter raw data
        filtered = data.copy()
        # Notch filter for alternating current
        if self.config.low_pass > 49:
            filtered.notch_filter(50)
            
        filtered.filter(
            l_freq=self.config.high_pass,
            h_freq=self.config.low_pass
        )
        return filtered

class MneResampleStep(PreprocessingStepPort):
    def process(self, data: mne.io.Raw) -> mne.io.Raw:
        resampled = data.copy()
        print(f'\033[90mResampling data from {data.info["sfreq"]} Hz to {self.config.target_sample_rate} Hz\033[0m')
        resampled.resample(sfreq=self.config.target_sample_rate)
        return resampled

class MneICAStep(PreprocessingStepPort):
    def process(self, data: mne.io.Raw) -> mne.io.Raw:
        if not self.config.ica:
            return data
            
        ica_data = data.copy()
        # Set channel types correctly before ICA
        try:
            ica_data.set_channel_types({ch: 'eog' for ch in self.config.eog_channels if ch in ica_data.ch_names})
        except ValueError:
            pass # Channels might already be right or missing

        ica = mne.preprocessing.ICA(n_components=15, max_iter='auto', random_state=42)
        ica.fit(ica_data)
        
        # Apply EOG & Muscle artifact rejection
        eog_idx, _ = ica.find_bads_eog(ica_data)
        muscle_idx, _ = ica.find_bads_muscle(ica_data)
        
        ica.exclude = eog_idx + muscle_idx
        ica.apply(ica_data)
        
        return ica_data

class MneReferencingStep(PreprocessingStepPort):
    def process(self, data: mne.io.Raw) -> mne.io.Raw:
        ref_data = data.copy()
        removed_channels = []

        # 1. Apply EEG Reference
        if self.config.reference_channels == ["average"]:
            print("\033[90mApplying Common Average Reference (CAR)\033[0m")
            ref_data.set_eeg_reference(ref_channels="average")
        elif self.config.reference_channels:
            print(f"\033[90mApplying EEG Reference: {self.config.reference_channels}\033[0m")
            ref_data.set_eeg_reference(ref_channels=self.config.reference_channels)
            # Track specified reference channels for removal if they exist in raw data
            for ch in self.config.reference_channels:
                if ch in ref_data.ch_names:
                    removed_channels.append(ch)
        
        # 2. Identify EOG channels for removal
        if self.config.eog_channels:
            for ch in self.config.eog_channels:
                if ch in ref_data.ch_names:
                    removed_channels.append(ch)

        # 3. Physically remove the channels
        if removed_channels:
            print(f"\033[90mRemoving channels: {removed_channels}\033[0m")
            ref_data.drop_channels(removed_channels)
            
        # Store removed channels in the info object for later retrieval by EpochingStep
        # MNE doesn't have a standard field for this, so we use 'description' or a custom attribute if we can.
        # However, passing it through the Raw object is tricky. 
        # Better: we can attach it to the Raw object as a dynamic attribute.
        ref_data._removed_channels = removed_channels
        
        return ref_data

class MneEpochingStep(PreprocessingStepPort):
    def process(self, data: mne.io.Raw) -> TrialData:
        events, event_id = mne.events_from_annotations(data)
        
        # Build MNE event selection and label mapping from config.class_map.
        # config.class_map: { raw_marker_string -> integer_class_label }
        # Strips whitespace from annotation strings to handle brittle recordings (e.g. ' 99 ' -> '99')
        clean_event_id: dict = {}
        mne_to_label: dict[int, int] = {}
        
        for event_name, mne_id in event_id.items():
            clean_name = str(event_name).strip()
            if clean_name in self.config.class_map:
                clean_event_id[event_name] = mne_id
                mne_to_label[mne_id] = self.config.class_map[clean_name]
        
        if not clean_event_id:
            raise ValueError(
                f"No matching events found after cleaning annotation strings.\n"
                f"  Config class_map keys : {list(self.config.class_map.keys())}\n"
                f"  Annotations in data   : {[str(k).strip() for k in event_id.keys()]}"
            )
            
        print(f"\033[90mEpoching events : { {str(k).strip(): v for k, v in clean_event_id.items()} }\033[0m")
        print(f"\033[90mLabel mapping   : {mne_to_label}\033[0m")
        
        # Epoch window and baseline are fully configurable
        tmin = self.config.tmin
        tmax = self.config.tmax
        baseline = (self.config.baseline_tmin, self.config.baseline_tmax) if self.config.baseline_correction else None
        
        epochs = mne.Epochs(
            data, 
            events, 
            event_id=clean_event_id, 
            tmin=tmin, 
            tmax=tmax, 
            baseline=baseline,
            preload=True
        )
        
        # Extract underlying numpy arrays: X shape (trials, channels, time)
        X = epochs.get_data()
        labels_from_mne = epochs.events[:, -1]
        
        # Remap MNE's arbitrary integer event IDs to true model integer labels
        y = np.array([mne_to_label[m_id] for m_id in labels_from_mne])
        
        # Derive human-readable class list from the class_map.
        # Group marker strings that share the same integer label.
        unique_labels = sorted(set(self.config.class_map.values()))
        label_to_markers: dict[int, list[str]] = {}
        for marker, label in self.config.class_map.items():
            label_to_markers.setdefault(label, []).append(marker)
        class_names = ["/".join(sorted(label_to_markers[lbl])) for lbl in unique_labels]
        
        # Recover removed channels from raw data if present
        removed_channels = getattr(data, "_removed_channels", [])

        metadata = TrialMetadata(
            subject_id=getattr(data, "subject_id", "unknown"),
            window_begin_sec=tmin,
            window_end_sec=tmax,
            sample_rate=int(data.info['sfreq']),
            channels=list(data.ch_names),
            eog_channels=self.config.eog_channels or [],
            reference_channels=self.config.reference_channels or [],
            high_pass_freq=self.config.high_pass,
            low_pass_freq=self.config.low_pass,
            notch_filtered_50hz=(self.config.low_pass > 49),
            ica_applied=self.config.ica,
            baseline_corrected=self.config.baseline_correction,
            filterbank_stepsize=self.config.filterbank_stepsize,
            removed_channels=removed_channels,
            classes=class_names,
            class_map=self.config.class_map,
        )
        
        return TrialData(X=X, y=y, metadata=metadata)
