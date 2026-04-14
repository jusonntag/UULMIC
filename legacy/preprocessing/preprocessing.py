import mne
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

class Preprocessor:
    def __init__(
            self,
            data_dir: Path | str = Path().cwd(),
            high_pass: float = 4.0,
            low_pass: float = 40.0,
            freq_step_size: bool | float = False,
            window_begin: float = -0.5,
            window_end: float = 4.0,
            events_of_interest: List[str] = ['11', '12', '13'],#, '21', '22', '23'],
            target_sample_rate: int = 128,
            reference_channel: List[str] = ['M1', 'M2'],
            eog_channel: List[str] = ['VEOG', 'HEOG'],
            model_name: str = 'EEGNet',
            save_resting: bool = True,
            resting_state: bool | List[str] = ['eyes_open', 'eyes_closed'],
            resting_window: Tuple[float, float] = (2.5, 182.5),
            event_keys: List[str] = ['98', '99'],
            ica: bool = True,
            ) -> None:
        
        # Initialize data_dir attribute
        if isinstance(data_dir, Path):
            self.data_dir = data_dir
        elif isinstance(data_dir, str):
            self.data_dir = Path(data_dir)
        else:
            raise ValueError('Path must be either type string or Path!')
        
        # TODO try to create data dir?
        
        # initialize & create save paths
        self.save_dir = self.data_dir.parent / 'preprocessed'
        self.resting_dir = self.data_dir.parent / 'resting'
        self.save_dir.mkdir(exist_ok =  True)
        self.resting_dir.mkdir(exist_ok = True)

        # initialize instance attributes
        self.high_pass = high_pass
        self.low_pass = low_pass
        self.freq_step_size = freq_step_size
        self.window_begin = window_begin
        self.window_end = window_end
        self.events_of_interest = events_of_interest
        self.target_sample_rate = target_sample_rate
        self.reference_channel = reference_channel
        self.eog_channel = eog_channel
        self.model_name = model_name
        self.save_resing = save_resting
        self.resting_state = resting_state
        self.resting_window = resting_window
        self.event_keys = event_keys
        self.ica = ica

        # load paths of data for later preprocessing
        self.raw_files = self._get_data()
        print(f'Loaded {len(self.raw_files)} files from directory.')

        self._preprocess()

    def _get_data(self):
        raw_files = []
        for vp in self.data_dir.iterdir():
            if vp.is_dir():
                for file in vp.iterdir():
                    if file.parts[-1].endswith('.set'):
                        raw_files.append(self.data_dir / vp / file)
        return raw_files

    def _load_data(self, path: Path | str) -> mne.io.Raw:
        if path.parts[-1].endswith('.set'):
            raw = mne.io.read_raw_eeglab(path, preload = True)
        elif path.parts[-1].endswith('.fif'):
            raw = mne.oi.read_raw_fif(path, preload = True)
        else:
            raise ValueError(
                'Error while trying to load file path. Must be .set or .fif'
                )
        return raw
    
    def _extract_resting_state(
            self,
            raw: mne.io.Raw,
            events_from_annot: Dict[str, int],
            event_dict: Dict[str, int],
            event_keys: List[int],
            window: Tuple[float, float] = (2.5, 182.5),
            ) -> mne.Epochs:

        rest_dict = {}
        for key, value in event_dict.items():
            key = key.replace(' ', '') # strips weird fromatting
            if key in event_keys:
                rest_dict[key] = value

        resting_epochs = mne.Epochs(
            raw = raw,
            events = events_from_annot, 
            event_id = rest_dict,
            event_repeated = 'drop',
            tmin = window[0],
            tmax = window[1],
            preload = True,
            baseline = None,
            )
        return resting_epochs
    
    def _preprocess_resting(
            self,
            raw_eeg: mne.io.Raw,
            VP: str,
            events_from_annot: Dict[str, int],
            event_dict: Dict[str, int],
            ):
        resting_eeg = self._extract_resting_state(
                    raw = raw_eeg,
                    events_from_annot = events_from_annot,
                    event_dict = event_dict,
                    event_keys = self.event_keys,
                    window = self.resting_window,    
                    )

        for state in self.resting_state:
            Path(f'{self.resting_dir}/{state}').mkdir(exist_ok = True)
            resting_dir = self.resting_dir / state / f'{VP}_resting-epo.fif'
            resting_eeg.save(resting_dir, overwrite = True)
    
    def _apply_ica(self, raw: mne.io.Raw) -> mne.io.Raw:
        raw_ica = raw.copy()
        eog_evoked = mne.preprocessing.create_eog_epochs(raw_ica)
        eog_evoked.apply_baseline(baseline = (None, -0.2))
        ica = mne.preprocessing.ICA(
            n_components = 15,
            max_iter = 'auto',
            random_state = 42
            )
        ica.fit(raw_ica)
        ica.exclude = []
        eog_idx, eog_scores = ica.find_bads_eog(raw_ica)
        muscle_idx, muscle_scores = ica.find_bads_muscle(raw_ica)
        ica.exclude = eog_idx + muscle_idx
        ica.apply(raw_ica)
        raw_ica.set_eeg_reference(ref_channels =  self.reference_channel)
        raw_ica.drop_channels(ch_names = self.eog_channel)
        return raw_ica
  
    def _to_numpy(self, epochs: mne.Epochs, class_id) -> Tuple[np.ndarray, List[int]]:
        epochs_array = epochs[class_id].get_data()
        label = [class_id for _ in epochs_array]
        return epochs_array, label
    
    def _preprocess(self):
        print(f'\033[95m Starting preprocessing raw data for model: {self.model_name} \033[0m')
        for file in self.raw_files:
            raw_eeg = self._load_data(path = file)

            VP = [part for part in file.parts][-1].split('_')[0] # TODO make this less error prone
            
            # apply notch filter to eliminate flickering of light
            if self.low_pass > 49:
                raw_eeg = raw_eeg.notch_filter(50)

            # set VEOG & HEOG as type 'eog'
            raw_eeg = raw_eeg.set_channel_types({'VEOG': 'eog','HEOG': 'eog'})

            # resample to wanted samplefrequency
            raw_eeg = raw_eeg.resample(sfreq = self.target_sample_rate)

            # extract evnts from annotation
            events_from_annot, event_dict = mne.events_from_annotations(raw_eeg)

            # creates & cleans an event_dict
            dct = {}
            for key, value in event_dict.items():
                key = key.replace(' ', '') # strips weird formatting
                dct[key] = value
            event_dict = dct 

            # extract and preprocess resting states
            if self.save_resing and self.resting_state:
                self._preprocess_resting(
                    raw_eeg = raw_eeg,
                    VP = VP,
                    events_from_annot = events_from_annot,
                    event_dict = event_dict,
                )
            else:
                print('[INFO] No preprocessing and saving of resting state(s)!')

            epochs = []

            if self.freq_step_size:
                start = self.high_pass 
                for i in np.arange(
                    self.high_pass,         # start
                    self.low_pass,          # stop
                    self.freq_step_size     # step
                    ):
                    filtered_eeg = raw_eeg.filter(start, self.high_pass + i)
                    if self.ica:
                        filtered_eeg = self._apply_ica(filtered_eeg)
                    epochs.append(
                        mne.Epochs(
                        raw = filtered_eeg,
                        events = events_from_annot,
                        event_id = event_dict,
                        tmin = self.window_begin,
                        tmax = self.window_end,
                        event_repeated = 'drop',
                        preload = True,
                        baseline = None,
                        )
                    )
                    start = self.high_pass + i
            else:
                raw_eeg.filter(self.high_pass, self.low_pass)
                if self.ica:
                    raw_eeg = self._apply_ica(raw_eeg)
                epochs.append(
                    mne.Epochs(
                    raw = raw_eeg,
                    events = events_from_annot,
                    event_id = event_dict,
                    tmin = self.window_begin,
                    tmax = self.window_end,
                    event_repeated = 'drop',
                    preload = True,
                    baseline = None,
                    )
                )
            ####
            # TODO transform epochs to np.arrays as func
            # extract numpy arrays from epochs per class
            all_x_per_file = []
            all_y_per_file = []

            for event in self.events_of_interest:
                # cllect all frequency bins for event
                event_bins = []
                for e in epochs:
                    # x shape: (n_epochs, n_channels, n_times)
                    x, y = self._to_numpy(epochs=e, class_id=event)
                    event_bins.append(x)

                if self.freq_step_size:
                    # stack bins on new axis: (n_epochs, n_channels, n_times, n_bins)
                    event_x = np.stack(event_bins, axis=-1) 
                else:
                    # just (n_epochs, n_channels, n_times)
                    event_x = event_bins[0]

                all_x_per_file.append(event_x)
                all_y_per_file.extend(y)

            X = np.concat(all_x_per_file, axis=0)
            Y = np.array(all_y_per_file)
            #Y = [item.strip() for item in Y]
            ####
            print('---')
            print(X.shape)
            print(Y.shape)
            print('---')
            ####
            # TODO name creation and saving of np.array func
            # create file name based on applied preprocessing steps
            filter = f'{self.high_pass}_{self.low_pass}'
            if self.freq_step_size:
                filter = f'{filter}_Step{self.freq_step_size}'
            
            window = f'{abs(self.window_begin)+self.window_end}'
            
            sr = f'{self.target_sample_rate}'
            
            if self.ica:
                ica = f'ICA'
            else:
                ica = 'NO_ICA'

            file_name = f'{VP}_{self.model_name}_X_{filter}_{window}_{ica}_{sr}.npy'
            file_name_y = file_name.replace('X', 'Y')
            
            save_dir = self.save_dir / self.model_name / VP
            save_dir.mkdir(parents = True, exist_ok = True)

            np.save(save_dir / file_name, X)
            np.save(save_dir / file_name_y, Y)
            ####