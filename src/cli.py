import argparse

# Monkey-patch argparse to fix Hydra incompatibility with Python 3.14+
_original_add_argument = argparse.ArgumentParser.add_argument
def _patched_add_argument(self, *args, **kwargs):
    if 'help' in kwargs and kwargs['help'].__class__.__name__ == 'LazyCompletionHelp':
        kwargs['help'] = repr(kwargs['help'])
    return _original_add_argument(self, *args, **kwargs)
argparse.ArgumentParser.add_argument = _patched_add_argument

import hydra
from omegaconf import DictConfig
from pathlib import Path
from dotenv import load_dotenv


@hydra.main(version_base="1.3", config_path="../configs", config_name="default")
def main(cfg: DictConfig):
    from src.adapters.data.mne_adapter import MneDataLoaderAdapter
    from src.adapters.tracking.wandb_adapter import WandbTrackerAdapter
    from src.core.domain.config import ModelConfig, EEGNetConfig
    from src.use_cases.train import run_training_usecase
    from src.utils.concurrency import ConcurrencyManager

    load_dotenv()

    print(f"--- Running UULMI CLI ---")
    print(f"Mode: {cfg.mode}")
    
    # 1. Setup automatic safe concurrency & environment variables
    concurrency_settings = ConcurrencyManager.setup_optimal_workers(estimated_gb_per_task=1.5)
    max_workers = concurrency_settings['optimal_workers']
    
    loader = MneDataLoaderAdapter(data_dir=cfg.data.raw_path)
    
    if cfg.mode == "train":
        # 1. First, resolve path and load data to detect shape
        data_dir = Path(cfg.data.epoched_path)
        print(f"Loading data from {data_dir} to detect shape...")
        all_trials = loader.load_training_data(data_dir = data_dir)
        
        if not all_trials:
             raise FileNotFoundError(f"No training data found in {data_dir}")
        
        # 2. Extract shape
        detected_channels = all_trials[0].X.shape[1]
        detected_samples = all_trials[0].X.shape[2]
        print(f"Dynamic Shape Detection: {detected_channels} channels, {detected_samples} samples")
        
        # Determine models to train
        models_to_train = cfg.get("models_to_train", [cfg.model.name])
        if "all" in models_to_train:
            models_to_train = [p.stem for p in Path("configs/model").glob("*.yaml")]
            
        print(f"Models scheduled for training: {models_to_train}")
        
        from omegaconf import OmegaConf
        import datetime
        run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for model_name in models_to_train:
            print(f"\n{'='*40}\nInitializing Model Pipeline: {model_name}\n{'='*40}")
            
            # Dynamically load the model config
            model_cfg_path = Path(f"configs/model/{model_name}.yaml")
            if not model_cfg_path.exists():
                print(f"Warning: Config for {model_name} not found at {model_cfg_path}. Skipping.")
                continue
                
            model_cfg = OmegaConf.load(model_cfg_path)
            # Inject dynamic shape
            model_cfg.channels = detected_channels
            model_cfg.samples = detected_samples

            # Setup Tracker for this model run
            tracker = WandbTrackerAdapter(project_name="UULMI", run_name=f"run_{model_name}_{run_timestamp}")

            # 3. Instantiate model based on config
            if model_name == "eegnet":
                from src.adapters.models.pytorch.architectures.eegnet import EEGNet
                from src.adapters.models.pytorch_adapter import PyTorchModelAdapter
                
                # Convert DictConfig to dict for Pydantic validation
                model_cfg_dict = OmegaConf.to_container(model_cfg, resolve=True)
                model_config = EEGNetConfig(**model_cfg_dict)
                model = PyTorchModelAdapter(config=model_config, model=EEGNet(config=model_config))
            else:
                print(f"Model architecture '{model_name}' not fully implemented in CLI yet. Skipping.")
                continue
                
            model_save_dir = Path(cfg.get("model_save_dir", "data/trained_models"))
            from src.core.domain.config import PreprocessingConfig
            prep_config = PreprocessingConfig(**cfg.preprocessing)
                
            # 4. Run use case with pre-loaded trials
            run_training_usecase(
                model=model,
                tracker=tracker,
                all_trials=all_trials,
                model_save_dir=model_save_dir,
                run_id=f"{run_timestamp}_{model_name}",
                prep_config=prep_config
            )
        
    elif cfg.mode == "preprocess":
        from src.use_cases.preprocess import run_preprocessing_usecase
        from src.adapters.preprocessing.mne_preprocessing import (
            MneFilterStep, 
            MneICAStep, 
            MneReferencingStep, 
            MneEpochingStep,
            MneResampleStep
        )
        from src.core.domain.config import PreprocessingConfig
        
        prep_config = PreprocessingConfig(**cfg.preprocessing)
        steps = [
            MneFilterStep(config=prep_config)
        ]
        if prep_config.ica:
            steps.append(MneICAStep(config=prep_config))
        
        if prep_config.target_sample_rate:
            steps.append(MneResampleStep(config=prep_config))

        # Add Referencing Step (handles CAR or specific channels + removal)
        steps.append(MneReferencingStep(config=prep_config))
            
        # Append Epoching Extraction as final preprocessing step
        steps.append(MneEpochingStep(config=prep_config))
            
        run_preprocessing_usecase(
            vp_ids=cfg.data.vps,
            loader=loader,
            pipeline_steps=steps,
            output_dir=Path(cfg.data.preprocessed_path),
            max_workers=max_workers
        )

if __name__ == "__main__":
    main()
