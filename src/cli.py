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
        tracker = WandbTrackerAdapter(project_name="UULMI_Refactored", run_name=f"run_{cfg.model.name}")
        
        # Instantiate model based on config
        if cfg.model.name == "eegnet":
            from src.adapters.models.pytorch.architectures.eegnet import EEGNet
            from src.adapters.models.pytorch_adapter import PyTorchModelAdapter
            
            # Use specific Pydantic model for validation and derived fields (F2)
            model_config = EEGNetConfig(**cfg.model)
            model = PyTorchModelAdapter(config=model_config, model=EEGNet(config=model_config))
        else:
            raise ValueError(f"Model {cfg.model.name} not implemented yet.")
            
        data_dir = Path(cfg.data.preprocessed_path) / "Epochs" if not hasattr(cfg.data, "epoched_path") else Path(cfg.data.epoched_path) 
        
        # Run use case
        run_training_usecase(
            model=model,
            data_loader=loader,
            tracker=tracker,
            data_dir=data_dir
        )
        
    elif cfg.mode == "preprocess":
        from src.use_cases.preprocess import run_preprocessing_usecase
        from src.adapters.preprocessing.mne_preprocessing import (
            MneFilterStep, 
            MneICAStep, 
            MneReferencingStep, 
            MneEpochingStep
        )
        from src.core.domain.config import PreprocessingConfig
        
        prep_config = PreprocessingConfig(**cfg.preprocessing)
        steps = [
            MneFilterStep(config=prep_config)
        ]
        if prep_config.ica:
            steps.append(MneICAStep(config=prep_config))
            
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
