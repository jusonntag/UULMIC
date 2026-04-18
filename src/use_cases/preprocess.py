from typing import List
from pathlib import Path
import os
import concurrent.futures
import numpy as np
from src.core.ports.loader import DataLoaderPort
from src.core.ports.processor import PreprocessingStepPort
from src.core.domain.trial import TrialData
from src.core.domain.config import PreprocessingConfig

def _save_processed_data(processed_data: Any, vp_id: str, vp_out: Path):
    """Internal helper to save data based on its type."""
    vp_out.mkdir(parents=True, exist_ok=True)
    
    if hasattr(processed_data, "save"):
        # Output is still MNE Raw/Epochs
        out_file = vp_out / f"{vp_id}_preprocessed.fif"
        processed_data.save(out_file, overwrite=True)
    elif isinstance(processed_data, TrialData):
        # Output has been extracted into Machine Learning tensors
        x_path = vp_out / f"{vp_id}_X.npy"
        y_path = vp_out / f"{vp_id}_Y.npy"
        json_path = vp_out / f"{vp_id}_metadata.json"
        
        np.save(str(x_path), processed_data.X)
        np.save(str(y_path), processed_data.y)
        
        with open(json_path, 'w') as f:
            f.write(processed_data.metadata.model_dump_json(indent=4))
    else:
        raise TypeError(f"Unknown pipeline output type: {type(processed_data)}")

def preprocess_single_vp(vp_id: str, loader: DataLoaderPort, steps: List[PreprocessingStepPort], output_dir: Path):
    """Preprocess a single subject and save result, potentially splitting into multiple datasets."""
    try:
        raw_data = loader.load_raw_subject(vp_id)
        # Attach subject_id for downstream metadata creation
        raw_data.subject_id = vp_id
        
        # 1. Identify "Base" steps vs "Final" steps.
        # Everything before the last step (assumed to be Epoching) is Base.
        if not steps:
            return
            
        base_steps = steps[:-1]
        final_step = steps[-1]
        config: PreprocessingConfig = final_step.config
        
        # 2. Run Base Pipeline
        processed_data = raw_data
        for step in base_steps:
            print(f"[\033[94m{vp_id}\033[0m] Processing with \033[96m{step.__class__.__name__}\033[0m...")
            processed_data = step.process(processed_data)
        
        # 3. Handle Splitting or Single Run
        if config.splits:
            print(f"[\033[94m{vp_id}\033[0m] \033[93mSplitting data into\033[0m: \033[95m{list(config.splits.keys())}\033[0m")
            for split_name, markers in config.splits.items():
                print(f"[\033[94m{vp_id}\033[0m] \033[92mExecuting split\033[0m: \033[95m{split_name}\033[0m")
                # Create a temporary config override for this split
                # We filter the class_map to only include markers for this split
                split_class_map = {m: label for m, label in config.class_map.items() if m in markers}
                
                if not split_class_map:
                    print(f"[{vp_id}] Warning: Split '{split_name}' has no matching markers in class_map. Skipping.")
                    continue
                
                # Clone final step with temporary split config
                # Note: We create a shallow copy of config to avoid polluting other threads
                temp_config = config.model_copy()
                temp_config.class_map = split_class_map
                
                # Re-instantiate the final step (Epoching) with the split config
                # We use the same class as the final_step
                split_step = final_step.__class__(config=temp_config)
                
                try:
                    split_result = split_step.process(processed_data)
                    save_path = output_dir / split_name / vp_id
                    _save_processed_data(split_result, vp_id, save_path)
                except Exception as split_err:
                    print(f"[\033[94m{vp_id}\033[0m] \033[91mError in split '{split_name}': {split_err}\033[0m")
        else:
            # Legacy behavior: just run the final step once
            processed_data = final_step.process(processed_data)
            _save_processed_data(processed_data, vp_id, output_dir / vp_id)
            
        print(f"[\033[94m{vp_id}\033[0m] \033[92mFinished preprocessing.\033[0m")
        
    except Exception as e:
        print(f"[\033[94m{vp_id}\033[0m] \033[91mError: {e}\033[0m")
        import traceback
        traceback.print_exc()

def run_preprocessing_usecase(
    vp_ids: List[str], 
    loader: DataLoaderPort, 
    pipeline_steps: List[PreprocessingStepPort],
    output_dir: Path,
    max_workers: int | None = None
):
    # Free-threaded python will truly parallelize this without GIL
    if max_workers is None:
        # Default safety logic: limit to CPU cores or max 4 to avoid OOM
        max_workers = min(os.cpu_count() or 4, 4)
        
    print(f"Running preprocessing on {len(vp_ids)} participants with {max_workers} threads...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(preprocess_single_vp, vp, loader, pipeline_steps, output_dir)
            for vp in vp_ids
        ]
        concurrent.futures.wait(futures)
