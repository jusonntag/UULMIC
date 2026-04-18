from src.core.ports.model import BaseModelPort
from src.core.ports.loader import DataLoaderPort
from src.core.ports.tracker import TrackerPort
from pathlib import Path

from typing import List
from src.core.domain.trial import TrialData
import numpy as np

from src.core.domain.config import PreprocessingConfig
from sklearn.model_selection import train_test_split

def run_training_usecase(
    model: BaseModelPort, 
    tracker: TrackerPort,
    all_trials: List[TrialData],
    model_save_dir: Path,
    run_id: str,
    prep_config: PreprocessingConfig
):
    try:
        if not all_trials:
            print("No training data provided.")
            return

        mode = model.config.training_mode
        print(f"Executing training mode: {mode}")
        
        # Ensure save directory exists
        run_save_dir = model_save_dir / run_id
        run_save_dir.mkdir(parents=True, exist_ok=True)

        if mode in ["independent", "sequential"]:
            # Train one model for each individual subject/dataset
            for trial_data in all_trials:
                subject_id = trial_data.metadata.subject_id
                print(f"\n--- Training on Subject: {subject_id} ---")
                
                # 2. Reset model weights if in independent mode
                if mode == "independent":
                    print(f"Resetting model for {subject_id}...")
                    model.reset()
                
                # 3. Track Parameters
                tracker.log_params(model.get_params())
                
                # 4. Split Data
                test_size = prep_config.test_size
                seed = prep_config.split_seed
                print(f"Splitting data with test_size={test_size}...")
                
                X_train, X_test, y_train, y_test = train_test_split(
                    trial_data.X, 
                    trial_data.y, 
                    test_size=test_size, 
                    random_state=seed,
                    stratify=trial_data.y if len(np.unique(trial_data.y)) > 1 else None
                )
                print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

                # 5. Train
                model.fit(X_train, y_train, tracker=tracker)
                
                # 6. Evaluate
                metrics = model.evaluate(X_test, y_test)
                
                # Log prefixed metrics (e.g. VP110_accuracy) for summary tracking
                # Distinguish between train and test accuracy if needed, 
                # but 'accuracy' from evaluate is typically the test accuracy.
                prefixed_metrics = {f"{subject_id}_test_{k}": v for k, v in metrics.items()}
                tracker.log_metrics(prefixed_metrics)
                
                # 7. Save Model
                save_path = run_save_dir / f"{subject_id}_{model.config.name}.pt"
                model.save(str(save_path))
                
                print(f"Finished {subject_id} with metrics: {metrics}")
                print(f"Model saved to {save_path}")

        elif mode == "combined":
            # Switch for combined dataset training (to be implemented)
            print("INFO: Combined training mode selected. Merging logic not yet implemented.")
            # TODO: Implement data concatenation and single training run
            pass

    finally:
        tracker.finish()
