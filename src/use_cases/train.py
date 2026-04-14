from src.core.ports.model import BaseModelPort
from src.core.ports.loader import DataLoaderPort
from src.core.ports.tracker import TrackerPort
from pathlib import Path

def run_training_usecase(
    model: BaseModelPort, 
    data_loader: DataLoaderPort, 
    tracker: TrackerPort,
    data_dir: Path
):
    try:
        # 1. Load preprocessed data
        print(f"Loading data from {data_dir}...")
        trial_data = data_loader.load_training_data(data_dir=data_dir)
        
        # 2. Track Parameters
        tracker.log_params(model.get_params())
        
        # 3. Train
        print("Starting training...")
        model.fit(trial_data.X, trial_data.y)
        
        # 4. Evaluate
        print("Evaluating model...")
        metrics = model.evaluate(trial_data.X, trial_data.y)
        tracker.log_metrics(metrics)
        print(f"Finished with metrics: {metrics}")
        
    finally:
        tracker.finish()
