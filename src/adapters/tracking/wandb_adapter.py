import wandb
from typing import Any, Dict

from src.core.ports.tracker import TrackerPort

class WandbTrackerAdapter(TrackerPort):
    def __init__(self, project_name: str, group_name: str | None = None, run_name: str | None = None):
        self.run = wandb.init(
            project=project_name,
            group=group_name,
            name=run_name,
            reinit=True
        )
        
    def log_params(self, params: Dict[str, Any]) -> None:
        wandb.config.update(params, allow_val_change=True)
        
    def log_metrics(self, metrics: Dict[str, float], step: int | None = None) -> None:
        wandb.log(metrics, step=step)
        
    def finish(self) -> None:
        wandb.finish()
