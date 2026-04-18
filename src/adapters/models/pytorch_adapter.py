import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from typing import Dict, Any, Type, Callable

from src.core.ports.model import BaseModelPort
from src.core.domain.config import ModelConfig

class PyTorchModelAdapter(BaseModelPort):
    """
    A generic PyTorch Trainer adapter. 
    It can wrap ANY torch.nn.Module (EEGNet, FBCNet, etc) so you don't need
    to write a new adapter for every architectural change.
    """
    def __init__(
        self, 
        config: ModelConfig, 
        model: nn.Module
    ):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = model.to(self.device)
        self.criterion = self._get_criterion(self.config.loss)
        self.optimizer_cls = self._get_optimizer(self.config.optimizer)
        
        # We need weight_decay for L2 regularization
        self.optimizer = self.optimizer_cls(
            self.model.parameters(), 
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )

    def _get_criterion(self, loss_name: str) -> nn.Module:
        loss_map = {
            "cross_entropy": nn.CrossEntropyLoss(),
            "nll": nn.NLLLoss(),
            "mse": nn.MSELoss()
        }
        if loss_name not in loss_map:
            raise ValueError(f"Loss {loss_name} not supported.")
        return loss_map[loss_name]

    def _get_optimizer(self, opt_name: str) -> Type[torch.optim.Optimizer]:
        opt_map = {
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
            "adamw": torch.optim.AdamW
        }
        if opt_name not in opt_map:
            raise ValueError(f"Optimizer {opt_name} not supported.")
        return opt_map[opt_name]
        
    def fit(self, X: np.ndarray, y: np.ndarray, tracker: Optional['TrackerPort'] = None) -> None:
        # Wrap in dataloader (X is passed exactly as provided by TrialData)
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        
        print(f"\033[90mTraining model for {self.config.epochs} epochs with batch size {self.config.batch_size}\033[0m")
        self.model.train()
        for epoch in range(self.config.epochs):
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
                # Accuracy calculation
                _, predicted = torch.max(outputs.data, 1)
                total_samples += batch_y.size(0)
                total_correct += (predicted == batch_y).sum().item()
            
            avg_loss = total_loss / len(loader)
            train_acc = total_correct / total_samples
            
            if tracker:
                tracker.log_metrics({
                    "train_loss": avg_loss,
                    "train_acc": train_acc
                }, step=epoch)
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"\033[90mEpoch {epoch + 1}/{self.config.epochs} - loss: {avg_loss:.4f} - acc: {train_acc:.4f}\033[0m")
                
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
        return predicted.cpu().numpy()
        
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        preds = self.predict(X)
        accuracy = (preds == y).mean()
        # You can expand this to include F1, kappa, etc.
        return {"accuracy": float(accuracy)}
        
    def get_params(self) -> Dict[str, Any]:
        # Log training config and total model parameters count
        params = self.config.model_dump()
        params['total_parameters'] = sum(p.numel() for p in self.model.parameters())
        params['model_class'] = self.model.__class__.__name__
        return params

    def reset(self) -> None:
        """Reset model weights and optimizer to a fresh state"""
        def weight_reset(m):
            # Check for reset_parameters attribute (common in most torch layers)
            reset_parameters = getattr(m, "reset_parameters", None)
            if callable(reset_parameters):
                m.reset_parameters()
        
        for layer in self.model.modules():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        # self.model.apply(weight_reset)

        # Re-initialize optimizer to clear momentum/state
        self.optimizer = self.optimizer_cls(
            self.model.parameters(), 
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )

    def save(self, path: str) -> None:
        """Save the model state dictionary to the specified path"""
        torch.save(self.model.state_dict(), path)
