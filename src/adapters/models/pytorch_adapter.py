import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from typing import Dict, Any, Type, Callable, Optional
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score

from src.core.ports.model import BaseModelPort
from src.core.domain.config import ModelConfig

class PyTorchModelAdapter(BaseModelPort):
    """
    A generic PyTorch Trainer adapter. 
    It can wrap ANY torch.nn.Module (EEGNet, FBCNet, etc).
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
        
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None, tracker: Optional['TrackerPort'] = None) -> None:
        # Wrap in dataloader (X is passed exactly as provided by TrialData)
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)

        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.LongTensor(y_val).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        
        print(f"\033[90mTraining model for {self.config.epochs} epochs with batch size {self.config.batch_size}\033[0m")
        for epoch in range(self.config.epochs):
            self.model.train()
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
            
            metrics_dict = {
                "train_loss": avg_loss,
                "train_acc": train_acc
            }
            
            val_loss_val = None
            val_acc_val = None
            if X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    outputs = self.model(X_val_tensor)
                    val_loss = self.criterion(outputs, y_val_tensor).item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_acc = (predicted == y_val_tensor).sum().item() / y_val_tensor.size(0)
                    
                    val_loss_val = val_loss
                    val_acc_val = val_acc
                    
                    metrics_dict["test_loss"] = val_loss
                    metrics_dict["test_acc"] = val_acc
            
            if tracker:
                tracker.log_metrics(metrics_dict)
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print_str = f"\033[90mEpoch {epoch + 1}/{self.config.epochs} - loss: {avg_loss:.4f} - acc: {train_acc:.4f}"
                if val_loss_val is not None:
                    print_str += f" - test_loss: {val_loss_val:.4f} - test_acc: {val_acc_val:.4f}"
                print_str += "\033[0m"
                print(print_str)
                
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
        precision = precision_score(y, preds, average='weighted', zero_division=0)
        recall = recall_score(y, preds, average='weighted', zero_division=0)
        f1 = f1_score(y, preds, average='weighted', zero_division=0)
        kappa = cohen_kappa_score(y, preds)

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "kappa": float(kappa)
        }
        
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
