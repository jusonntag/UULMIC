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
        model: nn.Module, 
        criterion: nn.Module = None, 
        optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam
    ):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = model.to(self.device)
        self.criterion = criterion if criterion else nn.CrossEntropyLoss()
        self.optimizer = optimizer_cls(self.model.parameters(), lr=self.config.lr)
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # Wrap in dataloader (X is passed exactly as provided by TrialData)
        X_tensor = torch.FloatTensor(X)
            
        y_tensor = torch.LongTensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.config.epochs):
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
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
