"""PyTorch MLP model for AeroSurrogate."""

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class MLPNetwork(nn.Module):
    """Multi-layer perceptron regression network."""

    def __init__(self, input_dim: int, hidden_layers: List[int], dropout: float = 0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class MLPModel(BaseModel):
    """PyTorch MLP regression model for single-target prediction."""

    def __init__(self, target_name: str, input_dim: int = 20, params: Optional[Dict] = None):
        super().__init__(name=f"mlp_{target_name}", params=params)
        self.target_name = target_name
        self.input_dim = input_dim

        default_params = {
            "hidden_layers": [128, 64, 32],
            "dropout": 0.1,
            "learning_rate": 0.001,
            "epochs": 200,
            "batch_size": 64,
            "patience": 20,
            "random_state": 42,
        }
        if params:
            default_params.update(params)
        self.params = default_params

        self.device = torch.device("cpu")
        self.network: Optional[MLPNetwork] = None
        self._build_network()

    def _build_network(self) -> None:
        """Initialize the neural network."""
        torch.manual_seed(self.params["random_state"])
        self.network = MLPNetwork(
            input_dim=self.input_dim,
            hidden_layers=self.params["hidden_layers"],
            dropout=self.params["dropout"],
        ).to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> None:
        """Train MLP with early stopping."""
        self.input_dim = X.shape[1]
        self._build_network()

        X_t = torch.FloatTensor(X).to(self.device)
        y_t = torch.FloatTensor(y).to(self.device)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.params["batch_size"], shuffle=True)

        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.params["learning_rate"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        criterion = nn.MSELoss()

        # Validation data
        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val_t = torch.FloatTensor(X_val).to(self.device)
            y_val_t = torch.FloatTensor(y_val).to(self.device)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        self.network.train()
        for epoch in range(self.params["epochs"]):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                pred = self.network(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(X_batch)
            epoch_loss /= len(X_t)

            # Validation
            if has_val:
                self.network.eval()
                with torch.no_grad():
                    val_pred = self.network(X_val_t)
                    val_loss = criterion(val_pred, y_val_t).item()
                self.network.train()
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.network.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.params["patience"]:
                        logger.info(f"Early stopping at epoch {epoch + 1}, best val_loss={best_val_loss:.6f}")
                        break
            else:
                scheduler.step(epoch_loss)

        if best_state is not None:
            self.network.load_state_dict(best_state)

        self.is_fitted = True
        logger.info(f"MLP model '{self.name}' trained for {epoch + 1} epochs")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using trained MLP."""
        if not self.is_fitted or self.network is None:
            raise RuntimeError(f"Model '{self.name}' is not fitted.")

        self.network.eval()
        X_t = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            preds = self.network(X_t).cpu().numpy()
        return preds

    def save(self, path: str) -> None:
        """Save model state dict and metadata."""
        if self.network is None:
            raise RuntimeError("No network to save.")
        torch.save({
            "state_dict": self.network.state_dict(),
            "input_dim": self.input_dim,
            "params": self.params,
        }, path)
        logger.info(f"MLP model saved to {path}")

    def load(self, path: str) -> None:
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.input_dim = checkpoint["input_dim"]
        self.params = checkpoint["params"]
        self._build_network()
        self.network.load_state_dict(checkpoint["state_dict"])
        self.is_fitted = True
        logger.info(f"MLP model loaded from {path}")
