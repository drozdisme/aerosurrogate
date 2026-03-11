"""DeepONet (Deep Operator Network) for surface field predictions.

Learns the operator: (geometry, flow conditions) -> Cp(x)
using a branch-trunk architecture.

Branch net: encodes the input parameters (geometry + flow).
Trunk net: encodes the query location (surface arc-length coordinate).
Output: dot product of branch and trunk latent representations.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class BranchNet(nn.Module):
    """Encodes input parameters into a latent representation."""

    def __init__(self, input_dim: int, hidden_layers: List[int], latent_dim: int):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers.extend([nn.Linear(prev, h), nn.GELU(), nn.LayerNorm(h)])
            prev = h
        layers.append(nn.Linear(prev, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TrunkNet(nn.Module):
    """Encodes query locations into a latent representation."""

    def __init__(self, input_dim: int, hidden_layers: List[int], latent_dim: int):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers.extend([nn.Linear(prev, h), nn.GELU(), nn.LayerNorm(h)])
            prev = h
        layers.append(nn.Linear(prev, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepONet(nn.Module):
    """Deep Operator Network for field prediction.

    Given input parameters u and query point y:
        output(u, y) = sum_k branch_k(u) * trunk_k(y) + bias
    """

    def __init__(
        self,
        param_dim: int,
        coord_dim: int = 1,
        branch_layers: List[int] = None,
        trunk_layers: List[int] = None,
        latent_dim: int = 64,
    ):
        super().__init__()
        branch_layers = branch_layers or [256, 128, 64]
        trunk_layers = trunk_layers or [128, 64, 64]

        self.branch = BranchNet(param_dim, branch_layers, latent_dim)
        self.trunk = TrunkNet(coord_dim, trunk_layers, latent_dim)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(
        self, params: torch.Tensor, coords: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            params: (batch_size, param_dim) — input parameters.
            coords: (n_points, coord_dim) — query coordinates.

        Returns:
            (batch_size, n_points) — predicted field values.
        """
        b = self.branch(params)    # (batch, latent)
        t = self.trunk(coords)     # (n_points, latent)
        # Dot product: (batch, latent) @ (latent, n_points) -> (batch, n_points)
        out = torch.matmul(b, t.T) + self.bias
        return out


class DeepONetModel:
    """Training and inference wrapper for DeepONet."""

    def __init__(self, param_dim: int, params: Optional[Dict] = None):
        self.param_dim = param_dim
        self.device = torch.device("cpu")

        default_params = {
            "branch_layers": [256, 128, 64],
            "trunk_layers": [128, 64, 64],
            "latent_dim": 64,
            "learning_rate": 0.0005,
            "epochs": 500,
            "batch_size": 32,
            "patience": 40,
            "n_surface_points": 200,
        }
        if params:
            default_params.update(params)
        self.params = default_params

        self.network: Optional[DeepONet] = None
        self.is_fitted = False
        self._build()

    def _build(self) -> None:
        torch.manual_seed(42)
        self.network = DeepONet(
            param_dim=self.param_dim,
            coord_dim=1,
            branch_layers=self.params["branch_layers"],
            trunk_layers=self.params["trunk_layers"],
            latent_dim=self.params["latent_dim"],
        ).to(self.device)

    def fit(
        self,
        X_params: np.ndarray,
        Y_fields: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        Y_val: Optional[np.ndarray] = None,
    ) -> None:
        """Train DeepONet on parameter-field pairs.

        Args:
            X_params: (n_samples, param_dim) — scaled input parameters.
            Y_fields: (n_samples, n_points) — target Cp distributions.
        """
        n_points = Y_fields.shape[1]

        # Surface coordinates as uniform arc-length parameterization
        coords = torch.linspace(0, 1, n_points).unsqueeze(1).to(self.device)

        X_t = torch.FloatTensor(X_params).to(self.device)
        Y_t = torch.FloatTensor(Y_fields).to(self.device)

        dataset = TensorDataset(X_t, Y_t)
        loader = DataLoader(dataset, batch_size=self.params["batch_size"], shuffle=True)

        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.params["learning_rate"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
        criterion = nn.MSELoss()

        has_val = X_val is not None and Y_val is not None
        if has_val:
            Xv = torch.FloatTensor(X_val).to(self.device)
            Yv = torch.FloatTensor(Y_val).to(self.device)

        best_loss = float("inf")
        best_state = None
        patience_counter = 0

        self.network.train()
        for epoch in range(self.params["epochs"]):
            epoch_loss = 0.0
            for X_batch, Y_batch in loader:
                optimizer.zero_grad()
                pred = self.network(X_batch, coords)
                loss = criterion(pred, Y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(X_batch)
            epoch_loss /= len(X_t)

            val_loss = epoch_loss
            if has_val:
                self.network.eval()
                with torch.no_grad():
                    val_pred = self.network(Xv, coords)
                    val_loss = criterion(val_pred, Yv).item()
                self.network.train()

            scheduler.step(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.network.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.params["patience"]:
                    logger.info(f"DeepONet early stopping at epoch {epoch+1}")
                    break

        if best_state:
            self.network.load_state_dict(best_state)

        self.is_fitted = True
        logger.info(f"DeepONet trained for {epoch+1} epochs, best loss={best_loss:.6f}")

    def predict(self, X_params: np.ndarray, n_points: int = 200) -> np.ndarray:
        """Predict Cp field for given parameters.

        Returns array of shape (n_samples, n_points).
        """
        if not self.is_fitted:
            raise RuntimeError("DeepONet not trained")

        coords = torch.linspace(0, 1, n_points).unsqueeze(1).to(self.device)
        X_t = torch.FloatTensor(X_params).to(self.device)

        self.network.eval()
        with torch.no_grad():
            pred = self.network(X_t, coords).cpu().numpy()
        return pred

    def save(self, path: str) -> None:
        torch.save({
            "state_dict": self.network.state_dict(),
            "param_dim": self.param_dim,
            "params": self.params,
        }, path)
        logger.info(f"DeepONet saved to {path}")

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.param_dim = checkpoint["param_dim"]
        self.params = checkpoint["params"]
        self._build()
        self.network.load_state_dict(checkpoint["state_dict"])
        self.is_fitted = True
        logger.info(f"DeepONet loaded from {path}")
