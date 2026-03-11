"""Fourier Neural Operator (FNO) for Aerodynamic Field Prediction.

Predicts the pressure coefficient distribution Cp(x) for arbitrary
surface grid resolutions — the model is resolution-independent by design.

Architecture (1D FNO operating on surface arc-length coordinate):
    1. Lift: project input channels → width
    2. L Fourier layers: spectral_conv(x) + W(x) → σ(·)
    3. Project: width → 128 → 1

Input encoding strategy:
    - Encode (geometry_params, flow_conditions) as constant features
      broadcast along the spatial dimension alongside x ∈ [0,1].
    - This is equivalent to the "parameterized FNO" formulation.

Reference: Li et al. (2020) https://arxiv.org/abs/2010.08895
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models.fno.spectral_conv import SpectralConv1d

logger = logging.getLogger(__name__)


class FNOBlock1d(nn.Module):
    """Single FNO layer: spectral convolution + bypass linear + activation."""

    def __init__(self, width: int, n_modes: int):
        super().__init__()
        self.spectral_conv = SpectralConv1d(width, width, n_modes)
        self.bypass = nn.Conv1d(width, width, kernel_size=1)  # pointwise W
        self.norm = nn.InstanceNorm1d(width, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, width, N)"""
        return F.gelu(self.norm(self.spectral_conv(x) + self.bypass(x)))


class FNO1d(nn.Module):
    """1D Fourier Neural Operator for Cp(x) prediction.

    Parameters
    ----------
    param_dim : int
        Dimension of the parameter vector (geometry + flow conditions).
    n_modes : int
        Number of Fourier modes to retain per layer (default 16).
    width : int
        Hidden channel width (default 64).
    n_layers : int
        Number of Fourier layers (default 4).
    """

    def __init__(
        self,
        param_dim: int,
        n_modes: int = 16,
        width: int = 64,
        n_layers: int = 4,
    ):
        super().__init__()
        self.param_dim = param_dim
        self.n_modes = n_modes
        self.width = width
        self.n_layers = n_layers

        # Input: 1 spatial coordinate + param_dim conditions → width
        self.lift = nn.Linear(1 + param_dim, width)

        # Fourier layers
        self.fno_blocks = nn.ModuleList([
            FNOBlock1d(width, n_modes) for _ in range(n_layers)
        ])

        # Projection: width → 128 → 1
        self.proj = nn.Sequential(
            nn.Linear(width, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        params: torch.Tensor,       # (B, param_dim)
        coords: torch.Tensor,       # (B, N) or (N,) — surface grid x ∈ [0,1]
    ) -> torch.Tensor:
        """Predict Cp field at given surface coordinates.

        Returns:
            (B, N) — Cp values at the queried coordinates.
        """
        B = params.shape[0]

        # Normalize coords to (B, N, 1)
        if coords.dim() == 1:
            coords = coords.unsqueeze(0).expand(B, -1)   # (B, N)
        N = coords.shape[-1]
        x = coords.unsqueeze(-1)                          # (B, N, 1)

        # Broadcast params along spatial axis: (B, N, param_dim)
        p = params.unsqueeze(1).expand(B, N, -1)

        # Concatenate: (B, N, 1 + param_dim)
        inp = torch.cat([x, p], dim=-1)

        # Lift to (B, N, width), transpose to (B, width, N) for Conv1d
        v = self.lift(inp).permute(0, 2, 1)              # (B, width, N)

        # Apply Fourier layers
        for block in self.fno_blocks:
            v = block(v)

        # Project: (B, N, width) → (B, N, 1) → (B, N)
        v = v.permute(0, 2, 1)                            # (B, N, width)
        out = self.proj(v).squeeze(-1)                    # (B, N)

        return out


class FNOModel:
    """Training and inference wrapper for FNO1d.

    Replaces DeepONetModel with identical interface for drop-in
    substitution in the training pipeline.

    Parameters
    ----------
    param_dim : int
        Dimension of the input parameter vector.
    params : dict, optional
        Override default hyperparameters.
    """

    DEFAULT_PARAMS: Dict = {
        "n_modes": 16,
        "width": 64,
        "n_layers": 4,
        "learning_rate": 5e-4,
        "epochs": 500,
        "batch_size": 32,
        "patience": 50,
        "n_surface_points": 200,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
    }

    def __init__(
        self,
        param_dim: int,
        params: Optional[Dict] = None,
        device: Optional[str] = None,
    ):
        self.param_dim = param_dim
        self.hparams = {**self.DEFAULT_PARAMS, **(params or {})}
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.network: Optional[FNO1d] = None
        self.is_fitted = False
        self.train_history: List[Dict] = []
        self._build()

    def _build(self) -> None:
        torch.manual_seed(42)
        self.network = FNO1d(
            param_dim=self.param_dim,
            n_modes=self.hparams["n_modes"],
            width=self.hparams["width"],
            n_layers=self.hparams["n_layers"],
        ).to(self.device)

        n_params = sum(p.numel() for p in self.network.parameters())
        logger.info(
            f"FNO1d built: param_dim={self.param_dim}, "
            f"modes={self.hparams['n_modes']}, width={self.hparams['width']}, "
            f"layers={self.hparams['n_layers']} | "
            f"Total parameters: {n_params:,}"
        )

    def fit(
        self,
        X_params: np.ndarray,         # (N, param_dim) — scaled flow + geometry params
        Y_fields: np.ndarray,         # (N, n_points) — Cp distributions
        X_val: Optional[np.ndarray] = None,
        Y_val: Optional[np.ndarray] = None,
    ) -> None:
        """Train FNO on (parameter, Cp field) pairs.

        The surface coordinate grid is inferred from n_points = Y_fields.shape[1].
        The model is resolution-independent: at inference time you may query
        any n_points without retraining.
        """
        n_points = Y_fields.shape[1]

        # Surface grid: uniform arc-length parameterization
        coords = torch.linspace(0, 1, n_points).to(self.device)   # (n_points,)

        X_t = torch.FloatTensor(X_params).to(self.device)
        Y_t = torch.FloatTensor(Y_fields).to(self.device)

        dataset = TensorDataset(X_t, Y_t)
        loader = DataLoader(
            dataset,
            batch_size=self.hparams["batch_size"],
            shuffle=True,
            pin_memory=self.device.type == "cuda",
            num_workers=0,
        )

        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.hparams["learning_rate"],
            weight_decay=self.hparams["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams["learning_rate"],
            epochs=self.hparams["epochs"],
            steps_per_epoch=len(loader),
            pct_start=0.1,
        )

        has_val = X_val is not None and Y_val is not None
        if has_val:
            Xv = torch.FloatTensor(X_val).to(self.device)
            Yv = torch.FloatTensor(Y_val).to(self.device)

        best_loss = float("inf")
        best_state: Optional[Dict] = None
        patience_counter = 0

        for epoch in range(self.hparams["epochs"]):
            self.network.train()
            epoch_loss = 0.0

            for X_batch, Y_batch in loader:
                optimizer.zero_grad(set_to_none=True)

                # coords broadcast: (B, n_points) — same grid for all samples
                batch_coords = coords.unsqueeze(0).expand(len(X_batch), -1)
                pred = self.network(X_batch, batch_coords)   # (B, n_points)

                # Relative L2 loss (standard FNO training objective)
                loss = self._relative_l2(pred, Y_batch)

                loss.backward()
                if self.hparams["grad_clip"] > 0:
                    nn.utils.clip_grad_norm_(
                        self.network.parameters(), self.hparams["grad_clip"]
                    )
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item() * len(X_batch)

            epoch_loss /= len(X_t)

            val_loss = epoch_loss
            if has_val:
                self.network.eval()
                with torch.no_grad():
                    batch_coords = coords.unsqueeze(0).expand(len(Xv), -1)
                    val_pred = self.network(Xv, batch_coords)
                    val_loss = self._relative_l2(val_pred, Yv).item()
                self.network.train()

            self.train_history.append({
                "epoch": epoch + 1,
                "train_loss": epoch_loss,
                "val_loss": val_loss,
                "lr": scheduler.get_last_lr()[0],
            })

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.network.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.hparams["patience"]:
                    logger.info(f"FNO early stopping at epoch {epoch + 1}")
                    break

            if (epoch + 1) % 50 == 0:
                logger.info(
                    f"FNO epoch {epoch+1:4d} | "
                    f"train={epoch_loss:.5f} | "
                    f"val={val_loss:.5f} | "
                    f"best={best_loss:.5f} | "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )

        if best_state:
            self.network.load_state_dict(best_state)

        self.is_fitted = True
        logger.info(
            f"FNO training complete. Best val loss: {best_loss:.6f} "
            f"(relative L2) after {epoch+1} epochs"
        )

    def predict(
        self,
        X_params: np.ndarray,
        n_points: int = 200,
        coords: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict Cp field at arbitrary resolution.

        Args:
            X_params : (n_samples, param_dim) scaled parameters.
            n_points : Resolution of the output grid (default 200).
            coords : Optional custom x-coordinates ∈ [0,1] of shape (n_points,).
                     If None, uses uniform grid.

        Returns:
            (n_samples, n_points) Cp field predictions.
        """
        if not self.is_fitted:
            raise RuntimeError("FNOModel not trained — call fit() first")

        if coords is None:
            query = torch.linspace(0, 1, n_points).to(self.device)
        else:
            query = torch.FloatTensor(coords).to(self.device)

        X_t = torch.FloatTensor(X_params).to(self.device)

        self.network.eval()
        results = []
        batch_size = 64

        with torch.no_grad():
            for i in range(0, len(X_t), batch_size):
                X_b = X_t[i:i + batch_size]
                grid = query.unsqueeze(0).expand(len(X_b), -1)
                pred = self.network(X_b, grid)                  # (B, n_points)
                results.append(pred.cpu().detach().float().numpy())

        return np.concatenate(results, axis=0)

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            "state_dict": self.network.state_dict(),
            "param_dim": self.param_dim,
            "hparams": self.hparams,
            "train_history": self.train_history,
            "model_class": "FNO1d",
        }, path)
        logger.info(f"FNO saved → {path}")

    def load(self, path: str) -> None:
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.param_dim = checkpoint["param_dim"]
        self.hparams = checkpoint["hparams"]
        self.train_history = checkpoint.get("train_history", [])
        self._build()
        self.network.load_state_dict(checkpoint["state_dict"])
        self.is_fitted = True
        logger.info(f"FNO loaded from {path}")

    # ── Loss functions ─────────────────────────────────────────────────────────

    @staticmethod
    def _relative_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Relative L2 loss: ||pred - target||_2 / ||target||_2 (per sample, mean)."""
        diff = pred - target
        loss = torch.norm(diff, dim=-1) / (torch.norm(target, dim=-1) + 1e-8)
        return loss.mean()

    @staticmethod
    def _mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target)
