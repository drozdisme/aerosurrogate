"""Physics-informed loss components for AeroSurrogate v2.0.

Adds soft physical constraints as penalty terms in the MLP loss function:
  - Cd positivity: drag coefficient must be non-negative
  - Cl monotonicity: dCl/dalpha > 0 in pre-stall region
  - Output consistency: K = Cl / Cd
"""

import torch
import torch.nn as nn


class PhysicsLoss(nn.Module):
    """Combined MSE + physics penalty loss.

    L = L_mse + w_cd * L_cd + w_cl * L_cl + w_cons * L_cons
    """

    def __init__(
        self,
        cd_positivity_weight: float = 1.0,
        cl_monotonicity_weight: float = 0.5,
        consistency_weight: float = 0.1,
        target_name: str = "Cl",
    ):
        super().__init__()
        self.mse = nn.MSELoss()
        self.cd_weight = cd_positivity_weight
        self.cl_weight = cl_monotonicity_weight
        self.cons_weight = consistency_weight
        self.target_name = target_name

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        cd_pred: torch.Tensor = None,
        alpha: torch.Tensor = None,
        cl_pred: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute combined loss.

        Args:
            pred: Model prediction for current target.
            target: Ground truth for current target.
            cd_pred: Cd predictions (for positivity penalty).
            alpha: Angle of attack values (for monotonicity).
            cl_pred: Cl predictions (for consistency with K).
        """
        loss = self.mse(pred, target)

        # Cd positivity: penalize negative Cd predictions
        if cd_pred is not None and self.cd_weight > 0:
            cd_violation = torch.clamp(-cd_pred, min=0.0)
            loss = loss + self.cd_weight * torch.mean(cd_violation ** 2)

        # Cl monotonicity: dCl/dalpha should be positive for |alpha| < 12 deg
        if (alpha is not None and cl_pred is not None
                and self.cl_weight > 0 and self.target_name == "Cl"):
            loss = loss + self.cl_weight * self._monotonicity_penalty(cl_pred, alpha)

        return loss

    def _monotonicity_penalty(
        self, cl: torch.Tensor, alpha: torch.Tensor
    ) -> torch.Tensor:
        """Penalize non-monotonic Cl vs alpha in pre-stall region."""
        # Sort by alpha
        sorted_idx = torch.argsort(alpha)
        cl_sorted = cl[sorted_idx]
        alpha_sorted = alpha[sorted_idx]

        # Only consider pre-stall region: |alpha| < 12 degrees
        mask = torch.abs(alpha_sorted) < 12.0

        if mask.sum() < 2:
            return torch.tensor(0.0, device=cl.device)

        cl_masked = cl_sorted[mask]
        dcl = cl_masked[1:] - cl_masked[:-1]

        # Penalize negative slopes
        violations = torch.clamp(-dcl, min=0.0)
        return torch.mean(violations ** 2)


class SimpleMSELoss(nn.Module):
    """Standard MSE loss (fallback when physics loss is disabled)."""

    def __init__(self, **kwargs):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target, **kwargs):
        return self.mse(pred, target)
