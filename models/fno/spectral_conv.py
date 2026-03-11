"""Spectral Convolution Layer for Fourier Neural Operator.

Implements the spectral convolution operator:
    (F^{-1} ○ R ○ F)(v)(x)

where F is the DFT, R is a learnable linear transform on Fourier modes,
and F^{-1} is the inverse DFT.

Reference: Li et al. (2020) "Fourier Neural Operator for Parametric PDEs"
           https://arxiv.org/abs/2010.08895
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    """1D Spectral Convolution — operates on signal of shape (B, C_in, N).

    Parameters
    ----------
    in_channels : int
    out_channels : int
    n_modes : int
        Number of Fourier modes to keep (low-frequency modes).
    """

    def __init__(self, in_channels: int, out_channels: int, n_modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes

        # Complex-valued weight tensor: (in_ch, out_ch, n_modes)
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, n_modes, 2)
        )

    def _complex_mul1d(
        self,
        x: torch.Tensor,      # (B, in_ch, n_modes) complex
        w: torch.Tensor,      # (in_ch, out_ch, n_modes) complex
    ) -> torch.Tensor:
        """Batched complex matrix–vector multiply along mode dimension."""
        # x: (B, in_ch, n_modes), w: (in_ch, out_ch, n_modes)
        return torch.einsum("bim,iom->bom", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, in_channels, N)  real tensor

        Returns:
            out : (B, out_channels, N) real tensor
        """
        B, C, N = x.shape

        # FFT along last dimension
        x_ft = torch.fft.rfft(x, dim=-1)              # (B, C, N//2+1) complex

        # Keep only lowest n_modes Fourier coefficients
        out_ft = torch.zeros(
            B, self.out_channels, N // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )

        w_complex = torch.view_as_complex(self.weights)  # (in, out, n_modes)

        out_ft[:, :, :self.n_modes] = self._complex_mul1d(
            x_ft[:, :, :self.n_modes], w_complex
        )

        # Inverse FFT → back to physical space
        out = torch.fft.irfft(out_ft, n=N, dim=-1)    # (B, out_channels, N)
        return out


class SpectralConv2d(nn.Module):
    """2D Spectral Convolution — operates on field of shape (B, C_in, H, W).

    Independently transforms modes along each spatial dimension.

    Parameters
    ----------
    in_channels, out_channels : int
    n_modes_x, n_modes_y : int
        Number of Fourier modes to retain along each dimension.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes_x: int,
        n_modes_y: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes_x = n_modes_x
        self.n_modes_y = n_modes_y

        scale = 1.0 / (in_channels * out_channels)

        # Four quadrant weights (standard FNO-2D decomposition)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, n_modes_x, n_modes_y, 2)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, n_modes_x, n_modes_y, 2)
        )

    def _complex_mul2d(
        self,
        x: torch.Tensor,   # (B, in_ch, mx, my) complex
        w: torch.Tensor,   # (in_ch, out_ch, mx, my) complex
    ) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, in_channels, H, W)

        Returns:
            (B, out_channels, H, W)
        """
        B, C, H, W = x.shape

        x_ft = torch.fft.rfft2(x, dim=(-2, -1))       # (B, C, H, W//2+1)

        out_ft = torch.zeros(
            B, self.out_channels, H, W // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )

        w1 = torch.view_as_complex(self.weights1)
        w2 = torch.view_as_complex(self.weights2)

        # Upper-left quadrant
        out_ft[:, :, :self.n_modes_x, :self.n_modes_y] = self._complex_mul2d(
            x_ft[:, :, :self.n_modes_x, :self.n_modes_y], w1
        )
        # Lower-left quadrant (negative frequencies in H)
        out_ft[:, :, -self.n_modes_x:, :self.n_modes_y] = self._complex_mul2d(
            x_ft[:, :, -self.n_modes_x:, :self.n_modes_y], w2
        )

        out = torch.fft.irfft2(out_ft, s=(H, W), dim=(-2, -1))   # (B, out_ch, H, W)
        return out
