#!/usr/bin/env python3
"""
Log-Gaussian Cox Process (LGCP) density estimator from particle positions.

Infers a smooth log-intensity field f on a regular 3D grid from the raw
particle positions (no deposition / binning).  The likelihood is the
inhomogeneous Poisson point-process likelihood evaluated at the exact
particle coordinates via trilinear interpolation:

    log p(positions | f) = sum_j f(x_j) - sum_i exp(f_i) * V_cell

where j runs over particles and i over grid cells.

Prior: Gaussian process with empirical power spectrum S(k) on a periodic
grid (FFT-diagonal covariance).

Inference: Laplace approximation — Newton-CG to find the MAP of the
posterior, with stochastic diagonal probing for posterior variance.

References
----------
- Moller, Syversveen & Waagepetersen (1998), "Log Gaussian Cox Processes"
- Rasmussen & Williams (2006), ch. 3.4 (Laplace approximation)
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.fft import fftn, ifftn, fftfreq
from scipy import sparse

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# ---------------------------------------------------------------------------
#  Paths to mini-ramses and denoise_cube utilities
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
MINI_RAMSES = ROOT / "mini-ramses-dev"
RAMSES_PIC = ROOT / "ramses-pic"
RUNDIR = ROOT / "rundir"

sys.path.insert(0, str(MINI_RAMSES / "utils" / "py"))
sys.path.insert(0, str(RUNDIR))


# ===================================================================
#  Trilinear interpolation matrix
# ===================================================================

def _trilinear_col_weights(
    positions: np.ndarray,
    nx: int,
    box_size: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (col_idx, weights) for the 8 grid nodes per particle, shape (8*n_part,).

    Used for building the sparse matrix and for PyTorch scatter_add.
    """
    n_part = positions.shape[0]
    dx = box_size / nx

    fx = positions[:, 0] / dx - 0.5
    fy = positions[:, 1] / dx - 0.5
    fz = positions[:, 2] / dx - 0.5

    ix0 = np.floor(fx).astype(np.int64)
    iy0 = np.floor(fy).astype(np.int64)
    iz0 = np.floor(fz).astype(np.int64)
    wx1 = (fx - ix0).astype(np.float64)
    wy1 = (fy - iy0).astype(np.float64)
    wz1 = (fz - iz0).astype(np.float64)
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1
    wz0 = 1.0 - wz1

    ix0 = ix0 % nx
    iy0 = iy0 % nx
    iz0 = iz0 % nx
    ix1 = (ix0 + 1) % nx
    iy1 = (iy0 + 1) % nx
    iz1 = (iz0 + 1) % nx

    ixs = [ix0, ix1]
    iys = [iy0, iy1]
    izs = [iz0, iz1]
    wxs = [wx0, wx1]
    wys = [wy0, wy1]
    wzs = [wz0, wz1]

    col_idx = np.empty(n_part * 8, dtype=np.int64)
    weights = np.empty(n_part * 8, dtype=np.float64)
    idx = 0
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                flat_cell = ixs[di] * (nx * nx) + iys[dj] * nx + izs[dk]
                w = wxs[di] * wys[dj] * wzs[dk]
                col_idx[idx * n_part:(idx + 1) * n_part] = flat_cell
                weights[idx * n_part:(idx + 1) * n_part] = w
                idx += 1
    return col_idx, weights


def build_trilinear_matrix(
    positions: np.ndarray,
    nx: int,
    box_size: float = 1.0,
) -> sparse.csr_matrix:
    """Build sparse trilinear interpolation matrix A (N_part x N_cells).

    For each particle at (x, y, z), find the 8 surrounding grid nodes and
    compute trilinear weights.  ``A @ f_flat`` evaluates the grid field f
    at every particle position; ``A.T @ ones`` gives the soft-count per cell.

    Parameters
    ----------
    positions : (N_part, 3) float array
        Particle positions in [0, box_size).
    nx : int
        Grid resolution (cubic: nx^3 cells).
    box_size : float
        Domain size.

    Returns
    -------
    A : sparse CSR matrix, shape (N_part, nx^3)
    """
    n_part = positions.shape[0]
    n_cells = nx * nx * nx
    col_idx, weights = _trilinear_col_weights(positions, nx, box_size)
    row_idx = np.repeat(np.arange(n_part), 8)
    A = sparse.csr_matrix((weights, (row_idx, col_idx)), shape=(n_part, n_cells))
    return A


# ===================================================================
#  Python CIC deposit (for initial PSD estimate, no Fortran needed)
# ===================================================================

def cic_deposit(
    positions: np.ndarray,
    masses: np.ndarray,
    nx: int,
    box_size: float = 1.0,
) -> np.ndarray:
    """Cloud-in-cell deposit of particles onto a regular grid.

    Returns
    -------
    density : (nx, nx, nx) array
        Mass per cell (not divided by cell volume).
    """
    dx = box_size / nx
    grid = np.zeros((nx, nx, nx), dtype=np.float64)

    fx = positions[:, 0] / dx - 0.5
    fy = positions[:, 1] / dx - 0.5
    fz = positions[:, 2] / dx - 0.5

    ix0 = np.floor(fx).astype(np.int64) % nx
    iy0 = np.floor(fy).astype(np.int64) % nx
    iz0 = np.floor(fz).astype(np.int64) % nx
    ix1 = (ix0 + 1) % nx
    iy1 = (iy0 + 1) % nx
    iz1 = (iz0 + 1) % nx

    wx1 = (fx - np.floor(fx)).astype(np.float64)
    wy1 = (fy - np.floor(fy)).astype(np.float64)
    wz1 = (fz - np.floor(fz)).astype(np.float64)
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1
    wz0 = 1.0 - wz1

    ixs = [ix0, ix1]
    iys = [iy0, iy1]
    izs = [iz0, iz1]
    wxs = [wx0, wx1]
    wys = [wy0, wy1]
    wzs = [wz0, wz1]

    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                w = masses * wxs[di] * wys[dj] * wzs[dk]
                np.add.at(grid, (ixs[di], iys[dj], izs[dk]), w)

    return grid


# ===================================================================
#  Fourier helpers (self-contained, no denoise_cube dependency at import)
# ===================================================================

def _build_k2_grid(shape: tuple[int, ...], box_size: float = 1.0) -> np.ndarray:
    """Return |k|^2 on a 3-D Fourier grid matching ``np.fft.fftn`` layout."""
    k_arrays = [fftfreq(n, d=box_size / n) * 2.0 * np.pi for n in shape]
    K = np.meshgrid(*k_arrays, indexing="ij")
    return K[0] ** 2 + K[1] ** 2 + K[2] ** 2


def _estimate_signal_psd(
    field: np.ndarray,
    noise_psd: float,
    box_size: float = 1.0,
    floor_snr: float = 0.003,
) -> np.ndarray:
    """Estimate signal PSD by subtracting white noise from the periodogram.

    Returns S(k) on the full 3-D Fourier grid, always > 0.
    """
    shape = field.shape
    n_total = int(np.prod(shape))
    power_3d = np.abs(fftn(field)) ** 2 / n_total

    k2 = _build_k2_grid(shape, box_size)
    k_flat = np.sqrt(k2).ravel()
    pow_flat = power_3d.ravel()
    nonzero = k_flat > 0

    n_bins = max(min(shape) // 2, 16)
    k_max = k_flat[nonzero].max()
    k_edges = np.linspace(0, k_max, n_bins + 1)
    k_centers = 0.5 * (k_edges[:-1] + k_edges[1:])

    pow_binned = np.zeros(n_bins)
    for i in range(n_bins):
        mask = nonzero & (k_flat >= k_edges[i]) & (k_flat < k_edges[i + 1])
        if mask.any():
            pow_binned[i] = np.mean(pow_flat[mask])

    sig_raw = pow_binned - noise_psd
    sig_binned = np.maximum(sig_raw, noise_psd * floor_snr)
    sig_binned = np.maximum(sig_binned, 1e-30)

    S_flat = np.interp(k_flat, k_centers, sig_binned)
    S_3d = S_flat.reshape(shape)
    S_3d.flat[0] = S_flat[nonzero].max() * 10.0
    return np.maximum(S_3d, 1e-30)


# ===================================================================
#  CG solver
# ===================================================================

def _cg_solve(
    matvec,
    rhs: np.ndarray,
    x0: np.ndarray | None = None,
    tol: float = 1e-6,
    maxiter: int = 200,
) -> np.ndarray:
    """Conjugate gradient for symmetric positive-definite system."""
    x = np.zeros_like(rhs) if x0 is None else x0.copy()
    r = rhs - matvec(x)
    p = r.copy()
    rs_old = np.sum(r * r)
    rhs_norm = np.sqrt(np.sum(rhs * rhs))
    if rhs_norm == 0:
        return x
    for _ in range(maxiter):
        Ap = matvec(p)
        pAp = np.sum(p * Ap)
        if pAp <= 0:
            break
        alpha = rs_old / pAp
        x += alpha * p
        r -= alpha * Ap
        rs_new = np.sum(r * r)
        if np.sqrt(rs_new) < tol * rhs_norm:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x


# ===================================================================
#  LGCP MAP solver
# ===================================================================

@dataclass
class LGCPResult:
    """Container for LGCP inference results."""
    density: np.ndarray       # Inferred density on grid (physical units)
    log_intensity: np.ndarray # MAP log-intensity f on grid
    variance: np.ndarray      # Posterior variance of f per cell
    n_newton: int             # Newton iterations used
    soft_counts: np.ndarray   # A^T @ 1 (trilinear soft counts per cell)


def lgcp_map_estimate(
    A: sparse.csr_matrix,
    n_particles: int,
    nx: int,
    V_cell: float,
    signal_psd: np.ndarray,
    prior_mean: float | None = None,
    max_newton: int = 30,
    newton_tol: float = 1e-5,
    cg_tol: float = 1e-6,
    cg_maxiter: int = 300,
    n_variance_probes: int = 30,
    verbose: bool = False,
) -> dict:
    r"""Find the MAP of the LGCP posterior via Newton-CG.

    Posterior:
        log p(f | positions) ∝ sum_j [A f]_j  -  sum_i exp(f_i) V_cell
                                - (1/2)(f-μ)^T K^{-1} (f-μ)

    Gradient of neg-log-posterior w.r.t. f_i:
        exp(f_i) V_cell  -  [A^T 1]_i  +  [K^{-1}(f-μ)]_i

    Hessian diagonal (likelihood part):
        W_i = exp(f_i) V_cell

    Parameters
    ----------
    A : sparse (N_part, N_cells)
        Trilinear interpolation matrix.
    n_particles : int
    nx : int
    V_cell : float
    signal_psd : (nx, nx, nx) array
        Prior power spectrum S(k).
    prior_mean : float, optional
        Prior mean of the latent field (scalar). Defaults to
        log(n_particles / N_cells) = log of expected intensity per cell.
    max_newton, newton_tol, cg_tol, cg_maxiter, n_variance_probes : tuning
    verbose : bool

    Returns
    -------
    dict with 'mode', 'variance', 'soft_counts', 'n_newton'.
    """
    shape = (nx, nx, nx)
    n_cells = nx ** 3

    S = signal_psd.astype(np.float64)
    S_inv = np.where(S > 1e-30, 1.0 / S, 0.0)

    # Soft counts: A^T @ 1  (how much trilinear weight each cell receives)
    ones_part = np.ones(n_particles, dtype=np.float64)
    soft_counts = np.asarray(A.T @ ones_part).ravel()
    soft_counts_3d = soft_counts.reshape(shape)

    # Prior mean
    if prior_mean is None:
        mean_intensity = n_particles / (n_cells * V_cell)
        mu = np.log(max(mean_intensity, 1e-10))
    else:
        mu = prior_mean

    # Initialize f from soft counts (smoothed to avoid log(0))
    sc_safe = np.maximum(soft_counts, 0.5)
    f = np.log(sc_safe / V_cell).reshape(shape)

    def kinv_matvec(v: np.ndarray) -> np.ndarray:
        return np.real(ifftn(fftn(v) * S_inv))

    n_iter_done = 0
    for it in range(max_newton):
        exp_f = np.exp(f)
        W = exp_f * V_cell  # Hessian diagonal (likelihood)

        # Gradient of neg-log-posterior:
        #   d/df_i [-log p(f|data)] = exp(f_i)*V_cell - soft_counts_i + [K^{-1}(f-mu)]_i
        grad_nll = W - soft_counts_3d + kinv_matvec(f - mu)

        grad_norm = float(np.sqrt(np.mean(grad_nll ** 2)))
        if verbose:
            ll = float(np.sum(soft_counts_3d * f) - np.sum(exp_f * V_cell))
            prior_term = float(-0.5 * np.sum((f - mu) * kinv_matvec(f - mu)))
            print(f"  Newton {it:3d}: |grad|={grad_norm:.3e}, "
                  f"LL={ll:.2f}, prior={prior_term:.2f}")

        if grad_norm < newton_tol:
            if verbose:
                print(f"  Converged at iteration {it}.")
            n_iter_done = it + 1
            break

        def hessian_matvec(v: np.ndarray) -> np.ndarray:
            return W * v + kinv_matvec(v)

        # Newton direction: H @ delta_f = -grad_nll
        delta_f = _cg_solve(hessian_matvec, -grad_nll, tol=cg_tol, maxiter=cg_maxiter)

        # Backtracking line search on log-posterior
        obj_current = ll + prior_term if verbose else (
            float(np.sum(soft_counts_3d * f) - np.sum(exp_f * V_cell))
            - 0.5 * float(np.sum((f - mu) * kinv_matvec(f - mu)))
        )
        step = 1.0
        for _ in range(20):
            f_trial = f + step * delta_f
            exp_trial = np.exp(f_trial)
            obj_trial = (
                float(np.sum(soft_counts_3d * f_trial) - np.sum(exp_trial * V_cell))
                - 0.5 * float(np.sum((f_trial - mu) * kinv_matvec(f_trial - mu)))
            )
            ascent = float(np.sum(-grad_nll * delta_f))
            if obj_trial >= obj_current + 1e-4 * step * ascent:
                break
            step *= 0.5
        if verbose and step < 1.0:
            print(f"           step={step:.4f}")
        f = f + step * delta_f
        n_iter_done = it + 1
    else:
        n_iter_done = max_newton

    # Posterior variance via stochastic diagonal probing
    W_final = np.exp(f) * V_cell

    def hessian_final(v: np.ndarray) -> np.ndarray:
        return W_final * v + kinv_matvec(v)

    rng = np.random.default_rng(42)
    diag_sum = np.zeros(shape, dtype=np.float64)
    for _ in range(n_variance_probes):
        z = rng.choice([-1.0, 1.0], size=shape)
        solve_z = _cg_solve(hessian_final, z, tol=cg_tol, maxiter=cg_maxiter)
        diag_sum += z * solve_z
    variance = diag_sum / n_variance_probes

    return {
        "mode": f,
        "variance": variance,
        "soft_counts": soft_counts_3d,
        "n_newton": n_iter_done,
    }


# ===================================================================
#  LGCP MAP solver (PyTorch)
# ===================================================================

# Clamp log-intensity to avoid exp(f) overflow (float32: exp(~88) = inf)
F_MAX = 30.0


def _trilinear_col_weights_torch(
    positions: "torch.Tensor",
    nx: int,
    box_size: float = 1.0,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Compute (col_idx, weights) for trilinear interpolation on device.
    positions: (N, 3) on device. Returns col_idx (long), weights (same dtype as positions),
    each shape (8*N,), on the same device. No CPU transfer."""
    n_part = positions.shape[0]
    dx = box_size / nx
    dtype = positions.dtype

    fx = positions[:, 0] / dx - 0.5
    fy = positions[:, 1] / dx - 0.5
    fz = positions[:, 2] / dx - 0.5

    ix0 = torch.floor(fx).long()
    iy0 = torch.floor(fy).long()
    iz0 = torch.floor(fz).long()
    wx1 = (fx - ix0.to(dtype))
    wy1 = (fy - iy0.to(dtype))
    wz1 = (fz - iz0.to(dtype))
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1
    wz0 = 1.0 - wz1

    ix0 = torch.remainder(ix0, nx)
    iy0 = torch.remainder(iy0, nx)
    iz0 = torch.remainder(iz0, nx)
    ix1 = torch.remainder(ix0 + 1, nx)
    iy1 = torch.remainder(iy0 + 1, nx)
    iz1 = torch.remainder(iz0 + 1, nx)

    col_parts = []
    weight_parts = []
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                ixs = [ix0, ix1]
                iys = [iy0, iy1]
                izs = [iz0, iz1]
                wxs = [wx0, wx1]
                wys = [wy0, wy1]
                wzs = [wz0, wz1]
                flat = ixs[di] * (nx * nx) + iys[dj] * nx + izs[dk]
                w = wxs[di] * wys[dj] * wzs[dk]
                col_parts.append(flat)
                weight_parts.append(w)
    col_idx_t = torch.cat(col_parts, dim=0)
    weights_t = torch.cat(weight_parts, dim=0)
    return col_idx_t, weights_t


def _check_nan_torch(t: "torch.Tensor", name: str) -> None:
    if torch.isnan(t).any().item() or torch.isinf(t).any().item():
        nnan = torch.isnan(t).sum().item()
        ninf = torch.isinf(t).sum().item()
        raise ValueError(f"LGCP Torch: {name} has nans={nnan}, infs={ninf}")


def _cg_solve_torch(
    matvec,
    rhs: "torch.Tensor",
    x0: "torch.Tensor | None" = None,
    tol: float = 1e-6,
    maxiter: int = 200,
    precond: "Callable[[torch.Tensor], torch.Tensor] | None" = None,
) -> "torch.Tensor":
    """Conjugate gradient for symmetric positive-definite system (PyTorch).
    If precond is given, uses preconditioned CG (M r = z, then p = z, etc.)."""
    x = torch.zeros_like(rhs) if x0 is None else x0.clone()
    r = rhs - matvec(x)
    z = precond(r) if precond is not None else r
    p = z.clone()
    rho = torch.sum(r * z).item()
    rhs_norm = torch.sqrt(torch.sum(rhs * rhs)).item()
    if rhs_norm == 0 or not np.isfinite(rho) or not np.isfinite(rhs_norm):
        return x
    for _ in range(maxiter):
        Ap = matvec(p)
        pAp = torch.sum(p * Ap).item()
        if not np.isfinite(pAp) or pAp <= 0:
            break
        alpha = rho / pAp
        if not np.isfinite(alpha):
            break
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.sum(r * r).item()
        if (rs_new ** 0.5) < tol * rhs_norm:
            break
        z = precond(r) if precond is not None else r
        rho_new = torch.sum(r * z).item()
        if not np.isfinite(rho_new) or rho <= 0:
            break
        p = z + (rho_new / rho) * p
        rho = rho_new
    return x


def lgcp_map_estimate_torch(
    n_particles: int,
    nx: int,
    V_cell: float,
    signal_psd: np.ndarray,
    device: str = "cpu",
    positions: "np.ndarray | torch.Tensor | None" = None,
    col_idx: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    prior_mean: float | None = None,
    max_newton: int = 30,
    newton_tol: float = 1e-5,
    cg_tol: float = 1e-6,
    cg_maxiter: int = 300,
    n_variance_probes: int = 30,
    verbose: bool = False,
) -> dict:
    """PyTorch LGCP MAP solver. FFT and CG run on the chosen device (cuda/mps/cpu).

    Either pass positions (preferred): (N, 3) array or tensor; col_idx/weights are
    computed on device (no CPU transfer). Or pass col_idx, weights (numpy) for
    backward compatibility.
    MPS does not support float64; when device is 'mps' we use float32.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for lgcp_map_estimate_torch")
    shape = (nx, nx, nx)
    n_cells = nx ** 3

    solver_device = device
    dtype = torch.float32 if device == "mps" else torch.float64

    if positions is not None:
        # Compute col_idx, weights on device from positions (no CPU transfer)
        box_size = float(nx) * (V_cell ** (1.0 / 3.0))
        if not isinstance(positions, torch.Tensor):
            positions_t = torch.from_numpy(np.asarray(positions, dtype=np.float64)).to(
                solver_device, dtype=dtype
            )
        else:
            positions_t = positions.to(solver_device, dtype=dtype)
        col_idx_t, weights_t = _trilinear_col_weights_torch(positions_t, nx, box_size)
    else:
        if col_idx is None or weights is None:
            raise ValueError("lgcp_map_estimate_torch requires either positions or (col_idx, weights)")
        col_idx_t = torch.from_numpy(col_idx).to(solver_device)
        weights_t = torch.from_numpy(weights).to(solver_device, dtype=dtype)

    # Soft counts: scatter_add weights by col_idx
    soft_counts_flat = torch.zeros(n_cells, device=solver_device, dtype=dtype)
    soft_counts_flat.scatter_add_(0, col_idx_t, weights_t)
    soft_counts_3d = soft_counts_flat.reshape(shape)
    _check_nan_torch(soft_counts_3d, "soft_counts_3d")

    np_dtype = np.float32 if dtype == torch.float32 else np.float64
    S = np.maximum(signal_psd.astype(np_dtype), 1e-30)
    S_inv = np.where(S > 1e-30, 1.0 / S, 0.0)
    S_inv_t = torch.from_numpy(S_inv).to(solver_device, dtype=dtype)

    if prior_mean is None:
        mean_intensity = n_particles / (n_cells * V_cell)
        mu = float(np.log(max(mean_intensity, 1e-10)))
    else:
        mu = prior_mean
    mu_t = torch.tensor(mu, device=solver_device, dtype=dtype)

    sc_safe = torch.clamp(soft_counts_3d, min=0.5)
    f = torch.clamp(torch.log(sc_safe / V_cell), min=-F_MAX, max=F_MAX)
    _check_nan_torch(f, "f (init)")

    # Mean of diag(K^{-1}) for diagonal preconditioner (H = W + K^{-1})
    k_prior_mean = torch.mean(S_inv_t).item()

    def kinv_matvec(v: "torch.Tensor") -> "torch.Tensor":
        v_fft = torch.fft.fftn(v)
        out = torch.fft.ifftn(v_fft * S_inv_t).real
        return out

    n_iter_done = 0
    for it in range(max_newton):
        exp_f = torch.exp(f)
        W = exp_f * V_cell

        grad_nll = W - soft_counts_3d + kinv_matvec(f - mu_t)
        _check_nan_torch(grad_nll, f"grad_nll (Newton {it})")
        grad_norm = torch.sqrt(torch.mean(grad_nll ** 2)).item()

        if verbose:
            ll = (torch.sum(soft_counts_3d * f) - torch.sum(exp_f * V_cell)).item()
            prior_term = (-0.5 * torch.sum((f - mu_t) * kinv_matvec(f - mu_t))).item()
            print(f"  Newton {it:3d}: |grad|={grad_norm:.3e}, LL={ll:.2f}, prior={prior_term:.2f}")

        if grad_norm < newton_tol:
            if verbose:
                print(f"  Converged at iteration {it}.")
            n_iter_done = it + 1
            break

        def hessian_matvec(v: "torch.Tensor") -> "torch.Tensor":
            return W * v + kinv_matvec(v)

        # Diagonal preconditioner M_ii = 1/(W_i + k_prior_mean) to reduce CG iterations
        diag_precond = 1.0 / (W + k_prior_mean)

        def precond(v: "torch.Tensor") -> "torch.Tensor":
            return diag_precond * v

        # Inexact Newton: looser CG tol when far from solution (fewer FFTs per step)
        cg_tol_eff = max(cg_tol, min(1e-2, float(grad_norm) * 0.1))

        delta_f = _cg_solve_torch(
            hessian_matvec, -grad_nll, tol=cg_tol_eff, maxiter=cg_maxiter, precond=precond
        )
        _check_nan_torch(delta_f, f"delta_f (Newton {it})")

        if not verbose:
            ll = (torch.sum(soft_counts_3d * f) - torch.sum(exp_f * V_cell)).item()
            prior_term = (-0.5 * torch.sum((f - mu_t) * kinv_matvec(f - mu_t))).item()
        obj_current = ll + prior_term
        step = 1.0
        ascent = torch.sum(-grad_nll * delta_f).item()
        if not np.isfinite(ascent):
            ascent = 0.0
        for _ in range(20):
            f_trial = torch.clamp(f + step * delta_f, min=-F_MAX, max=F_MAX)
            exp_trial = torch.exp(f_trial)
            obj_trial = (
                (torch.sum(soft_counts_3d * f_trial) - torch.sum(exp_trial * V_cell))
                - 0.5 * torch.sum((f_trial - mu_t) * kinv_matvec(f_trial - mu_t))
            ).item()
            if not np.isfinite(obj_trial):
                step *= 0.5
                continue
            if obj_trial >= obj_current + 1e-4 * step * ascent:
                break
            step *= 0.5
        if verbose and step < 1.0:
            print(f"           step={step:.4f}")
        f = torch.clamp(f + step * delta_f, min=-F_MAX, max=F_MAX)
        _check_nan_torch(f, f"f (after Newton {it})")
        n_iter_done = it + 1
    else:
        n_iter_done = max_newton

    W_final = torch.exp(f) * V_cell

    def hessian_final(v: "torch.Tensor") -> "torch.Tensor":
        return W_final * v + kinv_matvec(v)

    if n_variance_probes > 0:
        diag_precond_final = 1.0 / (W_final + k_prior_mean)

        def precond_final(v: "torch.Tensor") -> "torch.Tensor":
            return diag_precond_final * v

        torch.manual_seed(42)
        diag_sum = torch.zeros(shape, device=solver_device, dtype=dtype)
        for _ in range(n_variance_probes):
            z = torch.where(torch.rand(shape, device=solver_device, dtype=dtype) < 0.5,
                            torch.tensor(-1.0, device=solver_device, dtype=dtype),
                            torch.tensor(1.0, device=solver_device, dtype=dtype))
            solve_z = _cg_solve_torch(
                hessian_final, z, tol=cg_tol, maxiter=cg_maxiter, precond=precond_final
            )
            diag_sum = diag_sum + z * solve_z
        variance = (diag_sum / n_variance_probes).cpu().numpy()
        _check_nan_torch(torch.from_numpy(variance), "variance")
    else:
        variance = np.zeros(shape, dtype=np.float64)

    _check_nan_torch(f, "mode (final)")

    return {
        "mode": f.cpu().numpy(),
        "variance": variance,
        "soft_counts": soft_counts_3d.cpu().numpy(),
        "n_newton": n_iter_done,
    }


# ===================================================================
#  Main entry point
# ===================================================================

def lgcp_denoise(
    positions: np.ndarray,
    masses: np.ndarray,
    nx: int,
    box_size: float = 1.0,
    n_eff: float | None = None,
    floor_snr: float = 0.003,
    max_newton: int = 30,
    newton_tol: float = 1e-5,
    cg_tol: float = 1e-6,
    cg_maxiter: int = 300,
    n_variance_probes: int = 30,
    verbose: bool = False,
    use_torch: bool = True,
    device: str = "mps",
) -> LGCPResult:
    """Infer density on a grid from particle positions via LGCP.

    Parameters
    ----------
    positions : (N_part, 3) array
        Particle positions in [0, box_size).
    masses : (N_part,) array
        Particle masses (used for CIC initial PSD estimate; uniform is fine).
    nx : int
        Grid resolution.
    box_size : float
        Domain size.
    n_eff : float or None
        Mean particles per cell at mean density (for noise floor in PSD).
        If None (default), computed as n_particles / nx^3.
    floor_snr : float
        Minimum signal-to-noise ratio for PSD floor (default 0.003).
    max_newton, newton_tol, cg_tol, cg_maxiter, n_variance_probes : tuning
    verbose : bool
    use_torch : bool
        If True (default) and PyTorch is available, use PyTorch solver (default device: MPS).
    device : str
        Device for PyTorch: "cuda", "mps", or "cpu". FFT and CG both run on this device.

    Returns
    -------
    LGCPResult with density, log_intensity, variance, n_newton, soft_counts.
    """
    n_part = positions.shape[0]
    V_cell = (box_size / nx) ** 3
    n_cells = nx ** 3
    if n_eff is None:
        n_eff = max(float(n_part) / n_cells, 0.5)
    if verbose:
        print(f"LGCP: {n_part} particles, nx={nx}, box={box_size}, n_eff={n_eff:.2f}")

    # Step 1: For NumPy path, build trilinear matrix A. For PyTorch path, col_idx/weights are computed on device from positions.
    if not use_torch or not TORCH_AVAILABLE:
        if verbose:
            print("  Building trilinear interpolation matrix ...")
        A = build_trilinear_matrix(positions, nx, box_size)

    # Step 2: CIC deposit for initial PSD estimate
    if verbose:
        print("  CIC deposit for initial PSD estimate ...")
    cic_density = cic_deposit(positions, masses, nx, box_size)
    rho_mean = float(np.mean(cic_density))
    if rho_mean <= 0:
        raise ValueError("All-zero CIC deposit; no particles in domain?")
    delta = cic_density / rho_mean - 1.0

    noise_psd = 1.0 / n_eff
    if verbose:
        print(f"  Noise PSD = 1/n_eff = {noise_psd:.4e}")
    signal_psd = _estimate_signal_psd(delta, noise_psd, box_size, floor_snr=floor_snr)

    # Step 3: MAP inference
    if verbose:
        print("  Running Newton-CG MAP inference ...")
    if use_torch and TORCH_AVAILABLE:
        result = lgcp_map_estimate_torch(
            n_particles=n_part, nx=nx, V_cell=V_cell, signal_psd=signal_psd,
            device=device, positions=positions,
            max_newton=max_newton, newton_tol=newton_tol,
            cg_tol=cg_tol, cg_maxiter=cg_maxiter,
            n_variance_probes=n_variance_probes, verbose=verbose,
        )
    else:
        if use_torch and not TORCH_AVAILABLE and verbose:
            print("  PyTorch not available; using NumPy solver.")
        result = lgcp_map_estimate(
            A, n_part, nx, V_cell, signal_psd,
            max_newton=max_newton, newton_tol=newton_tol,
            cg_tol=cg_tol, cg_maxiter=cg_maxiter,
            n_variance_probes=n_variance_probes, verbose=verbose,
        )

    # Convert log-intensity to density
    # f = log(intensity), intensity = particles per unit volume
    # density = intensity * (total_mass / n_particles)
    total_mass = float(np.sum(masses))
    mass_per_particle = total_mass / n_part
    intensity = np.exp(result["mode"])
    density = intensity * mass_per_particle

    # Renormalize so mean density matches the CIC mean
    density *= rho_mean / float(np.mean(density)) if float(np.mean(density)) > 0 else 1.0

    if verbose:
        print(f"  Newton iterations: {result['n_newton']}")
        print(f"  Density range: [{density.min():.4e}, {density.max():.4e}]")

    return LGCPResult(
        density=density,
        log_intensity=result["mode"],
        variance=result["variance"],
        n_newton=result["n_newton"],
        soft_counts=result["soft_counts"],
    )


# ===================================================================
#  Cube I/O (self-contained)
# ===================================================================

def save_cube_fortran(cube: np.ndarray, output_file: str | Path) -> None:
    """Write a 3-D cube in Fortran unformatted layout (same as part2cube)."""
    nx, ny, nz = cube.shape
    with open(output_file, "wb") as f:
        rec = np.int32(3 * 4)
        rec.tofile(f)
        np.array([nx, ny, nz], dtype=np.int32).tofile(f)
        rec.tofile(f)
        flat = cube.astype(np.float32).flatten(order="F")
        rec = np.int32(flat.nbytes)
        rec.tofile(f)
        flat.tofile(f)
        rec.tofile(f)


# ===================================================================
#  Particle I/O
# ===================================================================

def load_particles(
    run_dir: str | Path,
    output_num: int,
    prefix: str = "dust",
    is_ramses: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Read particle positions and masses from a mini-ramses or RAMSES output.

    Returns
    -------
    positions : (N, 3) array
    masses : (N,) array
    """
    run_dir = Path(run_dir)

    if is_ramses:
        sys.path.insert(0, str(RAMSES_PIC / "utils" / "py"))
        import ramses_io as ram
        import os
        orig_cwd = os.getcwd()
        try:
            os.chdir(run_dir)
            p = ram.rd_part(output_num)
        finally:
            os.chdir(orig_cwd)
        positions = np.column_stack([p.xp[0], p.xp[1], p.xp[2]])
        masses = p.mp.copy()
    else:
        import miniramses as ram
        p = ram.rd_part(output_num, path=str(run_dir) + "/", prefix=prefix, silent=True)
        positions = np.column_stack([p.pos[0], p.pos[1], p.pos[2]])
        masses = p.mass.copy()

    # Filter out NaN or out-of-domain particles
    valid = np.all(np.isfinite(positions), axis=1)
    if not np.all(valid):
        n_bad = int(np.sum(~valid))
        positions = positions[valid]
        masses = masses[valid]
        print(f"  Filtered {n_bad} particles with NaN/inf positions")

    return positions.astype(np.float64), masses.astype(np.float64)


# ===================================================================
#  CLI
# ===================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="LGCP density estimator from particle positions."
    )
    ap.add_argument("--input-dir", type=str, required=True,
                    help="Run directory containing output_XXXXX/")
    ap.add_argument("--output-num", type=int, required=True,
                    help="Snapshot number")
    ap.add_argument("--prefix", type=str, default="dust",
                    help="Particle prefix (default: dust)")
    ap.add_argument("--nx", type=int, default=64,
                    help="Grid resolution (default: 64)")
    ap.add_argument("--n-eff", type=float, default=None, metavar="N",
                    help="Mean particles per cell for noise floor (default: auto from particles/nx^3)")
    ap.add_argument("--floor-snr", type=float, default=0.003,
                    help="PSD floor SNR (default: 0.003)")
    ap.add_argument("--box-size", type=float, default=1.0,
                    help="Physical box size (default: 1.0)")
    ap.add_argument("--output", type=str, default=None,
                    help="Output .cube file (default: lgcp_<prefix>_XXXXX.cube in input-dir)")
    ap.add_argument("--is-ramses", action="store_true",
                    help="Use RAMSES particle reader instead of mini-ramses")
    ap.add_argument("--no-torch", action="store_true",
                    help="Disable PyTorch solver (use NumPy backend)")
    ap.add_argument("--device", type=str, default="mps",
                    choices=("cuda", "mps", "cpu"),
                    help="Device for PyTorch: cuda, mps, or cpu (default: mps)")
    ap.add_argument("--max-newton", type=int, default=30)
    ap.add_argument("--n-variance-probes", type=int, default=30,
                    help="Number of Hutchinson probes for posterior variance (0 to skip, faster)")
    ap.add_argument("--no-variance", action="store_true",
                    help="Skip variance estimation (same as --n-variance-probes 0)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.input_dir)
    if not run_dir.exists():
        print(f"Error: directory {run_dir} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Loading particles from {run_dir} (output {args.output_num}, prefix={args.prefix}) ...")
    positions, masses = load_particles(run_dir, args.output_num, args.prefix, args.is_ramses)
    print(f"  {len(positions)} particles loaded")

    result = lgcp_denoise(
        positions, masses,
        nx=args.nx, box_size=args.box_size, n_eff=args.n_eff, floor_snr=args.floor_snr,
        max_newton=args.max_newton,
        n_variance_probes=0 if args.no_variance else args.n_variance_probes,
        verbose=args.verbose,
        use_torch=not args.no_torch,
        device=args.device,
    )

    out_file = args.output
    if out_file is None:
        out_file = run_dir / f"lgcp_{args.prefix}_{args.output_num:05d}.cube"
    else:
        out_file = Path(out_file)

    save_cube_fortran(result.density, out_file)
    print(f"Saved density cube to {out_file}")
    print(f"  Newton iterations: {result.n_newton}")
    print(f"  Density: mean={np.mean(result.density):.4e}, "
          f"min={result.density.min():.4e}, max={result.density.max():.4e}")


if __name__ == "__main__":
    main()
