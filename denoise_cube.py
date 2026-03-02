#!/usr/bin/env python3
"""
Bayesian GP denoising of 3D particle density cubes from mini-ramses-dev.

Statistical Model
-----------------
Monte Carlo tracer and dust particles in mini-ramses-dev are stochastic
realizations of an underlying smooth density field.  When deposited onto a
grid (CIC/TSC/PCS), each cell's density is a noisy observation whose
variance is dominated by finite-particle Poisson statistics.

We model the latent log-intensity field as a Gaussian process with a
stationary kernel on a periodic 3D grid.  Because the kernel is stationary
and the grid is periodic, the prior covariance matrix K is circulant and
diagonalized by the DFT:

    K = F^H  diag(S(k))  F

where S(k) is the kernel power spectrum.  This reduces all GP algebra from
O(N^3) to O(N log N).

Two inference methods are provided:

1. **Wiener filter** (``method='wiener'``): exact GP posterior mean for
   Gaussian noise.  Computes ``f_hat(k) = S(k)/(S(k)+N(k)) * y_hat(k)``
   in a single FFT round-trip.

2. **Laplace GP** (``method='laplace'``): posterior mode under a Poisson
   likelihood via Newton-CG in log-count space.  Each Newton step solves
   ``(W + K^{-1}) df = grad`` with conjugate gradient, where the K^{-1}
   mat-vec costs one FFT pair.  Posterior variance is estimated by
   stochastic diagonal probing of ``(W + K^{-1})^{-1}``.

3. **Gaussian SNR** (``method='gaussian_snr'``): smooth with a Gaussian
   kernel whose width is set by the signal-to-noise ratio:
   :math:`\\sigma = \\mathrm{scale}/\\sqrt{n_\\mathrm{eff}}` in grid cells.
   Equivalent to convolving with a Gaussian of that width.  No signal
   spectrum estimation; the same filter can be applied to gas for
   comparable tracer–gas plots.

Noise Models
------------
- **Poisson** (default): observed count n_i ~ Poisson(lambda_i).  The
  density cube is related to counts by  n = rho * n_eff / rho_mean.
- **Gaussian**: homoscedastic or heteroscedastic additive noise.

Kernels
-------
- Empirical (default): non-parametric power spectrum estimated directly
  from the data.  Best for turbulent fields whose power spectrum follows
  a broad power-law — a single-length-scale parametric kernel cannot
  capture this and will over-smooth.
- Matern-3/2: once-differentiable; single characteristic length scale.
- Matern-5/2: twice-differentiable.
- RBF / squared-exponential: infinitely smooth.

References
----------
- Rasmussen & Williams (2006), *Gaussian Processes for Machine Learning*,
  ch. 2 (GP regression) and ch. 3.4 (Laplace approximation).
- Rue & Held (2005), *Gaussian Markov Random Fields*, ch. 4.4 (Poisson GP).
- Hockney & Eastwood (1988), *Computer Simulation Using Particles*,
  ch. 5 (CIC/TSC deposition and aliasing).

Usage
-----
As a library::

    from denoise_cube import denoise_cube
    result = denoise_cube(density_cube, n_eff=16, method='laplace')
    denoised = result['denoised']
    variance = result['variance']

From the command line::

    python denoise_cube.py --input my_cube.cube --n-eff 16 --output denoised.cube
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.fft import fftn, ifftn, fftfreq
from scipy.optimize import minimize_scalar

# ---------------------------------------------------------------------------
# Paths: use local utils/ when running from particle_analysis repo
# ---------------------------------------------------------------------------
_PKG_DIR = Path(__file__).resolve().parent
MINI_RAMSES = _PKG_DIR
MINI_PART2CUBE = str(MINI_RAMSES / "utils" / "f90" / "part2cube")
sys.path.insert(0, str(MINI_RAMSES / "utils" / "py"))

# ===================================================================
#  Data I/O
# ===================================================================

def read_cube_fortran(cube_file: str | Path) -> np.ndarray:
    """Read a 3-D density cube written by ``part2cube`` (Fortran unformatted).

    The file layout is two Fortran records:
      1. ``nx, ny, nz``  (int32)
      2. ``cube(nx*ny*nz)``  (float32 or float64, column-major)

    Returns
    -------
    cube : ndarray, shape (nx, ny, nz)
    """
    cube_file = Path(cube_file)
    file_size = cube_file.stat().st_size
    with open(cube_file, "rb") as f:
        _ = np.fromfile(f, dtype=np.int32, count=1)[0]
        nx, ny, nz = np.fromfile(f, dtype=np.int32, count=3)
        _ = np.fromfile(f, dtype=np.int32, count=1)[0]
        n_cells = nx * ny * nz
        expected_f32 = 28 + n_cells * 4
        dtype = np.float32 if file_size == expected_f32 else np.float64
        _ = np.fromfile(f, dtype=np.int32, count=1)[0]
        cube = np.fromfile(f, dtype=dtype, count=n_cells)
        _ = np.fromfile(f, dtype=np.int32, count=1)[0]
    return cube.reshape((nx, ny, nz), order="F")


def save_cube_fortran(cube: np.ndarray, output_file: str | Path) -> None:
    """Write a 3-D cube in the same Fortran unformatted layout as ``part2cube``.

    Parameters
    ----------
    cube : ndarray, shape (nx, ny, nz)
    output_file : path
    """
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


def load_tracer_cube(
    run_dir: str | Path,
    output_num: int,
    nx: int,
    *,
    prefix: str = "trac",
    dep: str = "NGP",
    part2cube: str | None = None,
) -> np.ndarray:
    """Generate a tracer density cube by calling the Fortran ``part2cube``.

    Parameters
    ----------
    run_dir : path
        Directory containing ``output_XXXXX/`` snapshots.
    output_num : int
        Snapshot number.
    nx : int
        Grid resolution (same in x, y, z).
    prefix : str
        Particle file prefix (``'trac'`` or ``'dust'``).
    dep : str
        Deposition scheme: ``'NGP'`` (default), ``'CIC'``, ``'TSC'``, or ``'PCS'``.
    part2cube : str, optional
        Path to the part2cube executable.  Defaults to the mini-ramses-dev
        build.

    Returns
    -------
    cube : ndarray, shape (nx, nx, nx)
    """
    run_dir = Path(run_dir)
    if part2cube is None:
        part2cube = MINI_PART2CUBE
    output_dir_name = f"output_{output_num:05d}"
    cube_file = run_dir / f"{prefix}_{output_num:05d}.cube"
    cmd = [
        part2cube, "-inp", output_dir_name, "-pre", prefix,
        "-nx", str(nx), "-ny", str(nx), "-nz", str(nx),
        "-per", ".true.", "-dep", dep,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(run_dir))
    if result.returncode != 0:
        raise RuntimeError(f"part2cube failed: {result.stderr}")
    temp_cube = run_dir / f"{prefix}.cube"
    if temp_cube.exists() and temp_cube != cube_file:
        temp_cube.rename(cube_file)
    if not cube_file.exists():
        raise FileNotFoundError(f"part2cube did not produce {cube_file}")
    return read_cube_fortran(cube_file)


def load_gas_cube(run_dir: str | Path, output_num: int) -> np.ndarray:
    """Build a gas density cube from AMR cell data via ``miniramses``.

    Parameters
    ----------
    run_dir : path
        Directory containing ``output_XXXXX/`` snapshots.
    output_num : int
        Snapshot number.

    Returns
    -------
    cube : ndarray, shape (nx, ny, nz)
    """
    import miniramses as ram  # lazy import; requires MINI_RAMSES on sys.path

    c = ram.rd_cell(output_num, path=str(run_dir).rstrip("/") + "/")
    cube = ram.mk_cube(c.x[0], c.x[1], c.x[2], c.dx, c.u[0])
    return cube.T


# ===================================================================
#  Kernel power spectra
# ===================================================================

def _build_k2_grid(shape: tuple[int, ...], box_size: float = 1.0) -> np.ndarray:
    """Return |k|^2 on a 3-D Fourier grid matching ``np.fft.fftn`` layout."""
    k_arrays = [
        fftfreq(n, d=box_size / n) * 2.0 * np.pi for n in shape
    ]
    K = np.meshgrid(*k_arrays, indexing="ij")
    return K[0] ** 2 + K[1] ** 2 + K[2] ** 2


def kernel_power_spectrum(
    k2: np.ndarray,
    kernel: str = "matern32",
    length_scale: float = 2.0,
    amplitude: float = 1.0,
) -> np.ndarray:
    r"""Spectral density S(k) of a stationary kernel in 3-D.

    For a GP on a periodic grid the prior covariance is circulant with
    eigenvalues given by the *spectral density* evaluated at the grid
    wavenumbers.

    Supported kernels (all isotropic):

    - ``'rbf'``: :math:`S(k) = A^2 (2\pi)^{3/2} \ell^3
      \exp(-\ell^2 k^2/2)`.
    - ``'matern32'``: Matern with :math:`\nu=3/2`.
      :math:`S(k) \propto (\alpha^2 + k^2)^{-5/2}` where
      :math:`\alpha = \sqrt{3}/\ell`.
    - ``'matern52'``: Matern with :math:`\nu=5/2`.
      :math:`S(k) \propto (\alpha^2 + k^2)^{-4}` where
      :math:`\alpha = \sqrt{5}/\ell`.

    Parameters
    ----------
    k2 : ndarray
        Squared wavenumber grid, |k|^2.
    kernel : str
    length_scale : float
        Characteristic length in *grid cells*.
    amplitude : float
        Signal standard deviation.

    Returns
    -------
    S : ndarray, same shape as *k2*
    """
    ell = length_scale
    A = amplitude
    d = 3  # spatial dimension

    if kernel == "rbf":
        S = A**2 * (2.0 * np.pi * ell**2) ** (d / 2.0) * np.exp(-0.5 * ell**2 * k2)
    elif kernel == "matern32":
        nu = 1.5
        alpha2 = 2.0 * nu / ell**2
        # Normalisation: integral of S(k) over all k = A^2
        #   S(k) = A^2 * C_d * (alpha^2 + k^2)^{-(nu + d/2)}
        # where C_d is chosen so that (2pi)^{-d} int S(k) d^d k = A^2.
        p = nu + d / 2.0  # = 3.0
        from scipy.special import gamma as gammafn

        C = (
            A**2
            * (2.0**d * np.pi ** (d / 2.0) * gammafn(p))
            / gammafn(nu)
            * alpha2**nu
        )
        S = C * (alpha2 + k2) ** (-p)
    elif kernel == "matern52":
        nu = 2.5
        alpha2 = 2.0 * nu / ell**2
        p = nu + d / 2.0  # = 4.0
        from scipy.special import gamma as gammafn

        C = (
            A**2
            * (2.0**d * np.pi ** (d / 2.0) * gammafn(p))
            / gammafn(nu)
            * alpha2**nu
        )
        S = C * (alpha2 + k2) ** (-p)
    else:
        raise ValueError(f"Unknown kernel {kernel!r}.  Choose rbf/matern32/matern52.")

    return S


def deposition_window(
    shape: tuple[int, ...],
    dep: str = "NGP",
    box_size: float = 1.0,
) -> np.ndarray:
    r"""Squared Fourier-space deposition window :math:`|W(k)|^2`.

    For CIC (order 2), TSC (order 3), and PCS (order 4) the window
    function along each axis is :math:`[\sin(k h/2)/(k h/2)]^p`
    where p is the order and h is the cell size.  The 3-D window is
    the product over axes.

    Returns
    -------
    W2 : ndarray, same shape as a cube of size *shape*
        Values in [0, 1].  W2[0,0,0] = 1.
    """
    order = {"NGP": 1, "CIC": 2, "TSC": 3, "PCS": 4}[dep.upper()]
    W2 = np.ones(shape, dtype=np.float64)
    for ax, n in enumerate(shape):
        h = box_size / n
        k = fftfreq(n, d=box_size / n) * 2.0 * np.pi
        arg = k * h / 2.0
        with np.errstate(divide="ignore", invalid="ignore"):
            sinc = np.where(np.abs(arg) < 1e-12, 1.0, np.sin(arg) / arg)
        w1d = np.abs(sinc) ** order
        sl = [np.newaxis] * len(shape)
        sl[ax] = slice(None)
        W2 *= w1d[tuple(sl)] ** 2
    return W2


def _deposit_noise_factor(dep: str | None) -> float:
    r"""Compound Poisson noise variance factor for a deposition scheme.

    When particles are deposited onto a grid, each particle contributes a
    **random weight** to its cell (and possibly neighbors).  If particle
    positions are uniformly random within cells, these weights are random
    variables whose variance reduces the effective noise relative to pure
    Poisson (NGP) counting.

    **CIC weight PDF (3-D).**  A particle at fractional position
    :math:`(u,v,w)` deposits weight :math:`W = (1-u)(1-v)(1-w)` to
    its own cell, where :math:`u,v,w \sim \text{Uniform}(0,1)`.
    So :math:`W` is the product of three i.i.d. uniforms, with PDF

    .. math::

        f_W(w) = \tfrac{1}{2}(-\ln w)^2, \quad 0 < w \le 1

    (equivalently :math:`-\ln W \sim \text{Gamma}(3,1)`).

    A cell's deposit is a **compound Poisson** sum
    :math:`X = \sum_{i=1}^{N} W_i` where :math:`N` counts all
    particles in the :math:`p^3` surrounding cells and each :math:`W_i`
    has the same marginal distribution (by symmetry of the uniform).
    The variance ratio vs.\ Poisson is

    .. math::

        \alpha \;=\; \frac{\operatorname{Var}(X)}{\lambda}
               \;=\; \Bigl(\sum_{j=1}^{p} E[w_j^2]\Bigr)^{\!3}

    where the 1-D sum runs over the :math:`p` cells one particle
    contributes to in each dimension.

    ===== === ======================================== ===========
    Dep.   p   :math:`\sum_j E[w_j^2]` (1-D)           α (3-D)
    ===== === ======================================== ===========
    NGP    1   1                                        1.000
    CIC    2   2/3                                      8/27 ≈ 0.296
    TSC    3   11/20                                    0.166
    PCS    4   ≈ 0.4794                                 0.110
    ===== === ======================================== ===========

    Returns
    -------
    float
        Noise variance as a fraction of Poisson variance.
        Multiply the Poisson per-cell variance by this factor to
        obtain the compound Poisson per-cell variance.
    """
    if dep is None:
        return 1.0
    dep_upper = dep.upper()
    if dep_upper == "NGP":
        return 1.0
    if dep_upper == "CIC":
        # 1-D: E[(1-U)^2] + E[U^2] = 1/3 + 1/3 = 2/3
        return (2.0 / 3.0) ** 3  # 8/27
    if dep_upper == "TSC":
        # 1-D: E[w_0^2] + E[w_+^2] + E[w_-^2] = 9/20 + 1/20 + 1/20 = 11/20
        return (11.0 / 20.0) ** 3
    if dep_upper == "PCS":
        # Cubic B-spline; 1-D sum numerically integrated = 0.47937
        return 0.47937 ** 3
    return 1.0


def empirical_power_spectrum(
    cube: np.ndarray,
    n_eff: float,
    box_size: float = 1.0,
    dep: str | None = None,
    psd_floor: str = "flat",
    floor_snr: float = 0.01,
) -> np.ndarray:
    r"""Non-parametric signal power spectrum estimated from the data.

    For turbulent density fields the true power spectrum is a broad
    power-law; no single-length-scale parametric kernel can capture it.
    This function estimates S(k) directly:

    1. Compute the raw 3-D periodogram :math:`|\hat{y}(k)|^2 / N`.
    2. Compute the noise floor :math:`N(k) = 1/n_{\rm eff}`
       (optionally shaped by the deposition window :math:`|W(k)|^2`).
    3. Bin the periodogram and noise into isotropic radial shells using
       the **mean** (unbiased for the power spectrum; the periodogram
       follows a chi-squared distribution whose median is 0.69x the mean,
       so using median would bias S(k) low and cause over-filtering).
    4. Subtract the binned noise to get the signal power per shell.
    5. Interpolate back onto the full 3-D FFT grid.

    Parameters
    ----------
    cube : ndarray (nx, ny, nz)
        Observed density field.
    n_eff : float
        Mean particles per cell at the mean density.
    box_size : float
    dep : str, optional
        Deposition scheme for noise window correction.
    psd_floor : ``'powerlaw'`` or ``'flat'``
        How to regularize the signal PSD at high k where noise dominates.
        ``'powerlaw'`` fits a power law to the high-SNR low-k bins and
        extrapolates.  ``'flat'`` (default) sets the floor to
        ``noise_binned * floor_snr``.
    floor_snr : float
        Minimum assumed signal-to-noise ratio per radial bin (used by
        the ``'flat'`` floor).

    Returns
    -------
    S : ndarray (nx, ny, nz)
        Estimated signal power spectrum on the FFT grid.  Always >= a
        small positive floor so the GP prior is well-defined everywhere.
    """
    shape = cube.shape
    rho_mean = float(np.mean(cube))
    if rho_mean <= 0:
        rho_mean = 1.0

    delta = cube / rho_mean - 1.0
    delta_k = fftn(delta)
    n_total = np.prod(shape)
    power_3d = np.abs(delta_k) ** 2 / n_total

    k2 = _build_k2_grid(shape, box_size)
    k_mag = np.sqrt(k2)

    noise_floor = 1.0 / n_eff
    if dep is not None:
        noise_3d = noise_floor * deposition_window(shape, dep, box_size)
    else:
        noise_3d = np.full(shape, noise_floor)

    k_flat = k_mag.ravel()
    pow_flat = power_3d.ravel()
    noise_flat = noise_3d.ravel()
    nonzero = k_flat > 0
    k_nz = k_flat[nonzero]

    n_bins = max(min(shape) // 2, 16)
    k_edges = np.linspace(0, k_nz.max(), n_bins + 1)
    k_centers = 0.5 * (k_edges[:-1] + k_edges[1:])

    pow_binned = np.zeros(n_bins)
    noise_binned = np.zeros(n_bins)
    for i in range(n_bins):
        mask = nonzero & (k_flat >= k_edges[i]) & (k_flat < k_edges[i + 1])
        if mask.any():
            pow_binned[i] = np.mean(pow_flat[mask])
            noise_binned[i] = np.mean(noise_flat[mask])

    sig_raw = pow_binned - noise_binned
    if psd_floor == "powerlaw":
        sig_binned = _powerlaw_floor(k_centers, sig_raw, noise_binned, pow_binned)
    else:
        sig_binned = np.maximum(sig_raw, noise_binned * floor_snr)
        sig_binned = np.maximum(sig_binned, 1e-30)

    S_flat = np.interp(k_flat, k_centers, sig_binned)
    S_3d = S_flat.reshape(shape)

    S_3d.flat[0] = S_flat[nonzero].max() * 10.0

    return np.maximum(S_3d, 1e-30)


def estimate_kernel_params(
    cube: np.ndarray,
    n_eff: float,
    kernel: str = "matern32",
    box_size: float = 1.0,
    dep: str | None = None,
) -> dict:
    """Estimate kernel length_scale and amplitude from the observed power spectrum.

    Algorithm
    ---------
    1. Compute the isotropic power spectrum of the density fluctuation field.
    2. Estimate the white-noise floor from the Poisson shot noise level
       ``N = rho_mean / n_eff`` (in density units) which gives a flat power
       spectrum contribution.  If a deposition window is specified, the noise
       power is ``N * |W(k)|^2``.
    3. Subtract the noise floor to get the signal power spectrum.
    4. Fit the parametric kernel S(k) to the signal power by least-squares
       in log-space over the radial bins.

    Returns
    -------
    dict with keys ``'length_scale'``, ``'amplitude'``, ``'noise_floor'``.
    """
    shape = cube.shape
    rho_mean = np.mean(cube)
    if rho_mean <= 0:
        rho_mean = 1.0

    delta = cube / rho_mean - 1.0
    delta_k = fftn(delta)
    N = np.prod(shape)
    power_3d = np.abs(delta_k) ** 2 / N

    k2 = _build_k2_grid(shape, box_size)
    k_mag = np.sqrt(k2).ravel()
    power_flat = power_3d.ravel()
    nonzero = k_mag > 0
    k_mag = k_mag[nonzero]
    power_flat = power_flat[nonzero]

    # Noise PSD of the fluctuation field: Var(delta_i) = 1/n_eff
    noise_floor = 1.0 / n_eff
    if dep is not None:
        W2 = deposition_window(shape, dep, box_size)
        noise_flat = (noise_floor * W2.ravel()[nonzero])
    else:
        noise_flat = noise_floor

    signal_flat = np.maximum(power_flat - noise_flat, 1e-30)

    n_bins = min(shape) // 2
    k_edges = np.linspace(0, k_mag.max(), n_bins + 1)
    k_centers = 0.5 * (k_edges[:-1] + k_edges[1:])
    signal_binned = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (k_mag >= k_edges[i]) & (k_mag < k_edges[i + 1])
        if mask.any():
            signal_binned[i] = np.mean(signal_flat[mask])
        else:
            signal_binned[i] = 1e-30

    valid = signal_binned > 1e-20
    if valid.sum() < 3:
        return {"length_scale": 2.0, "amplitude": 1.0, "noise_floor": noise_floor}

    k_fit = k_centers[valid]
    s_fit = signal_binned[valid]
    log_s = np.log(s_fit)
    weights = np.ones_like(log_s)

    def _objective(log_ell):
        ell = np.exp(log_ell)
        S_model = kernel_power_spectrum(
            k_fit**2, kernel=kernel, length_scale=ell, amplitude=1.0
        )
        log_S = np.log(np.maximum(S_model, 1e-30))
        offset = np.average(log_s - log_S, weights=weights)
        residual = (log_s - log_S - offset) ** 2
        return np.average(residual, weights=weights)

    # Grid search over plausible length scales (0.5 to 20 cells)
    best_loss = np.inf
    best_log_ell = np.log(2.0)
    for trial_ell in np.logspace(np.log10(0.5), np.log10(20.0), 40):
        loss = _objective(np.log(trial_ell))
        if loss < best_loss:
            best_loss = loss
            best_log_ell = np.log(trial_ell)

    res = minimize_scalar(
        _objective,
        bounds=(best_log_ell - 1.0, best_log_ell + 1.0),
        method="bounded",
    )
    ell_opt = np.exp(res.x)

    S_unit = kernel_power_spectrum(k_fit**2, kernel=kernel, length_scale=ell_opt, amplitude=1.0)
    log_S_unit = np.log(np.maximum(S_unit, 1e-30))
    log_amp2 = np.average(log_s - log_S_unit, weights=weights)
    amp_opt = np.sqrt(np.exp(log_amp2))

    return {
        "length_scale": float(ell_opt),
        "amplitude": float(amp_opt),
        "noise_floor": float(noise_floor),
    }


# ===================================================================
#  Noise models
# ===================================================================

class NoiseModel(ABC):
    """Abstract base for observation noise models."""

    @abstractmethod
    def variance(self, field: np.ndarray) -> np.ndarray:
        """Per-cell noise variance (may depend on the current field estimate)."""

    @abstractmethod
    def log_likelihood(self, observed: np.ndarray, predicted: np.ndarray) -> float:
        """Scalar log-likelihood of *observed* given *predicted* (up to const)."""

    @abstractmethod
    def grad_neg_log_lik(self, observed: np.ndarray, f: np.ndarray) -> np.ndarray:
        r"""Gradient of -log p(y|f) w.r.t. the latent field *f*.

        For Poisson with log-link, f = log(lambda) and the gradient is
        ``exp(f) - y``.  For Gaussian, f is the mean and the gradient is
        ``(f - y) / sigma^2``.
        """

    @abstractmethod
    def hessian_diag(self, f: np.ndarray) -> np.ndarray:
        r"""Diagonal of the Hessian of -log p(y|f) w.r.t. *f*.

        For Poisson: ``exp(f)``.  For Gaussian: ``1/sigma^2``.
        """


@dataclass
class PoissonNoise(NoiseModel):
    """Poisson noise model for particle count data.

    The latent field *f* represents log-counts:  ``n ~ Poisson(exp(f))``.

    Parameters
    ----------
    n_eff : float
        Mean number of tracer/dust particles per cell at the mean density.
        Used to convert a density cube to effective counts via
        ``counts = density * n_eff / mean(density)``.
    """

    n_eff: float = 16.0

    def density_to_counts(self, rho: np.ndarray) -> np.ndarray:
        rho_mean = np.mean(rho)
        if rho_mean <= 0:
            rho_mean = 1.0
        return rho * self.n_eff / rho_mean

    def counts_to_density(self, counts: np.ndarray, rho_mean: float) -> np.ndarray:
        return counts * rho_mean / self.n_eff

    def variance(self, field: np.ndarray) -> np.ndarray:
        return np.exp(field)

    def log_likelihood(self, observed: np.ndarray, f: np.ndarray) -> float:
        lam = np.exp(f)
        return float(np.sum(observed * f - lam))

    def grad_neg_log_lik(self, observed: np.ndarray, f: np.ndarray) -> np.ndarray:
        return np.exp(f) - observed

    def hessian_diag(self, f: np.ndarray) -> np.ndarray:
        return np.exp(f)


@dataclass
class GaussianNoise(NoiseModel):
    """Homoscedastic Gaussian noise.

    Parameters
    ----------
    variance_val : float
        Constant noise variance.
    """

    variance_val: float = 1.0

    def variance(self, field: np.ndarray) -> np.ndarray:
        return np.full_like(field, self.variance_val)

    def log_likelihood(self, observed: np.ndarray, f: np.ndarray) -> float:
        return float(-0.5 * np.sum((observed - f) ** 2) / self.variance_val)

    def grad_neg_log_lik(self, observed: np.ndarray, f: np.ndarray) -> np.ndarray:
        return (f - observed) / self.variance_val

    def hessian_diag(self, f: np.ndarray) -> np.ndarray:
        return np.full_like(f, 1.0 / self.variance_val)


@dataclass
class GaussianHeteroscedastic(NoiseModel):
    """Heteroscedastic Gaussian noise with a per-cell variance cube.

    Parameters
    ----------
    variance_cube : ndarray
        Per-cell variance (must match the cube shape).
    """

    variance_cube: np.ndarray = field(default_factory=lambda: np.ones(1))

    def variance(self, field: np.ndarray) -> np.ndarray:
        return self.variance_cube

    def log_likelihood(self, observed: np.ndarray, f: np.ndarray) -> float:
        return float(-0.5 * np.sum((observed - f) ** 2 / self.variance_cube))

    def grad_neg_log_lik(self, observed: np.ndarray, f: np.ndarray) -> np.ndarray:
        return (f - observed) / self.variance_cube

    def hessian_diag(self, f: np.ndarray) -> np.ndarray:
        return 1.0 / self.variance_cube


# ===================================================================
#  Wiener filter
# ===================================================================

def wiener_denoise(
    cube: np.ndarray,
    kernel_spectrum: np.ndarray,
    noise_spectrum: np.ndarray | float,
) -> dict:
    r"""FFT Wiener filter — exact GP posterior mean for Gaussian noise.

    Computes

    .. math::

        \hat{f}(k) = \frac{S(k)}{S(k) + N(k)} \, \hat{y}(k)

    where S(k) is the prior signal power spectrum and N(k) the noise power.

    For Poisson noise the Gaussian approximation sets
    ``N(k) = rho_mean / n_eff`` (white).

    Parameters
    ----------
    cube : ndarray (nx, ny, nz)
        Observed density field (or fluctuation field delta = rho/rho_mean - 1).
    kernel_spectrum : ndarray (nx, ny, nz)
        Prior signal power spectrum S(k) on the FFT grid.
    noise_spectrum : ndarray or float
        Noise power spectrum N(k).  A scalar is broadcast to all modes.

    Returns
    -------
    dict with:
        - ``'denoised'``: posterior mean (real-space cube).
        - ``'variance'``: posterior variance estimate per cell.
        - ``'filter'``: the Wiener filter array W(k).
    """
    y_hat = fftn(cube)
    S = kernel_spectrum
    N = noise_spectrum

    W = S / (S + N)
    # DC mode: pass through (the mean is not "noise")
    W.flat[0] = 1.0

    f_hat = W * y_hat
    denoised = np.real(ifftn(f_hat))

    # Posterior variance: diag of (K^{-1} + Sigma^{-1})^{-1} in Fourier domain
    # = S * N / (S + N) per mode; transform back for per-cell variance
    var_spectrum = S * N / (S + N)
    var_spectrum.flat[0] = 0.0
    n_total = np.prod(cube.shape)
    variance = np.real(ifftn(np.ones_like(cube) * var_spectrum)) / n_total

    return {"denoised": denoised, "variance": variance, "filter": W}


# ===================================================================
#  Gaussian SNR smoothing
# ===================================================================

def gaussian_snr_denoise(
    cube: np.ndarray,
    n_eff: float,
    box_size: float = 1.0,
    scale: float = 2.0,
) -> dict:
    r"""Smooth with a Gaussian kernel whose width is set by the signal-to-noise ratio.

    Uses a single global Gaussian in Fourier space:
    :math:`G(k) = \exp(-k^2 \sigma^2/2)` with
    :math:`\sigma = \mathrm{scale} / \sqrt{n_\mathrm{eff}}` in grid cells.
    Lower n_eff (more noise) → wider Gaussian (more smoothing).

    The filter is linear and the same G(k) can be applied to gas for
    comparable tracer–gas plots.

    Parameters
    ----------
    cube : ndarray (nx, ny, nz)
        Density field (e.g. tracer or gas).
    n_eff : float
        Mean particles per cell at mean density (sets noise level).
    box_size : float
        Physical box size (default 1.0).
    scale : float
        Multiplier for sigma; sigma_cells = scale / sqrt(n_eff). Default 2.

    Returns
    -------
    dict with ``denoised``, ``variance`` (approximate), ``filter`` (G(k)).
    """
    shape = cube.shape
    nx = shape[0]
    n_eff = max(float(n_eff), 0.5)
    sigma_cells = scale / np.sqrt(n_eff)
    dx = box_size / nx
    sigma_phys = sigma_cells * dx

    k2 = _build_k2_grid(shape, box_size)
    k_mag = np.sqrt(np.maximum(k2, 0.0))
    G = np.exp(-0.5 * (k_mag * sigma_phys) ** 2).astype(np.float64)
    G.flat[0] = 1.0

    cube_hat = fftn(cube)
    denoised = np.real(ifftn(G * cube_hat))

    # Approximate posterior variance: (1 - G^2) * noise_psd per mode, negligible for display
    noise_psd = 1.0 / n_eff
    var_spectrum = (1.0 - G * G) * noise_psd
    var_spectrum.flat[0] = 0.0
    n_total = np.prod(shape)
    variance = np.real(ifftn(np.ones(shape) * var_spectrum)) / n_total

    return {"denoised": denoised, "variance": variance, "filter": G}


# ===================================================================
#  Log-space Wiener filter
# ===================================================================

def _powerlaw_floor(
    k_centers: np.ndarray,
    sig_raw: np.ndarray,
    noise_binned: np.ndarray,
    pow_binned: np.ndarray,
    snr_threshold: float = 2.0,
) -> np.ndarray:
    """Regularize the noise-subtracted signal PSD with a power-law model.

    At high k the noise-subtracted PSD becomes unreliable (negative or
    dominated by noise).  This fits a power law ``S(k) = C * k^b`` to the
    high-SNR low-k bins and smoothly blends between the data and the
    model based on per-bin SNR.  Where SNR is high the data is trusted;
    where it is low the power-law extrapolation takes over.

    The blend weight per bin is ``w = clip(SNR / snr_threshold, 0, 1)``,
    giving ``sig = w * sig_raw + (1 - w) * sig_model``.  This avoids
    both the over-suppression of a flat floor and the noise leakage of
    a hard ``max(sig_raw, sig_model)``.

    Parameters
    ----------
    k_centers : ndarray (n_bins,)
        Bin centres in wavenumber.
    sig_raw : ndarray (n_bins,)
        ``pow_binned - noise_binned`` (may be negative).
    noise_binned : ndarray (n_bins,)
        Mean noise power per bin.
    pow_binned : ndarray (n_bins,)
        Mean observed power per bin (used for flat-floor fallback).
    snr_threshold : float
        SNR level at which the blend fully trusts the data.

    Returns
    -------
    sig_reg : ndarray (n_bins,)
        Regularized signal PSD, always > 0.
    """
    trusted = (sig_raw > 0) & (sig_raw > snr_threshold * noise_binned) & (k_centers > 0)

    if np.sum(trusted) < 3:
        sig_reg = np.maximum(sig_raw, pow_binned * 1e-3)
        return np.maximum(sig_reg, 1e-30)

    log_k = np.log(k_centers[trusted])
    log_s = np.log(sig_raw[trusted])
    b, a = np.polyfit(log_k, log_s, 1)
    b = np.clip(b, -10.0, -0.5)

    k_min_pos = k_centers[k_centers > 0].min()
    sig_model = np.exp(a + b * np.log(np.maximum(k_centers, k_min_pos)))

    snr = np.where(noise_binned > 0, sig_raw / noise_binned, 0.0)
    weight = np.clip(snr / snr_threshold, 0.0, 1.0)
    sig_reg = weight * np.maximum(sig_raw, 1e-30) + (1.0 - weight) * sig_model
    return np.maximum(sig_reg, 1e-30)


def _estimate_signal_psd(
    field: np.ndarray,
    noise_psd: np.ndarray | float,
    box_size: float = 1.0,
    psd_floor: str = "flat",
    floor_snr: float = 0.01,
) -> np.ndarray:
    """Estimate signal power spectrum by subtracting noise from the periodogram.

    Uses mean-binned isotropic radial shells (unbiased for chi-squared
    periodogram).  Shared helper for both linear and log-space filters.

    Parameters
    ----------
    field : ndarray (nx, ny, nz)
        Zero-mean field whose PSD to estimate.
    noise_psd : ndarray or float
        Noise power spectrum (same shape or scalar).
    box_size : float
    psd_floor : ``'powerlaw'`` or ``'flat'``
        How to regularize the signal PSD at high k where noise dominates.
        ``'powerlaw'`` fits a power law to the high-SNR low-k bins and
        extrapolates.  ``'flat'`` (default) sets the floor to
        ``noise_binned * floor_snr``.
    floor_snr : float
        Minimum assumed signal-to-noise ratio per radial bin (used by
        the ``'flat'`` floor).  The Wiener weight at noise-dominated
        scales is approximately ``floor_snr``.

    Returns
    -------
    S : ndarray (nx, ny, nz)
        Estimated signal PSD, always > 0.
    """
    shape = field.shape
    n_total = np.prod(shape)
    power_3d = np.abs(fftn(field)) ** 2 / n_total

    k2 = _build_k2_grid(shape, box_size)
    k_flat = np.sqrt(k2).ravel()
    pow_flat = power_3d.ravel()
    noise_flat = (noise_psd * np.ones(shape)).ravel() if np.ndim(noise_psd) == 0 else noise_psd.ravel()
    nonzero = k_flat > 0

    n_bins = max(min(shape) // 2, 16)
    k_edges = np.linspace(0, k_flat[nonzero].max(), n_bins + 1)
    k_centers = 0.5 * (k_edges[:-1] + k_edges[1:])

    pow_binned = np.zeros(n_bins)
    noise_binned = np.zeros(n_bins)
    for i in range(n_bins):
        mask = nonzero & (k_flat >= k_edges[i]) & (k_flat < k_edges[i + 1])
        if mask.any():
            pow_binned[i] = np.mean(pow_flat[mask])
            noise_binned[i] = np.mean(noise_flat[mask])

    sig_raw = pow_binned - noise_binned
    if psd_floor == "powerlaw":
        sig_binned = _powerlaw_floor(k_centers, sig_raw, noise_binned, pow_binned)
    else:
        sig_binned = np.maximum(sig_raw, noise_binned * floor_snr)
        sig_binned = np.maximum(sig_binned, 1e-30)

    S_flat = np.interp(k_flat, k_centers, sig_binned)
    S_3d = S_flat.reshape(shape)
    S_3d.flat[0] = S_flat[nonzero].max() * 10.0
    return np.maximum(S_3d, 1e-30)


def log_wiener_denoise(
    cube: np.ndarray,
    n_eff: float,
    box_size: float = 1.0,
    dep: str | None = None,
    deconvolve: bool = False,
    n_iter: int = 3,
    tol: float = 1e-5,
    n_strata: int = 3,
    gas_ref: np.ndarray | None = None,
    psd_floor: str = "flat",
    floor_snr: float = 0.01,
    noise_model: str = "poisson",
    verbose: bool = False,
) -> dict:
    r"""Iterative Wiener filter in log-density space.

    Turbulent density fields are approximately log-normal, so
    ``s = \ln(\rho / \bar\rho)`` is nearly Gaussian — the regime where
    the Wiener filter is optimal.

    Poisson noise in log-space is approximately additive Gaussian with
    per-cell variance ``1/\lambda_i`` (delta-method).  Because the noise
    is heteroscedastic, the filter iterates: each pass updates the noise
    PSD using the current signal estimate, then re-applies the Wiener
    filter.

    Zero-count cells are handled with a continuity correction
    ``n \to \max(n, 0.5)`` and receive high noise variance, so the filter
    relies on neighboring information for those cells.

    Parameters
    ----------
    cube : ndarray (nx, ny, nz)
        Observed density field.
    n_eff : float
        Mean particles per cell at the mean density.
    box_size : float
    dep : str, optional
        Deposition scheme.  Used for noise PSD shaping (``'poisson'``
        model) or to compute the compound Poisson variance factor
        (``'compound_poisson'`` model).
    deconvolve : bool
        If True and *dep* is set, apply a post-hoc Wiener deconvolution
        to undo the deposition window smoothing after the log-space filter.
    n_iter : int
        Maximum number of filter iterations (default 3).  The loop stops
        early if the relative change in the log-density estimate falls
        below *tol*.
    tol : float
        Convergence tolerance: stop when
        ‖s_est_new − s_est_old‖ / ‖s_est_old‖ < tol (default 1e-5).
    n_strata : int
        Number of density strata for heteroscedastic noise handling.
        ``n_strata=3`` (default) splits cells into equal-count density
        quantiles and applies a separate Wiener filter per stratum, so
        high-density (low-noise) cells are filtered less aggressively than
        low-density (high-noise) cells.  ``n_strata=1`` uses a single
        global noise level (original algorithm).
    noise_model : ``'poisson'`` or ``'compound_poisson'``
        ``'poisson'`` (default) uses per-cell variance ``1/λ`` shaped
        by the deposition window ``|W(k)|²`` in Fourier space.
        ``'compound_poisson'`` uses per-cell variance ``α/λ`` where
        ``α`` is the deposit noise factor (see `_deposit_noise_factor`);
        the noise PSD is white (no deposition window shaping), since
        the reduced variance already accounts for the deposition.
        Both models are spectrally equivalent for the standard case.
    verbose : bool

    Returns
    -------
    dict with:
        - ``'denoised'``: denoised density cube.
        - ``'variance'``: posterior variance of ln(rho) per cell.
        - ``'n_iter'``: iterations performed.
    """
    shape = cube.shape
    rho_mean = float(np.mean(cube))
    if rho_mean <= 0:
        rho_mean = 1.0

    counts = cube * (n_eff / rho_mean)
    counts_safe = np.maximum(counts, 0.5)

    s_obs = np.log(counts_safe / n_eff)
    mu_s = float(np.mean(s_obs))
    s_centered = s_obs - mu_s

    s_est = s_obs.copy()

    use_compound = noise_model == "compound_poisson"
    alpha = _deposit_noise_factor(dep) if use_compound else 1.0

    # For "poisson" model, shape noise with the deposition window |W(k)|^2.
    # For "compound_poisson" or NGP, noise is white (factor already in alpha).
    _dep_for_window = dep if (not use_compound and dep is not None
                              and dep.upper() != "NGP") else None
    dep_window = (deposition_window(shape, _dep_for_window, box_size)
                  if _dep_for_window is not None else None)

    if verbose and use_compound:
        print(f"  Compound Poisson noise factor α={alpha:.4f} "
              f"(dep={dep})")

    n_iter_done = 0
    for it in range(n_iter):
        s_est_prev = s_est.copy()
        lambda_est = n_eff * np.exp(s_est)
        sigma2 = alpha / np.maximum(lambda_est, 0.5)

        noise_psd_global = float(np.mean(sigma2))
        if dep_window is not None:
            noise_psd_for_S = noise_psd_global * dep_window
        else:
            noise_psd_for_S = noise_psd_global
        S = _estimate_signal_psd(s_centered, noise_psd_for_S, box_size, psd_floor=psd_floor, floor_snr=floor_snr)

        s_hat = fftn(s_centered)

        if n_strata <= 1:
            # Original scalar-mean path
            noise_psd = noise_psd_for_S
            W = S / (S + noise_psd)
            W_arr = W if np.ndim(W) > 0 else np.full(shape, W)
            W_arr.flat[0] = 1.0
            s_centered_filt = np.real(ifftn(W_arr * s_hat))

            if verbose:
                print(f"  log_wiener iter {it}: N_psd={noise_psd_global:.4f}, "
                      f"mean(W)={float(np.mean(W_arr)):.3f}")
        else:
            # Density-stratified path: different noise level per stratum
            quantile_edges = np.linspace(0, 1, n_strata + 1)
            thresholds = np.quantile(lambda_est, quantile_edges)
            s_centered_filt = np.empty_like(s_centered)

            if verbose:
                stratum_info = []

            for j in range(n_strata):
                lo, hi = thresholds[j], thresholds[j + 1]
                if j == 0:
                    mask = lambda_est < hi
                elif j == n_strata - 1:
                    mask = lambda_est >= lo
                else:
                    mask = (lambda_est >= lo) & (lambda_est < hi)

                N_j = float(np.mean(sigma2[mask]))
                if dep_window is not None:
                    noise_j = N_j * dep_window
                else:
                    noise_j = N_j
                W_j = S / (S + noise_j)
                W_j_arr = W_j if np.ndim(W_j) > 0 else np.full(shape, W_j)
                W_j_arr.flat[0] = 1.0

                filt_j = np.real(ifftn(W_j_arr * s_hat))
                s_centered_filt[mask] = filt_j[mask]

                if verbose:
                    stratum_info.append(
                        f"s{j}(N={N_j:.4f},W={float(np.mean(W_j_arr)):.3f},"
                        f"n={int(mask.sum())})"
                    )

            if verbose:
                print(f"  log_wiener iter {it}: strata=[{', '.join(stratum_info)}]")

        s_est = mu_s + s_centered_filt
        n_iter_done = it + 1

        # Convergence: relative change in log-density estimate
        if it > 0:
            norm_prev = np.sqrt(np.mean(s_est_prev**2)) + 1e-30
            rel_change = np.sqrt(np.mean((s_est - s_est_prev) ** 2)) / norm_prev
            if rel_change < tol:
                if verbose:
                    print(f"  Converged at iter {n_iter_done}: rel_change={rel_change:.2e} < {tol}")
                break

    # Posterior variance in log-space: S*N/(S+N) per mode, averaged to real space
    # Use the global noise PSD for the variance estimate
    noise_psd_final = noise_psd_for_S
    var_spectrum = S * noise_psd_final / (S + noise_psd_final)
    if np.ndim(var_spectrum) == 0:
        var_spectrum = np.full(shape, var_spectrum)
    var_spectrum.flat[0] = 0.0
    n_total = np.prod(shape)
    variance = np.real(ifftn(np.ones(shape) * var_spectrum)) / n_total

    # Global Wiener filter (same transfer function) for applying to other fields (e.g. gas)
    W_global = S / (S + noise_psd_final)
    if np.ndim(W_global) == 0:
        W_global = np.full(shape, float(W_global))
    W_global = np.asarray(W_global, dtype=np.float64)
    W_global.flat[0] = 1.0

    denoised_rho = rho_mean * np.exp(s_est)

    if deconvolve and dep is not None:
        if verbose:
            print("  Applying post-hoc Wiener deconvolution ...")
        denoised_rho = _wiener_deconvolve(
            denoised_rho, dep=dep,
            residual_noise_psd=var_spectrum,
            box_size=box_size,
            verbose=verbose,
        )

    if gas_ref is not None:
        if verbose:
            print("  Applying cross-power spectrum calibration ...")
        denoised_rho = _calibrate_power_spectrum(
            denoised_rho, gas_ref,
            box_size=box_size,
            verbose=verbose,
        )

    return {"denoised": denoised_rho, "variance": variance, "n_iter": n_iter_done, "filter": W_global}


def _wiener_deconvolve(
    cube: np.ndarray,
    dep: str,
    residual_noise_psd: np.ndarray | float,
    box_size: float = 1.0,
    verbose: bool = False,
) -> np.ndarray:
    r"""Post-hoc Wiener deconvolution of the deposition window.

    Applies the standard Wiener deconvolution filter in density-contrast
    space to undo the smoothing introduced by the particle deposition
    kernel H(k):

    .. math::

        W(k) = \frac{H^*(k)\,S(k)}{|H(k)|^2\,S(k) + N(k)}

    where S(k) is the signal power estimated from the (already denoised)
    input and N(k) is the residual noise power after the log-space filter.

    Parameters
    ----------
    cube : ndarray (nx, ny, nz)
        Denoised density field (output of the log-space Wiener filter).
    dep : str
        Deposition scheme (``'CIC'``, ``'TSC'``, ``'PCS'``).
    residual_noise_psd : ndarray or float
        Residual noise power spectrum after denoising (from the posterior
        variance spectrum).
    box_size : float
    verbose : bool

    Returns
    -------
    deconvolved : ndarray (nx, ny, nz)
        Deconvolved density field.
    """
    shape = cube.shape
    rho_mean = float(np.mean(cube))
    if rho_mean <= 0:
        rho_mean = 1.0

    delta = cube / rho_mean - 1.0

    W2 = deposition_window(shape, dep, box_size)
    H = np.sqrt(W2)

    S = _estimate_signal_psd(delta, residual_noise_psd, box_size)

    N = residual_noise_psd
    if np.ndim(N) == 0:
        N = np.full(shape, float(N))

    denom = W2 * S + N
    denom = np.maximum(denom, 1e-30)
    W_deconv = H * S / denom
    W_deconv.flat[0] = 1.0

    delta_k = fftn(delta)
    deconv_delta = np.real(ifftn(W_deconv * delta_k))
    deconvolved = rho_mean * (1.0 + deconv_delta)
    deconvolved = np.maximum(deconvolved, rho_mean * 1e-6)

    if verbose:
        mean_gain = float(np.mean(W_deconv))
        print(f"    Deconvolution mean filter gain: {mean_gain:.3f}")

    return deconvolved


def _calibrate_power_spectrum(
    denoised: np.ndarray,
    gas_ref: np.ndarray,
    box_size: float = 1.0,
    t_clamp: tuple[float, float] = (0.5, 4.0),
    verbose: bool = False,
) -> np.ndarray:
    r"""Post-hoc transfer function calibration against a gas reference.

    Adjusts Fourier amplitudes of the denoised field so its isotropic
    power spectrum matches the gas reference.  Phases are preserved —
    only the per-k amplitude is rescaled.

    Given P_den(k) and P_gas(k) (isotropically binned), the correction is

    .. math::

        T(k) = \text{clamp}\!\bigl(P_{\rm gas}(k) / P_{\rm den}(k),\;
               T_{\min},\; T_{\max}\bigr)

    and the calibrated Fourier coefficients are multiplied by
    :math:`\sqrt{T(k)}`.

    Parameters
    ----------
    denoised : ndarray (nx, ny, nz)
        Denoised density cube.
    gas_ref : ndarray (nx, ny, nz)
        Reference gas density cube (same shape).
    box_size : float
    t_clamp : tuple of float
        (T_min, T_max) range to which the transfer function is clamped.
    verbose : bool

    Returns
    -------
    calibrated : ndarray (nx, ny, nz)
        Calibrated density cube.
    """
    shape = denoised.shape
    den_mean = float(np.mean(denoised))
    gas_mean = float(np.mean(gas_ref))
    if den_mean <= 0:
        den_mean = 1.0
    if gas_mean <= 0:
        gas_mean = 1.0

    delta_den = denoised / den_mean - 1.0
    delta_gas = gas_ref / gas_mean - 1.0

    n_total = np.prod(shape)
    P_den = np.abs(fftn(delta_den)) ** 2 / n_total
    P_gas = np.abs(fftn(delta_gas)) ** 2 / n_total

    k2 = _build_k2_grid(shape, box_size)
    k_flat = np.sqrt(k2).ravel()
    nonzero = k_flat > 0

    n_bins = max(min(shape) // 2, 16)
    k_edges = np.linspace(0, k_flat[nonzero].max(), n_bins + 1)
    k_centers = 0.5 * (k_edges[:-1] + k_edges[1:])

    P_den_binned = np.zeros(n_bins)
    P_gas_binned = np.zeros(n_bins)
    for i in range(n_bins):
        mask = nonzero & (k_flat >= k_edges[i]) & (k_flat < k_edges[i + 1])
        if mask.any():
            P_den_binned[i] = np.mean(P_den.ravel()[mask])
            P_gas_binned[i] = np.mean(P_gas.ravel()[mask])

    T_binned = P_gas_binned / np.maximum(P_den_binned, 1e-30)
    T_binned = np.clip(T_binned, t_clamp[0], t_clamp[1])

    sqrt_T_flat = np.sqrt(np.interp(k_flat, k_centers, T_binned))
    sqrt_T_3d = sqrt_T_flat.reshape(shape)
    sqrt_T_3d.flat[0] = 1.0

    delta_den_k = fftn(delta_den)
    cal_delta = np.real(ifftn(sqrt_T_3d * delta_den_k))
    calibrated = den_mean * (1.0 + cal_delta)
    calibrated = np.maximum(calibrated, den_mean * 1e-6)

    if verbose:
        mean_T = float(np.mean(T_binned))
        print(f"    Power spectrum calibration: mean T(k)={mean_T:.3f}, "
              f"range=[{T_binned.min():.3f}, {T_binned.max():.3f}]")

    return calibrated


# ===================================================================
#  Laplace GP with Poisson likelihood
# ===================================================================

def _cg_solve(
    matvec,
    rhs: np.ndarray,
    x0: np.ndarray | None = None,
    tol: float = 1e-6,
    maxiter: int = 200,
) -> np.ndarray:
    """Conjugate gradient solver for a symmetric positive-definite system.

    Parameters
    ----------
    matvec : callable
        ``matvec(v)`` returns A @ v for the linear system A x = rhs.
    rhs : ndarray
    x0 : ndarray, optional
        Initial guess (zeros if omitted).
    tol : float
        Relative residual tolerance.
    maxiter : int

    Returns
    -------
    x : ndarray
    """
    if x0 is None:
        x = np.zeros_like(rhs)
    else:
        x = x0.copy()
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


def laplace_gp_denoise(
    counts: np.ndarray,
    kernel_spectrum: np.ndarray,
    noise_model: NoiseModel | None = None,
    prior_mean: float | None = None,
    max_newton: int = 30,
    newton_tol: float = 1e-5,
    cg_tol: float = 1e-6,
    cg_maxiter: int = 200,
    n_variance_probes: int = 30,
    verbose: bool = False,
) -> dict:
    r"""Laplace-approximation GP posterior for non-Gaussian likelihoods.

    Finds the posterior mode :math:`f^*` of

    .. math::

        \log p(f \mid y) \;\propto\;
        \sum_i \bigl[y_i f_i - e^{f_i}\bigr]
        - \tfrac12 (f - \mu)^\top K^{-1} (f - \mu)

    via Newton's method.  Each Newton step solves

    .. math::

        (W + K^{-1})\,\Delta f = \nabla

    with conjugate gradient, where the :math:`K^{-1} v` matvec is
    one FFT round-trip.

    Parameters
    ----------
    counts : ndarray (nx, ny, nz)
        Observed counts (non-negative; need not be integer for CIC/TSC).
    kernel_spectrum : ndarray (nx, ny, nz)
        Prior power spectrum S(k), matching the FFT grid.
    noise_model : NoiseModel, optional
        Defaults to ``PoissonNoise()``.
    prior_mean : float, optional
        Prior mean of the latent field.  Defaults to ``log(mean(counts))``.
    max_newton : int
    newton_tol : float
        Convergence threshold on the relative gradient norm.
    cg_tol, cg_maxiter : float, int
        Conjugate gradient parameters.
    n_variance_probes : int
        Number of random probing vectors for diagonal variance estimation.
    verbose : bool

    Returns
    -------
    dict with:
        - ``'mode'``: posterior mode of f (log-counts).
        - ``'denoised_counts'``: exp(mode), the MAP count estimate.
        - ``'variance'``: approximate posterior variance of f per cell.
        - ``'n_newton'``: number of Newton iterations used.
    """
    if noise_model is None:
        noise_model = PoissonNoise()

    shape = counts.shape
    y = counts.astype(np.float64)

    S = kernel_spectrum.astype(np.float64)
    S_inv = np.where(S > 1e-30, 1.0 / S, 0.0)

    mean_count = float(np.mean(y[y > 0])) if np.any(y > 0) else 1.0
    if prior_mean is None:
        mu = np.log(max(mean_count, 1e-3))
    else:
        mu = prior_mean

    # Initialise f at a safe log of the observations
    f = np.log(np.maximum(y, 0.5))

    def kinv_matvec(v: np.ndarray) -> np.ndarray:
        """Apply K^{-1} v  =  ifft(fft(v) / S)."""
        return np.real(ifftn(fftn(v) * S_inv))

    for it in range(max_newton):
        W = noise_model.hessian_diag(f)
        grad = -noise_model.grad_neg_log_lik(y, f) - kinv_matvec(f - mu)

        grad_norm = np.sqrt(np.mean(grad**2))
        if verbose:
            ll = noise_model.log_likelihood(y, f)
            prior_term = -0.5 * np.sum((f - mu) * kinv_matvec(f - mu))
            print(f"  Newton {it:3d}: |grad|={grad_norm:.3e}, "
                  f"LL={ll:.2f}, prior={prior_term:.2f}")

        if grad_norm < newton_tol * max(mean_count, 1.0):
            if verbose:
                print(f"  Converged at iteration {it}.")
            break

        def hessian_matvec(v: np.ndarray) -> np.ndarray:
            return W * v + kinv_matvec(v)

        delta_f = _cg_solve(hessian_matvec, grad, tol=cg_tol, maxiter=cg_maxiter)

        # Back-tracking line search
        obj_current = (
            noise_model.log_likelihood(y, f)
            - 0.5 * np.sum((f - mu) * kinv_matvec(f - mu))
        )
        step = 1.0
        for _ in range(20):
            f_trial = f + step * delta_f
            obj_trial = (
                noise_model.log_likelihood(y, f_trial)
                - 0.5 * np.sum((f_trial - mu) * kinv_matvec(f_trial - mu))
            )
            if obj_trial >= obj_current + 1e-4 * step * np.sum(grad * delta_f):
                break
            step *= 0.5
        f = f + step * delta_f

    # --- Posterior variance via stochastic diagonal probing ---
    W_final = noise_model.hessian_diag(f)

    def hessian_matvec_final(v: np.ndarray) -> np.ndarray:
        return W_final * v + kinv_matvec(v)

    rng = np.random.default_rng(42)
    diag_sum = np.zeros(shape, dtype=np.float64)
    for _ in range(n_variance_probes):
        z = rng.choice([-1.0, 1.0], size=shape)
        solve_z = _cg_solve(hessian_matvec_final, z, tol=cg_tol, maxiter=cg_maxiter)
        diag_sum += z * solve_z

    variance = diag_sum / n_variance_probes

    return {
        "mode": f,
        "denoised_counts": np.exp(f),
        "variance": variance,
        "n_newton": it + 1 if "it" in dir() else 0,
    }


# ===================================================================
#  Main entry point
# ===================================================================

@dataclass
class DenoiseResult:
    """Container for denoising results.

    Attributes
    ----------
    denoised : ndarray
        Denoised density cube in the original density units.
    variance : ndarray
        Posterior variance estimate per cell (in log-density units for Poisson,
        density units for Gaussian).
    params : dict
        Estimated or supplied kernel hyperparameters.
    method : str
        Method used (``'wiener'`` or ``'laplace'``).
    wiener_filter : ndarray or None
        Global Wiener filter W(k) in Fourier space (same shape as cube).
        Only set for log_wiener; use to apply the same transfer to gas.
    """

    denoised: np.ndarray
    variance: np.ndarray
    params: dict
    method: str
    wiener_filter: np.ndarray | None = None


def denoise_cube(
    cube: np.ndarray,
    n_eff: float = 16.0,
    method: Literal["wiener", "laplace", "log_wiener", "gaussian_snr"] = "log_wiener",
    noise_model_type: Literal["poisson", "gaussian"] = "poisson",
    kernel: str = "empirical",
    length_scale: float | None = None,
    amplitude: float | None = None,
    dep: str | None = None,
    deconvolve: bool = False,
    n_strata: int = 3,
    tol: float = 1e-5,
    gas_ref: np.ndarray | None = None,
    psd_floor: str = "flat",
    floor_snr: float = 0.01,
    noise_model: str = "poisson",
    box_size: float = 1.0,
    verbose: bool = False,
    **kwargs,
) -> DenoiseResult:
    """Denoise a 3-D particle density cube.

    This is the main entry point.  It estimates kernel hyperparameters
    (if not supplied), builds the spectral model, and dispatches to
    the selected inference method.

    Parameters
    ----------
    cube : ndarray (nx, ny, nz)
        Raw density cube (e.g. from ``part2cube``).
    n_eff : float
        Mean number of particles per cell at the average density.
    method : ``'wiener'``, ``'log_wiener'``, or ``'laplace'``
        Inference algorithm.  ``'log_wiener'`` (default) applies an
        iterative Wiener filter in log-density space — best for
        log-normal turbulent fields.  ``'wiener'`` uses a linear-space
        Gaussian-noise approximation.  ``'laplace'`` uses the full
        Poisson likelihood (slowest, most rigorous).
    noise_model_type : ``'poisson'`` or ``'gaussian'``
        Which noise model to use.
    kernel : str
        GP kernel: ``'empirical'`` (default, non-parametric — best for
        turbulent fields), ``'matern32'``, ``'matern52'``, or ``'rbf'``.
    length_scale : float, optional
        Kernel length scale in grid cells (parametric kernels only).
        Auto-estimated if omitted.
    amplitude : float, optional
        Kernel amplitude (parametric kernels only).  Auto-estimated if
        omitted.
    dep : str, optional
        Deposition scheme (``'CIC'``, ``'TSC'``, ``'PCS'``).  When given,
        the shot-noise power spectrum is corrected for the deposition window.
    deconvolve : bool
        If True and *dep* is set, apply a Wiener deconvolution filter
        that simultaneously removes noise and undoes the deposition window
        smoothing.  The filter is
        ``W(k) = H*(k) S_gas(k) / (|H(k)|^2 S_gas(k) + N(k))``
        where H(k) is the deposition transfer function.  Supported by the
        ``'wiener'`` and ``'log_wiener'`` methods.
    n_strata : int
        Number of density strata for heteroscedastic noise handling in the
        ``'log_wiener'`` method.  ``3`` (default) splits cells into
        equal-count density quantiles.  ``1`` uses a single global
        noise level (original algorithm).  Values > 1 split cells into
        equal-count density quantiles and apply a separate Wiener filter
        per stratum.  Ignored by other methods.
    tol : float
        Convergence tolerance for ``'log_wiener'``: stop when the relative
        change in the log-density estimate is below *tol* (default 1e-5).
        Ignored by other methods.
    box_size : float
        Physical box size (default 1.0, matching RAMSES code units).
    verbose : bool
    **kwargs
        Forwarded to the inference routine (e.g. ``max_newton``, ``cg_tol``).

    Returns
    -------
    DenoiseResult
    """
    cube = np.asarray(cube, dtype=np.float64)
    shape = cube.shape
    rho_mean = float(np.mean(cube))
    if rho_mean <= 0:
        raise ValueError("Cube has non-positive mean density; cannot denoise.")

    # log_wiener handles its own spectral estimation internally
    if method == "log_wiener":
        if verbose:
            print("Running iterative log-space Wiener filter ...")
        kwargs_log = {k: v for k, v in kwargs.items() if k != "gaussian_scale"}
        res = log_wiener_denoise(
            cube, n_eff=n_eff, box_size=box_size, dep=dep,
            deconvolve=deconvolve, n_strata=n_strata, tol=tol,
            gas_ref=gas_ref, psd_floor=psd_floor, floor_snr=floor_snr,
            noise_model=noise_model, verbose=verbose, **kwargs_log,
        )
        return DenoiseResult(
            denoised=res["denoised"],
            variance=res["variance"],
            params={"kernel": "empirical (log-space)"},
            method="log_wiener",
            wiener_filter=res.get("filter"),
        )

    # Gaussian smoothing with width set by SNR (sigma = scale / sqrt(n_eff))
    if method == "gaussian_snr":
        if verbose:
            print("Running Gaussian SNR smoothing ...")
        scale = kwargs.get("gaussian_scale", 2.0)
        res = gaussian_snr_denoise(cube, n_eff=n_eff, box_size=box_size, scale=scale)
        if verbose:
            sigma_cells = scale / np.sqrt(n_eff)
            print(f"  sigma = {sigma_cells:.3f} cells (scale={scale}, n_eff={n_eff})")
        return DenoiseResult(
            denoised=res["denoised"],
            variance=res["variance"],
            params={"kernel": "gaussian_snr", "scale": scale, "sigma_cells": scale / np.sqrt(n_eff)},
            method="gaussian_snr",
            wiener_filter=res["filter"],
        )

    # --- Build spectral model (for wiener / laplace) ---
    if kernel == "empirical":
        if verbose:
            print("Building empirical (non-parametric) signal power spectrum ...")
        S = empirical_power_spectrum(cube, n_eff, box_size=box_size, dep=dep, psd_floor=psd_floor, floor_snr=floor_snr)
        params = {"kernel": "empirical"}
    else:
        auto_params = {}
        if length_scale is None or amplitude is None:
            if verbose:
                print("Estimating kernel hyperparameters from the power spectrum ...")
            auto_params = estimate_kernel_params(
                cube, n_eff=n_eff, kernel=kernel, box_size=box_size, dep=dep
            )
            if verbose:
                print(f"  length_scale = {auto_params['length_scale']:.3f} cells")
                print(f"  amplitude    = {auto_params['amplitude']:.3e}")
                print(f"  noise_floor  = {auto_params['noise_floor']:.3e}")

        ell = length_scale if length_scale is not None else auto_params["length_scale"]
        amp = amplitude if amplitude is not None else auto_params["amplitude"]
        params = {"length_scale": ell, "amplitude": amp, "kernel": kernel}

        k2 = _build_k2_grid(shape, box_size)
        S = kernel_power_spectrum(k2, kernel=kernel, length_scale=ell, amplitude=amp)

    S = np.maximum(S, 1e-30)

    if method == "wiener":
        delta = cube / rho_mean - 1.0
        noise_psd = 1.0 / n_eff
        if dep is not None:
            N = noise_psd * deposition_window(shape, dep, box_size)
        else:
            N = noise_psd

        if deconvolve and dep is not None:
            # Wiener deconvolution: simultaneously denoise and undo the
            # deposition window H(k).  We estimate S_gas from the
            # isotropically-averaged (radially-binned) periodogram to avoid
            # noise amplification from per-mode division by |H|^2 near zeros.
            W2 = deposition_window(shape, dep, box_size)
            H = np.sqrt(W2)
            k2_grid = _build_k2_grid(shape, box_size)
            k_mag = np.sqrt(k2_grid)
            k_flat = k_mag.ravel()
            nz = k_flat > 0

            P_obs = np.abs(fftn(delta)) ** 2 / np.prod(shape)
            n_bins_d = max(min(shape) // 2, 16)
            k_edges_d = np.linspace(0, k_flat[nz].max(), n_bins_d + 1)
            k_centers_d = 0.5 * (k_edges_d[:-1] + k_edges_d[1:])

            P_bin = np.zeros(n_bins_d)
            N_bin = np.zeros(n_bins_d)
            H2_bin = np.zeros(n_bins_d)
            for i in range(n_bins_d):
                mask_b = nz & (k_flat >= k_edges_d[i]) & (k_flat < k_edges_d[i + 1])
                if mask_b.any():
                    P_bin[i] = np.mean(P_obs.ravel()[mask_b])
                    N_bin[i] = np.mean(N.ravel()[mask_b]) if np.ndim(N) > 0 else float(N)
                    H2_bin[i] = np.mean(W2.ravel()[mask_b])

            S_gas_bin = np.maximum(P_bin - N_bin, P_bin * 1e-3) / np.maximum(H2_bin, 1e-4)
            S_gas_bin = np.maximum(S_gas_bin, 1e-30)
            S_gas_3d = np.interp(k_flat, k_centers_d, S_gas_bin).reshape(shape)
            S_gas_3d.flat[0] = S_gas_3d.ravel()[nz].max() * 10.0

            W_deconv = H * S_gas_3d / (W2 * S_gas_3d + N)
            W_deconv.flat[0] = 1.0
            delta_k = fftn(delta)
            denoised_delta = np.real(ifftn(W_deconv * delta_k))
            if verbose:
                print("Applied Wiener deconvolution filter (dep + denoise).")
            var_spectrum = S_gas_3d * N / (W2 * S_gas_3d + N)
            var_spectrum.flat[0] = 0.0
            n_total = np.prod(shape)
            variance = np.real(ifftn(np.ones(shape) * var_spectrum)) / n_total
        else:
            res = wiener_denoise(delta, S, N)
            denoised_delta = res["denoised"]
            variance = res["variance"]

        denoised_rho = rho_mean * (1.0 + denoised_delta)
        denoised_rho = np.maximum(denoised_rho, rho_mean * 1e-6)
        return DenoiseResult(
            denoised=denoised_rho,
            variance=variance,
            params=params,
            method="wiener",
        )

    elif method == "laplace":
        if noise_model_type == "poisson":
            nm = PoissonNoise(n_eff=n_eff)
            counts = nm.density_to_counts(cube)
        elif noise_model_type == "gaussian":
            sigma2 = rho_mean / n_eff
            nm = GaussianNoise(variance_val=sigma2)
            counts = cube
        else:
            raise ValueError(f"Unknown noise model {noise_model_type!r}")

        res = laplace_gp_denoise(
            counts, S, noise_model=nm, verbose=verbose, **kwargs
        )

        if noise_model_type == "poisson":
            denoised_density = nm.counts_to_density(res["denoised_counts"], rho_mean)
        else:
            denoised_density = res["denoised_counts"]

        denoised_density = np.maximum(denoised_density, rho_mean * 1e-6)
        return DenoiseResult(
            denoised=denoised_density,
            variance=res["variance"],
            params=params,
            method="laplace",
        )
    else:
        raise ValueError(
            f"Unknown method {method!r}.  "
            "Choose 'wiener', 'log_wiener', 'laplace', or 'gaussian_snr'."
        )


# ===================================================================
#  CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Bayesian GP denoising of 3-D particle density cubes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to a Fortran-unformatted cube file (from part2cube).",
    )
    parser.add_argument(
        "--output", required=True,
        help="Path for the denoised output cube.",
    )
    parser.add_argument(
        "--method", default="log_wiener",
        choices=["wiener", "log_wiener", "laplace", "gaussian_snr"],
        help="Inference method (default: log_wiener). "
             "gaussian_snr: Gaussian smoothing with width = scale/sqrt(n_eff).",
    )
    parser.add_argument(
        "--noise-model", default="poisson", choices=["poisson", "gaussian"],
        dest="noise_model",
        help="Noise model (default: poisson).",
    )
    parser.add_argument(
        "--kernel", default="empirical",
        choices=["empirical", "matern32", "matern52", "rbf"],
        help="GP kernel (default: empirical — non-parametric, best for turbulence).",
    )
    parser.add_argument(
        "--length-scale", type=float, default=None, dest="length_scale",
        help="Kernel length scale in grid cells (auto-estimated if omitted).",
    )
    parser.add_argument(
        "--amplitude", type=float, default=None,
        help="Kernel amplitude (auto-estimated if omitted).",
    )
    parser.add_argument(
        "--n-eff", type=float, required=True, dest="n_eff",
        help="Mean number of particles per cell at the average density.",
    )
    parser.add_argument(
        "--dep", default=None, choices=["NGP", "CIC", "TSC", "PCS"],
        help="Deposition scheme for shot-noise window correction.",
    )
    parser.add_argument(
        "--deposit-noise", default="poisson", dest="deposit_noise",
        choices=["poisson", "compound_poisson"],
        help="Deposit noise model (log_wiener only): 'poisson' (default) "
             "uses 1/lambda per-cell variance shaped by the deposition "
             "window |W(k)|^2; 'compound_poisson' uses the deposit weight "
             "PDF to compute a reduced per-cell variance alpha/lambda "
             "(white noise, no window shaping).",
    )
    parser.add_argument(
        "--deconvolve", action="store_true",
        help="Simultaneously denoise and deconvolve the deposition window (wiener and log_wiener methods).",
    )
    parser.add_argument(
        "--n-strata", type=int, default=3, dest="n_strata",
        help="Number of density strata for heteroscedastic noise (log_wiener only). "
             "Default 3 = stratified filter; 1 = original single-level.",
    )
    parser.add_argument(
        "--tol", type=float, default=1e-5, dest="tol",
        help="Convergence tolerance for log_wiener: stop when relative change < tol (default 1e-5).",
    )
    parser.add_argument(
        "--gas-cube", default=None, dest="gas_cube", metavar="GAS_CUBE",
        help="Path to a reference gas density cube (.cube) for cross-power "
             "spectrum calibration.  When provided, the denoised field's "
             "Fourier amplitudes are rescaled so its power spectrum matches "
             "the gas reference.  Only affects log_wiener method.",
    )
    parser.add_argument(
        "--box-size", type=float, default=1.0, dest="box_size",
        help="Physical box size (default: 1.0).",
    )
    parser.add_argument(
        "--psd-floor", default="flat", dest="psd_floor",
        choices=["powerlaw", "flat"],
        help="Signal PSD regularization at high k: 'flat' (default) "
             "uses noise_binned * floor_snr; 'powerlaw' extrapolates "
             "a power law from high-SNR low-k bins.",
    )
    parser.add_argument(
        "--floor-snr", type=float, default=0.01, dest="floor_snr",
        help="Minimum signal-to-noise ratio for the 'flat' PSD floor "
             "(default 0.01).  The Wiener weight at noise-dominated "
             "scales is approximately this value.",
    )
    parser.add_argument(
        "--gaussian-scale", type=float, default=2.0, dest="gaussian_scale",
        help="For method gaussian_snr: sigma (cells) = scale/sqrt(n_eff). Default 2.",
    )
    parser.add_argument(
        "--save-variance", action="store_true", dest="save_variance",
        help="Also save the posterior variance cube (<output>.var).",
    )
    parser.add_argument(
        "--save-filter", action="store_true", dest="save_filter",
        help="Also save the filter G(k) as <output_stem>_filter.npy "
             "(log_wiener, gaussian_snr). Use to apply the same transfer to gas.",
    )
    parser.add_argument(
        "--validate", default=None, metavar="GAS_CUBE",
        help="Path to a reference gas cube for validation diagnostics.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print progress information.",
    )
    args = parser.parse_args()

    if args.verbose:
        print(f"Reading cube from {args.input} ...")
    cube = read_cube_fortran(args.input)
    if args.verbose:
        print(f"  Shape: {cube.shape}, mean={np.mean(cube):.4e}")

    gas_ref = None
    if args.gas_cube is not None:
        if args.verbose:
            print(f"Reading gas reference cube from {args.gas_cube} ...")
        gas_ref = read_cube_fortran(args.gas_cube)
        if args.verbose:
            print(f"  Gas ref shape: {gas_ref.shape}, mean={np.mean(gas_ref):.4e}")

    result = denoise_cube(
        cube,
        n_eff=args.n_eff,
        method=args.method,
        noise_model_type=args.noise_model,
        kernel=args.kernel,
        length_scale=args.length_scale,
        amplitude=args.amplitude,
        deconvolve=args.deconvolve,
        n_strata=args.n_strata,
        tol=args.tol,
        gas_ref=gas_ref,
        psd_floor=args.psd_floor,
        floor_snr=args.floor_snr,
        noise_model=args.deposit_noise,
        dep=args.dep,
        box_size=args.box_size,
        verbose=args.verbose,
        gaussian_scale=args.gaussian_scale,
    )

    if args.verbose:
        print(f"Writing denoised cube to {args.output} ...")
    save_cube_fortran(result.denoised, args.output)

    if args.save_variance:
        var_path = Path(args.output).with_suffix(".var")
        if args.verbose:
            print(f"Writing variance cube to {var_path} ...")
        save_cube_fortran(result.variance, var_path)

    if args.save_filter and result.wiener_filter is not None:
        filter_path = Path(args.output).parent / (Path(args.output).stem + "_filter.npy")
        if args.verbose:
            print(f"Writing Wiener filter to {filter_path} ...")
        np.save(filter_path, result.wiener_filter)

    if args.verbose:
        print("Done.")
        print(f"  Method:       {result.method}")
        print(f"  Kernel:       {result.params['kernel']}")
        if "length_scale" in result.params:
            print(f"  Length scale: {result.params['length_scale']:.3f} cells")
            print(f"  Amplitude:    {result.params['amplitude']:.3e}")

    if args.validate:
        _validate(cube, result.denoised, args.validate)


def _validate(raw: np.ndarray, denoised: np.ndarray, gas_path: str) -> None:
    """Print diagnostic comparison of raw and denoised against a reference."""
    gas = read_cube_fortran(gas_path).astype(np.float64)
    gas = gas * (np.mean(raw) / np.mean(gas))
    rho_mean = np.mean(raw)
    delta_gas = gas / rho_mean - 1.0

    print("\n=== Validation vs reference gas cube ===")
    header = f"{'':18s} {'RMSE(lin)':>10s} {'scatter(log)':>13s} {'corr(log)':>10s} {'neg_cells':>10s}"
    print(header)
    print("-" * len(header))

    for label, field in [("Raw tracer", raw), ("Denoised", denoised)]:
        delta_f = field / rho_mean - 1.0
        rmse = np.sqrt(np.mean((delta_f - delta_gas) ** 2))
        mask = (gas > 0) & (field > 0)
        log_f = np.log10(field[mask])
        log_g = np.log10(gas[mask])
        scatter = np.std(log_f - log_g)
        corr = np.corrcoef(log_g, log_f)[0, 1]
        neg = int(np.sum(field <= 0))
        print(f"{label:18s} {rmse:10.4f} {scatter:13.4f} {corr:10.4f} {neg:10d}")


if __name__ == "__main__":
    main()
