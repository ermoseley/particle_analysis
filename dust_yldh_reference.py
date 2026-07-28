#!/usr/bin/env python3
"""Offline float64 YLDH gyroresonance reference and table scaffold.

This module evaluates the balanced-cascade, ``n=1`` gyroresonant
Fokker--Planck coefficients in Yan, Lazarian & Draine (2004, their Appendix
B).  It includes the low-beta fast-mode spectrum from their Equation B8 and
the trans-Alfvénic anisotropic Alfvén spectrum from Equation B7.  The hard
lower resonance cutoff follows their Section 4: modes slower than the grain
gyroperiod do not violate the adiabatic invariant.

The calculation is deliberately an offline reference.  It uses float64
quadrature and exposes every local turbulence and damping parameter.  A
generated fixed-context reference table contains only float32 drift and a
lower-triangular noise factor.  Interpolating that factor, rather than the
diffusion-tensor entries, preserves positive semidefiniteness.

The implementation does not include transit-time damping (``n=0``),
imbalanced turbulence, resonance broadening beyond the YLD04 Lorentzian, or
an internally predicted damping scale, slow modes, high-beta fast modes, or
the dynamic charge/context axes required by production.  It also leaves the
conversion to mini-RAMSES code units to the runtime implementation.  These
omissions keep the scaffold honest: the production code must supply locally
appropriate cell-scale amplitudes and damping cutoffs before YLDH can be
selected.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.integrate import quad, quad_vec
from scipy.special import jv

E_ESU = 4.803204712570263e-10
C_LIGHT = 2.99792458e10
M_HYDROGEN = 1.6735575e-24
MICRON_CM = 1.0e-4
PC_CM = 3.085677581491367e18

MODEL_REVISION = "YLD04-balanced-n1-fast-lowbeta-alfven-v2"
TABLE_REVISION = "YLDH-wpar-muadb-drift-cholesky-f32-v2"
INTEGRATION_METHOD = "adaptive-inner-resonance-split-v1"
DEFAULT_INTEGRATION_ATOL = 0.0
DEFAULT_INTEGRATION_RTOL = 3.0e-7
DEFAULT_INTEGRATION_LIMIT = 200
PSD_MATERIAL_RTOL = 1024.0 * np.finfo(np.float64).eps


@dataclass(frozen=True)
class TurbulenceContext:
    """Local magnetic and cascade state in Gaussian cgs units."""

    name: str
    magnetic_field_gauss: float
    mass_density_g_cm3: float
    injection_scale_cm: float
    alfven_speed_cm_s: float
    fast_phase_speed_cm_s: float
    alfven_injection_velocity_cm_s: float
    fast_injection_velocity_cm_s: float
    alfven_cutoff_parallel_cm_inv: float
    fast_cutoff_cm_inv: float

    def validate(self) -> None:
        values = asdict(self)
        values.pop("name")
        for name, value in values.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class Grain:
    """Spherical grain state used by the offline reference."""

    radius_cm: float
    bulk_density_g_cm3: float
    charge_number: float

    def validate(self) -> None:
        if not np.isfinite(self.radius_cm) or self.radius_cm <= 0.0:
            raise ValueError("radius_cm must be finite and positive")
        if (
            not np.isfinite(self.bulk_density_g_cm3)
            or self.bulk_density_g_cm3 <= 0.0
        ):
            raise ValueError(
                "bulk_density_g_cm3 must be finite and positive"
            )
        if not np.isfinite(self.charge_number):
            raise ValueError("charge_number must be finite")

    @property
    def mass_g(self) -> float:
        return (
            (4.0 * np.pi / 3.0)
            * self.bulk_density_g_cm3
            * self.radius_cm**3
        )


@dataclass(frozen=True)
class PrimitiveDiffusion:
    """Diffusion coefficients in YLD04 coordinates ``(p, xi)``."""

    d_pp: float
    d_p_xi: float
    d_xi_xi: float
    fast_support_fraction: float
    alfven_support_fraction: float

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [[self.d_pp, self.d_p_xi], [self.d_p_xi, self.d_xi_xi]],
            dtype=np.float64,
        )


def yld04_cnm_context() -> TurbulenceContext:
    """Return the idealized CNM values quoted by YLD04.

    The Alfvén cutoff is the parallel cutoff stated in their Section 5.  The
    fast cutoff is the isotropic cutoff in the same paragraph.
    """

    return TurbulenceContext(
        name="YLD04_CNM",
        magnetic_field_gauss=6.0e-6,
        mass_density_g_cm3=1.4 * M_HYDROGEN * 30.0,
        injection_scale_cm=0.64 * PC_CM,
        alfven_speed_cm_s=2.0e5,
        fast_phase_speed_cm_s=2.0e5,
        alfven_injection_velocity_cm_s=2.0e5,
        fast_injection_velocity_cm_s=2.0e5,
        alfven_cutoff_parallel_cm_inv=4.0e-16,
        fast_cutoff_cm_inv=7.0e-15,
    )


def grain_gyrofrequency_s(grain: Grain, magnetic_field_gauss: float) -> float:
    """Return the bare Gaussian-cgs grain gyrofrequency ``|Ze|B/(mc)``."""

    grain.validate()
    if not np.isfinite(magnetic_field_gauss) or magnetic_field_gauss <= 0.0:
        raise ValueError("magnetic_field_gauss must be finite and positive")
    return (
        abs(grain.charge_number)
        * E_ESU
        * magnetic_field_gauss
        / (grain.mass_g * C_LIGHT)
    )


def _gauss_legendre_interval(
    count: int, lower: float, upper: float
) -> tuple[np.ndarray, np.ndarray]:
    if count < 4:
        raise ValueError("quadrature count must be at least four")
    node, weight = np.polynomial.legendre.leggauss(count)
    coordinate = 0.5 * ((upper - lower) * node + upper + lower)
    return coordinate, 0.5 * (upper - lower) * weight


def _validate_integration_controls(
    quadrature_order: int,
    integration_rtol: float,
    integration_limit: int,
) -> None:
    if quadrature_order < 4:
        raise ValueError("outer quadrature order must be at least four")
    if (
        not np.isfinite(integration_rtol)
        or integration_rtol <= 0.0
        or integration_rtol >= 1.0
    ):
        raise ValueError("integration_rtol must lie in (0, 1)")
    if integration_limit < 32:
        raise ValueError("integration_limit must be at least 32")


def _interior_resonance_split(
    lower: float,
    upper: float,
    peak: float,
    fractional_width: float,
) -> list[float]:
    """Return one point just beyond an endpoint resonance when useful."""

    if not (lower <= peak <= upper):
        return []
    offset = max(8.0 * abs(fractional_width), 1.0e-10)
    split = peak + offset
    if lower < split < upper:
        return [split]
    return []


def _fast_mode_coefficients(
    context: TurbulenceContext,
    grain: Grain,
    speed_cm_s: float,
    pitch_cosine: float,
    omega_s: float,
    quadrature_order: int,
    integration_rtol: float,
    integration_limit: int,
) -> tuple[float, float, float]:
    """Return the low-beta fast contribution using adaptive log-k integrals."""

    k_min = 1.0 / context.injection_scale_cm
    k_max = context.fast_cutoff_cm_inv
    if omega_s == 0.0 or k_max <= k_min:
        return 0.0, 0.0, 0.0

    zeta, weight_zeta = np.polynomial.legendre.leggauss(quadrature_order)
    v_perp = speed_cm_s * np.sqrt(max(1.0 - pitch_cosine**2, 0.0))
    phase_speed = context.fast_phase_speed_cm_s
    injection_velocity = context.fast_injection_velocity_cm_s
    cascade_amplitude = (injection_velocity / context.alfven_speed_cm_s) ** 2
    spectrum_normalization = (
        context.injection_scale_cm ** (-0.5)
        / (8.0 * np.pi)
        * cascade_amplitude
    )
    integral_pp = 0.0
    integral_xi = 0.0
    support_fraction = 0.0
    full_log_width = np.log(k_max / k_min)

    for zeta_value, zeta_weight in zip(zeta, weight_zeta, strict=True):
        velocity_mismatch = (
            phase_speed - speed_cm_s * pitch_cosine * zeta_value
        )
        denominator = abs(velocity_mismatch)
        if denominator == 0.0:
            continue
        k_resonant = omega_s / denominator
        k_lower = max(k_min, k_resonant)
        if k_lower >= k_max:
            continue

        log_lower = np.log(k_lower)
        log_upper = np.log(k_max)
        support_fraction += (
            zeta_weight * np.log(k_max / k_lower) / full_log_width
        )
        pitch_factor = (
            zeta_value + pitch_cosine * phase_speed / speed_cm_s
        ) ** 2

        def integrand(log_k: float) -> np.ndarray:
            k_value = np.exp(log_k)
            perpendicular_fraction = np.sqrt(
                max(1.0 - zeta_value**2, 0.0)
            )
            argument = (
                k_value * perpendicular_fraction * v_perp / omega_s
            )
            bessel_factor = (jv(2, argument) - jv(0, argument)) ** 2
            tau_k = (
                np.sqrt(context.injection_scale_cm / k_value)
                * phase_speed
                / injection_velocity**2
            )
            tau_inverse = 1.0 / tau_k
            mismatch = k_value * velocity_mismatch - omega_s
            resonance = tau_inverse / (tau_inverse**2 + mismatch**2)
            spectrum = spectrum_normalization * k_value ** (-3.5)
            measure = 2.0 * np.pi * k_value**3
            common = measure * resonance * spectrum * bessel_factor
            return np.array([common, common * pitch_factor])

        points: list[float] = []
        if velocity_mismatch > 0.0:
            peak_log_k = np.log(omega_s / velocity_mismatch)
            if log_lower <= peak_log_k <= log_upper:
                peak_k = np.exp(peak_log_k)
                peak_tau = (
                    np.sqrt(context.injection_scale_cm / peak_k)
                    * phase_speed
                    / injection_velocity**2
                )
                width_log_k = (1.0 / peak_tau) / omega_s
                points = _interior_resonance_split(
                    log_lower, log_upper, peak_log_k, width_log_k
                )
        integral, integration_error = quad_vec(
            integrand,
            log_lower,
            log_upper,
            epsabs=DEFAULT_INTEGRATION_ATOL,
            epsrel=integration_rtol,
            norm="max",
            limit=integration_limit,
            points=points,
        )
        integral_scale = max(float(np.max(np.abs(integral))), 1.0e-300)
        if integration_error > 10.0 * integration_rtol * integral_scale:
            raise RuntimeError("fast resonance integral did not converge")
        integral_pp += zeta_weight * float(integral[0])
        integral_xi += zeta_weight * float(integral[1])

    prefactor = 0.5 * np.pi * omega_s**2 * (1.0 - pitch_cosine**2)
    d_pp = (
        prefactor
        * grain.mass_g**2
        * context.alfven_speed_cm_s**2
        * integral_pp
    )
    d_xi_xi = prefactor * integral_xi
    return d_pp, d_xi_xi, float(0.5 * support_fraction)


def _alfven_branch_coefficients(
    context: TurbulenceContext,
    grain: Grain,
    speed_cm_s: float,
    pitch_cosine: float,
    omega_s: float,
    propagation_sign: int,
    quadrature_order: int,
    integration_rtol: float,
    integration_limit: int,
) -> tuple[float, float, float]:
    """Return one signed GS95 Alfvén propagation branch.

    For ``k_parallel = sigma |k_parallel|`` and
    ``omega = |k_parallel| V_A``, Appendix B gives the mismatch
    ``|k_parallel| (V_A - sigma v xi) - Omega`` and the pitch factor
    ``(1 + sigma xi V_A/v)^2``.
    """

    k_perp_min = 1.0 / context.injection_scale_cm
    k_parallel_max = context.alfven_cutoff_parallel_cm_inv
    k_perp_max = (
        k_parallel_max * context.injection_scale_cm ** (1.0 / 3.0)
    ) ** 1.5
    if omega_s == 0.0 or k_perp_max <= k_perp_min:
        return 0.0, 0.0, 0.0

    if propagation_sign not in (-1, 1):
        raise ValueError("propagation_sign must be -1 or +1")
    parallel_velocity = (
        context.alfven_speed_cm_s
        - propagation_sign * speed_cm_s * pitch_cosine
    )
    denominator = abs(parallel_velocity)
    if denominator == 0.0:
        return 0.0, 0.0, 0.0
    k_parallel_resonant = omega_s / denominator
    if k_parallel_resonant >= k_parallel_max:
        return 0.0, 0.0, 0.0

    log_k_perp, weight_log_k = _gauss_legendre_interval(
        quadrature_order, np.log(k_perp_min), np.log(k_perp_max)
    )
    v_perp = speed_cm_s * np.sqrt(max(1.0 - pitch_cosine**2, 0.0))
    injection_velocity = context.alfven_injection_velocity_cm_s
    cascade_amplitude = (
        injection_velocity / context.alfven_speed_cm_s
    ) ** 2
    spectrum_normalization = (
        context.injection_scale_cm ** (-1.0 / 3.0)
        / (12.0 * np.pi)
        * cascade_amplitude
    )
    integral = 0.0
    lower_fraction = k_parallel_resonant / k_parallel_max
    peak_fraction = (
        omega_s / (parallel_velocity * k_parallel_max)
        if parallel_velocity > 0.0
        else np.inf
    )

    for log_k_value, log_k_weight in zip(
        log_k_perp, weight_log_k, strict=True
    ):
        k_perp = np.exp(log_k_value)
        argument = k_perp * v_perp / omega_s
        bessel_factor = (jv(2, argument) + jv(0, argument)) ** 2
        tau_k = (
            context.injection_scale_cm
            / injection_velocity
            * (k_perp * context.injection_scale_cm) ** (-2.0 / 3.0)
        )
        tau_inverse = 1.0 / tau_k
        anisotropy_rate = (
            context.injection_scale_cm ** (1.0 / 3.0)
            * k_parallel_max
            / k_perp ** (2.0 / 3.0)
        )

        def integrand(parallel_fraction: float) -> float:
            mismatch = (
                k_parallel_max
                * parallel_fraction
                * parallel_velocity
                - omega_s
            )
            resonance = tau_inverse / (tau_inverse**2 + mismatch**2)
            anisotropy = np.exp(-anisotropy_rate * parallel_fraction)
            return k_parallel_max * resonance * anisotropy

        width_fraction = (
            tau_inverse
            / (parallel_velocity * k_parallel_max)
            if parallel_velocity > 0.0
            else 0.0
        )
        points = _interior_resonance_split(
            lower_fraction,
            1.0,
            peak_fraction,
            width_fraction,
        )
        inner_integral, integration_error = quad(
            integrand,
            lower_fraction,
            1.0,
            epsabs=DEFAULT_INTEGRATION_ATOL,
            epsrel=integration_rtol,
            limit=integration_limit,
            points=points,
        )
        if integration_error > (
            10.0
            * integration_rtol
            * max(abs(inner_integral), 1.0e-300)
        ):
            raise RuntimeError("Alfvén resonance integral did not converge")
        spectrum = spectrum_normalization * k_perp ** (-10.0 / 3.0)
        measure = 2.0 * np.pi * k_perp**2
        integral += (
            log_k_weight
            * measure
            * spectrum
            * bessel_factor
            * inner_integral
        )

    pitch_factor = (
        1.0
        + propagation_sign
        * pitch_cosine
        * context.alfven_speed_cm_s
        / speed_cm_s
    ) ** 2
    prefactor = 0.5 * np.pi * omega_s**2 * (1.0 - pitch_cosine**2)
    d_pp = (
        prefactor
        * grain.mass_g**2
        * context.alfven_speed_cm_s**2
        * integral
    )
    d_xi_xi = prefactor * pitch_factor * integral
    return d_pp, d_xi_xi, float(1.0 - lower_fraction)


def _alfven_mode_coefficients(
    context: TurbulenceContext,
    grain: Grain,
    speed_cm_s: float,
    pitch_cosine: float,
    omega_s: float,
    quadrature_order: int,
    integration_rtol: float,
    integration_limit: int,
) -> tuple[float, float, float]:
    """Return the sum of the two balanced GS95 Alfvén branches."""

    positive = _alfven_branch_coefficients(
        context,
        grain,
        speed_cm_s,
        pitch_cosine,
        omega_s,
        1,
        quadrature_order,
        integration_rtol,
        integration_limit,
    )
    negative = _alfven_branch_coefficients(
        context,
        grain,
        speed_cm_s,
        pitch_cosine,
        omega_s,
        -1,
        quadrature_order,
        integration_rtol,
        integration_limit,
    )
    return (
        positive[0] + negative[0],
        positive[1] + negative[1],
        0.5 * (positive[2] + negative[2]),
    )


def primitive_diffusion(
    context: TurbulenceContext,
    grain: Grain,
    speed_cm_s: float,
    pitch_cosine: float,
    *,
    modes: Iterable[str] = ("fast", "alfven"),
    quadrature_order: int = 64,
    integration_rtol: float = DEFAULT_INTEGRATION_RTOL,
    integration_limit: int = DEFAULT_INTEGRATION_LIMIT,
) -> PrimitiveDiffusion:
    """Evaluate balanced YLDH diffusion in ``(p, xi)``.

    ``D_p_xi`` is exactly zero because the magnetic--velocity cross
    correlation vanishes for the balanced cascade assumed by YLD04.
    """

    context.validate()
    grain.validate()
    if not np.isfinite(speed_cm_s) or speed_cm_s <= 0.0:
        raise ValueError("speed_cm_s must be finite and positive")
    if (
        not np.isfinite(pitch_cosine)
        or pitch_cosine < -1.0
        or pitch_cosine > 1.0
    ):
        raise ValueError("pitch_cosine must lie in [-1, 1]")
    selected_modes = tuple(modes)
    unknown = set(selected_modes) - {"fast", "alfven"}
    if unknown:
        raise ValueError(f"unknown modes: {sorted(unknown)}")
    _validate_integration_controls(
        quadrature_order, integration_rtol, integration_limit
    )

    omega_s = grain_gyrofrequency_s(grain, context.magnetic_field_gauss)
    d_pp = 0.0
    d_xi_xi = 0.0
    fast_support = 0.0
    alfven_support = 0.0
    if "fast" in selected_modes:
        contribution = _fast_mode_coefficients(
            context,
            grain,
            speed_cm_s,
            pitch_cosine,
            omega_s,
            quadrature_order,
            integration_rtol,
            integration_limit,
        )
        d_pp += contribution[0]
        d_xi_xi += contribution[1]
        fast_support = contribution[2]
    if "alfven" in selected_modes:
        contribution = _alfven_mode_coefficients(
            context,
            grain,
            speed_cm_s,
            pitch_cosine,
            omega_s,
            quadrature_order,
            integration_rtol,
            integration_limit,
        )
        d_pp += contribution[0]
        d_xi_xi += contribution[1]
        alfven_support = contribution[2]

    return PrimitiveDiffusion(
        d_pp=max(float(d_pp), 0.0),
        d_p_xi=0.0,
        d_xi_xi=max(float(d_xi_xi), 0.0),
        fast_support_fraction=fast_support,
        alfven_support_fraction=alfven_support,
    )


def transform_diffusion_to_w_mu(
    diffusion: PrimitiveDiffusion,
    grain_mass_g: float,
    magnetic_field_gauss: float,
    speed_cm_s: float,
    pitch_cosine: float,
) -> np.ndarray:
    """Transform diffusion from ``(p, xi)`` to ``(w_parallel, mu_adb)``.

    The code variable is ``mu_adb = v_perp^2/(2B)``.  This is why the
    perpendicular term contains ``B`` rather than the gyrofrequency.
    """

    s = float(speed_cm_s)
    xi = float(pitch_cosine)
    a = diffusion.d_pp / grain_mass_g**2
    c = diffusion.d_p_xi / grain_mass_g
    e = diffusion.d_xi_xi
    d_ww = xi**2 * a + 2.0 * s * xi * c + s**2 * e
    d_mu_mu = (
        s**2 * (1.0 - xi**2) ** 2 * a
        - 2.0 * s**3 * xi * (1.0 - xi**2) * c
        + s**4 * xi**2 * e
    ) / magnetic_field_gauss**2
    d_w_mu = (
        s * xi * (1.0 - xi**2) * a
        + s**2 * (1.0 - 2.0 * xi**2) * c
        - s**3 * xi * e
    ) / magnetic_field_gauss
    return np.array(
        [[d_ww, d_w_mu], [d_w_mu, d_mu_mu]], dtype=np.float64
    )


def _conservative_primitive_drift(
    d_pp: np.ndarray,
    d_p_xi: np.ndarray,
    d_xi_xi: np.ndarray,
    momentum_g_cm_s: np.ndarray,
    pitch_cosine: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Itô drift implied by the conservative YLD04 operator.

    The phase-space measure is ``p^2 dp dxi``.  Therefore
    ``b_i = J^-1 d_j(J D_ij)`` with ``J=p^2``.
    """

    edge_order = 2
    d_dpp_dp = np.gradient(
        d_pp, momentum_g_cm_s, axis=0, edge_order=edge_order
    )
    d_dpxi_dxi = np.gradient(
        d_p_xi, pitch_cosine, axis=1, edge_order=edge_order
    )
    d_dpxi_dp = np.gradient(
        d_p_xi, momentum_g_cm_s, axis=0, edge_order=edge_order
    )
    d_dxixi_dxi = np.gradient(
        d_xi_xi, pitch_cosine, axis=1, edge_order=edge_order
    )
    momentum = momentum_g_cm_s[:, None]
    drift_p = d_dpp_dp + d_dpxi_dxi + 2.0 * d_pp / momentum
    drift_xi = d_dpxi_dp + d_dxixi_dxi + 2.0 * d_p_xi / momentum
    return drift_p, drift_xi


def _transform_drift_to_w_mu(
    drift_p: np.ndarray,
    drift_xi: np.ndarray,
    d_pp: np.ndarray,
    d_p_xi: np.ndarray,
    d_xi_xi: np.ndarray,
    grain_mass_g: float,
    magnetic_field_gauss: float,
    speed_cm_s: np.ndarray,
    pitch_cosine: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the full Itô coordinate transformation, including Hessians."""

    s = speed_cm_s[:, None]
    xi = pitch_cosine[None, :]
    mass = grain_mass_g
    field = magnetic_field_gauss

    drift_w = (
        xi * drift_p / mass
        + s * drift_xi
        + 2.0 * d_p_xi / mass
    )
    drift_mu = (
        s * (1.0 - xi**2) * drift_p / (mass * field)
        - s**2 * xi * drift_xi / field
        + d_pp * (1.0 - xi**2) / (mass**2 * field)
        - 4.0 * d_p_xi * s * xi / (mass * field)
        - d_xi_xi * s**2 / field
    )
    return drift_w, drift_mu


def psd_lower_factor(
    matrix: np.ndarray,
    *,
    coordinate_scales: np.ndarray | None = None,
    material_rtol: float = PSD_MATERIAL_RTOL,
) -> tuple[np.ndarray, float]:
    """Return ``L`` such that ``L L^T = 2 D_psd``.

    Factorization is performed after a diagonal congruence scaling, so the
    unlike units and dynamic ranges of ``w_parallel`` and ``mu_adb`` are never
    compared directly.  Roundoff-level negative eigenvalues are projected to
    zero in scaled coordinates; material indefiniteness is an error.
    """

    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (2, 2) or np.any(~np.isfinite(matrix)):
        raise ValueError("matrix must be a finite 2x2 array")
    if (
        not np.isfinite(material_rtol)
        or material_rtol <= 0.0
        or material_rtol >= 1.0
    ):
        raise ValueError("material_rtol must lie in (0, 1)")
    symmetric = 0.5 * (matrix + matrix.T)

    if coordinate_scales is None:
        diagonal = np.abs(np.diag(symmetric))
        coordinate_scale = np.sqrt(diagonal)
        coordinate_scale[coordinate_scale == 0.0] = 1.0
    else:
        coordinate_scale = np.asarray(coordinate_scales, dtype=np.float64)
        if (
            coordinate_scale.shape != (2,)
            or np.any(~np.isfinite(coordinate_scale))
            or np.any(coordinate_scale <= 0.0)
        ):
            raise ValueError("coordinate_scales must be two finite positives")

    scaled = symmetric / np.outer(coordinate_scale, coordinate_scale)
    eigenvalue, eigenvector = np.linalg.eigh(scaled)
    denominator = max(
        float(np.linalg.norm(scaled)), np.finfo(np.float64).tiny
    )
    if float(np.min(eigenvalue)) < -material_rtol * denominator:
        raise ValueError("diffusion tensor is materially indefinite")
    projected = (
        eigenvector * np.maximum(eigenvalue, 0.0)
    ) @ eigenvector.T
    projection_correction = float(
        np.linalg.norm(projected - scaled) / denominator
    )

    covariance_rate = 2.0 * projected
    tolerance = material_rtol * max(
        float(np.max(np.abs(covariance_rate))),
        np.finfo(np.float64).tiny,
    )
    a = float(covariance_rate[0, 0])
    b = float(covariance_rate[1, 0])
    c = float(covariance_rate[1, 1])
    if a > tolerance:
        l_11 = np.sqrt(a)
        l_21 = b / l_11
        residual = c - l_21**2
        if residual < -tolerance:
            raise ValueError("scaled PSD factorization failed")
        l_22 = np.sqrt(max(residual, 0.0))
    else:
        if abs(b) > tolerance:
            raise ValueError("zero-variance coordinate has nonzero covariance")
        l_11 = 0.0
        l_21 = 0.0
        l_22 = np.sqrt(max(c, 0.0))

    lower_scaled = np.array(
        [[l_11, 0.0], [l_21, l_22]], dtype=np.float64
    )
    reconstruction = lower_scaled @ lower_scaled.T
    reconstruction_correction = float(
        np.linalg.norm(reconstruction - covariance_rate)
        / max(np.linalg.norm(covariance_rate), np.finfo(np.float64).tiny)
    )
    lower = coordinate_scale[:, None] * lower_scaled
    return lower, max(projection_correction, reconstruction_correction)


def build_runtime_table(
    context: TurbulenceContext,
    grain: Grain,
    speed_axis_cm_s: np.ndarray,
    pitch_axis: np.ndarray,
    *,
    modes: Iterable[str] = ("fast", "alfven"),
    quadrature_order: int = 64,
    integration_rtol: float = DEFAULT_INTEGRATION_RTOL,
    integration_limit: int = DEFAULT_INTEGRATION_LIMIT,
) -> dict[str, np.ndarray | float | int | str]:
    """Build a fixed-context float32 table on a ``(speed, pitch)`` grid.

    Runtime interpolation should be applied to the drift and factor entries.
    With two independent normal draws ``eta``, the update is
    ``dy = drift*dt + L*eta*sqrt(dt)``.

    This table is an offline reference for one grain and one turbulence
    context.  It is not the dynamic charge/context table required by Stage 13.
    """

    context.validate()
    grain.validate()
    _validate_integration_controls(
        quadrature_order, integration_rtol, integration_limit
    )
    speed = np.asarray(speed_axis_cm_s, dtype=np.float64)
    pitch = np.asarray(pitch_axis, dtype=np.float64)
    if (
        speed.ndim != 1
        or pitch.ndim != 1
        or speed.size < 3
        or pitch.size < 3
    ):
        raise ValueError("table axes must be one-dimensional with >=3 points")
    if np.any(~np.isfinite(speed)) or np.any(speed <= 0.0):
        raise ValueError("speed axis must be finite and positive")
    if np.any(np.diff(speed) <= 0.0):
        raise ValueError("speed axis must be strictly increasing")
    if (
        np.any(~np.isfinite(pitch))
        or np.any(pitch <= -1.0)
        or np.any(pitch >= 1.0)
        or np.any(np.diff(pitch) <= 0.0)
    ):
        raise ValueError(
            "pitch axis must be strictly increasing inside (-1, 1)"
        )

    shape = (speed.size, pitch.size)
    d_pp = np.empty(shape, dtype=np.float64)
    d_p_xi = np.empty(shape, dtype=np.float64)
    d_xi_xi = np.empty(shape, dtype=np.float64)
    fast_support = np.empty(shape, dtype=np.float64)
    alfven_support = np.empty(shape, dtype=np.float64)
    selected_modes = tuple(modes)
    for i_speed, speed_value in enumerate(speed):
        for i_pitch, pitch_value in enumerate(pitch):
            diffusion = primitive_diffusion(
                context,
                grain,
                float(speed_value),
                float(pitch_value),
                modes=selected_modes,
                quadrature_order=quadrature_order,
                integration_rtol=integration_rtol,
                integration_limit=integration_limit,
            )
            d_pp[i_speed, i_pitch] = diffusion.d_pp
            d_p_xi[i_speed, i_pitch] = diffusion.d_p_xi
            d_xi_xi[i_speed, i_pitch] = diffusion.d_xi_xi
            fast_support[i_speed, i_pitch] = (
                diffusion.fast_support_fraction
            )
            alfven_support[i_speed, i_pitch] = (
                diffusion.alfven_support_fraction
            )

    momentum = grain.mass_g * speed
    drift_p, drift_xi = _conservative_primitive_drift(
        d_pp, d_p_xi, d_xi_xi, momentum, pitch
    )
    drift_w, drift_mu = _transform_drift_to_w_mu(
        drift_p,
        drift_xi,
        d_pp,
        d_p_xi,
        d_xi_xi,
        grain.mass_g,
        context.magnetic_field_gauss,
        speed,
        pitch,
    )

    factor_11 = np.empty(shape, dtype=np.float64)
    factor_21 = np.empty(shape, dtype=np.float64)
    factor_22 = np.empty(shape, dtype=np.float64)
    maximum_projection = 0.0
    for i_speed, speed_value in enumerate(speed):
        for i_pitch, pitch_value in enumerate(pitch):
            diffusion = PrimitiveDiffusion(
                d_pp=float(d_pp[i_speed, i_pitch]),
                d_p_xi=float(d_p_xi[i_speed, i_pitch]),
                d_xi_xi=float(d_xi_xi[i_speed, i_pitch]),
                fast_support_fraction=float(
                    fast_support[i_speed, i_pitch]
                ),
                alfven_support_fraction=float(
                    alfven_support[i_speed, i_pitch]
                ),
            )
            transformed = transform_diffusion_to_w_mu(
                diffusion,
                grain.mass_g,
                context.magnetic_field_gauss,
                float(speed_value),
                float(pitch_value),
            )
            factor, correction = psd_lower_factor(transformed)
            factor_11[i_speed, i_pitch] = factor[0, 0]
            factor_21[i_speed, i_pitch] = factor[1, 0]
            factor_22[i_speed, i_pitch] = factor[1, 1]
            maximum_projection = max(maximum_projection, correction)

    physics_arrays = {
        "speed_cm_s": speed.astype(np.float32),
        "pitch_cosine": pitch.astype(np.float32),
        "drift_w_parallel": drift_w.astype(np.float32),
        "drift_mu_adb": drift_mu.astype(np.float32),
        "noise_l11": factor_11.astype(np.float32),
        "noise_l21": factor_21.astype(np.float32),
        "noise_l22": factor_22.astype(np.float32),
        "fast_support_fraction": fast_support.astype(np.float32),
        "alfven_support_fraction": alfven_support.astype(np.float32),
    }
    return {
        **physics_arrays,
        "model_revision": MODEL_REVISION,
        "table_revision": TABLE_REVISION,
        "reference_scope": "fixed-context-single-grain-cgs",
        "modes": ",".join(selected_modes),
        "integration_method": INTEGRATION_METHOD,
        "integration_absolute_tolerance": DEFAULT_INTEGRATION_ATOL,
        "integration_relative_tolerance": integration_rtol,
        "outer_quadrature_order": quadrature_order,
        "integration_subdivision_limit": integration_limit,
        "factorization_scaling": "per-node-diagonal-congruence",
        "maximum_psd_projection_relative": maximum_projection,
    }


def write_runtime_table(
    path: Path,
    table: dict[str, np.ndarray | float | int | str],
    context: TurbulenceContext,
    grain: Grain,
) -> None:
    """Write a compressed table with float32 physics arrays."""

    path = Path(path)
    metadata = {
        "model_revision": table["model_revision"],
        "table_revision": table["table_revision"],
        "reference_scope": table["reference_scope"],
        "modes": table["modes"],
        "integration_method": table["integration_method"],
        "integration_absolute_tolerance": table[
            "integration_absolute_tolerance"
        ],
        "integration_relative_tolerance": table[
            "integration_relative_tolerance"
        ],
        "outer_quadrature_order": table["outer_quadrature_order"],
        "integration_subdivision_limit": table[
            "integration_subdivision_limit"
        ],
        "factorization_scaling": table["factorization_scaling"],
        "context_cgs": asdict(context),
        "grain_cgs": asdict(grain),
        "maximum_psd_projection_relative": table[
            "maximum_psd_projection_relative"
        ],
        "update": "dy = drift*dt + L*normal(0,1)*sqrt(dt)",
    }
    payload = {
        name: value
        for name, value in table.items()
        if isinstance(value, np.ndarray)
    }
    for name, value in payload.items():
        if value.dtype != np.float32:
            raise ValueError(f"{name} is not float32")
    payload["metadata_json"] = np.asarray(
        json.dumps(metadata, sort_keys=True)
    )
    np.savez_compressed(path, **payload)


def _default_grain() -> Grain:
    # A representative large active grain.  The charge is an input from the
    # Stage 9 equilibrium closure, not a second charging prescription.
    return Grain(
        radius_cm=0.1 * MICRON_CM,
        bulk_density_g_cm3=3.5,
        charge_number=40.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("yldh_cnm_reference_f32.npz"),
    )
    parser.add_argument("--n-speed", type=int, default=20)
    parser.add_argument("--n-pitch", type=int, default=17)
    parser.add_argument("--quadrature-order", type=int, default=64)
    parser.add_argument(
        "--integration-rtol",
        type=float,
        default=DEFAULT_INTEGRATION_RTOL,
    )
    parser.add_argument(
        "--integration-limit",
        type=int,
        default=DEFAULT_INTEGRATION_LIMIT,
    )
    parser.add_argument(
        "--modes",
        default="fast,alfven",
        help="comma-separated subset of fast,alfven",
    )
    args = parser.parse_args()

    context = yld04_cnm_context()
    grain = _default_grain()
    speed = np.geomspace(1.0e4, 5.0e6, args.n_speed)
    pitch = np.linspace(-0.95, 0.95, args.n_pitch)
    modes = tuple(item.strip() for item in args.modes.split(",") if item.strip())
    table = build_runtime_table(
        context,
        grain,
        speed,
        pitch,
        modes=modes,
        quadrature_order=args.quadrature_order,
        integration_rtol=args.integration_rtol,
        integration_limit=args.integration_limit,
    )
    write_runtime_table(args.output, table, context, grain)
    summary = {
        "output": str(args.output),
        "bytes": args.output.stat().st_size,
        "model_revision": MODEL_REVISION,
        "table_revision": TABLE_REVISION,
        "shape": [args.n_speed, args.n_pitch],
        "modes": modes,
        "reference_scope": table["reference_scope"],
        "integration_method": table["integration_method"],
        "integration_absolute_tolerance": table[
            "integration_absolute_tolerance"
        ],
        "integration_relative_tolerance": table[
            "integration_relative_tolerance"
        ],
        "outer_quadrature_order": table["outer_quadrature_order"],
        "integration_subdivision_limit": table[
            "integration_subdivision_limit"
        ],
        "maximum_psd_projection_relative": table[
            "maximum_psd_projection_relative"
        ],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
