#!/usr/bin/env python3
"""Offline float64 YLD04 gyroresonance reference and table generator.

This module evaluates the balanced-cascade, ``|n|=1`` gyroresonant
Fokker--Planck coefficients in Yan, Lazarian & Draine (2004, their Appendix
B).  It uses the trans-Alfvénic anisotropic Alfvén tensor in their Equation
B3 and the low-beta fast-mode tensor in their Equation B4.  The hard
lower resonance cutoff follows their Section 4: modes slower than the grain
gyroperiod do not violate the adiabatic invariant.

The calculation is deliberately an offline reference.  It uses float64
quadrature and exposes every local turbulence and damping parameter.  The
``yldh04_balanced_gyro`` generator tabulates the dimensionless coordinates
``R=v/(|Omega|L)``, ``u=v/V_A``, and pitch cosine for one declared set of
balanced-mode amplitudes and dimensionless damping cutoffs.  It stores only
float32 drift and a lower-triangular noise factor.  Interpolating that factor,
rather than the diffusion-tensor entries, preserves positive semidefiniteness.

The implementation does not include transit-time damping (``n=0``),
imbalanced turbulence, resonance broadening beyond the YLD04 Lorentzian, or
an internally predicted damping scale, slow modes, high-beta fast modes, or
dynamic turbulence-amplitude and damping axes.  It is therefore a bounded
reference for the explicitly named reduced selector, not a full YLDH model.
The production scattering model remains off.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
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

MODEL_REVISION = "YLD04-balanced-absn1-fast-lowbeta-alfven-v3"
TABLE_REVISION = "YLDH-wpar-muadb-drift-cholesky-f32-v2"
RUNTIME_DIMENSIONLESS_TABLE_REVISION = (
    "yldh04-balanced-gyro-logR-logu-xi-f32-v3"
)
REDUCED_SELECTOR = "yldh04_balanced_gyro"
INTEGRATION_METHOD = "adaptive-inner-resonance-split-v1"
DEFAULT_INTEGRATION_ATOL = 0.0
DEFAULT_INTEGRATION_RTOL = 3.0e-7
DEFAULT_INTEGRATION_LIMIT = 200
PSD_MATERIAL_RTOL = 1024.0 * np.finfo(np.float64).eps
DEFAULT_RUNTIME_N_RESONANCE = 21
DEFAULT_RUNTIME_N_SPEED = 13
DEFAULT_RUNTIME_N_PITCH = 13
DEFAULT_RUNTIME_SPEED_REFINEMENT_INTERVALS = (0, 4, 6, 7, 8)
DEFAULT_RUNTIME_PITCH_REFINEMENT_INTERVALS = (0, 5, 6, 11)
DEFAULT_RUNTIME_PITCH_SECOND_REFINEMENT_INTERVALS = (6, 7, 8, 9)
DEFAULT_RUNTIME_RESONANCE_MIN = 3.0e-5
DEFAULT_RUNTIME_RESONANCE_MAX = 0.1
DEFAULT_RUNTIME_SPEED_MIN = 0.05
DEFAULT_RUNTIME_SPEED_MAX = 25.0
DEFAULT_RUNTIME_PITCH_MAX = 0.95
DEFAULT_RUNTIME_QUADRATURE_ORDER = 40
REDUCED_MODEL_EXCLUSIONS = (
    "transit-time damping (n=0)",
    "slow modes",
    "high-beta fast modes",
    "imbalanced turbulence",
    "charge-distribution dynamics",
    "dynamic turbulence-context axes",
    "production use",
)


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
    """Return the low-beta fast contribution using YLD04 B1 and B4."""

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
    """Return one signed GS95 Alfvén branch using YLD04 B1 and B3.

    For ``k_parallel = sigma |k_parallel|`` and
    ``omega = |k_parallel| V_A``, Equation B1 gives the mismatch
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
    """Evaluate reduced balanced YLD04 ``|n|=1`` diffusion in ``(p, xi)``.

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


def transform_diffusion_to_dimensionless_u_pitch(
    diffusion: PrimitiveDiffusion,
    grain_mass_g: float,
    context: TurbulenceContext,
) -> np.ndarray:
    """Return diffusion in ``u=v/V_A`` and pitch for ``tau=t V_A/L``."""

    if not np.isfinite(grain_mass_g) or grain_mass_g <= 0.0:
        raise ValueError("grain_mass_g must be finite and positive")
    context.validate()
    length = context.injection_scale_cm
    speed = context.alfven_speed_cm_s
    return np.array(
        [
            [
                diffusion.d_pp * length / (grain_mass_g**2 * speed**3),
                diffusion.d_p_xi * length / (grain_mass_g * speed**2),
            ],
            [
                diffusion.d_p_xi * length / (grain_mass_g * speed**2),
                diffusion.d_xi_xi * length / speed,
            ],
        ],
        dtype=np.float64,
    )


def grain_with_matched_gyrofrequency(
    grain_geometry: Grain,
    magnetic_field_gauss: float,
    absolute_gyrofrequency_s: float,
) -> Grain:
    """Return the same grain geometry with ``|Z|`` chosen to match ``Omega``."""

    grain_geometry.validate()
    if not np.isfinite(magnetic_field_gauss) or magnetic_field_gauss <= 0.0:
        raise ValueError("magnetic_field_gauss must be finite and positive")
    if (
        not np.isfinite(absolute_gyrofrequency_s)
        or absolute_gyrofrequency_s <= 0.0
    ):
        raise ValueError(
            "absolute_gyrofrequency_s must be finite and positive"
        )
    charge_number = (
        absolute_gyrofrequency_s
        * grain_geometry.mass_g
        * C_LIGHT
        / (E_ESU * magnetic_field_gauss)
    )
    return Grain(
        radius_cm=grain_geometry.radius_cm,
        bulk_density_g_cm3=grain_geometry.bulk_density_g_cm3,
        charge_number=charge_number,
    )


def dimensionless_balanced_mode_signature(
    context: TurbulenceContext,
) -> dict[str, float | str]:
    """Return every fixed dimensionless turbulence parameter in the table."""

    context.validate()
    length = context.injection_scale_cm
    alfven_speed = context.alfven_speed_cm_s
    return {
        "propagation_balance": "equal-forward-backward-Cij-zero",
        "fast_phase_speed_over_V_A": (
            context.fast_phase_speed_cm_s / alfven_speed
        ),
        "alfven_injection_velocity_over_V_A": (
            context.alfven_injection_velocity_cm_s / alfven_speed
        ),
        "fast_injection_velocity_over_V_A": (
            context.fast_injection_velocity_cm_s / alfven_speed
        ),
        "alfven_parallel_k_max_L": (
            context.alfven_cutoff_parallel_cm_inv * length
        ),
        "fast_isotropic_k_max_L": context.fast_cutoff_cm_inv * length,
    }


def _validate_log_axis(axis: np.ndarray, name: str) -> None:
    if axis.ndim != 1 or axis.size < 3:
        raise ValueError(f"{name} must be one-dimensional with >=3 points")
    if np.any(~np.isfinite(axis)) or np.any(np.diff(axis) <= 0.0):
        raise ValueError(f"{name} must be finite and strictly increasing")


def _conservative_dimensionless_ru_drift(
    d_uu: np.ndarray,
    d_u_pitch: np.ndarray,
    d_pitch_pitch: np.ndarray,
    log_resonance_ratio: np.ndarray,
    log_speed_over_alfven: np.ndarray,
    pitch_cosine: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return complete Itô drift on ``(log R, log u, xi)`` tables.

    The physical momentum derivative holds
    ``g=|Omega|L/V_A=u/R`` fixed.  Consequently
    ``d/du|g=(d/dlog(u)+d/dlog(R))/u``.  The phase-space Jacobian contributes
    the usual ``2 D_uj/u`` terms.
    """

    shape = (
        log_resonance_ratio.size,
        log_speed_over_alfven.size,
        pitch_cosine.size,
    )
    for name, value in (
        ("d_uu", d_uu),
        ("d_u_pitch", d_u_pitch),
        ("d_pitch_pitch", d_pitch_pitch),
    ):
        if np.asarray(value).shape != shape:
            raise ValueError(f"{name} does not match the three table axes")
    _validate_log_axis(log_resonance_ratio, "log_resonance_ratio")
    _validate_log_axis(log_speed_over_alfven, "log_speed_over_alfven")
    _validate_table_axes(
        np.exp(log_speed_over_alfven), pitch_cosine
    )

    edge_order = 2
    speed = np.exp(log_speed_over_alfven)[None, :, None]
    d_duu_du_fixed_gyro = (
        np.gradient(
            d_uu,
            log_resonance_ratio,
            axis=0,
            edge_order=edge_order,
        )
        + np.gradient(
            d_uu,
            log_speed_over_alfven,
            axis=1,
            edge_order=edge_order,
        )
    ) / speed
    d_du_pitch_du_fixed_gyro = (
        np.gradient(
            d_u_pitch,
            log_resonance_ratio,
            axis=0,
            edge_order=edge_order,
        )
        + np.gradient(
            d_u_pitch,
            log_speed_over_alfven,
            axis=1,
            edge_order=edge_order,
        )
    ) / speed
    drift_u = (
        d_duu_du_fixed_gyro
        + np.gradient(
            d_u_pitch,
            pitch_cosine,
            axis=2,
            edge_order=edge_order,
        )
        + 2.0 * d_uu / speed
    )
    drift_pitch = (
        d_du_pitch_du_fixed_gyro
        + np.gradient(
            d_pitch_pitch,
            pitch_cosine,
            axis=2,
            edge_order=edge_order,
        )
        + 2.0 * d_u_pitch / speed
    )
    return drift_u, drift_pitch


def _validate_table_axes(
    speed: np.ndarray,
    pitch: np.ndarray,
) -> None:
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


def build_dimensionless_balanced_gyro_ru_table(
    context: TurbulenceContext,
    reference_grain_geometry: Grain,
    log_resonance_ratio_axis: np.ndarray,
    log_speed_over_alfven_axis: np.ndarray,
    pitch_axis: np.ndarray,
    *,
    quadrature_order: int = DEFAULT_RUNTIME_QUADRATURE_ORDER,
    integration_rtol: float = DEFAULT_INTEGRATION_RTOL,
    integration_limit: int = DEFAULT_INTEGRATION_LIMIT,
) -> dict[str, np.ndarray | float | int | str]:
    """Build the runtime-shaped reduced table on ``(log R, log u, xi)``.

    ``R=v/(|Omega|L)`` and ``u=v/V_A``.  At each ``(R,u)`` the generator
    chooses the charge on an arbitrary reference grain geometry to reproduce
    ``|Omega|=u V_A/(R L)``.  Grain mass then cancels from the normalized
    ``(u,xi)`` tensor.
    """

    context.validate()
    reference_grain_geometry.validate()
    _validate_integration_controls(
        quadrature_order, integration_rtol, integration_limit
    )
    log_resonance = np.asarray(
        log_resonance_ratio_axis, dtype=np.float64
    )
    log_speed = np.asarray(log_speed_over_alfven_axis, dtype=np.float64)
    pitch = np.asarray(pitch_axis, dtype=np.float64)
    _validate_log_axis(log_resonance, "log_resonance_ratio")
    _validate_log_axis(log_speed, "log_speed_over_alfven")
    _validate_table_axes(np.exp(log_speed), pitch)

    mode_signature = dimensionless_balanced_mode_signature(context)
    generator_power = (
        float(mode_signature["alfven_injection_velocity_over_V_A"]) ** 2
        + float(mode_signature["fast_injection_velocity_over_V_A"]) ** 2
    )
    if generator_power <= 0.0:
        raise ValueError("balanced-mode input power must be positive")
    resonance = np.exp(log_resonance)
    speed_ratio = np.exp(log_speed)
    shape = (resonance.size, speed_ratio.size, pitch.size)
    d_uu = np.empty(shape, dtype=np.float64)
    d_u_pitch = np.empty(shape, dtype=np.float64)
    d_pitch_pitch = np.empty(shape, dtype=np.float64)
    fast_support = np.empty(shape, dtype=np.float64)
    alfven_support = np.empty(shape, dtype=np.float64)
    minimum_charge = np.inf
    maximum_charge = 0.0
    for i_resonance, resonance_value in enumerate(resonance):
        for i_speed, speed_value in enumerate(speed_ratio):
            speed_cm_s = speed_value * context.alfven_speed_cm_s
            omega_s = (
                speed_cm_s
                / (resonance_value * context.injection_scale_cm)
            )
            grain = grain_with_matched_gyrofrequency(
                reference_grain_geometry,
                context.magnetic_field_gauss,
                omega_s,
            )
            minimum_charge = min(minimum_charge, abs(grain.charge_number))
            maximum_charge = max(maximum_charge, abs(grain.charge_number))
            for i_pitch, pitch_value in enumerate(pitch):
                try:
                    diffusion = primitive_diffusion(
                        context,
                        grain,
                        float(speed_cm_s),
                        float(pitch_value),
                        modes=("fast", "alfven"),
                        quadrature_order=quadrature_order,
                        integration_rtol=integration_rtol,
                        integration_limit=integration_limit,
                    )
                except RuntimeError as error:
                    raise RuntimeError(
                        "table integration failed at "
                        f"R={resonance_value:.9e}, "
                        f"u={speed_value:.9e}, "
                        f"xi={pitch_value:.9e}"
                    ) from error
                dimensionless = (
                    transform_diffusion_to_dimensionless_u_pitch(
                        diffusion, grain.mass_g, context
                    )
                    / generator_power
                )
                index = (i_resonance, i_speed, i_pitch)
                d_uu[index] = dimensionless[0, 0]
                d_u_pitch[index] = dimensionless[0, 1]
                d_pitch_pitch[index] = dimensionless[1, 1]
                fast_support[index] = diffusion.fast_support_fraction
                alfven_support[index] = diffusion.alfven_support_fraction

    drift_u, drift_pitch = _conservative_dimensionless_ru_drift(
        d_uu,
        d_u_pitch,
        d_pitch_pitch,
        log_resonance,
        log_speed,
        pitch,
    )
    factor_11 = np.empty(shape, dtype=np.float64)
    factor_21 = np.empty(shape, dtype=np.float64)
    factor_22 = np.empty(shape, dtype=np.float64)
    maximum_projection = 0.0
    for index in np.ndindex(shape):
        factor, correction = psd_lower_factor(
            np.array(
                [
                    [d_uu[index], d_u_pitch[index]],
                    [d_u_pitch[index], d_pitch_pitch[index]],
                ]
            )
        )
        factor_11[index] = factor[0, 0]
        factor_21[index] = factor[1, 0]
        factor_22[index] = factor[1, 1]
        maximum_projection = max(maximum_projection, correction)

    return {
        "log_resonance_ratio": log_resonance.astype(np.float32),
        "log_speed_over_alfven": log_speed.astype(np.float32),
        "pitch_cosine": pitch.astype(np.float32),
        "drift_speed_over_alfven": drift_u.astype(np.float32),
        "drift_pitch_cosine": drift_pitch.astype(np.float32),
        "noise_l11": factor_11.astype(np.float32),
        "noise_l21": factor_21.astype(np.float32),
        "noise_l22": factor_22.astype(np.float32),
        "fast_support_fraction": fast_support.astype(np.float32),
        "alfven_support_fraction": alfven_support.astype(np.float32),
        "selector": REDUCED_SELECTOR,
        "model_revision": MODEL_REVISION,
        "table_revision": RUNTIME_DIMENSIONLESS_TABLE_REVISION,
        "reference_scope": (
            "dimensionless-fixed-balanced-mode-amplitudes-and-cutoffs"
        ),
        "modes": "alfven-balanced-absn1,fast-lowbeta-absn1",
        "integration_method": INTEGRATION_METHOD,
        "integration_absolute_tolerance": DEFAULT_INTEGRATION_ATOL,
        "integration_relative_tolerance": integration_rtol,
        "outer_quadrature_order": quadrature_order,
        "integration_subdivision_limit": integration_limit,
        "factorization_scaling": "per-node-diagonal-congruence",
        "maximum_psd_projection_relative": maximum_projection,
        "generator_minimum_absolute_charge_number": minimum_charge,
        "generator_maximum_absolute_charge_number": maximum_charge,
        "generator_input_power_proxy": generator_power,
        "basis_input_power_proxy": 1.0,
    }


def _runtime_dimensionless_table_metadata(
    table: dict[str, np.ndarray | float | int | str],
    context: TurbulenceContext,
    reference_grain_geometry: Grain,
) -> dict:
    context.validate()
    reference_grain_geometry.validate()
    signature = dimensionless_balanced_mode_signature(context)
    alfven_power = float(
        signature["alfven_injection_velocity_over_V_A"]
    ) ** 2
    fast_power = float(
        signature["fast_injection_velocity_over_V_A"]
    ) ** 2
    total_power = alfven_power + fast_power
    log_resonance = np.asarray(table["log_resonance_ratio"])
    log_speed = np.asarray(table["log_speed_over_alfven"])
    pitch = np.asarray(table["pitch_cosine"])
    return {
        "schema_version": 1,
        "selector": table["selector"],
        "production_default": False,
        "model_revision": table["model_revision"],
        "table_revision": table["table_revision"],
        "reference_scope": table["reference_scope"],
        "included_modes": table["modes"],
        "excluded_physics": list(REDUCED_MODEL_EXCLUSIONS),
        "primary_reference": {
            "citation": (
                "Yan, Lazarian & Draine 2004, ApJ, 616, 895"
            ),
            "doi": "10.1086/425111",
            "arxiv": "astro-ph/0408173",
            "gyroresonance_operator": "Appendix B1",
            "balanced_correlations": "Appendix B2",
            "alfven_tensor": "Appendix B3",
            "low_beta_fast_tensor": "Appendix B4",
        },
        "coordinates": {
            "axes": [
                "log_R with R=v/(|Omega|L)",
                "log_u with u=v/V_A",
                "xi=cos(pitch)",
            ],
            "logarithm": "natural",
            "time": "tau=t*V_A/L",
            "interpolation": (
                "tensor-product local PCHIP of drift and lower-factor "
                "entries in (log_R,log_u,xi); trilinear support diagnostics"
            ),
            "stochastic_update": (
                "dy=drift*d_tau+L_factor*N(0,I)*sqrt(d_tau)"
            ),
            "drift_derivative": (
                "d/du at fixed |Omega|L/V_A "
                "=(d/dlog_u+d/dlog_R)/u"
            ),
        },
        "fixed_dimensionless_mode_model": signature,
        "generator_context_cgs": asdict(context),
        "generator_reference_grain_geometry_cgs": {
            "radius_cm": reference_grain_geometry.radius_cm,
            "bulk_density_g_cm3": (
                reference_grain_geometry.bulk_density_g_cm3
            ),
            "mass_g": reference_grain_geometry.mass_g,
            "charge_policy": (
                "chosen at each (R,u) to match |Omega|; not a table axis"
            ),
            "minimum_absolute_charge_number": table[
                "generator_minimum_absolute_charge_number"
            ],
            "maximum_absolute_charge_number": table[
                "generator_maximum_absolute_charge_number"
            ],
        },
        "grain_scaling": {
            "invariant": (
                "normalized tensor depends on |Omega|, not grain mass, "
                "radius, or charge separately"
            ),
            "runtime_charge_sign": (
                "abs(Omega) selects scattering; Lorentz sign remains separate"
            ),
        },
        "mode_partition": {
            "propagation_balance": (
                "equal forward/backward intensity; C_ij=0"
            ),
            "family_partition_basis": (
                "squared injection velocity of each included family"
            ),
            "alfven_power_proxy": alfven_power,
            "fast_power_proxy": fast_power,
            "alfven_fraction": alfven_power / total_power,
            "fast_fraction": fast_power / total_power,
        },
        "wave_power_scaling": {
            "basis_total_power_proxy": table["basis_input_power_proxy"],
            "generator_total_power_proxy_before_normalization": table[
                "generator_input_power_proxy"
            ],
            "external_amplitude_name": "A_wave",
            "allowed_amplitude": "finite and nonnegative",
            "tensor_rule": "D=A_wave*D_basis",
            "drift_rule": "b=A_wave*b_basis",
            "noise_rule": "L=sqrt(A_wave)*L_basis",
            "sgs_mapping_if_sigma_is_total_wave_rms": (
                "A_wave=(sigma_sgs/V_A)^2"
            ),
            "fixed_under_rescaling": (
                "Alfven/fast partition, decorrelation kernel, and cutoffs"
            ),
            "limitation": (
                "This scales correlation-tensor power only; it does not "
                "recompute amplitude-dependent cascade decorrelation."
            ),
        },
        "envelope": {
            "log_resonance_ratio": [
                float(log_resonance[0]),
                float(log_resonance[-1]),
            ],
            "resonance_ratio": [
                float(np.exp(log_resonance[0])),
                float(np.exp(log_resonance[-1])),
            ],
            "log_speed_over_alfven": [
                float(log_speed[0]),
                float(log_speed[-1]),
            ],
            "speed_over_alfven": [
                float(np.exp(log_speed[0])),
                float(np.exp(log_speed[-1])),
            ],
            "pitch_cosine": [float(pitch[0]), float(pitch[-1])],
            "boundary_policy": (
                "closed tabulated envelope; reject, never clamp or extrapolate"
            ),
            "support_transition_policy": (
                "reject a queried cell containing both zero and nonzero "
                "resonance support; PCHIP neighbors set monotone slopes only"
            ),
            "mode_model_policy": (
                "exact declared dimensionless amplitudes and cutoffs"
            ),
        },
        "integration": {
            "method": table["integration_method"],
            "absolute_tolerance": table[
                "integration_absolute_tolerance"
            ],
            "relative_tolerance": table[
                "integration_relative_tolerance"
            ],
            "outer_quadrature_order": table["outer_quadrature_order"],
            "subdivision_limit": table[
                "integration_subdivision_limit"
            ],
        },
        "factorization": {
            "method": table["factorization_scaling"],
            "maximum_psd_projection_relative": table[
                "maximum_psd_projection_relative"
            ],
        },
    }


def require_runtime_balanced_mode_signature(
    metadata: dict,
    context: TurbulenceContext,
    *,
    relative_tolerance: float = 2.0e-7,
) -> None:
    """Reject turbulence amplitudes or cutoffs outside the declared model."""

    if (
        not np.isfinite(relative_tolerance)
        or relative_tolerance <= 0.0
        or relative_tolerance >= 1.0
    ):
        raise ValueError("relative_tolerance must lie in (0,1)")
    expected = metadata.get("fixed_dimensionless_mode_model")
    measured = dimensionless_balanced_mode_signature(context)
    if not isinstance(expected, dict) or set(expected) != set(measured):
        raise ValueError("dimensionless mode signature is incomplete")
    for name, value in measured.items():
        reference = expected[name]
        if isinstance(value, str):
            matched = value == reference
        else:
            matched = np.isclose(
                value,
                reference,
                rtol=relative_tolerance,
                atol=0.0,
            )
        if not matched:
            raise ValueError(
                f"{name} differs from the table's fixed mode model"
            )


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


_RUNTIME_DIMENSIONLESS_TABLE_ARRAYS = (
    "log_resonance_ratio",
    "log_speed_over_alfven",
    "pitch_cosine",
    "drift_speed_over_alfven",
    "drift_pitch_cosine",
    "noise_l11",
    "noise_l21",
    "noise_l22",
    "fast_support_fraction",
    "alfven_support_fraction",
)
_RUNTIME_DIMENSIONLESS_INTERPOLATED_ARRAYS = (
    "drift_speed_over_alfven",
    "drift_pitch_cosine",
    "noise_l11",
    "noise_l21",
    "noise_l22",
    "fast_support_fraction",
    "alfven_support_fraction",
)


def _validate_runtime_dimensionless_table_arrays(
    table: dict[str, np.ndarray | float | int | str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    missing = set(_RUNTIME_DIMENSIONLESS_TABLE_ARRAYS) - set(table)
    if missing:
        raise ValueError(f"runtime table is missing {sorted(missing)}")
    log_resonance = np.asarray(table["log_resonance_ratio"])
    log_speed = np.asarray(table["log_speed_over_alfven"])
    pitch = np.asarray(table["pitch_cosine"])
    _validate_log_axis(log_resonance, "log_resonance_ratio")
    _validate_log_axis(log_speed, "log_speed_over_alfven")
    _validate_table_axes(np.exp(log_speed), pitch)
    shape = (log_resonance.size, log_speed.size, pitch.size)
    for name in _RUNTIME_DIMENSIONLESS_TABLE_ARRAYS:
        value = np.asarray(table[name])
        if name == "log_resonance_ratio":
            expected_shape = (log_resonance.size,)
        elif name == "log_speed_over_alfven":
            expected_shape = (log_speed.size,)
        elif name == "pitch_cosine":
            expected_shape = (pitch.size,)
        else:
            expected_shape = shape
        if value.shape != expected_shape:
            raise ValueError(
                f"{name} has shape {value.shape}, not {expected_shape}"
            )
        if value.dtype != np.float32:
            raise ValueError(f"{name} is not float32")
        if np.any(~np.isfinite(value)):
            raise ValueError(f"{name} contains non-finite values")
    if np.any(np.asarray(table["noise_l11"]) < 0.0):
        raise ValueError("noise_l11 contains negative entries")
    if np.any(np.asarray(table["noise_l22"]) < 0.0):
        raise ValueError("noise_l22 contains negative entries")
    for name in ("fast_support_fraction", "alfven_support_fraction"):
        support = np.asarray(table[name])
        if np.any((support < 0.0) | (support > 1.0)):
            raise ValueError(f"{name} lies outside [0,1]")
    return log_resonance, log_speed, pitch


def _closed_axis_bracket(
    axis: np.ndarray,
    value: float,
) -> tuple[int, np.float32]:
    nearest = int(np.argmin(np.abs(axis - value)))
    snap_tolerance = (
        2.0
        * np.finfo(np.float32).eps
        * max(1.0, abs(float(axis[nearest])))
    )
    if abs(value - float(axis[nearest])) <= snap_tolerance:
        if nearest == axis.size - 1:
            return nearest - 1, np.float32(1.0)
        return nearest, np.float32(0.0)
    if value <= float(axis[0]):
        return 0, np.float32(0.0)
    if value >= float(axis[-1]):
        return axis.size - 2, np.float32(1.0)
    lower = int(np.searchsorted(axis, value, side="right") - 1)
    weight = (value - float(axis[lower])) / (
        float(axis[lower + 1]) - float(axis[lower])
    )
    return lower, np.float32(weight)


def _weighted_offsets(weight: np.float32) -> tuple[int, ...]:
    if weight == np.float32(0.0):
        return (0,)
    if weight == np.float32(1.0):
        return (1,)
    return (0, 1)


def _pchip_axis_indices(
    axis_size: int,
    lower: int,
    weight: np.float32,
) -> tuple[int, ...]:
    if weight == np.float32(0.0):
        return (lower,)
    if weight == np.float32(1.0):
        return (lower + 1,)
    return tuple(range(max(0, lower - 1), min(axis_size, lower + 3)))


def _pchip_node_derivative(
    axis: np.ndarray,
    values: np.ndarray,
    index: int,
) -> np.float32:
    axis = np.asarray(axis, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    if axis.ndim != 1 or values.shape != axis.shape or axis.size < 3:
        raise ValueError("PCHIP input must have matching axes of length >=3")
    spacing = np.diff(axis)
    slope = np.diff(values) / spacing
    zero = np.float32(0.0)
    three = np.float32(3.0)
    if index == 0:
        derivative = (
            (np.float32(2.0) * spacing[0] + spacing[1]) * slope[0]
            - spacing[0] * slope[1]
        ) / (spacing[0] + spacing[1])
        if np.sign(derivative) != np.sign(slope[0]):
            return zero
        if (
            np.sign(slope[0]) != np.sign(slope[1])
            and abs(derivative) > three * abs(slope[0])
        ):
            return np.float32(three * slope[0])
        return np.float32(derivative)
    if index == axis.size - 1:
        derivative = (
            (np.float32(2.0) * spacing[-1] + spacing[-2]) * slope[-1]
            - spacing[-1] * slope[-2]
        ) / (spacing[-1] + spacing[-2])
        if np.sign(derivative) != np.sign(slope[-1]):
            return zero
        if (
            np.sign(slope[-1]) != np.sign(slope[-2])
            and abs(derivative) > three * abs(slope[-1])
        ):
            return np.float32(three * slope[-1])
        return np.float32(derivative)

    slope_lower = slope[index - 1]
    slope_upper = slope[index]
    if (
        slope_lower == zero
        or slope_upper == zero
        or np.sign(slope_lower) != np.sign(slope_upper)
    ):
        return zero
    weight_lower = (
        np.float32(2.0) * spacing[index] + spacing[index - 1]
    )
    weight_upper = (
        spacing[index] + np.float32(2.0) * spacing[index - 1]
    )
    return np.float32(
        (weight_lower + weight_upper)
        / (weight_lower / slope_lower + weight_upper / slope_upper)
    )


def _pchip_interpolate_1d(
    axis: np.ndarray,
    values: np.ndarray,
    value: float,
) -> np.float32:
    axis = np.asarray(axis, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    if axis.size == 1:
        return np.float32(values[0])
    lower, weight = _closed_axis_bracket(
        axis.astype(np.float64), float(value)
    )
    if weight == np.float32(0.0):
        return np.float32(values[lower])
    if weight == np.float32(1.0):
        return np.float32(values[lower + 1])
    derivative_lower = _pchip_node_derivative(axis, values, lower)
    derivative_upper = _pchip_node_derivative(axis, values, lower + 1)
    spacing = np.float32(axis[lower + 1] - axis[lower])
    t = weight
    t_squared = np.float32(t * t)
    t_cubed = np.float32(t_squared * t)
    return np.float32(
        (
            np.float32(2.0) * t_cubed
            - np.float32(3.0) * t_squared
            + np.float32(1.0)
        )
        * values[lower]
        + (t_cubed - np.float32(2.0) * t_squared + t)
        * spacing
        * derivative_lower
        + (
            -np.float32(2.0) * t_cubed
            + np.float32(3.0) * t_squared
        )
        * values[lower + 1]
        + (t_cubed - t_squared) * spacing * derivative_upper
    )


def _tensor_pchip_interpolate(
    axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    values: np.ndarray,
    coordinates: tuple[float, float, float],
    indices: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
) -> np.float32:
    resonance_index, speed_index, pitch_index = indices
    pitch_stage = np.empty(
        (len(resonance_index), len(speed_index)), dtype=np.float32
    )
    for i_resonance, source_resonance in enumerate(resonance_index):
        for i_speed, source_speed in enumerate(speed_index):
            pitch_stage[i_resonance, i_speed] = _pchip_interpolate_1d(
                axes[2][list(pitch_index)],
                values[
                    source_resonance,
                    source_speed,
                    list(pitch_index),
                ],
                coordinates[2],
            )
    speed_stage = np.empty(len(resonance_index), dtype=np.float32)
    for i_resonance in range(len(resonance_index)):
        speed_stage[i_resonance] = _pchip_interpolate_1d(
            axes[1][list(speed_index)],
            pitch_stage[i_resonance, :],
            coordinates[1],
        )
    return _pchip_interpolate_1d(
        axes[0][list(resonance_index)],
        speed_stage,
        coordinates[0],
    )


def interpolate_dimensionless_balanced_gyro_ru(
    table: dict[str, np.ndarray | float | int | str],
    resonance_ratio: float,
    speed_over_alfven: float,
    pitch_cosine: float,
) -> dict[str, np.float32]:
    """PCHIP-interpolate the reduced factor on its closed envelope."""

    log_resonance_axis, log_speed_axis, pitch_axis = (
        _validate_runtime_dimensionless_table_arrays(table)
    )
    resonance_value = float(resonance_ratio)
    speed_value = float(speed_over_alfven)
    pitch_value = float(pitch_cosine)
    if not np.isfinite(resonance_value) or resonance_value <= 0.0:
        raise ValueError("resonance_ratio must be finite and positive")
    if not np.isfinite(speed_value) or speed_value <= 0.0:
        raise ValueError("speed_over_alfven must be finite and positive")
    if not np.isfinite(pitch_value):
        raise ValueError("pitch_cosine must be finite")

    resonance_bounds = np.exp(log_resonance_axis.astype(np.float64))
    speed_bounds = np.exp(log_speed_axis.astype(np.float64))
    if (
        resonance_value < resonance_bounds[0]
        or resonance_value > resonance_bounds[-1]
    ):
        raise ValueError("resonance_ratio lies outside the closed envelope")
    if speed_value < speed_bounds[0] or speed_value > speed_bounds[-1]:
        raise ValueError("speed_over_alfven lies outside the closed envelope")
    if pitch_value < pitch_axis[0] or pitch_value > pitch_axis[-1]:
        raise ValueError("pitch_cosine lies outside the closed envelope")

    i_resonance, weight_resonance = _closed_axis_bracket(
        log_resonance_axis.astype(np.float64), np.log(resonance_value)
    )
    i_speed, weight_speed = _closed_axis_bracket(
        log_speed_axis.astype(np.float64), np.log(speed_value)
    )
    i_pitch, weight_pitch = _closed_axis_bracket(
        pitch_axis.astype(np.float64), pitch_value
    )
    pchip_indices = (
        _pchip_axis_indices(
            log_resonance_axis.size, i_resonance, weight_resonance
        ),
        _pchip_axis_indices(log_speed_axis.size, i_speed, weight_speed),
        _pchip_axis_indices(pitch_axis.size, i_pitch, weight_pitch),
    )
    support_indices = (
        tuple(
            i_resonance + offset
            for offset in _weighted_offsets(weight_resonance)
        ),
        tuple(
            i_speed + offset
            for offset in _weighted_offsets(weight_speed)
        ),
        tuple(
            i_pitch + offset
            for offset in _weighted_offsets(weight_pitch)
        ),
    )
    for name in ("fast_support_fraction", "alfven_support_fraction"):
        support = np.asarray(table[name])
        weighted_support = [
            support[resonance_index, speed_index, pitch_index]
            for resonance_index in support_indices[0]
            for speed_index in support_indices[1]
            for pitch_index in support_indices[2]
        ]
        if min(weighted_support) == 0.0 and max(weighted_support) > 0.0:
            mode = name.removesuffix("_support_fraction")
            raise ValueError(
                f"lookup crosses unresolved {mode} resonance-support boundary"
            )
    one = np.float32(1.0)
    result: dict[str, np.float32] = {}
    for name in _RUNTIME_DIMENSIONLESS_INTERPOLATED_ARRAYS:
        values = np.asarray(table[name])
        if name not in (
            "fast_support_fraction",
            "alfven_support_fraction",
        ):
            result[name] = _tensor_pchip_interpolate(
                (
                    log_resonance_axis,
                    log_speed_axis,
                    pitch_axis,
                ),
                values,
                (
                    np.log(resonance_value),
                    np.log(speed_value),
                    pitch_value,
                ),
                pchip_indices,
            )
            continue
        interpolated_resonance = []
        for resonance_offset in (0, 1):
            interpolated_speed = []
            for speed_offset in (0, 1):
                lower = values[
                    i_resonance + resonance_offset,
                    i_speed + speed_offset,
                    i_pitch,
                ]
                upper = values[
                    i_resonance + resonance_offset,
                    i_speed + speed_offset,
                    i_pitch + 1,
                ]
                interpolated_speed.append(
                    (one - weight_pitch) * lower + weight_pitch * upper
                )
            interpolated_resonance.append(
                (one - weight_speed) * interpolated_speed[0]
                + weight_speed * interpolated_speed[1]
            )
        result[name] = np.float32(
            (one - weight_resonance) * interpolated_resonance[0]
            + weight_resonance * interpolated_resonance[1]
        )
    return result


def scale_balanced_gyro_wave_power(
    lookup: dict[str, np.float32],
    wave_power_amplitude: float,
) -> dict[str, np.float32]:
    """Scale one unit-power lookup while keeping its mode model fixed."""

    amplitude = float(wave_power_amplitude)
    if not np.isfinite(amplitude) or amplitude < 0.0:
        raise ValueError("wave_power_amplitude must be finite and nonnegative")
    missing = set(_RUNTIME_DIMENSIONLESS_INTERPOLATED_ARRAYS) - set(lookup)
    if missing:
        raise ValueError(f"lookup is missing {sorted(missing)}")
    amplitude_f32 = np.float32(amplitude)
    noise_scale = np.float32(np.sqrt(amplitude))
    result: dict[str, np.float32] = {}
    for name in _RUNTIME_DIMENSIONLESS_INTERPOLATED_ARRAYS:
        value = np.float32(lookup[name])
        if name.startswith("drift_"):
            result[name] = np.float32(amplitude_f32 * value)
        elif name.startswith("noise_"):
            result[name] = np.float32(noise_scale * value)
        else:
            result[name] = value
    return result


def write_dimensionless_balanced_gyro_ru_table(
    path: Path,
    table: dict[str, np.ndarray | float | int | str],
    context: TurbulenceContext,
    reference_grain_geometry: Grain,
) -> tuple[str, Path]:
    """Write the runtime-shaped reduced table and SHA-256 sidecar."""

    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("runtime dimensionless table path must end in .npz")
    _validate_runtime_dimensionless_table_arrays(table)
    metadata = _runtime_dimensionless_table_metadata(
        table, context, reference_grain_geometry
    )
    payload = {
        name: np.asarray(table[name])
        for name in _RUNTIME_DIMENSIONLESS_TABLE_ARRAYS
    }
    payload["metadata_json"] = np.asarray(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    )
    np.savez_compressed(path, **payload)
    digest = sha256_file(path)
    manifest = path.with_suffix(path.suffix + ".sha256")
    manifest.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest, manifest


def load_dimensionless_balanced_gyro_ru_table(
    path: Path,
) -> tuple[dict[str, np.ndarray], dict]:
    """Load and validate the runtime-shaped reduced table."""

    with np.load(Path(path), allow_pickle=False) as stored:
        table = {
            name: np.asarray(stored[name])
            for name in _RUNTIME_DIMENSIONLESS_TABLE_ARRAYS
        }
        metadata = json.loads(str(stored["metadata_json"]))
    _validate_runtime_dimensionless_table_arrays(table)
    if metadata.get("selector") != REDUCED_SELECTOR:
        raise ValueError("table selector is not yldh04_balanced_gyro")
    if (
        metadata.get("table_revision")
        != RUNTIME_DIMENSIONLESS_TABLE_REVISION
    ):
        raise ValueError("runtime dimensionless table revision mismatch")
    if metadata.get("production_default") is not False:
        raise ValueError("reduced selector must remain off in production")
    return table, metadata


_FORTRAN_RUNTIME_ARRAY_NAMES = {
    "log_resonance_ratio": "dust_yldh_log_r",
    "log_speed_over_alfven": "dust_yldh_log_u",
    "pitch_cosine": "dust_yldh_xi",
    "drift_speed_over_alfven": "dust_yldh_drift_u",
    "drift_pitch_cosine": "dust_yldh_drift_xi",
    "noise_l11": "dust_yldh_l11",
    "noise_l21": "dust_yldh_l21",
    "noise_l22": "dust_yldh_l22",
}
_FORTRAN_RUNTIME_SUPPORT_NAMES = {
    "fast_support_fraction": "dust_yldh_fast_supported",
    "alfven_support_fraction": "dust_yldh_alfven_supported",
}


def _fortran_f32_literal(value: np.float32) -> str:
    return f"{float(np.float32(value)):.9e}_4"


def _fortran_array_declaration(
    source_name: str,
    fortran_name: str,
    value: np.ndarray,
) -> list[str]:
    value = np.asarray(value)
    if value.dtype != np.float32 or np.any(~np.isfinite(value)):
        raise ValueError(f"{source_name} must be finite float32")
    literals = [
        _fortran_f32_literal(item)
        for item in value.ravel(order="F")
    ]
    if value.ndim == 1:
        opening = (
            f"real(kind=4), parameter :: {fortran_name}"
            f"({value.size}) = [ &"
        )
        closing = "]"
    elif value.ndim == 3:
        opening = (
            f"real(kind=4), parameter :: {fortran_name}"
            "(dust_yldh_nr,dust_yldh_nu,dust_yldh_nxi) = reshape([ &"
        )
        closing = (
            "], [dust_yldh_nr,dust_yldh_nu,dust_yldh_nxi])"
        )
    else:
        raise ValueError(f"{source_name} must have one or three dimensions")
    lines = [opening]
    for start in range(0, len(literals), 4):
        chunk = literals[start : start + 4]
        final = start + len(chunk) == len(literals)
        lines.append(
            "  " + ", ".join(chunk) + (" &" if final else ", &")
        )
    lines.append(closing)
    return lines


def _fortran_support_declaration(
    source_name: str,
    fortran_name: str,
    value: np.ndarray,
) -> list[str]:
    value = np.asarray(value)
    if value.dtype != np.float32 or np.any(~np.isfinite(value)):
        raise ValueError(f"{source_name} must be finite float32")
    literals = [
        ".true." if item > 0.0 else ".false."
        for item in value.ravel(order="F")
    ]
    lines = [
        (
            f"logical, parameter :: {fortran_name}"
            "(dust_yldh_nr,dust_yldh_nu,dust_yldh_nxi) = reshape([ &"
        )
    ]
    for start in range(0, len(literals), 8):
        chunk = literals[start : start + 8]
        final = start + len(chunk) == len(literals)
        lines.append(
            "  " + ", ".join(chunk) + (" &" if final else ", &")
        )
    lines.append(
        "], [dust_yldh_nr,dust_yldh_nu,dust_yldh_nxi])"
    )
    return lines


def write_cuda_fortran_balanced_gyro_include(
    path: Path,
    table: dict[str, np.ndarray | float | int | str],
    source_table_sha256: str,
) -> tuple[str, Path]:
    """Emit deterministic float32 CUDA-Fortran constants and a checksum."""

    _validate_runtime_dimensionless_table_arrays(table)
    if (
        len(source_table_sha256) != 64
        or any(character not in "0123456789abcdef" for character in source_table_sha256)
    ):
        raise ValueError("source_table_sha256 must be lowercase SHA-256")
    path = Path(path)
    log_resonance = np.asarray(table["log_resonance_ratio"])
    log_speed = np.asarray(table["log_speed_over_alfven"])
    pitch = np.asarray(table["pitch_cosine"])
    revision = str(table["table_revision"])
    lines = [
        "! Generated by dust_yldh_reference.py; do not edit.",
        "! Reduced balanced |n|=1 gyroresonance; production default is off.",
        "! Interpolate drift and L with local tensor-product PCHIP.",
        "! Reject interpolation across mixed false/true support corners.",
        f"! source_npz_sha256={source_table_sha256}",
        (
            f"character(len={len(revision)}), parameter :: "
            f'dust_yldh_revision = "{revision}"'
        ),
        (
            "character(len=64), parameter :: dust_yldh_source_sha256 = "
            f'"{source_table_sha256}"'
        ),
        "logical, parameter :: dust_yldh_production_default = .false.",
        f"integer, parameter :: dust_yldh_nr = {log_resonance.size}",
        f"integer, parameter :: dust_yldh_nu = {log_speed.size}",
        f"integer, parameter :: dust_yldh_nxi = {pitch.size}",
    ]
    for source_name, fortran_name in _FORTRAN_RUNTIME_ARRAY_NAMES.items():
        lines.extend(
            _fortran_array_declaration(
                source_name,
                fortran_name,
                np.asarray(table[source_name]),
            )
        )
    for source_name, fortran_name in _FORTRAN_RUNTIME_SUPPORT_NAMES.items():
        lines.extend(
            _fortran_support_declaration(
                source_name,
                fortran_name,
                np.asarray(table[source_name]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    digest = sha256_file(path)
    manifest = path.with_suffix(path.suffix + ".sha256")
    manifest.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    return digest, manifest


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
    """Build the original fixed-context cgs diagnostic table.

    Interpolation should be applied to the drift and factor entries.
    With two independent normal draws ``eta``, the update is
    ``dy = drift*dt + L*eta*sqrt(dt)``.

    This retained compatibility helper uses the code coordinates
    ``(w_parallel, mu_adb)`` and dimensional cgs entries.  New validation for
    the reduced selector should use
    :func:`build_dimensionless_balanced_gyro_ru_table`.
    """

    context.validate()
    grain.validate()
    _validate_integration_controls(
        quadrature_order, integration_rtol, integration_limit
    )
    speed = np.asarray(speed_axis_cm_s, dtype=np.float64)
    pitch = np.asarray(pitch_axis, dtype=np.float64)
    _validate_table_axes(speed, pitch)

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
    """Write the retained fixed-context cgs diagnostic table."""

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


def _axis_with_midpoints(
    axis: np.ndarray,
    intervals: Iterable[int],
) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    midpoint = []
    for interval in intervals:
        if interval < 0 or interval >= axis.size - 1:
            raise ValueError("refinement interval lies outside the axis")
        midpoint.append(
            np.float64(0.5) * (axis[interval] + axis[interval + 1])
        )
    return np.asarray(sorted([*axis, *midpoint]), dtype=np.float64)


def default_runtime_dimensionless_axes() -> tuple[
    np.ndarray, np.ndarray, np.ndarray
]:
    """Return the compact nonuniform default axes selected by audit."""

    log_resonance = np.linspace(
        np.log(DEFAULT_RUNTIME_RESONANCE_MIN),
        np.log(DEFAULT_RUNTIME_RESONANCE_MAX),
        DEFAULT_RUNTIME_N_RESONANCE,
    )
    log_speed = _axis_with_midpoints(
        np.linspace(
            np.log(DEFAULT_RUNTIME_SPEED_MIN),
            np.log(DEFAULT_RUNTIME_SPEED_MAX),
            DEFAULT_RUNTIME_N_SPEED,
        ).astype(np.float32).astype(np.float64),
        DEFAULT_RUNTIME_SPEED_REFINEMENT_INTERVALS,
    )
    pitch = _axis_with_midpoints(
        np.linspace(
            -DEFAULT_RUNTIME_PITCH_MAX,
            DEFAULT_RUNTIME_PITCH_MAX,
            DEFAULT_RUNTIME_N_PITCH,
        ).astype(np.float32).astype(np.float64),
        DEFAULT_RUNTIME_PITCH_REFINEMENT_INTERVALS,
    )
    pitch = _axis_with_midpoints(
        pitch.astype(np.float32).astype(np.float64),
        DEFAULT_RUNTIME_PITCH_SECOND_REFINEMENT_INTERVALS,
    )
    return log_resonance, log_speed, pitch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("yldh04_balanced_gyro_table_f32.npz"),
    )
    parser.add_argument(
        "--fortran-include",
        type=Path,
        default=None,
        help="runtime include path; defaults to the output stem plus .inc",
    )
    parser.add_argument(
        "--n-resonance", type=int, default=DEFAULT_RUNTIME_N_RESONANCE
    )
    parser.add_argument(
        "--n-speed",
        type=int,
        default=None,
        help="uniform speed-axis override; default uses 18 audited nodes",
    )
    parser.add_argument(
        "--n-pitch",
        type=int,
        default=None,
        help="uniform pitch-axis override; default uses 21 audited nodes",
    )
    parser.add_argument(
        "--resonance-min",
        type=float,
        default=DEFAULT_RUNTIME_RESONANCE_MIN,
    )
    parser.add_argument(
        "--resonance-max",
        type=float,
        default=DEFAULT_RUNTIME_RESONANCE_MAX,
    )
    parser.add_argument(
        "--speed-min",
        type=float,
        default=DEFAULT_RUNTIME_SPEED_MIN,
    )
    parser.add_argument(
        "--speed-max",
        type=float,
        default=DEFAULT_RUNTIME_SPEED_MAX,
    )
    parser.add_argument(
        "--pitch-max",
        type=float,
        default=DEFAULT_RUNTIME_PITCH_MAX,
    )
    parser.add_argument(
        "--quadrature-order",
        type=int,
        default=DEFAULT_RUNTIME_QUADRATURE_ORDER,
    )
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
        help="comma-separated subset for --legacy-cgs-table only",
    )
    parser.add_argument(
        "--legacy-cgs-table",
        action="store_true",
        help="write the original dimensional diagnostic instead",
    )
    args = parser.parse_args()

    context = yld04_cnm_context()
    grain = _default_grain()
    n_speed = (
        DEFAULT_RUNTIME_N_SPEED
        if args.n_speed is None
        else args.n_speed
    )
    n_pitch = (
        DEFAULT_RUNTIME_N_PITCH
        if args.n_pitch is None
        else args.n_pitch
    )
    pitch = np.linspace(-args.pitch_max, args.pitch_max, n_pitch)
    digest = None
    manifest = None
    include_path = None
    include_digest = None
    include_manifest = None
    started = time.perf_counter()
    if args.legacy_cgs_table:
        speed = np.geomspace(1.0e4, 5.0e6, n_speed)
        modes = tuple(
            item.strip() for item in args.modes.split(",") if item.strip()
        )
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
        shape = [n_speed, n_pitch]
    else:
        if args.n_speed is None and args.n_pitch is None and (
            args.n_resonance == DEFAULT_RUNTIME_N_RESONANCE
            and args.resonance_min == DEFAULT_RUNTIME_RESONANCE_MIN
            and args.resonance_max == DEFAULT_RUNTIME_RESONANCE_MAX
            and args.speed_min == DEFAULT_RUNTIME_SPEED_MIN
            and args.speed_max == DEFAULT_RUNTIME_SPEED_MAX
            and args.pitch_max == DEFAULT_RUNTIME_PITCH_MAX
        ):
            log_resonance, log_speed, pitch = (
                default_runtime_dimensionless_axes()
            )
        else:
            log_resonance = np.linspace(
                np.log(args.resonance_min),
                np.log(args.resonance_max),
                args.n_resonance,
            )
            log_speed = np.linspace(
                np.log(args.speed_min),
                np.log(args.speed_max),
                n_speed,
            )
        modes = ("alfven-balanced-absn1", "fast-lowbeta-absn1")
        table = build_dimensionless_balanced_gyro_ru_table(
            context,
            grain,
            log_resonance,
            log_speed,
            pitch,
            quadrature_order=args.quadrature_order,
            integration_rtol=args.integration_rtol,
            integration_limit=args.integration_limit,
        )
        digest, manifest = write_dimensionless_balanced_gyro_ru_table(
            args.output, table, context, grain
        )
        include_path = (
            args.fortran_include
            if args.fortran_include is not None
            else args.output.with_suffix(".inc")
        )
        include_digest, include_manifest = (
            write_cuda_fortran_balanced_gyro_include(
                include_path, table, digest
            )
        )
        shape = [
            log_resonance.size,
            log_speed.size,
            pitch.size,
        ]
    elapsed = time.perf_counter() - started
    node_count = int(np.prod(shape))
    summary = {
        "output": str(args.output),
        "bytes": args.output.stat().st_size,
        "model_revision": MODEL_REVISION,
        "table_revision": table["table_revision"],
        "shape": shape,
        "nodes": node_count,
        "generation_seconds": elapsed,
        "seconds_per_node": elapsed / node_count,
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
    if digest is not None:
        summary["sha256"] = digest
        summary["sha256_manifest"] = str(manifest)
    if include_digest is not None:
        summary["fortran_include"] = str(include_path)
        summary["fortran_include_sha256"] = include_digest
        summary["fortran_include_sha256_manifest"] = str(
            include_manifest
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
