#!/usr/bin/env python3
"""Stage 9 equilibrium grain charging and compact runtime-table generator.

The provisional cell closure smoothly joins an ion-composition-aware dark OML
equilibrium to the published illuminated-silicate centroid of Ibanez-Mejia et
al. (2019), their Equations 17--19 and Table 1.  The join is an empirical
asymptotic blend, not a published current-balance solution.  It lets the
runtime consume Stage 8's external shielded radiation directly without
pretending that it includes the paper's CR-induced FUV component.  The current
HD23 active population has one astrodust material, so silicate is an explicit
provisional proxy until material-specific astrodust charging data exist.

The fit fixes the mean potential at ``a_ref = 0.1 micron``.  For the active
0.0575--0.92 micron grains, ``<Z>`` scales with radius and the variance scales
with radius, as expected for the large-grain constant-potential limit.  The
published coefficients end at 0.1 micron.  Applying the 0.1-micron silicate
fit with constant-potential scaling across the active span, especially through
0.92 micron, is an explicit extrapolation rather than published HD23 accuracy.

The offline reference uses float64.  ``build_runtime_table`` and
``write_fortran_include`` emit only float32 data for the production lookup.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ANGSTROM_CM = 1.0e-8
MICRON_CM = 1.0e-4
YEAR_S = 365.25 * 86400.0

E_ESU = 4.803204712570263e-10
K_BOLTZMANN = 1.380649e-16
M_ELECTRON = 9.1093837015e-28
M_HYDROGEN = 1.6735575e-24
M_CARBON = 12.0 * M_HYDROGEN
C_LIGHT = 2.99792458e10

REFERENCE_RADIUS_ANGSTROM = 1000.0
REFERENCE_RADIUS_CM = REFERENCE_RADIUS_ANGSTROM * ANGSTROM_CM
ACTIVE_MIN_MICRON = 0.057493992527317586
ACTIVE_MAX_MICRON = 0.9199038804370818
N_SIZE_KNOTS = 12
SIZE_KNOT_RADII_CM = np.geomspace(
    ACTIVE_MIN_MICRON * MICRON_CM,
    ACTIVE_MAX_MICRON * MICRON_CM,
    N_SIZE_KNOTS,
    dtype=np.float64,
)
# Backward-compatible analysis name.  These are interpolation knots, not the
# twelve continuous Stage 6/7 particle-size strata.
N_FAMILIES = N_SIZE_KNOTS
FAMILY_RADII_CM = SIZE_KNOT_RADII_CM

TABLE_AXIS_SCALE = 0.25
TABLE_ZREF_MIN = -512.0
TABLE_ZREF_MAX = 8192.0
TABLE_N_AXIS = 257
CLOSURE_REVISION = "DSOML-HCplus-IM19-asymptotic-blend-v2"
TABLE_REVISION = "IM19-silicate-size-knot-moments-aref1000A-v2"
NEUTRALITY_RTOL = 1.0e-5


@dataclass(frozen=True)
class SilicateFit:
    """Published Table 1 coefficients for one silicate grain radius."""

    radius_angstrom: float
    alpha: float
    k: float
    b: float
    h_z: float
    c_positive: float
    eta_positive: float
    d: float
    c_negative: float
    eta_negative: float


# Published Table 1 values.  These intentionally do not use the older values
# in the companion DustCharge repository.
SILICATE_FITS = {
    fit.radius_angstrom: fit
    for fit in (
        SilicateFit(3.5, 0.3263, 0.0149, -0.1212, 57.0, 0.4123, 0.2513, 0.1891, 0.4845, 0.3532),
        SilicateFit(5.0, 0.3141, 0.0372, -0.3043, 86.0, 0.2734, 0.2925, 0.3233, 0.3615, 0.6532),
        SilicateFit(10.0, 0.3535, 0.0494, -0.4865, 73.0, 0.4353, 0.7459, 0.4451, 0.1053, 0.5803),
        SilicateFit(50.0, 0.5115, 0.0717, -0.4106, 107.0, 1.0758, 1.7832, 0.5860, -1.0379e3, 7.7069e3),
        SilicateFit(100.0, 0.3525, 0.6591, -0.1649, 384.0, 1.6245, 2.8390, 0.6346, -4.2075e2, 1.9840e3),
        SilicateFit(500.0, 0.3643, 2.6283, 0.5217, 345.0, 4.0732, 11.0200, 0.6797, -0.2418, 0.5910),
        SilicateFit(1000.0, 0.3927, 3.6493, 0.8389, 372.0, 5.9813, 20.6410, 0.6961, -0.1885, 0.4237),
    )
}
REFERENCE_FIT = SILICATE_FITS[REFERENCE_RADIUS_ANGSTROM]


@dataclass(frozen=True)
class ChargeMoments:
    """Normalized discrete-charge moments needed by Stage 9."""

    normalization: float
    mean: float
    second: float
    variance: float
    z2_log_abs_z: float


@dataclass(frozen=True)
class Environment:
    """One charging and timescale audit state in cgs units."""

    name: str
    n_h_cm3: float
    temperature_k: float
    electron_fraction: float
    hydrogen_ion_fraction: float
    radiation_habing: float
    magnetic_field_gauss: float
    timestep_s: float

    @property
    def electron_density_cm3(self) -> float:
        return self.n_h_cm3 * self.electron_fraction


REPRESENTATIVE_ENVIRONMENTS = (
    Environment("WNM", 0.9, 7000.0, 0.012, 0.010, 1.52, 5.0e-6, YEAR_S),
    Environment("CNM", 36.0, 70.0, 1.8e-4, 1.0e-4, 0.60, 10.0e-6, YEAR_S),
    Environment("WIM", 0.01, 8000.0, 1.0, 1.0, 1.0, 5.0e-6, YEAR_S),
    # External shielded radiation only: no implicit CR-induced FUV.
    Environment("shielded_molecular", 3.0e4, 14.4, 9.0e-8, 0.0, 0.0, 100.0e-6, YEAR_S),
)

# Exact T, nH, ne, GUV, and B entries from the idealized ISM phases in YLD04
# Table 1.  That table does not split the positive charge between H+ and heavy
# ions.  The audit therefore assigns the electron density to H+ in the WNM and
# WIM and to the effective singly charged heavy-ion proxy in neutral/cloud
# phases.  The one-year timestep is an audit choice, not a YLD04 table entry.
YLD04_TABLE1_ENVIRONMENTS = (
    Environment("YLD04_CNM", 30.0, 100.0, 0.03 / 30.0, 0.0, 1.0, 6.0e-6, YEAR_S),
    Environment("YLD04_WNM", 0.3, 6000.0, 0.03 / 0.3, 0.03 / 0.3, 1.0, 5.8e-6, YEAR_S),
    Environment("YLD04_WIM", 0.1, 8000.0, 0.0991 / 0.1, 0.0991 / 0.1, 1.0, 3.35e-6, YEAR_S),
    Environment("YLD04_MC", 300.0, 25.0, 0.03 / 300.0, 0.0, 0.1, 11.0e-6, YEAR_S),
    Environment("YLD04_DC1", 1.0e4, 10.0, 0.01 / 1.0e4, 0.0, 0.01, 80.0e-6, YEAR_S),
    Environment("YLD04_DC2", 1.0e4, 10.0, 0.001 / 1.0e4, 0.0, 0.001, 80.0e-6, YEAR_S),
)

CANONICAL_ENVIRONMENTS = (
    *REPRESENTATIVE_ENVIRONMENTS,
    *YLD04_TABLE1_ENVIRONMENTS,
)


def _positive_finite(name: str, value) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if np.any(~np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must be finite and positive")
    return array


def charging_parameter(temperature_k, electron_density_cm3, radiation_habing):
    """Return ``psi = Gtot sqrt(T) / ne`` from Equation 17."""
    temperature = _positive_finite("temperature_k", temperature_k)
    electron_density = _positive_finite(
        "electron_density_cm3", electron_density_cm3
    )
    radiation = np.asarray(radiation_habing, dtype=np.float64)
    if np.any(~np.isfinite(radiation)) or np.any(radiation < 0.0):
        raise ValueError("radiation_habing must be finite and nonnegative")
    return radiation * np.sqrt(temperature) / electron_density


def dark_potential_nu(
    n_h_cm3,
    electron_density_cm3,
    hydrogen_ion_fraction,
):
    """Solve the H+/heavy-ion dark OML balance for ``e phi/(kT)``.

    ``n_H+ = xHII nH`` and ``n_heavy = max(ne-n_H+, 0)``.  Both ions are
    singly charged; the residual channel uses carbon mass as an effective
    heavy-ion proxy.  It is not literally C+ in every environment.
    """
    n_h = _positive_finite("n_h_cm3", n_h_cm3)
    electron_density = _positive_finite(
        "electron_density_cm3", electron_density_cm3
    )
    x_hii = np.asarray(hydrogen_ion_fraction, dtype=np.float64)
    if (
        np.any(~np.isfinite(x_hii))
        or np.any(x_hii < 0.0)
        or np.any(x_hii > 1.0 + NEUTRALITY_RTOL)
    ):
        raise ValueError("hydrogen_ion_fraction must lie in [0, 1]")
    n_h, electron_density, x_hii = np.broadcast_arrays(
        n_h, electron_density, x_hii
    )
    n_h_plus = x_hii * n_h
    if np.any(n_h_plus > electron_density * (1.0 + NEUTRALITY_RTOL)):
        raise ValueError("xHII*nH cannot materially exceed ne")
    n_h_plus = np.minimum(n_h_plus, electron_density)
    n_c_plus = np.maximum(electron_density - n_h_plus, 0.0)
    current_ratio = (
        n_h_plus * np.sqrt(M_ELECTRON / M_HYDROGEN)
        + n_c_plus * np.sqrt(M_ELECTRON / M_CARBON)
    ) / electron_density
    lower = np.where(current_ratio <= 1.0, -32.0, 0.0)
    upper = np.where(current_ratio <= 1.0, 0.0, 32.0)
    for _ in range(56):
        midpoint = 0.5 * (lower + upper)
        residual = np.where(
            midpoint >= 0.0,
            1.0 + midpoint - current_ratio * np.exp(-midpoint),
            np.exp(midpoint) - current_ratio * (1.0 - midpoint),
        )
        lower = np.where(residual < 0.0, midpoint, lower)
        upper = np.where(residual < 0.0, upper, midpoint)
    return 0.5 * (lower + upper)


def photocharging_increment(
    temperature_k,
    electron_density_cm3,
    radiation_habing,
):
    """Return the external-FUV-activated IM19 illuminated centroid."""
    psi = charging_parameter(
        temperature_k, electron_density_cm3, radiation_habing
    )
    fit = REFERENCE_FIT
    activation = -np.expm1(-psi / fit.h_z)
    return activation * (fit.b + fit.k * np.power(psi, fit.alpha))


def reference_mean_charge(
    n_h_cm3,
    temperature_k,
    electron_density_cm3,
    hydrogen_ion_fraction,
    radiation_habing,
):
    """Return the empirical dark-to-illuminated blend at 0.1 micron.

    The exponential weight enforces the dark OML limit at zero external FUV
    and the published IM19 centroid ``b+k psi**alpha`` at large ``psi``.
    It is a practical asymptotic interpolation, not a published current
    balance between those limits.
    """
    temperature = _positive_finite("temperature_k", temperature_k)
    dark_nu = dark_potential_nu(
        n_h_cm3, electron_density_cm3, hydrogen_ion_fraction
    )
    dark_charge = (
        dark_nu
        * REFERENCE_RADIUS_CM
        * K_BOLTZMANN
        * temperature
        / (E_ESU * E_ESU)
    )
    psi = charging_parameter(
        temperature, electron_density_cm3, radiation_habing
    )
    dark_weight = np.exp(-psi / REFERENCE_FIT.h_z)
    return dark_weight * dark_charge + photocharging_increment(
        temperature, electron_density_cm3, radiation_habing
    )


def reference_mean_charge_float32(
    n_h_cm3,
    temperature_k,
    electron_density_cm3,
    hydrogen_ion_fraction,
    radiation_habing,
):
    """Emulate the production float32 hybrid closure."""
    n_h = np.asarray(n_h_cm3, dtype=np.float32)
    temperature = np.asarray(temperature_k, dtype=np.float32)
    electron_density = np.asarray(electron_density_cm3, dtype=np.float32)
    x_hii = np.asarray(hydrogen_ion_fraction, dtype=np.float32)
    radiation = np.asarray(radiation_habing, dtype=np.float32)
    if (
        np.any(~np.isfinite(n_h))
        or np.any(n_h <= np.float32(0.0))
        or np.any(~np.isfinite(temperature))
        or np.any(temperature <= np.float32(0.0))
        or np.any(~np.isfinite(electron_density))
        or np.any(electron_density <= np.float32(0.0))
        or np.any(~np.isfinite(x_hii))
        or np.any(x_hii < np.float32(0.0))
        or np.any(x_hii > np.float32(1.0 + NEUTRALITY_RTOL))
        or np.any(~np.isfinite(radiation))
        or np.any(radiation < np.float32(0.0))
    ):
        raise ValueError("float32 charging inputs are outside their valid domain")
    n_h, temperature, electron_density, x_hii, radiation = np.broadcast_arrays(
        n_h, temperature, electron_density, x_hii, radiation
    )
    n_h_plus = x_hii * n_h
    if np.any(
        n_h_plus
        > electron_density * np.float32(1.0 + NEUTRALITY_RTOL)
    ):
        raise ValueError("float32 charging inputs violate charge neutrality")
    n_h_plus = np.minimum(n_h_plus, electron_density)
    n_c_plus = np.maximum(
        electron_density - n_h_plus, np.float32(0.0)
    )
    ratio = (
        n_h_plus
        * np.sqrt(np.float32(M_ELECTRON / M_HYDROGEN))
        + n_c_plus * np.sqrt(np.float32(M_ELECTRON / M_CARBON))
    ) / electron_density
    lower = np.where(
        ratio <= np.float32(1.0), np.float32(-32.0), np.float32(0.0)
    )
    upper = np.where(
        ratio <= np.float32(1.0), np.float32(0.0), np.float32(32.0)
    )
    for _ in range(32):
        midpoint = np.float32(0.5) * (lower + upper)
        residual = np.where(
            midpoint >= np.float32(0.0),
            np.float32(1.0)
            + midpoint
            - ratio * np.exp(-midpoint),
            np.exp(midpoint) - ratio * (np.float32(1.0) - midpoint),
        )
        lower = np.where(residual < np.float32(0.0), midpoint, lower)
        upper = np.where(residual < np.float32(0.0), upper, midpoint)
    dark_nu = np.float32(0.5) * (lower + upper)
    dark_charge = (
        dark_nu
        * np.float32(REFERENCE_RADIUS_CM)
        * np.float32(K_BOLTZMANN)
        * temperature
        / np.float32(E_ESU * E_ESU)
    )
    psi = radiation * np.sqrt(temperature) / electron_density
    k = np.float32(REFERENCE_FIT.k)
    b = np.float32(REFERENCE_FIT.b)
    h_z = np.float32(REFERENCE_FIT.h_z)
    alpha = np.float32(REFERENCE_FIT.alpha)
    activation = -np.expm1(-psi / h_z)
    photo = activation * (b + k * np.power(psi, alpha))
    return np.exp(-psi / h_z) * dark_charge + photo


def reference_width(reference_charge):
    """Return the fitted 0.1 micron charge-distribution width."""
    charge = np.asarray(reference_charge, dtype=np.float64)
    fit = REFERENCE_FIT
    positive = fit.c_positive * (
        -np.expm1(-np.maximum(charge, 0.0) / fit.eta_positive)
    ) + fit.d
    negative = fit.c_negative * (
        -np.expm1(-np.maximum(-charge, 0.0) / fit.eta_negative)
    ) + fit.d
    return np.where(charge >= 0.0, positive, negative)


def mean_potential_statvolt(reference_charge):
    """Return the material cell potential ``phi = e <Zref> / aref``."""
    return E_ESU * np.asarray(reference_charge, dtype=np.float64) / REFERENCE_RADIUS_CM


def grain_mean_charge(radius_cm, reference_charge):
    """Return ``<Z(a)> = (a/aref) <Zref>``."""
    radius = _positive_finite("radius_cm", radius_cm)
    return radius / REFERENCE_RADIUS_CM * np.asarray(
        reference_charge, dtype=np.float64
    )


def grain_charge_width(radius_cm, reference_charge):
    """Return the constant-potential large-grain width, ``sigma^2 propto a``."""
    radius = _positive_finite("radius_cm", radius_cm)
    return reference_width(reference_charge) * np.sqrt(
        radius / REFERENCE_RADIUS_CM
    )


def charge_to_mass_cgs(
    radius_cm,
    grain_density_g_cm3,
    potential_statvolt,
):
    """Return ``q/m = 3 phi / (4 pi rho_gr a^2)`` in Gaussian cgs."""
    radius = _positive_finite("radius_cm", radius_cm)
    density = _positive_finite(
        "grain_density_g_cm3", grain_density_g_cm3
    )
    return (
        3.0
        * np.asarray(potential_statvolt, dtype=np.float64)
        / (4.0 * np.pi * density * radius**2)
    )


def charge_parameter_code(
    charge_to_mass_esu_per_g,
    units_length_cm,
    units_density_g_cm3,
):
    """Convert physical ``q/m`` to the dimensionless RAMSES coefficient.

    In Gaussian cgs, ``dv/dt=(q/mc) v cross B`` and RAMSES uses
    ``B_phys=v_unit sqrt(4 pi rho_unit) B_code``.  With
    ``t_unit=L_unit/v_unit``, the dimensionless multiplier of
    ``v_code cross B_code`` is therefore

    ``(q/m) L_unit sqrt(4 pi rho_unit) / c``.
    """
    charge_to_mass = np.asarray(charge_to_mass_esu_per_g, dtype=np.float64)
    if np.any(~np.isfinite(charge_to_mass)):
        raise ValueError("charge_to_mass_esu_per_g must be finite")
    unit_l = _positive_finite("units_length_cm", units_length_cm)
    unit_d = _positive_finite(
        "units_density_g_cm3", units_density_g_cm3
    )
    return (
        charge_to_mass
        * unit_l
        * np.sqrt(4.0 * np.pi * unit_d)
        / C_LIGHT
    )


def grain_charge_parameter_code(
    radius_cm,
    grain_density_g_cm3,
    potential_statvolt,
    units_length_cm,
    units_density_g_cm3,
):
    """Return the dimensionless code charge parameter for one grain."""
    return charge_parameter_code(
        charge_to_mass_cgs(
            radius_cm, grain_density_g_cm3, potential_statvolt
        ),
        units_length_cm,
        units_density_g_cm3,
    )


def discrete_charge_distribution(
    mean_charge: float,
    sigma_charge: float,
    *,
    tail_sigma: float = 12.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a normalized integer-Z Gaussian approximation to ``f_Z``.

    The continuous Gaussian location is adjusted so that the normalized
    integer distribution has the requested first moment.  This matters on the
    negative branch, where the fitted width approaches half a charge state.
    """
    if not np.isfinite(mean_charge):
        raise ValueError("mean_charge must be finite")
    if not np.isfinite(sigma_charge) or sigma_charge <= 0.0:
        raise ValueError("sigma_charge must be finite and positive")
    if tail_sigma < 8.0:
        raise ValueError("tail_sigma must be at least 8")

    def at_location(location: float) -> tuple[np.ndarray, np.ndarray, float]:
        half_width = max(8.0, tail_sigma * sigma_charge)
        lower = int(np.floor(location - half_width))
        upper = int(np.ceil(location + half_width))
        charge = np.arange(lower, upper + 1, dtype=np.int64)
        exponent = -0.5 * (
            (charge.astype(np.float64) - location) / sigma_charge
        ) ** 2
        exponent -= np.max(exponent)
        probability = np.exp(exponent)
        probability /= np.sum(probability, dtype=np.float64)
        first = float(np.dot(probability, charge.astype(np.float64)))
        return charge, probability, first

    lower_location = mean_charge - 1.0
    upper_location = mean_charge + 1.0
    for _ in range(64):
        midpoint = 0.5 * (lower_location + upper_location)
        _, _, first = at_location(midpoint)
        if first < mean_charge:
            lower_location = midpoint
        else:
            upper_location = midpoint
    charge, probability, _ = at_location(
        0.5 * (lower_location + upper_location)
    )
    return charge, probability


def charge_moments(charge: np.ndarray, probability: np.ndarray) -> ChargeMoments:
    """Return normalization, first, second, variance, and log moment."""
    z = np.asarray(charge, dtype=np.float64)
    f_z = np.asarray(probability, dtype=np.float64)
    if z.ndim != 1 or f_z.shape != z.shape or z.size == 0:
        raise ValueError("charge and probability must be matching 1-D arrays")
    if np.any(~np.isfinite(f_z)) or np.any(f_z < 0.0):
        raise ValueError("probability must be finite and nonnegative")
    normalization = float(np.sum(f_z, dtype=np.float64))
    if normalization <= 0.0:
        raise ValueError("probability has zero normalization")
    normalized = f_z / normalization
    mean = float(np.dot(normalized, z))
    second = float(np.dot(normalized, z * z))
    log_abs_z = np.zeros_like(z)
    nonzero = z != 0.0
    log_abs_z[nonzero] = np.log(np.abs(z[nonzero]))
    z2_log = float(np.dot(normalized, z * z * log_abs_z))
    return ChargeMoments(
        normalization=float(np.sum(normalized, dtype=np.float64)),
        mean=mean,
        second=second,
        variance=max(second - mean * mean, 0.0),
        z2_log_abs_z=z2_log,
    )


def distribution_for_radius(
    radius_cm: float,
    reference_charge: float,
) -> tuple[np.ndarray, np.ndarray, ChargeMoments]:
    """Construct the normalized large-grain ``f_Z`` for one radius."""
    mean = float(grain_mean_charge(radius_cm, reference_charge))
    width = float(grain_charge_width(radius_cm, reference_charge))
    charge, probability = discrete_charge_distribution(mean, width)
    return charge, probability, charge_moments(charge, probability)


def coulomb_log_moment(
    charge: np.ndarray,
    probability: np.ndarray,
    log_c: float,
) -> float:
    """Return ``sum_Z f_Z Z^2 ln(C/|Z|)`` without a charge floor."""
    if not np.isfinite(log_c):
        raise ValueError("log_c must be finite")
    moments = charge_moments(charge, probability)
    return moments.second * log_c - moments.z2_log_abs_z


def yld04_coulomb_log_scale(
    temperature_k: float,
    ion_density_cm3: float,
) -> float:
    """Return ``ln(A)`` in the YLD04 charge-distribution drag moment.

    ``A = 3 (kT)^(3/2) / [2 e^3 sqrt(pi ni)]`` in Gaussian cgs, so that
    ``ln Lambda(Z) = ln(A/abs(Z))`` for nonzero integer charge.
    """
    temperature = float(_positive_finite("temperature_k", temperature_k))
    ion_density = float(
        _positive_finite("ion_density_cm3", ion_density_cm3)
    )
    return float(
        np.log(
            3.0 * (K_BOLTZMANN * temperature) ** 1.5
            / (
                2.0
                * E_ESU**3
                * np.sqrt(np.pi * ion_density)
            )
        )
    )


def ion_number_densities_cm3(
    n_h_cm3: float,
    electron_density_cm3: float,
    hydrogen_ion_fraction: float,
) -> tuple[float, float]:
    """Return H+ and effective singly charged heavy-ion number densities."""
    n_h = float(_positive_finite("n_h_cm3", n_h_cm3))
    electron_density = float(
        _positive_finite("electron_density_cm3", electron_density_cm3)
    )
    x_hii = float(hydrogen_ion_fraction)
    if (
        not np.isfinite(x_hii)
        or x_hii < 0.0
        or x_hii > 1.0 + NEUTRALITY_RTOL
    ):
        raise ValueError("hydrogen_ion_fraction must lie in [0, 1]")
    n_h_plus = x_hii * n_h
    if n_h_plus > electron_density * (1.0 + NEUTRALITY_RTOL):
        raise ValueError("xHII*nH cannot materially exceed ne")
    n_h_plus = min(n_h_plus, electron_density)
    return n_h_plus, max(electron_density - n_h_plus, 0.0)


def coulomb_drag_rate_s(
    radius_cm: float,
    grain_density_g_cm3: float,
    n_h_cm3: float,
    temperature_k: float,
    electron_density_cm3: float,
    hydrogen_ion_fraction: float,
    charge: np.ndarray,
    probability: np.ndarray,
) -> tuple[float, float, float]:
    """Return subsonic distribution-averaged Coulomb rate and diagnostics.

    H+ and the residual singly charged heavy-ion proxy are summed with their
    separate ``n_i sqrt(m_i)`` weights.  Charge neutrality makes their total
    number density equal to ``ne``.  The returned tuple is
    ``(nu_coulomb, weighted_charge_moment, lnA)``.
    """
    radius = float(_positive_finite("radius_cm", radius_cm))
    density = float(
        _positive_finite("grain_density_g_cm3", grain_density_g_cm3)
    )
    temperature = float(_positive_finite("temperature_k", temperature_k))
    electron_density = float(
        _positive_finite("electron_density_cm3", electron_density_cm3)
    )
    n_h_plus, n_heavy = ion_number_densities_cm3(
        n_h_cm3, electron_density, hydrogen_ion_fraction
    )
    log_c = yld04_coulomb_log_scale(temperature, electron_density)
    moment = max(
        coulomb_log_moment(charge, probability, log_c),
        0.0,
    )
    grain_mass = 4.0 * np.pi * density * radius**3 / 3.0
    species_weight = (
        n_h_plus * np.sqrt(M_HYDROGEN)
        + n_heavy * np.sqrt(M_CARBON)
    )
    prefactor = (
        (8.0 * np.pi / 3.0)
        * E_ESU**4
        * species_weight
        / (
            np.sqrt(2.0 * np.pi)
            * grain_mass
            * (K_BOLTZMANN * temperature) ** 1.5
        )
    )
    return float(prefactor * moment), float(moment), log_c


def electron_collection_rate_s(
    radius_cm: float,
    charge,
    temperature_k: float,
    electron_density_cm3: float,
    *,
    sticking: float = 0.5,
):
    """Conservative large-grain electron arrival rate for the timescale audit.

    The classical attractive/repulsive Coulomb factors retain the current
    scale.  Draine--Sutin polarization corrections are omitted and can be
    material at low reduced temperature, even for the active radii.  This is
    therefore an order-of-magnitude separation audit, not a WD01 current
    solver.
    """
    radius = float(_positive_finite("radius_cm", radius_cm))
    temperature = float(_positive_finite("temperature_k", temperature_k))
    electron_density = float(
        _positive_finite("electron_density_cm3", electron_density_cm3)
    )
    if not np.isfinite(sticking) or not 0.0 < sticking <= 1.0:
        raise ValueError("sticking must lie in (0, 1]")
    z = np.asarray(charge, dtype=np.float64)
    tau = radius * K_BOLTZMANN * temperature / (E_ESU * E_ESU)
    focusing = np.empty_like(z)
    attractive = z >= 0.0
    focusing[attractive] = 1.0 + z[attractive] / tau
    focusing[~attractive] = np.exp(
        np.maximum(z[~attractive] / tau, -80.0)
    )
    thermal_speed = np.sqrt(
        8.0 * K_BOLTZMANN * temperature / (np.pi * M_ELECTRON)
    )
    return (
        sticking
        * electron_density
        * np.pi
        * radius**2
        * thermal_speed
        * focusing
    )


def charging_time_s(
    radius_cm: float,
    temperature_k: float,
    electron_density_cm3: float,
    charge: np.ndarray,
    probability: np.ndarray,
) -> float:
    """Estimate ``tau_Z = sigma_Z^2 / sum f_Z J_tot(Z)``.

    The fitted equilibrium ``f_Z`` and electron collection current set the
    rate scale.  Detailed balance gives an equal probability-weighted reverse
    current, hence ``sum f_Z J_tot = 2 sum f_Z J_e``.  This is a conservative,
    explicitly documented audit estimate, not a runtime field.
    """
    moments = charge_moments(charge, probability)
    electron_rate = electron_collection_rate_s(
        radius_cm,
        charge,
        temperature_k,
        electron_density_cm3,
    )
    total_transition_rate = 2.0 * float(
        np.dot(np.asarray(probability, dtype=np.float64), electron_rate)
    )
    if total_transition_rate <= 0.0:
        return np.inf
    return moments.variance / total_transition_rate


def larmor_time_s(
    radius_cm: float,
    grain_density_g_cm3: float,
    mean_charge: float,
    magnetic_field_gauss: float,
) -> float:
    """Return the inverse gyrofrequency ``1/|Omega|`` (stricter than 2pi/Omega)."""
    radius = float(_positive_finite("radius_cm", radius_cm))
    density = float(
        _positive_finite("grain_density_g_cm3", grain_density_g_cm3)
    )
    magnetic_field = float(
        _positive_finite("magnetic_field_gauss", magnetic_field_gauss)
    )
    if mean_charge == 0.0:
        return np.inf
    grain_mass = 4.0 * np.pi * density * radius**3 / 3.0
    return grain_mass * C_LIGHT / (
        abs(mean_charge) * E_ESU * magnetic_field
    )


def epstein_drag_time_s(
    radius_cm: float,
    grain_density_g_cm3: float,
    n_h_cm3: float,
    temperature_k: float,
    *,
    mean_mass_per_h: float = 1.4,
) -> float:
    """Return a thermal Epstein stopping-time estimate."""
    radius = float(_positive_finite("radius_cm", radius_cm))
    density = float(
        _positive_finite("grain_density_g_cm3", grain_density_g_cm3)
    )
    n_h = float(_positive_finite("n_h_cm3", n_h_cm3))
    temperature = float(_positive_finite("temperature_k", temperature_k))
    gas_density = mean_mass_per_h * M_HYDROGEN * n_h
    thermal_speed = np.sqrt(
        8.0 * K_BOLTZMANN * temperature
        / (np.pi * mean_mass_per_h * M_HYDROGEN)
    )
    return density * radius / (gas_density * thermal_speed)


def audit_environment(
    environment: Environment,
    *,
    grain_density_g_cm3: float = 2.0,
) -> list[dict[str, float]]:
    """Audit equilibrium charging at all twelve runtime size knots."""
    z_ref = float(
        reference_mean_charge(
            environment.n_h_cm3,
            environment.temperature_k,
            environment.electron_density_cm3,
            environment.hydrogen_ion_fraction,
            environment.radiation_habing,
        )
    )
    records: list[dict[str, float]] = []
    for size_knot, radius in enumerate(SIZE_KNOT_RADII_CM, start=1):
        charge, probability, moments = distribution_for_radius(radius, z_ref)
        tau_z = charging_time_s(
            radius,
            environment.temperature_k,
            environment.electron_density_cm3,
            charge,
            probability,
        )
        tau_l = larmor_time_s(
            radius,
            grain_density_g_cm3,
            moments.mean,
            environment.magnetic_field_gauss,
        )
        tau_epstein = epstein_drag_time_s(
            radius,
            grain_density_g_cm3,
            environment.n_h_cm3,
            environment.temperature_k,
        )
        nu_coulomb, coulomb_moment, log_c = coulomb_drag_rate_s(
            radius,
            grain_density_g_cm3,
            environment.n_h_cm3,
            environment.temperature_k,
            environment.electron_density_cm3,
            environment.hydrogen_ion_fraction,
            charge,
            probability,
        )
        tau_coulomb = (
            1.0 / nu_coulomb if nu_coulomb > 0.0 else np.inf
        )
        tau_drag = 1.0 / (1.0 / tau_epstein + nu_coulomb)
        comparison = min(environment.timestep_s, tau_l, tau_drag)
        records.append(
            {
                "size_knot": size_knot,
                "radius_micron": radius / MICRON_CM,
                "reference_mean_charge": z_ref,
                "mean_charge": moments.mean,
                "variance": moments.variance,
                "tau_z_s": tau_z,
                "tau_l_s": tau_l,
                "tau_epstein_s": tau_epstein,
                "tau_coulomb_s": tau_coulomb,
                "tau_drag_total_s": tau_drag,
                "coulomb_log_scale": log_c,
                "coulomb_charge_moment": coulomb_moment,
                "dt_s": environment.timestep_s,
                "tau_z_over_min": tau_z / comparison,
            }
        )
    return records


def _runtime_axis() -> tuple[np.ndarray, np.ndarray]:
    s_min = np.arcsinh(TABLE_ZREF_MIN / TABLE_AXIS_SCALE)
    s_max = np.arcsinh(TABLE_ZREF_MAX / TABLE_AXIS_SCALE)
    transformed = np.linspace(s_min, s_max, TABLE_N_AXIS, dtype=np.float64)
    reference_charge = TABLE_AXIS_SCALE * np.sinh(transformed)
    return transformed, reference_charge


def build_runtime_table() -> dict[str, np.ndarray | float | int | str]:
    """Build the compact float32 table used by the production drag closure."""
    transformed, reference_charge = _runtime_axis()
    second = np.empty((N_SIZE_KNOTS, TABLE_N_AXIS), dtype=np.float32)
    z2_log = np.empty_like(second)
    for size_knot, radius in enumerate(SIZE_KNOT_RADII_CM):
        for index, z_ref in enumerate(reference_charge):
            _, _, moments = distribution_for_radius(radius, float(z_ref))
            second[size_knot, index] = np.float32(moments.second)
            z2_log[size_knot, index] = np.float32(
                moments.z2_log_abs_z
            )
    return {
        "revision": TABLE_REVISION,
        "material": "HD23 astrodust (provisional silicate proxy)",
        "axis": "s=asinh(Zref/axis_scale)",
        "axis_min": np.float32(transformed[0]),
        "axis_step": np.float32(transformed[1] - transformed[0]),
        "axis_scale": np.float32(TABLE_AXIS_SCALE),
        "reference_charge_min": np.float32(TABLE_ZREF_MIN),
        "reference_charge_max": np.float32(TABLE_ZREF_MAX),
        "axis_count": TABLE_N_AXIS,
        "size_knot_radii_cm": SIZE_KNOT_RADII_CM.astype(np.float32),
        # Compatibility with the first Stage 9 include generator.
        "family_radii_cm": SIZE_KNOT_RADII_CM.astype(np.float32),
        "second_moment": second,
        "z2_log_abs_z": z2_log,
    }


def runtime_lookup_float32(
    table: dict[str, np.ndarray | float | int | str],
    family_index: int,
    reference_charge: float,
) -> tuple[float, float]:
    """Emulate the production float32 transformed-axis linear lookup."""
    if not 0 <= family_index < N_SIZE_KNOTS:
        raise IndexError("family_index must be in [0, 12)")
    z_ref = np.float32(reference_charge)
    if (
        not np.isfinite(z_ref)
        or z_ref < np.float32(table["reference_charge_min"])
        or z_ref > np.float32(table["reference_charge_max"])
    ):
        raise ValueError("reference charge lies outside the audited runtime table")
    scale = np.float32(table["axis_scale"])
    s = np.arcsinh(z_ref / scale).astype(np.float32)
    axis_min = np.float32(table["axis_min"])
    axis_step = np.float32(table["axis_step"])
    coordinate = (s - axis_min) / axis_step
    if coordinate < np.float32(0.0) or coordinate > np.float32(TABLE_N_AXIS - 1):
        raise ValueError("reference charge lies outside the audited runtime table")
    lower = int(np.floor(coordinate))
    upper = min(lower + 1, TABLE_N_AXIS - 1)
    fraction = np.float32(coordinate - np.float32(lower))
    second = np.asarray(table["second_moment"], dtype=np.float32)
    z2_log = np.asarray(table["z2_log_abs_z"], dtype=np.float32)
    m2 = second[family_index, lower] + fraction * (
        second[family_index, upper] - second[family_index, lower]
    )
    m2log = z2_log[family_index, lower] + fraction * (
        z2_log[family_index, upper] - z2_log[family_index, lower]
    )
    return float(m2), float(m2log)


def runtime_lookup_radius_float32(
    table: dict[str, np.ndarray | float | int | str],
    radius_cm: float,
    reference_charge: float,
) -> tuple[float, float]:
    """Emulate production lookup for a continuous Stage 6/7 grain radius.

    The twelve stored radii are logarithmic interpolation knots.  Each
    bracketing knot first uses the transformed reference-charge lookup, then
    the positive moments are interpolated geometrically in log radius.  Exact
    knot queries return the underlying float32 table value.
    """
    radius = np.float32(radius_cm)
    radii = np.asarray(table["size_knot_radii_cm"], dtype=np.float32)
    if (
        not np.isfinite(radius)
        or radius < radii[0]
        or radius > radii[-1]
    ):
        raise ValueError("radius lies outside the audited size-knot range")
    upper = int(np.searchsorted(radii, radius, side="left"))
    if upper == 0:
        return runtime_lookup_float32(table, 0, reference_charge)
    if upper == N_SIZE_KNOTS:
        return runtime_lookup_float32(
            table, N_SIZE_KNOTS - 1, reference_charge
        )
    if radius == radii[upper]:
        return runtime_lookup_float32(table, upper, reference_charge)
    lower = upper - 1
    weight = np.float32(
        np.log(radius / radii[lower])
        / np.log(radii[upper] / radii[lower])
    )
    m2_lo, m2log_lo = runtime_lookup_float32(
        table, lower, reference_charge
    )
    m2_hi, m2log_hi = runtime_lookup_float32(
        table, upper, reference_charge
    )

    def positive_log_interp(lo: float, hi: float) -> float:
        lo32 = np.float32(lo)
        hi32 = np.float32(hi)
        if lo32 <= 0.0 or hi32 <= 0.0:
            return float(lo32 + weight * (hi32 - lo32))
        return float(
            np.exp(
                np.log(lo32)
                + weight * (np.log(hi32) - np.log(lo32))
            ).astype(np.float32)
        )

    return (
        positive_log_interp(m2_lo, m2_hi),
        positive_log_interp(m2log_lo, m2log_hi),
    )


def _fortran_values(values: np.ndarray, *, per_line: int = 4) -> str:
    flat = np.asarray(values, dtype=np.float32).ravel(order="C")
    tokens = [f"{value:.9e}_4" for value in flat]
    lines = []
    for start in range(0, len(tokens), per_line):
        suffix = ", &" if start + per_line < len(tokens) else " &"
        lines.append("  " + ", ".join(tokens[start : start + per_line]) + suffix)
    return "\n".join(lines)


def write_fortran_include(
    path: str | Path,
    table: dict[str, np.ndarray | float | int | str] | None = None,
) -> Path:
    """Write direct runtime-ready float32 Fortran parameter arrays."""
    destination = Path(path)
    runtime = build_runtime_table() if table is None else table
    radii = np.asarray(runtime["family_radii_cm"], dtype=np.float32)
    second = np.asarray(runtime["second_moment"], dtype=np.float32)
    z2_log = np.asarray(runtime["z2_log_abs_z"], dtype=np.float32)
    text = f"""! Generated by particle_analysis/dust_charge_equilibrium.py.
! IM19 silicate proxy; twelve radius lookup knots; float32 runtime data.
integer, parameter :: dust_charge_nfamily = {N_FAMILIES}
integer, parameter :: dust_charge_naxis = {TABLE_N_AXIS}
real(kind=4), parameter :: dust_charge_axis_min = {float(runtime['axis_min']):.9e}_4
real(kind=4), parameter :: dust_charge_axis_step = {float(runtime['axis_step']):.9e}_4
real(kind=4), parameter :: dust_charge_axis_scale = {float(runtime['axis_scale']):.9e}_4
real(kind=4), parameter :: dust_charge_reference_min = {float(runtime['reference_charge_min']):.9e}_4
real(kind=4), parameter :: dust_charge_reference_max = {float(runtime['reference_charge_max']):.9e}_4
real(kind=4), parameter :: dust_charge_radius_cm(dust_charge_nfamily) = [ &
{_fortran_values(radii)}
]
real(kind=4), parameter :: dust_charge_m2(dust_charge_naxis,dust_charge_nfamily) = reshape([ &
{_fortran_values(second)}
], [dust_charge_naxis,dust_charge_nfamily])
real(kind=4), parameter :: dust_charge_m2log(dust_charge_naxis,dust_charge_nfamily) = reshape([ &
{_fortran_values(z2_log)}
], [dust_charge_naxis,dust_charge_nfamily])
"""
    destination.write_text(text)
    return destination


def load_environments(path: str | Path) -> tuple[Environment, ...]:
    """Load production states from a JSON list for a post-merge audit.

    Each record supplies ``name``, ``n_h_cm3``, ``temperature_k``,
    ``hydrogen_ion_fraction``, ``radiation_habing``,
    ``magnetic_field_gauss``, and ``timestep_s``, plus either
    ``electron_fraction`` or ``electron_density_cm3``.
    """
    payload = json.loads(Path(path).read_text())
    records = payload.get("environments", payload) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError("environment JSON must be a list or contain 'environments'")
    environments = []
    for index, record in enumerate(records):
        values = dict(record)
        if "electron_fraction" not in values:
            values["electron_fraction"] = (
                values.pop("electron_density_cm3") / values["n_h_cm3"]
            )
        environments.append(
            Environment(
                name=str(values.get("name", f"state_{index:05d}")),
                n_h_cm3=float(values["n_h_cm3"]),
                temperature_k=float(values["temperature_k"]),
                electron_fraction=float(values["electron_fraction"]),
                hydrogen_ion_fraction=float(
                    values["hydrogen_ion_fraction"]
                ),
                radiation_habing=float(values["radiation_habing"]),
                magnetic_field_gauss=float(values["magnetic_field_gauss"]),
                timestep_s=float(values["timestep_s"]),
            )
        )
    return tuple(environments)


def write_audit(
    path: str | Path,
    environments: tuple[Environment, ...] = CANONICAL_ENVIRONMENTS,
) -> Path:
    """Write an all-size-knot timescale audit as JSON."""
    destination = Path(path)
    results = {}
    all_ratios = []
    for environment in environments:
        records = audit_environment(environment)
        results[environment.name] = {
            "input": asdict(environment),
            "size_knots": records,
        }
        all_ratios.extend(record["tau_z_over_min"] for record in records)
    payload = {
        "revision": CLOSURE_REVISION,
        "moment_table_revision": TABLE_REVISION,
        "method": (
            "tau_Z=variance/(2<Je>); fitted f_Z plus detailed-balance reverse "
            "current; classical large-grain electron collection; total drag "
            "combines Epstein with distribution-averaged subsonic Coulomb "
            "drag from H+ and an effective singly charged heavy-ion proxy"
        ),
        "comparison": (
            "tau_Z/min(dt,1/abs(Omega),"
            "tau_total[Epstein+Coulomb])"
        ),
        "grain_density_g_cm3": 2.0,
        "scope": (
            "representative plus exact YLD04 Table 1 baseline"
            if environments is CANONICAL_ENVIRONMENTS
            else "caller-supplied production states"
        ),
        "limitations": (
            "Draine-Sutin polarization, a full WD01 current balance, and "
            "supersonic Coulomb roll-off are omitted; tau_Z is an "
            "order-of-magnitude equilibrium-separation proxy"
        ),
        "ion_composition_assumption": (
            "nHplus=xHII*nH; ne-nHplus is an effective singly charged "
            "heavy-ion channel with carbon mass. YLD04 Table 1 supplies ne "
            "but not xHII; this audit assigns WNM/WIM electrons to H+ and "
            "neutral/cloud electrons to the heavy-ion proxy"
        ),
        "post_merge_requirement": (
            "repeat with actual Stage 8 T, ne, xHII/rho, shielded radiation, "
            "B, and dt states from the combined tree"
        ),
        "maximum_tau_z_over_min": max(all_ratios),
        "equilibrium_separated_by_factor_ten": max(all_ratios) < 0.1,
        "environments": results,
    }
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fortran",
        type=Path,
        default=Path("stage9_charge_table_f32.inc"),
        help="float32 runtime include to write",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("stage9_charge_timescale_audit.json"),
        help="float64 canonical timescale audit to write",
    )
    parser.add_argument(
        "--environment-json",
        type=Path,
        help="optional actual production states for the timescale audit",
    )
    args = parser.parse_args()
    write_fortran_include(args.fortran)
    environments = (
        load_environments(args.environment_json)
        if args.environment_json is not None
        else CANONICAL_ENVIRONMENTS
    )
    write_audit(args.audit, environments)
    print(f"WROTE {args.fortran}")
    print(f"WROTE {args.audit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
