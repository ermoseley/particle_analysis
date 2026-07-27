#!/usr/bin/env python3
"""HD23 astrodust/PAH distributions and log-size quadrature helpers.

The coefficients are the published best fit from Hensley & Draine (2023),
Equations 18 and 25 and their Table 1.  Radii passed to this module are in cm.
The distributions are ``(1 / n_H) dn / da`` and therefore have units
``H^-1 cm^-1``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import numpy as np

ANGSTROM_CM = 1.0e-8
MICRON_CM = 1.0e-4

HD23_REVISION = "HD23-2023-eq18-eq25-v1"
MINIRAMSES_MH_G = 1.6605390e-24

ASTRODUST_MIN_CM = 4.5 * ANGSTROM_CM
ASTRODUST_MAX_CM = 5.0 * MICRON_CM
PAH_MIN_CM = 4.0 * ANGSTROM_CM

# HD23 Equation 18 and Table 1.
PAH_B = np.array([7.52e-7, 8.09e-10], dtype=np.float64)
PAH_A0_CM = np.array([4.0, 30.0], dtype=np.float64) * ANGSTROM_CM
PAH_SIGMA = 0.40

# HD23 Equation 25 and Table 1.
ASTRODUST_B = 3.31e-10
ASTRODUST_A0_CM = 63.8 * ANGSTROM_CM
ASTRODUST_SIGMA = 0.353
ASTRODUST_POLY = np.array(
    [-3.40, -0.807, 0.157, 7.96e-3, -1.68e-3],
    dtype=np.float64,
)
ASTRODUST_BROAD_A0 = 2.97e-5

# Densities used for the paper's derived mass/volume checks.  The simulation
# grain density remains an explicit argument to the weighting routines.
PUBLISHED_ASTRODUST_DENSITY = 2.74
PUBLISHED_PAH_DENSITY = 2.0


def _array_result(radius_cm, evaluator: Callable[[np.ndarray], np.ndarray]):
    radius = np.asarray(radius_cm, dtype=np.float64)
    scalar = radius.ndim == 0
    flat = radius.reshape(-1)
    out = evaluator(flat).reshape(radius.shape)
    return float(out) if scalar else out


def astrodust_small_dnda(radius_cm):
    """Return the small-lognormal term of HD23 Equation 25."""

    def evaluate(radius: np.ndarray) -> np.ndarray:
        out = np.zeros_like(radius)
        valid = (radius >= ASTRODUST_MIN_CM) & (radius <= ASTRODUST_MAX_CM)
        a = radius[valid]
        log_ratio = np.log(a / ASTRODUST_A0_CM)
        out[valid] = ASTRODUST_B / a * np.exp(
            -0.5 * (log_ratio / ASTRODUST_SIGMA) ** 2
        )
        return out

    return _array_result(radius_cm, evaluate)


def astrodust_broad_dnda(radius_cm):
    """Return the broad polynomial-exponential term of HD23 Equation 25."""

    def evaluate(radius: np.ndarray) -> np.ndarray:
        out = np.zeros_like(radius)
        valid = (radius >= ASTRODUST_MIN_CM) & (radius <= ASTRODUST_MAX_CM)
        a = radius[valid]
        log_a_angstrom = np.log(a / ANGSTROM_CM)
        exponent = np.zeros_like(log_a_angstrom)
        power = log_a_angstrom.copy()
        for coefficient in ASTRODUST_POLY:
            exponent += coefficient * power
            power *= log_a_angstrom
        out[valid] = ASTRODUST_BROAD_A0 / a * np.exp(exponent)
        return out

    return _array_result(radius_cm, evaluate)


def astrodust_dnda(radius_cm):
    """Return the complete astrodust distribution from HD23 Equation 25."""
    return astrodust_small_dnda(radius_cm) + astrodust_broad_dnda(radius_cm)


def pah_dnda(radius_cm):
    """Return the two-lognormal PAH distribution from HD23 Equation 18."""

    def evaluate(radius: np.ndarray) -> np.ndarray:
        out = np.zeros_like(radius)
        valid = radius > PAH_MIN_CM
        a = radius[valid]
        for amplitude, center in zip(PAH_B, PAH_A0_CM):
            log_ratio = np.log(a / center)
            out[valid] += amplitude / a * np.exp(
                -0.5 * (log_ratio / PAH_SIGMA) ** 2
            )
        return out

    return _array_result(radius_cm, evaluate)


COMPONENTS: dict[str, Callable] = {
    "small": astrodust_small_dnda,
    "broad": astrodust_broad_dnda,
    "astrodust": astrodust_dnda,
    "pah": pah_dnda,
}


def integrate_moment(
    component: str,
    power: int,
    lower_cm: float,
    upper_cm: float,
    *,
    quadrature_order: int = 256,
) -> float:
    """Integrate ``a**power * (dn/n_H/da) da`` using ``ln(a)``.

    The ``a`` Jacobian is explicit in the integrand.  In particular, powers
    0, 2, and 3 are the number, unweighted area, and volume kernels.
    """
    if component not in COMPONENTS:
        raise ValueError(f"Unknown HD23 component {component!r}")
    if lower_cm <= 0.0 or upper_cm <= lower_cm:
        raise ValueError("Moment bounds must satisfy 0 < lower_cm < upper_cm")
    if quadrature_order < 16:
        raise ValueError("quadrature_order must be at least 16")
    nodes, weights = _legendre_rule(quadrature_order)
    log_lower = np.log(lower_cm)
    log_upper = np.log(upper_cm)
    midpoint = 0.5 * (log_lower + log_upper)
    half_width = 0.5 * (log_upper - log_lower)
    log_radius = midpoint + half_width * nodes
    radius = np.exp(log_radius)
    integrand = radius ** (power + 1) * COMPONENTS[component](radius)
    return float(half_width * np.dot(weights, integrand))


@lru_cache(maxsize=None)
def _legendre_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Cache the NumPy Gauss-Legendre rule used by all moment integrals."""
    return np.polynomial.legendre.leggauss(order)


@dataclass(frozen=True)
class PopulationMoments:
    """Number, geometric ``a^2``, volume, and mass per H nucleus."""

    number_per_h: float
    area2_per_h: float
    volume_per_h: float
    mass_per_h: float

    @property
    def surface_area_per_h(self) -> float:
        return 4.0 * np.pi * self.area2_per_h

    def scaled(self, factor: float) -> "PopulationMoments":
        return PopulationMoments(
            number_per_h=self.number_per_h * factor,
            area2_per_h=self.area2_per_h * factor,
            volume_per_h=self.volume_per_h * factor,
            mass_per_h=self.mass_per_h * factor,
        )


def population_moments(
    component: str,
    lower_cm: float,
    upper_cm: float,
    *,
    grain_density: float,
) -> PopulationMoments:
    """Integrate the basic moments of one analytic component."""
    if grain_density <= 0.0:
        raise ValueError("grain_density must be positive")
    number = integrate_moment(component, 0, lower_cm, upper_cm)
    area2 = integrate_moment(component, 2, lower_cm, upper_cm)
    volume = (4.0 * np.pi / 3.0) * integrate_moment(
        component, 3, lower_cm, upper_cm
    )
    return PopulationMoments(number, area2, volume, grain_density * volume)


@dataclass(frozen=True)
class HD23Partition:
    """Analytic HD23 components with the broad term split active/passive."""

    small: PopulationMoments
    broad_active: PopulationMoments
    broad_passive: PopulationMoments
    pah: PopulationMoments

    @property
    def broad_total(self) -> PopulationMoments:
        return _sum_moments(self.broad_active, self.broad_passive)

    @property
    def total(self) -> PopulationMoments:
        return _sum_moments(
            self.small,
            self.broad_active,
            self.broad_passive,
            self.pah,
        )


def _sum_moments(*moments: PopulationMoments) -> PopulationMoments:
    return PopulationMoments(
        number_per_h=sum(value.number_per_h for value in moments),
        area2_per_h=sum(value.area2_per_h for value in moments),
        volume_per_h=sum(value.volume_per_h for value in moments),
        mass_per_h=sum(value.mass_per_h for value in moments),
    )


def hd23_partition(
    active_min_cm: float,
    active_max_cm: float,
    *,
    astrodust_density: float = PUBLISHED_ASTRODUST_DENSITY,
    pah_density: float = PUBLISHED_PAH_DENSITY,
) -> HD23Partition:
    """Partition the broad term while retaining small astrodust and PAHs."""
    lo = max(active_min_cm, ASTRODUST_MIN_CM)
    hi = min(active_max_cm, ASTRODUST_MAX_CM)
    if hi <= lo:
        raise ValueError("Active interval does not overlap the HD23 domain")

    small = population_moments(
        "small",
        ASTRODUST_MIN_CM,
        ASTRODUST_MAX_CM,
        grain_density=astrodust_density,
    )
    active = population_moments(
        "broad",
        lo,
        hi,
        grain_density=astrodust_density,
    )
    passive_parts: list[PopulationMoments] = []
    if lo > ASTRODUST_MIN_CM:
        passive_parts.append(
            population_moments(
                "broad",
                ASTRODUST_MIN_CM,
                lo,
                grain_density=astrodust_density,
            )
        )
    if hi < ASTRODUST_MAX_CM:
        passive_parts.append(
            population_moments(
                "broad",
                hi,
                ASTRODUST_MAX_CM,
                grain_density=astrodust_density,
            )
        )
    passive = _sum_moments(*passive_parts)
    pah = population_moments(
        "pah",
        PAH_MIN_CM,
        ASTRODUST_MAX_CM,
        grain_density=pah_density,
    )
    return HD23Partition(small, active, passive, pah)


def dust_normalization_scale(
    published_mass_per_h: float,
    *,
    dust_to_gas: float,
    hydrogen_mass_fraction: float,
    hydrogen_mass_g: float = MINIRAMSES_MH_G,
) -> float:
    """Scale per-H moments to the mini-RAMSES total dust/gas convention."""
    if published_mass_per_h <= 0.0 or dust_to_gas <= 0.0:
        raise ValueError("Dust masses and dust_to_gas must be positive")
    if not 0.0 < hydrogen_mass_fraction <= 1.0:
        raise ValueError("hydrogen_mass_fraction must lie in (0, 1]")
    target_mass_per_h = dust_to_gas * hydrogen_mass_g / hydrogen_mass_fraction
    return target_mass_per_h / published_mass_per_h


def normalized_hd23_partition(
    active_min_cm: float,
    active_max_cm: float,
    *,
    dust_to_gas: float,
    hydrogen_mass_fraction: float,
    astrodust_density: float,
    pah_scale: float = 1.0,
    pah_density: float = PUBLISHED_PAH_DENSITY,
) -> tuple[HD23Partition, float]:
    """Return the Stage-6 partition normalized to the selected total D/G."""
    if pah_scale < 0.0:
        raise ValueError("pah_scale must be nonnegative")
    raw = hd23_partition(
        active_min_cm,
        active_max_cm,
        astrodust_density=astrodust_density,
        pah_density=pah_density,
    )
    raw = HD23Partition(
        small=raw.small,
        broad_active=raw.broad_active,
        broad_passive=raw.broad_passive,
        pah=raw.pah.scaled(pah_scale),
    )
    scale = dust_normalization_scale(
        raw.total.mass_per_h,
        dust_to_gas=dust_to_gas,
        hydrogen_mass_fraction=hydrogen_mass_fraction,
    )
    return (
        HD23Partition(
            small=raw.small.scaled(scale),
            broad_active=raw.broad_active.scaled(scale),
            broad_passive=raw.broad_passive.scaled(scale),
            pah=raw.pah.scaled(scale),
        ),
        scale,
    )


@dataclass(frozen=True)
class LogFamilyWeights:
    """Distinct number, mass, and ``a^2`` weights for log-size samples."""

    number: np.ndarray
    mass: np.ndarray
    area2: np.ndarray

    def scaled(self, factor: float) -> "LogFamilyWeights":
        return LogFamilyWeights(
            number=self.number * factor,
            mass=self.mass * factor,
            area2=self.area2 * factor,
        )


def log_family_weights(
    radius_cm,
    delta_ln_a,
    *,
    component: str = "broad",
    grain_density: float,
) -> LogFamilyWeights:
    """Return quadrature weights for samples uniform in ``ln(a)``.

    For ``f(a) = (1/n_H) dn/da``, a log-family represents
    ``dN/n_H = a f(a) dln(a)``.  Its mass and area weights are therefore
    proportional to ``a^4 f(a)`` and ``a^3 f(a)`` respectively, not to equal
    weights or ``a^3`` alone.
    """
    if component not in COMPONENTS:
        raise ValueError(f"Unknown HD23 component {component!r}")
    if grain_density <= 0.0:
        raise ValueError("grain_density must be positive")
    radius = np.asarray(radius_cm, dtype=np.float64)
    widths = np.asarray(delta_ln_a, dtype=np.float64)
    if np.any(radius <= 0.0) or np.any(widths <= 0.0):
        raise ValueError("Radii and log-size widths must be positive")
    number = radius * COMPONENTS[component](radius) * widths
    mass = (4.0 * np.pi / 3.0) * grain_density * radius**3 * number
    area2 = radius**2 * number
    return LogFamilyWeights(number=number, mass=mass, area2=area2)


def log_bin_widths(radius_cm) -> np.ndarray:
    """Return midpoint-rule ``dln(a)`` widths for ordered family centers."""
    radius = np.asarray(radius_cm, dtype=np.float64)
    if radius.ndim != 1 or radius.size < 2:
        raise ValueError("At least two one-dimensional family centers are required")
    if np.any(radius <= 0.0) or np.any(np.diff(radius) <= 0.0):
        raise ValueError("Family centers must be positive and strictly increasing")
    log_radius = np.log(radius)
    edges = np.empty(radius.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (log_radius[:-1] + log_radius[1:])
    edges[0] = log_radius[0] - 0.5 * (log_radius[1] - log_radius[0])
    edges[-1] = log_radius[-1] + 0.5 * (log_radius[-1] - log_radius[-2])
    return np.diff(edges)


def radius_code_to_cm(
    size_code,
    *,
    unit_density: float,
    unit_length: float,
    grain_density: float,
):
    """Convert mini-RAMSES GC ``size`` to physical effective radius."""
    if unit_density <= 0.0 or unit_length <= 0.0 or grain_density <= 0.0:
        raise ValueError("Unit conversion and grain density must be positive")
    return (
        np.asarray(size_code, dtype=np.float64)
        * unit_density
        * unit_length
        / grain_density
    )
