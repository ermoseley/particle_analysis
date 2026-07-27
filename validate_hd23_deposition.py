#!/usr/bin/env python3
"""Independent Stage 6/7 HD23 and dust-deposition validation.

With no ``--output-dir`` this runs fast analytic and synthetic stencil checks.
For one or more outputs it reconstructs number, mass, or ``a^2`` payloads from
massless GC dumps, deposits them with the production CIC/TSC convention, and
checks conservation, absolute normalization, log-size-family masses, and
particle-output rank-count invariance.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path

import numpy as np

from dust_hd23 import (
    ANGSTROM_CM,
    ASTRODUST_BROAD_A0,
    ASTRODUST_MAX_CM,
    ASTRODUST_MIN_CM,
    HD23_REVISION,
    MICRON_CM,
    MINIRAMSES_MH_G,
    PUBLISHED_ASTRODUST_DENSITY,
    PUBLISHED_PAH_DENSITY,
    astrodust_broad_dnda,
    hd23_partition,
    integrate_moment,
    log_family_weights,
    normalized_hd23_partition,
)
from dust_projection import iter_dust_snapshot_blocks

DEFAULT_ACTIVE_MIN_MICRON = 0.057493992527317586
DEFAULT_ACTIVE_MAX_MICRON = 0.9199038804370818


def parse_info(path: Path) -> dict[str, str]:
    """Read scalar ``key = value`` metadata from a mini-RAMSES info file."""
    values: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip().lower()] = value.strip().strip("'\"")
    return values


def _metadata_value(
    values: dict[str, str],
    aliases: tuple[str, ...],
    *,
    required: bool = False,
) -> str | None:
    for key in aliases:
        if key in values:
            return values[key]
    if required:
        raise ValueError(f"Missing output metadata; expected one of {aliases}")
    return None


def _metadata_float(
    values: dict[str, str],
    aliases: tuple[str, ...],
    *,
    required: bool = False,
) -> float | None:
    value = _metadata_value(values, aliases, required=required)
    if value is None:
        return None
    return float(value.replace("D", "e").replace("d", "e").split()[0])


def _output_location(output_dir: Path) -> tuple[Path, int]:
    match = re.fullmatch(r"output_(\d+)", output_dir.name)
    if match is None:
        raise ValueError(f"Expected an output_XXXXX directory, got {output_dir}")
    return output_dir.parent, int(match.group(1))


def deposit_periodic_3d(
    positions: np.ndarray,
    payload: np.ndarray,
    nx: int,
    *,
    box_size: float,
    scheme: str,
    grid: np.ndarray | None = None,
) -> np.ndarray:
    """Deposit using the GC gather/deposit CIC or TSC cell convention."""
    pos = np.asarray(positions, dtype=np.float64)
    values = np.asarray(payload, dtype=np.float64).ravel()
    if pos.ndim != 2 or pos.shape[1] != 3 or pos.shape[0] != values.size:
        raise ValueError("positions must be (N, 3) and match payload")
    if nx <= 0 or box_size <= 0.0:
        raise ValueError("nx and box_size must be positive")
    if grid is None:
        grid = np.zeros((nx, nx, nx), dtype=np.float64)

    u = np.mod(pos, box_size) * (nx / box_size)
    indices: list[list[np.ndarray]] = []
    weights: list[list[np.ndarray]] = []
    if scheme == "cic":
        anchor = np.floor(u + 0.5).astype(np.int64)
        fraction = u + 0.5 - anchor
        for axis in range(3):
            indices.append(
                [(anchor[:, axis] - 1) % nx, anchor[:, axis] % nx]
            )
            weights.append(
                [1.0 - fraction[:, axis], fraction[:, axis]]
            )
    elif scheme == "tsc":
        anchor = np.floor(u).astype(np.int64)
        fraction = u - anchor
        for axis in range(3):
            frac = fraction[:, axis]
            indices.append(
                [
                    (anchor[:, axis] - 1) % nx,
                    anchor[:, axis] % nx,
                    (anchor[:, axis] + 1) % nx,
                ]
            )
            weights.append(
                [
                    0.5 * (1.0 - frac) ** 2,
                    0.75 - (frac - 0.5) ** 2,
                    0.5 * frac**2,
                ]
            )
    else:
        raise ValueError(f"Unknown deposition scheme {scheme!r}")

    width = 2 if scheme == "cic" else 3
    for offset in itertools.product(range(width), repeat=3):
        weight = values.copy()
        for axis in range(3):
            weight *= weights[axis][offset[axis]]
        np.add.at(
            grid,
            (
                indices[0][offset[0]],
                indices[1][offset[1]],
                indices[2][offset[2]],
            ),
            weight,
        )
    return grid


def analytic_checks() -> dict:
    """Check published integral values and the log-size Jacobian."""
    partition = hd23_partition(
        DEFAULT_ACTIVE_MIN_MICRON * MICRON_CM,
        DEFAULT_ACTIVE_MAX_MICRON * MICRON_CM,
        astrodust_density=PUBLISHED_ASTRODUST_DENSITY,
        pah_density=PUBLISHED_PAH_DENSITY,
    )
    astrodust = {
        "volume_per_h": partition.small.volume_per_h
        + partition.broad_total.volume_per_h,
        "surface_area_per_h": partition.small.surface_area_per_h
        + partition.broad_total.surface_area_per_h,
    }
    pah = {
        "volume_per_h": partition.pah.volume_per_h,
        "surface_area_per_h": partition.pah.surface_area_per_h,
    }

    published = {
        "astrodust_volume_per_h": 3.92e-27,
        "pah_volume_per_h": 5.51e-28,
        "astrodust_surface_area_per_h": 3.00e-21,
        "pah_surface_area_per_h": 1.74e-20,
    }
    relative_errors = {
        "astrodust_volume": abs(
            astrodust["volume_per_h"] / published["astrodust_volume_per_h"] - 1.0
        ),
        "pah_volume": abs(pah["volume_per_h"] / published["pah_volume_per_h"] - 1.0),
        "astrodust_surface_area": abs(
            astrodust["surface_area_per_h"]
            / published["astrodust_surface_area_per_h"]
            - 1.0
        ),
        "pah_surface_area": abs(
            pah["surface_area_per_h"] / published["pah_surface_area_per_h"] - 1.0
        ),
    }

    log_edges = np.linspace(
        np.log(DEFAULT_ACTIVE_MIN_MICRON * MICRON_CM),
        np.log(DEFAULT_ACTIVE_MAX_MICRON * MICRON_CM),
        4097,
    )
    centers = np.exp(0.5 * (log_edges[:-1] + log_edges[1:]))
    widths = np.diff(log_edges)
    weights = log_family_weights(
        centers,
        widths,
        component="broad",
        grain_density=PUBLISHED_ASTRODUST_DENSITY,
    )
    expected_number = integrate_moment(
        "broad", 0, np.exp(log_edges[0]), np.exp(log_edges[-1])
    )
    expected_mass = (
        4.0
        * np.pi
        / 3.0
        * PUBLISHED_ASTRODUST_DENSITY
        * integrate_moment(
            "broad", 3, np.exp(log_edges[0]), np.exp(log_edges[-1])
        )
    )
    expected_area2 = integrate_moment(
        "broad", 2, np.exp(log_edges[0]), np.exp(log_edges[-1])
    )
    jacobian_errors = {
        "number": abs(float(np.sum(weights.number)) / expected_number - 1.0),
        "mass": abs(float(np.sum(weights.mass)) / expected_mass - 1.0),
        "area2": abs(float(np.sum(weights.area2)) / expected_area2 - 1.0),
    }
    broad = partition.broad_total
    active_fractions = {
        "mass": partition.broad_active.mass_per_h / broad.mass_per_h,
        "area2": partition.broad_active.area2_per_h / broad.area2_per_h,
    }
    partition_closure_errors = {}
    for cutoff_micron in (0.10, 0.30):
        cut = hd23_partition(
            cutoff_micron * MICRON_CM,
            DEFAULT_ACTIVE_MAX_MICRON * MICRON_CM,
            astrodust_density=PUBLISHED_ASTRODUST_DENSITY,
            pah_density=PUBLISHED_PAH_DENSITY,
        ).broad_total
        partition_closure_errors[f"{cutoff_micron:.2f}_micron"] = {
            "number": abs(cut.number_per_h / broad.number_per_h - 1.0),
            "mass": abs(cut.mass_per_h / broad.mass_per_h - 1.0),
            "area2": abs(cut.area2_per_h / broad.area2_per_h - 1.0),
        }
    # Table 1 coefficients are published at only three significant figures,
    # so their direct integrals do not exactly recover the full-precision
    # derived values in Table 2 (the high-a volume is especially sensitive).
    table_tolerances = {
        "astrodust_volume": 0.25,
        "pah_volume": 0.03,
        "astrodust_surface_area": 0.06,
        "pah_surface_area": 0.03,
    }
    passed = all(
        relative_errors[name] <= tolerance
        for name, tolerance in table_tolerances.items()
    ) and max(jacobian_errors.values()) < 2.0e-6 and max(
        error
        for cutoff in partition_closure_errors.values()
        for error in cutoff.values()
    ) < 2.0e-11
    return {
        "passed": passed,
        "computed": {"astrodust": astrodust, "pah": pah},
        "published": published,
        "published_relative_errors": relative_errors,
        "published_table_tolerances": table_tolerances,
        "published_coefficients_are_rounded": True,
        "log_jacobian_relative_errors": jacobian_errors,
        "broad_active_fractions": active_fractions,
        "explicit_cutoff_partition_closure_errors": partition_closure_errors,
    }


def synthetic_stencil_checks() -> dict:
    """Exercise conservation, uniform fill, and periodic translation."""
    nx = 8
    centers_1d = (np.arange(nx, dtype=np.float64) + 0.5) / nx
    mesh = np.meshgrid(centers_1d, centers_1d, centers_1d, indexing="ij")
    uniform_pos = np.column_stack([axis.ravel() for axis in mesh])
    uniform_payload = np.ones(uniform_pos.shape[0], dtype=np.float64)

    rng = np.random.default_rng(72123)
    random_pos = rng.random((257, 3))
    random_payload = np.exp(rng.normal(size=random_pos.shape[0]))
    results: dict[str, dict] = {}
    passed = True
    for scheme in ("cic", "tsc"):
        uniform = deposit_periodic_3d(
            uniform_pos, uniform_payload, nx, box_size=1.0, scheme=scheme
        )
        original = deposit_periodic_3d(
            random_pos, random_payload, nx, box_size=1.0, scheme=scheme
        )
        shifted_pos = random_pos.copy()
        shifted_pos[:, 0] = np.mod(shifted_pos[:, 0] + 1.0 / nx, 1.0)
        shifted = deposit_periodic_3d(
            shifted_pos, random_payload, nx, box_size=1.0, scheme=scheme
        )
        conservation = abs(float(np.sum(original)) / float(np.sum(random_payload)) - 1.0)
        uniform_error = float(np.max(np.abs(uniform - 1.0)))
        translation_error = float(
            np.max(np.abs(shifted - np.roll(original, 1, axis=0)))
        )
        scheme_passed = (
            conservation < 5.0e-14
            and uniform_error < 5.0e-14
            and translation_error < 2.0e-13
        )
        passed &= scheme_passed
        results[scheme] = {
            "passed": scheme_passed,
            "conservation_relative_error": conservation,
            "uniform_max_abs_error": uniform_error,
            "one_cell_translation_max_abs_error": translation_error,
        }
    return {"passed": passed, "schemes": results}


def _radius_factor(
    metadata: dict[str, str],
    args: argparse.Namespace,
    grain_density: float,
) -> tuple[float, str]:
    if args.size_in_micron:
        return MICRON_CM, "size_in_micron"
    if args.radius_code_to_cm is not None:
        return args.radius_code_to_cm, "command_line"
    exported = _metadata_float(
        metadata,
        ("dust_radius_code_to_cm", "hd23_radius_code_to_cm"),
    )
    if exported is not None:
        return exported, "output_metadata"
    unit_d = _metadata_float(metadata, ("unit_d",), required=True)
    unit_l = _metadata_float(metadata, ("unit_l",), required=True)
    return unit_d * unit_l / grain_density, "unit_d*unit_l/grain_density"


def _active_bounds(
    metadata: dict[str, str],
    args: argparse.Namespace,
) -> tuple[float, float, str]:
    lower = args.active_min_micron
    upper = args.active_max_micron
    source = "command_line"
    if lower is None:
        lower = _metadata_float(
            metadata,
            (
                "dust_active_min_micron",
                "dust_active_a_min_micron",
                "hd23_active_min_micron",
            ),
        )
        source = "output_metadata"
    if upper is None:
        upper = _metadata_float(
            metadata,
            (
                "dust_active_max_micron",
                "dust_active_a_max_micron",
                "hd23_active_max_micron",
            ),
        )
        source = "output_metadata"
    if lower is None or upper is None:
        lower = DEFAULT_ACTIVE_MIN_MICRON
        upper = DEFAULT_ACTIVE_MAX_MICRON
        source = "stage6_default"
    if lower <= 0.0 or upper <= lower:
        raise ValueError("Invalid active grain-size interval")
    return lower * MICRON_CM, upper * MICRON_CM, source


def validate_exported_moments(
    metadata: dict[str, str],
    *,
    active_min: float,
    active_max: float,
    grain_density: float,
) -> dict:
    """Compare every exported Stage-6 component moment to fresh quadrature."""
    dust_to_gas = _metadata_float(
        metadata, ("dust_to_gas_hd23", "dust_to_gas"), required=True
    )
    hydrogen_fraction = _metadata_float(
        metadata, ("dust_x_h", "x_h"), required=True
    )
    pah_scale = _metadata_float(metadata, ("dust_pah_scale",), required=True)
    partition, normalization = normalized_hd23_partition(
        active_min,
        active_max,
        dust_to_gas=dust_to_gas,
        hydrogen_mass_fraction=hydrogen_fraction,
        astrodust_density=grain_density,
        pah_scale=pah_scale,
    )
    components = {
        "active": partition.broad_active,
        "passive_broad": partition.broad_passive,
        "small": partition.small,
        "pah": partition.pah,
    }
    errors: dict[str, float] = {}
    for name, component in components.items():
        expected = {
            "number": component.number_per_h,
            "mass": component.mass_per_h,
            "a2": component.area2_per_h,
            "form": component.area2_per_h,
            "lw": component.area2_per_h,
        }
        for moment, reference in expected.items():
            exported = _metadata_float(
                metadata, (f"dust_{moment}_{name}",), required=True
            )
            scale = max(abs(reference), np.finfo(float).tiny)
            errors[f"{moment}_{name}"] = abs(exported - reference) / scale
    form_basis = _metadata_value(metadata, ("dust_form_basis",), required=True)
    lw_basis = _metadata_value(metadata, ("dust_lw_basis",), required=True)
    max_error = max(errors.values())
    return {
        "passed": (
            max_error <= 2.0e-9
            and form_basis == "area2_stage6"
            and lw_basis == "area2_stage6"
        ),
        "normalization": normalization,
        "max_relative_error": max_error,
        "relative_errors": errors,
        "formation_basis": form_basis,
        "lw_basis": lw_basis,
    }


def validate_provider_normalization(
    metadata: dict[str, str],
    *,
    relative_sums: dict[str, float],
    normalization: float,
    grain_density: float,
) -> dict:
    """Close the exported discrete and absolute Stage-6 normalizations."""
    dust_to_gas = _metadata_float(
        metadata, ("dust_to_gas_hd23", "dust_to_gas"), required=True
    )
    hydrogen_fraction = _metadata_float(
        metadata, ("dust_x_h", "x_h"), required=True
    )
    n_h = _metadata_float(metadata, ("dust_nh_ref_resolved",), required=True)
    boxlen = _metadata_float(metadata, ("boxlen",), required=True)
    unit_l = _metadata_float(metadata, ("unit_l",), required=True)
    a0_normalized = _metadata_float(
        metadata, ("dust_a0_normalized",), required=True
    )
    delta_ln_a = _metadata_float(metadata, ("dust_dln_a",), required=True)
    c2 = _metadata_float(metadata, ("dust_c2",), required=True)
    c3 = _metadata_float(metadata, ("dust_c3",), required=True)
    area_norm = _metadata_float(
        metadata, ("dust_area_weight_norm",), required=True
    )
    macro_number_norm = _metadata_float(
        metadata, ("dust_macro_number_norm",), required=True
    )
    macro_mass_norm = _metadata_float(
        metadata, ("dust_macro_mass_norm",), required=True
    )
    active_a2 = _metadata_float(metadata, ("dust_a2_active",), required=True)
    active_mass = _metadata_float(metadata, ("dust_mass_active",), required=True)

    volume = (boxlen * unit_l) ** 3
    four_pi_over_three = 4.0 * np.pi / 3.0
    expected = {
        "a0_normalized": ASTRODUST_BROAD_A0 * normalization,
        "c2": active_a2
        / (normalization * delta_ln_a * relative_sums["area2"]),
        "c3": active_mass
        / (normalization * delta_ln_a * relative_sums["mass"]),
        "area_weight_norm": n_h * a0_normalized * delta_ln_a * c2,
        "macro_number_norm": (
            n_h * volume * a0_normalized * delta_ln_a * c3
        ),
        "macro_mass_norm": (
            macro_number_norm * four_pi_over_three * grain_density
        ),
        "particle_area_sum": n_h * active_a2,
        "particle_macro_mass_sum": n_h * volume * active_mass,
        "total_mass_per_h": dust_to_gas
        * MINIRAMSES_MH_G
        / hydrogen_fraction,
    }
    actual = {
        "a0_normalized": a0_normalized,
        "c2": c2,
        "c3": c3,
        "area_weight_norm": area_norm,
        "macro_number_norm": macro_number_norm,
        "macro_mass_norm": macro_mass_norm,
        "particle_area_sum": (
            relative_sums["area2"] * area_norm / ASTRODUST_BROAD_A0
        ),
        "particle_macro_mass_sum": (
            relative_sums["mass"]
            * macro_mass_norm
            / (four_pi_over_three * grain_density * ASTRODUST_BROAD_A0)
        ),
        "total_mass_per_h": sum(
            _metadata_float(metadata, (f"dust_mass_{name}",), required=True)
            for name in ("active", "passive_broad", "small", "pah")
        ),
    }
    errors = {
        name: abs(actual[name] / expected[name] - 1.0)
        for name in expected
    }
    tolerance = 5.0e-5
    return {
        "passed": max(errors.values()) <= tolerance,
        "relative_tolerance": tolerance,
        "max_relative_error": max(errors.values()),
        "relative_errors": errors,
        "actual": actual,
        "expected": expected,
    }


def validate_output(
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Reconstruct and deposit one output in bounded particle chunks."""
    run_dir, output_num = _output_location(output_dir)
    metadata = parse_info(output_dir / "info.txt")
    revision = _metadata_value(
        metadata, ("dust_hd23_revision", "hd23_revision")
    )
    if revision != HD23_REVISION and not args.allow_revision_mismatch:
        raise ValueError(
            f"{output_dir}: dust_hd23_revision={revision!r}, expected {HD23_REVISION!r}"
        )
    grain_density = args.grain_density
    if grain_density is None:
        grain_density = _metadata_float(
            metadata,
            ("grain_bulk_density", "dust_grain_bulk_density"),
        )
    if grain_density is None:
        grain_density = 2.0
    radius_factor, radius_factor_source = _radius_factor(
        metadata, args, grain_density
    )
    active_min, active_max, active_source = _active_bounds(metadata, args)
    exported_moments = validate_exported_moments(
        metadata,
        active_min=active_min,
        active_max=active_max,
        grain_density=grain_density,
    )
    box_size = args.box_size
    if box_size is None:
        box_size = _metadata_float(metadata, ("boxlen",), required=True)

    schemes = ("cic", "tsc") if args.scheme == "both" else (args.scheme,)
    grids = {
        scheme: np.zeros((args.nx, args.nx, args.nx), dtype=np.float64)
        for scheme in schemes
    }
    bin_edges = np.geomspace(active_min, active_max, args.family_bins + 1)
    particle_mass_bins = np.zeros(args.family_bins, dtype=np.float64)
    sums = {"number": 0.0, "mass": 0.0, "area2": 0.0}
    count = 0
    active_count = 0
    observed_min = np.inf
    observed_max = 0.0
    has_stored_mass: bool | None = None

    for block in iter_dust_snapshot_blocks(
        run_dir, output_num, chunk_size=args.chunk_size
    ):
        radius = block.size * radius_factor
        weights = log_family_weights(
            radius,
            1.0,
            component="broad",
            grain_density=grain_density,
        )
        active = (radius >= active_min) & (radius <= active_max)
        payload = np.where(active, getattr(weights, args.field), 0.0)
        for name in sums:
            sums[name] += float(np.sum(getattr(weights, name)[active]))
        particle_mass_bins += np.histogram(
            radius[active], bins=bin_edges, weights=weights.mass[active]
        )[0]
        count += radius.size
        active_count += int(np.count_nonzero(active))
        observed_min = min(observed_min, float(np.min(radius)))
        observed_max = max(observed_max, float(np.max(radius)))
        block_has_mass = block.mass is not None
        if has_stored_mass is None:
            has_stored_mass = block_has_mass
        elif has_stored_mass != block_has_mass:
            raise ValueError("Inconsistent mass-field presence across dust files")
        for scheme in schemes:
            deposit_periodic_3d(
                block.pos,
                payload,
                args.nx,
                box_size=box_size,
                scheme=scheme,
                grid=grids[scheme],
            )

    if count == 0:
        raise ValueError(f"No finite dust particles found in {output_dir}")
    if active_count == 0:
        raise ValueError(f"No particles lie in the active interval for {output_dir}")
    expected_mass_bins = np.array(
        [
            (4.0 * np.pi / 3.0)
            * grain_density
            * integrate_moment("broad", 3, lo, hi)
            for lo, hi in zip(bin_edges[:-1], bin_edges[1:])
        ]
    )
    particle_fraction = particle_mass_bins / np.sum(particle_mass_bins)
    expected_fraction = expected_mass_bins / np.sum(expected_mass_bins)
    family_max_abs_error = float(
        np.max(np.abs(particle_fraction - expected_fraction))
    )
    provider_normalization = validate_provider_normalization(
        metadata,
        relative_sums=sums,
        normalization=exported_moments["normalization"],
        grain_density=grain_density,
    )
    deposition = {}
    passed = (
        family_max_abs_error <= args.family_atol
        and exported_moments["passed"]
        and provider_normalization["passed"]
    )
    for scheme, grid in grids.items():
        deposited_sum = float(np.sum(grid))
        source_sum = sums[args.field]
        relative_error = abs(deposited_sum / source_sum - 1.0)
        scheme_passed = relative_error <= args.conservation_rtol
        passed &= scheme_passed
        deposition[scheme] = {
            "passed": scheme_passed,
            "source_sum": source_sum,
            "deposited_sum": deposited_sum,
            "conservation_relative_error": relative_error,
        }
    result = {
        "passed": passed,
        "output_dir": str(output_dir.resolve()),
        "particle_count": count,
        "active_particle_count": active_count,
        "stored_mass_field": bool(has_stored_mass),
        "hd23_revision": revision,
        "grain_density_g_cm3": grain_density,
        "radius_code_to_cm": radius_factor,
        "radius_factor_source": radius_factor_source,
        "box_size": box_size,
        "active_interval_source": active_source,
        "active_min_micron": active_min / MICRON_CM,
        "active_max_micron": active_max / MICRON_CM,
        "exported_moments": exported_moments,
        "provider_normalization": provider_normalization,
        "observed_min_micron": observed_min / MICRON_CM,
        "observed_max_micron": observed_max / MICRON_CM,
        "relative_weight_sums": sums,
        "field_deposited": args.field,
        "deposition": deposition,
        "family_mass_fraction_max_abs_error": family_max_abs_error,
        "family_mass_fraction_tolerance": args.family_atol,
        "family_mass_fraction_particle": particle_fraction.tolist(),
        "family_mass_fraction_analytic": expected_fraction.tolist(),
    }
    return result, grids


def compare_outputs(
    outputs: list[tuple[dict, dict[str, np.ndarray]]],
    *,
    atol: float,
    rtol: float,
) -> dict:
    """Compare CPU reconstructions from particle outputs across rank counts."""
    if len(outputs) < 2:
        return {"passed": True, "comparisons": []}
    reference_report, reference_grids = outputs[0]
    comparisons = []
    passed = True
    for report, grids in outputs[1:]:
        for scheme, reference in reference_grids.items():
            candidate = grids[scheme]
            max_abs = float(np.max(np.abs(candidate - reference)))
            scale = max(float(np.max(np.abs(reference))), np.finfo(float).tiny)
            max_relative_to_peak = max_abs / scale
            equal = bool(np.allclose(candidate, reference, atol=atol, rtol=rtol))
            passed &= equal
            comparisons.append(
                {
                    "passed": equal,
                    "scheme": scheme,
                    "reference": reference_report["output_dir"],
                    "candidate": report["output_dir"],
                    "max_abs_error": max_abs,
                    "max_error_relative_to_reference_peak": max_relative_to_peak,
                }
            )
    return {"passed": passed, "comparisons": comparisons}


def parse_log_diagnostics(
    paths: list[Path],
    *,
    require_canary_check: bool = False,
) -> dict:
    """Collect Stage 7 area diagnostics without assuming a full log format."""
    lines: list[str] = []
    metrics: dict[str, list[float]] = {}
    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][-+]?\d+)?"
    pattern = re.compile(
        r"\b(dust_area_(?:ratio|lost_weight|checksum|source|field|lost|oob|nan|"
        r"zero_leaf|canary_checked|canary))"
        r"\s*[=:]\s*"
        f"({number})",
        re.IGNORECASE,
    )
    production_pattern = re.compile(
        r"dust\s+HD23\s+deposit\s+level\s*=\s*\d+\s+"
        r"source/intended/lost/field\s*=\s*"
        f"({number})\\s+({number})\\s+({number})\\s+({number})\\s+"
        r"missing/oob/canary\s*=\s*(\d+)\s+(\d+)\s+(\d+)",
        re.IGNORECASE,
    )
    for path in paths:
        for raw in path.read_text(errors="replace").splitlines():
            if "dust_area" not in raw.lower() and "hd23" not in raw.lower():
                continue
            lines.append(raw.strip())
            production = production_pattern.search(raw)
            if production is not None:
                source, intended, lost, field = (
                    float(value.replace("D", "e").replace("d", "e"))
                    for value in production.groups()[:4]
                )
                missing, oob, canary = (
                    float(value) for value in production.groups()[4:]
                )
                metrics.setdefault("dust_area_ratio", []).append(
                    field / source if source != 0.0 else float(field == 0.0)
                )
                metrics.setdefault("dust_area_intended_ratio", []).append(
                    intended / source if source != 0.0 else float(intended == 0.0)
                )
                metrics.setdefault("dust_area_lost", []).append(lost)
                metrics.setdefault("dust_area_missing", []).append(missing)
                metrics.setdefault("dust_area_oob", []).append(oob)
                metrics.setdefault("dust_area_canary", []).append(canary)
            for key, value in pattern.findall(raw):
                metrics.setdefault(key.lower(), []).append(
                    float(value.replace("D", "e").replace("d", "e"))
                )
    passed = True
    for key, values in metrics.items():
        if key == "dust_area_canary_checked":
            if require_canary_check:
                passed &= all(value == 1.0 for value in values)
        elif key.endswith(("checksum", "source", "field")):
            passed &= all(np.isfinite(value) for value in values)
        elif key == "dust_area_lost_weight":
            passed &= all(value == 0.0 for value in values)
        elif key.endswith("ratio"):
            passed &= all(abs(value - 1.0) <= 5.0e-5 for value in values)
        elif key.endswith(("lost", "missing", "oob", "nan", "canary")):
            passed &= all(value == 0.0 for value in values)
    if paths:
        required = {
            "dust_area_ratio",
            "dust_area_lost",
            "dust_area_oob",
            "dust_area_canary",
        }
        if require_canary_check:
            required.add("dust_area_canary_checked")
        passed &= required.issubset(metrics)
    return {"passed": passed, "metrics": metrics, "matched_lines": lines}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        action="append",
        type=Path,
        default=[],
        help="output_XXXXX directory; repeat to compare MPI decompositions",
    )
    parser.add_argument("--log", action="append", type=Path, default=[])
    parser.add_argument("--report", type=Path)
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument(
        "--box-size",
        type=float,
        help="simulation box size; defaults to each output's boxlen metadata",
    )
    parser.add_argument("--scheme", choices=("cic", "tsc", "both"), default="both")
    parser.add_argument(
        "--field", choices=("number", "mass", "area2"), default="area2"
    )
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--family-bins", type=int, default=16)
    parser.add_argument("--family-atol", type=float, default=5.0e-2)
    parser.add_argument("--conservation-rtol", type=float, default=5.0e-13)
    parser.add_argument("--compare-atol", type=float, default=1.0e-14)
    parser.add_argument("--compare-rtol", type=float, default=2.0e-12)
    parser.add_argument("--grain-density", type=float)
    parser.add_argument("--radius-code-to-cm", type=float)
    parser.add_argument("--size-in-micron", action="store_true")
    parser.add_argument("--active-min-micron", type=float)
    parser.add_argument("--active-max-micron", type=float)
    parser.add_argument(
        "--allow-revision-mismatch",
        action="store_true",
        help="permit an HD23 revision other than the exact Stage-6 revision",
    )
    parser.add_argument(
        "--require-canary-check",
        action="store_true",
        help="require logs to prove debug canaries were actually checked",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    analytic = analytic_checks()
    stencils = synthetic_stencil_checks()
    output_results = [validate_output(path, args) for path in args.output_dir]
    comparisons = compare_outputs(
        output_results, atol=args.compare_atol, rtol=args.compare_rtol
    )
    logs = parse_log_diagnostics(
        args.log,
        require_canary_check=args.require_canary_check,
    )
    report = {
        "passed": bool(
            analytic["passed"]
            and stencils["passed"]
            and comparisons["passed"]
            and logs["passed"]
            and all(result[0]["passed"] for result in output_results)
        ),
        "hd23_revision": HD23_REVISION,
        "analytic": analytic,
        "synthetic_deposition": stencils,
        "outputs": [result[0] for result in output_results],
        "mpi_comparisons": comparisons,
        "log_diagnostics": logs,
    }
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    print(text, end="")
    if args.report is not None:
        args.report.write_text(text)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
