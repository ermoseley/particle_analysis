#!/usr/bin/env python3
"""Focused validation for the Stage 9 equilibrium-charge reference."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dust_charge_equilibrium import (
    CANONICAL_ENVIRONMENTS,
    E_ESU,
    MICRON_CM,
    REFERENCE_FIT,
    REPRESENTATIVE_ENVIRONMENTS,
    SILICATE_FITS,
    SIZE_KNOT_RADII_CM,
    TABLE_ZREF_MAX,
    TABLE_ZREF_MIN,
    YLD04_TABLE1_ENVIRONMENTS,
    audit_environment,
    build_runtime_table,
    charge_parameter_code,
    coulomb_log_moment,
    dark_potential_nu,
    distribution_for_radius,
    grain_charge_parameter_code,
    reference_mean_charge,
    reference_mean_charge_float32,
    runtime_lookup_float32,
    runtime_lookup_radius_float32,
    write_audit,
    write_fortran_include,
    yld04_coulomb_log_scale,
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_published_coefficients() -> dict:
    _assert(len(SILICATE_FITS) == 7, "expected all seven published silicate rows")
    _assert(REFERENCE_FIT.alpha == 0.3927, "1000 A alpha drifted")
    _assert(REFERENCE_FIT.k == 3.6493, "1000 A k drifted")
    _assert(REFERENCE_FIT.b == 0.8389, "1000 A b drifted")
    _assert(REFERENCE_FIT.h_z == 372.0, "1000 A h_Z drifted")
    _assert(REFERENCE_FIT.eta_negative == 0.4237, "1000 A eta- drifted")
    return {"passed": True, "published_rows": len(SILICATE_FITS)}


def validate_canonical_environments() -> dict:
    expected_ranges = {
        # Signs and magnitudes for the provisional asymptotic blend.
        "WNM": (130.0, 160.0),
        "CNM": (35.0, 55.0),
        "WIM": (115.0, 145.0),
        "shielded_molecular": (-1.0, 0.0),
    }
    values = {}
    for environment in REPRESENTATIVE_ENVIRONMENTS:
        z_ref = float(
            reference_mean_charge(
                environment.n_h_cm3,
                environment.temperature_k,
                environment.electron_density_cm3,
                environment.hydrogen_ion_fraction,
                environment.radiation_habing,
            )
        )
        lower, upper = expected_ranges[environment.name]
        _assert(
            lower <= z_ref <= upper,
            f"{environment.name} charge {z_ref} outside [{lower}, {upper}]",
        )
        values[environment.name] = z_ref
    return {"passed": True, "reference_charge": values}


def validate_yld04_table1_environments() -> dict:
    expected = {
        "YLD04_CNM": (100.0, 30.0, 0.03, 1.0, 6.0e-6),
        "YLD04_WNM": (6000.0, 0.3, 0.03, 1.0, 5.8e-6),
        "YLD04_WIM": (8000.0, 0.1, 0.0991, 1.0, 3.35e-6),
        "YLD04_MC": (25.0, 300.0, 0.03, 0.1, 11.0e-6),
        "YLD04_DC1": (10.0, 1.0e4, 0.01, 0.01, 80.0e-6),
        "YLD04_DC2": (10.0, 1.0e4, 0.001, 0.001, 80.0e-6),
    }
    values = {}
    for environment in YLD04_TABLE1_ENVIRONMENTS:
        observed = (
            environment.temperature_k,
            environment.n_h_cm3,
            environment.electron_density_cm3,
            environment.radiation_habing,
            environment.magnetic_field_gauss,
        )
        _assert(
            np.allclose(
                observed,
                expected[environment.name],
                rtol=1.0e-14,
                atol=0.0,
            ),
            f"{environment.name} no longer matches YLD04 Table 1",
        )
        values[environment.name] = float(
            reference_mean_charge(
                environment.n_h_cm3,
                environment.temperature_k,
                environment.electron_density_cm3,
                environment.hydrogen_ion_fraction,
                environment.radiation_habing,
            )
        )
    _assert(values["YLD04_CNM"] > 0.0, "YLD04 CNM sign drifted")
    _assert(values["YLD04_WNM"] > 0.0, "YLD04 WNM sign drifted")
    _assert(values["YLD04_WIM"] > 0.0, "YLD04 WIM sign drifted")
    _assert(values["YLD04_DC1"] < 0.0, "YLD04 DC1 sign drifted")
    _assert(values["YLD04_DC2"] < 0.0, "YLD04 DC2 sign drifted")
    return {"passed": True, "reference_charge": values}


def validate_dark_collisional_baseline() -> dict:
    temperature = 100.0
    n_h = 30.0
    electron_density = 1.0e-2
    x_hii_carbon = 0.0
    x_hii_hydrogen = electron_density / n_h
    nu_carbon = float(
        dark_potential_nu(n_h, electron_density, x_hii_carbon)
    )
    nu_hydrogen = float(
        dark_potential_nu(n_h, electron_density, x_hii_hydrogen)
    )
    z_dark = float(
        reference_mean_charge(
            n_h, temperature, electron_density, x_hii_carbon, 0.0
        )
    )
    z_illuminated = float(
        reference_mean_charge(
            n_h, temperature, electron_density, x_hii_carbon, 10.0
        )
    )
    _assert(nu_carbon < nu_hydrogen < 0.0, "H/C ion mass was ignored")
    _assert(z_dark < 0.0, "dark collisional equilibrium is not negative")
    _assert(z_illuminated > 0.0, "external FUV did not overcome dark charging")
    return {
        "passed": True,
        "nu_carbon": nu_carbon,
        "nu_hydrogen": nu_hydrogen,
        "dark_reference_charge": z_dark,
        "illuminated_reference_charge": z_illuminated,
    }


def validate_distributions() -> dict:
    max_normalization_error = 0.0
    max_mean_error = 0.0
    minimum_zero_second = np.inf
    for z_ref in (-16.0, -1.0, 0.0, 1.0, 32.0, 160.0):
        for radius in SIZE_KNOT_RADII_CM:
            charge, probability, moments = distribution_for_radius(radius, z_ref)
            expected_mean = z_ref * radius / (0.1 * MICRON_CM)
            max_normalization_error = max(
                max_normalization_error, abs(moments.normalization - 1.0)
            )
            max_mean_error = max(max_mean_error, abs(moments.mean - expected_mean))
            _assert(np.all(np.isfinite(probability)), "non-finite f_Z")
            _assert(np.all(probability >= 0.0), "negative f_Z")
            _assert(moments.variance > 0.0, "zero charge variance")
            if z_ref == 0.0:
                minimum_zero_second = min(minimum_zero_second, moments.second)
                drag = coulomb_log_moment(charge, probability, 20.0)
                _assert(drag > 0.0, "Coulomb drag vanished at <Z>=0")
    _assert(max_normalization_error < 3.0e-15, "f_Z normalization error")
    # Integer sampling causes only a tiny centroid offset for these large grains.
    _assert(max_mean_error < 3.0e-5, "discrete f_Z first moment drift")
    return {
        "passed": True,
        "maximum_normalization_error": max_normalization_error,
        "maximum_mean_abs_error": max_mean_error,
        "minimum_second_moment_at_zero_mean": minimum_zero_second,
    }


def validate_float32_table() -> dict:
    table = build_runtime_table()
    _assert(
        np.asarray(table["second_moment"]).dtype == np.float32,
        "m2 table is not float32",
    )
    _assert(
        np.asarray(table["z2_log_abs_z"]).dtype == np.float32,
        "m2log table is not float32",
    )
    _assert(
        np.asarray(table["size_knot_radii_cm"]).dtype == np.float32,
        "radii table is not float32",
    )

    probe_zref = np.concatenate(
        (
            np.linspace(-60.0, -0.25, 41),
            np.linspace(-0.2, 0.2, 41),
            np.geomspace(0.25, 0.95 * TABLE_ZREF_MAX, 161),
        )
    )
    max_m2_relative = 0.0
    max_m2log_relative = 0.0
    max_drag_relative = 0.0
    actual_log_c = [
        yld04_coulomb_log_scale(
            environment.temperature_k,
            environment.electron_density_cm3,
        )
        for environment in CANONICAL_ENVIRONMENTS
    ]
    for family, radius in enumerate(SIZE_KNOT_RADII_CM):
        for z_ref in probe_zref:
            _, _, reference = distribution_for_radius(radius, float(z_ref))
            m2_f32, m2log_f32 = runtime_lookup_float32(
                table, family, float(z_ref)
            )
            max_m2_relative = max(
                max_m2_relative,
                abs(m2_f32 - reference.second) / max(reference.second, 1.0e-12),
            )
            max_m2log_relative = max(
                max_m2log_relative,
                abs(m2log_f32 - reference.z2_log_abs_z)
                / max(abs(reference.z2_log_abs_z), 1.0),
            )
            for log_c in actual_log_c:
                drag_reference = max(
                    reference.second * log_c
                    - reference.z2_log_abs_z,
                    0.0,
                )
                drag_f32 = max(m2_f32 * log_c - m2log_f32, 0.0)
                max_drag_relative = max(
                    max_drag_relative,
                    abs(drag_f32 - drag_reference)
                    / max(abs(drag_reference), 1.0e-10),
                )
    _assert(max_m2_relative < 6.0e-3, "float32 m2 interpolation error")
    _assert(max_m2log_relative < 1.2e-2, "float32 m2log interpolation error")
    _assert(max_drag_relative < 6.0e-3, "float32 Coulomb moment error")
    return {
        "passed": True,
        "maximum_m2_relative_error": max_m2_relative,
        "maximum_m2log_scaled_error": max_m2log_relative,
        "maximum_coulomb_moment_relative_error": max_drag_relative,
        "actual_coulomb_log_scale_range": [
            min(actual_log_c),
            max(actual_log_c),
        ],
    }


def validate_continuous_radius_lookup() -> dict:
    table = build_runtime_table()
    radii = []
    for lower, upper in zip(
        SIZE_KNOT_RADII_CM[:-1], SIZE_KNOT_RADII_CM[1:]
    ):
        radii.extend(
            np.geomspace(lower, upper, 7, dtype=np.float64)[1:-1]
        )
    probe_zref = np.concatenate(
        (
            np.linspace(-32.0, -0.2, 17),
            np.linspace(-0.15, 0.15, 13),
            np.geomspace(0.2, 2048.0, 49),
        )
    )
    log_c_values = [
        yld04_coulomb_log_scale(
            environment.temperature_k,
            environment.electron_density_cm3,
        )
        for environment in YLD04_TABLE1_ENVIRONMENTS
    ]
    max_m2_relative = 0.0
    max_m2log_scaled = 0.0
    max_drag_relative = 0.0
    for radius in radii:
        for z_ref in probe_zref:
            _, _, reference = distribution_for_radius(radius, float(z_ref))
            m2_f32, m2log_f32 = runtime_lookup_radius_float32(
                table, float(radius), float(z_ref)
            )
            max_m2_relative = max(
                max_m2_relative,
                abs(m2_f32 - reference.second)
                / max(reference.second, 1.0e-12),
            )
            max_m2log_scaled = max(
                max_m2log_scaled,
                abs(m2log_f32 - reference.z2_log_abs_z)
                / max(abs(reference.z2_log_abs_z), 1.0),
            )
            for log_c in log_c_values:
                drag_reference = max(
                    reference.second * log_c
                    - reference.z2_log_abs_z,
                    0.0,
                )
                drag_runtime = max(
                    m2_f32 * log_c - m2log_f32,
                    0.0,
                )
                max_drag_relative = max(
                    max_drag_relative,
                    abs(drag_runtime - drag_reference)
                    / max(abs(drag_reference), 1.0e-10),
                )
    _assert(max_m2_relative < 1.5e-2, "continuous-radius m2 error")
    # The integer-Z distribution changes nonlinearly near |Z|~1.  With only
    # twelve mandated size knots, the auxiliary log moment is allowed a wider
    # scaled error; the physical YLD04 Coulomb combination remains below 1.5%.
    _assert(max_m2log_scaled < 6.0e-2, "continuous-radius m2log error")
    _assert(max_drag_relative < 1.5e-2, "continuous-radius drag error")
    for outside in (
        0.99 * SIZE_KNOT_RADII_CM[0],
        1.01 * SIZE_KNOT_RADII_CM[-1],
    ):
        try:
            runtime_lookup_radius_float32(table, outside, 0.0)
        except ValueError:
            pass
        else:
            raise AssertionError("out-of-range radius was silently clamped")
    return {
        "passed": True,
        "probe_radius_count": len(radii),
        "maximum_m2_relative_error": max_m2_relative,
        "maximum_m2log_scaled_error": max_m2log_scaled,
        "maximum_coulomb_moment_relative_error": max_drag_relative,
    }


def validate_runtime_axis_envelope() -> dict:
    """Ensure the compact table does not silently clamp the accepted envelope."""
    temperatures = np.geomspace(10.0, 2.0e4, 19)
    electron_densities = np.geomspace(1.0e-5, 10.0, 17)
    radiation_fields = np.concatenate(([0.0], np.geomspace(1.0e-5, 10.0, 15)))
    minimum = np.inf
    maximum = -np.inf
    n_h = 10.0
    for hydrogen_ion_share in (0.0, 0.5, 1.0):
        for temperature in temperatures:
            for electron_density in electron_densities:
                x_hii = hydrogen_ion_share * electron_density / n_h
                charge = reference_mean_charge(
                    n_h,
                    temperature,
                    electron_density,
                    x_hii,
                    radiation_fields,
                )
                minimum = min(minimum, float(np.min(charge)))
                maximum = max(maximum, float(np.max(charge)))
    _assert(minimum > TABLE_ZREF_MIN, "runtime table lower axis clamps envelope")
    _assert(maximum < TABLE_ZREF_MAX, "runtime table upper axis clamps envelope")
    table = build_runtime_table()
    for boundary in (TABLE_ZREF_MIN, TABLE_ZREF_MAX):
        runtime_lookup_float32(table, 0, boundary)
    for outside in (
        np.nextafter(np.float32(TABLE_ZREF_MIN), np.float32(-np.inf)),
        np.nextafter(np.float32(TABLE_ZREF_MAX), np.float32(np.inf)),
    ):
        try:
            runtime_lookup_float32(table, 0, outside)
        except ValueError:
            pass
        else:
            raise AssertionError("out-of-envelope lookup was silently clamped")
    return {
        "passed": True,
        "minimum_reference_charge": minimum,
        "maximum_reference_charge": maximum,
        "table_bounds": [TABLE_ZREF_MIN, TABLE_ZREF_MAX],
    }


def validate_float32_cell_closure() -> dict:
    temperatures = np.geomspace(10.0, 2.0e4, 19)
    electron_densities = np.geomspace(1.0e-5, 10.0, 17)
    radiation_fields = np.concatenate(([0.0], np.geomspace(1.0e-5, 10.0, 15)))
    maximum_relative_error = 0.0
    n_h = 10.0
    for hydrogen_ion_share in (0.0, 0.5, 1.0):
        for temperature in temperatures:
            for electron_density in electron_densities:
                x_hii = hydrogen_ion_share * electron_density / n_h
                reference = reference_mean_charge(
                    n_h,
                    temperature,
                    electron_density,
                    x_hii,
                    radiation_fields,
                )
                runtime = reference_mean_charge_float32(
                    n_h,
                    temperature,
                    electron_density,
                    x_hii,
                    radiation_fields,
                ).astype(np.float64)
                relative = np.abs(runtime - reference) / np.maximum(
                    np.abs(reference), 1.0
                )
                maximum_relative_error = max(
                    maximum_relative_error, float(np.max(relative))
                )
    _assert(
        maximum_relative_error < 5.0e-5,
        "float32 cell potential closure drift",
    )
    return {
        "passed": True,
        "maximum_scaled_error": maximum_relative_error,
    }


def validate_input_neutrality() -> dict:
    valid = float(
        reference_mean_charge(1.0, 100.0, 1.0e-2, 1.0e-2, 0.0)
    )
    _assert(np.isfinite(valid), "neutral state was rejected")
    for invalid_xhii in (-1.0e-3, 1.001):
        try:
            reference_mean_charge(
                1.0, 100.0, 1.0e-2, invalid_xhii, 0.0
            )
        except ValueError:
            pass
        else:
            raise AssertionError("invalid xHII was accepted")
    try:
        reference_mean_charge(1.0, 100.0, 1.0e-2, 2.0e-2, 0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("xHII*nH > ne was accepted")
    return {"passed": True}


def validate_code_charge_parameter() -> dict:
    unit_l = 3.0857e18
    unit_d = 1.50492957435e-20
    rho_gr = 2.0
    radius_reference = 1.0 * MICRON_CM
    potential = 100.0 * E_ESU / radius_reference
    reference_parameter = float(
        grain_charge_parameter_code(
            radius_reference,
            rho_gr,
            potential,
            unit_l,
            unit_d,
        )
    )
    peak_parameter = float(
        grain_charge_parameter_code(
            0.23 * MICRON_CM,
            rho_gr,
            potential,
            unit_l,
            unit_d,
        )
    )
    _assert(
        np.isclose(
            peak_parameter,
            4851.240862014688,
            rtol=2.0e-15,
            atol=0.0,
        ),
        "Gaussian-cgs to code charge conversion drifted",
    )
    direct = float(
        charge_parameter_code(
            100.0
            * E_ESU
            / (4.0 * np.pi * rho_gr * radius_reference**3 / 3.0),
            unit_l,
            unit_d,
        )
    )
    _assert(
        np.isclose(reference_parameter, direct, rtol=2.0e-15),
        "q/m and potential code conversions disagree",
    )
    return {
        "passed": True,
        "charge_parameter_at_1micron": reference_parameter,
        "charge_parameter_at_0p23micron": peak_parameter,
        "dimensionless": True,
    }


def validate_timescales() -> dict:
    maximum_ratio = 0.0
    maximum_state = None
    heavy_proxy_coulomb_rate_seen = False
    for environment in CANONICAL_ENVIRONMENTS:
        for record in audit_environment(environment):
            ratio = record["tau_z_over_min"]
            _assert(np.isfinite(ratio), "non-finite charging timescale ratio")
            _assert(
                record["tau_drag_total_s"] <= record["tau_epstein_s"],
                "total drag is slower than Epstein alone",
            )
            if (
                environment.hydrogen_ion_fraction == 0.0
                and environment.electron_density_cm3 > 0.0
                and np.isfinite(record["tau_coulomb_s"])
            ):
                heavy_proxy_coulomb_rate_seen = True
            if ratio > maximum_ratio:
                maximum_ratio = ratio
                maximum_state = {
                    "environment": environment.name,
                    "size_knot": record["size_knot"],
                    "radius_micron": record["radius_micron"],
                }
    _assert(
        maximum_ratio < 0.1,
        "instantaneous charging is not separated by a factor of ten",
    )
    _assert(
        heavy_proxy_coulomb_rate_seen,
        "heavy-ion proxy failed to contribute Coulomb drag",
    )
    return {
        "passed": True,
        "maximum_tau_z_over_min": maximum_ratio,
        "maximum_state": maximum_state,
        "heavy_proxy_coulomb_rate_seen": heavy_proxy_coulomb_rate_seen,
    }


def validate_writers() -> dict:
    include = Path("/tmp/stage9_charge_table_f32.test.inc")
    audit = Path("/tmp/stage9_charge_timescale_audit.test.json")
    try:
        write_fortran_include(include)
        write_audit(audit)
        include_text = include.read_text()
        payload = json.loads(audit.read_text())
        _assert("_4" in include_text, "Fortran table lacks float32 literals")
        _assert("real(kind=8)" not in include_text, "Fortran table contains float64")
        _assert(
            "dust_charge_reference_min" in include_text
            and "dust_charge_reference_max" in include_text,
            "Fortran table lacks strict reference-charge bounds",
        )
        _assert(payload["maximum_tau_z_over_min"] < 0.1, "audit JSON failed")
    finally:
        include.unlink(missing_ok=True)
        audit.unlink(missing_ok=True)
    return {"passed": True}


def main() -> int:
    checks = {
        "published_coefficients": validate_published_coefficients(),
        "dark_collisional_baseline": validate_dark_collisional_baseline(),
        "canonical_environments": validate_canonical_environments(),
        "yld04_table1_environments": validate_yld04_table1_environments(),
        "normalized_distributions": validate_distributions(),
        "input_neutrality": validate_input_neutrality(),
        "float32_cell_closure": validate_float32_cell_closure(),
        "code_charge_parameter": validate_code_charge_parameter(),
        "runtime_axis_envelope": validate_runtime_axis_envelope(),
        "float32_runtime_table": validate_float32_table(),
        "continuous_radius_lookup": validate_continuous_radius_lookup(),
        "charging_timescales": validate_timescales(),
        "writers": validate_writers(),
    }
    print(json.dumps({"passed": True, "checks": checks}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
