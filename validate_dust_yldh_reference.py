#!/usr/bin/env python3
"""Focused checks for the offline YLDH reference and float32 table."""

from __future__ import annotations

import json
import re
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from dust_yldh_reference import (
    DEFAULT_INTEGRATION_ATOL,
    DEFAULT_INTEGRATION_LIMIT,
    DEFAULT_INTEGRATION_RTOL,
    Grain,
    INTEGRATION_METHOD,
    PrimitiveDiffusion,
    REDUCED_MODEL_EXCLUSIONS,
    RUNTIME_DIMENSIONLESS_TABLE_REVISION,
    _alfven_branch_coefficients,
    _closed_axis_bracket,
    _conservative_dimensionless_ru_drift,
    _conservative_primitive_drift,
    _pchip_axis_indices,
    _tensor_pchip_interpolate,
    build_dimensionless_balanced_gyro_ru_table,
    build_runtime_table,
    default_runtime_dimensionless_axes,
    grain_with_matched_gyrofrequency,
    grain_gyrofrequency_s,
    interpolate_dimensionless_balanced_gyro_ru,
    load_dimensionless_balanced_gyro_ru_table,
    primitive_diffusion,
    psd_lower_factor,
    require_runtime_balanced_mode_signature,
    scale_balanced_gyro_wave_power,
    sha256_file,
    transform_diffusion_to_dimensionless_u_pitch,
    transform_diffusion_to_w_mu,
    write_cuda_fortran_balanced_gyro_include,
    write_dimensionless_balanced_gyro_ru_table,
    write_runtime_table,
    yld04_cnm_context,
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_gaussian_gyrofrequency() -> dict:
    context = yld04_cnm_context()
    grain = Grain(0.1e-4, 3.5, 40.0)
    omega = grain_gyrofrequency_s(grain, context.magnetic_field_gauss)
    doubled_charge = grain_gyrofrequency_s(
        Grain(grain.radius_cm, grain.bulk_density_g_cm3, 80.0),
        context.magnetic_field_gauss,
    )
    doubled_radius = grain_gyrofrequency_s(
        Grain(2.0 * grain.radius_cm, grain.bulk_density_g_cm3, 40.0),
        context.magnetic_field_gauss,
    )
    _assert(omega > 0.0, "gyrofrequency is not positive")
    _assert(
        np.isclose(doubled_charge / omega, 2.0, rtol=2.0e-15),
        "gyrofrequency lost linear charge scaling",
    )
    _assert(
        np.isclose(doubled_radius / omega, 1.0 / 8.0, rtol=2.0e-15),
        "gyrofrequency lost inverse mass scaling",
    )
    return {"passed": True, "omega_s": omega}


def validate_resonance_support() -> dict:
    context = yld04_cnm_context()
    resolved = Grain(0.1e-4, 3.5, 40.0)
    unresolved = Grain(0.001e-4, 3.5, 40.0)
    active = primitive_diffusion(
        context,
        resolved,
        1.0e5,
        0.0,
        modes=("fast",),
        quadrature_order=28,
    )
    cutoff = primitive_diffusion(
        context,
        unresolved,
        1.0e5,
        0.0,
        modes=("fast", "alfven"),
        quadrature_order=28,
    )
    neutral = primitive_diffusion(
        context,
        Grain(0.1e-4, 3.5, 0.0),
        1.0e5,
        0.0,
        modes=("fast", "alfven"),
        quadrature_order=16,
    )
    _assert(active.d_pp > 0.0, "supported fast resonance has zero Dpp")
    _assert(active.d_xi_xi > 0.0, "supported fast resonance has zero D_xixi")
    _assert(
        active.fast_support_fraction > 0.0,
        "supported fast resonance was not counted",
    )
    _assert(cutoff.d_pp == 0.0, "sub-cutoff grain retained diffusion")
    _assert(cutoff.d_xi_xi == 0.0, "sub-cutoff grain retained scattering")
    _assert(neutral.d_pp == 0.0, "neutral grain retained YLDH diffusion")
    return {
        "passed": True,
        "active_d_pp": active.d_pp,
        "active_d_xi_xi": active.d_xi_xi,
        "active_fast_support_fraction": active.fast_support_fraction,
    }


def validate_balanced_alfven_relation() -> dict:
    context = yld04_cnm_context()
    # This point lies inside the declared Alfvén support.  Doubling its charge
    # would move k_res above the cutoff and would test an empty integral rather
    # than the balanced-cascade coefficient relation.
    grain = Grain(0.2e-4, 3.5, 40.0)
    speed = 3.0e5
    pitch = 0.25
    omega = grain_gyrofrequency_s(grain, context.magnetic_field_gauss)
    branches = {}
    for propagation_sign in (-1, 1):
        branch = _alfven_branch_coefficients(
            context,
            grain,
            speed,
            pitch,
            omega,
            propagation_sign,
            24,
            1.0e-8,
            DEFAULT_INTEGRATION_LIMIT,
        )
        _assert(branch[0] > 0.0, "Alfvén branch has zero support")
        measured = (grain.mass_g * speed) ** 2 * branch[1] / branch[0]
        expected = (
            speed / context.alfven_speed_cm_s
            + propagation_sign * pitch
        ) ** 2
        _assert(
            np.isclose(measured, expected, rtol=3.0e-13),
            "signed Alfvén branch violates the Appendix-B ratio",
        )
        branches[propagation_sign] = {
            "d_pp": branch[0],
            "d_xi_xi": branch[1],
            "measured_ratio": measured,
            "expected_ratio": expected,
        }

    diffusion = primitive_diffusion(
        context,
        grain,
        speed,
        pitch,
        modes=("alfven",),
        quadrature_order=24,
        integration_rtol=1.0e-8,
    )
    mirrored = primitive_diffusion(
        context,
        grain,
        speed,
        -pitch,
        modes=("alfven",),
        quadrature_order=24,
        integration_rtol=1.0e-8,
    )
    _assert(diffusion.d_pp > 0.0, "Alfvén resonance has zero support")
    _assert(
        diffusion.alfven_support_fraction > 0.0,
        "Alfvén validation point lies outside the declared cutoff",
    )
    _assert(diffusion.d_p_xi == 0.0, "balanced cascade produced D_p_xi")
    _assert(
        np.isclose(diffusion.d_pp, mirrored.d_pp, rtol=3.0e-12),
        "balanced Alfvén Dpp is not even in pitch cosine",
    )
    _assert(
        np.isclose(diffusion.d_xi_xi, mirrored.d_xi_xi, rtol=3.0e-12),
        "balanced Alfvén D_xixi is not even in pitch cosine",
    )
    _assert(
        np.isclose(
            diffusion.alfven_support_fraction,
            mirrored.alfven_support_fraction,
            rtol=3.0e-12,
        ),
        "balanced Alfvén support is not even in pitch cosine",
    )
    return {
        "passed": True,
        "branches": branches,
        "pitch_symmetry_relative": float(
            abs(diffusion.d_pp - mirrored.d_pp) / diffusion.d_pp
        ),
    }


def validate_adaptive_coefficient_convergence() -> dict:
    context = yld04_cnm_context()
    cases = (
        ("fast", Grain(0.1e-4, 3.5, 40.0), 1.0e5, 0.0),
        ("alfven", Grain(0.2e-4, 3.5, 40.0), 3.0e5, 0.25),
        ("fast_near_cutoff", Grain(0.0575e-4, 3.5, 40.0), 1.0e5, 0.0),
    )
    report = {}
    maximum_outer_error = 0.0
    maximum_tolerance_error = 0.0
    for name, grain, speed, pitch in cases:
        mode = "alfven" if name == "alfven" else "fast"
        medium = primitive_diffusion(
            context,
            grain,
            speed,
            pitch,
            modes=(mode,),
            quadrature_order=64,
            integration_rtol=3.0e-7,
        )
        fine = primitive_diffusion(
            context,
            grain,
            speed,
            pitch,
            modes=(mode,),
            quadrature_order=96,
            integration_rtol=3.0e-8,
        )
        tight = primitive_diffusion(
            context,
            grain,
            speed,
            pitch,
            modes=(mode,),
            quadrature_order=96,
            integration_rtol=3.0e-9,
        )
        _assert(fine.d_pp > 0.0, f"{name} convergence point has zero Dpp")
        _assert(
            fine.d_xi_xi > 0.0,
            f"{name} convergence point has zero D_xixi",
        )
        outer_error = max(
            abs(medium.d_pp - fine.d_pp) / fine.d_pp,
            abs(medium.d_xi_xi - fine.d_xi_xi) / fine.d_xi_xi,
        )
        tolerance_error = max(
            abs(tight.d_pp - fine.d_pp) / tight.d_pp,
            abs(tight.d_xi_xi - fine.d_xi_xi) / tight.d_xi_xi,
        )
        maximum_outer_error = max(maximum_outer_error, outer_error)
        maximum_tolerance_error = max(
            maximum_tolerance_error, tolerance_error
        )
        report[name] = {
            "outer_order_relative": float(outer_error),
            "inner_tolerance_relative": float(tolerance_error),
        }
    _assert(
        maximum_outer_error < 1.0e-3,
        "outer quadrature has not converged at a reference point",
    )
    _assert(
        maximum_tolerance_error < 3.0e-6,
        "adaptive resonance integral has not converged",
    )
    return {
        "passed": True,
        "integration_method": INTEGRATION_METHOD,
        "maximum_outer_order_relative": maximum_outer_error,
        "maximum_inner_tolerance_relative": maximum_tolerance_error,
        "cases": report,
    }


def validate_coordinate_transform() -> dict:
    mass = 2.3
    field = 0.7
    speed = 4.1
    pitch = -0.35
    diffusion = PrimitiveDiffusion(1.7, 0.13, 0.42, 0.0, 0.0)
    transformed = transform_diffusion_to_w_mu(
        diffusion, mass, field, speed, pitch
    )
    jacobian = np.array(
        [
            [pitch / mass, speed],
            [
                speed * (1.0 - pitch**2) / (mass * field),
                -(speed**2) * pitch / field,
            ],
        ]
    )
    expected = jacobian @ diffusion.matrix @ jacobian.T
    _assert(
        np.allclose(transformed, expected, rtol=2.0e-15, atol=2.0e-15),
        "(p,xi) to (w,mu_adb) tensor transform is wrong",
    )
    eigenvalue = np.linalg.eigvalsh(transformed)
    _assert(np.min(eigenvalue) >= 0.0, "coordinate transform broke PSD")
    return {
        "passed": True,
        "minimum_eigenvalue": float(np.min(eigenvalue)),
    }


def validate_phase_space_drift() -> dict:
    momentum = np.geomspace(1.0, 8.0, 7)
    pitch = np.linspace(-0.8, 0.8, 7)
    d_pp = np.full((momentum.size, pitch.size), 3.0)
    d_p_xi = np.zeros_like(d_pp)
    d_xi_xi = np.full_like(d_pp, 0.4)
    drift_p, drift_xi = _conservative_primitive_drift(
        d_pp, d_p_xi, d_xi_xi, momentum, pitch
    )
    expected_p = 6.0 / momentum[:, None]
    _assert(
        np.allclose(drift_p, expected_p, rtol=2.0e-14, atol=2.0e-14),
        "p^2 phase-space measure is absent from the drift",
    )
    _assert(
        np.max(np.abs(drift_xi)) < 2.0e-15,
        "constant pitch diffusion produced spurious drift",
    )
    return {"passed": True}


def validate_psd_factor() -> dict:
    matrix = np.array([[2.0, -0.6], [-0.6, 0.4]])
    factor, correction = psd_lower_factor(matrix)
    _assert(correction < 2.0e-15, "valid PSD matrix was projected")
    _assert(
        np.allclose(factor @ factor.T, 2.0 * matrix, rtol=2.0e-15),
        "noise factor does not reconstruct 2D",
    )
    mixed_scale = np.array([[1.57830333e-2, 0.0], [0.0, 4.94874499e19]])
    mixed_factor, mixed_correction = psd_lower_factor(mixed_scale)
    mixed_reconstructed = 0.5 * mixed_factor @ mixed_factor.T
    _assert(mixed_factor[0, 0] > 0.0, "mixed-unit factor lost D_ww")
    _assert(
        np.allclose(
            np.diag(mixed_reconstructed),
            np.diag(mixed_scale),
            rtol=3.0e-14,
        ),
        "mixed-unit factor lost a component variance",
    )
    _assert(
        mixed_correction < 3.0e-14,
        "mixed-unit PSD factor required a correction",
    )

    roundoff_indefinite = np.array(
        [[1.0, 1.0 + 5.0e-14], [1.0 + 5.0e-14, 1.0]]
    )
    _, roundoff_correction = psd_lower_factor(roundoff_indefinite)
    _assert(
        0.0 < roundoff_correction < 1.0e-12,
        "roundoff-level PSD projection was not reported",
    )

    indefinite = np.array([[1.0, 2.0], [2.0, 1.0]])
    try:
        psd_lower_factor(indefinite)
    except ValueError:
        materially_indefinite_rejected = True
    else:
        materially_indefinite_rejected = False
    _assert(
        materially_indefinite_rejected,
        "materially indefinite tensor was silently projected",
    )
    return {
        "passed": True,
        "mixed_scale_reconstruction_relative": mixed_correction,
        "roundoff_projection_relative": roundoff_correction,
    }


def validate_float32_runtime_table() -> dict:
    context = yld04_cnm_context()
    grain = Grain(0.1e-4, 3.5, 40.0)
    speed = np.geomspace(3.0e4, 2.0e6, 5)
    pitch = np.linspace(-0.9, 0.9, 5)
    table = build_runtime_table(
        context,
        grain,
        speed,
        pitch,
        modes=("fast", "alfven"),
        quadrature_order=48,
    )
    arrays = {
        name: value
        for name, value in table.items()
        if isinstance(value, np.ndarray)
    }
    for name, value in arrays.items():
        _assert(value.dtype == np.float32, f"{name} is not float32")
        _assert(np.all(np.isfinite(value)), f"{name} is non-finite")

    minimum_eigenvalue = np.inf
    maximum_factor_relative_error = 0.0
    maximum_component_relative_error = 0.0
    for i_speed, speed_value in enumerate(speed):
        for i_pitch, pitch_value in enumerate(pitch):
            factor = np.array(
                [
                    [table["noise_l11"][i_speed, i_pitch], 0.0],
                    [
                        table["noise_l21"][i_speed, i_pitch],
                        table["noise_l22"][i_speed, i_pitch],
                    ],
                ],
                dtype=np.float64,
            )
            covariance_rate = factor @ factor.T
            minimum_eigenvalue = min(
                minimum_eigenvalue,
                float(np.min(np.linalg.eigvalsh(covariance_rate))),
            )
            diffusion = primitive_diffusion(
                context,
                grain,
                float(speed_value),
                float(pitch_value),
                modes=("fast", "alfven"),
                quadrature_order=48,
            )
            reference = 2.0 * transform_diffusion_to_w_mu(
                diffusion,
                grain.mass_g,
                context.magnetic_field_gauss,
                float(speed_value),
                float(pitch_value),
            )
            error = np.linalg.norm(covariance_rate - reference)
            scale = max(np.linalg.norm(reference), 1.0e-100)
            maximum_factor_relative_error = max(
                maximum_factor_relative_error, float(error / scale)
            )
            for diagonal_index in (0, 1):
                reference_variance = reference[
                    diagonal_index, diagonal_index
                ]
                if reference_variance > 0.0:
                    component_error = abs(
                        covariance_rate[diagonal_index, diagonal_index]
                        - reference_variance
                    ) / reference_variance
                    maximum_component_relative_error = max(
                        maximum_component_relative_error,
                        float(component_error),
                    )
    _assert(
        minimum_eigenvalue >= -1.0e-20,
        "float32 factor produced a non-PSD covariance",
    )
    _assert(
        maximum_factor_relative_error < 2.0e-5,
        "float32 factor no longer matches the float64 reference",
    )
    _assert(
        maximum_component_relative_error < 2.0e-5,
        "float32 factor lost a component variance",
    )
    _assert(
        np.count_nonzero(table["noise_l11"]) > 0,
        "float32 table lost the parallel stochastic channel",
    )
    _assert(
        table["maximum_psd_projection_relative"] < 2.0e-12,
        "physical tensor required a material PSD projection",
    )

    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "table.npz"
        write_runtime_table(output, table, context, grain)
        with np.load(output, allow_pickle=False) as stored:
            _assert(stored["noise_l11"].dtype == np.float32, "stored factor drift")
            metadata = json.loads(str(stored["metadata_json"]))
            _assert(
                metadata["update"]
                == "dy = drift*dt + L*normal(0,1)*sqrt(dt)",
                "runtime update convention is ambiguous",
            )
            _assert(
                metadata["reference_scope"]
                == "fixed-context-single-grain-cgs",
                "fixed-context table is mislabeled",
            )
            _assert(
                metadata["integration_method"] == INTEGRATION_METHOD,
                "integration method is absent from metadata",
            )
            _assert(
                metadata["integration_relative_tolerance"]
                == DEFAULT_INTEGRATION_RTOL,
                "integration tolerance is absent from metadata",
            )
            _assert(
                metadata["integration_absolute_tolerance"]
                == DEFAULT_INTEGRATION_ATOL,
                "integration absolute tolerance is absent from metadata",
            )
            _assert(
                metadata["outer_quadrature_order"] == 48,
                "outer quadrature order is absent from metadata",
            )
        stored_bytes = output.stat().st_size
    _assert(stored_bytes < 16 * 1024, "focused runtime table is not compact")
    return {
        "passed": True,
        "minimum_covariance_rate_eigenvalue": minimum_eigenvalue,
        "maximum_factor_relative_error": maximum_factor_relative_error,
        "maximum_component_relative_error": maximum_component_relative_error,
        "stored_bytes": stored_bytes,
    }


def _lookup_covariance(lookup: dict[str, np.float32]) -> np.ndarray:
    factor = np.array(
        [
            [lookup["noise_l11"], 0.0],
            [lookup["noise_l21"], lookup["noise_l22"]],
        ],
        dtype=np.float64,
    )
    return factor @ factor.T


def validate_complete_ru_ito_drift() -> dict:
    log_resonance = np.linspace(-2.0, 2.0, 5)
    log_speed = np.linspace(-1.0, 1.0, 5)
    pitch = np.linspace(-0.8, 0.8, 5)
    log_r = log_resonance[:, None, None]
    log_u = log_speed[None, :, None]
    xi = pitch[None, None, :]
    d_uu = 20.0 + 2.0 * log_r + 3.0 * log_u + 0.5 * xi
    d_u_pitch = 5.0 + 0.7 * log_r - 0.2 * log_u + 0.3 * xi
    d_pitch_pitch = 15.0 - 0.1 * log_r + 0.5 * log_u + 0.8 * xi
    drift_u, drift_pitch = _conservative_dimensionless_ru_drift(
        d_uu,
        d_u_pitch,
        d_pitch_pitch,
        log_resonance,
        log_speed,
        pitch,
    )
    speed = np.exp(log_speed)[None, :, None]
    expected_u = 5.0 / speed + 0.3 + 2.0 * d_uu / speed
    expected_pitch = (
        0.5 / speed + 0.8 + 2.0 * d_u_pitch / speed
    )
    maximum_error = max(
        float(np.max(np.abs(drift_u - expected_u))),
        float(np.max(np.abs(drift_pitch - expected_pitch))),
    )
    _assert(
        maximum_error < 2.0e-13,
        "log-R contribution is missing from the fixed-gyro Ito drift",
    )
    return {"passed": True, "maximum_absolute_error": maximum_error}


def validate_local_pchip_factor_interpolation() -> dict:
    axes = (
        np.asarray([-3.0, -1.4, -0.2, 0.7, 2.1], dtype=np.float32),
        np.asarray([-2.0, -0.8, 0.1, 1.6, 3.0], dtype=np.float32),
        np.asarray([-0.9, -0.3, 0.0, 0.4, 0.9], dtype=np.float32),
    )
    rr, uu, xx = np.meshgrid(*axes, indexing="ij")
    values = np.asarray(
        2.0 + np.exp(0.15 * rr) + 0.2 * uu**2 + 0.1 * xx,
        dtype=np.float32,
    )
    reference = RegularGridInterpolator(
        tuple(axis.astype(np.float64) for axis in axes),
        values.astype(np.float64),
        method="pchip",
    )
    maximum_relative_error = 0.0
    for coordinates in (
        (-2.2, -1.1, -0.5),
        (-0.6, 0.6, 0.2),
        (1.4, 2.2, 0.7),
    ):
        lower_weight = tuple(
            _closed_axis_bracket(
                axis.astype(np.float64), coordinate
            )
            for axis, coordinate in zip(axes, coordinates)
        )
        indices = tuple(
            _pchip_axis_indices(axis.size, lower, weight)
            for axis, (lower, weight) in zip(axes, lower_weight)
        )
        measured = _tensor_pchip_interpolate(
            axes, values, coordinates, indices
        )
        expected = np.asarray(reference(coordinates)).item()
        maximum_relative_error = max(
            maximum_relative_error,
            abs(float(measured) - expected) / abs(expected),
        )
    _assert(
        maximum_relative_error < 2.0e-6,
        "local float32 PCHIP differs from the SciPy reference",
    )
    return {
        "passed": True,
        "maximum_relative_error": maximum_relative_error,
    }


def validate_matched_gyro_grain_invariance() -> dict:
    context = yld04_cnm_context()
    geometries = (
        Grain(0.0575e-4, 2.2, 1.0),
        Grain(0.92e-4, 3.5, 1.0),
    )
    maximum_relative_error = 0.0
    maximum_gyro_error = 0.0
    for omega_s, speed_ratio, pitch in (
        (1.0e-11, 0.4, -0.35),
        (3.0e-10, 1.5, 0.2),
        (2.0e-9, 5.0, 0.65),
    ):
        tensors = []
        for geometry in geometries:
            grain = grain_with_matched_gyrofrequency(
                geometry, context.magnetic_field_gauss, omega_s
            )
            measured_omega = grain_gyrofrequency_s(
                grain, context.magnetic_field_gauss
            )
            maximum_gyro_error = max(
                maximum_gyro_error,
                abs(measured_omega - omega_s) / omega_s,
            )
            diffusion = primitive_diffusion(
                context,
                grain,
                speed_ratio * context.alfven_speed_cm_s,
                pitch,
                modes=("fast", "alfven"),
                quadrature_order=32,
            )
            tensors.append(
                transform_diffusion_to_dimensionless_u_pitch(
                    diffusion, grain.mass_g, context
                )
            )
        scale = max(np.linalg.norm(tensors[0]), 1.0e-100)
        maximum_relative_error = max(
            maximum_relative_error,
            float(np.linalg.norm(tensors[1] - tensors[0]) / scale),
        )
    _assert(
        maximum_gyro_error < 3.0e-16,
        "matched grain geometry did not preserve gyrofrequency",
    )
    _assert(
        maximum_relative_error < 3.0e-13,
        "normalized tensor depends on grain mass or size at fixed Omega",
    )
    return {
        "passed": True,
        "maximum_gyrofrequency_relative_error": maximum_gyro_error,
        "maximum_tensor_relative_error": maximum_relative_error,
        "mass_ratio": geometries[1].mass_g / geometries[0].mass_g,
    }


def _runtime_basis_reference_covariance(
    context,
    reference_grain_geometry,
    resonance_ratio,
    speed_over_alfven,
    pitch_cosine,
    quadrature_order,
):
    speed_cm_s = speed_over_alfven * context.alfven_speed_cm_s
    omega_s = speed_cm_s / (
        resonance_ratio * context.injection_scale_cm
    )
    grain = grain_with_matched_gyrofrequency(
        reference_grain_geometry,
        context.magnetic_field_gauss,
        omega_s,
    )
    diffusion = primitive_diffusion(
        context,
        grain,
        speed_cm_s,
        pitch_cosine,
        modes=("fast", "alfven"),
        quadrature_order=quadrature_order,
    )
    generator_power = (
        (
            context.alfven_injection_velocity_cm_s
            / context.alfven_speed_cm_s
        )
        ** 2
        + (
            context.fast_injection_velocity_cm_s
            / context.alfven_speed_cm_s
        )
        ** 2
    )
    return (
        2.0
        * transform_diffusion_to_dimensionless_u_pitch(
            diffusion, grain.mass_g, context
        )
        / generator_power
    )


def validate_runtime_ru_table() -> dict:
    context = yld04_cnm_context()
    grain_geometry = Grain(0.1e-4, 3.5, 1.0)
    log_resonance = np.linspace(np.log(1.0e-4), np.log(0.1), 7)
    log_speed = np.linspace(np.log(0.25), np.log(6.0), 6)
    pitch = np.linspace(-0.8, 0.8, 7)
    quadrature_order = 28
    started = time.perf_counter()
    table = build_dimensionless_balanced_gyro_ru_table(
        context,
        grain_geometry,
        log_resonance,
        log_speed,
        pitch,
        quadrature_order=quadrature_order,
    )
    generation_seconds = time.perf_counter() - started
    shape = (log_resonance.size, log_speed.size, pitch.size)
    for name, value in table.items():
        if isinstance(value, np.ndarray):
            _assert(value.dtype == np.float32, f"{name} is not float32")
            _assert(np.all(np.isfinite(value)), f"{name} is non-finite")
    _assert(
        table["drift_speed_over_alfven"].shape == shape,
        "runtime table has the wrong axis order",
    )

    maximum_node_relative_error = 0.0
    for i_resonance, i_speed, i_pitch in (
        (1, 1, 1),
        (3, 2, 3),
        (5, 4, 5),
    ):
        resonance_value = float(
            np.exp(table["log_resonance_ratio"][i_resonance])
        )
        speed_value = float(
            np.exp(table["log_speed_over_alfven"][i_speed])
        )
        pitch_value = float(table["pitch_cosine"][i_pitch])
        lookup = interpolate_dimensionless_balanced_gyro_ru(
            table, resonance_value, speed_value, pitch_value
        )
        for name in (
            "drift_speed_over_alfven",
            "drift_pitch_cosine",
            "noise_l11",
            "noise_l21",
            "noise_l22",
        ):
            _assert(
                lookup[name]
                == table[name][i_resonance, i_speed, i_pitch],
                f"runtime interpolation is not exact at a {name} node",
            )
        reference = _runtime_basis_reference_covariance(
            context,
            grain_geometry,
            resonance_value,
            speed_value,
            pitch_value,
            quadrature_order,
        )
        covariance = _lookup_covariance(lookup)
        error = np.linalg.norm(covariance - reference) / max(
            np.linalg.norm(reference), 1.0e-100
        )
        maximum_node_relative_error = max(
            maximum_node_relative_error, float(error)
        )
    _assert(
        maximum_node_relative_error < 2.0e-5,
        "runtime float32 factor disagrees with float64 reference",
    )

    maximum_midpoint_relative_error = 0.0
    minimum_interpolated_eigenvalue = np.inf
    rejected_support_transition_cells = 0
    accepted_zero_support_cells = 0
    accepted_active_cells = []
    for i_resonance in range(log_resonance.size - 1):
        for i_speed in range(log_speed.size - 1):
            for i_pitch in range(pitch.size - 1):
                resonance_mid = float(
                    np.exp(
                        0.5
                        * (
                            table["log_resonance_ratio"][i_resonance]
                            + table["log_resonance_ratio"][
                                i_resonance + 1
                            ]
                        )
                    )
                )
                speed_mid = float(
                    np.exp(
                        0.5
                        * (
                            table["log_speed_over_alfven"][i_speed]
                            + table["log_speed_over_alfven"][i_speed + 1]
                        )
                    )
                )
                pitch_mid = 0.5 * float(
                    table["pitch_cosine"][i_pitch]
                    + table["pitch_cosine"][i_pitch + 1]
                )
                try:
                    lookup = interpolate_dimensionless_balanced_gyro_ru(
                        table, resonance_mid, speed_mid, pitch_mid
                    )
                except ValueError as error:
                    _assert(
                        "resonance-support boundary" in str(error),
                        "an in-envelope midpoint failed for the wrong reason",
                    )
                    rejected_support_transition_cells += 1
                    continue
                covariance = _lookup_covariance(lookup)
                minimum_interpolated_eigenvalue = min(
                    minimum_interpolated_eigenvalue,
                    float(np.min(np.linalg.eigvalsh(covariance))),
                )
                if (
                    lookup["fast_support_fraction"] == 0.0
                    and lookup["alfven_support_fraction"] == 0.0
                ):
                    accepted_zero_support_cells += 1
                else:
                    accepted_active_cells.append(
                        (
                            resonance_mid,
                            speed_mid,
                            pitch_mid,
                            covariance,
                        )
                    )
    _assert(
        rejected_support_transition_cells > 0,
        "small runtime table did not exercise support-boundary rejection",
    )
    _assert(
        len(accepted_active_cells) >= 3,
        "smooth active cells were incorrectly rejected",
    )
    for support_value in (0.0, 1.0):
        uniform_support_table = dict(table)
        for name in (
            "fast_support_fraction",
            "alfven_support_fraction",
        ):
            uniform_support_table[name] = np.full_like(
                table[name], support_value
            )
        interpolate_dimensionless_balanced_gyro_ru(
            uniform_support_table,
            float(np.exp(table["log_resonance_ratio"][3])),
            float(np.exp(table["log_speed_over_alfven"][3])),
            0.5
            * float(
                table["pitch_cosine"][3]
                + table["pitch_cosine"][4]
            ),
        )
    sample_indices = {
        len(accepted_active_cells) // 4,
        len(accepted_active_cells) // 2,
        3 * len(accepted_active_cells) // 4,
    }
    for sample_index in sorted(sample_indices):
        resonance_mid, speed_mid, pitch_mid, covariance = (
            accepted_active_cells[sample_index]
        )
        reference = _runtime_basis_reference_covariance(
            context,
            grain_geometry,
            resonance_mid,
            speed_mid,
            pitch_mid,
            quadrature_order,
        )
        error = np.linalg.norm(covariance - reference) / max(
            np.linalg.norm(reference), 1.0e-100
        )
        maximum_midpoint_relative_error = max(
            maximum_midpoint_relative_error, float(error)
        )
    _assert(
        minimum_interpolated_eigenvalue >= -1.0e-20,
        "trilinear lower-factor interpolation broke PSD",
    )
    _assert(
        maximum_midpoint_relative_error < 0.5,
        "small-grid runtime interpolation lost the direct tensor",
    )

    central = interpolate_dimensionless_balanced_gyro_ru(
        table,
        float(np.exp(table["log_resonance_ratio"][3])),
        float(np.exp(table["log_speed_over_alfven"][3])),
        float(table["pitch_cosine"][3]),
    )
    central_covariance = _lookup_covariance(central)
    for amplitude in (0.0, 0.25, 4.0):
        scaled = scale_balanced_gyro_wave_power(central, amplitude)
        scaled_covariance = _lookup_covariance(scaled)
        _assert(
            np.allclose(
                scaled_covariance,
                amplitude * central_covariance,
                rtol=3.0e-7,
                atol=1.0e-30,
            ),
            "external wave power did not scale the tensor linearly",
        )
        for name in (
            "drift_speed_over_alfven",
            "drift_pitch_cosine",
        ):
            _assert(
                np.isclose(
                    scaled[name],
                    amplitude * central[name],
                    rtol=3.0e-7,
                    atol=1.0e-30,
                ),
                "external wave power did not scale the Ito drift",
            )
    for invalid_amplitude in (-1.0, np.nan):
        try:
            scale_balanced_gyro_wave_power(central, invalid_amplitude)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid wave-power amplitude was accepted")

    resonance_bounds = np.exp(
        table["log_resonance_ratio"].astype(np.float64)
    )
    speed_bounds = np.exp(
        table["log_speed_over_alfven"].astype(np.float64)
    )
    for resonance_value, speed_value, pitch_value in (
        (0.99 * resonance_bounds[0], 1.0, 0.0),
        (1.01 * resonance_bounds[-1], 1.0, 0.0),
        (1.0e-2, 0.99 * speed_bounds[0], 0.0),
        (1.0e-2, 1.01 * speed_bounds[-1], 0.0),
        (1.0e-2, 1.0, float(table["pitch_cosine"][0]) - 0.01),
        (1.0e-2, 1.0, float(table["pitch_cosine"][-1]) + 0.01),
    ):
        try:
            interpolate_dimensionless_balanced_gyro_ru(
                table, resonance_value, speed_value, pitch_value
            )
        except ValueError:
            pass
        else:
            raise AssertionError("out-of-envelope runtime state was accepted")

    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "runtime_balanced_gyro.npz"
        digest, manifest = write_dimensionless_balanced_gyro_ru_table(
            output, table, context, grain_geometry
        )
        _assert(digest == sha256_file(output), "runtime SHA-256 is wrong")
        _assert(manifest.is_file(), "runtime SHA-256 sidecar is absent")
        include = Path(temporary) / "runtime_balanced_gyro.inc"
        duplicate = Path(temporary) / "duplicate.inc"
        include_digest, include_manifest = (
            write_cuda_fortran_balanced_gyro_include(
                include, table, digest
            )
        )
        duplicate_digest, _ = write_cuda_fortran_balanced_gyro_include(
            duplicate, table, digest
        )
        _assert(
            include_digest == duplicate_digest,
            "CUDA-Fortran include emission is not deterministic",
        )
        _assert(
            include_manifest.read_text(encoding="ascii")
            == f"{include_digest}  {include.name}\n",
            "CUDA-Fortran SHA-256 sidecar is wrong",
        )
        include_text = include.read_text(encoding="ascii")
        _assert(
            RUNTIME_DIMENSIONLESS_TABLE_REVISION in include_text
            and digest in include_text,
            "CUDA-Fortran include is not bound to the source table",
        )
        _assert(
            "dust_yldh_fast_supported" in include_text
            and "dust_yldh_alfven_supported" in include_text,
            "CUDA-Fortran include lacks resonance-support masks",
        )
        _assert(
            max(map(len, include_text.splitlines())) <= 132,
            "CUDA-Fortran include exceeds the free-form line limit",
        )
        literal_text = re.findall(
            r"([-+]?\d+\.\d{9}e[-+]\d+)_4", include_text
        )
        emitted = np.asarray(literal_text, dtype=np.float32)
        expected = np.concatenate(
            [
                np.asarray(table[name]).ravel(order="F")
                for name in (
                    "log_resonance_ratio",
                    "log_speed_over_alfven",
                    "pitch_cosine",
                    "drift_speed_over_alfven",
                    "drift_pitch_cosine",
                    "noise_l11",
                    "noise_l21",
                    "noise_l22",
                )
            ]
        )
        _assert(
            np.array_equal(emitted, expected),
            "CUDA-Fortran text does not round-trip every float32 value",
        )
        loaded, metadata = load_dimensionless_balanced_gyro_ru_table(
            output
        )
        for name, value in loaded.items():
            _assert(
                np.array_equal(value, table[name]),
                f"runtime {name} changed during round trip",
            )
        _assert(
            metadata["table_revision"]
            == RUNTIME_DIMENSIONLESS_TABLE_REVISION,
            "runtime table revision is absent",
        )
        _assert(
            metadata["production_default"] is False,
            "reduced selector was enabled for production",
        )
        _assert(
            metadata["excluded_physics"] == list(REDUCED_MODEL_EXCLUSIONS),
            "runtime reduced-model exclusions are incomplete",
        )
        _assert(
            metadata["wave_power_scaling"]["tensor_rule"]
            == "D=A_wave*D_basis",
            "unit wave-power basis is not declared",
        )
        _assert(
            "reject a queried cell"
            in metadata["envelope"]["support_transition_policy"],
            "runtime support-transition policy is absent",
        )
        _assert(
            "PCHIP" in metadata["coordinates"]["interpolation"],
            "runtime PCHIP policy is absent",
        )
        require_runtime_balanced_mode_signature(metadata, context)
        scaled_context = replace(
            context,
            magnetic_field_gauss=2.0 * context.magnetic_field_gauss,
            mass_density_g_cm3=3.0 * context.mass_density_g_cm3,
            injection_scale_cm=2.0 * context.injection_scale_cm,
            alfven_speed_cm_s=3.0 * context.alfven_speed_cm_s,
            fast_phase_speed_cm_s=3.0 * context.fast_phase_speed_cm_s,
            alfven_injection_velocity_cm_s=(
                3.0 * context.alfven_injection_velocity_cm_s
            ),
            fast_injection_velocity_cm_s=(
                3.0 * context.fast_injection_velocity_cm_s
            ),
            alfven_cutoff_parallel_cm_inv=(
                0.5 * context.alfven_cutoff_parallel_cm_inv
            ),
            fast_cutoff_cm_inv=0.5 * context.fast_cutoff_cm_inv,
        )
        require_runtime_balanced_mode_signature(metadata, scaled_context)
        try:
            require_runtime_balanced_mode_signature(
                metadata,
                replace(
                    context,
                    fast_cutoff_cm_inv=(
                        1.01 * context.fast_cutoff_cm_inv
                    ),
                ),
            )
        except ValueError:
            pass
        else:
            raise AssertionError("changed damping cutoff was accepted")
        stored_bytes = output.stat().st_size
        include_bytes = include.stat().st_size
    _assert(stored_bytes < 32 * 1024, "small runtime table is not compact")
    return {
        "passed": True,
        "shape": list(shape),
        "nodes": int(np.prod(shape)),
        "generation_seconds": generation_seconds,
        "seconds_per_node": generation_seconds / np.prod(shape),
        "maximum_node_float32_relative_error": (
            maximum_node_relative_error
        ),
        "maximum_midpoint_interpolation_relative_error": (
            maximum_midpoint_relative_error
        ),
        "minimum_interpolated_covariance_eigenvalue": (
            minimum_interpolated_eigenvalue
        ),
        "rejected_support_transition_cells": (
            rejected_support_transition_cells
        ),
        "accepted_zero_support_cells": accepted_zero_support_cells,
        "accepted_active_cells": len(accepted_active_cells),
        "stored_bytes": stored_bytes,
        "fortran_include_bytes": include_bytes,
        "fortran_include_sha256_verified": True,
        "sha256_verified": True,
    }


def validate_checked_default_artifacts() -> dict:
    root = Path(__file__).resolve().parent
    npz = root / "yldh04_balanced_gyro_table_f32.npz"
    include = root / "yldh04_balanced_gyro_table_f32.inc"
    audit_path = root / "yldh04_balanced_gyro_audit.json"
    table, metadata = load_dimensionless_balanced_gyro_ru_table(npz)
    for name, expected in zip(
        (
            "log_resonance_ratio",
            "log_speed_over_alfven",
            "pitch_cosine",
        ),
        default_runtime_dimensionless_axes(),
    ):
        _assert(
            np.array_equal(table[name], expected.astype(np.float32)),
            f"checked default {name} differs from the generator",
        )
    npz_digest = sha256_file(npz)
    include_digest = sha256_file(include)
    _assert(
        npz.with_suffix(".npz.sha256").read_text(encoding="ascii")
        == f"{npz_digest}  {npz.name}\n",
        "checked default NPZ manifest is stale",
    )
    _assert(
        include.with_suffix(".inc.sha256").read_text(encoding="ascii")
        == f"{include_digest}  {include.name}\n",
        "checked default include manifest is stale",
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    _assert(
        audit["artifacts"]["npz_sha256"] == npz_digest
        and audit["artifacts"]["cuda_fortran_include_sha256"]
        == include_digest,
        "checked default audit hashes are stale",
    )
    error = audit["midpoint_audit"][
        "accepted_relative_covariance_error"
    ]
    _assert(
        error["p95"] < 0.10 and error["maximum"] < 0.25,
        "checked default interpolation audit missed its error gate",
    )
    _assert(
        metadata["coordinates"]["interpolation"].startswith(
            "tensor-product local PCHIP"
        ),
        "checked default does not declare PCHIP",
    )
    return {
        "passed": True,
        "shape": list(table["noise_l11"].shape),
        "npz_bytes": npz.stat().st_size,
        "npz_sha256": npz_digest,
        "fortran_include_bytes": include.stat().st_size,
        "fortran_include_sha256": include_digest,
        "audit_p95_relative_error": error["p95"],
        "audit_maximum_relative_error": error["maximum"],
    }


def main() -> None:
    checks = {
        "gaussian_gyrofrequency": validate_gaussian_gyrofrequency(),
        "resonance_support": validate_resonance_support(),
        "balanced_alfven_relation": validate_balanced_alfven_relation(),
        "adaptive_coefficient_convergence": (
            validate_adaptive_coefficient_convergence()
        ),
        "coordinate_transform": validate_coordinate_transform(),
        "phase_space_drift": validate_phase_space_drift(),
        "psd_factor": validate_psd_factor(),
        "float32_runtime_table": validate_float32_runtime_table(),
        "complete_ru_ito_drift": validate_complete_ru_ito_drift(),
        "local_pchip_factor_interpolation": (
            validate_local_pchip_factor_interpolation()
        ),
        "matched_gyro_grain_invariance": (
            validate_matched_gyro_grain_invariance()
        ),
        "runtime_ru_table": validate_runtime_ru_table(),
        "checked_default_artifacts": validate_checked_default_artifacts(),
    }
    report = {"passed": True, "checks": checks}
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
