#!/usr/bin/env python3
"""Focused checks for the offline YLDH reference and float32 table."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from dust_yldh_reference import (
    DEFAULT_INTEGRATION_ATOL,
    DEFAULT_INTEGRATION_LIMIT,
    DEFAULT_INTEGRATION_RTOL,
    Grain,
    INTEGRATION_METHOD,
    PrimitiveDiffusion,
    _alfven_branch_coefficients,
    _conservative_primitive_drift,
    build_runtime_table,
    grain_gyrofrequency_s,
    primitive_diffusion,
    psd_lower_factor,
    transform_diffusion_to_w_mu,
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
    }
    report = {"passed": True, "checks": checks}
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
