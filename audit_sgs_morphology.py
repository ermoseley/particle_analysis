#!/usr/bin/env python3
"""Audit production SGS transport and projected dust/gas morphology.

Examples
--------
python audit_sgs_morphology.py /run/output_00001 /run/output_00002 \
    --report sgs_morphology.json
python audit_sgs_morphology.py /run/output_00002 --elapsed-time 0.02 \
    --projection-axis z --report sgs_morphology.json
python audit_sgs_morphology.py --self-test --report /tmp/sgs_self_test.json

The reader is intentionally limited to the fixed-level periodic geometry used
by production.  Hydro octs and dust particles are streamed; the four 3-D
fields needed for the centered-gradient calculation are temporary float32
memmaps.  All reported distribution moments are exact streaming reductions.
Quantiles and histograms use a documented deterministic bounded sample.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import tempfile
from typing import Iterable

import numpy as np

from column_utils import dust_pos_plane
from dust_hd23 import (
    ASTRODUST_BROAD_A0,
    HD23_REVISION,
    MICRON_CM,
    astrodust_broad_dnda,
)
from dust_projection import iter_dust_snapshot_blocks, read_dust_npart_tot
from ramses_streaming import (
    iter_mesh_field_blocks,
    regular_output_files,
)


SCHEMA = "mini-ramses-sgs-morphology-v1"
QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
DEFAULT_ACTIVE_MIN_MICRON = 0.057493992527317586
DEFAULT_ACTIVE_MAX_MICRON = 0.9199038804370818


def parse_info(path: Path) -> dict[str, str]:
    """Read scalar ``key = value`` metadata."""
    values: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].split("!", 1)[0].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip().lower()] = value.strip().strip("'\"")
    return values


def metadata_float(
    metadata: dict[str, str],
    names: Iterable[str],
    *,
    required: bool = False,
) -> float | None:
    """Read the first available floating-point metadata alias."""
    for name in names:
        if name.lower() in metadata:
            token = metadata[name.lower()].replace("D", "e").replace("d", "e")
            return float(token.split()[0])
    if required:
        raise ValueError(f"Missing output metadata; expected one of {tuple(names)}")
    return None


def metadata_int(
    metadata: dict[str, str],
    name: str,
    *,
    required: bool = True,
) -> int | None:
    """Read integer-valued metadata."""
    value = metadata_float(metadata, (name,), required=required)
    return None if value is None else int(value)


def metadata_logical(
    metadata: dict[str, str],
    name: str,
    *,
    required: bool = True,
) -> bool | None:
    """Read a Fortran-style logical assignment."""
    if name.lower() not in metadata:
        if required:
            raise ValueError(f"Missing output metadata {name}")
        return None
    value = metadata[name.lower()].strip().lower()
    if value in ("t", ".true.", "true"):
        return True
    if value in ("f", ".false.", "false"):
        return False
    raise ValueError(f"Invalid logical value {name}={metadata[name.lower()]!r}")


def production_configuration(
    output_dir: Path,
    metadata: dict[str, str],
) -> dict:
    """Fail closed unless the output used the intended SGS dust walk."""
    namelist_path = output_dir / "namelist.txt"
    if not namelist_path.is_file():
        raise ValueError(f"{namelist_path}: required production namelist copy missing")
    namelist = parse_info(namelist_path)
    info_cs = metadata_float(
        metadata, ("smagorinsky_lilly_constant",), required=True
    )
    namelist_cs = metadata_float(
        namelist, ("smagorinsky_lilly_constant",), required=True
    )
    assert info_cs is not None and namelist_cs is not None
    expected_cs = 0.17
    if not (
        math.isclose(info_cs, expected_cs, rel_tol=0.0, abs_tol=1.0e-12)
        and math.isclose(namelist_cs, expected_cs, rel_tol=0.0, abs_tol=1.0e-12)
    ):
        raise ValueError(
            f"{output_dir}: expected C_s={expected_cs}, "
            f"found info={info_cs}, namelist={namelist_cs}"
        )
    info_equilibrium = metadata_logical(metadata, "equilibrium_sgs")
    namelist_equilibrium = metadata_logical(namelist, "equilibrium_sgs")
    sgs_turb = metadata_logical(namelist, "sgs_turb")
    diffusive_kicks = metadata_logical(namelist, "diffusive_kicks")
    info_scattering = metadata.get("dust_scattering_model")
    namelist_scattering = namelist.get("dust_scattering_model")
    expected_scattering = "response_markov"
    if info_equilibrium or namelist_equilibrium:
        raise ValueError(f"{output_dir}: equilibrium_sgs must be false")
    if not sgs_turb or not diffusive_kicks:
        raise ValueError(
            f"{output_dir}: the direct SGS walk requires sgs_turb and diffusive_kicks"
        )
    if (
        info_scattering != expected_scattering
        or namelist_scattering != expected_scattering
    ):
        raise ValueError(
            f"{output_dir}: expected dust_scattering_model={expected_scattering}, "
            f"found info={info_scattering!r}, namelist={namelist_scattering!r}"
        )
    return {
        "smagorinsky_lilly_constant": expected_cs,
        "equilibrium_sgs": False,
        "sgs_turb": True,
        "diffusive_kicks": True,
        "dust_scattering_model": expected_scattering,
    }


def output_location(output_dir: Path) -> tuple[Path, int]:
    """Return run directory and output number from ``output_NNNNN``."""
    output_dir = output_dir.expanduser().resolve()
    match = re.fullmatch(r"output_(\d+)", output_dir.name)
    if match is None or not output_dir.is_dir():
        raise ValueError(f"Expected an output_NNNNN directory, got {output_dir}")
    return output_dir.parent, int(match.group(1))


def descriptor_names(output_dir: Path) -> tuple[str, ...]:
    """Read the ordered regular-output primitive field descriptors."""
    lines = (output_dir / "hydro_header.txt").read_text().splitlines()
    if not lines or "=" not in lines[0]:
        raise ValueError(f"{output_dir / 'hydro_header.txt'}: missing nvar")
    nvar = int(lines[0].split("=", 1)[1])
    names: list[str | None] = [None] * nvar
    pattern = re.compile(r"variable #\s*(\d+):\s*(\S+)")
    for line in lines[1:]:
        match = pattern.fullmatch(line.strip())
        if match:
            index = int(match.group(1)) - 1
            if not 0 <= index < nvar:
                raise ValueError(f"Invalid descriptor index in {line!r}")
            names[index] = match.group(2)
    if any(name is None for name in names):
        raise ValueError(f"{output_dir / 'hydro_header.txt'}: incomplete descriptors")
    return tuple(str(name) for name in names)


def require_field(names: tuple[str, ...], aliases: tuple[str, ...]) -> tuple[int, str]:
    """Resolve one required primitive field."""
    for alias in aliases:
        if alias in names:
            return names.index(alias), alias
    raise ValueError(f"Missing field; expected one of {aliases}, found {names}")


@dataclass
class StreamingDistribution:
    """Exact moments plus a deterministic bounded sample."""

    total_size: int
    max_samples: int
    require_all_finite: bool = True

    def __post_init__(self) -> None:
        self.count = 0
        self.finite_count = 0
        self.zero_count = 0
        self.minimum = math.inf
        self.maximum = -math.inf
        self.total = 0.0
        self.total2 = 0.0
        self.stride = max(1, math.ceil(self.total_size / self.max_samples))
        self.samples: list[np.ndarray] = []

    def update(self, values: np.ndarray, offset: int) -> None:
        """Add a canonical contiguous block starting at global ``offset``."""
        flat = np.asarray(values).ravel()
        self.count += flat.size
        finite = np.isfinite(flat)
        self.finite_count += int(np.count_nonzero(finite))
        if not np.all(finite):
            flat_valid = flat[finite].astype(np.float64, copy=False)
        else:
            flat_valid = flat.astype(np.float64, copy=False)
        if flat_valid.size:
            self.minimum = min(self.minimum, float(np.min(flat_valid)))
            self.maximum = max(self.maximum, float(np.max(flat_valid)))
            self.zero_count += int(np.count_nonzero(flat_valid == 0.0))
            self.total += float(np.sum(flat_valid, dtype=np.float64))
            self.total2 += float(
                np.sum(flat_valid * flat_valid, dtype=np.float64)
            )

        first = (-offset) % self.stride
        sampled = flat[first:: self.stride]
        sampled = sampled[np.isfinite(sampled)]
        if sampled.size:
            self.samples.append(np.asarray(sampled, dtype=np.float64))

    def report(self) -> dict:
        """Return JSON-safe exact moments and sampled distribution shape."""
        if self.count != self.total_size:
            raise ValueError(
                f"Distribution consumed {self.count} values, expected {self.total_size}"
            )
        if self.require_all_finite and self.finite_count != self.count:
            raise ValueError(
                f"Distribution has {self.count - self.finite_count} non-finite values"
            )
        if self.finite_count == 0:
            raise ValueError("Distribution has no finite support")
        sample = (
            np.concatenate(self.samples)
            if self.samples
            else np.empty(0, dtype=np.float64)
        )
        if sample.size > self.max_samples + 1:
            sample = sample[: self.max_samples + 1]
        mean = self.total / self.finite_count
        variance = max(self.total2 / self.finite_count - mean * mean, 0.0)
        result = {
            "count": self.count,
            "finite_count": self.finite_count,
            "finite_fraction": self.finite_count / self.count,
            "zero_fraction_of_finite": self.zero_count / self.finite_count,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": mean,
            "rms": math.sqrt(self.total2 / self.finite_count),
            "standard_deviation": math.sqrt(variance),
            "quantile_method": "deterministic canonical-stride sample",
            "sample_stride": self.stride,
            "sample_count": int(sample.size),
            "sample_quantiles": {
                f"p{100.0 * q:g}": float(np.quantile(sample, q))
                for q in QUANTILES
            },
        }
        positive = sample[sample > 0.0]
        if positive.size:
            log_values = np.log10(positive)
            lo = float(np.min(log_values))
            hi = float(np.max(log_values))
            if hi == lo:
                edges = np.array([lo - 0.5, hi + 0.5])
            else:
                edges = np.linspace(lo, hi, 65)
            counts, edges = np.histogram(log_values, bins=edges)
            result["positive_sample_log10_histogram"] = {
                "edges": edges.tolist(),
                "counts": counts.astype(int).tolist(),
            }
        else:
            result["positive_sample_log10_histogram"] = {
                "edges": [],
                "counts": [],
            }
        return result


@dataclass
class FieldWorkspace:
    """Disk-backed fixed-grid fields and projected gas column."""

    temp: tempfile.TemporaryDirectory
    kappa: np.memmap
    bx: np.memmap
    by: np.memmap
    bz: np.memmap
    gas_column: np.ndarray
    nside: int
    dx: float
    cell_count: int
    sgs_descriptor: str

    def close(self) -> None:
        """Flush mappings and remove their temporary directory."""
        for field in (self.kappa, self.bx, self.by, self.bz):
            field.flush()
        del self.kappa, self.bx, self.by, self.bz
        self.temp.cleanup()


def create_memmap(root: Path, name: str, shape: tuple[int, ...], dtype) -> np.memmap:
    """Create a zeroed temporary memmap."""
    mapping = np.memmap(root / name, dtype=dtype, mode="w+", shape=shape)
    mapping[:] = 0
    return mapping


def load_amr_maps(
    output_dir: Path,
    *,
    level: int,
    ndim: int,
) -> tuple[dict[int, np.memmap], int]:
    """Open fixed-level AMR Cartesian oct keys as read-only memmaps."""
    maps: dict[int, np.memmap] = {}
    total_octs = 0
    for file_index, path in enumerate(
        regular_output_files(output_dir, "amr"), start=1
    ):
        header = np.fromfile(path, dtype="<i4", count=3)
        if header.size != 3:
            raise ValueError(f"{path}: truncated AMR header")
        file_ndim, levelmin, levelmax = map(int, header)
        if (file_ndim, levelmin, levelmax) != (ndim, level, level):
            raise ValueError(
                f"{path}: expected fixed ({ndim}, {level}, {level}), "
                f"found {tuple(map(int, header))}"
            )
        noct = int(np.fromfile(path, dtype="<i4", count=1, offset=12)[0])
        expected = 16 + noct * (ndim + 1) * 4
        if path.stat().st_size != expected:
            raise ValueError(f"{path}: size={path.stat().st_size}, expected={expected}")
        maps[file_index] = np.memmap(
            path,
            dtype="<i4",
            mode="r",
            offset=16,
            shape=(noct, ndim + 1),
        )
        total_octs += noct
    return maps, total_octs


def plane_flat_indices(
    ix: np.ndarray,
    iy: np.ndarray,
    iz: np.ndarray,
    *,
    axis: str,
    nside: int,
) -> np.ndarray:
    """Flatten coordinates perpendicular to the selected LOS."""
    if axis == "x":
        return iy * nside + iz
    if axis == "y":
        return ix * nside + iz
    return ix * nside + iy


def build_field_workspace(
    output_dir: Path,
    *,
    projection_axis: str,
    chunk_octs: int,
    scratch_dir: Path | None,
) -> tuple[FieldWorkspace, dict]:
    """Stream one regular fixed-grid output into four temporary fields."""
    run_dir, output_num = output_location(output_dir)
    metadata = parse_info(output_dir / "info.txt")
    ndim = metadata_int(metadata, "ndim")
    levelmin = metadata_int(metadata, "levelmin")
    levelmax = metadata_int(metadata, "levelmax")
    if ndim != 3 or levelmin != levelmax:
        raise ValueError(
            f"{output_dir}: this production audit requires ndim=3 and "
            f"levelmin=levelmax, found {ndim}, {levelmin}, {levelmax}"
        )
    level = int(levelmin)
    nside = 1 << level
    boxlen = metadata_float(metadata, ("boxlen",), required=True)
    assert boxlen is not None
    dx = boxlen / nside
    total_cells = nside**3

    names = descriptor_names(output_dir)
    rho_index, rho_name = require_field(names, ("density",))
    sgs_index, sgs_name = require_field(
        names, ("turb_kinetic_energy", "SGS_turb_energy_density")
    )
    bx_index, bx_name = require_field(names, ("magnetic_field_x",))
    by_index, by_name = require_field(names, ("magnetic_field_y",))
    bz_index, bz_name = require_field(names, ("magnetic_field_z",))

    scratch_parent = None if scratch_dir is None else str(scratch_dir)
    temp = tempfile.TemporaryDirectory(prefix="sgs_morphology_", dir=scratch_parent)
    temp_path = Path(temp.name)
    shape = (nside, nside, nside)
    kappa_grid = create_memmap(temp_path, "kappa.f32", shape, "<f4")
    bx_grid = create_memmap(temp_path, "bx.f32", shape, "<f4")
    by_grid = create_memmap(temp_path, "by.f32", shape, "<f4")
    bz_grid = create_memmap(temp_path, "bz.f32", shape, "<f4")
    coverage = create_memmap(temp_path, "coverage.u8", (total_cells,), np.uint8)
    gas_column = np.zeros((nside, nside), dtype=np.float64)

    amr_maps, total_octs = load_amr_maps(
        output_dir, level=level, ndim=ndim
    )
    expected_octs = total_cells // 8
    if total_octs != expected_octs:
        raise ValueError(
            f"{output_dir}: {total_octs} octs, expected {expected_octs}"
        )

    bit = np.arange(8, dtype=np.int64)
    bits = (
        (bit & 1),
        ((bit >> 1) & 1),
        ((bit >> 2) & 1),
    )
    cells_seen = 0
    for block in iter_mesh_field_blocks(
        run_dir,
        output_num,
        "hydro",
        chunk_octs=chunk_octs,
        expected_nvar=len(names),
    ):
        if block.level != level:
            raise ValueError(f"{output_dir}: unexpected hydro level {block.level}")
        nblock = block.values.shape[0]
        amr = amr_maps[block.file_index][
            block.oct_start : block.oct_start + nblock
        ]
        if amr.shape[0] != nblock:
            raise ValueError(f"{output_dir}: AMR/hydro block length mismatch")
        if np.any(amr[:, 3] != 0):
            raise ValueError(f"{output_dir}: refined cells in fixed-level output")
        keys = np.asarray(amr[:, :3], dtype=np.int64)
        oct_linear = (keys[:, 0] * (nside // 2) + keys[:, 1]) * (
            nside // 2
        ) + keys[:, 2]
        if np.unique(oct_linear).size != oct_linear.size:
            raise ValueError(f"{output_dir}: duplicate oct key within one block")

        ix = (2 * keys[:, 0, None] + bits[0][None, :]).reshape(-1)
        iy = (2 * keys[:, 1, None] + bits[1][None, :]).reshape(-1)
        iz = (2 * keys[:, 2, None] + bits[2][None, :]).reshape(-1)
        if (
            np.any(ix < 0)
            or np.any(ix >= nside)
            or np.any(iy < 0)
            or np.any(iy >= nside)
            or np.any(iz < 0)
            or np.any(iz >= nside)
        ):
            raise ValueError(f"{output_dir}: AMR key lies outside the periodic box")
        linear = (ix * nside + iy) * nside + iz
        if np.any(coverage[linear] != 0):
            raise ValueError(f"{output_dir}: duplicate cell coverage")

        rho = np.asarray(
            block.values[:, rho_index, :], dtype=np.float64
        ).reshape(-1)
        sgs = np.asarray(
            block.values[:, sgs_index, :], dtype=np.float64
        ).reshape(-1)
        bx = np.asarray(block.values[:, bx_index, :]).reshape(-1)
        by = np.asarray(block.values[:, by_index, :]).reshape(-1)
        bz = np.asarray(block.values[:, bz_index, :]).reshape(-1)
        if (
            np.any(~np.isfinite(rho))
            or np.any(rho <= 0.0)
            or np.any(~np.isfinite(sgs))
            or np.any(sgs < 0.0)
            or np.any(~np.isfinite(bx))
            or np.any(~np.isfinite(by))
            or np.any(~np.isfinite(bz))
        ):
            raise ValueError(f"{output_dir}: invalid density, SGS, or magnetic field")

        if sgs_name == "turb_kinetic_energy":
            e_sgs_density = rho * sgs
        else:
            e_sgs_density = sgs
        kappa = dx * np.sqrt((2.0 / 3.0) * e_sgs_density / rho)
        kappa_grid[ix, iy, iz] = kappa.astype(np.float32)
        bx_grid[ix, iy, iz] = bx
        by_grid[ix, iy, iz] = by
        bz_grid[ix, iy, iz] = bz
        coverage[linear] = 1
        plane = plane_flat_indices(
            ix, iy, iz, axis=projection_axis, nside=nside
        )
        gas_column.ravel()[:] += np.bincount(
            plane,
            weights=rho * dx,
            minlength=nside * nside,
        )
        cells_seen += linear.size

    if cells_seen != total_cells or int(np.sum(coverage, dtype=np.int64)) != total_cells:
        raise ValueError(
            f"{output_dir}: covered {cells_seen} cells, expected {total_cells}"
        )
    for mapping in amr_maps.values():
        del mapping
    coverage.flush()
    del coverage
    for field in (kappa_grid, bx_grid, by_grid, bz_grid):
        field.flush()

    workspace = FieldWorkspace(
        temp=temp,
        kappa=kappa_grid,
        bx=bx_grid,
        by=by_grid,
        bz=bz_grid,
        gas_column=gas_column,
        nside=nside,
        dx=dx,
        cell_count=total_cells,
        sgs_descriptor=sgs_name,
    )
    layout = {
        "ndim": ndim,
        "level": level,
        "nside": nside,
        "boxlen_code": boxlen,
        "dx_code": dx,
        "cell_count": total_cells,
        "oct_count": total_octs,
        "nfile": metadata_int(metadata, "nfile"),
        "primitive_fields": list(names),
        "resolved_fields": {
            "density": rho_name,
            "sgs": sgs_name,
            "magnetic": [bx_name, by_name, bz_name],
        },
        "scratch_peak_bytes_estimate": 17 * total_cells,
        "scratch_note": (
            "Four float32 3-D fields plus one temporary byte-per-cell "
            "coverage map; use node-local --scratch-dir at 512^3."
        ),
    }
    return workspace, layout


def centered_difference(
    field: np.memmap,
    x_indices: np.ndarray,
    *,
    derivative_axis: int,
    dx: float,
) -> np.ndarray:
    """Centered periodic derivative for one x slab."""
    nside = field.shape[0]
    if derivative_axis == 0:
        plus = np.asarray(field[(x_indices + 1) % nside], dtype=np.float64)
        minus = np.asarray(field[(x_indices - 1) % nside], dtype=np.float64)
        return (plus - minus) / (2.0 * dx)
    current = np.asarray(field[x_indices], dtype=np.float64)
    plus = np.roll(current, -1, axis=derivative_axis)
    minus = np.roll(current, 1, axis=derivative_axis)
    return (plus - minus) / (2.0 * dx)


def transport_diagnostics(
    workspace: FieldWorkspace,
    *,
    elapsed_time: float,
    projection_axis: str,
    slab_planes: int,
    max_samples: int,
    b_floor: float,
) -> dict:
    """Measure local transport scales and cell-centred drift reconstructions."""
    nside = workspace.nside
    total = workspace.cell_count
    dx = workspace.dx
    distributions = {
        "kappa_code": StreamingDistribution(total, max_samples),
        "diffusion_1d_cells": StreamingDistribution(total, max_samples),
        "diffusion_magnetized_2d_rms_cells": StreamingDistribution(
            total, max_samples
        ),
        "diffusion_full_orbit_3d_rms_cells": StreamingDistribution(
            total, max_samples
        ),
        "isotropic_ito_drift_speed_code": StreamingDistribution(
            total, max_samples
        ),
        "isotropic_ito_drift_displacement_cells": StreamingDistribution(
            total, max_samples
        ),
        "projected_ito_drift_speed_code": StreamingDistribution(
            total, max_samples, require_all_finite=False
        ),
        "projected_ito_drift_displacement_cells": StreamingDistribution(
            total, max_samples, require_all_finite=False
        ),
    }
    drift_los_sum = np.zeros((nside, nside), dtype=np.float64)
    magnetic_valid = 0
    offset = 0

    for start in range(0, nside, slab_planes):
        stop = min(start + slab_planes, nside)
        xs = np.arange(start, stop, dtype=np.int64)
        kappa = np.asarray(workspace.kappa[xs], dtype=np.float64)
        ell1 = np.sqrt(2.0 * kappa * elapsed_time) / dx
        ell2 = np.sqrt(4.0 * kappa * elapsed_time) / dx
        ell3 = np.sqrt(6.0 * kappa * elapsed_time) / dx
        distributions["kappa_code"].update(kappa, offset)
        distributions["diffusion_1d_cells"].update(ell1, offset)
        distributions["diffusion_magnetized_2d_rms_cells"].update(ell2, offset)
        distributions["diffusion_full_orbit_3d_rms_cells"].update(ell3, offset)

        bx = np.asarray(workspace.bx[xs], dtype=np.float64)
        by = np.asarray(workspace.by[xs], dtype=np.float64)
        bz = np.asarray(workspace.bz[xs], dtype=np.float64)
        bmag = np.sqrt(bx * bx + by * by + bz * bz)
        valid = np.isfinite(bmag) & (bmag > b_floor)
        magnetic_valid += int(np.count_nonzero(valid))
        safe = np.where(valid, bmag, 1.0)
        bhat = (bx / safe, by / safe, bz / safe)

        grad_kappa = tuple(
            centered_difference(
                workspace.kappa, xs, derivative_axis=axis, dx=dx
            )
            for axis in range(3)
        )
        isotropic_drift = np.sqrt(
            sum(component * component for component in grad_kappa)
        )
        distributions["isotropic_ito_drift_speed_code"].update(
            isotropic_drift, offset
        )
        distributions["isotropic_ito_drift_displacement_cells"].update(
            isotropic_drift * elapsed_time / dx, offset
        )
        grad_bmag = tuple(
            centered_difference_magnitude(
                workspace.bx,
                workspace.by,
                workspace.bz,
                xs,
                derivative_axis=axis,
                dx=dx,
            )
            for axis in range(3)
        )
        grad_bhat = [
            [
                centered_difference_normalized_vector(
                    workspace.bx,
                    workspace.by,
                    workspace.bz,
                    xs,
                    component=i,
                    derivative_axis=j,
                    dx=dx,
                    b_floor=b_floor,
                )
                for j in range(3)
            ]
            for i in range(3)
        ]

        # Match the mover: project each discrete d_j(bhat) onto the tangent
        # plane before forming curvature.
        for j in range(3):
            radial = sum(bhat[i] * grad_bhat[i][j] for i in range(3))
            for i in range(3):
                grad_bhat[i][j] -= radial * bhat[i]
        curvature = tuple(
            sum(grad_bhat[i][j] * bhat[j] for j in range(3))
            for i in range(3)
        )
        grad_kappa_dot_b = sum(grad_kappa[i] * bhat[i] for i in range(3))
        bhat_grad_bmag = sum(grad_bmag[i] * bhat[i] for i in range(3))
        drift = tuple(
            grad_kappa[i]
            - grad_kappa_dot_b * bhat[i]
            - kappa
            * (
                curvature[i]
                - bhat[i] * bhat_grad_bmag / safe
            )
            for i in range(3)
        )
        drift_mag = np.sqrt(sum(component * component for component in drift))
        drift_mag = np.where(valid, drift_mag, np.nan)
        distributions["projected_ito_drift_speed_code"].update(
            drift_mag, offset
        )
        distributions["projected_ito_drift_displacement_cells"].update(
            drift_mag * elapsed_time / dx, offset
        )

        projected = np.where(valid, drift_mag, 0.0)
        if projection_axis == "x":
            drift_los_sum += np.sum(projected, axis=0)
        elif projection_axis == "y":
            drift_los_sum[start:stop, :] += np.sum(projected, axis=1)
        else:
            drift_los_sum[start:stop, :] += np.sum(projected, axis=2)
        offset += kappa.size

    reports = {name: value.report() for name, value in distributions.items()}
    drift_los_mean = drift_los_sum / nside
    reports["projected_ito_drift_speed_code"][
        "magnetic_valid_count"
    ] = magnetic_valid
    reports["projected_ito_drift_speed_code"][
        "magnetic_valid_fraction"
    ] = magnetic_valid / total
    reports["projected_ito_drift_speed_code"][
        "null_cells_are_excluded_from_distribution"
    ] = True
    reports["projected_ito_drift_speed_code"][
        "reconstruction_note"
    ] = (
        "Cell-centred centered-periodic reconstruction of the magnetized "
        "continuum expression; the mover evaluates CIC/TSC derivatives at "
        "particle positions."
    )
    reports["projected_los_mean_drift_speed_code"] = scalar_map_summary(
        drift_los_mean
    )
    return reports


def centered_difference_magnitude(
    bx: np.memmap,
    by: np.memmap,
    bz: np.memmap,
    x_indices: np.ndarray,
    *,
    derivative_axis: int,
    dx: float,
) -> np.ndarray:
    """Centered periodic derivative of cell-centered ``|B|``."""
    nside = bx.shape[0]

    def magnitude(indices: np.ndarray) -> np.ndarray:
        x = np.asarray(bx[indices], dtype=np.float64)
        y = np.asarray(by[indices], dtype=np.float64)
        z = np.asarray(bz[indices], dtype=np.float64)
        return np.sqrt(x * x + y * y + z * z)

    if derivative_axis == 0:
        plus = magnitude((x_indices + 1) % nside)
        minus = magnitude((x_indices - 1) % nside)
    else:
        current = magnitude(x_indices)
        plus = np.roll(current, -1, axis=derivative_axis)
        minus = np.roll(current, 1, axis=derivative_axis)
    return (plus - minus) / (2.0 * dx)


def centered_difference_normalized_vector(
    bx: np.memmap,
    by: np.memmap,
    bz: np.memmap,
    x_indices: np.ndarray,
    *,
    component: int,
    derivative_axis: int,
    dx: float,
    b_floor: float,
) -> np.ndarray:
    """Centered periodic derivative of one cell-normalized B component."""
    fields = (bx, by, bz)
    nside = bx.shape[0]

    def normalized(indices: np.ndarray) -> np.ndarray:
        x = np.asarray(bx[indices], dtype=np.float64)
        y = np.asarray(by[indices], dtype=np.float64)
        z = np.asarray(bz[indices], dtype=np.float64)
        magnitude = np.sqrt(x * x + y * y + z * z)
        valid = magnitude > b_floor
        return np.where(
            valid,
            np.asarray(fields[component][indices], dtype=np.float64)
            / np.where(valid, magnitude, 1.0),
            0.0,
        )

    if derivative_axis == 0:
        plus = normalized((x_indices + 1) % nside)
        minus = normalized((x_indices - 1) % nside)
    else:
        current = normalized(x_indices)
        plus = np.roll(current, -1, axis=derivative_axis)
        minus = np.roll(current, 1, axis=derivative_axis)
    return (plus - minus) / (2.0 * dx)


def scalar_map_summary(values: np.ndarray) -> dict:
    """Compact exact summary for a 2-D map."""
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return {"count": int(values.size), "finite_count": 0}
    valid = values[finite]
    return {
        "count": int(values.size),
        "finite_count": int(valid.size),
        "minimum": float(np.min(valid)),
        "maximum": float(np.max(valid)),
        "mean": float(np.mean(valid)),
        "rms": float(np.sqrt(np.mean(valid * valid))),
    }


def fast_cic_deposit_2d(
    grid: np.ndarray,
    positions: np.ndarray,
    weights: np.ndarray,
    *,
    box_size: float,
) -> None:
    """Periodic cell-centered CIC deposit using bounded ``bincount`` passes."""
    nside = grid.shape[0]
    uv = np.mod(np.asarray(positions, dtype=np.float64), box_size)
    scaled = uv * (nside / box_size) - 0.5
    lower = np.floor(scaled).astype(np.int64)
    frac = scaled - lower
    lower %= nside
    upper = (lower + 1) % nside
    flat = grid.ravel()
    weights = np.asarray(weights, dtype=np.float64)
    for du in range(2):
        iu = upper[:, 0] if du else lower[:, 0]
        wu = frac[:, 0] if du else 1.0 - frac[:, 0]
        for dv in range(2):
            iv = upper[:, 1] if dv else lower[:, 1]
            wv = frac[:, 1] if dv else 1.0 - frac[:, 1]
            flat += np.bincount(
                iu * nside + iv,
                weights=weights * wu * wv,
                minlength=nside * nside,
            )


def dust_weight_configuration(metadata: dict[str, str]) -> dict:
    """Resolve the exact relative/absolute HD23 macro-mass conversion."""
    revision = metadata.get("dust_hd23_revision")
    if revision != HD23_REVISION:
        raise ValueError(
            f"dust_hd23_revision={revision!r}, expected {HD23_REVISION!r}"
        )
    grain_density = metadata_float(
        metadata,
        ("dust_grain_bulk_density", "grain_bulk_density"),
        required=True,
    )
    assert grain_density is not None
    radius_factor = metadata_float(
        metadata,
        ("dust_radius_code_to_cm", "hd23_radius_code_to_cm"),
        required=True,
    )
    assert radius_factor is not None
    active_min = metadata_float(
        metadata,
        (
            "dust_active_min_micron",
            "dust_active_a_min_micron",
            "hd23_active_min_micron",
        ),
        required=True,
    )
    active_max = metadata_float(
        metadata,
        (
            "dust_active_max_micron",
            "dust_active_a_max_micron",
            "hd23_active_max_micron",
        ),
        required=True,
    )
    assert active_min is not None and active_max is not None
    macro_norm = metadata_float(
        metadata, ("dust_macro_mass_norm",), required=True
    )
    assert macro_norm is not None
    active_min_code = active_min * MICRON_CM / radius_factor
    active_max_code = active_max * MICRON_CM / radius_factor
    active_min_stored = float(
        np.nextafter(np.float32(active_min_code), np.float32(-np.inf))
    )
    active_max_stored = float(
        np.nextafter(np.float32(active_max_code), np.float32(np.inf))
    )
    return {
        "hd23_revision": revision,
        "grain_density_g_cm3": grain_density,
        "radius_code_to_cm": radius_factor,
        "active_min_cm": active_min * MICRON_CM,
        "active_max_cm": active_max * MICRON_CM,
        "active_min_stored_code": active_min_stored,
        "active_max_stored_code": active_max_stored,
        "macro_mass_norm": macro_norm,
    }


def project_dust(
    output_dir: Path,
    *,
    nside: int,
    projection_axis: str,
    particle_chunk: int,
    box_size: float,
) -> tuple[np.ndarray, dict]:
    """Stream the production dust output into a common 2-D mass map."""
    run_dir, output_num = output_location(output_dir)
    metadata = parse_info(output_dir / "info.txt")
    config = dust_weight_configuration(metadata)
    grid = np.zeros((nside, nside), dtype=np.float64)
    particle_count = 0
    active_count = 0
    source_sum = 0.0
    stored_mass: bool | None = None
    declared_particle_count = read_dust_npart_tot(output_dir)

    for block in iter_dust_snapshot_blocks(
        run_dir, output_num, chunk_size=particle_chunk
    ):
        block_has_mass = block.mass is not None
        if stored_mass is None:
            stored_mass = block_has_mass
        elif stored_mass != block_has_mass:
            raise ValueError(f"{output_dir}: inconsistent dust mass fields")
        if block.mass is not None:
            weights = np.asarray(block.mass, dtype=np.float64)
            active = np.ones(weights.size, dtype=bool)
            weighting = "stored particle mass"
        else:
            radius = block.size * config["radius_code_to_cm"]
            active = (
                np.isfinite(radius)
                & (block.size >= config["active_min_stored_code"])
                & (block.size <= config["active_max_stored_code"])
            )
            relative = (
                astrodust_broad_dnda(radius) * radius / ASTRODUST_BROAD_A0
            )
            weights = config["macro_mass_norm"] * relative * radius**3
            weighting = "output-metadata HD23 macro mass"
            weights = np.where(active, weights, 0.0)
        valid = active & np.isfinite(weights) & (weights >= 0.0)
        particle_count += weights.size
        active_count += int(np.count_nonzero(valid))
        source_sum += float(np.sum(weights[valid], dtype=np.float64))
        if np.any(valid):
            fast_cic_deposit_2d(
                grid,
                dust_pos_plane(block.pos[valid], projection_axis),
                weights[valid],
                box_size=box_size,
            )

    if particle_count != declared_particle_count:
        raise ValueError(
            f"{output_dir}: streamed {particle_count} finite particles, "
            f"dust_header.txt declares {declared_particle_count}"
        )
    if active_count != particle_count:
        raise ValueError(
            f"{output_dir}: {particle_count - active_count} production grains "
            "lie outside the declared active HD23 interval or have invalid weights"
        )
    if particle_count == 0 or source_sum <= 0.0:
        raise ValueError(f"{output_dir}: no positive active dust weight")
    if stored_mass:
        raise ValueError(
            f"{output_dir}: production GC dust unexpectedly stores particle mass"
        )
    deposited_sum = float(np.sum(grid, dtype=np.float64))
    conservation = abs(deposited_sum / source_sum - 1.0)
    report = {
        "particle_count": particle_count,
        "declared_particle_count": declared_particle_count,
        "active_particle_count": active_count,
        "stored_mass_field": bool(stored_mass),
        "weighting": weighting,
        "dust_component": (
            "kinetic active broad astrodust only; excludes the passive broad, "
            "small-astrodust, and PAH reservoirs"
        ),
        "hd23_revision": config["hd23_revision"],
        "grain_density_g_cm3": config["grain_density_g_cm3"],
        "macro_mass_norm": config["macro_mass_norm"],
        "source_weight_sum": source_sum,
        "deposited_weight_sum": deposited_sum,
        "cic_conservation_relative_error": conservation,
        "radius_code_to_cm": config["radius_code_to_cm"],
        "active_min_micron": config["active_min_cm"] / MICRON_CM,
        "active_max_micron": config["active_max_cm"] / MICRON_CM,
    }
    return grid, report


def morphology_diagnostics(
    gas_column: np.ndarray,
    dust_column: np.ndarray,
    *,
    box_size: float,
    kmax_fraction: float,
) -> dict:
    """Common-grid log-contrast correlation and periodic 2-D spectra."""
    gas = np.asarray(gas_column, dtype=np.float64)
    dust = np.asarray(dust_column, dtype=np.float64)
    if gas.shape != dust.shape or gas.ndim != 2 or gas.shape[0] != gas.shape[1]:
        raise ValueError("Gas and dust projections must be matching square maps")
    valid = (
        np.isfinite(gas)
        & np.isfinite(dust)
        & (gas > 0.0)
        & (dust > 0.0)
    )
    if np.count_nonzero(valid) < 3:
        raise ValueError("Too few jointly positive pixels for morphology")
    gas_contrast = np.full_like(gas, np.nan)
    dust_contrast = np.full_like(dust, np.nan)
    gas_contrast[valid] = np.log10(gas[valid] / np.mean(gas[valid]))
    dust_contrast[valid] = np.log10(dust[valid] / np.mean(dust[valid]))
    g = gas_contrast[valid]
    d = dust_contrast[valid]
    g_std = float(np.std(g))
    d_std = float(np.std(d))
    pearson = (
        None
        if g_std == 0.0 or d_std == 0.0
        else float(np.mean((g - np.mean(g)) * (d - np.mean(d))) / (g_std * d_std))
    )
    pixel = {
        "joint_positive_count": int(g.size),
        "joint_positive_fraction": float(g.size / gas.size),
        "pearson_log_contrast": pearson,
        "gas_log10_contrast_std": g_std,
        "dust_log10_contrast_std": d_std,
        "rms_log10_dust_minus_gas": float(np.sqrt(np.mean((d - g) ** 2))),
    }

    # Spectra require complete periodic maps; a sparse projection is a failed
    # estimand, not something to fill with an arbitrary floor.
    if not np.all(valid):
        return {
            "pixel": pixel,
            "spectrum": None,
            "spectrum_reason": "projection contains nonpositive or nonfinite pixels",
        }
    gmap = gas_contrast - np.mean(gas_contrast)
    dmap = dust_contrast - np.mean(dust_contrast)
    fg = np.fft.rfft2(gmap)
    fd = np.fft.rfft2(dmap)
    nside = gas.shape[0]
    kx = np.fft.fftfreq(nside) * nside
    ky = np.fft.rfftfreq(nside) * nside
    kr = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
    hermitian_weight = np.full(ky.size, 2.0)
    hermitian_weight[0] = 1.0
    if nside % 2 == 0:
        hermitian_weight[-1] = 1.0
    weight = np.broadcast_to(hermitian_weight[None, :], kr.shape)
    kmax = max(1, int(math.floor(kmax_fraction * nside)))
    bins: list[dict] = []
    for k in range(1, kmax + 1):
        mask = (kr >= k - 0.5) & (kr < k + 0.5)
        if not np.any(mask):
            continue
        mode_weight = weight[mask]
        norm = float(np.sum(mode_weight))
        pgg = float(np.sum(mode_weight * np.abs(fg[mask]) ** 2) / norm)
        pdd = float(np.sum(mode_weight * np.abs(fd[mask]) ** 2) / norm)
        pdg = float(
            np.sum(mode_weight * np.real(fd[mask] * np.conj(fg[mask]))) / norm
        )
        correlation = None if pgg <= 0.0 or pdd <= 0.0 else pdg / math.sqrt(pgg * pdd)
        transfer = None if pgg <= 0.0 else pdg / pgg
        amplitude = None if pgg <= 0.0 else math.sqrt(max(pdd, 0.0) / pgg)
        bins.append(
            {
                "k_box_cycles": k,
                "wavelength_cells": nside / k,
                "unique_rfft_modes": int(np.count_nonzero(mask)),
                "hermitian_mode_weight": norm,
                "gas_power": pgg,
                "dust_power": pdd,
                "cross_power": pdg,
                "cross_correlation": correlation,
                "regression_transfer_dust_from_gas": transfer,
                "amplitude_ratio_sqrt_Pdd_over_Pgg": amplitude,
            }
        )
    return {
        "pixel": pixel,
        "spectrum": {
            "projection_is_periodic": True,
            "mean_subtracted_log10_contrasts": True,
            "box_size_code": box_size,
            "k_bin_width_box_cycles": 1.0,
            "kmax_fraction_of_grid": kmax_fraction,
            "kmax_box_cycles": kmax,
            "bins": bins,
        },
    }


def elapsed_time_for_outputs(
    outputs: list[Path],
    override: float | None,
) -> tuple[float, str, list[float]]:
    """Resolve a supplied interval or derive it from output metadata."""
    times: list[float] = []
    for output in outputs:
        metadata = parse_info(output / "info.txt")
        time_value = metadata_float(metadata, ("time",), required=True)
        assert time_value is not None
        times.append(time_value)
    if override is not None:
        if override <= 0.0:
            raise ValueError("--elapsed-time must be positive")
        return override, "command_line", times
    if len(outputs) == 2:
        elapsed = times[1] - times[0]
        source = "output_metadata_difference"
    else:
        elapsed = times[0]
        source = "single_output_metadata_time_since_zero"
    if elapsed <= 0.0:
        raise ValueError(
            "Metadata-derived elapsed time is not positive; pass --elapsed-time"
        )
    return elapsed, source, times


def compare_reports(first: dict, second: dict) -> dict:
    """Describe endpoint change without declaring acceptance."""
    contract_keys = (
        "nside",
        "boxlen_code",
        "primitive_fields",
    )
    for key in contract_keys:
        if first["layout"][key] != second["layout"][key]:
            raise ValueError(f"Output comparison changed layout field {key}")
    if first["configuration"] != second["configuration"]:
        raise ValueError("Output comparison changed the production SGS contract")
    dust_keys = (
        "hd23_revision",
        "radius_code_to_cm",
        "active_min_micron",
        "active_max_micron",
        "grain_density_g_cm3",
        "macro_mass_norm",
        "declared_particle_count",
    )
    for key in dust_keys:
        if first["dust_projection"][key] != second["dust_projection"][key]:
            raise ValueError(f"Output comparison changed HD23 field {key}")
    p0 = first["morphology"]["pixel"]["pearson_log_contrast"]
    p1 = second["morphology"]["pixel"]["pearson_log_contrast"]
    spectral: list[dict] = []
    s0 = first["morphology"].get("spectrum")
    s1 = second["morphology"].get("spectrum")
    if s0 is not None and s1 is not None:
        by_k = {row["k_box_cycles"]: row for row in s0["bins"]}
        for final in s1["bins"]:
            initial = by_k.get(final["k_box_cycles"])
            if initial is None:
                continue
            a0 = initial["amplitude_ratio_sqrt_Pdd_over_Pgg"]
            a1 = final["amplitude_ratio_sqrt_Pdd_over_Pgg"]
            r0 = initial["cross_correlation"]
            r1 = final["cross_correlation"]
            spectral.append(
                {
                    "k_box_cycles": final["k_box_cycles"],
                    "wavelength_cells": final["wavelength_cells"],
                    "amplitude_ratio_final_over_initial": (
                        None if a0 in (None, 0.0) or a1 is None else a1 / a0
                    ),
                    "cross_correlation_change": (
                        None if r0 is None or r1 is None else r1 - r0
                    ),
                }
            )
    return {
        "pearson_log_contrast_change": (
            None if p0 is None or p1 is None else p1 - p0
        ),
        "spectral_change": spectral,
        "acceptance_status": None,
        "note": (
            "Descriptive endpoint change; it does not isolate SGS transport "
            "without a matched kappa-off control, and no pass threshold is imposed."
        ),
    }


def analyze_output(
    output_dir: Path,
    *,
    elapsed_time: float,
    metadata_time: float,
    args: argparse.Namespace,
) -> dict:
    """Run the complete bounded audit for one output."""
    metadata = parse_info(output_dir / "info.txt")
    configuration = production_configuration(output_dir, metadata)
    workspace, layout = build_field_workspace(
        output_dir,
        projection_axis=args.projection_axis,
        chunk_octs=args.chunk_octs,
        scratch_dir=args.scratch_dir,
    )
    try:
        transport = transport_diagnostics(
            workspace,
            elapsed_time=elapsed_time,
            projection_axis=args.projection_axis,
            slab_planes=args.slab_planes,
            max_samples=args.max_samples,
            b_floor=args.b_floor,
        )
        dust_column, dust_report = project_dust(
            output_dir,
            nside=workspace.nside,
            projection_axis=args.projection_axis,
            particle_chunk=args.particle_chunk,
            box_size=layout["boxlen_code"],
        )
        morphology = morphology_diagnostics(
            workspace.gas_column,
            dust_column,
            box_size=layout["boxlen_code"],
            kmax_fraction=args.kmax_fraction,
        )
        unit_l = metadata_float(metadata, ("unit_l",))
        unit_t = metadata_float(metadata, ("unit_t",))
        if unit_l is not None and unit_t is not None:
            transport["unit_conversion"] = {
                "kappa_code_to_cm2_s": unit_l * unit_l / unit_t,
                "drift_speed_code_to_cm_s": unit_l / unit_t,
            }
        return {
            "output": str(output_dir.resolve()),
            "metadata_time_code": metadata_time,
            "configuration": configuration,
            "layout": layout,
            "transport": transport,
            "gas_projection": scalar_map_summary(workspace.gas_column),
            "dust_projection": dust_report,
            "morphology": morphology,
        }
    finally:
        workspace.close()


def run_audit(outputs: list[Path], args: argparse.Namespace) -> dict:
    """Analyze one or two endpoint outputs."""
    if len(outputs) not in (1, 2):
        raise ValueError("Provide one or two output_NNNNN directories")
    outputs = [path.expanduser().resolve() for path in outputs]
    elapsed, elapsed_source, times = elapsed_time_for_outputs(
        outputs, args.elapsed_time
    )
    reports = [
        analyze_output(
            output,
            elapsed_time=elapsed,
            metadata_time=metadata_time,
            args=args,
        )
        for output, metadata_time in zip(outputs, times)
    ]
    result = {
        "schema": SCHEMA,
        "estimand": {
            "kappa_sgs": "dx*sqrt((2/3)*E_sgs_density/rho)",
            "primitive_sgs_note": (
                "Regular output turb_kinetic_energy is E_sgs_density/rho; "
                "the audit reconstructs E_sgs_density before applying the formula."
            ),
            "diffusion_1d_cells": "sqrt(2*kappa_sgs*elapsed_time)/dx",
            "diffusion_magnetized_2d_rms_cells": (
                "sqrt(4*kappa_sgs*elapsed_time)/dx"
            ),
            "diffusion_full_orbit_3d_rms_cells": (
                "sqrt(6*kappa_sgs*elapsed_time)/dx"
            ),
            "isotropic_full_orbit_ito_drift": "grad(kappa)",
            "magnetized_projected_ito_drift": (
                "grad(kappa)-bhat*(bhat.grad(kappa))"
                "-kappa*((bhat.grad)bhat-bhat*(bhat.grad|B|)/|B|)"
            ),
            "gradient": "second-order centered periodic cell-centered difference",
            "gradient_scope": (
                "Eulerian reconstruction of the same continuum expressions; "
                "not the particle-local CIC/TSC gather"
            ),
            "morphology_fields": (
                "log10 of each projected field divided by its own positive-pixel mean"
            ),
            "projection_scope": (
                "Whole-box 2-D projection; it may hide 3-D washout and the dust "
                "map contains only kinetic active broad astrodust."
            ),
            "spectral_transfer": "P_dg/P_gg on periodic 2-D projected maps",
            "spectral_correlation": "P_dg/sqrt(P_dd*P_gg)",
        },
        "analysis_domain": {
            "geometry": "fixed-level periodic",
            "projection_axis": args.projection_axis,
            "elapsed_time_code": elapsed,
            "elapsed_time_source": elapsed_source,
            "transport_time_assumption": (
                "Each output's local coefficient is held fixed over the stated "
                "interval; these are endpoint transport scales, not an integral "
                "of each particle's time-varying kappa."
            ),
            "magnetic_valid_threshold_code": f"|B| > {args.b_floor:g}",
            "resolved_spectral_kmax_fraction": args.kmax_fraction,
            "distribution_max_samples": args.max_samples,
            "acceptance_threshold": None,
            "acceptance_note": (
                "This is a descriptive audit. Morphology changes are not "
                "causally attributable to SGS transport without a matched "
                "kappa-off control; choose a baseline before assigning pass/fail."
            ),
        },
        "outputs": reports,
    }
    if len(reports) == 2:
        result["comparison"] = compare_reports(reports[0], reports[1])
    return result


def write_synthetic_output(
    root: Path,
    output_num: int,
    *,
    nside: int,
    time_value: float,
    phase: float,
) -> dict:
    """Write one small fixed-grid regular output for ``--self-test``."""
    output = root / f"output_{output_num:05d}"
    output.mkdir(parents=True)
    level = int(round(math.log2(nside)))
    if 1 << level != nside:
        raise ValueError("Synthetic nside must be a power of two")
    dx = 1.0 / nside
    noct_side = nside // 2
    keys = np.array(
        [
            (i, j, k)
            for i in range(noct_side)
            for j in range(noct_side)
            for k in range(noct_side)
        ],
        dtype="<i4",
    )
    noct = keys.shape[0]
    bit = np.arange(8)
    ix = (2 * keys[:, 0, None] + (bit & 1)[None, :]).reshape(-1)
    iy = (2 * keys[:, 1, None] + ((bit >> 1) & 1)[None, :]).reshape(-1)
    iz = (2 * keys[:, 2, None] + ((bit >> 2) & 1)[None, :]).reshape(-1)
    x = (ix + 0.5) * dx
    y = (iy + 0.5) * dx
    z = (iz + 0.5) * dx
    rho = (
        1.0
        + 0.16 * np.sin(2.0 * np.pi * (x + phase))
        + 0.08 * np.cos(2.0 * np.pi * y)
        + 0.04 * np.sin(2.0 * np.pi * z)
    )
    kappa = 0.02 * (1.0 + 0.25 * np.sin(2.0 * np.pi * x))
    sgs_specific = 1.5 * (kappa / dx) ** 2
    values = np.zeros((noct, 10, 8), dtype="<f4")
    values[:, 0, :] = rho.reshape(noct, 8)
    values[:, 4, :] = 0.1
    values[:, 7, :] = 1.0
    values[:, 8, :] = sgs_specific.reshape(noct, 8)
    values[:, 9, :] = 0.1

    info = (
        "ncpu       =          1\n"
        "nfile      =          1\n"
        "ndim       =          3\n"
        f"levelmin   =          {level}\n"
        f"levelmax   =          {level}\n"
        "boxlen     = 1.000000000000000E+00\n"
        f"time       = {time_value:.15E}\n"
        "unit_l     = 3.085677581491367E+18\n"
        "unit_t     = 3.155760000000000E+13\n"
        "unit_d     = 1.000000000000000E+00\n"
        "equilibrium_sgs =F\n"
        "smagorinsky_lilly_constant =1.700000000000000E-01\n"
        "dust_scattering_model =response_markov\n"
        "dust_hd23_revision =HD23-2023-eq18-eq25-v1\n"
        "dust_grain_bulk_density =2.000000000000000E+00\n"
        "dust_radius_code_to_cm =1.000000000000000E+00\n"
        f"dust_active_min_micron ={DEFAULT_ACTIVE_MIN_MICRON:.15E}\n"
        f"dust_active_max_micron ={DEFAULT_ACTIVE_MAX_MICRON:.15E}\n"
        "dust_macro_mass_norm =1.000000000000000E+40\n"
    )
    (output / "info.txt").write_text(info)
    (output / "namelist.txt").write_text(
        "&HYDRO_PARAMS\n"
        "  sgs_turb=.true.\n"
        "  equilibrium_sgs=.false.\n"
        "  smagorinsky_lilly_constant=0.17d0\n"
        "/\n"
        "&DUST_PARAMS\n"
        "  dust_scattering_model='response_markov'\n"
        "  diffusive_kicks=.true.\n"
        "/\n"
    )
    names = (
        "density",
        "velocity_x",
        "velocity_y",
        "velocity_z",
        "thermal_pressure",
        "magnetic_field_x",
        "magnetic_field_y",
        "magnetic_field_z",
        "turb_kinetic_energy",
        "H2_fraction",
    )
    hydro_header = ["nvar        =         10"]
    hydro_header.extend(
        f"variable #{index:2d}: {name}" for index, name in enumerate(names, 1)
    )
    (output / "hydro_header.txt").write_text("\n".join(hydro_header) + "\n")

    with (output / "amr.00001").open("wb") as handle:
        np.array([3, level, level, noct], dtype="<i4").tofile(handle)
        amr_payload = np.column_stack(
            (keys, np.zeros(noct, dtype="<i4"))
        ).astype("<i4")
        amr_payload.tofile(handle)
    with (output / "hydro.00001").open("wb") as handle:
        np.array([3, 10, level, level, noct], dtype="<i4").tofile(handle)
        values.tofile(handle)

    positions = np.column_stack((x, y, z)).astype("<f4")
    npart = positions.shape[0]
    radius_micron = 0.20 + 0.025 * (
        np.sin(2.0 * np.pi * (x + phase))
        + 0.5 * np.cos(2.0 * np.pi * y)
        + 0.25 * np.sin(2.0 * np.pi * z)
    )
    sizes = (radius_micron * MICRON_CM).astype("<f4")
    zeros = np.zeros(npart, dtype="<f4")
    ones = np.ones(npart, dtype="<f4")
    fields = "pos vel size charge vpara mu_adb gc_mode birth_id"
    dust_header = (
        " Total number of particles\n"
        f" {npart}\n"
        " Total number of files\n"
        " 1\n"
        " Particle fields\n"
        f"{fields}\n"
        "id_bytes=8\n"
    )
    (output / "dust_header.txt").write_text(dust_header)
    with (output / "dust.00001").open("wb") as handle:
        np.array([3, npart], dtype="<i4").tofile(handle)
        for component in range(3):
            positions[:, component].tofile(handle)
        for _ in range(3):
            zeros.tofile(handle)
        sizes.tofile(handle)
        zeros.tofile(handle)
        zeros.tofile(handle)
        zeros.tofile(handle)
        ones.tofile(handle)
        np.arange(1, npart + 1, dtype="<i8").tofile(handle)

    stored_kappa = dx * np.sqrt(
        (2.0 / 3.0) * values[:, 8, :].astype(np.float64)
    )
    canonical = np.empty((nside, nside, nside), dtype=np.float64)
    canonical[ix, iy, iz] = stored_kappa.reshape(-1)
    expected_grad = (
        np.roll(canonical, -1, axis=0)
        - np.roll(canonical, 1, axis=0)
    ) / (2.0 * dx)
    return {
        "output": output,
        "kappa_min": float(np.min(canonical)),
        "kappa_max": float(np.max(canonical)),
        "drift_max": float(np.max(np.abs(expected_grad))),
    }


def run_self_test(args: argparse.Namespace) -> dict:
    """Create and validate a disposable two-output fixture under ``/tmp``."""
    root = Path(tempfile.mkdtemp(prefix="sgs_morphology_fixture_", dir="/tmp"))
    first = write_synthetic_output(
        root, 1, nside=8, time_value=0.0, phase=0.0
    )
    second = write_synthetic_output(
        root, 2, nside=8, time_value=0.125, phase=0.125
    )
    result = run_audit([first["output"], second["output"]], args)
    measured = result["outputs"][0]["transport"]
    checks = {
        "metadata_elapsed": (
            result["analysis_domain"]["elapsed_time_source"]
            == "output_metadata_difference"
            and abs(result["analysis_domain"]["elapsed_time_code"] - 0.125) < 1.0e-15
        ),
        "cell_coverage": result["outputs"][0]["layout"]["cell_count"] == 8**3,
        "kappa_min": math.isclose(
            measured["kappa_code"]["minimum"],
            first["kappa_min"],
            rel_tol=2.0e-7,
            abs_tol=1.0e-12,
        ),
        "kappa_max": math.isclose(
            measured["kappa_code"]["maximum"],
            first["kappa_max"],
            rel_tol=2.0e-7,
            abs_tol=1.0e-12,
        ),
        "centered_projected_drift": math.isclose(
            measured["projected_ito_drift_speed_code"]["maximum"],
            first["drift_max"],
            rel_tol=2.0e-6,
            abs_tol=1.0e-12,
        ),
        "dust_cic_conservation": all(
            report["dust_projection"]["cic_conservation_relative_error"] < 2.0e-13
            for report in result["outputs"]
        ),
        "morphology_is_finite": all(
            report["morphology"]["pixel"]["pearson_log_contrast"] is not None
            and math.isfinite(
                report["morphology"]["pixel"]["pearson_log_contrast"]
            )
            for report in result["outputs"]
        ),
        "resolved_spectrum_present": all(
            report["morphology"]["spectrum"] is not None
            and len(report["morphology"]["spectrum"]["bins"]) > 0
            for report in result["outputs"]
        ),
    }
    result["self_test"] = {
        "fixture_root": str(root),
        "checks": checks,
        "passed": all(checks.values()),
    }
    if not result["self_test"]["passed"]:
        raise RuntimeError(f"Synthetic self-test failed: {checks}")
    return result


def arguments() -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        description=(
            "Bounded fixed-grid SGS transport and projected dust/gas morphology audit"
        )
    )
    parser.add_argument("outputs", nargs="*", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--elapsed-time", type=float)
    parser.add_argument("--projection-axis", choices=("x", "y", "z"), default="z")
    parser.add_argument("--chunk-octs", type=int, default=65_536)
    parser.add_argument("--particle-chunk", type=int, default=1_000_000)
    parser.add_argument("--slab-planes", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=1_000_000)
    parser.add_argument(
        "--b-floor",
        type=float,
        default=1.0e-22,
        help="magnetic support floor (production smallc*1e-10 = 1e-22)",
    )
    parser.add_argument("--kmax-fraction", type=float, default=0.25)
    parser.add_argument("--scratch-dir", type=Path)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="write and audit a disposable two-output fixture under /tmp",
    )
    args = parser.parse_args()
    if args.self_test and args.outputs:
        parser.error("--self-test does not accept output paths")
    if not args.self_test and len(args.outputs) not in (1, 2):
        parser.error("provide one or two output_NNNNN directories")
    if args.chunk_octs <= 0 or args.particle_chunk <= 0 or args.slab_planes <= 0:
        parser.error("chunk sizes and slab planes must be positive")
    if args.max_samples < 100:
        parser.error("--max-samples must be at least 100")
    if args.b_floor < 0.0:
        parser.error("--b-floor must be nonnegative")
    if not 0.0 < args.kmax_fraction <= 0.5:
        parser.error("--kmax-fraction must lie in (0, 0.5]")
    if args.scratch_dir is not None:
        args.scratch_dir = args.scratch_dir.expanduser().resolve()
        if not args.scratch_dir.is_dir():
            parser.error("--scratch-dir must already exist")
    return args


def main() -> None:
    """CLI entry point."""
    args = arguments()
    result = run_self_test(args) if args.self_test else run_audit(args.outputs, args)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report is None:
        print(payload, end="")
    else:
        args.report.expanduser().resolve().write_text(payload)
        print(f"Wrote {args.report.expanduser().resolve()}")
    if args.self_test:
        print(
            "Synthetic self-test PASS: "
            f"{result['self_test']['fixture_root']}"
        )


if __name__ == "__main__":
    main()
