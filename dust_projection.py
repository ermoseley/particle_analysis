#!/usr/bin/env python3
"""Shared dust readers and projection helpers for particle-analysis scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from .column_utils import cic_deposit_2d, dust_pos_plane
except ImportError:
    from column_utils import cic_deposit_2d, dust_pos_plane

VECTOR_FLOAT_FIELDS = frozenset({"pos", "vel", "accel", "angmom"})
INT32_BLOCK_FIELDS = frozenset({"level"})
ID_HEADER_FIELDS = frozenset({"birth_id", "id", "identity", "merging_id", "tracking_id"})


@dataclass(frozen=True)
class DustSnapshot:
    """Dust particle data extracted from one output snapshot."""

    pos: np.ndarray
    mass: np.ndarray
    size: np.ndarray
    particle_id: np.ndarray


def read_dust_header_fields(output_dir: Path) -> list[str]:
    """Return ordered field names from ``dust_header.txt``."""
    header_path = output_dir / "dust_header.txt"
    fields: list[str] = []
    after_particle_fields = False
    for raw in header_path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("Particle fields"):
            after_particle_fields = True
            continue
        if not after_particle_fields:
            continue
        fields.extend(line.split())
    if not fields:
        raise ValueError(f"No particle fields found in {header_path}")
    return fields


def read_output_ndim(output_dir: Path) -> int:
    """Read ``ndim`` from ``info.txt`` in the output directory."""
    info_path = output_dir / "info.txt"
    for raw in info_path.read_text().splitlines():
        if raw.strip().startswith("ndim"):
            _, value = raw.split("=", 1)
            return int(value)
    raise ValueError(f"Could not read ndim from {info_path}")


def read_dust_npart_tot(output_dir: Path) -> int:
    """Read the total dust particle count from ``dust_header.txt``."""
    header_path = output_dir / "dust_header.txt"
    lines = [ln.strip() for ln in header_path.read_text().splitlines()]
    for i, line in enumerate(lines):
        if line.startswith("Total number of particles"):
            return int(lines[i + 1].split()[0])
    raise ValueError(f"Could not read total particle count from {header_path}")


def _expand_real_block_specs(fields: list[str], ndim: int) -> list[str]:
    """Expand header field names into their ordered float32 stream blocks."""
    reals: list[str] = []
    for name in fields:
        if name in INT32_BLOCK_FIELDS or name in ID_HEADER_FIELDS:
            continue
        if name in VECTOR_FLOAT_FIELDS:
            for idim in range(ndim):
                reals.append(f"{name}_{idim}")
        else:
            reals.append(name)
    return reals


def _first_int_field_index(fields: list[str]) -> int:
    """Return the index of the first integer field in header order."""
    for i, name in enumerate(fields):
        nl = name.lower()
        if nl in INT32_BLOCK_FIELDS or nl in ID_HEADER_FIELDS:
            return i
    return len(fields)


def _extract_particle_id(mm: bytes, npart: int, fields: list[str], ndim: int) -> np.ndarray:
    """Read particle identity from the integer tail of a dust stream."""
    real_fields = _expand_real_block_specs(fields, ndim)
    offset = 8 + 4 * npart * len(real_fields)
    for name in fields[_first_int_field_index(fields) :]:
        nl = name.lower()
        if nl in INT32_BLOCK_FIELDS:
            offset += 4 * npart
        elif nl in ID_HEADER_FIELDS:
            rem = len(mm) - offset
            if npart <= 0:
                raise ValueError("npart must be positive")
            id_bytes = rem // npart
            if rem != id_bytes * npart or id_bytes not in (4, 8):
                raise ValueError(
                    f"dust stream ID field {name!r}: cannot infer width (rem={rem}, npart={npart})"
                )
            if id_bytes == 4:
                arr = np.frombuffer(mm, dtype=np.int32, count=npart, offset=offset).astype(np.int64)
            else:
                arr = np.frombuffer(mm, dtype=np.int64, count=npart, offset=offset)
            if nl in ("birth_id", "id", "identity"):
                return arr
            offset += id_bytes * npart
        else:
            raise ValueError(f"Unexpected field {name!r} in integer tail of dust stream")
    raise ValueError("dust_header has no birth_id / id / identity field for grain binning")


def read_dust_snapshot(run_dir: Path, output_num: int) -> DustSnapshot:
    """Read dust particle positions, masses, sizes, and IDs for one output."""
    output_dir = Path(run_dir) / f"output_{output_num:05d}"
    fields = read_dust_header_fields(output_dir)
    ndim = read_output_ndim(output_dir)
    real_fields = _expand_real_block_specs(fields, ndim)

    if "mass" not in real_fields or "size" not in real_fields:
        raise ValueError(f"dust_header.txt in {output_dir} must include mass and size")

    dust_files = sorted(output_dir.glob("dust.*"))
    if not dust_files:
        return DustSnapshot(
            pos=np.empty((0, 3), dtype=np.float64),
            mass=np.empty((0,), dtype=np.float64),
            size=np.empty((0,), dtype=np.float64),
            particle_id=np.empty((0,), dtype=np.int64),
        )

    pos_list: list[np.ndarray] = []
    mass_list: list[np.ndarray] = []
    size_list: list[np.ndarray] = []
    id_list: list[np.ndarray] = []

    for path in dust_files:
        mm = path.read_bytes()
        if len(mm) < 8:
            continue
        npart = int(np.frombuffer(mm, dtype=np.int32, count=1, offset=4)[0])
        if npart <= 0:
            continue

        stride = npart * 4
        offset = 8
        real_data: dict[str, np.ndarray] = {}
        for name in real_fields:
            real_data[name] = np.frombuffer(mm, dtype=np.float32, count=npart, offset=offset).astype(np.float64)
            offset += stride

        pos = np.full((npart, 3), np.nan, dtype=np.float64)
        for idim in range(min(ndim, 3)):
            pos[:, idim] = real_data[f"pos_{idim}"]
        mass = real_data["mass"]
        size = real_data["size"]
        particle_id = _extract_particle_id(mm, npart, fields, ndim)

        valid = np.isfinite(mass) & np.isfinite(size)
        if ndim >= 1:
            valid &= np.isfinite(pos[:, 0])
        if ndim >= 2:
            valid &= np.isfinite(pos[:, 1])
        if ndim >= 3:
            valid &= np.isfinite(pos[:, 2])
        if not np.any(valid):
            continue

        pos_list.append(pos[valid])
        mass_list.append(mass[valid])
        size_list.append(size[valid])
        id_list.append(particle_id[valid])

    if not pos_list:
        return DustSnapshot(
            pos=np.empty((0, 3), dtype=np.float64),
            mass=np.empty((0,), dtype=np.float64),
            size=np.empty((0,), dtype=np.float64),
            particle_id=np.empty((0,), dtype=np.int64),
        )

    return DustSnapshot(
        pos=np.concatenate(pos_list, axis=0),
        mass=np.concatenate(mass_list, axis=0),
        size=np.concatenate(size_list, axis=0),
        particle_id=np.concatenate(id_list, axis=0),
    )


def valid_dust_particle_mask(snapshot: DustSnapshot, axis: str = "x") -> np.ndarray:
    """Mask particles valid for a dust LOS projection."""
    if snapshot.mass.size == 0:
        return np.zeros((0,), dtype=bool)
    pos_xy = dust_pos_plane(snapshot.pos, axis)
    return (
        np.isfinite(pos_xy[:, 0])
        & np.isfinite(pos_xy[:, 1])
        & np.isfinite(snapshot.mass)
        & np.isfinite(snapshot.size)
        & (snapshot.size > 0.0)
    )


def project_dust_moments(
    snapshot: DustSnapshot,
    nx: int,
    axis: str = "x",
    box_size: float = 1.0,
    include_second_moment: bool = False,
) -> dict[str, np.ndarray]:
    """Project direct dust moments onto a 2D LOS map."""
    valid = valid_dust_particle_mask(snapshot, axis=axis)
    zero = np.zeros((nx, nx), dtype=np.float64)
    if not np.any(valid):
        out = {"sum_m": zero.copy(), "sum_ma": zero.copy()}
        if include_second_moment:
            out["sum_ma2"] = zero.copy()
        return out

    pos_xy = dust_pos_plane(snapshot.pos[valid], axis)
    mass = snapshot.mass[valid]
    size = snapshot.size[valid]

    out = {
        "sum_m": cic_deposit_2d(pos_xy, mass, nx, box_size=box_size),
        "sum_ma": cic_deposit_2d(pos_xy, mass * size, nx, box_size=box_size),
    }
    if include_second_moment:
        out["sum_ma2"] = cic_deposit_2d(pos_xy, mass * size * size, nx, box_size=box_size)
    return out


def mean_size_from_moments(sum_m: np.ndarray, sum_ma: np.ndarray) -> np.ndarray:
    """Return the mass-weighted mean size from projected moments."""
    mean = np.full_like(sum_m, np.nan, dtype=np.float64)
    np.divide(sum_ma, sum_m, out=mean, where=sum_m > 0.0)
    return mean


def std_size_from_moments(sum_m: np.ndarray, sum_ma: np.ndarray, sum_ma2: np.ndarray) -> np.ndarray:
    """Return the mass-weighted LOS standard deviation of size."""
    mean = mean_size_from_moments(sum_m, sum_ma)
    second = np.full_like(sum_m, np.nan, dtype=np.float64)
    np.divide(sum_ma2, sum_m, out=second, where=sum_m > 0.0)
    var = second - mean * mean
    var = np.where(np.isfinite(var), np.maximum(var, 0.0), np.nan)
    return np.sqrt(var)


def global_mass_weighted_mean_size(snapshot: DustSnapshot) -> float:
    """Return the global dust-mass-weighted mean grain size for a snapshot."""
    valid = np.isfinite(snapshot.mass) & np.isfinite(snapshot.size) & (snapshot.size > 0.0)
    if not np.any(valid):
        raise ValueError("No positive finite dust particles for mean size")
    mass = snapshot.mass[valid]
    size = snapshot.size[valid]
    total_mass = float(np.sum(mass))
    if total_mass <= 0.0:
        raise ValueError("Dust total mass must be positive for mean size")
    return float(np.sum(mass * size) / total_mass)


def median_size_per_bin(sizes: np.ndarray, bin_idx: np.ndarray, n_bins: int) -> np.ndarray:
    """Median physical size in each bin index (NaN if bin empty)."""
    med = np.full(n_bins, np.nan, dtype=np.float64)
    for b in range(n_bins):
        m = (bin_idx == b) & np.isfinite(sizes) & (sizes > 0.0)
        if np.any(m):
            med[b] = float(np.median(sizes[m]))
    return med


def logsize_bin_edges_from_sizes(sizes: np.ndarray, n_bins: int) -> np.ndarray:
    """Return ``n_bins+1`` log-spaced edges spanning ``sizes``."""
    s = np.asarray(sizes, dtype=np.float64)
    s = s[np.isfinite(s) & (s > 0.0)]
    if s.size == 0:
        raise ValueError("No positive dust grain sizes to build log-size bin edges")
    lo = float(np.min(s))
    hi = float(np.max(s))
    if hi <= lo * (1.0 + 1e-12):
        lo = lo * 0.99
        hi = hi * 1.01
    return np.logspace(np.log10(lo), np.log10(hi), n_bins + 1)


def bin_idx_from_logsize(sizes: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Assign each particle to ``[0, n_bins-1]`` from log-size edges."""
    s = np.asarray(sizes, dtype=np.float64)
    e = np.asarray(edges, dtype=np.float64)
    n_bins = e.size - 1
    idx = np.searchsorted(e, s, side="right") - 1
    return np.clip(idx, 0, n_bins - 1)


def legacy_binned_mean_size_map(
    run_dir: Path,
    output_num: int,
    nx: int,
    axis: str,
    box_size: float,
    grain_bins: str,
    n_bins: int,
    logsize_edges: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, DustSnapshot]:
    """Reproduce the current binned-median LOS effective size map."""
    snapshot = read_dust_snapshot(run_dir, output_num)
    if snapshot.mass.size == 0:
        z = np.zeros((nx, nx), dtype=np.float64)
        return z.copy(), np.full((nx, nx), np.nan, dtype=np.float64), snapshot

    if grain_bins == "identity":
        output_dir = Path(run_dir) / f"output_{output_num:05d}"
        npart_tot = read_dust_npart_tot(output_dir)
        particles_per_bin = max(1, npart_tot // n_bins)
        bin_idx = np.zeros_like(snapshot.particle_id, dtype=np.int64)
        ok_id = snapshot.particle_id > 0
        bin_idx[ok_id] = np.clip(
            (snapshot.particle_id[ok_id] - 1) // particles_per_bin,
            0,
            n_bins - 1,
        )
        bin_idx[~ok_id] = -1
    elif grain_bins == "logsize":
        if logsize_edges is None:
            raise ValueError("logsize_edges is required when grain_bins='logsize'")
        bin_idx = np.full_like(snapshot.particle_id, -1, dtype=np.int64)
        ok_id = snapshot.particle_id > 0
        bin_idx[ok_id] = bin_idx_from_logsize(snapshot.size[ok_id], logsize_edges)
    else:
        raise ValueError(f"Unknown grain_bins: {grain_bins!r}")

    med = median_size_per_bin(snapshot.size, bin_idx, n_bins)
    safe_bin = np.clip(bin_idx, 0, n_bins - 1)
    median_at_particle = med[safe_bin]
    median_at_particle[~ok_id] = np.nan

    valid = valid_dust_particle_mask(snapshot, axis=axis) & ok_id & np.isfinite(median_at_particle)
    if not np.any(valid):
        z = np.zeros((nx, nx), dtype=np.float64)
        return z.copy(), np.full((nx, nx), np.nan, dtype=np.float64), snapshot

    pos_xy = dust_pos_plane(snapshot.pos[valid], axis)
    mass = snapshot.mass[valid]
    eff_size = median_at_particle[valid]
    sum_m = cic_deposit_2d(pos_xy, mass, nx, box_size=box_size)
    sum_weighted = cic_deposit_2d(pos_xy, mass * eff_size, nx, box_size=box_size)
    mean_size = mean_size_from_moments(sum_m, sum_weighted)
    return sum_m, mean_size, snapshot
