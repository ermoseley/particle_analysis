#!/usr/bin/env python3
"""Shared dust readers and projection helpers for particle-analysis scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from .column_utils import cic_deposit_2d, dust_pos_plane
    from .ramses_streaming import read_info_int, regular_output_files
except ImportError:
    from column_utils import cic_deposit_2d, dust_pos_plane
    from ramses_streaming import read_info_int, regular_output_files

VECTOR_FLOAT_FIELDS = frozenset({"pos", "vel", "accel", "angmom"})
INT32_BLOCK_FIELDS = frozenset({"level"})
ID_HEADER_FIELDS = frozenset({"birth_id", "id", "identity", "merging_id", "tracking_id"})


@dataclass(frozen=True)
class DustSnapshot:
    """Dust particle data extracted from one output snapshot."""

    pos: np.ndarray
    mass: np.ndarray | None
    size: np.ndarray
    particle_id: np.ndarray

    @property
    def has_mass(self) -> bool:
        """Whether the output stream contains an explicit particle mass."""
        return self.mass is not None


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
        if line.startswith("GC restart trailer") or line.startswith("id_bytes="):
            break
        fields.extend(line.split())
    if not fields:
        raise ValueError(f"No particle fields found in {header_path}")
    return fields


def read_output_ndim(output_dir: Path) -> int:
    """Read ``ndim`` from ``info.txt`` in the output directory."""
    return read_info_int(output_dir, "ndim")


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


def _particle_id_layout(
    stream_nbytes: int,
    npart: int,
    fields: list[str],
    ndim: int,
) -> tuple[int, np.dtype]:
    """Locate the primary particle-ID block in a header-described stream."""
    offset = 8 + 4 * npart * len(_expand_real_block_specs(fields, ndim))
    for name in fields[_first_int_field_index(fields) :]:
        nl = name.lower()
        if nl in INT32_BLOCK_FIELDS:
            offset += 4 * npart
        elif nl in ID_HEADER_FIELDS:
            remaining = stream_nbytes - offset
            width = remaining // npart
            if remaining != width * npart or width not in (4, 8):
                raise ValueError(
                    f"dust stream ID field {name!r}: cannot infer width "
                    f"(remaining={remaining}, npart={npart})"
                )
            if nl in ("birth_id", "id", "identity"):
                return offset, np.dtype(np.int32 if width == 4 else np.int64)
            offset += width * npart
        else:
            raise ValueError(f"Unexpected field {name!r} in integer tail of dust stream")
    raise ValueError("dust_header has no birth_id / id / identity field for grain binning")


def iter_dust_snapshot_blocks(
    run_dir: Path,
    output_num: int,
    *,
    chunk_size: int = 1_000_000,
):
    """Yield bounded, header-aware blocks from mass-bearing or massless GC output."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    output_dir = Path(run_dir) / f"output_{output_num:05d}"
    fields = read_dust_header_fields(output_dir)
    ndim = read_output_ndim(output_dir)
    real_fields = _expand_real_block_specs(fields, ndim)
    if "size" not in real_fields:
        raise ValueError(f"dust_header.txt in {output_dir} must include size")
    has_mass = "mass" in real_fields
    expected_particles = read_dust_npart_tot(output_dir)
    particles_seen = 0

    for path in regular_output_files(output_dir, "dust"):
        stream = np.memmap(path, dtype=np.uint8, mode="r")
        if stream.size < 8:
            raise ValueError(f"{path}: truncated dust header")
        file_ndim, npart = map(
            int, np.frombuffer(stream, dtype="<i4", count=2, offset=0)
        )
        if file_ndim != ndim or npart < 0:
            raise ValueError(
                f"{path}: invalid ndim/count header [{file_ndim}, {npart}]"
            )
        particles_seen += npart
        if npart == 0:
            if stream.size != 8:
                raise ValueError(f"{path}: empty dust file has trailing payload")
            del stream
            continue
        real_offsets = {
            name: 8 + index * npart * 4 for index, name in enumerate(real_fields)
        }
        id_offset, id_dtype = _particle_id_layout(
            stream.size, npart, fields, ndim
        )
        for start in range(0, npart, chunk_size):
            count = min(chunk_size, npart - start)
            pos = np.full((count, 3), np.nan, dtype=np.float64)
            for idim in range(min(ndim, 3)):
                pos[:, idim] = np.frombuffer(
                    stream,
                    dtype=np.float32,
                    count=count,
                    offset=real_offsets[f"pos_{idim}"] + 4 * start,
                )
            size = np.array(
                np.frombuffer(
                    stream,
                    dtype=np.float32,
                    count=count,
                    offset=real_offsets["size"] + 4 * start,
                ),
                dtype=np.float64,
                copy=True,
            )
            mass = None
            if has_mass:
                mass = np.array(
                    np.frombuffer(
                        stream,
                        dtype=np.float32,
                        count=count,
                        offset=real_offsets["mass"] + 4 * start,
                    ),
                    dtype=np.float64,
                    copy=True,
                )
            particle_id = np.array(
                np.frombuffer(
                    stream,
                    dtype=id_dtype,
                    count=count,
                    offset=id_offset + id_dtype.itemsize * start,
                ),
                dtype=np.int64,
                copy=True,
            )
            valid = np.isfinite(size)
            if mass is not None:
                valid &= np.isfinite(mass)
            for idim in range(min(ndim, 3)):
                valid &= np.isfinite(pos[:, idim])
            if np.any(valid):
                yield DustSnapshot(
                    pos=pos[valid],
                    mass=mass[valid] if mass is not None else None,
                    size=size[valid],
                    particle_id=particle_id[valid],
                )
        del stream
    if particles_seen != expected_particles:
        raise ValueError(
            f"{output_dir}: dust files contain {particles_seen} particles, "
            f"dust_header.txt declares {expected_particles}"
        )


def read_dust_snapshot(run_dir: Path, output_num: int) -> DustSnapshot:
    """Materialize one dust output; use the block iterator at production scale."""
    output_dir = Path(run_dir) / f"output_{output_num:05d}"
    fields = read_dust_header_fields(output_dir)
    ndim = read_output_ndim(output_dir)
    real_fields = _expand_real_block_specs(fields, ndim)

    if "size" not in real_fields:
        raise ValueError(f"dust_header.txt in {output_dir} must include size")
    has_mass = "mass" in real_fields

    pos_list: list[np.ndarray] = []
    mass_list: list[np.ndarray] = []
    size_list: list[np.ndarray] = []
    id_list: list[np.ndarray] = []

    for block in iter_dust_snapshot_blocks(run_dir, output_num):
        pos_list.append(block.pos)
        if block.mass is not None:
            mass_list.append(block.mass)
        size_list.append(block.size)
        id_list.append(block.particle_id)

    if not pos_list:
        return DustSnapshot(
            pos=np.empty((0, 3), dtype=np.float64),
            mass=np.empty((0,), dtype=np.float64) if has_mass else None,
            size=np.empty((0,), dtype=np.float64),
            particle_id=np.empty((0,), dtype=np.int64),
        )

    return DustSnapshot(
        pos=np.concatenate(pos_list, axis=0),
        mass=np.concatenate(mass_list, axis=0) if has_mass else None,
        size=np.concatenate(size_list, axis=0),
        particle_id=np.concatenate(id_list, axis=0),
    )


def valid_dust_particle_mask(
    snapshot: DustSnapshot,
    axis: str = "x",
    *,
    require_mass: bool = True,
) -> np.ndarray:
    """Mask particles valid for a dust LOS projection."""
    if snapshot.size.size == 0:
        return np.zeros((0,), dtype=bool)
    if require_mass and snapshot.mass is None:
        raise ValueError(
            "This dust output has no mass field; provide explicit reconstructed "
            "HD23 weights instead of treating the following size block as mass"
        )
    pos_xy = dust_pos_plane(snapshot.pos, axis)
    valid = (
        np.isfinite(pos_xy[:, 0])
        & np.isfinite(pos_xy[:, 1])
        & np.isfinite(snapshot.size)
        & (snapshot.size > 0.0)
    )
    if require_mass:
        valid &= np.isfinite(snapshot.mass)
    return valid


def project_weighted_dust_moments(
    snapshot: DustSnapshot,
    weights: np.ndarray,
    nx: int,
    axis: str = "x",
    box_size: float = 1.0,
    include_second_moment: bool = False,
) -> dict[str, np.ndarray]:
    """Project an explicit particle weight and its size moments with CIC."""
    weights = np.asarray(weights, dtype=np.float64).ravel()
    if weights.size != snapshot.size.size:
        raise ValueError("weights and dust snapshot length mismatch")
    valid = valid_dust_particle_mask(snapshot, axis=axis, require_mass=False)
    valid &= np.isfinite(weights)
    zero = np.zeros((nx, nx), dtype=np.float64)
    if not np.any(valid):
        out = {"sum_w": zero.copy(), "sum_wa": zero.copy()}
        if include_second_moment:
            out["sum_wa2"] = zero.copy()
        return out

    pos_xy = dust_pos_plane(snapshot.pos[valid], axis)
    weight = weights[valid]
    size = snapshot.size[valid]
    out = {
        "sum_w": cic_deposit_2d(pos_xy, weight, nx, box_size=box_size),
        "sum_wa": cic_deposit_2d(pos_xy, weight * size, nx, box_size=box_size),
    }
    if include_second_moment:
        out["sum_wa2"] = cic_deposit_2d(
            pos_xy, weight * size * size, nx, box_size=box_size
        )
    return out


def project_dust_moments(
    snapshot: DustSnapshot,
    nx: int,
    axis: str = "x",
    box_size: float = 1.0,
    include_second_moment: bool = False,
) -> dict[str, np.ndarray]:
    """Project direct dust moments onto a 2D LOS map."""
    if snapshot.mass is None:
        raise ValueError(
            "Direct mass moments require an explicit mass field; use "
            "project_weighted_dust_moments with reconstructed HD23 weights"
        )
    weighted = project_weighted_dust_moments(
        snapshot,
        snapshot.mass,
        nx,
        axis=axis,
        box_size=box_size,
        include_second_moment=include_second_moment,
    )
    out = {
        "sum_m": weighted["sum_w"],
        "sum_ma": weighted["sum_wa"],
    }
    if include_second_moment:
        out["sum_ma2"] = weighted["sum_wa2"]
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


def global_weighted_mean_size(snapshot: DustSnapshot, weights: np.ndarray) -> float:
    """Return the global mean size for an explicit particle weight."""
    weights = np.asarray(weights, dtype=np.float64).ravel()
    if weights.size != snapshot.size.size:
        raise ValueError("weights and dust snapshot length mismatch")
    valid = np.isfinite(weights) & np.isfinite(snapshot.size) & (snapshot.size > 0.0)
    if not np.any(valid):
        raise ValueError("No positive finite dust particles for mean size")
    weight = weights[valid]
    size = snapshot.size[valid]
    total_weight = float(np.sum(weight))
    if total_weight <= 0.0:
        raise ValueError("Dust total weight must be positive for mean size")
    return float(np.sum(weight * size) / total_weight)


def global_mass_weighted_mean_size(snapshot: DustSnapshot) -> float:
    """Return the global dust-mass-weighted mean grain size for a snapshot."""
    if snapshot.mass is None:
        raise ValueError(
            "This dust output has no mass field; reconstruct HD23 mass weights first"
        )
    return global_weighted_mean_size(snapshot, snapshot.mass)


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
    if snapshot.size.size == 0:
        z = np.zeros((nx, nx), dtype=np.float64)
        return z.copy(), np.full((nx, nx), np.nan, dtype=np.float64), snapshot
    if snapshot.mass is None:
        raise ValueError(
            "Legacy binned projection requires a stored mass field and is not "
            "valid for massless GC dust output"
        )

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
