#!/usr/bin/env python3
"""Shared gas and dust column-density helpers for particle-analysis scripts."""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np

# Self-contained: use miniramses from this repo only (no sibling mini-ramses-dev).
SCRIPT_DIR = Path(__file__).resolve().parent
UTILS_PY = SCRIPT_DIR / "utils" / "py"
if UTILS_PY.is_dir():
    sys.path.insert(0, str(UTILS_PY))

# Optional scipy for resampling.
try:
    from scipy.ndimage import zoom
except ImportError:
    zoom = None


def cic_deposit_2d(
    pos_xy: np.ndarray,
    masses: np.ndarray,
    nx: int,
    box_size: float = 1.0,
) -> np.ndarray:
    """Cloud-in-cell deposit of particles onto a 2D grid."""
    pos_xy = np.asarray(pos_xy, dtype=np.float64)
    if pos_xy.ndim != 2 or pos_xy.shape[1] != 2:
        raise ValueError("pos_xy must have shape (N, 2)")
    masses = np.asarray(masses, dtype=np.float64).ravel()
    if pos_xy.shape[0] != masses.shape[0]:
        raise ValueError("pos_xy and masses length mismatch")

    dx = box_size / nx
    grid = np.zeros((nx, nx), dtype=np.float64)

    fx = pos_xy[:, 0] / dx - 0.5
    fy = pos_xy[:, 1] / dx - 0.5

    ix0 = np.floor(fx).astype(np.int64) % nx
    iy0 = np.floor(fy).astype(np.int64) % nx
    ix1 = (ix0 + 1) % nx
    iy1 = (iy0 + 1) % nx

    wx1 = (fx - np.floor(fx)).astype(np.float64)
    wy1 = (fy - np.floor(fy)).astype(np.float64)
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1

    for di in range(2):
        for dj in range(2):
            ixs = ix1 if di else ix0
            iys = iy1 if dj else iy0
            w = masses * (wx1 if di else wx0) * (wy1 if dj else wy0)
            np.add.at(grid, (ixs, iys), w)

    return grid


def _resample_cube(cube: np.ndarray, target_nx: int, target_ny: int, target_nz: int) -> np.ndarray:
    """Resample 3D array to target shape via linear interpolation."""
    if zoom is None:
        raise RuntimeError("scipy.ndimage.zoom required for gas resampling")
    cx, cy, cz = cube.shape
    factors = (target_nx / cx, target_ny / cy, target_nz / cz)
    return zoom(cube.astype(np.float64), factors, order=1).astype(np.float64)


def column_density(cube: np.ndarray, axis: str = "z") -> np.ndarray:
    """Integrate a 3D cube along the requested line of sight."""
    ax = {"x": 0, "y": 1, "z": 2}[axis.lower()]
    n = cube.shape[ax]
    return np.sum(cube, axis=ax) * (1.0 / n)


def read_cube_fortran(cube_file: Path) -> np.ndarray:
    """Read a simple Fortran cube file produced by these analysis tools."""
    with open(cube_file, "rb") as f:
        _ = np.fromfile(f, dtype=np.int32, count=1)[0]
        nx, ny, nz = np.fromfile(f, dtype=np.int32, count=3)
        _ = np.fromfile(f, dtype=np.int32, count=1)[0]
        n_cells = nx * ny * nz
        cube = np.fromfile(f, dtype=np.float32, count=n_cells)
        _ = np.fromfile(f, dtype=np.int32, count=1)[0]
    return cube.reshape((nx, ny, nz), order="F").astype(np.float64)


def save_cube_fortran(cube: np.ndarray, output_file: Path) -> None:
    """Write a Fortran-format cube file compatible with existing scripts."""
    nx, ny, nz = cube.shape
    with open(output_file, "wb") as f:
        np.array([3 * 4], dtype=np.int32).tofile(f)
        np.array([nx, ny, nz], dtype=np.int32).tofile(f)
        np.array([3 * 4], dtype=np.int32).tofile(f)
        flat = cube.astype(np.float32).flatten(order="F")
        np.array([flat.nbytes], dtype=np.int32).tofile(f)
        flat.tofile(f)
        np.array([flat.nbytes], dtype=np.int32).tofile(f)


@contextlib.contextmanager
def _silent_stdout():
    """Temporarily redirect stdout to suppress miniramses/mk_cube prints."""
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old


def get_gas_column(
    run_dir: Path,
    output_num: int,
    nx: int,
    cache: bool = False,
    axis: str = "z",
) -> np.ndarray:
    """Load gas density cube, resample to ``nx^3``, and return a column density map."""
    run_dir = Path(run_dir)
    gas_file = run_dir / f"gas_{output_num:05d}.cube"
    if cache and gas_file.exists():
        cube = read_cube_fortran(gas_file)
        if cube.shape != (nx, nx, nx):
            cube = _resample_cube(cube, nx, nx, nx)
        return column_density(cube, axis=axis)

    import miniramses as ram

    with _silent_stdout():
        c = ram.rd_cell(output_num, path=str(run_dir) + "/")
        cube = ram.mk_cube(c.x[0], c.x[1], c.x[2], c.dx, c.u[0])
    cube = np.asarray(cube).T
    cx, cy, cz = cube.shape
    if (cx, cy, cz) != (nx, nx, nx):
        cube = _resample_cube(cube, nx, nx, nx)
    if cache:
        save_cube_fortran(cube, gas_file)
    return column_density(cube, axis=axis)


def dust_pos_plane(pos: np.ndarray, axis: str) -> np.ndarray:
    """Select coordinates perpendicular to the line-of-sight ``axis``."""
    ax = axis.lower()
    if ax == "z":
        return pos[:, [0, 1]]
    if ax == "y":
        return pos[:, [0, 2]]
    if ax == "x":
        return pos[:, [1, 2]]
    raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis!r}")


# Backward-compatible alias for older analysis scripts.
_dust_pos_plane = dust_pos_plane


def get_dust_column(
    run_dir: Path,
    output_num: int,
    nx: int,
    box_size: float = 1.0,
    axis: str = "z",
) -> np.ndarray:
    """Load dust particles and deposit onto a 2D grid with CIC; return column mass map."""
    import miniramses as ram

    p = ram.rd_part(output_num, path=str(run_dir) + "/", prefix="dust", silent=True)
    if p.npart == 0:
        return np.zeros((nx, nx), dtype=np.float64)

    pos = np.column_stack([p.pos[i] for i in range(p.pos.shape[0])])
    masses = np.asarray(p.mass, dtype=np.float64).ravel()
    pos_xy = dust_pos_plane(pos, axis)
    valid = np.isfinite(pos_xy[:, 0]) & np.isfinite(pos_xy[:, 1])
    if not np.any(valid):
        return np.zeros((nx, nx), dtype=np.float64)
    return cic_deposit_2d(pos_xy[valid], masses[valid], nx, box_size=box_size)
