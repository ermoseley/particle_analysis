#!/usr/bin/env python3
"""
Two-panel figure: log10(rho_gas / <rho_gas>) on a mid-plane density slice, and
log10(Sigma_gas / <Sigma_gas>) for the column density along the same line-of-sight
axis (perpendicular to the slice). Uses inferno and the same cube / orientation
conventions as make_column_density_video.py.

Example:
  python plot_rho_sigma_norm.py /path/to/run 42 --axis z --nx 128 -o figure.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import matplotlib

# Non-interactive backend must be set before pyplot is imported.
if "--no-display" in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

SCRIPT_DIR = Path(__file__).resolve().parent
UTILS_PY = SCRIPT_DIR / "utils" / "py"
if UTILS_PY.is_dir():
    sys.path.insert(0, str(UTILS_PY))

from column_utils import (
    _resample_cube,
    _silent_stdout,
    column_density,
    read_cube_fortran,
    save_cube_fortran,
)
from video_common import FLOOR, orient_like_column_video


def load_gas_cube(
    run_dir: Path,
    output_num: int,
    nx: int,
    cache: bool,
) -> np.ndarray:
    """Return resampled gas density cube (nx, ny, nz) in code units."""
    run_dir = Path(run_dir)
    gas_file = run_dir / f"gas_{output_num:05d}.cube"
    if cache and gas_file.exists():
        cube = read_cube_fortran(gas_file)
        if cube.shape != (nx, nx, nx):
            cube = _resample_cube(cube, nx, nx, nx)
        return cube

    import miniramses as ram

    with _silent_stdout():
        c = ram.rd_cell(output_num, path=str(run_dir) + "/")
        cube = ram.mk_cube(c.x[0], c.x[1], c.x[2], c.dx, c.u[0])
    cube = np.asarray(cube).T
    if cube.shape != (nx, nx, nx):
        cube = _resample_cube(cube, nx, nx, nx)
    if cache:
        save_cube_fortran(cube, gas_file)
    return cube


def mid_plane_rho_slice(cube: np.ndarray, axis: str) -> np.ndarray:
    """Mid-plane slice of rho perpendicular to the column integration axis."""
    ax = axis.lower()
    nx, ny, nz = cube.shape
    if ax == "z":
        return cube[:, :, nz // 2]
    if ax == "y":
        return cube[:, ny // 2, :]
    if ax == "x":
        return cube[nx // 2, :, :]
    raise ValueError("axis must be 'x', 'y', or 'z'")


def log_ratio_field(field: np.ndarray) -> np.ndarray:
    """log10(field / mean(field)) with a floor for nonpositive values."""
    m = float(np.mean(field))
    if m <= 0.0 or not np.isfinite(m):
        m = 1.0
    ratio = field / m
    return np.log10(np.maximum(ratio, FLOOR))


def percentile_vmin_vmax(
    data: np.ndarray,
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> tuple[float, float]:
    """Robust color limits from percentiles of finite values."""
    v = np.asarray(data, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return -1.0, 1.0
    vmin, vmax = np.percentile(v, [p_low, p_high])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin = float(np.min(v))
        vmax = float(np.max(v))
        if vmin >= vmax:
            vmax = vmin + 1e-9
    return float(vmin), float(vmax)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot log10(rho/<rho>) slice and log10(Sigma/<Sigma>) with inferno.",
    )
    parser.add_argument("run_dir", type=Path, help="RAMSES run directory (contains output_*)")
    parser.add_argument("nout", type=int, help="Output number (e.g. 12 for output_00012)")
    parser.add_argument("--nx", type=int, default=128, help="Cube resolution per side (default 128)")
    parser.add_argument(
        "--axis",
        type=str,
        default="z",
        choices=("x", "y", "z"),
        help="Line-of-sight for Sigma; slice is the perpendicular mid-plane (default z)",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Read/write gas_NNNNN.cube in the run directory",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: run_dir/rho_sigma_norm_NNNNN.png)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Use non-interactive backend (no GUI)",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    cube = load_gas_cube(run_dir, args.nout, args.nx, args.cache)
    axis = args.axis.lower()

    rho_slice = mid_plane_rho_slice(cube, axis)
    sigma = column_density(cube, axis=axis)

    log_rho = log_ratio_field(rho_slice)
    log_sig = log_ratio_field(sigma)

    vr = percentile_vmin_vmax(log_rho)
    vs = percentile_vmin_vmax(log_sig)

    out = args.output
    if out is None:
        out = run_dir / f"rho_sigma_norm_{args.nout:05d}.png"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
    display_rho = orient_like_column_video(log_rho)
    display_sig = orient_like_column_video(log_sig)

    im0 = axes[0].imshow(
        display_rho,
        origin="lower",
        cmap="inferno",
        norm=Normalize(vmin=vr[0], vmax=vr[1]),
        aspect="equal",
    )
    axes[0].set_title(r"$\log_{10}(\rho_{\rm gas}/\langle\rho_{\rm gas}\rangle)$ (slice)")
    if axis == "z":
        axes[0].set_xlabel("x index")
        axes[0].set_ylabel("y index")
    elif axis == "y":
        axes[0].set_xlabel("x index")
        axes[0].set_ylabel("z index")
    else:
        axes[0].set_xlabel("y index")
        axes[0].set_ylabel("z index")

    im1 = axes[1].imshow(
        display_sig,
        origin="lower",
        cmap="inferno",
        norm=Normalize(vmin=vs[0], vmax=vs[1]),
        aspect="equal",
    )
    axes[1].set_title(r"$\log_{10}(\Sigma_{\rm gas}/\langle\Sigma_{\rm gas}\rangle)$")
    if axis == "z":
        axes[1].set_xlabel("x index")
        axes[1].set_ylabel("y index")
    elif axis == "y":
        axes[1].set_xlabel("x index")
        axes[1].set_ylabel("z index")
    else:
        axes[1].set_xlabel("y index")
        axes[1].set_ylabel("z index")

    fig.colorbar(im0, ax=axes[0], shrink=0.75, label=r"$\log_{10}(\rho/\langle\rho\rangle)$")
    fig.colorbar(im1, ax=axes[1], shrink=0.75, label=r"$\log_{10}(\Sigma/\langle\Sigma\rangle)$")

    fig.savefig(out, dpi=150, bbox_inches="tight")
    if not args.no_display:
        plt.show()
    else:
        plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
