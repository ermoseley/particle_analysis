#!/usr/bin/env python3
"""
Fast column-density video for guiding-center dust runs.

Reads mini-ramses outputs, builds 128x128 gas and dust column-density maps
(gas via 3D cube + sum; dust via 2D CIC from particles), renders frames with
fixed log colorbar, then encodes to video with ffmpeg.

Usage (on cluster):
  python make_column_density_video.py --run-dir /path/to/run --start 1 --end 152
"""
from __future__ import annotations

import argparse
import contextlib
import io
import subprocess
import sys
from pathlib import Path

import numpy as np

# Agg before pyplot for headless/speed
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm

# Self-contained: use miniramses from this repo only (no sibling mini-ramses-dev).
SCRIPT_DIR = Path(__file__).resolve().parent
UTILS_PY = SCRIPT_DIR / "utils" / "py"
if UTILS_PY.is_dir():
    sys.path.insert(0, str(UTILS_PY))

# Optional scipy for resampling
try:
    from scipy.ndimage import zoom
except ImportError:
    zoom = None

FLOOR = 1e-30  # small floor for log10(column density)


def cic_deposit_2d(
    positions: np.ndarray,
    masses: np.ndarray,
    nx: int,
    box_size: float = 1.0,
) -> np.ndarray:
    """Cloud-in-cell deposit of particles onto a 2D grid (x, y only).

    positions: (N, 2) or (N, 3) in [0, box_size]; only first two columns used.
    Returns mass per cell, shape (nx, nx).
    """
    pos_xy = np.asarray(positions[:, :2], dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64).ravel()
    if pos_xy.shape[0] != masses.shape[0]:
        raise ValueError("positions and masses length mismatch")
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
    ax = {"x": 0, "y": 1, "z": 2}[axis.lower()]
    n = cube.shape[ax]
    return np.sum(cube, axis=ax) * (1.0 / n)


def read_cube_fortran(cube_file: Path) -> np.ndarray:
    with open(cube_file, "rb") as f:
        _ = np.fromfile(f, dtype=np.int32, count=1)[0]
        nx, ny, nz = np.fromfile(f, dtype=np.int32, count=3)
        _ = np.fromfile(f, dtype=np.int32, count=1)[0]
        n_cells = nx * ny * nz
        cube = np.fromfile(f, dtype=np.float32, count=n_cells)
        _ = np.fromfile(f, dtype=np.int32, count=1)[0]
    return cube.reshape((nx, ny, nz), order="F").astype(np.float64)


def save_cube_fortran(cube: np.ndarray, output_file: Path) -> None:
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
) -> np.ndarray:
    """Load gas density cube, resample to nx^3, return column density along z."""
    run_dir = Path(run_dir)
    gas_file = run_dir / f"gas_{output_num:05d}.cube"
    if cache and gas_file.exists():
        cube = read_cube_fortran(gas_file)
        if cube.shape != (nx, nx, nx):
            cube = _resample_cube(cube, nx, nx, nx)
        return column_density(cube, axis="z")

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
    return column_density(cube, axis="z")


def get_dust_column(run_dir: Path, output_num: int, nx: int, box_size: float = 1.0) -> np.ndarray:
    """Load dust particles and deposit onto 2D grid with CIC; return column mass map."""
    import miniramses as ram
    p = ram.rd_part(output_num, path=str(run_dir) + "/", prefix="dust", silent=True)
    if p.npart == 0:
        return np.zeros((nx, nx), dtype=np.float64)
    # p.pos is (ndim, npart); convert to (npart, ndim)
    pos = np.column_stack([p.pos[i] for i in range(p.pos.shape[0])])
    masses = np.asarray(p.mass, dtype=np.float64).ravel()
    # Drop particles with invalid (x, y) to avoid NaN in CIC
    valid = np.isfinite(pos[:, 0]) & np.isfinite(pos[:, 1])
    if not np.any(valid):
        return np.zeros((nx, nx), dtype=np.float64)
    pos = pos[valid]
    masses = masses[valid]
    return cic_deposit_2d(pos, masses, nx, box_size=box_size)


def get_output_numbers(run_dir: Path, start: int | None, end: int | None) -> list[int]:
    """Return sorted list of output numbers in run_dir in [start, end] (inclusive)."""
    run_dir = Path(run_dir)
    outs = []
    for d in run_dir.iterdir():
        if d.is_dir() and d.name.startswith("output_"):
            try:
                num = int(d.name.replace("output_", ""))
                outs.append(num)
            except ValueError:
                pass
    if not outs:
        raise FileNotFoundError(f"No output_* directories in {run_dir}")
    outs = sorted(outs)
    if start is not None:
        outs = [n for n in outs if n >= start]
    if end is not None:
        outs = [n for n in outs if n <= end]
    if not outs:
        raise ValueError("No outputs in requested start/end range")
    return outs


def compute_colorbar_range(
    gas_col: np.ndarray,
    dust_col: np.ndarray,
    floor: float = FLOOR,
) -> tuple[float, float]:
    """Compute vmin, vmax for log10 column density (slightly larger than data, centered at mean)."""
    g = np.log10(gas_col + floor)
    d = np.log10(dust_col + floor)
    valid = np.isfinite(g) & np.isfinite(d) & ((gas_col > 0) | (dust_col > 0))
    if not np.any(valid):
        return 1e-3, 1e1
    all_log = np.concatenate([g[valid].ravel(), d[valid].ravel()])
    mean_log = np.mean(all_log)
    min_log = np.min(all_log)
    max_log = np.max(all_log)
    half_span = max((max_log - min_log) / 2 * 1.1, np.std(all_log) * 1.5)
    half_span = max(half_span, 0.5)
    vmin = 10 ** (mean_log - half_span)
    vmax = 10 ** (mean_log + half_span)
    return vmin, vmax


# Default frame size: 1080p (1920x1080) true HD
FRAME_WIDTH_PX = 1920
FRAME_HEIGHT_PX = 1080
FRAME_DPI = 96

RATIO_FLOOR = 1e-30  # floor for gas when computing dust/gas ratio


def compute_ratio_colorbar_range(ratio_map: np.ndarray, floor: float = RATIO_FLOOR) -> tuple[float, float]:
    """Compute vmin, vmax for log10(dust/gas) (symmetric about 0, slightly larger than data range)."""
    log_ratio = np.log10(np.where(ratio_map > 0, ratio_map, np.nan))
    valid = np.isfinite(log_ratio)
    if not np.any(valid):
        return -1.0, 1.0
    vals = log_ratio[valid]
    mean_log = np.mean(vals)
    min_log = np.min(vals)
    max_log = np.max(vals)
    half_span = max((max_log - min_log) / 2 * 1.1, np.std(vals) * 1.5)
    half_span = max(half_span, 0.3)
    vmin = mean_log - half_span
    vmax = mean_log + half_span
    return vmin, vmax


def render_frame(
    gas_col: np.ndarray,
    dust_col: np.ndarray,
    vmin: float,
    vmax: float,
    out_path: Path,
    dpi: int | None = None,
) -> None:
    """Draw gas and dust column density panels with shared log colorbar; save PNG."""
    if dpi is None:
        dpi = FRAME_DPI
    figsize = (FRAME_WIDTH_PX / dpi, FRAME_HEIGHT_PX / dpi)
    norm = LogNorm(vmin=vmin, vmax=vmax)
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    axes[0].imshow(gas_col.T, origin="lower", norm=norm, cmap="inferno")
    axes[0].set_title("Gas column density")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_aspect("equal")
    axes[1].imshow(dust_col.T, origin="lower", norm=norm, cmap="inferno")
    axes[1].set_title("Dust column density")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_aspect("equal")
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="inferno"),
        ax=axes,
        label=r"$\log_{10}$ column density",
        shrink=0.6,
    )
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def render_ratio_frame(
    gas_col: np.ndarray,
    dust_col: np.ndarray,
    vmin: float,
    vmax: float,
    out_path: Path,
    dpi: int | None = None,
) -> None:
    """Draw single panel: log10(dust/gas) mass ratio; save PNG."""
    if dpi is None:
        dpi = FRAME_DPI
    figsize = (FRAME_WIDTH_PX / dpi, FRAME_HEIGHT_PX / dpi)
    ratio = np.where(gas_col > 0, dust_col / (gas_col + RATIO_FLOOR), np.nan)
    log_ratio = np.log10(np.where(ratio > 0, ratio, np.nan))
    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    # Diverging norm: center at 0 (log10(dust/gas)=0 => dust/gas=1)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    im = ax.imshow(
        log_ratio.T,
        origin="lower",
        cmap="RdBu_r",
        norm=norm,
    )
    ax.set_title(r"$\log_{10}$(dust / gas)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, label=r"$\log_{10}$(dust / gas)", shrink=0.8)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Make column-density video (gas + dust) from mini-ramses outputs.",
    )
    ap.add_argument("--run-dir", type=Path, default=Path("."), help="Run directory (output_XXXXX)")
    ap.add_argument("--start", type=int, default=None, help="First output number (inclusive)")
    ap.add_argument("--end", type=int, default=None, help="Last output number (inclusive)")
    ap.add_argument("--nx", type=int, default=128, help="Grid size (default 128)")
    ap.add_argument("--fps", type=int, default=24, help="Output video FPS (default 24)")
    ap.add_argument("--out-video", type=Path, default=Path("column_density.mp4"), help="Output video path")
    ap.add_argument(
        "--frames-dir",
        type=Path,
        default=None,
        help="Directory for frame PNGs (default: run_dir/frames_col)",
    )
    ap.add_argument("--cache-gas", action="store_true", help="Cache gas cubes as gas_XXXXX.cube")
    ap.add_argument("--keep-frames", action="store_true", help="Do not delete frames dir after encoding")
    ap.add_argument(
        "--ffmpeg-only",
        action="store_true",
        help="Skip rendering; only run ffmpeg on existing frame_*.png in --frames-dir",
    )
    ap.add_argument(
        "--ratio-only",
        action="store_true",
        help="Make only dust-to-gas mass ratio video (frames in run_dir/frames_ratio)",
    )
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory not found: {run_dir}")

    # Separate frame dir and default video name for ratio vs column-density
    default_frames_subdir = "frames_ratio" if args.ratio_only else "frames_col"
    frames_dir = args.frames_dir if args.frames_dir is not None else run_dir / default_frames_subdir
    frames_dir = frames_dir.resolve()

    if args.ratio_only and not args.out_video.is_absolute() and args.out_video.name == "column_density.mp4":
        args.out_video = Path("dust_to_gas_ratio.mp4")

    if args.ffmpeg_only:
        # Discover existing frame_XXXXX.png and encode only
        import re
        if not frames_dir.is_dir():
            raise SystemExit(f"Frames directory not found: {frames_dir}")
        frame_pattern = re.compile(r"frame_(\d+)\.png$")
        found = []
        for f in frames_dir.iterdir():
            if f.is_file():
                m = frame_pattern.match(f.name)
                if m:
                    found.append((int(m.group(1)), f.name))
        if not found:
            raise SystemExit(f"No frame_*.png found in {frames_dir}")
        frame_names = [name for _, name in sorted(found)]
        print(f"Found {len(frame_names)} frames in {frames_dir}")
    elif args.ratio_only:
        # Dust-to-gas ratio video only
        output_numbers = get_output_numbers(run_dir, args.start, args.end)
        frames_dir.mkdir(parents=True, exist_ok=True)

        last = output_numbers[-1]
        print(f"Computing ratio colorbar range from last output {last} ...")
        gas_col_last = get_gas_column(run_dir, last, args.nx, cache=args.cache_gas)
        dust_col_last = get_dust_column(run_dir, last, args.nx)
        mg = np.mean(gas_col_last)
        md = np.mean(dust_col_last)
        if mg <= 0:
            mg = 1.0
        if md <= 0:
            md = 1.0
        gas_col_last = gas_col_last / mg
        dust_col_last = dust_col_last / md
        ratio_last = np.where(gas_col_last > 0, dust_col_last / (gas_col_last + RATIO_FLOOR), np.nan)
        vmin, vmax = compute_ratio_colorbar_range(ratio_last)
        print(f"  vmin={vmin:.2f}, vmax={vmax:.2f} (log10 dust/gas)")

        n_frames = len(output_numbers)
        for i, output_num in enumerate(output_numbers):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"Frame {i + 1}/{n_frames} (output_{output_num:05d})")
            gas_col = get_gas_column(run_dir, output_num, args.nx, cache=args.cache_gas)
            dust_col = get_dust_column(run_dir, output_num, args.nx)
            mg = np.mean(gas_col)
            md = np.mean(dust_col)
            if mg <= 0:
                mg = 1.0
            if md <= 0:
                md = 1.0
            gas_col = gas_col / mg
            dust_col = dust_col / md
            out_path = frames_dir / f"frame_{output_num:05d}.png"
            render_ratio_frame(gas_col, dust_col, vmin, vmax, out_path)
    else:
        # Column-density (gas + dust) video
        output_numbers = get_output_numbers(run_dir, args.start, args.end)
        frames_dir.mkdir(parents=True, exist_ok=True)

        # Precompute colorbar from last output
        last = output_numbers[-1]
        print(f"Computing colorbar range from last output {last} ...")
        gas_col_last = get_gas_column(run_dir, last, args.nx, cache=args.cache_gas)
        dust_col_last = get_dust_column(run_dir, last, args.nx)
        # Normalize to mean 1 for comparability (same as plot_dust_gas)
        mg = np.mean(gas_col_last)
        md = np.mean(dust_col_last)
        if mg <= 0:
            mg = 1.0
        if md <= 0:
            md = 1.0
        gas_col_last = gas_col_last / mg
        dust_col_last = dust_col_last / md
        vmin, vmax = compute_colorbar_range(gas_col_last, dust_col_last)
        print(f"  vmin={vmin:.2e}, vmax={vmax:.2e}")

        # Render all frames
        n_frames = len(output_numbers)
        for i, output_num in enumerate(output_numbers):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"Frame {i + 1}/{n_frames} (output_{output_num:05d})")
            gas_col = get_gas_column(run_dir, output_num, args.nx, cache=args.cache_gas)
            dust_col = get_dust_column(run_dir, output_num, args.nx)
            mg = np.mean(gas_col)
            md = np.mean(dust_col)
            if mg <= 0:
                mg = 1.0
            if md <= 0:
                md = 1.0
            gas_col = gas_col / mg
            dust_col = dust_col / md
            out_path = frames_dir / f"frame_{output_num:05d}.png"
            render_frame(gas_col, dust_col, vmin, vmax, out_path)

    # Use concat demuxer so non-contiguous output numbers (e.g. --start 10 --end 20) work.
    list_file = frames_dir / "frame_list.txt"
    duration_sec = 1.0 / args.fps
    with open(list_file, "w") as f:
        if args.ffmpeg_only:
            for name in frame_names:
                f.write(f"file '{name}'\n")
                f.write(f"duration {duration_sec}\n")
            f.write(f"file '{frame_names[-1]}'\n")
        else:
            for output_num in output_numbers:
                f.write(f"file 'frame_{output_num:05d}.png'\n")
                f.write(f"duration {duration_sec}\n")
            f.write(f"file 'frame_{output_numbers[-1]:05d}.png'\n")

    out_video = args.out_video.resolve()
    if not out_video.is_absolute():
        out_video = (run_dir / out_video).resolve()
    out_video.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "18",  # high quality (default 23; lower = better, 18 ~ visually lossless)
        str(out_video),
    ]
    print("Running ffmpeg ...")
    result = subprocess.run(cmd, cwd=str(frames_dir))
    if result.returncode != 0:
        # Fallback for builds without libx264 (e.g. some conda ffmpeg)
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c:v", "mpeg4", "-pix_fmt", "yuv420p",
            "-q:v", "2",  # high quality for mpeg4 (2-5 = high)
            str(out_video),
        ]
        result = subprocess.run(cmd, cwd=str(frames_dir))
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg failed with code {result.returncode}")
    print(f"Saved {out_video}")

    if not args.ffmpeg_only and not args.keep_frames:
        import shutil
        shutil.rmtree(frames_dir)
        print("Removed frames directory.")
    elif not args.ffmpeg_only:
        print(f"Frames left in {frames_dir}")


if __name__ == "__main__":
    main()
