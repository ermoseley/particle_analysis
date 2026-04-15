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
from pathlib import Path

import numpy as np

# Agg before pyplot for headless/speed.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm

from column_utils import (
    cic_deposit_2d,
    column_density,
    dust_pos_plane as _dust_pos_plane,
    get_dust_column,
    get_gas_column,
    read_cube_fortran,
    save_cube_fortran,
)
from video_common import (
    FLOOR,
    FRAME_DPI,
    FRAME_HEIGHT_PX,
    FRAME_WIDTH_PX,
    RATIO_FLOOR,
    discover_frame_names,
    encode_frames,
    get_output_numbers,
    write_concat_frame_list,
)


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
        frame_names = discover_frame_names(frames_dir)
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

    list_file = frames_dir / "frame_list.txt"
    if args.ffmpeg_only:
        write_concat_frame_list(list_file, args.fps, frame_names=frame_names)
    else:
        write_concat_frame_list(list_file, args.fps, output_numbers=output_numbers)

    out_video = args.out_video.resolve()
    if not out_video.is_absolute():
        out_video = (run_dir / out_video).resolve()
    out_video.parent.mkdir(parents=True, exist_ok=True)

    print("Running ffmpeg ...")
    encode_frames(list_file, frames_dir, out_video)
    print(f"Saved {out_video}")

    if not args.ffmpeg_only and not args.keep_frames:
        import shutil
        shutil.rmtree(frames_dir)
        print("Removed frames directory.")
    elif not args.ffmpeg_only:
        print(f"Frames left in {frames_dir}")


if __name__ == "__main__":
    main()
