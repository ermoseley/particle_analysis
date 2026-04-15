#!/usr/bin/env python3
"""
Video: gas column density in colorcet isolum (hue ~ log gas / mean), dust column
as multiplicative darkening (clear → black). Spatial means for normalization and
color limits are taken from the last output only; every frame uses those same
means so the field evolves against a fixed reference. Each colorbar's vmin/vmax
is the min and max of that log field on the last frame only (not per-frame).

No titles or axis annotations; left grayscale colorbar (dust log), right isolum
colorbar (gas log). Default projection integrates along x (yz plane).

Usage:
  python make_dust_alpha_gas_video.py --run-dir /path/to/run --start 1 --end 50
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import colorcet as cc
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

from make_column_density_video import (
    FLOOR,
    FRAME_DPI,
    FRAME_HEIGHT_PX,
    FRAME_WIDTH_PX,
    get_dust_column,
    get_gas_column,
    get_output_numbers,
)

GAS_CMAP = cc.cm["isolum"]
DUST_CMAP = LinearSegmentedColormap.from_list("dust_wb", [(1.0, 1.0, 1.0), (0.0, 0.0, 0.0)])

DUST_CB_LABEL = r"$\log_{10}\rho_{\rm dust}/\langle \rho_{\rm dust}\rangle$"
GAS_CB_LABEL = r"$\log_{10}\rho_{\rm gas}/\langle \rho_{\rm gas}\rangle$"


def _orient_like_column_video(arr: np.ndarray) -> np.ndarray:
    """Match imshow orientation used in make_column_density_video (transpose first two dims)."""
    if arr.ndim == 2:
        return arr.T
    return np.transpose(arr, (1, 0, 2))


def log_vmin_vmax_from_last_frame(log_last: np.ndarray) -> tuple[float, float]:
    """vmin/vmax = finite min/max of the log field on the last snapshot only."""
    v = np.asarray(log_last, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return -1.0, 1.0
    vmin = float(np.min(v))
    vmax = float(np.max(v))
    if vmin >= vmax:
        vmax = vmin + 1e-9
    return vmin, vmax


def render_frame(
    log_g: np.ndarray,
    log_d: np.ndarray,
    vmin_g: float,
    vmax_g: float,
    vmin_d: float,
    vmax_d: float,
    out_path: Path,
    box_size: float = 1.0,
    dpi: int | None = None,
) -> None:
    if dpi is None:
        dpi = FRAME_DPI
    figsize = (FRAME_WIDTH_PX / dpi, FRAME_HEIGHT_PX / dpi)

    norm_g = Normalize(vmin=vmin_g, vmax=vmax_g, clip=True)
    norm_d = Normalize(vmin=vmin_d, vmax=vmax_d, clip=True)

    sm_g = ScalarMappable(norm=norm_g, cmap=GAS_CMAP)
    rgba = sm_g.to_rgba(log_g)
    rgb = rgba[..., :3]
    alpha = norm_d(log_d)
    alpha = np.clip(alpha, 0.0, 1.0)
    out_rgb = (1.0 - alpha[..., None]) * rgb
    out_rgb = np.clip(out_rgb, 0.0, 1.0)

    disp = _orient_like_column_video(out_rgb)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(
        disp,
        origin="lower",
        extent=(0.0, box_size, 0.0, box_size),
        interpolation="nearest",
    )
    ax.set_axis_off()
    ax.set_aspect("equal")

    divider = make_axes_locatable(ax)
    cax_d = divider.append_axes("left", size="3.8%", pad=0.14)
    cax_g = divider.append_axes("right", size="3.8%", pad=0.14)

    cb_d = fig.colorbar(
        ScalarMappable(norm=norm_d, cmap=DUST_CMAP),
        cax=cax_d,
        orientation="vertical",
    )
    cb_d.set_label(DUST_CB_LABEL)
    # Left colorbar: put ticks on outer (left) side, axis label on inner (right) side
    cb_d.ax.yaxis.set_ticks_position("left")
    cb_d.ax.yaxis.set_label_position("right")

    cb_g = fig.colorbar(
        ScalarMappable(norm=norm_g, cmap=GAS_CMAP),
        cax=cax_g,
        orientation="vertical",
    )
    cb_g.set_label(GAS_CB_LABEL)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Gas (isolum) + dust (alpha) column-density video from mini-ramses outputs.",
    )
    ap.add_argument("--run-dir", type=Path, default=Path("."), help="Run directory (output_XXXXX)")
    ap.add_argument("--start", type=int, default=None, help="First output number (inclusive)")
    ap.add_argument("--end", type=int, default=None, help="Last output number (inclusive)")
    ap.add_argument("--nx", type=int, default=128, help="Grid size (default 128)")
    ap.add_argument(
        "--axis",
        type=str,
        default="x",
        choices=("x", "y", "z"),
        help="Line-of-sight for column integration (default x)",
    )
    ap.add_argument("--box-size", type=float, default=1.0, help="Domain size for imshow extent")
    ap.add_argument("--fps", type=int, default=24, help="Output video FPS (default 24)")
    ap.add_argument(
        "--out-video",
        type=Path,
        default=Path("dust_alpha_gas.mp4"),
        help="Output video path (default dust_alpha_gas.mp4)",
    )
    ap.add_argument(
        "--frames-dir",
        type=Path,
        default=None,
        help="Directory for frame PNGs (default: run_dir/frames_dust_alpha)",
    )
    ap.add_argument("--cache-gas", action="store_true", help="Cache gas cubes as gas_XXXXX.cube")
    ap.add_argument("--keep-frames", action="store_true", help="Do not delete frames dir after encoding")
    ap.add_argument(
        "--ffmpeg-only",
        action="store_true",
        help="Skip rendering; only run ffmpeg on existing frame_*.png in --frames-dir",
    )
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory not found: {run_dir}")

    axis = args.axis.lower()
    frames_dir = args.frames_dir if args.frames_dir is not None else run_dir / "frames_dust_alpha"
    frames_dir = frames_dir.resolve()

    frame_names: list[str] = []
    output_numbers: list[int] = []

    if args.ffmpeg_only:
        if not frames_dir.is_dir():
            raise SystemExit(f"Frames directory not found: {frames_dir}")
        frame_pattern = re.compile(r"frame_(\d+)\.png$")
        found: list[tuple[int, str]] = []
        for f in frames_dir.iterdir():
            if f.is_file():
                m = frame_pattern.match(f.name)
                if m:
                    found.append((int(m.group(1)), f.name))
        if not found:
            raise SystemExit(f"No frame_*.png found in {frames_dir}")
        frame_names = [name for _, name in sorted(found)]
        print(f"Found {len(frame_names)} frames in {frames_dir}")
    else:
        output_numbers = get_output_numbers(run_dir, args.start, args.end)
        frames_dir.mkdir(parents=True, exist_ok=True)

        last = output_numbers[-1]
        print(f"Computing color limits from last output {last} (axis={axis}) ...")
        gas_col_last = get_gas_column(run_dir, last, args.nx, cache=args.cache_gas, axis=axis)
        dust_col_last = get_dust_column(run_dir, last, args.nx, axis=axis)
        mg = np.mean(gas_col_last)
        md = np.mean(dust_col_last)
        if mg <= 0:
            mg = 1.0
        if md <= 0:
            md = 1.0
        gas_n = gas_col_last / mg
        dust_n = dust_col_last / md
        log_g_last = np.log10(gas_n + FLOOR)
        log_d_last = np.log10(dust_n + FLOOR)
        vmin_g, vmax_g = log_vmin_vmax_from_last_frame(log_g_last)
        vmin_d, vmax_d = log_vmin_vmax_from_last_frame(log_d_last)
        print(f"  gas log:  vmin={vmin_g:.4f}, vmax={vmax_g:.4f}")
        print(f"  dust log: vmin={vmin_d:.4f}, vmax={vmax_d:.4f}")

        n_frames = len(output_numbers)
        for i, output_num in enumerate(output_numbers):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"Frame {i + 1}/{n_frames} (output_{output_num:05d})")
            gas_col = get_gas_column(run_dir, output_num, args.nx, cache=args.cache_gas, axis=axis)
            dust_col = get_dust_column(run_dir, output_num, args.nx, axis=axis)
            # Same mg, md as last frame (not per-frame means)
            log_g = np.log10(gas_col / mg + FLOOR)
            log_d = np.log10(dust_col / md + FLOOR)
            out_path = frames_dir / f"frame_{output_num:05d}.png"
            render_frame(
                log_g,
                log_d,
                vmin_g,
                vmax_g,
                vmin_d,
                vmax_d,
                out_path,
                box_size=args.box_size,
            )

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
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        str(out_video),
    ]
    print("Running ffmpeg ...")
    result = subprocess.run(cmd, cwd=str(frames_dir))
    if result.returncode != 0:
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-c:v",
            "mpeg4",
            "-pix_fmt",
            "yuv420p",
            "-q:v",
            "2",
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
