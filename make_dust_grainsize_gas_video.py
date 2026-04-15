#!/usr/bin/env python3
"""
Video: gas column density in colorcet CET_I3 (isoluminant cyan-magenta) with
dust overlay whose alpha is set by dust column density and whose color is set by
the line-of-sight mass-weighted mean grain size.

Spatial means for gas and dust-column normalization, the grain-size conversion to
microns, and all colorbar limits are derived from the last output only.

No titles or axes. Colorbars:
  - left:  dust column density (white -> black), ticks/label on the left
  - right: gas column density (CET_I3)
  - bottom: grain size (yellow -> black), tick labels in microns

Usage:
  python make_dust_grainsize_gas_video.py --run-dir /path/to/run --start 1 --end 50
"""
from __future__ import annotations

import argparse
import re
import subprocess
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
    _dust_pos_plane,
    cic_deposit_2d,
    get_gas_column,
    get_output_numbers,
)


GAS_CMAP = cc.cm["CET_I3"]
DUST_COL_CMAP = LinearSegmentedColormap.from_list("dust_col_wb", [(1.0, 1.0, 1.0), (0.0, 0.0, 0.0)])
GRAIN_CMAP = LinearSegmentedColormap.from_list("grain_yk", [(1.0, 1.0, 0.0), (0.0, 0.0, 0.0)])

DUST_COL_LABEL = r"$\log_{10}\rho_{\rm dust}/\langle \rho_{\rm dust}\rangle$"
GAS_COL_LABEL = r"$\log_{10}\rho_{\rm gas}/\langle \rho_{\rm gas}\rangle$"
GRAIN_LABEL = r"$a\ [{\rm \mu m}]$"


def _orient_like_column_video(arr: np.ndarray) -> np.ndarray:
    """Match imshow orientation used in the existing column-density scripts."""
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


def read_dust_header_fields(output_dir: Path) -> list[str]:
    """Return ordered field names from dust_header.txt."""
    header_path = output_dir / "dust_header.txt"
    fields: list[str] = []
    for raw in header_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("Total number"):
            continue
        if line.startswith("Particle fields"):
            continue
        fields.extend(line.split())
    if not fields:
        raise ValueError(f"No particle fields found in {header_path}")
    return fields


def read_output_ndim(output_dir: Path) -> int:
    """Read `ndim` from info.txt in the output directory."""
    info_path = output_dir / "info.txt"
    for raw in info_path.read_text().splitlines():
        if raw.strip().startswith("ndim"):
            _, value = raw.split("=", 1)
            return int(value)
    raise ValueError(f"Could not read ndim from {info_path}")


def _expand_real_block_specs(fields: list[str], ndim: int) -> list[str]:
    """Expand header field names into the ordered float32 stream blocks."""
    int_fields = {"level", "birth_id", "id", "identity", "merging_id", "tracking_id"}
    vector_fields = {"pos", "vel", "accel", "angmom"}
    reals: list[str] = []
    for name in fields:
        if name in int_fields:
            continue
        if name in vector_fields:
            for idim in range(ndim):
                reals.append(f"{name}_{idim}")
        else:
            reals.append(name)
    return reals


def read_dust_pos_mass_size(run_dir: Path, output_num: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read dust stream files for one snapshot, extracting only positions, mass, and size.

    The field order is parsed from dust_header.txt so extra float fields like
    charge/mu_adb/vpara are skipped safely.
    """
    output_dir = Path(run_dir) / f"output_{output_num:05d}"
    fields = read_dust_header_fields(output_dir)
    ndim = read_output_ndim(output_dir)
    real_fields = _expand_real_block_specs(fields, ndim)

    if "mass" not in real_fields or "size" not in real_fields:
        raise ValueError(f"dust_header.txt in {output_dir} must include mass and size")

    dust_files = sorted(output_dir.glob("dust.*"))
    if not dust_files:
        return np.empty((0, 3), dtype=np.float64), np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64)

    pos_list: list[np.ndarray] = []
    mass_list: list[np.ndarray] = []
    size_list: list[np.ndarray] = []

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

        pos = np.column_stack([real_data[f"pos_{idim}"] for idim in range(ndim)])
        mass = real_data["mass"]
        size = real_data["size"]

        valid = (
            np.isfinite(pos[:, 0])
            & np.isfinite(pos[:, 1])
            & np.isfinite(pos[:, 2])
            & np.isfinite(mass)
            & np.isfinite(size)
        )
        if not np.any(valid):
            continue
        pos_list.append(pos[valid])
        mass_list.append(mass[valid])
        size_list.append(size[valid])

    if not pos_list:
        return np.empty((0, 3), dtype=np.float64), np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64)

    return (
        np.concatenate(pos_list, axis=0),
        np.concatenate(mass_list, axis=0),
        np.concatenate(size_list, axis=0),
    )


def get_dust_column_and_mean_size(
    run_dir: Path,
    output_num: int,
    nx: int,
    axis: str = "x",
    box_size: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return dust column density, LOS mass-weighted mean grain size, and raw size values.

    mean_size[pix] = sum(m_i * a_i) / sum(m_i)
    """
    pos, masses, sizes = read_dust_pos_mass_size(run_dir, output_num)
    if masses.size == 0:
        z = np.zeros((nx, nx), dtype=np.float64)
        return z.copy(), np.full((nx, nx), np.nan, dtype=np.float64), sizes

    pos_xy = _dust_pos_plane(pos, axis)
    valid = (
        np.isfinite(pos_xy[:, 0])
        & np.isfinite(pos_xy[:, 1])
        & np.isfinite(masses)
        & np.isfinite(sizes)
        & (sizes > 0.0)
    )
    if not np.any(valid):
        z = np.zeros((nx, nx), dtype=np.float64)
        return z.copy(), np.full((nx, nx), np.nan, dtype=np.float64), sizes[valid]

    pos_xy = pos_xy[valid]
    masses = masses[valid]
    sizes = sizes[valid]

    sum_m = cic_deposit_2d(pos_xy, masses, nx, box_size=box_size)
    sum_ma = cic_deposit_2d(pos_xy, masses * sizes, nx, box_size=box_size)

    mean_size = np.full_like(sum_m, np.nan, dtype=np.float64)
    np.divide(sum_ma, sum_m, out=mean_size, where=sum_m > 0.0)
    return sum_m, mean_size, sizes


def size_scale_to_micron(size_last: np.ndarray, peak_micron: float = 0.23) -> float:
    """Map the geometric-center peak of the simulated size range to `peak_micron`."""
    size_last = np.asarray(size_last, dtype=np.float64)
    size_last = size_last[np.isfinite(size_last) & (size_last > 0.0)]
    if size_last.size == 0:
        raise ValueError("No positive dust grain sizes found in last snapshot")

    a_min = float(np.min(size_last))
    a_max = float(np.max(size_last))
    if a_max <= 0.0 or a_min <= 0.0:
        raise ValueError("Dust grain sizes must be positive")

    log_a_peak = np.log10(a_min) + 0.5 * np.log10(a_max / a_min)
    a_peak_dimless = 10.0 ** log_a_peak
    return peak_micron / a_peak_dimless


def grain_colorbar_ticks(vmin: float, vmax: float) -> tuple[np.ndarray, list[str]]:
    """Generate bottom colorbar ticks shown in microns."""
    ticks = np.linspace(vmin, vmax, 5)
    labels = [f"{0.23 * (10.0 ** t):.3g}" for t in ticks]
    return ticks, labels


def render_frame(
    log_g: np.ndarray,
    log_dcol: np.ndarray,
    log_grain: np.ndarray,
    vmin_g: float,
    vmax_g: float,
    vmin_dcol: float,
    vmax_dcol: float,
    vmin_grain: float,
    vmax_grain: float,
    out_path: Path,
    box_size: float = 1.0,
    dpi: int | None = None,
) -> None:
    if dpi is None:
        dpi = FRAME_DPI
    figsize = (FRAME_WIDTH_PX / dpi, FRAME_HEIGHT_PX / dpi)

    norm_g = Normalize(vmin=vmin_g, vmax=vmax_g, clip=True)
    norm_dcol = Normalize(vmin=vmin_dcol, vmax=vmax_dcol, clip=True)
    norm_grain = Normalize(vmin=vmin_grain, vmax=vmax_grain, clip=True)

    gas_rgb = ScalarMappable(norm=norm_g, cmap=GAS_CMAP).to_rgba(log_g)[..., :3]
    grain_rgb = ScalarMappable(norm=norm_grain, cmap=GRAIN_CMAP).to_rgba(log_grain)[..., :3]
    alpha = np.clip(norm_dcol(log_dcol), 0.0, 1.0)
    alpha = np.where(np.isfinite(log_dcol), alpha, 0.0)

    rgb = (1.0 - alpha[..., None]) * gas_rgb + alpha[..., None] * grain_rgb
    rgb = np.clip(rgb, 0.0, 1.0)
    disp = _orient_like_column_video(rgb)

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
    cax_d = divider.append_axes("left", size="3.8%", pad=0.16)
    cax_g = divider.append_axes("right", size="3.8%", pad=0.16)
    cax_a = divider.append_axes("bottom", size="6.0%", pad=0.20)

    cb_d = fig.colorbar(
        ScalarMappable(norm=norm_dcol, cmap=DUST_COL_CMAP),
        cax=cax_d,
        orientation="vertical",
    )
    cb_d.set_label(DUST_COL_LABEL)
    # Dust label on the left of the colorbar; ticks on the right (toward plot) to avoid overlap
    cb_d.ax.yaxis.set_ticks_position("right")
    cb_d.ax.yaxis.set_label_position("left")

    cb_g = fig.colorbar(
        ScalarMappable(norm=norm_g, cmap=GAS_CMAP),
        cax=cax_g,
        orientation="vertical",
    )
    cb_g.set_label(GAS_COL_LABEL)

    cb_a = fig.colorbar(
        ScalarMappable(norm=norm_grain, cmap=GRAIN_CMAP),
        cax=cax_a,
        orientation="horizontal",
    )
    tick_vals, tick_labels = grain_colorbar_ticks(vmin_grain, vmax_grain)
    cb_a.set_ticks(tick_vals)
    cb_a.set_ticklabels(tick_labels)
    cb_a.set_label(GRAIN_LABEL)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Gas density + LOS mass-weighted dust grain-size video from mini-ramses outputs.",
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
        default=Path("dust_grainsize_gas.mp4"),
        help="Output video path (default dust_grainsize_gas.mp4)",
    )
    ap.add_argument(
        "--frames-dir",
        type=Path,
        default=None,
        help="Directory for frame PNGs (default: run_dir/frames_dust_grainsize)",
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
    frames_dir = args.frames_dir if args.frames_dir is not None else run_dir / "frames_dust_grainsize"
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
        print(f"Computing reference limits from last output {last} (axis={axis}) ...")
        gas_col_last = get_gas_column(run_dir, last, args.nx, cache=args.cache_gas, axis=axis)
        dust_col_last, mean_size_last, size_last = get_dust_column_and_mean_size(
            run_dir, last, args.nx, axis=axis, box_size=args.box_size
        )

        mg_last = float(np.mean(gas_col_last))
        md_last = float(np.mean(dust_col_last))
        if mg_last <= 0.0:
            mg_last = 1.0
        if md_last <= 0.0:
            md_last = 1.0

        to_micron = size_scale_to_micron(size_last)
        mean_size_last_micron = mean_size_last * to_micron

        log_g_last = np.log10(gas_col_last / mg_last + FLOOR)
        log_dcol_last = np.log10(dust_col_last / md_last + FLOOR)
        log_grain_last = np.log10(mean_size_last_micron / 0.23)

        vmin_g, vmax_g = log_vmin_vmax_from_last_frame(log_g_last)
        vmin_dcol, vmax_dcol = log_vmin_vmax_from_last_frame(log_dcol_last)
        vmin_grain, vmax_grain = log_vmin_vmax_from_last_frame(log_grain_last)

        print(f"  gas log:        vmin={vmin_g:.4f}, vmax={vmax_g:.4f}")
        print(f"  dust col log:   vmin={vmin_dcol:.4f}, vmax={vmax_dcol:.4f}")
        print(f"  grain size log: vmin={vmin_grain:.4f}, vmax={vmax_grain:.4f}")
        print(f"  grain scale:    peak=0.23 micron, factor={to_micron:.6e} micron/code")

        n_frames = len(output_numbers)
        for i, output_num in enumerate(output_numbers):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"Frame {i + 1}/{n_frames} (output_{output_num:05d})")

            gas_col = get_gas_column(run_dir, output_num, args.nx, cache=args.cache_gas, axis=axis)
            dust_col, mean_size, _ = get_dust_column_and_mean_size(
                run_dir, output_num, args.nx, axis=axis, box_size=args.box_size
            )

            mean_size_micron = mean_size * to_micron
            log_g = np.log10(gas_col / mg_last + FLOOR)
            log_dcol = np.log10(dust_col / md_last + FLOOR)
            log_grain = np.log10(mean_size_micron / 0.23)

            out_path = frames_dir / f"frame_{output_num:05d}.png"
            render_frame(
                log_g,
                log_dcol,
                log_grain,
                vmin_g,
                vmax_g,
                vmin_dcol,
                vmax_dcol,
                vmin_grain,
                vmax_grain,
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
