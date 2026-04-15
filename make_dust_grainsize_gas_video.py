#!/usr/bin/env python3
"""
Video: gas column density in colorcet CET_I3 (isoluminant cyan-magenta) with
dust overlay whose alpha is set by dust column density and whose color encodes a
grain-size statistic along the line of sight.

Default mode is ``mean-dev``: deposit the direct dust moments ``Σm`` and
``Σ(m a)`` from the stored particle ``size`` field, then display
``log10(a_mean_los / a_ref_last)`` where ``a_ref_last`` is the last snapshot's
global dust-mass-weighted mean grain size. This makes true mean-size deviations
visible while preserving the fixed-last-frame normalization used by the other
video scripts.

``mean-abs`` shows the direct LOS mass-weighted mean grain size.
``legacy-binned`` preserves the older 16-bin median-in-bin surrogate.

Use ``--nx 128`` (default) to match a 128^3 column sampling; coarser ``--nx``
is only for quick tests.

Gas and dust column log-contrast colorbars use the **last** snapshot in the
requested range: symmetric limits ``±deltav`` about 0, with
``deltav = max(|min|, |max|)`` of ``log10(ρ/⟨ρ⟩)`` on that frame.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import colorcet as cc
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from column_utils import get_gas_column
from dust_projection import (
    global_mass_weighted_mean_size,
    legacy_binned_mean_size_map,
    logsize_bin_edges_from_sizes,
    mean_size_from_moments,
    project_dust_moments,
    read_dust_snapshot,
)
from video_common import (
    FLOOR,
    FRAME_DPI,
    FRAME_HEIGHT_PX,
    FRAME_WIDTH_PX,
    discover_frame_names,
    encode_frames,
    get_output_numbers,
    log_vmin_vmax_from_last_frame,
    orient_like_column_video,
    symmetric_log_vmin_vmax_from_last_frame,
    write_concat_frame_list,
)

GAS_CMAP = cc.cm["CET_I3"]
DUST_COL_CMAP = LinearSegmentedColormap.from_list("dust_col_wb", [(1.0, 1.0, 1.0), (0.0, 0.0, 0.0)])
# Yellow (small grains) → black (large grains); same for mean-dev and mean-abs.
GRAIN_CMAP = LinearSegmentedColormap.from_list("grain_yk", [(1.0, 1.0, 0.0), (0.0, 0.0, 0.0)])

N_GRAIN_BINS = 16

# Column integration grid per side (128×128 maps); matches typical 128³ runs.
COLUMN_NX_DEFAULT = 128

DUST_COL_LABEL = r"$\log_{10}\rho_{\rm dust}/\langle \rho_{\rm dust}\rangle$"
GAS_COL_LABEL = r"$\log_{10}\rho_{\rm gas}/\langle \rho_{\rm gas}\rangle$"
GRAIN_LABEL = r"$a\ [{\rm \mu m}]$"


def grain_colorbar_ticks(vmin: float, vmax: float, ref_micron: float) -> tuple[np.ndarray, list[str]]:
    """Generate bottom colorbar ticks shown in microns."""
    ticks = np.linspace(vmin, vmax, 5)
    labels = [f"{ref_micron * (10.0 ** t):.3g}" for t in ticks]
    return ticks, labels


def compute_direct_mean_size_map(
    run_dir: Path,
    output_num: int,
    nx: int,
    axis: str,
    box_size: float,
):
    """Return dust column mass, direct LOS mean size, and the dust snapshot."""
    snapshot = read_dust_snapshot(run_dir, output_num)
    moments = project_dust_moments(snapshot, nx, axis=axis, box_size=box_size)
    sum_m = moments["sum_m"]
    mean_size = mean_size_from_moments(sum_m, moments["sum_ma"])
    return sum_m, mean_size, snapshot


def compute_grain_stat_map(
    run_dir: Path,
    output_num: int,
    nx: int,
    *,
    axis: str,
    box_size: float,
    field_mode: str,
    grain_bins: str,
    logsize_edges: np.ndarray | None = None,
):
    """Return dust column mass, grain statistic map, and the dust snapshot."""
    if field_mode == "legacy-binned":
        return legacy_binned_mean_size_map(
            run_dir,
            output_num,
            nx,
            axis=axis,
            box_size=box_size,
            grain_bins=grain_bins,
            n_bins=N_GRAIN_BINS,
            logsize_edges=logsize_edges,
        )
    return compute_direct_mean_size_map(run_dir, output_num, nx, axis=axis, box_size=box_size)


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
    field_mode: str,
    grain_ref_micron: float,
    out_path: Path,
    box_size: float = 1.0,
    dpi: int | None = None,
) -> None:
    """Render one composite frame."""
    if dpi is None:
        dpi = FRAME_DPI
    figsize = (FRAME_WIDTH_PX / dpi, FRAME_HEIGHT_PX / dpi)

    norm_g = Normalize(vmin=vmin_g, vmax=vmax_g, clip=True)
    norm_dcol = Normalize(vmin=vmin_dcol, vmax=vmax_dcol, clip=True)
    if field_mode == "mean-dev":
        norm_grain = TwoSlopeNorm(vmin=vmin_grain, vcenter=0.0, vmax=vmax_grain)
    else:
        norm_grain = Normalize(vmin=vmin_grain, vmax=vmax_grain, clip=True)
    grain_cmap = GRAIN_CMAP

    gas_rgb = ScalarMappable(norm=norm_g, cmap=GAS_CMAP).to_rgba(log_g)[..., :3]
    grain_rgb = ScalarMappable(norm=norm_grain, cmap=grain_cmap).to_rgba(log_grain)[..., :3]
    alpha = np.clip(norm_dcol(log_dcol), 0.0, 1.0)
    alpha = np.where(np.isfinite(log_dcol), alpha, 0.0)

    rgb = (1.0 - alpha[..., None]) * gas_rgb + alpha[..., None] * grain_rgb
    rgb = np.clip(rgb, 0.0, 1.0)
    disp = orient_like_column_video(rgb)

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
    cb_d.set_label(DUST_COL_LABEL, labelpad=14)
    cb_d.ax.yaxis.set_ticks_position("left")
    cb_d.ax.yaxis.set_label_position("left")

    cb_g = fig.colorbar(
        ScalarMappable(norm=norm_g, cmap=GAS_CMAP),
        cax=cax_g,
        orientation="vertical",
    )
    cb_g.set_label(GAS_COL_LABEL)

    cb_a = fig.colorbar(
        ScalarMappable(norm=norm_grain, cmap=grain_cmap),
        cax=cax_a,
        orientation="horizontal",
    )
    tick_vals, tick_labels = grain_colorbar_ticks(vmin_grain, vmax_grain, grain_ref_micron)
    cb_a.set_ticks(tick_vals)
    cb_a.set_ticklabels(tick_labels)
    cb_a.set_label(GRAIN_LABEL)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Gas (CET_I3) + dust column alpha + LOS grain-size statistic.",
    )
    ap.add_argument("--run-dir", type=Path, default=Path("."), help="Run directory (output_XXXXX)")
    ap.add_argument("--start", type=int, default=None, help="First output number (inclusive)")
    ap.add_argument("--end", type=int, default=None, help="Last output number (inclusive)")
    ap.add_argument(
        "--nx",
        type=int,
        default=COLUMN_NX_DEFAULT,
        help=f"Column-density grid resolution per side (default {COLUMN_NX_DEFAULT}, e.g. 128³-style maps)",
    )
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
    ap.add_argument(
        "--field-mode",
        type=str,
        default="mean-dev",
        choices=("mean-dev", "mean-abs", "legacy-binned"),
        help=(
            "mean-dev (default): direct LOS mass-weighted mean-size deviation relative to the "
            "last snapshot global mean. mean-abs: direct LOS mass-weighted mean size. "
            "legacy-binned: older 16-bin median-in-bin surrogate."
        ),
    )
    ap.add_argument(
        "--grain-bins",
        type=str,
        default="logsize",
        choices=("identity", "logsize"),
        help=(
            "Legacy binning for --field-mode legacy-binned only. logsize: 16 log-spaced size bins. "
            "identity: 16 equal particle-ID ranges."
        ),
    )
    ap.add_argument(
        "--grain-micron-per-code",
        type=float,
        default=23.0,
        help=(
            "Linear code→µm scale: a[µm] = a[code] × this value. "
            "Default 23 corresponds to median 0.23 µm at size 0.01 code (0.23/0.01)."
        ),
    )
    ap.add_argument(
        "--grain-ref-micron",
        type=float,
        default=0.23,
        help="Reference grain size in µm for mean-abs color scale log10(a/a_ref) (default 0.23).",
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
        frame_names = discover_frame_names(frames_dir)
        print(f"Found {len(frame_names)} frames in {frames_dir}")
    else:
        output_numbers = get_output_numbers(run_dir, args.start, args.end)
        frames_dir.mkdir(parents=True, exist_ok=True)

        last = output_numbers[-1]
        print(f"Computing reference limits from last output {last} (axis={axis}) ...")
        logsize_edges: np.ndarray | None = None
        snapshot_last = read_dust_snapshot(run_dir, last)
        if args.field_mode == "legacy-binned" and args.grain_bins == "logsize":
            logsize_edges = logsize_bin_edges_from_sizes(snapshot_last.size, N_GRAIN_BINS)
            print(f"  grain-bins=logsize: {N_GRAIN_BINS} log-spaced edges from last output sizes")

        gas_col_last = get_gas_column(run_dir, last, args.nx, cache=args.cache_gas, axis=axis)
        dust_col_last, grain_stat_last, snapshot_last = compute_grain_stat_map(
            run_dir,
            last,
            args.nx,
            axis=axis,
            box_size=args.box_size,
            field_mode=args.field_mode,
            grain_bins=args.grain_bins,
            logsize_edges=logsize_edges,
        )

        mg_last = float(np.mean(gas_col_last))
        md_last = float(np.mean(dust_col_last))
        if mg_last <= 0.0:
            mg_last = 1.0
        if md_last <= 0.0:
            md_last = 1.0

        to_micron = float(args.grain_micron_per_code)
        grain_stat_last_micron = grain_stat_last * to_micron
        ref_mean_size_last = global_mass_weighted_mean_size(snapshot_last)
        ref_mean_size_last_micron = ref_mean_size_last * to_micron
        a_ref_um = float(args.grain_ref_micron)

        log_g_last = np.log10(gas_col_last / mg_last + FLOOR)
        log_dcol_last = np.log10(dust_col_last / md_last + FLOOR)
        if args.field_mode == "mean-dev":
            log_grain_last = np.log10(grain_stat_last / ref_mean_size_last)
            vmin_grain, vmax_grain = symmetric_log_vmin_vmax_from_last_frame(log_grain_last)
            grain_ref_micron = ref_mean_size_last_micron
        else:
            log_grain_last = np.log10(grain_stat_last_micron / a_ref_um)
            vmin_grain, vmax_grain = log_vmin_vmax_from_last_frame(log_grain_last)
            grain_ref_micron = a_ref_um

        # Symmetric log₁₀(ρ/⟨ρ⟩) colorbars from last frame: ± deltav with
        # deltav = max(|min|, |max|) over all pixels (center 0, clip OOB on other frames).
        vmin_g, vmax_g = symmetric_log_vmin_vmax_from_last_frame(log_g_last)
        vmin_dcol, vmax_dcol = symmetric_log_vmin_vmax_from_last_frame(log_dcol_last)

        g_data_min, g_data_max = log_vmin_vmax_from_last_frame(log_g_last)
        d_data_min, d_data_max = log_vmin_vmax_from_last_frame(log_dcol_last)
        g_deltav = float(vmax_g)
        d_deltav = float(vmax_dcol)
        print(
            f"  gas log:        symmetric ±{g_deltav:.4f} "
            f"(last-frame log₁₀ range {g_data_min:.4f} … {g_data_max:.4f})"
        )
        print(
            f"  dust col log:   symmetric ±{d_deltav:.4f} "
            f"(last-frame log₁₀ range {d_data_min:.4f} … {d_data_max:.4f})"
        )
        print(f"  grain size log: vmin={vmin_grain:.4f}, vmax={vmax_grain:.4f}")
        print(
            f"  grain scale:    {to_micron:g} micron/code "
            f"(linear; ref display {a_ref_um:g} µm for mean-abs)"
        )
        if args.field_mode == "mean-dev":
            print(f"  grain ref mean: {ref_mean_size_last_micron:.6e} micron")

        n_frames = len(output_numbers)
        for i, output_num in enumerate(output_numbers):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"Frame {i + 1}/{n_frames} (output_{output_num:05d})")

            gas_col = get_gas_column(run_dir, output_num, args.nx, cache=args.cache_gas, axis=axis)
            dust_col, grain_stat, _ = compute_grain_stat_map(
                run_dir,
                output_num,
                args.nx,
                axis=axis,
                box_size=args.box_size,
                field_mode=args.field_mode,
                grain_bins=args.grain_bins,
                logsize_edges=logsize_edges,
            )

            log_g = np.log10(gas_col / mg_last + FLOOR)
            log_dcol = np.log10(dust_col / md_last + FLOOR)
            if args.field_mode == "mean-dev":
                log_grain = np.log10(grain_stat / ref_mean_size_last)
            else:
                grain_stat_micron = grain_stat * to_micron
                log_grain = np.log10(grain_stat_micron / a_ref_um)

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
                args.field_mode,
                grain_ref_micron,
                out_path,
                box_size=args.box_size,
            )

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
