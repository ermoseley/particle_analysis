#!/usr/bin/env python3
"""
Plot gas, raw tracer, and GP-denoised tracer for multiple runs.

Runs: mc_tracer_ramses, mc_tracer_cic, sgs_tracer_cic_eq0, tracer_cic.

Produces seven figures:
  1. Column density maps (integrated along z): gas, tracer, denoised — N rows x 3 columns.
  2. Mid-plane density slices: gas, tracer, denoised — N rows x 3 columns.
  3. Joint histograms: gas vs tracer — N rows x 3 columns (noisy | tracer denoised | both denoised).
  4. Power spectra: before denoising (left), after denoising (right) — one panel each, all runs overlaid.
  5. Tracer/gas column-density ratio: no denoising | tracer denoised | same Wiener W(k) applied to gas — N rows x 3 columns.
  6. Gas column density: before (raw) vs after applying the tracer Wiener filter W(k) to gas — N rows x 2 columns.
  7. 1D PDF of log₁₀(tracer/gas) cell-centered density: same three scenarios — N panels (one per run), 3 curves each.

This repo is self-contained: denoise_cube.py, utils/py/miniramses.py, and
utils/f90/part2cube (build with make there) are included. For RAMSES runs
either add ramses-pic under this repo or pre-build gas/tracer cubes.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm

# Paths: all under this repo (particle_analysis)
RUNDIR = Path(__file__).resolve().parent
ROOT = RUNDIR
RUNDIR_TOP = RUNDIR
MINI_RAMSES = RUNDIR
RAMSES_PIC = RUNDIR / "ramses-pic"  # optional; for RAMSES runs put gas/tracer cubes in run dirs or add ramses-pic
MINI_PART2CUBE = str(RUNDIR / "utils" / "f90" / "part2cube")
RAMSES_PART2CUBE = str(RAMSES_PIC / "utils" / "f90" / "part2cube") if RAMSES_PIC.exists() else MINI_PART2CUBE
DENOISE_SCRIPT = RUNDIR / "denoise_cube.py"

sys.path.insert(0, str(RUNDIR / "utils" / "py"))

OUTPUT_NUM = 3
NX = 64
N_EFF = 16
LOG_SIGMA_RANGE = 2.5  # Plot range in log space: mean +/- this many sigma (of log10(gas))


def read_cube_fortran(cube_file: Path) -> np.ndarray:
    cube_file = Path(cube_file)
    file_size = cube_file.stat().st_size
    with open(cube_file, "rb") as f:
        _ = np.fromfile(f, dtype=np.int32, count=1)[0]
        nx, ny, nz = np.fromfile(f, dtype=np.int32, count=3)
        _ = np.fromfile(f, dtype=np.int32, count=1)[0]
        n_cells = nx * ny * nz
        expected_f32 = 28 + n_cells * 4
        dtype = np.float32 if file_size == expected_f32 else np.float64
        _ = np.fromfile(f, dtype=np.int32, count=1)[0]
        cube = np.fromfile(f, dtype=dtype, count=n_cells)
        _ = np.fromfile(f, dtype=np.int32, count=1)[0]
    return cube.reshape((nx, ny, nz), order="F")


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


def get_gas_cube(run_dir: Path, output_num: int, nx: int, is_ramses: bool) -> np.ndarray:
    run_dir = Path(run_dir)
    gas_file = run_dir / f"gas_{output_num:05d}.cube"
    if gas_file.exists():
        return read_cube_fortran(gas_file)

    if is_ramses:
        if not RAMSES_PIC.exists():
            raise FileNotFoundError(
                f"RAMSES run but gas cube missing and ramses-pic not found at {RAMSES_PIC}. "
                "Either put gas_{output_num:05d}.cube in the run dir or add ramses-pic under this repo."
            )
        sys.path.insert(0, str(RAMSES_PIC / "utils" / "py"))
        import ramses_io as ram
        orig_cwd = os.getcwd()
        try:
            os.chdir(run_dir)
            c = ram.rd_cell(output_num)
            from miniramses import mk_cube
            cube = mk_cube(c.x[0], c.x[1], c.x[2], c.dx, c.u[0])
            cube = cube.T
        finally:
            os.chdir(orig_cwd)
    else:
        import miniramses as ram
        c = ram.rd_cell(output_num, path=str(run_dir) + "/")
        cube = ram.mk_cube(c.x[0], c.x[1], c.x[2], c.dx, c.u[0])
        cube = cube.T

    save_cube_fortran(cube, gas_file)
    return cube


def get_tracer_cube(run_dir: Path, output_num: int, nx: int, is_ramses: bool, dep: str = "NGP") -> np.ndarray:
    run_dir = Path(run_dir)
    prefix = "trac"
    cube_file = run_dir / f"{prefix}_{output_num:05d}.cube"
    if cube_file.exists():
        return read_cube_fortran(cube_file)

    out_dir = f"output_{output_num:05d}"
    if is_ramses:
        if not RAMSES_PIC.exists():
            raise FileNotFoundError(
                f"RAMSES run but tracer cube missing and ramses-pic not found at {RAMSES_PIC}. "
                "Either put trac_{output_num:05d}.cube in the run dir or add ramses-pic under this repo."
            )
        part2cube = RAMSES_PART2CUBE
        cmd = [
            part2cube, "-inp", out_dir, "-out", cube_file.name,
            "-nx", str(nx), "-ny", str(nx), "-nz", str(nx), "-per", ".true.", "-str", ".true.",
        ]
    else:
        part2cube = MINI_PART2CUBE
        cmd = [
            part2cube, "-inp", out_dir, "-pre", prefix,
            "-nx", str(nx), "-ny", str(nx), "-nz", str(nx), "-per", ".true.", "-dep", dep,
        ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(run_dir))
    if result.returncode != 0:
        raise RuntimeError(f"part2cube failed: {result.stderr}")
    temp = run_dir / f"{prefix}.cube"
    if temp.exists() and temp != cube_file:
        temp.rename(cube_file)
    if not cube_file.exists():
        raise FileNotFoundError(f"part2cube did not create {cube_file}")
    return read_cube_fortran(cube_file)


def _denoised_suffix(
    dep: str | None, deconvolve: bool, n_strata: int,
    gas_cube_path: str | None, psd_floor: str, floor_snr: float, deposit_noise: str,
) -> str:
    suffix = "_deconv" if deconvolve else ""
    if dep is not None:
        suffix += f"_{dep.lower()}"
    if n_strata > 1:
        suffix += f"_s{n_strata}"
    if gas_cube_path is not None:
        suffix += "_cal"
    if psd_floor == "powerlaw":
        suffix += "_ppow"
    if floor_snr != 0.01:
        suffix += f"_fsnr{floor_snr:g}"
    if deposit_noise == "compound_poisson":
        suffix += "_cpn"
    return suffix


def get_denoised_cube(
    run_dir: Path, output_num: int, n_eff: float = N_EFF, method: str = "wiener",
    dep: str | None = "NGP", deconvolve: bool = False, n_strata: int = 3,
    gas_cube_path: str | None = None, psd_floor: str = "flat",
    floor_snr: float = 0.01, deposit_noise: str = "poisson",
    force: bool = False, save_filter: bool = True,
) -> np.ndarray:
    run_dir = Path(run_dir)
    prefix = "trac"
    tracer_cube_file = run_dir / f"{prefix}_{output_num:05d}.cube"
    suffix = _denoised_suffix(dep, deconvolve, n_strata, gas_cube_path, psd_floor, floor_snr, deposit_noise)
    denoised_file = run_dir / f"denoised_{method}{suffix}_{output_num:05d}.cube"
    if denoised_file.exists() and not force:
        return read_cube_fortran(denoised_file)
    if denoised_file.exists():
        denoised_file.unlink()
    if not tracer_cube_file.exists():
        raise FileNotFoundError(f"Tracer cube not found: {tracer_cube_file}")

    cmd = [
        sys.executable,
        str(DENOISE_SCRIPT),
        "--input", str(tracer_cube_file),
        "--output", str(denoised_file),
        "--n-eff", str(n_eff),
        "--method", method,
        "--psd-floor", psd_floor,
        "--floor-snr", str(floor_snr),
        "--deposit-noise", deposit_noise,
        "-v",
    ]
    if dep is not None:
        cmd += ["--dep", dep]
    if deconvolve:
        cmd += ["--deconvolve"]
    if n_strata > 1:
        cmd += ["--n-strata", str(n_strata)]
    if gas_cube_path is not None:
        cmd += ["--gas-cube", str(gas_cube_path)]
    if save_filter:
        cmd += ["--save-filter"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(RUNDIR_TOP))
    if result.returncode != 0:
        raise RuntimeError(f"denoise_cube failed: {result.stderr}\n{result.stdout}")
    print(result.stdout)
    return read_cube_fortran(denoised_file)


def get_denoised_gas_cube(
    run_dir: Path, output_num: int, n_eff: float = N_EFF, method: str = "wiener",
    n_strata: int = 3, psd_floor: str = "flat", floor_snr: float = 0.01,
    force: bool = False,
) -> np.ndarray:
    """Run denoise_cube on the gas cube; gas has no deposition, so no --dep/--gas-cube."""
    run_dir = Path(run_dir)
    gas_file = run_dir / f"gas_{output_num:05d}.cube"
    suffix = f"_s{n_strata}" if n_strata > 1 else ""
    denoised_file = run_dir / f"denoised_gas_{method}{suffix}_{output_num:05d}.cube"
    if denoised_file.exists() and not force:
        return read_cube_fortran(denoised_file)
    if denoised_file.exists():
        denoised_file.unlink()
    if not gas_file.exists():
        raise FileNotFoundError(f"Gas cube not found: {gas_file}")

    cmd = [
        sys.executable,
        str(DENOISE_SCRIPT),
        "--input", str(gas_file),
        "--output", str(denoised_file),
        "--n-eff", str(n_eff),
        "--method", method,
        "--psd-floor", psd_floor,
        "--floor-snr", str(floor_snr),
        "-v",
    ]
    if n_strata > 1:
        cmd += ["--n-strata", str(n_strata)]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(RUNDIR_TOP))
    if result.returncode != 0:
        raise RuntimeError(f"denoise_cube (gas) failed: {result.stderr}\n{result.stdout}")
    print(result.stdout)
    return read_cube_fortran(denoised_file)


def get_wiener_filter(
    run_dir: Path, output_num: int, method: str = "wiener",
    dep: str | None = "NGP", deconvolve: bool = False, n_strata: int = 3,
    gas_cube_path: str | None = None, psd_floor: str = "flat",
    floor_snr: float = 0.01, deposit_noise: str = "poisson",
) -> np.ndarray:
    """Load the Wiener filter W(k) saved by denoise_cube --save-filter (same transfer as tracer)."""
    run_dir = Path(run_dir)
    suffix = _denoised_suffix(dep, deconvolve, n_strata, gas_cube_path, psd_floor, floor_snr, deposit_noise)
    filter_path = run_dir / f"denoised_{method}{suffix}_{output_num:05d}_filter.npy"
    if not filter_path.exists():
        raise FileNotFoundError(
            f"Wiener filter not found: {filter_path}. "
            "Run denoising with save_filter=True (default) to create it."
        )
    return np.load(filter_path)


def apply_wiener_to_gas(gas: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Apply the same Fourier-space Wiener filter W(k) to gas (real-space). Preserves mean (DC=1)."""
    from numpy.fft import fftn, ifftn
    gas_hat = fftn(gas)
    filtered = np.real(ifftn(W * gas_hat))
    return filtered


def column_density(cube: np.ndarray) -> np.ndarray:
    """Integrate density along z; multiply by dz = 1/nz for physical column density."""
    nz = cube.shape[2]
    return np.sum(cube, axis=2) * (1.0 / nz)


def mid_slice(cube: np.ndarray) -> np.ndarray:
    return cube[:, :, cube.shape[2] // 2]


def compute_power_spectrum(cube: np.ndarray, box_size: float = 1.0):
    """Spherically averaged 3D power spectrum P(k) from density cube (unit mean → density contrast)."""
    nx, ny, nz = cube.shape
    mean_density = np.mean(cube)
    if mean_density <= 0:
        return np.array([]), np.array([])
    delta = cube / mean_density - 1.0
    delta_k = np.fft.fftn(delta)
    power_3d = np.abs(delta_k) ** 2 / (nx * ny * nz) ** 2
    kx = np.fft.fftfreq(nx, d=box_size / nx) * 2 * np.pi
    ky = np.fft.fftfreq(ny, d=box_size / ny) * 2 * np.pi
    kz = np.fft.fftfreq(nz, d=box_size / nz) * 2 * np.pi
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    kmag = np.sqrt(KX**2 + KY**2 + KZ**2)
    kmag_flat = kmag.flatten()
    power_flat = power_3d.flatten()
    nonzero_mask = kmag_flat > 0
    kmag_flat = kmag_flat[nonzero_mask]
    power_flat = power_flat[nonzero_mask]
    k_min, k_max = np.min(kmag_flat), np.max(kmag_flat)
    n_bins = min(100, int((nx * ny * nz) ** (1 / 3) / 2))
    k_edges = np.logspace(np.log10(k_min), np.log10(k_max), n_bins + 1)
    k_bins = np.zeros(n_bins)
    power_spectrum = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (kmag_flat >= k_edges[i]) & (kmag_flat < k_edges[i + 1])
        if np.any(mask):
            k_bins[i] = np.sqrt(k_edges[i] * k_edges[i + 1])
            power_spectrum[i] = np.mean(power_flat[mask])
        else:
            k_bins[i] = np.sqrt(k_edges[i] * k_edges[i + 1])
            power_spectrum[i] = 0
    valid = power_spectrum > 0
    return k_bins[valid], power_spectrum[valid]


def make_panel(ax, data, title, log=True, vmin=None, vmax=None, cmap="viridis"):
    if log:
        data = np.maximum(data, 1e-30)
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        from matplotlib.colors import Normalize
        norm = Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(data.T, origin="lower", aspect="equal", cmap=cmap, norm=norm)
    ax.set_title(title, fontsize=11)
    plt.colorbar(im, ax=ax, shrink=0.7)
    return im


def plot_joint_histogram(ax, gas_flat, tracer_flat, title, xlabel="log\u2081\u2080(gas density)", ylabel="log\u2081\u2080(tracer density)", n_bins=50, cmap="Oranges", xlim=None, ylim=None):
    """2D joint histogram of gas vs tracer density in log-log, with 1:1 line and metrics."""
    mask = (gas_flat > 0) & (tracer_flat > 0)
    x = np.log10(gas_flat[mask])
    y = np.log10(tracer_flat[mask])
    if x.size == 0:
        ax.set_title(title)
        return
    if xlim is None:
        xlim = (np.percentile(x, 0.5), np.percentile(x, 99.5))
    if ylim is None:
        ylim = (np.percentile(y, 0.5), np.percentile(y, 99.5))
    x_edges = np.linspace(xlim[0], xlim[1], n_bins + 1)
    y_edges = np.linspace(ylim[0], ylim[1], n_bins + 1)
    hist, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    hist_plot = np.ma.masked_where(hist.T == 0, hist.T)
    im = ax.pcolormesh(x_edges, y_edges, hist_plot, cmap=cmap, shading="flat", norm=mcolors.LogNorm(vmin=1, vmax=max(hist_plot.max(), 10)))
    ax.plot(xlim, ylim, "k-", lw=1.5, alpha=0.5, label="1:1")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.tick_params(labelsize=9)

    scatter = np.std(y - x)
    corr = np.corrcoef(x, y)[0, 1]
    ax.text(
        0.03, 0.97,
        f"scatter = {scatter:.3f} dex\nr = {corr:.4f}",
        transform=ax.transAxes, fontsize=9, verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )
    return im


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Plot gas, tracer, and denoised column/slice maps.")
    ap.add_argument("--method", default="log_wiener",
                    choices=["wiener", "log_wiener", "laplace", "gaussian_snr"],
                    help="Denoising method: log_wiener (default, best for turbulence), "
                         "wiener (linear-space), or laplace (slowest, most rigorous)")
    ap.add_argument("--deconvolve", action="store_true",
                    help="Apply Wiener deconvolution of the deposition window after denoising")
    ap.add_argument("--n-strata", type=int, default=3, dest="n_strata",
                    help="Number of density strata for heteroscedastic noise "
                         "(3 = default stratified, 1 = original single-level)")
    ap.add_argument("--calibrate", action="store_true",
                    help="Enable cross-power spectrum calibration against the gas "
                         "reference cube (adjusts Fourier amplitudes to match gas P(k))")
    ap.add_argument("--psd-floor", default="flat", dest="psd_floor",
                    choices=["powerlaw", "flat"],
                    help="Signal PSD regularization: 'flat' (default) uses noise*floor_snr; "
                         "'powerlaw' extrapolates from high-SNR low-k bins")
    ap.add_argument("--floor-snr", type=float, default=0.01, dest="floor_snr",
                    help="Minimum signal-to-noise ratio for the 'flat' PSD floor "
                         "(default 0.01)")
    ap.add_argument("--dep", default="NGP", choices=["NGP", "CIC", "TSC", "PCS"],
                    help="Deposition scheme for tracer cubes and denoising "
                         "(default NGP). Only applies to mini-ramses runs; "
                         "RAMSES runs use their native deposition.")
    ap.add_argument("--deposit-noise", default="poisson", dest="deposit_noise",
                    choices=["poisson", "compound_poisson"],
                    help="Deposit noise model: 'poisson' (default) uses 1/lambda "
                         "variance with deposition window shaping; "
                         "'compound_poisson' uses the deposit weight PDF to "
                         "compute reduced variance alpha/lambda (white noise)")
    ap.add_argument("--force", action="store_true",
                    help="Re-run denoising even if cached cubes exist")
    args = ap.parse_args()

    runs = [
        {"dir": RUNDIR / "mc_tracer_ramses", "label": "MC tracer (RAMSES)",
         "is_ramses": True},
        {"dir": RUNDIR / "mc_tracer_cic", "label": "Ito-MC tracer (CIC)",
         "is_ramses": False},
        {"dir": RUNDIR / "sgs_tracer_cic_eq0", "label": "SGS tracer CIC (eq0)",
         "is_ramses": False},
        {"dir": RUNDIR / "tracer_cic", "label": "Tracer CIC",
         "is_ramses": False},
    ]

    # Deposition: RAMSES runs use native (no --dep); mini-ramses runs use --dep
    dep_for_run = lambda r: None if r["is_ramses"] else args.dep

    print("Loading gas, tracer, denoised tracer, and gas filtered with tracer Wiener ...")
    gas_cubes, tracer_cubes, denoised_cubes, gas_filtered_cubes = [], [], [], []
    for r in runs:
        d = Path(r["dir"])
        out_num = r.get("output_num", OUTPUT_NUM)
        dep = dep_for_run(r)
        print(f"  {d.name} ...")
        gas = get_gas_cube(d, out_num, NX, r["is_ramses"])
        tracer = get_tracer_cube(d, out_num, NX, r["is_ramses"], dep=dep or "NGP")
        gas_cube_path = str(d / f"gas_{out_num:05d}.cube") if args.calibrate else None
        denoised_cubes.append(get_denoised_cube(
            d, out_num, N_EFF, method=args.method,
            dep=dep, deconvolve=args.deconvolve,
            n_strata=args.n_strata, gas_cube_path=gas_cube_path,
            psd_floor=args.psd_floor, floor_snr=args.floor_snr,
            deposit_noise=args.deposit_noise, force=args.force,
            save_filter=True,
        ))
        try:
            W = get_wiener_filter(
                d, out_num, method=args.method,
                dep=dep, deconvolve=args.deconvolve, n_strata=args.n_strata,
                gas_cube_path=gas_cube_path, psd_floor=args.psd_floor,
                floor_snr=args.floor_snr, deposit_noise=args.deposit_noise,
            )
        except FileNotFoundError:
            print(f"  Wiener filter not found; re-running denoiser with --save-filter ...")
            get_denoised_cube(
                d, out_num, N_EFF, method=args.method,
                dep=dep, deconvolve=args.deconvolve,
                n_strata=args.n_strata, gas_cube_path=gas_cube_path,
                psd_floor=args.psd_floor, floor_snr=args.floor_snr,
                deposit_noise=args.deposit_noise, force=True,
                save_filter=True,
            )
            W = get_wiener_filter(
                d, out_num, method=args.method,
                dep=dep, deconvolve=args.deconvolve, n_strata=args.n_strata,
                gas_cube_path=gas_cube_path, psd_floor=args.psd_floor,
                floor_snr=args.floor_snr, deposit_noise=args.deposit_noise,
            )
        gas_filtered = apply_wiener_to_gas(gas, W)
        gas_filtered_cubes.append(gas_filtered)
        # Renormalize each cube so mean = 1 for comparable scales
        mean_gas = np.mean(gas)
        mean_tracer = np.mean(tracer)
        mean_denoised = np.mean(denoised_cubes[-1])
        mean_gas_filt = np.mean(gas_filtered)
        if mean_gas > 0:
            gas = gas / mean_gas
        if mean_tracer > 0:
            tracer = tracer / mean_tracer
        if mean_denoised > 0:
            denoised_cubes[-1] = denoised_cubes[-1] / mean_denoised
        if mean_gas_filt > 0:
            gas_filtered_cubes[-1] = gas_filtered_cubes[-1] / mean_gas_filt
        gas_cubes.append(gas)
        tracer_cubes.append(tracer)

    gas_col = [column_density(c) for c in gas_cubes]
    tracer_col = [column_density(c) for c in tracer_cubes]
    denoised_col = [column_density(c) for c in denoised_cubes]
    gas_filtered_col = [column_density(c) for c in gas_filtered_cubes]

    all_gas_col = np.concatenate([g[g > 0].ravel() for g in gas_col])
    log_gas_col = np.log10(all_gas_col)
    mean_log = np.mean(log_gas_col)
    std_log = np.std(log_gas_col)
    vmin_col = 10 ** (mean_log - LOG_SIGMA_RANGE * std_log)
    vmax_col = 10 ** (mean_log + LOG_SIGMA_RANGE * std_log)

    n_runs = len(runs)
    fig1, axes1 = plt.subplots(n_runs, 3, figsize=(14, 3 * n_runs))
    for row, (r, g, t, d) in enumerate(zip(runs, gas_col, tracer_col, denoised_col)):
        make_panel(axes1[row, 0], g, "Gas column density", vmin=vmin_col, vmax=vmax_col)
        make_panel(axes1[row, 1], t, "Tracer column density", vmin=vmin_col, vmax=vmax_col)
        make_panel(axes1[row, 2], d, "Denoised column density", vmin=vmin_col, vmax=vmax_col)
        axes1[row, 0].set_ylabel(r["label"], fontsize=12)
    fig1.suptitle("Column density (integrated along z)", fontsize=14)
    fig1.tight_layout()
    out1 = RUNDIR / "denoised_column_density.png"
    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"Saved {out1}")
    plt.close(fig1)

    gas_sl = [mid_slice(c) for c in gas_cubes]
    tracer_sl = [mid_slice(c) for c in tracer_cubes]
    denoised_sl = [mid_slice(c) for c in denoised_cubes]
    all_gas_sl = np.concatenate([g[g > 0].ravel() for g in gas_sl])
    log_gas_sl = np.log10(all_gas_sl)
    mean_log_sl = np.mean(log_gas_sl)
    std_log_sl = np.std(log_gas_sl)
    vmin_sl = 10 ** (mean_log_sl - LOG_SIGMA_RANGE * std_log_sl)
    vmax_sl = 10 ** (mean_log_sl + LOG_SIGMA_RANGE * std_log_sl)

    fig2, axes2 = plt.subplots(n_runs, 3, figsize=(14, 3 * n_runs))
    for row, (r, g, t, d) in enumerate(zip(runs, gas_sl, tracer_sl, denoised_sl)):
        make_panel(axes2[row, 0], g, "Gas density (slice)", vmin=vmin_sl, vmax=vmax_sl)
        make_panel(axes2[row, 1], t, "Tracer density (slice)", vmin=vmin_sl, vmax=vmax_sl)
        make_panel(axes2[row, 2], d, "Denoised density (slice)", vmin=vmin_sl, vmax=vmax_sl)
        axes2[row, 0].set_ylabel(r["label"], fontsize=12)
    fig2.suptitle("Mid-plane density slice (z = L_z/2)", fontsize=14)
    fig2.tight_layout()
    out2 = RUNDIR / "denoised_density_slice.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved {out2}")
    plt.close(fig2)

    # --- Figure 3: Joint histograms gas vs tracer (noisy | tracer denoised | both denoised) ---
    fig3, axes3 = plt.subplots(n_runs, 3, figsize=(14, 5 * n_runs))
    if n_runs == 1:
        axes3 = axes3.reshape(1, -1)
    all_gas = np.concatenate([g.flatten() for g in gas_cubes])
    all_tracer = np.concatenate([t.flatten() for t in tracer_cubes])
    all_denoised = np.concatenate([d.flatten() for d in denoised_cubes])
    all_gas_filt = np.concatenate([g.flatten() for g in gas_filtered_cubes])
    log_g = np.log10(all_gas[all_gas > 0])
    log_t = np.log10(all_tracer[all_tracer > 0])
    log_d = np.log10(all_denoised[all_denoised > 0])
    log_gf = np.log10(all_gas_filt[all_gas_filt > 0])
    lim_lo = min(log_g.min(), log_t.min(), log_d.min(), log_gf.min())
    lim_hi = max(log_g.max(), log_t.max(), log_d.max(), log_gf.max())
    xlim = ylim = (lim_lo, lim_hi)

    for row, (r, gas, tracer, denoised, gas_filt) in enumerate(zip(runs, gas_cubes, tracer_cubes, denoised_cubes, gas_filtered_cubes)):
        g_flat = gas.flatten()
        t_flat = tracer.flatten()
        d_flat = denoised.flatten()
        gf_flat = gas_filt.flatten()
        plot_joint_histogram(
            axes3[row, 0], g_flat, t_flat,
            f"{r['label']} — gas vs tracer (noisy)",
            xlim=xlim, ylim=ylim,
        )
        plot_joint_histogram(
            axes3[row, 1], g_flat, d_flat,
            f"{r['label']} — gas vs tracer (tracer denoised)",
            xlim=xlim, ylim=ylim,
        )
        plot_joint_histogram(
            axes3[row, 2], gf_flat, d_flat,
            f"{r['label']} — gas vs tracer (both denoised)",
            xlim=xlim, ylim=ylim,
        )
    fig3.suptitle("Joint histogram: gas density vs particle density (cell-by-cell)", fontsize=14)
    fig3.tight_layout()
    out3 = RUNDIR / "denoised_joint_histogram.png"
    fig3.savefig(out3, dpi=150, bbox_inches="tight")
    print(f"Saved {out3}")
    plt.close(fig3)

    # --- Figure 4: Power spectra before (left) and after (right) denoising ---
    fig4, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(12, 5))
    box_size = 1.0

    # Gas reference P(k) — average over all runs for a single reference line
    all_k_gas, all_p_gas = [], []
    for gas in gas_cubes:
        k_g, p_g = compute_power_spectrum(gas, box_size=box_size)
        if k_g.size > 0:
            all_k_gas.append(k_g)
            all_p_gas.append(p_g)
    if all_k_gas:
        k_ref = all_k_gas[0]
        p_ref = np.mean(all_p_gas, axis=0)
        ax_before.loglog(k_ref, p_ref, "k--", lw=2, alpha=0.7, label="Gas (ref)")
        ax_after.loglog(k_ref, p_ref, "k--", lw=2, alpha=0.7, label="Gas (ref)")

    for r, tracer, denoised in zip(runs, tracer_cubes, denoised_cubes):
        k_t, p_t = compute_power_spectrum(tracer, box_size=box_size)
        k_d, p_d = compute_power_spectrum(denoised, box_size=box_size)
        if k_t.size > 0:
            ax_before.loglog(k_t, p_t, "-", label=r["label"], alpha=0.9)
        if k_d.size > 0:
            ax_after.loglog(k_d, p_d, "-", label=r["label"], alpha=0.9)
    ax_before.set_xlabel("k")
    ax_before.set_ylabel("P(k)")
    ax_before.set_title("Before denoising (tracer)")
    ax_before.legend(loc="best", fontsize=9)
    ax_before.grid(True, alpha=0.3)
    ax_after.set_xlabel("k")
    ax_after.set_ylabel("P(k)")
    ax_after.set_title("After denoising")
    ax_after.legend(loc="best", fontsize=9)
    ax_after.grid(True, alpha=0.3)
    fig4.suptitle("Tracer power spectrum P(k)", fontsize=14)
    fig4.tight_layout()
    out4 = RUNDIR / "denoised_power_spectrum.png"
    fig4.savefig(out4, dpi=150, bbox_inches="tight")
    print(f"Saved {out4}")
    plt.close(fig4)

    # --- Figure 5: Tracer/gas column-density ratio (raw | tracer denoised | same W on gas) ---
    def safe_ratio(num: np.ndarray, denom: np.ndarray, eps: float = 1e-30) -> np.ndarray:
        out = np.where(denom > eps, num / denom, np.nan)
        return np.clip(out, 0.5, 2.0)

    ratio_vmin, ratio_vmax = 0.5, 2.0
    fig5, axes5 = plt.subplots(n_runs, 3, figsize=(12, 3 * n_runs))
    if n_runs == 1:
        axes5 = axes5.reshape(1, -1)
    for row, (r, tc, gc, dc, gfc) in enumerate(zip(runs, tracer_col, gas_col, denoised_col, gas_filtered_col)):
        r_neither = safe_ratio(tc, gc)
        r_tracer_only = safe_ratio(dc, gc)
        r_same_W = safe_ratio(dc, gfc)
        make_panel(axes5[row, 0], r_neither, "Tracer/gas (no denoising)", log=False,
                   vmin=ratio_vmin, vmax=ratio_vmax, cmap="RdBu_r")
        make_panel(axes5[row, 1], r_tracer_only, "Tracer/gas (tracer denoised)", log=False,
                   vmin=ratio_vmin, vmax=ratio_vmax, cmap="RdBu_r")
        make_panel(axes5[row, 2], r_same_W, "Tracer/gas (same W applied to gas)", log=False,
                   vmin=ratio_vmin, vmax=ratio_vmax, cmap="RdBu_r")
        axes5[row, 0].set_ylabel(r["label"], fontsize=12)
    fig5.suptitle("Column-density ratio: tracer / gas (1 = same scaling)", fontsize=14)
    fig5.tight_layout()
    out5 = RUNDIR / "denoised_ratio_column.png"
    fig5.savefig(out5, dpi=150, bbox_inches="tight")
    print(f"Saved {out5}")
    plt.close(fig5)

    # --- Figure 6: Gas column density before vs after applying tracer Wiener W(k) ---
    all_gas_filt_col = np.concatenate([g[g > 0].ravel() for g in gas_filtered_col])
    all_gas_both = np.concatenate([all_gas_col, all_gas_filt_col])
    log_gas_both = np.log10(all_gas_both)
    mean_log_both = np.mean(log_gas_both)
    std_log_both = np.std(log_gas_both)
    vmin_gas_col = 10 ** (mean_log_both - LOG_SIGMA_RANGE * std_log_both)
    vmax_gas_col = 10 ** (mean_log_both + LOG_SIGMA_RANGE * std_log_both)

    fig6, axes6 = plt.subplots(n_runs, 2, figsize=(10, 3 * n_runs))
    if n_runs == 1:
        axes6 = axes6.reshape(1, -1)
    for row, (r, gc, gfc) in enumerate(zip(runs, gas_col, gas_filtered_col)):
        make_panel(axes6[row, 0], gc, "Gas column density (raw)", vmin=vmin_gas_col, vmax=vmax_gas_col)
        make_panel(axes6[row, 1], gfc, "Gas column density (tracer W applied)", vmin=vmin_gas_col, vmax=vmax_gas_col)
        axes6[row, 0].set_ylabel(r["label"], fontsize=12)
    fig6.suptitle("Gas column density: before vs after applying tracer Wiener filter W(k)", fontsize=14)
    fig6.tight_layout()
    out6 = RUNDIR / "denoised_gas_column_before_after.png"
    fig6.savefig(out6, dpi=150, bbox_inches="tight")
    print(f"Saved {out6}")
    plt.close(fig6)

    # --- Figure 7: 1D PDF of log10(tracer/gas) cell-centered density (3 scenarios) ---
    def log_ratio_pdf(ax, gas_flat, tracer_flat, denoised_tracer_flat, gas_filtered_flat, title, n_bins=80, xlim=(-2, 2)):
        """Plot PDF of log10(tracer/gas) for no denoising, tracer denoised, same W on gas."""
        eps = 1e-30
        edges = np.linspace(xlim[0], xlim[1], n_bins + 1)
        for label, num, denom in [
            ("no denoising", tracer_flat, gas_flat),
            ("tracer denoised", denoised_tracer_flat, gas_flat),
            ("same W on gas", denoised_tracer_flat, gas_filtered_flat),
        ]:
            mask = (num > eps) & (denom > eps)
            log_rat = np.log10(num[mask] / denom[mask])
            hist, _ = np.histogram(log_rat, bins=edges, density=True)
            cents = 0.5 * (edges[:-1] + edges[1:])
            ax.plot(cents, hist, "-", label=label, alpha=0.9)
        ax.axvline(0, color="k", ls="--", alpha=0.5)
        ax.set_xlim(xlim)
        ax.set_ylim(bottom=0)
        ax.set_xlabel(r"log₁₀(tracer / gas)")
        ax.set_ylabel("PDF")
        ax.set_title(title, fontsize=12)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

    fig7, axes7 = plt.subplots(n_runs, 1, figsize=(7, 4 * n_runs))
    if n_runs == 1:
        axes7 = [axes7]
    for row, (r, gas, tracer, denoised, gas_filt) in enumerate(zip(runs, gas_cubes, tracer_cubes, denoised_cubes, gas_filtered_cubes)):
        log_ratio_pdf(
            axes7[row],
            gas.flatten(), tracer.flatten(), denoised.flatten(), gas_filt.flatten(),
            r["label"],
        )
    fig7.suptitle("Cell-centered density ratio log₁₀(tracer/gas): 1D PDF", fontsize=14)
    fig7.tight_layout()
    out7 = RUNDIR / "denoised_log_ratio_pdf.png"
    fig7.savefig(out7, dpi=150, bbox_inches="tight")
    print(f"Saved {out7}")
    plt.close(fig7)


if __name__ == "__main__":
    main()
