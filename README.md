# Particle analysis

Plot gas, raw tracer, and denoised tracer density from mini-ramses (or RAMSES) runs. Self-contained: denoising and plotting logic plus required utils live in this repo.

## Dependencies (Python)

Install with:

```bash
pip install -r requirements.txt
```

Requires: **numpy**, **matplotlib**, **scipy**, **astropy** (for `miniramses` when building gas cubes from AMR), and **colorcet** (for `make_dust_alpha_gas_video.py`).

## Build part2cube

The script builds tracer density cubes from particle outputs using the Fortran `part2cube` tool. Build it once:

```bash
cd utils/f90 && make && cd ../..
```

You need `gfortran`. The binary `utils/f90/part2cube` is used by `plot_denoised_tracer.py`.

## Layout

- **plot_denoised_tracer.py** — main script: loads or builds gas/tracer cubes, runs denoising, produces figures.
- **denoise_cube.py** — Wiener / Gaussian SNR denoising; called as a subprocess and used for the saved Wiener filter.
- **utils/py/miniramses.py** — read AMR outputs and build gas density cubes (`rd_cell`, `mk_cube`).
- **utils/f90/part2cube.f90** — build 3D tracer density cubes from particle files (NGP/CIC/TSC/PCS).
- **video_common.py** — shared frame sizing, last-frame log ranges, frame-list writing, and `ffmpeg` encode helpers for the video scripts.
- **column_utils.py** — shared gas and dust column helpers (`get_gas_column`, `get_dust_column`, CIC deposition, projection helpers).
- **dust_projection.py** — raw `dust.*` reader and shared dust LOS moment projections (`Σm`, `Σm a`, optional `Σm a^2`) plus the legacy binned-median path.
- **make_column_density_video.py** — gas + dust column-density frames and MP4.
- **make_dust_alpha_gas_video.py** — same inputs, but gas uses colorcet **isolum** (log column / mean) and dust modulates darkness (alpha); default projection integrates along **x** (`--axis x`).
- **make_dust_grainsize_gas_video.py** — gas uses colorcet **CET_I3**; dust alpha follows dust column density; default dust hue shows the direct LOS mass-weighted mean-size deviation `log10(a_mean_los / a_ref_last)` from the stored particle `size` field, with `a_ref_last` taken from the last snapshot global dust-mass-weighted mean. The older 16-bin median-in-bin surrogate remains available via `--field-mode legacy-binned` (default **`--nx 128`** for 128³-style maps; default projection **x**).

## Column-density videos

From this directory, with `ffmpeg` on your `PATH`:

```bash
# Side-by-side gas and dust (log scale, inferno), integrate along z
python make_column_density_video.py --run-dir /path/to/run --start 1 --end 50

# Gas hue (isolum) + dust as shade; default integrate along x
python make_dust_alpha_gas_video.py --run-dir /path/to/run --start 1 --end 50

# Gas CET_I3 + dust grain-size color: default is direct LOS mean-size deviation
python make_dust_grainsize_gas_video.py --run-dir /path/to/run --start 1 --end 50

# Absolute LOS mean grain size instead of deviation
python make_dust_grainsize_gas_video.py --run-dir /path/to/run --start 1 --end 50 --field-mode mean-abs

# Legacy 16-bin median-in-bin surrogate (log-size bins by default)
python make_dust_grainsize_gas_video.py --run-dir /path/to/run --start 1 --end 50 --field-mode legacy-binned

# Same grain-size movie but line-of-sight along z (matches classic xy maps)
python make_dust_grainsize_gas_video.py --run-dir /path/to/run --start 1 --end 50 --axis z
```

Re-encode existing frames only:

```bash
python make_dust_alpha_gas_video.py --ffmpeg-only --frames-dir /path/to/run/frames_dust_alpha
python make_dust_grainsize_gas_video.py --ffmpeg-only --frames-dir /path/to/run/frames_dust_grainsize
```

## Grain-size interpretation

For the current `mini-ramses-dev` grafic dust IC, the size-spectrum path assigns particle sizes through a shuffled map `size_index = perm(idp)` in `pm/input_part_grafic.f90`. That means contiguous particle IDs are **not** contiguous grain-size bins. The default grain-size movie therefore works from the stored particle `size` field directly:

- project `Σ(m a)` and `Σm` with the same 2D CIC kernel
- sum along the line of sight implicitly during that projection
- divide to get `a_mean_los = Σ(m a) / Σm`

The legacy binned mode is retained for comparison or workflows that intentionally want a 16-bin surrogate, but it is not the physically correct default for shuffled grafic size spectra.

## Run directories

By default the script expects run directories next to it (e.g. `mc_tracer_ramses/`, `mc_tracer_cic/`, …). Each run dir should contain:

- **output_00003/** (or `output_XXXXX/`) — snapshot/particle output.
- Optionally **gas_00003.cube** and **trac_00003.cube**; if missing, the script builds them (gas via `miniramses`, tracer via `part2cube`).

Edit the `runs` list in `plot_denoised_tracer.py` to point at your run dirs and labels.

## RAMSES runs

For RAMSES (full code) runs you can either:

1. Put **ramses-pic** under this repo (`particle_analysis/ramses-pic/`) so the script can use its `part2cube` and `ramses_io` for gas/tracer cubes, or  
2. Pre-build **gas_XXXXX.cube** and **trac_XXXXX.cube** in each run dir so the script only loads them.

## Usage

```bash
# Default: log_wiener, n_strata=3, dep=NGP
python plot_denoised_tracer.py

# CIC deposition, force re-run denoising
python plot_denoised_tracer.py --dep CIC --n-strata 3 --force
```

Figures are written next to the script (e.g. `denoised_column_density.png`, `denoised_ratio_column.png`, …).
