# Particle analysis

Plot gas, raw tracer, and denoised tracer density from mini-ramses (or RAMSES) runs. Self-contained: denoising and plotting logic plus required utils live in this repo.

## Dependencies (Python)

Install with:

```bash
pip install -r requirements.txt
```

Requires: **numpy**, **matplotlib**, **scipy**, **astropy** (for `miniramses` when building gas cubes from AMR).

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
