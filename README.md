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
- **dust_hd23.py** — published HD23 Equation 18/25 distributions, independent log-size quadrature, active/passive partitioning, and distinct number/mass/area family weights.
- **validate_hd23_deposition.py** — fast Stage 6/7 analytic, normalization, CIC/TSC, family-mass, and rank-count reconstruction checks for massless GC outputs.
- **dust_charge_equilibrium.py** — Stage 9 float64 equilibrium-charge reference, twelve-size-knot float32 Coulomb-moment table generator for continuous grain radii, Gaussian-cgs code-unit conversion, and Epstein+Coulomb charging-timescale audit.
- **validate_dust_charge_equilibrium.py** — focused Stage 9 distribution, float32-table, charge-zero, and timescale checks.
- **dust_yldh_reference.py** — offline float64 YLD04 `|n|=1` reference and compact float32 `yldh04_balanced_gyro` table generator.
- **validate_dust_yldh_reference.py** — focused resonance, PSD, interpolation, precision, and strict-envelope checks for that reduced table.
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

## HD23 and surface-area validation

Run the dependency-light analytic and synthetic CIC/TSC checks:

```bash
python validate_hd23_deposition.py
```

Validate a Stage 6/7 output, or compare particle-output reconstructions written
with different MPI rank counts:

```bash
python validate_hd23_deposition.py \
  --output-dir /path/to/run_1rank/output_00010 \
  --output-dir /path/to/run_8rank/output_00010 \
  --nx 64 --scheme both --field area2 --report hd23_validation.json
```

The output `info.txt` should carry
`dust_hd23_revision = HD23-2023-eq18-eq25-v1`, the resolved active size
bounds, `dust_grain_bulk_density`, and either `dust_radius_code_to_cm` or the
usual `unit_d` and `unit_l`. The validator streams each rank file in bounded
chunks and reconstructs the log-sampled broad-component weights with the
required Jacobian. It also closes the exported C2/C3, area, macro-mass, and
total dust-to-gas normalizations. Actual post-flush GPU-field rank invariance
is checked separately from `dust_area_checksum` in the mini-RAMSES Stage 6/7
MPI gate; a CPU reconstruction is not a substitute for that check.

- number weight is proportional to `a f(a) dln(a)`
- mass weight is proportional to `a^4 f(a) dln(a)`
- deposited area weight is proportional to `a^3 f(a) dln(a)`

Here `f(a) = (1/n_H) dn/da`. Keep these three weights distinct.

Regular GC dust outputs do not store particle mass: their scalar block after
velocity is `size`. The legacy `miniramses.rd_part` and `part2cube` readers
interpret that block as mass and therefore produce a physically wrong dust
column. `column_utils.get_dust_column` now rejects such outputs; use the HD23
reconstruction path explicitly.

## Reduced YLD04 gyroresonance reference

Generate the practical dimensionless float32 table, deterministic
CUDA-Fortran include, and SHA-256 sidecars:

```bash
python dust_yldh_reference.py
python validate_dust_yldh_reference.py
```

The checked-in default table and CUDA-Fortran include carry independent
SHA-256 sidecars. `yldh04_balanced_gyro_audit.json` records their grid,
generation cost, strict support-boundary rejection rate, and direct-reference
interpolation errors.

The named selector contains balanced forward/backward `|n|=1`
gyroresonance from the YLD04 Appendix B3 Alfvén tensor and B4 low-beta fast
tensor. It does not contain TTD, slow modes, high-beta fast modes, imbalanced
turbulence, or dynamic cascade/damping axes, and it remains off for
production. Its axes are `log R`, `log u`, and pitch cosine, with
`R=v/(|Ω|L)` and `u=v/V_A`; grain mass and size cancel at matched `|Ω|`.
The unit equal-mode basis may be multiplied by one finite nonnegative wave
power, while its Alfvén/fast partition, decorrelation kernel, and cutoffs stay
fixed. The audited default uses a compact nonuniform 21×18×21 grid. Local
tensor-product PCHIP interpolation of the float32 lower factor preserves PSD;
support diagnostics remain trilinear. Lookup outside the closed envelope,
with a changed mode model, or across a zero/nonzero resonance-support
boundary is rejected. Exact nodes and faces remain usable.
`YLDH_RUNTIME_PCHIP.md` gives the exact boundary, slope, axis-order, and
float32-rounding contract for the CUDA implementation.

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
