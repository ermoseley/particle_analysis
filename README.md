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
- **ramses_streaming.py** — bounded readers for all regular `nfile` rank files, including hydro-like payloads and the 12-field six-ray stream.
- **audit_sgs_morphology.py** — fixed-grid production audit for the direct SGS dust diffusivity, diffusion length, projected Itô drift, and common-grid dust/gas morphology.
- **dust_hd23.py** — published HD23 Equation 18/25 distributions, independent log-size quadrature, active/passive partitioning, and distinct number/mass/area family weights.
- **validate_hd23_deposition.py** — fast Stage 6/7 analytic, normalization, CIC/TSC, family-mass, and rank-count reconstruction checks for massless GC outputs.
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
reconstruction path explicitly. At production particle counts, consume
`iter_dust_snapshot_blocks` rather than materializing `read_dust_snapshot`;
both readers honor the exact `nfile` rank-file set declared by `info.txt`.

## SGS transport and morphology audit

Run on a final output, supplying the interval if it cannot be inferred as the
time since zero:

```bash
python audit_sgs_morphology.py /path/to/run/output_00002 \
  --elapsed-time 0.02 --report sgs_morphology.json
```

For an initial/final pair, the default interval is the difference between
their `info.txt` times:

```bash
python audit_sgs_morphology.py \
  /path/to/run/output_00001 /path/to/run/output_00002 \
  --projection-axis z \
  --scratch-dir "${SLURM_TMPDIR:?SLURM_TMPDIR must be set}" \
  --report sgs_morphology.json
```

The script requires the production fixed-level periodic geometry. It evaluates
`kappa_sgs = dx*sqrt((2/3)*E_sgs/rho)`, the per-coordinate, magnetized 2-D,
and full-orbit 3-D RMS diffusion lengths in cell units, and cell-centred
reconstructions of the isotropic and magnetized Itô drifts. The latter uses the
same continuum expression as the mover, but not its particle-local CIC/TSC
gradient gather. It streams octs and particles, keeps the required 3-D float32
fields in temporary disk-backed arrays, and reconstructs massless GC dust with
the output's HD23 macro-mass metadata. Pixel correlation and `P_dg/P_gg`,
`P_dg/sqrt(P_dd*P_gg)` spectra use projected log contrasts on one common grid.
The default resolved range is `1 <= k <= N/4`; `|B| > --b-floor` defines the
projected-drift support, with the default `1e-22` matching the production
mover's `smallc*1e-10`. Distribution moments are exact; reported quantiles use
a deterministic bounded sample.

Each diffusion length freezes the output's local coefficient over the stated
interval; it is an endpoint scale, not an integral along a particle history.
Endpoint morphology is descriptive and cannot isolate SGS diffusion without a
matched `kappa_sgs=0` control. A whole-box 2-D projection may also hide 3-D
washout, and the dust map contains the kinetic active broad-astrodust
population rather than the passive small-grain and PAH reservoirs. No
morphology pass threshold is assumed.

The audit fails closed unless output metadata and the copied namelist select
the production contract: HD23 revision `HD23-2023-eq18-eq25-v1`,
`C_s=0.17`, `equilibrium_sgs=.false.`, active SGS and diffusive kicks, and
`response_markov`. At 512³ its temporary maps peak near 2.125 GiB. Run it as a
separate CPU batch with node-local `--scratch-dir`, not on a login node or
inside the short GPU gate.

Quick dependency-light validation creates its fixture only under `/tmp`:

```bash
python audit_sgs_morphology.py --self-test \
  --report /tmp/sgs_morphology_self_test.json
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
