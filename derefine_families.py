#!/usr/bin/env python3
"""
Quasi-Lagrangian mesh de-refinement of the CIC dust and gas density cubes
(Moseley thesis, Sec. 4.4.2.1), one structure per grain family, size group, and
for all grains together.

The structure is built from CIC particle COUNTS, not mass. Starting from the whole
box, a node (2^L cells on a side) is split into its eight children only if every
child holds at least `threshold` particles; otherwise every cell in the node takes
the node's volume mean. Splitting stops at the native grid, so each final cell holds
at least `threshold` particles, nested exactly, and total mass is conserved. Read
bottom-up this is the thesis rule: merge an oct that has a deficient cell, then test
the merged oct as one cell inside its parent oct.

The dust density of the case and the gas density are both painted with the
structure. Inputs are the CIC cubes written by reduce_cubes_mf (the cache whose
prefix is recorded in py_out/dust/cic/metadata.json) and py_out/{gas,dust/cic}.

Example:
  python3 derefine_families.py /path/to/turb256/output_00009
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

NFAM = 12
DEFAULT_GROUPS = "small=1-4,medium=5-8,large=9-12"
DEFAULT_THRESHOLD = 1.0
METHOD = (
    "Top-down from the whole box: a node is split into its 8 children only if every "
    "child holds at least `threshold` CIC particles; otherwise all cells in the node "
    "take the node's volume mean. Gate = particle count (sum of the case's family "
    "count cubes); painted fields = the case's dust density and gas/rho. Node sums "
    "in float64, outputs float32. Moseley thesis Sec. 4.4.2.1."
)


# ----------------------------------------------------------------------------
# Core: pyramid, structure, painting
# ----------------------------------------------------------------------------
def _up(a):
    """Repeat every element 2x along each axis."""
    return a.repeat(2, 0).repeat(2, 1).repeat(2, 2)


def block_sum(a):
    """float64 sum over disjoint 2x2x2 blocks."""
    m = a.shape[0] // 2
    return a.reshape(m, 2, m, 2, m, 2).sum(axis=(1, 3, 5), dtype=np.float64)


def pyramid(a):
    """Node sums at every scale: sums[L] has side n >> L; sums[0] is `a` itself."""
    n = a.shape[0]
    if a.shape != (n, n, n) or n < 1 or n & (n - 1):
        raise ValueError(f"need a cubic grid with power-of-two side, got {a.shape}")
    sums = [a]
    while sums[-1].shape[0] > 1:
        sums.append(block_sum(sums[-1]))
    return sums


def find_structure(gate_sums, threshold):
    """Boolean masks term[L]: the nodes at scale L that are final cells."""
    lmax = len(gate_sums) - 1
    term = [None] * (lmax + 1)
    reached = np.ones((1, 1, 1), dtype=bool)  # nodes whose ancestors were all split
    for L in range(lmax, -1, -1):
        if L == 0:
            term[0] = reached
            break
        m = gate_sums[L].shape[0]
        children_ok = (gate_sums[L - 1] >= threshold).reshape(m, 2, m, 2, m, 2).all(axis=(1, 3, 5))
        term[L] = reached & ~children_ok
        reached = _up(reached & children_ok)
    return term


def paint(base, term):
    """Final cells take the volume mean of `base`; native cells keep their value."""
    sums = pyramid(base)
    lmax = len(term) - 1
    value = None
    for L in range(lmax, 0, -1):
        mean = sums[L] / float(8**L)
        value = np.where(term[L], mean, 0.0 if value is None else _up(value))
    if value is None:
        return np.asarray(base, dtype=np.float32).copy()
    return np.where(term[0], base, _up(value)).astype(np.float32)


def level_map(term):
    """uint8 cube: L where the base cell sits in a final cell 2^L cells wide."""
    lmax = len(term) - 1
    value = None
    for L in range(lmax, -1, -1):
        up = np.uint8(0) if value is None else _up(value)
        value = np.where(term[L], np.uint8(L), up).astype(np.uint8)
    return value


def derefine(gate, fields, threshold=DEFAULT_THRESHOLD):
    """Return (level, [painted fields], stats) for a count cube and density cubes."""
    gate = np.asarray(gate, dtype=np.float64)
    gate_sums = pyramid(gate)
    term = find_structure(gate_sums, threshold)
    n = gate.shape[0]
    nodes = [int(t.sum()) for t in term]
    min_count = min(float(gate_sums[L][term[L]].min()) for L in range(len(term)) if nodes[L])
    stats = {
        "nodes_by_level": nodes,
        "volume_fraction_by_level": [nodes[L] * 8**L / n**3 for L in range(len(term))],
        "n_final_cells": int(sum(nodes)),
        "min_particles_per_final_cell": min_count,
    }
    return level_map(term), [paint(f, term) for f in fields], stats


def verify_structure(level, gate, threshold):
    """Check a level cube against the rule from the counts alone; return the minimum
    particle count over final cells. Raises AssertionError on any violation."""
    n = level.shape[0]
    gate_sums = pyramid(np.asarray(gate, dtype=np.float64))
    lmax = len(gate_sums) - 1
    assert level.max() <= lmax
    min_count = np.inf
    for L in range(lmax + 1):
        if L == 0:
            final = level == 0
        else:
            m = n >> L
            b = level.reshape(m, 2**L, m, 2**L, m, 2**L)
            blocks_min = b.min(axis=(1, 3, 5))
            final = (b == L).all(axis=(1, 3, 5))
            # A block is either one final cell or holds no cell labelled L.
            assert not ((b == L).any(axis=(1, 3, 5)) & ~final).any(), f"ragged cell at L={L}"
            split = blocks_min < L
            if L >= 1:
                m1 = gate_sums[L - 1].shape[0]
                kids_ok = (gate_sums[L - 1] >= threshold).reshape(
                    m1 // 2, 2, m1 // 2, 2, m1 // 2, 2).all(axis=(1, 3, 5))
                assert kids_ok[split].all(), f"split node with a deficient child at L={L}"
                # A final cell that was not split must have a deficient child.
                assert (~kids_ok[final]).all(), f"final cell that should have split at L={L}"
        if final.any():
            min_count = min(min_count, float(gate_sums[L][final].min()))
    assert min_count >= threshold, f"final cell with {min_count} < {threshold} particles"
    return min_count


def verify_fields(level, dust, gas):
    """Dust and gas are constant within each final cell (read from the written cubes)."""
    n = level.shape[0]
    for L in range(1, int(level.max()) + 1):
        m = n >> L
        full = (level.reshape(m, 2**L, m, 2**L, m, 2**L) == L).all(axis=(1, 3, 5))
        for name, f in (("dust", dust), ("gas", gas)):
            b = f.reshape(m, 2**L, m, 2**L, m, 2**L)
            flat = b.max(axis=(1, 3, 5)) == b.min(axis=(1, 3, 5))
            assert flat[full].all(), f"{name} not constant in L={L} cells"


# ----------------------------------------------------------------------------
# Cases and I/O
# ----------------------------------------------------------------------------
def parse_groups(text):
    groups = {}
    for item in filter(None, text.split(",")):
        name, rng = item.split("=")
        lo, _, hi = rng.partition("-")
        lo, hi = int(lo), int(hi or lo)
        if not 1 <= lo <= hi <= NFAM:
            raise ValueError(f"bad family range in {item!r}")
        groups[name] = list(range(lo, hi + 1))
    return groups


def case_list(groups):
    """name -> family numbers (1-based). Families first, then groups, then all."""
    cases = {f"{f:02d}": [f] for f in range(1, NFAM + 1)}
    for name, fams in groups.items():
        if name in cases or name == "all":
            raise ValueError(f"group name {name!r} is reserved")
        cases[name] = fams
    cases["all"] = list(range(1, NFAM + 1))
    return cases


def file_record(path):
    st = Path(path).stat()
    return {"path": str(path), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def script_commit():
    here = Path(__file__).resolve().parent
    try:
        sha = subprocess.run(["git", "-C", str(here), "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(here), "status", "--porcelain", "--",
                                Path(__file__).name], capture_output=True, text=True,
                               check=True).stdout.strip()
        return sha + ("+modified" if dirty else "")
    except Exception:
        return "unknown"


def load_sum(arrays):
    """float64 sum of memory-mapped cubes (one cube is returned untouched)."""
    if len(arrays) == 1:
        return arrays[0]
    total = np.zeros(arrays[0].shape, dtype=np.float64)
    for a in arrays:
        total += a
    return total


def check_provenance(root, n, log):
    """Cache vs published cubes, and per-family count totals; returns a record."""
    cic = root / "dust" / "cic"
    for marker in (root / "COMPLETE", cic / "COMPLETE"):
        if not marker.exists():
            raise RuntimeError(f"missing {marker}")
    meta = json.loads((cic / "metadata.json").read_text())
    prefix = meta["cache_prefix"]
    cnt = np.load(prefix + "_dust_num_bins.npy", mmap_mode="r")
    rho_bins = np.load(prefix + "_dust_rho_bins.npy", mmap_mode="r")
    cache_gas = np.load(prefix + "_gas_rho.npy", mmap_mode="r")
    gas = np.load(root / "gas" / "rho.npy", mmap_mode="r")
    shape = (NFAM, n, n, n)
    assert cnt.shape == shape and rho_bins.shape == shape and gas.shape == (n, n, n), "bad shapes"
    assert cnt.dtype == np.float32 and rho_bins.dtype == np.float32 and gas.dtype == np.float32
    npart = json.loads(Path(prefix + "_meta.json").read_text())["npart"]
    rec = {"cache_prefix": prefix, "npart": npart}

    def max_abs_diff(a, b):
        return max(float(np.abs(a[i:i + 32].astype(np.float64) - b[i:i + 32]).max())
                   for i in range(0, n, 32))

    rec["gas_max_abs_diff_vs_cache"] = max_abs_diff(gas, cache_gas)
    rec["rho_family_max_abs_diff_vs_cache"] = []
    for b in range(NFAM):
        fam = np.load(cic / f"rho_{b + 1:02d}.npy", mmap_mode="r")
        rec["rho_family_max_abs_diff_vs_cache"].append(max_abs_diff(fam, rho_bins[b]))
    totals = []
    for b in range(NFAM):
        assert float(cnt[b].min()) >= 0.0 and np.isfinite(cnt[b]).all(), f"bad counts family {b + 1}"
        totals.append(float(cnt[b].sum(dtype=np.float64)))
    rec["count_total_by_family"] = totals
    rec["count_total"] = float(sum(totals))
    rec["count_total_rel_diff_vs_npart"] = rec["count_total"] / npart - 1.0
    if abs(rec["count_total_rel_diff_vs_npart"]) > 1e-6:
        raise RuntimeError(f"count totals disagree with npart: {rec}")
    if rec["gas_max_abs_diff_vs_cache"] != 0.0 or any(rec["rho_family_max_abs_diff_vs_cache"]):
        raise RuntimeError(f"published cubes differ from the cache: {rec}")
    log(f"provenance ok: gas and 12 family cubes identical to cache; "
        f"counts total {rec['count_total']:.1f} vs npart {npart} "
        f"(rel {rec['count_total_rel_diff_vs_npart']:.1e})")
    return rec, cnt


def process_case(name, fams, cnt, root, out, threshold):
    n = cnt.shape[1]
    cic = root / "dust" / "cic"
    gas = np.load(root / "gas" / "rho.npy", mmap_mode="r")
    gate = np.zeros((n, n, n), dtype=np.float64)
    for f in fams:
        gate += cnt[f - 1]
    if name == "all":
        dust = np.load(cic / "rho_total.npy", mmap_mode="r")
    else:
        dust = load_sum([np.load(cic / f"rho_{f:02d}.npy", mmap_mode="r") for f in fams])
    level, (dust_out, gas_out), stats = derefine(gate, [dust, gas], threshold)
    np.save(out / f"dust_{name}.npy", dust_out)
    np.save(out / f"gas_{name}.npy", gas_out)
    np.save(out / f"level_{name}.npy", level)
    inputs = {"dust_in": float(np.sum(dust, dtype=np.float64)),
              "gas_in": float(np.sum(gas, dtype=np.float64))}
    return stats, inputs, gate


def verify_case(name, out, gate, inputs, threshold, tol):
    """Re-read the written cubes and check the acceptance criteria."""
    level = np.load(out / f"level_{name}.npy")
    dust = np.load(out / f"dust_{name}.npy", mmap_mode="r")
    gas = np.load(out / f"gas_{name}.npy", mmap_mode="r")
    assert level.dtype == np.uint8 and dust.dtype == np.float32 and gas.dtype == np.float32
    assert dust.shape == gas.shape == level.shape
    min_count = verify_structure(level, gate, threshold)
    verify_fields(level, dust, gas)
    res_d = abs(float(dust.sum(dtype=np.float64)) - inputs["dust_in"]) / inputs["dust_in"]
    res_g = abs(float(gas.sum(dtype=np.float64)) - inputs["gas_in"]) / inputs["gas_in"]
    nonfinite = int(np.sum(~np.isfinite(dust)) + np.sum(~np.isfinite(gas)))
    zero_dust = int(np.sum(np.asarray(dust) <= 0))
    assert res_d <= tol and res_g <= tol, f"mass residual dust {res_d:.2e} gas {res_g:.2e}"
    assert nonfinite == 0 and zero_dust == 0, f"{nonfinite} non-finite, {zero_dust} zero dust cells"
    return {"min_particles_per_final_cell": min_count, "dust_mass_residual": res_d,
            "gas_mass_residual": res_g, "zero_dust_cells": zero_dust, "nonfinite_cells": nonfinite}


def compare_reference(refdir, out, stats_by_case, cnt_n, log):
    """Cross-check family cubes against the paper's derefine_cubes output."""
    refdir = Path(refdir)
    ref_meta = json.loads(next(refdir.glob("*_cell_derefine_meta.json")).read_text())
    prefix = next(refdir.glob("*_cell_derefined_dust_bin00.npy")).name.replace("dust_bin00.npy", "")
    result = {"reference_dir": str(refdir), "families": {}}
    for b in range(NFAM):
        name = f"{b + 1:02d}"
        row = {}
        for kind in ("dust", "gas"):
            mine = np.load(out / f"{kind}_{name}.npy", mmap_mode="r")
            ref = np.load(refdir / f"{prefix}{kind}_bin{b:02d}.npy", mmap_mode="r")
            assert mine.shape == ref.shape
            worst = 0.0
            for i in range(0, mine.shape[0], 32):
                a = mine[i:i + 32].astype(np.float64)
                r = ref[i:i + 32].astype(np.float64)
                worst = max(worst, float((np.abs(a - r) / np.abs(r)).max()))
            row[f"{kind}_max_rel_diff"] = worst
        ref_nodes = {k: v for k, v in ref_meta["bins"][b]["nodes"].items()}
        mine_nodes = {f"{2**L}^3": c for L, c in enumerate(stats_by_case[name]["nodes_by_level"])
                      if L >= 1 and c}
        row["nodes_match"] = ref_nodes == mine_nodes
        row["cells_at_base_match"] = ref_meta["bins"][b]["cells_at_base"] == \
            stats_by_case[name]["nodes_by_level"][0]
        row["nodes_mine"], row["nodes_ref"] = mine_nodes, ref_nodes
        result["families"][name] = row
        log(f"reference {name}: dust {row['dust_max_rel_diff']:.2e} gas "
            f"{row['gas_max_rel_diff']:.2e} nodes_match={row['nodes_match']} "
            f"base_match={row['cells_at_base_match']}")
    return result


# ----------------------------------------------------------------------------
# README
# ----------------------------------------------------------------------------
README = """# De-refined dust and gas density, @N@³

Dust and gas density cubes for `turb@N@/output_00009` in which low-count regions are
averaged over larger cells, so that every cell holds at least one simulation particle
of the grain family it describes. Each family has only about one grain per cell on average,
so the native cubes in `dust/cic/` are dominated by sampling noise in low-density regions,
which produces a spurious tail to low densities.

## Rule

Consider each group of 2×2×2 cells. If any one of the eight holds fewer than one particle
(counted with the same cloud-in-cell weights as the density), all eight are replaced by
their volume-mean density. The merged group then acts as one cell inside a group of eight
larger cells, and the test repeats at that scale, up to the whole box. Gas density is
averaged over the same cells as the dust. This follows Moseley (2025, thesis) §4.4.2.1
with a threshold of one particle. Total dust mass and total gas mass are conserved, and no
smoothing is applied.

## Files

All cubes are `(@N@, @N@, @N@)` in x,y,z order, in code units, with the same normalization
as `dust/cic/`: float32, except `level_*` (uint8). Family `XX` is `dust/cic/rho_XX.npy`.

| File | Content |
| --- | --- |
| `dust_XX.npy` | Dust density of family `XX` (01–12), de-refined using that family's particles. |
| `gas_XX.npy` | Gas density averaged over the same cells as `dust_XX`. |
| `level_XX.npy` | Cell size of each base cell (below). |
@GROUPROWS@| `dust_all.npy`, `gas_all.npy`, `level_all.npy` | All grains (`dust/cic/rho_total.npy`), de-refined using the particles of all 12 families together. |

The grain radii of each family are listed in `dust/cic/README.md`. Radius ranges of the
groups:

@RADII@

## Use

The dust-to-gas density ratio of family `XX` is `dust_XX / gas_XX`. Always pair a dust cube
with the gas cube of the same name, not with `gas/rho.npy` or another case's gas cube,
since each case merges different cells. For all grains use `dust_all / gas_all`. Each of
`dust_all`, `dust_small`, `dust_medium` and `dust_large` is its own de-refinement of its
particles, not the sum of the de-refined family cubes.

`level_XX` records the size of the cell that contains each base cell. A value `L` means the
base cell belongs to a cell `2^L` base cells on a side, so 0 is native resolution and 1 is a
2×2×2 cell. Use it to select regions by resolution, for example `level == 0`.

```python
import numpy as np
p = 'turb@N@/output_00009/py_out/derefined/'
dust = np.load(p + 'dust_06.npy', mmap_mode='r')
gas = np.load(p + 'gas_06.npy', mmap_mode='r')
level = np.load(p + 'level_06.npy', mmap_mode='r')
ratio = dust / gas                      # dust-to-gas density ratio, family 06
native = np.asarray(level) == 0
print(ratio.mean(), ratio[native].mean())
```

## Caveats

The cubes are piecewise constant, and structure on scales below the local cell size is
absent. With about one particle per cell in each family, @FAMFRAC@ of the volume of a
family cube lies in cells of 2×2×2 or larger, and only @FAMNATIVE@ keeps the native
resolution. @GROUPFRAC@ The all-grain cube is much closer to native resolution,
with @ALLFRAC@ of its volume in merged cells, because it has 12 times as many particles.
Compare different cases only after checking their `level` cubes: a ratio built from a
small family and one built from all grains are resolved on different scales.

Volume fractions by cell size, from this run (percent of the volume):

@TABLE@
"""


def readme_text(n, cases, stats, bins):
    radii = {}
    for name, fams in cases.items():
        radii[name] = (bins[fams[0] - 1][0], bins[fams[-1] - 1][1])
    group_names = [c for c in cases if not c.isdigit() and c != "all"]
    group_rows = ""
    for g in group_names:
        fams = cases[g]
        group_rows += (f"| `dust_{g}.npy`, `gas_{g}.npy`, `level_{g}.npy` | Families "
                       f"{fams[0]:02d}–{fams[-1]:02d} together, de-refined using their "
                       f"combined particles. |\n")
    radii_lines = "\n".join(f"- `{g}` (families {cases[g][0]:02d}–{cases[g][-1]:02d}): "
                            f"{radii[g][0]:.3f}–{radii[g][1]:.3f} µm" for g in group_names)
    radii_lines += f"\n- `all`: {radii['all'][0]:.3f}–{radii['all'][1]:.3f} µm"

    def pct(name, levels):
        return 100.0 * sum(stats[name]["volume_fraction_by_level"][L] for L in levels)

    def rng(names, levels):
        v = [pct(c, levels) for c in names]
        return f"{min(v):.0f}–{max(v):.0f}%"

    fam_names = [c for c in cases if c.isdigit()]
    lmax = len(stats["all"]["volume_fraction_by_level"]) - 1
    merged, native = range(1, lmax + 1), [0]
    gmerged = ", ".join(f"{pct(g, merged):.0f}% for `{g}`" for g in group_names)
    rows = ["| Case | Radius [µm] | native | 2³ | 4³ | ≥ 8³ |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for name in cases:
        sel = [[0], [1], [2], range(3, lmax + 1)]
        cells = " | ".join(f"{pct(name, s):.1f}" for s in sel)
        rows.append(f"| `{name}` | {radii[name][0]:.3f}–{radii[name][1]:.3f} | {cells} |")
    text = README
    for key, val in {
        "@N@": str(n),
        "@GROUPROWS@": group_rows,
        "@RADII@": radii_lines,
        "@FAMFRAC@": rng(fam_names, merged),
        "@FAMNATIVE@": rng(fam_names, native),
        "@GROUPFRAC@": f"For the size groups the merged fraction is {gmerged}." if group_names else "",
        "@ALLFRAC@": f"{pct('all', merged):.0f}%",
        "@TABLE@": "\n".join(rows),
    }.items():
        text = text.replace(key, val)
    return text


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def run(snapshot, threshold, groups, tol, reference_dir):
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)

    snapshot = Path(snapshot).resolve()
    root = snapshot / "py_out"
    final = root / "derefined"
    if final.exists():
        raise FileExistsError(f"{final} exists; refusing to overwrite")
    n = json.loads((root / "metadata.json").read_text())["n"]
    cases = case_list(groups)
    cache_rec, cnt = check_provenance(root, n, log)
    out = root / f"derefined.incomplete_{os.environ.get('SLURM_JOB_ID', os.getpid())}"
    out.mkdir()
    log(f"writing to {out}")

    stats_by_case, verify_by_case = {}, {}
    for name, fams in cases.items():
        stats, inputs, gate = process_case(name, fams, cnt, root, out, threshold)
        ver = verify_case(name, out, gate, inputs, threshold, tol)
        stats.update(ver)
        stats["families"] = fams
        stats["dust_mass_in"], stats["gas_mass_in"] = inputs["dust_in"], inputs["gas_in"]
        stats_by_case[name] = stats
        frac = " ".join(f"{100 * v:.2f}%" for v in stats["volume_fraction_by_level"])
        log(f"case {name}: final cells {stats['n_final_cells']}, min particles "
            f"{stats['min_particles_per_final_cell']:.4f}, mass residual dust "
            f"{ver['dust_mass_residual']:.1e} gas {ver['gas_mass_residual']:.1e}; "
            f"volume by level [{frac}]")
        del gate

    reference = None
    if reference_dir:
        reference = compare_reference(reference_dir, out, stats_by_case, n, log)
        bad = [k for k, r in reference["families"].items()
               if not (r["nodes_match"] and r["cells_at_base_match"])
               or max(r["dust_max_rel_diff"], r["gas_max_rel_diff"]) > 1e-6]
        if bad:
            (out / "metadata.partial.json").write_text(json.dumps(reference, indent=2))
            raise RuntimeError(f"disagrees with reference for families {bad}; see {out}")

    cic = root / "dust" / "cic"
    inputs = [root / "COMPLETE", cic / "COMPLETE", cic / "metadata.json", cic / "bins.csv",
              root / "gas" / "rho.npy", cic / "rho_total.npy"]
    inputs += [cic / f"rho_{f:02d}.npy" for f in range(1, NFAM + 1)]
    prefix = cache_rec["cache_prefix"]
    inputs += [Path(prefix + s) for s in ("_dust_num_bins.npy", "_dust_rho_bins.npy",
                                           "_gas_rho.npy", "_meta.json")]
    with open(cic / "bins.csv") as f:
        bins = [(float(r["radius_min_um"]), float(r["radius_max_um"])) for r in csv.DictReader(f)]
    meta = {
        "snapshot": str(snapshot), "n": n, "threshold_particles": threshold,
        "method": METHOD, "script": Path(__file__).name, "script_commit": script_commit(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "host": platform.node(),
        "numpy": np.__version__, "python": sys.version.split()[0],
        "mass_residual_tolerance": tol, "wall_seconds": time.time() - t0,
        "inputs": [file_record(p) for p in inputs], "provenance": cache_rec,
        "level_meaning": "level L: base cell lies in a final cell 2^L base cells on a side",
        "cases": stats_by_case,
    }
    if reference:
        meta["reference_check"] = reference
    (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    (out / "README.md").write_text(readme_text(n, cases, stats_by_case, bins))
    (out / "COMPLETE").write_text(f"{len(cases)} cases written and verified from the written cubes.\n")
    if final.exists():
        raise FileExistsError(f"{final} appeared during the run")
    out.rename(final)
    log(f"DEREFINE_COMPLETE {final}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("snapshot", type=Path, help="turbN/output_NNNNN directory containing py_out/")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                   help="minimum CIC particles per final cell (default 1)")
    p.add_argument("--groups", default=DEFAULT_GROUPS,
                   help="size groups as name=lo-hi family ranges (default %(default)s); '' for none")
    p.add_argument("--mass-tol", type=float, default=1e-6, help="max relative mass residual")
    p.add_argument("--reference-dir", type=Path,
                   help="paper derefine_cubes output to cross-check families against")
    a = p.parse_args(argv)
    run(a.snapshot, a.threshold, parse_groups(a.groups), a.mass_tol, a.reference_dir)


if __name__ == "__main__":
    main()
