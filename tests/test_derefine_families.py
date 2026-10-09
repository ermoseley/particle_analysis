import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import derefine_families as dd  # noqa: E402


def reference(counts, dust, gas, thr):
    """Plain recursion over nodes: the rule as stated, with no vectorisation."""
    n = counts.shape[0]
    level = np.zeros((n,) * 3, np.uint8)
    out = [np.zeros((n,) * 3), np.zeros((n,) * 3)]

    def box(i, j, k, L):
        s = 2**L
        return (slice(i * s, (i + 1) * s), slice(j * s, (j + 1) * s), slice(k * s, (k + 1) * s))

    def rec(i, j, k, L):
        if L > 0:
            kids = [counts[box(2 * i + a, 2 * j + b, 2 * k + c, L - 1)].sum()
                    for a, b, c in itertools.product((0, 1), repeat=3)]
            if all(x >= thr for x in kids):
                for a, b, c in itertools.product((0, 1), repeat=3):
                    rec(2 * i + a, 2 * j + b, 2 * k + c, L - 1)
                return
        sl = box(i, j, k, L)
        level[sl] = L
        for o, f in zip(out, (dust, gas)):
            o[sl] = f[sl].mean()

    rec(0, 0, 0, int(np.log2(n)))
    return level, out


def random_case(n, seed, shape=0.5):
    rng = np.random.default_rng(seed)
    counts = rng.gamma(shape, 1.0 / shape, (n, n, n)).astype(np.float32)
    return counts, rng.random((n, n, n)).astype(np.float32), rng.random((n, n, n)).astype(np.float32) + 0.5


@pytest.mark.parametrize("n,seed,thr,shape", [(8, 0, 1.0, 0.5), (16, 1, 1.0, 0.8), (16, 2, 2.5, 2.0),
                                              (4, 3, 0.7, 0.3)])
def test_matches_recursive_reference(n, seed, thr, shape):
    counts, dust, gas = random_case(n, seed, shape)
    level, (d, g), _ = dd.derefine(counts, [dust, gas], thr)
    ref_level, (rd, rg) = reference(counts.astype(np.float64), dust.astype(np.float64),
                                    gas.astype(np.float64), thr)
    assert np.array_equal(level, ref_level)
    np.testing.assert_allclose(d, rd, rtol=2e-7)
    np.testing.assert_allclose(g, rg, rtol=2e-7)


def test_conservation_and_floor():
    counts, dust, gas = random_case(16, 5, 3.0)
    level, (d, g), stats = dd.derefine(counts, [dust, gas], 1.0)
    assert d.dtype == g.dtype == np.float32 and level.dtype == np.uint8
    assert abs(d.sum(dtype=np.float64) / dust.sum(dtype=np.float64) - 1) < 1e-6
    assert abs(g.sum(dtype=np.float64) / gas.sum(dtype=np.float64) - 1) < 1e-6
    assert np.isfinite(d).all() and (d > 0).all()
    # Independent acceptance checks on the finished cubes.
    assert dd.verify_structure(level, counts, 1.0) >= 1.0
    assert stats["min_particles_per_final_cell"] >= 1.0
    dd.verify_fields(level, d, g)
    assert sum(stats["volume_fraction_by_level"]) == pytest.approx(1.0)
    assert stats["n_final_cells"] == sum(stats["nodes_by_level"])
    # Something must actually have merged for the test to mean anything.
    assert (level > 0).any() and (level == 0).any()


def test_gas_shares_the_dust_structure():
    counts, dust, gas = random_case(8, 6)
    level, (d, g), _ = dd.derefine(counts, [dust, gas], 1.0)
    for L in range(1, int(level.max()) + 1):
        m = 8 >> L
        for f in (d, g):
            blocks = f.reshape(m, 2**L, m, 2**L, m, 2**L)
            lv = level.reshape(m, 2**L, m, 2**L, m, 2**L)
            full = (lv == L).all(axis=(1, 3, 5))
            assert (blocks.max(axis=(1, 3, 5)) == blocks.min(axis=(1, 3, 5)))[full].all()


def test_one_deficient_cell_merges_its_oct_even_if_the_oct_total_is_large():
    # The oct holds 7*1.5 + 0.5 = 11 particles, far above one, but one cell holds 0.5.
    # A gate on the oct total would leave it native; the per-child gate merges it.
    counts = np.full((4, 4, 4), 1.5, np.float32)
    counts[0, 0, 0] = 0.5
    assert counts[:2, :2, :2].sum() > 1.0
    dust = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
    level, (d, _), _ = dd.derefine(counts, [dust, dust], 1.0)
    expect = np.zeros((4, 4, 4), np.uint8)
    expect[:2, :2, :2] = 1
    assert np.array_equal(level, expect)
    assert np.all(d[:2, :2, :2] == np.float32(dust[:2, :2, :2].mean(dtype=np.float64)))
    assert np.array_equal(d[2:], dust[2:])


def test_merged_octs_are_one_cell_in_a_larger_oct():
    # One 2x2x2 oct holds 0.8 particles in total, so it is a deficient child of the
    # 4x4x4 octant, which then becomes a single cell.
    counts = np.full((8, 8, 8), 2.0, np.float32)
    counts[:2, :2, :2] = 0.1
    dust = np.random.default_rng(0).random((8, 8, 8)).astype(np.float32)
    level, (d, _), stats = dd.derefine(counts, [dust, dust], 1.0)
    assert np.all(level[:4, :4, :4] == 2)
    assert np.all(d[:4, :4, :4] == d[0, 0, 0])
    assert np.all(level[4:] == 0) and np.all(level[:4, 4:] == 0)
    assert stats["nodes_by_level"][2] == 1


def notebook_like(counts, dust, thr, levels):
    """The thesis notebook's loop: split a node whenever its children all clear the
    threshold, without asking whether the node itself was reached."""
    n = counts.shape[0]
    out = np.full(dust.shape, dust.mean())
    for lvl in range(levels):
        s = n >> lvl
        for i, j, k in itertools.product(range(2**lvl), repeat=3):
            h = s // 2
            kids = {(a, b, c): (slice((2 * i + a) * h, (2 * i + a + 1) * h),
                                slice((2 * j + b) * h, (2 * j + b + 1) * h),
                                slice((2 * k + c) * h, (2 * k + c + 1) * h))
                    for a, b, c in itertools.product((0, 1), repeat=3)}
            if all(counts[sl].sum() >= thr for sl in kids.values()):
                for sl in kids.values():
                    out[sl] = dust[sl].mean()
    return out


def test_strict_nesting_differs_from_notebook_overwrite():
    # The 4^3 octant at the origin has one deficient 2^3 child, so it is a single cell.
    # Its other 2^3 children are well populated; the notebook rule splits them anyway
    # and overwrites part of the merged cell, which breaks nesting and mass.
    counts = np.full((8, 8, 8), 2.0, np.float32)
    counts[:2, :2, :2] = 0.1
    dust = np.random.default_rng(1).random((8, 8, 8))
    _, (d, _), _ = dd.derefine(counts, [dust.astype(np.float32)] * 2, 1.0)
    assert np.unique(d[:4, :4, :4]).size == 1
    nb = notebook_like(counts.astype(np.float64), dust, 1.0, 3)
    assert np.unique(nb[:4, :4, :4]).size > 1
    assert abs(d.sum() - dust.sum()) < 1e-4 * dust.sum()


def test_threshold_option_changes_the_structure():
    counts, dust, gas = random_case(8, 7, 2.0)
    lo, _, _ = dd.derefine(counts, [dust, gas], 0.5)
    hi, _, _ = dd.derefine(counts, [dust, gas], 3.0)
    assert (hi > 0).mean() > (lo > 0).mean()


def test_pyramid_rejects_non_cubic_or_non_power_of_two():
    with pytest.raises(ValueError):
        dd.pyramid(np.zeros((6, 6, 6)))
    with pytest.raises(ValueError):
        dd.pyramid(np.zeros((4, 4, 2)))


def test_verify_structure_catches_a_wrong_level_map():
    counts = np.full((4, 4, 4), 1.5, np.float32)
    counts[0, 0, 0] = 0.5
    level, _, _ = dd.derefine(counts, [counts, counts], 1.0)
    bad = level.copy()
    bad[:2, :2, :2] = 0           # leave the deficient oct native
    with pytest.raises(AssertionError):
        dd.verify_structure(bad, counts, 1.0)
    bad = level.copy()
    bad[2:, 2:, 2:] = 1           # merge an oct whose cells are all populated
    with pytest.raises(AssertionError):
        dd.verify_structure(bad, counts, 1.0)


def test_parse_groups_and_cases():
    g = dd.parse_groups(dd.DEFAULT_GROUPS)
    assert g == {"small": [1, 2, 3, 4], "medium": [5, 6, 7, 8], "large": [9, 10, 11, 12]}
    cases = dd.case_list(g)
    assert list(cases)[:3] == ["01", "02", "03"] and list(cases)[-1] == "all"
    assert len(cases) == 16
    assert len(dd.case_list({})) == 13
    with pytest.raises(ValueError):
        dd.parse_groups("bad=0-3")


def make_snapshot(tmp_path, n=8, seed=0):
    rng = np.random.default_rng(seed)
    snap = tmp_path / "turb8" / "output_00009"
    root = snap / "py_out"
    cic = root / "dust" / "cic"
    (root / "gas").mkdir(parents=True)
    cic.mkdir(parents=True)
    cache = tmp_path / "cache"
    cache.mkdir()
    counts = rng.gamma(0.8, 1.25, (12, n, n, n)).astype(np.float32)
    rho = (rng.random((12, n, n, n)) * counts).astype(np.float32) + np.float32(1e-3)
    gas = (rng.random((n, n, n)) + 0.5).astype(np.float32)
    prefix = str(cache / "output_00009")
    np.save(prefix + "_dust_num_bins.npy", counts)
    np.save(prefix + "_dust_rho_bins.npy", rho)
    np.save(prefix + "_gas_rho.npy", gas)
    (cache / "output_00009_meta.json").write_text(json.dumps({"npart": float(counts.sum(dtype=np.float64))}))
    np.save(root / "gas" / "rho.npy", gas)
    for b in range(12):
        np.save(cic / f"rho_{b + 1:02d}.npy", rho[b])
    np.save(cic / "rho_total.npy", rho.sum(axis=0, dtype=np.float64).astype(np.float32))
    (root / "metadata.json").write_text(json.dumps({"n": n}))
    (cic / "metadata.json").write_text(json.dumps({"cache_prefix": prefix}))
    edges = np.geomspace(0.05, 1.0, 13)
    (cic / "bins.csv").write_text("file,radius_min_um,radius_max_um,mass_fraction\n" + "".join(
        f"rho_{b + 1:02d}.npy,{edges[b]},{edges[b + 1]},0.08\n" for b in range(12)))
    (root / "COMPLETE").write_text("")
    (cic / "COMPLETE").write_text("")
    return snap, counts, rho, gas


def test_end_to_end_on_a_synthetic_snapshot(tmp_path):
    snap, counts, rho, gas = make_snapshot(tmp_path)
    dd.run(snap, 1.0, dd.parse_groups(dd.DEFAULT_GROUPS), 1e-6, None)
    out = snap / "py_out" / "derefined"
    assert (out / "COMPLETE").exists() and (out / "README.md").exists()
    assert not list((snap / "py_out").glob("derefined.incomplete*"))
    names = [f"{b:02d}" for b in range(1, 13)] + ["small", "medium", "large", "all"]
    for name in names:
        for kind in ("dust", "gas", "level"):
            assert (out / f"{kind}_{name}.npy").exists()
    meta = json.loads((out / "metadata.json").read_text())
    assert set(meta["cases"]) == set(names)
    # Group gate is the sum of its families' counts, and its dust the sum of their cubes.
    gate = counts[4:8].sum(axis=0, dtype=np.float64)
    level = np.load(out / "level_medium.npy")
    assert dd.verify_structure(level, gate, 1.0) >= 1.0
    dust = np.load(out / "dust_medium.npy")
    assert abs(dust.sum(dtype=np.float64) / rho[4:8].sum(dtype=np.float64) - 1) < 1e-6
    # The all-grain structure is finer than any single family's.
    assert (np.load(out / "level_all.npy") > 0).mean() < (np.load(out / "level_03.npy") > 0).mean()
    text = (out / "README.md").read_text()
    assert "@" not in text and "8³" in text
    with pytest.raises(FileExistsError):
        dd.run(snap, 1.0, dd.parse_groups(dd.DEFAULT_GROUPS), 1e-6, None)


def test_run_rejects_a_cache_that_differs_from_the_published_cubes(tmp_path):
    snap, *_ = make_snapshot(tmp_path)
    gas = np.load(snap / "py_out" / "gas" / "rho.npy")
    gas[0, 0, 0] *= 1.01
    np.save(snap / "py_out" / "gas" / "rho.npy", gas)
    with pytest.raises(RuntimeError):
        dd.run(snap, 1.0, {}, 1e-6, None)
    assert not (snap / "py_out" / "derefined").exists()
