#!/usr/bin/env python3
"""Focused synthetic checks for regular multi-file output readers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from column_utils import cic_deposit_2d, dust_pos_plane, get_dust_column
from dust_projection import iter_dust_snapshot_blocks, read_dust_snapshot
from ramses_streaming import iter_mesh_field_blocks, iter_sixray_blocks


def make_output(root: Path, nfile: int, ndim: int = 3) -> Path:
    output = root / "output_00001"
    output.mkdir(parents=True)
    (output / "info.txt").write_text(
        f"nfile       = {nfile:11d}\n"
        f"ncpu        = {nfile:11d}\n"
        f"ndim        = {ndim:11d}\n"
    )
    return output


def write_dust_header(output: Path, npart: int, *, has_mass: bool = True) -> None:
    fields = "pos vel mass size birth_id" if has_mass else "pos vel size birth_id"
    (output / "dust_header.txt").write_text(
        "Total number of particles\n"
        f"{npart}\n"
        "Particle fields\n"
        f"{fields}\n"
        "id_bytes=8\n"
    )


def write_dust_file(
    path: Path,
    pos: np.ndarray,
    mass: np.ndarray,
    size: np.ndarray,
    particle_id: np.ndarray,
) -> None:
    pos = np.asarray(pos, dtype="<f4")
    mass = np.asarray(mass, dtype="<f4")
    size = np.asarray(size, dtype="<f4")
    particle_id = np.asarray(particle_id, dtype="<i8")
    npart = pos.shape[0]
    vel = np.zeros_like(pos)
    with path.open("wb") as stream:
        np.array([3, npart], dtype="<i4").tofile(stream)
        for component in pos.T:
            component.tofile(stream)
        for component in vel.T:
            component.tofile(stream)
        mass.tofile(stream)
        size.tofile(stream)
        particle_id.tofile(stream)


def write_mesh_file(path: Path, values: np.ndarray, level: int = 3) -> None:
    values = np.asarray(values, dtype="<f4")
    noct, nvar, cells_per_oct = values.shape
    if cells_per_oct != 8:
        raise ValueError("Synthetic 3D mesh payload must have eight cells per oct")
    with path.open("wb") as stream:
        np.array([3, nvar, level, level, noct], dtype="<i4").tofile(stream)
        values.tofile(stream)


class OutputStreamingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.pos = np.array(
            [
                [0.10, 0.20, 0.30],
                [0.45, 0.55, 0.65],
                [0.80, 0.90, 0.05],
                [0.99, 0.01, 0.50],
                [0.25, 0.75, 0.40],
            ],
            dtype=np.float32,
        )
        self.mass = np.array([1.0, 2.0, 0.5, 1.5, 3.0], dtype=np.float32)
        self.size = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        self.ids = np.array([11, 12, 13, 14, 15], dtype=np.int64)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _dust_run(self, name: str, splits: tuple[slice, ...]) -> Path:
        run = self.root / name
        output = make_output(run, len(splits))
        write_dust_header(output, self.pos.shape[0])
        for index, selection in enumerate(splits, start=1):
            write_dust_file(
                output / f"dust.{index:05d}",
                self.pos[selection],
                self.mass[selection],
                self.size[selection],
                self.ids[selection],
            )
        return run

    def test_dust_reader_preserves_one_file_and_streams_all_rank_files(self) -> None:
        one_file = self._dust_run("one", (slice(None),))
        two_files = self._dust_run("two", (slice(0, 2), slice(2, None)))

        snapshot = read_dust_snapshot(one_file, 1)
        np.testing.assert_array_equal(snapshot.particle_id, self.ids)
        np.testing.assert_allclose(snapshot.pos, self.pos)

        blocks = list(iter_dust_snapshot_blocks(two_files, 1, chunk_size=2))
        self.assertTrue(all(block.size.size <= 2 for block in blocks))
        np.testing.assert_array_equal(
            np.concatenate([block.particle_id for block in blocks]), self.ids
        )
        np.testing.assert_allclose(
            np.concatenate([block.mass for block in blocks]), self.mass
        )

    def test_dust_column_accumulates_chunks_without_materializing_particles(self) -> None:
        one_file = self._dust_run("column_one", (slice(None),))
        two_files = self._dust_run(
            "column_two", (slice(0, 1), slice(1, None))
        )
        expected = cic_deposit_2d(
            dust_pos_plane(self.pos, "z"), self.mass, 8
        )
        one = get_dust_column(one_file, 1, 8, chunk_size=2)
        two = get_dust_column(two_files, 1, 8, chunk_size=2)
        np.testing.assert_allclose(one, expected)
        np.testing.assert_allclose(two, expected)
        self.assertAlmostEqual(float(np.sum(two)), float(np.sum(self.mass)))

    def test_hydro_and_sixray_stream_every_configured_file(self) -> None:
        run = self.root / "mesh"
        output = make_output(run, 2)
        hydro = [
            np.arange(2 * 5 * 8, dtype=np.float32).reshape(2, 5, 8),
            np.arange(5 * 8, dtype=np.float32).reshape(1, 5, 8) + 1000,
        ]
        sixray = [
            np.arange(12 * 8, dtype=np.float32).reshape(1, 12, 8),
            np.arange(2 * 12 * 8, dtype=np.float32).reshape(2, 12, 8) + 2000,
        ]
        for index in range(2):
            write_mesh_file(output / f"hydro.{index + 1:05d}", hydro[index])
            write_mesh_file(output / f"sixray.{index + 1:05d}", sixray[index])

        hydro_blocks = list(
            iter_mesh_field_blocks(
                run, 1, "hydro", chunk_octs=1, expected_nvar=5
            )
        )
        sixray_blocks = list(iter_sixray_blocks(run, 1, chunk_octs=1))
        self.assertTrue(all(block.values.shape[0] == 1 for block in hydro_blocks))
        self.assertTrue(all(block.values.shape[0] == 1 for block in sixray_blocks))
        np.testing.assert_array_equal(
            np.concatenate([block.values for block in hydro_blocks]),
            np.concatenate(hydro),
        )
        np.testing.assert_array_equal(
            np.concatenate([block.values for block in sixray_blocks]),
            np.concatenate(sixray),
        )

    def test_missing_rank_file_is_rejected(self) -> None:
        run = self.root / "missing"
        output = make_output(run, 2)
        write_dust_header(output, self.pos.shape[0])
        write_dust_file(
            output / "dust.00001",
            self.pos,
            self.mass,
            self.size,
            self.ids,
        )
        with self.assertRaisesRegex(FileNotFoundError, "dust.00002"):
            list(iter_dust_snapshot_blocks(run, 1, chunk_size=2))


if __name__ == "__main__":
    unittest.main()
