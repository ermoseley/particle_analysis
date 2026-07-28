#!/usr/bin/env python3
"""Bounded readers for regular multi-file mini-RAMSES output payloads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterator

import numpy as np


def read_info_int(output_dir: Path, key: str) -> int:
    """Read one integer-valued key from a regular output ``info.txt``."""
    info_path = Path(output_dir) / "info.txt"
    for raw in info_path.read_text().splitlines():
        if "=" not in raw:
            continue
        name, value = raw.split("=", 1)
        if name.strip() == key:
            return int(value.split()[0])
    raise ValueError(f"Could not read {key} from {info_path}")


def regular_output_files(output_dir: Path, family: str) -> tuple[Path, ...]:
    """Return the exact rank-file set declared by ``nfile``."""
    output_dir = Path(output_dir)
    nfile = read_info_int(output_dir, "nfile")
    if nfile <= 0:
        raise ValueError(f"{output_dir / 'info.txt'} has invalid nfile={nfile}")

    expected = tuple(output_dir / f"{family}.{i:05d}" for i in range(1, nfile + 1))
    missing = [path for path in expected if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{output_dir}: missing {family} rank file(s): "
            + ", ".join(path.name for path in missing)
        )

    pattern = re.compile(rf"{re.escape(family)}\.\d{{5}}$")
    actual = tuple(
        sorted(path for path in output_dir.iterdir() if pattern.fullmatch(path.name))
    )
    if actual != expected:
        raise ValueError(
            f"{output_dir}: {family} files do not match configured nfile={nfile}"
        )
    return expected


@dataclass(frozen=True)
class MeshFieldBlock:
    """One bounded regular-output field block in file and level order."""

    family: str
    file_index: int
    level: int
    oct_start: int
    values: np.ndarray


def iter_mesh_field_blocks(
    run_dir: Path,
    output_num: int,
    family: str,
    *,
    chunk_octs: int = 1_000_000,
    expected_nvar: int | None = None,
) -> Iterator[MeshFieldBlock]:
    """Stream regular hydro-like payloads with shape ``(oct, var, cell)``."""
    if chunk_octs <= 0:
        raise ValueError("chunk_octs must be positive")

    output_dir = Path(run_dir) / f"output_{output_num:05d}"
    reference_layout: tuple[int, int, int, int] | None = None
    for file_index, path in enumerate(
        regular_output_files(output_dir, family), start=1
    ):
        header = np.fromfile(path, dtype="<i4", count=4)
        if header.size != 4:
            raise ValueError(f"{path}: truncated mesh-field header")
        ndim, nvar, levelmin, levelmax = map(int, header)
        if ndim <= 0 or nvar <= 0 or levelmax < levelmin:
            raise ValueError(f"{path}: invalid mesh-field header {header.tolist()}")
        if expected_nvar is not None and nvar != expected_nvar:
            raise ValueError(f"{path}: nvar={nvar}, expected {expected_nvar}")

        layout = (ndim, nvar, levelmin, levelmax)
        if reference_layout is None:
            reference_layout = layout
        elif layout != reference_layout:
            raise ValueError(
                f"{path}: layout {layout} differs from {reference_layout}"
            )

        nlevels = levelmax - levelmin + 1
        noct = np.fromfile(path, dtype="<i4", count=nlevels, offset=16)
        if noct.size != nlevels or np.any(noct < 0):
            raise ValueError(f"{path}: invalid per-level oct counts")

        cells_per_oct = 1 << ndim
        header_bytes = 16 + 4 * nlevels
        expected_bytes = header_bytes + (
            int(np.sum(noct, dtype=np.int64))
            * nvar
            * cells_per_oct
            * np.dtype("<f4").itemsize
        )
        if path.stat().st_size != expected_bytes:
            raise ValueError(
                f"{path}: size={path.stat().st_size}, expected={expected_bytes}"
            )

        offset = header_bytes
        for level, level_noct in zip(
            range(levelmin, levelmax + 1), map(int, noct)
        ):
            level_values = level_noct * nvar * cells_per_oct
            if level_noct:
                raw = np.memmap(
                    path,
                    dtype="<f4",
                    mode="r",
                    offset=offset,
                    shape=(level_noct, nvar, cells_per_oct),
                )
                for start in range(0, level_noct, chunk_octs):
                    stop = min(start + chunk_octs, level_noct)
                    yield MeshFieldBlock(
                        family=family,
                        file_index=file_index,
                        level=level,
                        oct_start=start,
                        values=np.array(raw[start:stop], copy=True),
                    )
                del raw
            offset += level_values * np.dtype("<f4").itemsize


def iter_sixray_blocks(
    run_dir: Path,
    output_num: int,
    *,
    chunk_octs: int = 1_000_000,
) -> Iterator[MeshFieldBlock]:
    """Stream all 12 directional-column fields from every output file."""
    yield from iter_mesh_field_blocks(
        run_dir,
        output_num,
        "sixray",
        chunk_octs=chunk_octs,
        expected_nvar=12,
    )
