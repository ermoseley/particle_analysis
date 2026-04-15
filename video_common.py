#!/usr/bin/env python3
"""Shared helpers for the particle-analysis video scripts."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np

FLOOR = 1e-30
RATIO_FLOOR = 1e-30

# Default frame size: 1080p true HD.
FRAME_WIDTH_PX = 1920
FRAME_HEIGHT_PX = 1080
FRAME_DPI = 96


def orient_like_column_video(arr: np.ndarray) -> np.ndarray:
    """Match the imshow orientation used by the column-density videos."""
    if arr.ndim == 2:
        return arr.T
    return np.transpose(arr, (1, 0, 2))


def log_vmin_vmax_from_last_frame(log_last: np.ndarray) -> tuple[float, float]:
    """Return finite min/max of the supplied log field."""
    v = np.asarray(log_last, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return -1.0, 1.0
    vmin = float(np.min(v))
    vmax = float(np.max(v))
    if vmin >= vmax:
        vmax = vmin + 1e-9
    return vmin, vmax


def symmetric_log_vmin_vmax_from_last_frame(log_last: np.ndarray) -> tuple[float, float]:
    """Return symmetric finite limits about zero for a log-deviation field."""
    v = np.asarray(log_last, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return -1.0, 1.0
    vmax = max(abs(float(np.min(v))), abs(float(np.max(v))))
    if vmax <= 0.0:
        vmax = 1e-9
    return -vmax, vmax


def get_output_numbers(run_dir: Path, start: int | None, end: int | None) -> list[int]:
    """Return sorted list of output numbers in ``run_dir`` within ``[start, end]``."""
    run_dir = Path(run_dir)
    outs: list[int] = []
    for path in run_dir.iterdir():
        if path.is_dir() and path.name.startswith("output_"):
            try:
                outs.append(int(path.name.replace("output_", "")))
            except ValueError:
                continue
    if not outs:
        raise FileNotFoundError(f"No output_* directories in {run_dir}")
    outs = sorted(outs)
    if start is not None:
        outs = [n for n in outs if n >= start]
    if end is not None:
        outs = [n for n in outs if n <= end]
    if not outs:
        raise ValueError("No outputs in requested start/end range")
    return outs


def discover_frame_names(frames_dir: Path) -> list[str]:
    """Return sorted ``frame_XXXXX.png`` names from ``frames_dir``."""
    frames_dir = Path(frames_dir)
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    frame_pattern = re.compile(r"frame_(\d+)\.png$")
    found: list[tuple[int, str]] = []
    for path in frames_dir.iterdir():
        if not path.is_file():
            continue
        match = frame_pattern.match(path.name)
        if match:
            found.append((int(match.group(1)), path.name))
    if not found:
        raise FileNotFoundError(f"No frame_*.png found in {frames_dir}")
    return [name for _, name in sorted(found)]


def write_concat_frame_list(
    list_file: Path,
    fps: int,
    *,
    frame_names: list[str] | None = None,
    output_numbers: list[int] | None = None,
) -> None:
    """Write an ffmpeg concat-demuxer frame list."""
    if (frame_names is None) == (output_numbers is None):
        raise ValueError("Provide exactly one of frame_names or output_numbers")

    duration_sec = 1.0 / fps
    with open(list_file, "w") as f:
        if frame_names is not None:
            for name in frame_names:
                f.write(f"file '{name}'\n")
                f.write(f"duration {duration_sec}\n")
            f.write(f"file '{frame_names[-1]}'\n")
        else:
            assert output_numbers is not None
            for output_num in output_numbers:
                f.write(f"file 'frame_{output_num:05d}.png'\n")
                f.write(f"duration {duration_sec}\n")
            f.write(f"file 'frame_{output_numbers[-1]:05d}.png'\n")


def encode_frames(list_file: Path, frames_dir: Path, out_video: Path) -> None:
    """Encode frames listed in ``list_file`` into ``out_video`` using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        str(out_video),
    ]
    result = subprocess.run(cmd, cwd=str(frames_dir))
    if result.returncode != 0:
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-c:v",
            "mpeg4",
            "-pix_fmt",
            "yuv420p",
            "-q:v",
            "2",
            str(out_video),
        ]
        result = subprocess.run(cmd, cwd=str(frames_dir))
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg failed with code {result.returncode}")
