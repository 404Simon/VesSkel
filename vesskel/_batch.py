"""Batch I/O helpers and worker function for multiprocessing."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import itk
import numpy as np
from PIL import Image

from vesskel.config import PipelineConfig
from vesskel.pipeline import analyze_binary_image


def _load_image(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
    elif suffix == ".mhd":
        arr = np.asarray(itk.imread(str(path)))
    else:
        with Image.open(path) as im:
            arr = np.asarray(im)

    if arr.ndim == 0:
        raise ValueError("Scalar input is not supported")

    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        arr = np.max(arr[..., :3], axis=-1)

    if arr.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D image, got shape={arr.shape}")

    return arr


def _sanitize_for_csv(value: object) -> object:
    if isinstance(value, (np.generic,)):
        return value.item()
    return value


def _write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _sanitize_for_csv(v) for k, v in row.items()})


def _save_skeleton(
    path: Path,
    skeleton: np.ndarray,
    *,
    npy: bool = True,
    png: bool = False,
) -> None:
    if npy:
        np.save(path.with_suffix(".npy"), skeleton.astype(np.uint8))
    if png:
        if skeleton.ndim != 2:
            raise ValueError("PNG skeleton output is only supported for 2D images")
        img = Image.fromarray((skeleton > 0).astype(np.uint8) * 255)
        img.save(path.with_suffix(".png"))


def _save_radius(path: Path, radius_matrix: np.ndarray) -> None:
    np.save(path.with_suffix(".npy"), radius_matrix.astype(np.float64))


def process_one(
    in_path: Path,
    safe_name: str,
    out_dir: Path,
    config: PipelineConfig,
) -> dict[str, object]:
    """Load, analyse, save one image. Returns summary row for agg CSV."""
    image = _load_image(in_path)
    result = analyze_binary_image(image=image, base_name=in_path.stem, config=config)

    image_out_dir = out_dir / safe_name
    image_out_dir.mkdir(parents=True, exist_ok=True)

    if config.output.write_skeleton_npy or config.output.write_skeleton_png:
        _save_skeleton(
            image_out_dir / f"{safe_name}_skeleton",
            result.skeleton,
            npy=config.output.write_skeleton_npy,
            png=config.output.write_skeleton_png,
        )

    if config.output.write_radius and result.radius_matrix is not None:
        _save_radius(image_out_dir / f"{safe_name}_radius", result.radius_matrix)

    if config.output.write_branch_csv and result.branch_records:
        _write_csv(image_out_dir / f"{safe_name}_branches.csv", result.branch_records)

    if config.output.write_node_csv and result.node_records:
        _write_csv(image_out_dir / f"{safe_name}_nodes.csv", result.node_records)

    return {"image": in_path.name, **result.summary_features}
