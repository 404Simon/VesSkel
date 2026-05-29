"""Standalone CLI for VesSkel batch analysis."""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from argcomplete import autocomplete, shellcode
from PIL import Image

from vesskel.config import (
    ExtractionConfig,
    OutputConfig,
    PipelineConfig,
    load_pipeline_config,
)
from vesskel.pipeline import analyze_binary_image

_SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".npy",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="vesskel",
        description="VesSkel CLI for batch-vessel-analysis.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Run batch analysis on one or more images using a config JSON.",
    )
    run_parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Input files, directories, or glob patterns.",
    )
    run_parser.add_argument(
        "--config",
        required=True,
        help="Path to pipeline config JSON (can be exported from napari).",
    )
    run_parser.add_argument(
        "--out",
        required=True,
        help="Output directory for CSVs and optional skeletons.",
    )
    run_parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search directories for supported image files.",
    )

    init_parser = subparsers.add_parser(
        "config-init",
        help="Create a starter config JSON.",
    )
    init_parser.add_argument("--out", required=True, help="Output config path.")

    validate_parser = subparsers.add_parser(
        "validate-config",
        help="Validate and print normalized config JSON.",
    )
    validate_parser.add_argument("--config", required=True, help="Config JSON path.")

    completions_parser = subparsers.add_parser(
        "completions",
        help="Print shell completion script to stdout.",
    )
    completions_parser.add_argument(
        "shell",
        choices=("bash", "zsh", "powershell"),
        help="Target shell.",
    )

    try:
        autocomplete(parser)
    except ImportError:
        pass
    return parser.parse_args()


def _discover_input_paths(inputs: list[str], recursive: bool) -> list[Path]:
    paths: set[Path] = set()

    for raw in inputs:
        token = Path(raw)
        if token.exists():
            if token.is_file():
                if token.suffix.lower() in _SUPPORTED_EXTENSIONS:
                    paths.add(token.resolve())
            elif token.is_dir():
                pattern = "**/*" if recursive else "*"
                for p in token.glob(pattern):
                    if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTENSIONS:
                        paths.add(p.resolve())
            continue

        for match in glob.glob(raw, recursive=recursive):
            p = Path(match)
            if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTENSIONS:
                paths.add(p.resolve())

    return sorted(paths)


def _load_image(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    else:
        with Image.open(path) as im:
            arr = np.asarray(im)

    if arr.ndim == 0:
        raise ValueError("Scalar input is not supported")

    # If image has channels, collapse to single binary mask
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


def _run_batch(args: argparse.Namespace) -> int:
    config = load_pipeline_config(Path(args.config))
    input_paths = _discover_input_paths(args.input, recursive=args.recursive)
    if not input_paths:
        raise ValueError("No input files found. Check --input and --recursive.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    output_name_counts: dict[str, int] = {}
    total = len(input_paths)

    for idx, in_path in enumerate(input_paths, 1):
        print(f"[{idx}/{total}] {in_path.name}", flush=True)

        image = _load_image(in_path)
        result = analyze_binary_image(
            image=image, base_name=in_path.stem, config=config
        )

        base_key = in_path.stem
        seen = output_name_counts.get(base_key, 0)
        output_name_counts[base_key] = seen + 1
        safe_name = base_key if seen == 0 else f"{base_key}_{seen + 1}"

        image_out_dir = out_dir / safe_name
        image_out_dir.mkdir(parents=True, exist_ok=True)

        if config.output.write_skeleton_npy or config.output.write_skeleton_png:
            _save_skeleton(
                path=image_out_dir / f"{safe_name}_skeleton",
                skeleton=result.skeleton,
                npy=config.output.write_skeleton_npy,
                png=config.output.write_skeleton_png,
            )

        if config.output.write_radius and result.radius_matrix is not None:
            _save_radius(
                image_out_dir / f"{safe_name}_radius",
                result.radius_matrix,
            )

        if config.output.write_branch_csv and result.branch_records:
            _write_csv(
                image_out_dir / f"{safe_name}_branches.csv",
                result.branch_records,
            )

        summary_row = {"image": in_path.name, **result.summary_features}
        summary_rows.append(summary_row)

    if config.output.write_summary_csv:
        _write_csv(out_dir / "summary.csv", summary_rows)

    print(
        f"Processed {len(input_paths)} image(s) with config '{args.config}'. "
        f"Outputs written to '{out_dir}'."
    )
    return 0


def _config_init(args: argparse.Namespace) -> int:
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = PipelineConfig(
        extraction=ExtractionConfig(),
        output=OutputConfig(),
    )
    with path.open("w") as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"Wrote starter config to '{path}'.")
    return 0


def _validate_config(args: argparse.Namespace) -> int:
    config = load_pipeline_config(Path(args.config))
    print(json.dumps(config.to_dict(), indent=2))
    print("Configuration is valid.")
    return 0


def _completions(args: argparse.Namespace) -> int:

    print(shellcode(["vesskel"], shell=args.shell))
    return 0


def main() -> int:
    args = _parse_args()
    if args.command == "run":
        return _run_batch(args)
    if args.command == "config-init":
        return _config_init(args)
    if args.command == "validate-config":
        return _validate_config(args)
    if args.command == "completions":
        return _completions(args)
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
