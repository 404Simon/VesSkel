"""Shared configuration models and helpers."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

CONFIG_SCHEMA_VERSION = 2


def _warn_unknown_keys(known: set[str], data: dict[str, Any]) -> None:
    unknown = set(data) - known
    if unknown:
        print(
            f"Warning: ignored unknown keys: {sorted(unknown)}",
            file=sys.stderr,
        )


@dataclass
class ExtractionConfig:
    """Configuration for what to extract from a skeleton."""

    branches: bool = False
    branch_text: bool = False
    summary: bool = False
    fractal_dimension: bool = False
    vessel_radius: bool = False
    junction_cleanup: bool = False
    cleanup_threshold_factor: float = 2.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "branches": self.branches,
            "branch_text": self.branch_text,
            "summary": self.summary,
            "fractal_dimension": self.fractal_dimension,
            "vessel_radius": self.vessel_radius,
            "junction_cleanup": self.junction_cleanup,
            "cleanup_threshold_factor": self.cleanup_threshold_factor,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExtractionConfig:
        _warn_unknown_keys({f.name for f in fields(cls)}, data)
        return cls(
            branches=data.get("branches", False),
            branch_text=data.get("branch_text", False),
            summary=data.get("summary", False),
            fractal_dimension=data.get("fractal_dimension", False),
            vessel_radius=data.get("vessel_radius", False),
            junction_cleanup=data.get("junction_cleanup", False),
            cleanup_threshold_factor=data.get("cleanup_threshold_factor", 2.5),
        )


@dataclass
class OutputConfig:
    """Output controls for batch CLI runs."""

    write_skeleton_npy: bool = True
    write_skeleton_png: bool = False
    write_summary_csv: bool = True
    write_branch_csv: bool = False
    write_radius: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "write_skeleton_npy": self.write_skeleton_npy,
            "write_skeleton_png": self.write_skeleton_png,
            "write_summary_csv": self.write_summary_csv,
            "write_branch_csv": self.write_branch_csv,
            "write_radius": self.write_radius,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> OutputConfig:
        data = data or {}
        _warn_unknown_keys({f.name for f in fields(cls)}, data)
        return cls(
            write_skeleton_npy=bool(data.get("write_skeleton_npy", True)),
            write_skeleton_png=bool(data.get("write_skeleton_png", False)),
            write_summary_csv=bool(data.get("write_summary_csv", True)),
            write_branch_csv=bool(data.get("write_branch_csv", False)),
            write_radius=bool(data.get("write_radius", False)),
        )


@dataclass
class PipelineConfig:
    """Top-level config shared by napari and batch CLI."""

    extraction: ExtractionConfig
    output: OutputConfig
    schema_version: int = CONFIG_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "extraction": self.extraction.to_dict(),
            "output": self.output.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        if not isinstance(data, dict):
            raise ValueError("Config JSON must be an object")

        _warn_unknown_keys({"schema_version", "extraction", "output"}, data)

        schema_version = int(data.get("schema_version", CONFIG_SCHEMA_VERSION))
        if schema_version != CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version={schema_version}. "
                f"Expected {CONFIG_SCHEMA_VERSION}."
            )

        extraction_data = data.get("extraction", {})
        output_data = data.get("output", {})

        if not isinstance(extraction_data, dict):
            raise ValueError("'extraction' must be an object")
        if not isinstance(output_data, dict):
            raise ValueError("'output' must be an object")

        return cls(
            extraction=ExtractionConfig.from_dict(extraction_data),
            output=OutputConfig.from_dict(output_data),
            schema_version=schema_version,
        )


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    """Load and parse a pipeline config from JSON file."""
    with Path(path).open() as f:
        return PipelineConfig.from_dict(json.load(f))


def save_pipeline_config(config: PipelineConfig, path: str | Path) -> None:
    """Save a pipeline config to JSON file."""
    with Path(path).open("w") as f:
        json.dump(config.to_dict(), f, indent=2)
