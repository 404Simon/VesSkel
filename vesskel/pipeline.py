"""Shared analysis pipeline used by napari and CLI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from skan import summarize

from vesskel._utils import to_binary
from vesskel.config import PipelineConfig
from vesskel.extraction import extract_skeleton_layers
from vesskel.features import build_vessel_graph, compute_radii, extract_vessel_features
from vesskel.thin import lee94_thin

if TYPE_CHECKING:
    from napari.types import LayerDataTuple


@dataclass
class AnalysisResult:
    """Container for single-image analysis outputs."""

    skeleton: np.ndarray
    layers: list["napari.types.LayerDataTuple"]
    summary_features: dict[str, float]
    branch_records: list[dict[str, object]]
    radius_matrix: np.ndarray | None = None


def analyze_binary_image(
    image: np.ndarray,
    base_name: str,
    config: PipelineConfig,
) -> AnalysisResult:
    """Run the full skeletonization + extraction pipeline for one image.

    Parameters
    ----------
    image : ndarray
        Input image array. Non-zero values are treated as foreground.
    base_name : str
        Base name used for generated layer names.
    config : PipelineConfig
        Full pipeline configuration.
    """
    binary = to_binary(image)
    skeleton = lee94_thin(binary)

    if not skeleton.any():
        return AnalysisResult(
            skeleton=skeleton,
            layers=[],
            summary_features={},
            branch_records=[],
        )

    graph = build_vessel_graph(skeleton)
    branch_data = summarize(graph, separator="-")

    radius_matrix = None
    radius_stats = None
    if config.extraction.vessel_radius:
        radius_matrix, radius_stats = compute_radii(binary, skeleton)

    summary_features: dict[str, float] = {}
    if config.extraction.summary:
        summary_features = extract_vessel_features(
            skeleton,
            graph,
            branch_data,
            include_fractal=config.extraction.fractal_dimension,
            radius_stats=radius_stats,
        )

    layers = extract_skeleton_layers(
        skeleton,
        base_name,
        graph=graph,
        branch_data=branch_data,
        config=config.extraction,
        features=summary_features if config.extraction.summary else None,
        radius_matrix=radius_matrix,
    )

    branch_records: list[dict[str, object]] = []
    if config.extraction.branches and not branch_data.empty:
        branch_records = branch_data.to_dict(orient="records")

    return AnalysisResult(
        skeleton=skeleton,
        layers=layers,
        summary_features=summary_features,
        branch_records=branch_records,
        radius_matrix=radius_matrix,
    )
