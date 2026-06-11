"""Shared analysis pipeline used by napari and CLI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import ndimage as ndi
from skan import summarize

from vesskel._utils import to_binary
from vesskel.config import PipelineConfig
from vesskel.features import (
    build_vessel_graph,
    compute_radii,
    compute_tortuosity,
    extract_vessel_features,
    per_segment_radii,
)
from vesskel.junction_cleanup import collapse_triangle_junctions
from vesskel.napari_layers import extract_skeleton_layers
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
    preprocessed_binary: np.ndarray | None = None


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

    # -- optional: morphological preprocessing -------------------------
    preprocessed_binary: np.ndarray | None = None
    if config.extraction.closing_iterations > 0:
        structure = ndi.generate_binary_structure(binary.ndim, 1)
        binary = ndi.binary_closing(
            binary, structure=structure, iterations=config.extraction.closing_iterations
        ).astype(binary.dtype)

    if config.extraction.fill_holes:
        before_fill = binary.copy()
        binary = ndi.binary_fill_holes(binary).astype(binary.dtype)

        if config.extraction.max_hole_size > 0:
            diff = binary.astype(np.int8) - before_fill.astype(np.int8)
            filled = diff > 0
            if filled.any():
                labels, n = ndi.label(filled)
                sizes = np.bincount(labels.ravel())
                big = sizes > config.extraction.max_hole_size
                big[0] = False
                revert = big[labels]
                binary[revert] = 0

    if config.extraction.closing_iterations > 0 or config.extraction.fill_holes:
        preprocessed_binary = binary

    skeleton = lee94_thin(binary)

    if not skeleton.any():
        return AnalysisResult(
            skeleton=skeleton,
            layers=[],
            summary_features={},
            branch_records=[],
        )

    # -- optional: collapse triangle junction artifacts -----------------
    # (requires EDT; done on the original skeleton before graph building)
    if config.extraction.junction_cleanup:
        rm_temp, _ = compute_radii(binary, skeleton)
        skeleton = collapse_triangle_junctions(
            skeleton,
            radius_matrix=rm_temp,
            threshold_factor=config.extraction.cleanup_threshold_factor,
        )

    # -- optional: vessel radius (EDT on the final skeleton) ------------
    radius_matrix = None
    radius_stats = None
    if config.extraction.vessel_radius:
        radius_matrix, radius_stats = compute_radii(binary, skeleton)

    # -- build graph & branch data on the (potentially cleaned) skeleton -
    graph = build_vessel_graph(skeleton)
    branch_data = summarize(graph, separator="-")

    if radius_matrix is not None and not branch_data.empty:
        per_seg = per_segment_radii(radius_matrix, graph, len(branch_data))
        for key, arr in per_seg.items():
            branch_data[key] = arr

    if not branch_data.empty:
        euclidean = branch_data["euclidean-distance"].to_numpy(dtype=float)
        branch_dist = branch_data["branch-distance"].to_numpy(dtype=float)
        tortuosity = compute_tortuosity(branch_dist, euclidean)
        branch_data["tortuosity"] = tortuosity
        straightness = np.full_like(branch_dist, np.nan, dtype=float)
        valid = branch_dist > 0
        straightness[valid] = euclidean[valid] / branch_dist[valid]
        branch_data["straightness"] = straightness

    summary_features: dict[str, float] = {}
    if config.extraction.summary:
        summary_features = extract_vessel_features(
            skeleton,
            graph,
            branch_data,
            binary=binary,
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
        preprocessed_binary=preprocessed_binary,
    )
