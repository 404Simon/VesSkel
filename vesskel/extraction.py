"""High-level napari-layer extraction API."""

from typing import TYPE_CHECKING

import numpy as np
from skan import summarize

from vesskel.config import ExtractionConfig
from vesskel.features import (
    build_vessel_graph,
    compute_tortuosity,
    extract_vessel_features,
)

if TYPE_CHECKING:
    from napari.types import LayerDataTuple


def extract_skeleton_layers(
    skeleton: np.ndarray,
    base_name: str,
    config: ExtractionConfig | None = None,
    features: dict[str, float] | None = None,
) -> list["napari.types.LayerDataTuple"]:
    """Extract visualization layers from a binary skeleton.

    Parameters
    ----------
    skeleton : ndarray
        Binary 2D or 3D skeleton array.
    base_name : str
        Base name for layer naming.
    config : ExtractionConfig, optional
        Configuration for what to extract. Defaults to all except fractal_dimension.
    features : dict, optional
        Pre-computed feature dictionary to avoid recomputation when caller
        already has it (e.g. from extract_vessel_features).
    """
    if config is None:
        config = ExtractionConfig()

    layers = []

    if config.branches:
        branch_layer = _extract_branch_features_layer(skeleton, base_name)
        if branch_layer is not None:
            layers.append(branch_layer)

            if config.branch_text:
                text_layer = _extract_branch_text_layer(branch_layer, base_name)
                layers.append(text_layer)

    if config.summary:
        summary_layer = _extract_summary_features_layer(
            skeleton,
            base_name,
            include_fractal=config.fractal_dimension,
            features=features,
        )
        layers.append(summary_layer)

    return layers


def _extract_branch_features_layer(
    skeleton: np.ndarray,
    base_name: str,
) -> "napari.types.LayerDataTuple | None":
    """Extract branch features and generate paths layer.

    Returns None if skeleton has no branches.
    """
    graph = build_vessel_graph(skeleton)
    branch_data = summarize(graph, separator="-")

    if branch_data.empty:
        return None

    branch_data = branch_data.reset_index(drop=True).copy()
    branch_data["branch_id"] = np.arange(len(branch_data), dtype=np.int64)

    # Compute tortuosity
    euclidean = branch_data["euclidean-distance"].to_numpy(dtype=float)
    branch_len = branch_data["branch-distance"].to_numpy(dtype=float)
    tortuosity = compute_tortuosity(branch_len, euclidean)
    tortuosity = np.nan_to_num(tortuosity, nan=1.0)
    branch_data["tortuosity"] = tortuosity

    # Get branch path coordinates
    path_data = [graph.path_coordinates(i) for i in range(len(branch_data))]

    # Determine if tortuosity varies significantly
    finite_tortuosity = tortuosity[np.isfinite(tortuosity)]
    varied_tortuosity = finite_tortuosity.size > 0 and float(
        np.min(finite_tortuosity)
    ) < float(np.max(finite_tortuosity))

    meta = {
        "name": f"{base_name}_branches",
        "shape_type": "path",
        "properties": branch_data,
        "face_color": "transparent",
        "edge_width": 0.5,
        "opacity": 0.95,
    }

    if varied_tortuosity:
        vmin = float(np.min(finite_tortuosity))
        vmax = float(np.max(finite_tortuosity))
        meta["edge_color"] = "tortuosity"
        meta["edge_colormap"] = "turbo"
        meta["edge_contrast_limits"] = (vmin, vmax)
    else:
        meta["edge_color"] = "#30d5c8"

    return (path_data, meta, "shapes")


def _extract_branch_text_layer(
    branch_layer: "napari.types.LayerDataTuple",
    base_name: str,
) -> "napari.types.LayerDataTuple":
    """Create text labels for branches."""
    path_data = branch_layer[0]
    branch_data = branch_layer[1]["properties"]

    # Compute label positions as mean of each path
    label_points = []
    for coords in path_data:
        if len(coords) == 0:
            label_points.append(np.zeros((coords.shape[1],), dtype=float))
            continue
        label_points.append(np.asarray(coords, dtype=float).mean(axis=0))

    points = np.asarray(label_points, dtype=float)
    meta = {
        "name": f"{base_name}_branch_text",
        "properties": branch_data,
        "symbol": "disc",
        "size": 1,
        "face_color": "transparent",
        "border_color": "transparent",
        "opacity": 1.0,
        "text": {
            "string": "id {branch_id} | L={branch-distance:.1f} | T={tortuosity:.2f}",
            "size": 9,
            "color": "white",
            "anchor": "center",
        },
    }
    return (points, meta, "points")


def _extract_summary_features_layer(
    skeleton: np.ndarray,
    base_name: str,
    include_fractal: bool = False,
    features: dict[str, float] | None = None,
) -> "napari.types.LayerDataTuple":
    """Extract global skeleton features and create summary point layer.

    Parameters
    ----------
    features : dict, optional
        Pre-computed feature dictionary. If provided, skips computation.
    """
    if features is None:
        features = extract_vessel_features(
            skeleton,
            include_fractal=include_fractal,
        )
    meta_features = {k: [v] for k, v in features.items()}

    # Find center of foreground
    fg = np.argwhere(skeleton > 0)
    if fg.size:
        center = fg.mean(axis=0, dtype=float)
    else:
        center = np.zeros(skeleton.ndim, dtype=float)

    points = np.asarray([center], dtype=float)
    meta = {
        "name": f"{base_name}_summary",
        "properties": meta_features,
        "symbol": "ring",
        "size": 8,
        "face_color": "transparent",
        "border_color": "yellow",
        "opacity": 0.9,
        "text": {
            "string": "summary",
            "size": 10,
            "color": "yellow",
            "anchor": "upper_left",
        },
    }
    return (points, meta, "points")
