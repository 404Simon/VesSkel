"""High-level napari-layer extraction API."""

from typing import TYPE_CHECKING

import numpy as np
from skan import Skeleton

from vesskel.config import ExtractionConfig
from vesskel.features import compute_tortuosity, extract_node_features

if TYPE_CHECKING:
    from napari.types import LayerDataTuple


def extract_skeleton_layers(
    skeleton: np.ndarray,
    base_name: str,
    graph: Skeleton,
    branch_data,
    config: ExtractionConfig | None = None,
    features: dict[str, float] | None = None,
    radius_matrix: np.ndarray | None = None,
) -> list["napari.types.LayerDataTuple"]:
    """Extract visualization layers from a binary skeleton.

    Parameters
    ----------
    skeleton : ndarray
        Binary 2D or 3D skeleton array.
    base_name : str
        Base name for layer naming.
    graph : Skeleton
        Pre-built skan Skeleton graph (e.g. from `build_vessel_graph`).
    branch_data : DataFrame
        Pre-computed branch summary (e.g. from `skan.summarize(graph, ...)`).
    config : ExtractionConfig, optional
        Configuration for what to extract. Defaults to all except fractal_dimension.
    features : dict, optional
        Pre-computed summary feature dictionary (e.g. from `extract_vessel_features`).
    radius_matrix : ndarray, optional
        Radius matrix from `compute_radii`. When provided, a napari image layer
        is added showing per-pixel vessel radius on the skeleton.
    """
    if config is None:
        config = ExtractionConfig()

    layers = []

    if config.branches:
        branch_layer = _extract_branch_features_layer(base_name, graph, branch_data)
        if branch_layer is not None:
            layers.append(branch_layer)

            if config.branch_text:
                text_layer = _extract_branch_text_layer(branch_layer, base_name)
                layers.append(text_layer)

    if config.nodes:
        node_layer = _extract_node_features_layer(base_name, graph, branch_data, radius_matrix=radius_matrix)
        if node_layer is not None:
            layers.append(node_layer)

    if config.summary:
        if features is None:
            raise ValueError(
                "features is required when summary is enabled"
            )
        summary_layer = _extract_summary_features_layer(
            skeleton,
            base_name,
            features=features,
        )
        layers.append(summary_layer)

    if radius_matrix is not None:
        radius_layer = _extract_radius_layer(radius_matrix, skeleton, base_name)
        if radius_layer is not None:
            layers.append(radius_layer)

    return layers


def _extract_radius_layer(
    radius_matrix: np.ndarray,
    skeleton: np.ndarray,
    base_name: str,
) -> "napari.types.LayerDataTuple | None":
    """Create an image layer showing per-pixel vessel radius on the skeleton."""
    if not np.any(radius_matrix):
        return None

    display = np.where(skeleton > 0, radius_matrix, np.nan)
    meta = {
        "name": f"{base_name}_radius",
        "colormap": "turbo",
        "blending": "additive",
        "opacity": 0.85,
    }
    return (display, meta, "image")


def _extract_branch_features_layer(
    base_name: str,
    graph: Skeleton,
    branch_data,
) -> "napari.types.LayerDataTuple | None":
    """Extract branch features and generate paths layer.

    Parameters
    ----------
    base_name : str
        Base name used for layer naming.
    graph : Skeleton
        Pre-built skan Skeleton graph.
    branch_data : DataFrame
        Pre-computed branch summary from `skan.summarize`.

    Returns
    -------
    LayerDataTuple or None
        Napari shapes layer for branch paths, or None if skeleton has no branches.
    """
    if branch_data.empty:
        return None

    branch_data = branch_data.reset_index(drop=True).copy()
    branch_data["branch_id"] = np.arange(len(branch_data), dtype=np.int64)

    euclidean = branch_data["euclidean-distance"].to_numpy(dtype=float)
    branch_len = branch_data["branch-distance"].to_numpy(dtype=float)
    tortuosity = compute_tortuosity(branch_len, euclidean)
    tortuosity = np.nan_to_num(tortuosity, nan=1.0)
    branch_data["tortuosity"] = tortuosity

    path_data = [graph.path_coordinates(i) for i in range(len(branch_data))]

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
    path_data = branch_layer[0]
    branch_data = branch_layer[1]["properties"]

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
    features: dict[str, float],
) -> "napari.types.LayerDataTuple":
    """Create a summary point layer displaying global skeleton features.

    Parameters
    ----------
    skeleton : ndarray
        Binary 2D or 3D skeleton array. Used to position the summary label.
    base_name : str
        Base name for layer naming.
    features : dict[str, float]
        Pre-computed summary feature dictionary
        (e.g. from `extract_vessel_features`).
    """
    meta_features = {k: [v] for k, v in features.items()}

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


def _extract_node_features_layer(
    base_name: str,
    graph: Skeleton,
    branch_data,
    radius_matrix: np.ndarray | None = None,
) -> "napari.types.LayerDataTuple | None":
    """Create a points layer showing graph nodes colored by degree."""
    node_records = extract_node_features(graph, branch_data, radius_matrix=radius_matrix)
    if not node_records:
        return None

    ndim = graph.coordinates.shape[1]
    points = np.array([tuple(r[f"coord_{d}"] for d in range(ndim)) for r in node_records], dtype=float)

    props = {k: [r[k] for r in node_records] for k in node_records[0].keys()
             if not k.startswith("coord_")}

    meta = {
        "name": f"{base_name}_nodes",
        "properties": props,
        "symbol": "disc",
        "size": 3,
        "face_color": "degree",
        "face_colormap": "viridis",
        "opacity": 0.8,
    }
    return (points, meta, "points")
