import time

import numpy as np
from napari.layers import Image, Labels
from napari.types import LayerDataTuple
from napari.utils.notifications import show_info
from skan import summarize

from vesskel.features import build_vessel_graph, extract_vessel_features
from vesskel.thin import lee94_thin


def _branch_features_layer_data(
    skeleton: np.ndarray,
    base_name: str,
) -> LayerDataTuple | None:
    graph = build_vessel_graph(skeleton)
    branch_data = summarize(graph, separator="-")

    if branch_data.empty:
        return None

    branch_data = branch_data.reset_index(drop=True).copy()
    branch_data["branch_id"] = np.arange(len(branch_data), dtype=np.int64)
    euclidean = branch_data["euclidean-distance"].to_numpy(dtype=float)
    branch_len = branch_data["branch-distance"].to_numpy(dtype=float)
    tortuosity = np.ones_like(branch_len, dtype=float)
    np.divide(branch_len, euclidean, out=tortuosity, where=euclidean > 0)
    branch_data["tortuosity"] = tortuosity

    path_data = [graph.path_coordinates(i) for i in range(len(branch_data))]
    finite_tortuosity = tortuosity[np.isfinite(tortuosity)]
    varied_tortuosity = finite_tortuosity.size > 0 and float(
        np.min(finite_tortuosity)
    ) < float(np.max(finite_tortuosity))

    meta = {
        "name": f"{base_name}_branches",
        "shape_type": "path",
        "features": branch_data,
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


def _branch_text_layer_data(
    branch_layer: LayerDataTuple,
    base_name: str,
) -> LayerDataTuple:
    path_data = branch_layer[0]
    branch_data = branch_layer[1]["features"]

    label_points = []
    for coords in path_data:
        if len(coords) == 0:
            label_points.append(np.zeros((coords.shape[1],), dtype=float))
            continue
        label_points.append(np.asarray(coords, dtype=float).mean(axis=0))

    points = np.asarray(label_points, dtype=float)
    meta = {
        "name": f"{base_name}_branch_text",
        "features": branch_data,
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


def _summary_features_layer_data(
    skeleton: np.ndarray,
    base_name: str,
) -> LayerDataTuple:
    feature_dict = extract_vessel_features(skeleton)
    features = {k: [v] for k, v in feature_dict.items()}

    fg = np.argwhere(skeleton > 0)
    if fg.size:
        center = fg.mean(axis=0, dtype=float)
    else:
        center = np.zeros(skeleton.ndim, dtype=float)

    points = np.asarray([center], dtype=float)
    meta = {
        "name": f"{base_name}_summary",
        "features": features,
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


def lee94_thin_widget(
    img: Image,
) -> LayerDataTuple:
    data = img.data
    binary = (data > 0).astype(np.uint8)
    n_fg = int(binary.sum())
    t0 = time.perf_counter()
    result = lee94_thin(binary)
    elapsed = time.perf_counter() - t0
    n_skel = int(result.sum())
    show_info(
        f"Thinned {img.name}: {n_fg} -> {n_skel} pixels "
        f"({100 * n_skel / n_fg:.1f}% of foreground) in {elapsed:.1f}s"
    )
    return (result, {"name": f"{img.name}_thinned"}, "labels")


def extract_branch_features_widget(
    img: Labels,
) -> list[LayerDataTuple]:
    data = img.data
    skeleton = (data > 0).astype(np.uint8)
    t0 = time.perf_counter()
    branch_layer = _branch_features_layer_data(skeleton, img.name)
    summary_layer = _summary_features_layer_data(skeleton, img.name)
    global_features = summary_layer[1]["features"]
    elapsed = time.perf_counter() - t0

    if branch_layer is None:
        show_info(f"No branches found in {img.name}.")
        return []

    branch_text_layer = _branch_text_layer_data(branch_layer, img.name)

    n_branches = len(branch_layer[1]["features"])
    n_components = int(global_features["num_components"][0])
    total_length = float(global_features["total_length"][0])
    show_info(
        f"Extracted {n_branches} branches from {img.name} "
        f"({n_components} components, total length {total_length:.1f}) in {elapsed:.1f}s. "
        f"Open Layers -> Visualize -> Features Table Widget to inspect full tables."
    )
    return [branch_layer, branch_text_layer, summary_layer]
