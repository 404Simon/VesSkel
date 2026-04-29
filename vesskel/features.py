import numpy as np
from numba import njit, prange
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from skan import Skeleton, summarize


@njit(parallel=True, cache=True)
def _box_count_2d(img: np.ndarray, scale: int) -> int:
    """Count occupied boxes of side `scale` in a 2D binary image."""
    H, W = img.shape
    nh = (H + scale - 1) // scale
    nw = (W + scale - 1) // scale
    count = 0
    for i in prange(nh):
        for j in range(nw):
            r0 = i * scale
            c0 = j * scale
            r1 = min(r0 + scale, H)
            c1 = min(c0 + scale, W)
            found = False
            for r in range(r0, r1):
                if found:
                    break
                for c in range(c0, c1):
                    if img[r, c]:
                        found = True
                        break
            if found:
                count += 1
    return count


@njit(parallel=True, cache=True)
def _box_count_3d(img: np.ndarray, scale: int) -> int:
    """Count occupied boxes of side `scale` in a 3D binary volume."""
    D, H, W = img.shape
    nd = (D + scale - 1) // scale
    nh = (H + scale - 1) // scale
    nw = (W + scale - 1) // scale
    count = 0
    for i in prange(nd):
        for j in range(nh):
            for k in range(nw):
                z0 = i * scale
                y0 = j * scale
                x0 = k * scale
                z1 = min(z0 + scale, D)
                y1 = min(y0 + scale, H)
                x1 = min(x0 + scale, W)
                found = False
                for z in range(z0, z1):
                    if found:
                        break
                    for y in range(y0, y1):
                        if found:
                            break
                        for x in range(x0, x1):
                            if img[z, y, x]:
                                found = True
                                break
                if found:
                    count += 1
    return count


def fractal_dimension(skeleton: np.ndarray) -> tuple[float, float]:
    """Estimate fractal dimension of a binary skeleton via box-counting.

    Returns
    -------
    fd : float
        Fractal dimension (slope magnitude of log-log fit).
    r2 : float
        R^2 of the log-log linear fit (fit quality indicator).
    """
    binary = (skeleton > 0).view(np.uint8)
    min_side = min(binary.shape)
    max_exp = int(np.floor(np.log2(min_side // 4)))
    if max_exp < 1:
        return 0.0, 0.0

    scales = np.array([2**k for k in range(1, max_exp + 1)], dtype=np.int64)

    box_counter = _box_count_2d if binary.ndim == 2 else _box_count_3d
    counts = np.array([box_counter(binary, int(s)) for s in scales], dtype=np.float64)

    # scales with zero count collapses the log
    valid = counts > 0
    if valid.sum() < 2:
        return 0.0, 0.0

    log_s = np.log(scales[valid].astype(np.float64))
    log_n = np.log(counts[valid])

    coeffs = np.polyfit(log_s, log_n, 1)
    fd = float(-coeffs[0])

    predicted = np.polyval(coeffs, log_s)
    ss_res = float(np.sum((log_n - predicted) ** 2))
    ss_tot = float(np.sum((log_n - log_n.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0

    return fd, float(r2)


def build_vessel_graph(skeleton: np.ndarray) -> Skeleton:
    """Build a graph representation from a binary vessel skeleton."""
    return Skeleton((skeleton > 0).astype(np.uint8))


def extract_vessel_features(
    skeleton: np.ndarray,
) -> dict[str, float]:
    """Extract graph-topology and segment statistics from a vessel skeleton."""
    graph = build_vessel_graph(skeleton)
    branch_data = summarize(graph, separator="-")
    if branch_data.empty:
        return {
            "num_nodes": 0.0,
            "num_edges": 0.0,
            "num_endpoints": 0.0,
            "num_bifurcations": 0.0,
            "total_length": 0.0,
            "mean_length": 0.0,
            "std_length": 0.0,
            "max_length": 0.0,
            "min_length": 0.0,
            "mean_tortuosity": 0.0,
            "std_tortuosity": 0.0,
            "num_components": 0.0,
            "mean_degree": 0.0,
            "max_degree": 0.0,
            "fractal_dimension": 0.0,
            "fractal_dimension_r2": 0.0,
        }

    fd, fd_r2 = fractal_dimension(skeleton)

    src_nodes = branch_data["node-id-src"].to_numpy(dtype=np.int64)
    dst_nodes = branch_data["node-id-dst"].to_numpy(dtype=np.int64)
    edge_nodes = np.concatenate((src_nodes, dst_nodes))
    unique_nodes = np.unique(edge_nodes)

    num_edges = int(len(branch_data))
    num_nodes = int(unique_nodes.size)

    max_node_id = int(np.max(unique_nodes))
    degrees = np.bincount(edge_nodes, minlength=max_node_id + 1)
    node_degrees = degrees[unique_nodes]

    num_endpoints = int(np.count_nonzero(node_degrees == 1))
    num_bifurcations = int(np.count_nonzero(node_degrees >= 3))
    mean_degree = float(np.mean(node_degrees)) if node_degrees.size else 0.0
    max_degree = float(np.max(node_degrees)) if node_degrees.size else 0.0

    node_to_index = {node: idx for idx, node in enumerate(unique_nodes)}
    src_idx = np.fromiter((node_to_index[src] for src in src_nodes), dtype=np.int64)
    dst_idx = np.fromiter((node_to_index[dst] for dst in dst_nodes), dtype=np.int64)
    adjacency = coo_matrix(
        (
            np.ones(src_idx.size * 2, dtype=np.uint8),
            (
                np.concatenate((src_idx, dst_idx)),
                np.concatenate((dst_idx, src_idx)),
            ),
        ),
        shape=(num_nodes, num_nodes),
    ).tocsr()
    num_components = int(
        connected_components(adjacency, directed=False, return_labels=False)
    )

    lengths = branch_data["branch-distance"].to_numpy(dtype=float)
    euclidean = branch_data["euclidean-distance"].to_numpy(dtype=float)

    if lengths.size:
        total_length = float(np.sum(lengths))
        mean_length = float(np.mean(lengths))
        std_length = float(np.std(lengths))
        max_length = float(np.max(lengths))
        min_length = float(np.min(lengths))
    else:
        total_length = 0.0
        mean_length = 0.0
        std_length = 0.0
        max_length = 0.0
        min_length = 0.0

    valid_tortuosity = euclidean > 0
    tortuosity = lengths[valid_tortuosity] / euclidean[valid_tortuosity]
    if tortuosity.size:
        mean_tortuosity = float(np.mean(tortuosity))
        std_tortuosity = float(np.std(tortuosity))
    else:
        mean_tortuosity = 0.0
        std_tortuosity = 0.0

    return {
        "num_nodes": float(num_nodes),
        "num_edges": float(num_edges),
        "num_endpoints": float(num_endpoints),
        "num_bifurcations": float(num_bifurcations),
        "total_length": total_length,
        "mean_length": mean_length,
        "std_length": std_length,
        "max_length": max_length,
        "min_length": min_length,
        "mean_tortuosity": mean_tortuosity,
        "std_tortuosity": std_tortuosity,
        "num_components": float(num_components),
        "mean_degree": mean_degree,
        "max_degree": max_degree,
        "fractal_dimension": fd,
        "fractal_dimension_r2": fd_r2,
    }
