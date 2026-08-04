import csv
import hashlib
from pathlib import Path

import numpy as np

BASELINE_DIR = Path(__file__).parent / "skeletons"
FEATURE_DIR = Path(__file__).parent / "features"


def skeleton_path(name: str) -> Path:
    return BASELINE_DIR / f"skeleton_{name}.npz"


def feature_path(name: str) -> Path:
    return FEATURE_DIR / f"features_{name}.csv"


def write_feature_csv(path: Path, features: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "value"])
        for key in sorted(features):
            writer.writerow([key, f"{features[key]:.17g}"])


def read_feature_csv(path: Path) -> dict[str, float]:
    loaded: dict[str, float] = {}
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header != ["feature", "value"]:
            raise ValueError("invalid feature csv header")
        for row in reader:
            if len(row) != 2:
                raise ValueError("invalid feature csv row")
            key, value = row
            loaded[key] = float(value)
    return loaded


def hash_array(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()


def cross_image(size: int = 32) -> np.ndarray:
    """Binary cross: one junction and four endpoints."""
    img = np.zeros((size, size), dtype=np.uint8)
    img[size // 2, size // 4 : 3 * size // 4] = 1
    img[size // 4 : 3 * size // 4, size // 2] = 1
    return img


def loop_image(size: int = 20) -> np.ndarray:
    """Binary rectangle ring: two junctions connected by parallel branches."""
    img = np.zeros((size, size), dtype=np.uint8)
    margin = size // 4
    inner = size - margin
    img[margin, margin:inner] = 1
    img[inner - 1, margin:inner] = 1
    img[margin:inner, margin] = 1
    img[margin:inner, inner - 1] = 1
    return img


def cross_volume(size: int = 16) -> np.ndarray:
    """Binary volume with two perpendicular lines crossing at the center."""
    vol = np.zeros((size, size, size), dtype=np.uint8)
    vol[size // 2, size // 2, :] = 1
    vol[size // 2, :, size // 2] = 1
    return vol


def line_volume(shape: tuple[int, int, int], axis: int = 0) -> np.ndarray:
    """Binary volume with a single straight line through the center along *axis*."""
    vol = np.zeros(shape, dtype=np.uint8)
    idx = [s // 2 for s in shape]
    idx[axis] = slice(None)
    vol[tuple(idx)] = 1
    return vol
