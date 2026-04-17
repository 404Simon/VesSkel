"""Regression test for 3D thinning using brain image from scikit-image."""

import csv
import hashlib
from pathlib import Path

import numpy as np
import pytest
from skimage import data

from vesskel.features import extract_vessel_features
from vesskel.thin import lee94_thin

BASELINE_DIR = Path(__file__).parent / "skeletons"
FEATURE_DIR = Path(__file__).parent / "features"


def _skeleton_path(name: str) -> Path:
    return BASELINE_DIR / f"skeleton_{name}.npz"


def _feature_path(name: str) -> Path:
    return FEATURE_DIR / f"features_{name}.csv"


def _compute_skeleton(image: np.ndarray) -> np.ndarray:
    return lee94_thin(image)


def _write_feature_csv(path: Path, features: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "value"])
        for key in sorted(features):
            writer.writerow([key, f"{features[key]:.17g}"])


def _read_feature_csv(path: Path) -> dict[str, float]:
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


def _hash_array(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()


class Test3DThinningRegression:
    """Run 3D thinning on brain image and compare against saved baselines."""

    @pytest.fixture(scope="class")
    def image(self):
        return data.brain()

    def test_skeleton_matches_baseline(self, image, request):
        skeleton = _compute_skeleton(image)
        features = extract_vessel_features(skeleton)
        name = "brain"
        baseline_file = _skeleton_path(name)
        feature_file = _feature_path(name)

        baseline_changed = False
        if request.config.getoption("--update-baseline") or not baseline_file.exists():
            BASELINE_DIR.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(baseline_file, skeleton=skeleton)
            baseline_changed = True

        features_changed = False
        if request.config.getoption("--update-baseline") or not feature_file.exists():
            FEATURE_DIR.mkdir(parents=True, exist_ok=True)
            _write_feature_csv(feature_file, features)
            features_changed = True

        if baseline_changed or features_changed:
            artifacts = []
            if baseline_changed:
                artifacts.append("skeleton baseline")
            if features_changed:
                artifacts.append("feature baseline")
            pytest.skip(f"{', '.join(artifacts)} created for baseline")

        with np.load(baseline_file) as data:
            baseline = data["skeleton"]

        assert (
            skeleton.shape == baseline.shape
        ), f"shape mismatch got {skeleton.shape}, expected {baseline.shape}"
        assert np.array_equal(
            skeleton, baseline
        ), f"skeleton differs (hash {_hash_array(skeleton)} vs {_hash_array(baseline)})"

        baseline_features = _read_feature_csv(feature_file)
        feature_keys = sorted(features)
        baseline_feature_keys = sorted(baseline_features)
        assert (
            feature_keys == baseline_feature_keys
        ), f"feature set differs (got {feature_keys}, expected {baseline_feature_keys})"

        feature_values = np.array([features[k] for k in feature_keys], dtype=np.float64)
        baseline_feature_values = np.array(
            [baseline_features[k] for k in feature_keys], dtype=np.float64
        )
        np.testing.assert_allclose(
            feature_values,
            baseline_feature_values,
            rtol=1e-8,
            atol=1e-10,
            err_msg="feature values differ from baseline",
        )
