"""Regression test suite.

On first run (or when --update-baseline is passed), skeletons and features are
generated and saved as baseline reference files.

On subsequent runs, skeletons and features are regenerated and compared against
saved baselines. Any difference is reported as a test failure, indicating a
regression (or an intentional algorithm change).
"""

import numpy as np
import pytest

from vesskel.features import extract_vessel_features
from vesskel.hrf import HRFDataset, preprocess_segmentation
from vesskel.thin import lee94_thin

from ._helpers import (
    BASELINE_DIR,
    feature_path,
    hash_array,
    read_feature_csv,
    skeleton_path,
    write_feature_csv,
)

HRF_PATH = "HRF"


def _compute_skeleton(dataset: HRFDataset, index: int) -> np.ndarray:
    _, seg, mask, _ = dataset.load_sample(index)
    cleaned = preprocess_segmentation(seg, mask)
    return lee94_thin(cleaned)


@pytest.fixture(scope="module")
def dataset():
    return HRFDataset(HRF_PATH)


class TestThinningRegression:
    """Run thinning on every HRF sample and compare against saved baselines."""

    @pytest.mark.parametrize(
        "index",
        range(len(HRFDataset(HRF_PATH))),
        ids=[info["name"] for info in HRFDataset(HRF_PATH).image_list],
    )
    def test_skeleton_matches_baseline(self, dataset, index, request):
        info = dataset.image_list[index]
        name = info["name"]
        skeleton = _compute_skeleton(dataset, index)
        features = extract_vessel_features(skeleton)
        baseline_file = skeleton_path(name)
        feature_file = feature_path(name)

        baseline_changed = False

        if request.config.getoption("--update-baseline") or not baseline_file.exists():
            BASELINE_DIR.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(baseline_file, skeleton=skeleton)
            baseline_changed = True

        features_changed = False
        if request.config.getoption("--update-baseline") or not feature_file.exists():
            write_feature_csv(feature_file, features)
            features_changed = True

        if baseline_changed or features_changed:
            baseline_action = (
                "updated"
                if request.config.getoption("--update-baseline")
                else "created"
            )
            artifacts = []
            if baseline_changed:
                artifacts.append("skeleton baseline")
            if features_changed:
                artifacts.append("feature baseline")
            pytest.skip(f"{', '.join(artifacts)} {baseline_action} for sample {name}")

        with np.load(baseline_file) as data:
            baseline = data["skeleton"]

        try:
            baseline_features = read_feature_csv(feature_file)
        except (OSError, ValueError) as exc:
            pytest.fail(f"Sample {name}: invalid feature baseline CSV: {exc}")

        assert skeleton.shape == baseline.shape, (
            f"Sample {name}: shape mismatch "
            f"got {skeleton.shape}, expected {baseline.shape}"
        )
        assert np.array_equal(skeleton, baseline), (
            f"Sample {name}: skeleton differs from baseline "
            f"(hash {hash_array(skeleton)} vs {hash_array(baseline)})"
        )

        feature_keys = sorted(features)
        baseline_feature_keys = sorted(baseline_features)
        assert feature_keys == baseline_feature_keys, (
            f"Sample {name}: feature set differs from baseline "
            f"(got {feature_keys}, expected {baseline_feature_keys})"
        )

        feature_values = np.array(
            [features[key] for key in feature_keys], dtype=np.float64
        )
        baseline_feature_values = np.array(
            [baseline_features[key] for key in feature_keys], dtype=np.float64
        )
        np.testing.assert_allclose(
            feature_values,
            baseline_feature_values,
            rtol=1e-8,
            atol=1e-10,
            err_msg=f"Sample {name}: feature values differ from baseline",
        )
