"""Regression test suite.

On first run (or when --update-baseline is passed), skeletons are generated
and saved as baseline reference files.

On subsequent runs, skeletons are regenerated and compared against the
saved baselines. Any difference is reported as a test failure, indicating
a regression (or an intentional algorithm change).
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from vesskel.hrf import HRFDataset, preprocess_segmentation
from vesskel.thin import lee94_thin

BASELINE_DIR = Path(__file__).parent / "skeletons"
HRF_PATH = "HRF"


def _skeleton_path(name: str) -> Path:
    return BASELINE_DIR / f"skeleton_{name}.npz"


def _compute_skeleton(dataset: HRFDataset, index: int) -> np.ndarray:
    _, seg, mask, _ = dataset.load_sample(index)
    cleaned = preprocess_segmentation(seg, mask)
    return lee94_thin(cleaned)


def _hash_array(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()


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
        baseline_file = _skeleton_path(name)

        if request.config.getoption("--update-baseline") or not baseline_file.exists():
            BASELINE_DIR.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(baseline_file, skeleton=skeleton)
            action = (
                "updated"
                if request.config.getoption("--update-baseline")
                else "created"
            )
            pytest.skip(f"Baseline {action} for sample {name}")

        baseline = np.load(baseline_file)["skeleton"]

        assert skeleton.shape == baseline.shape, (
            f"Sample {name}: shape mismatch "
            f"got {skeleton.shape}, expected {baseline.shape}"
        )
        assert np.array_equal(skeleton, baseline), (
            f"Sample {name}: skeleton differs from baseline "
            f"(hash {_hash_array(skeleton)} vs {_hash_array(baseline)})"
        )
