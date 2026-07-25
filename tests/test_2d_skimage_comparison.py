"""Test comparing vesskel.thin with skimage skeletonize(method='lee') on an HRF image."""

import numpy as np
import pytest
from skimage.morphology import skeletonize

from vesskel.hrf import HRFDataset, preprocess_segmentation
from vesskel.thin import lee94_thin

HRF_PATH = "HRF"


@pytest.mark.slow
class TestSkimageComparison2D:
    """Compare vesskel.thin with scikit-image skeletonize (Lee) on a single HRF image."""

    @pytest.fixture(scope="class")
    def sample_data(self):
        ds = HRFDataset(HRF_PATH)
        _, seg, mask, _ = ds.load_sample(0)
        cleaned = preprocess_segmentation(seg, mask)
        return cleaned

    def test_vesskel_vs_skimage_lee(self, sample_data):
        binary = sample_data > 0
        vesskel_skel = lee94_thin(sample_data)
        skimage_skel = skeletonize(binary, method="lee")

        assert vesskel_skel.shape == skimage_skel.shape, (
            f"shape mismatch: vesskel {vesskel_skel.shape} vs skimage {skimage_skel.shape}"
        )
        assert np.array_equal(vesskel_skel, skimage_skel), (
            "skeleton mismatch: algorithms produce different results"
        )
