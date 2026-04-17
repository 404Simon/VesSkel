"""Regression test comparing vesskel.thin with scikit-image skeletonize on brain image."""

import numpy as np
import pytest
from skimage import data
from skimage.morphology import skeletonize

from vesskel.thin import lee94_thin


class TestSkeletonizeComparison:
    """Compare vesskel.thin with scikit-image skeletonize on brain image."""

    @pytest.fixture(scope="class")
    def image(self):
        return data.brain()

    def test_vesskel_vs_scikit_skeletonize(self, image):
        vesskel_skel = lee94_thin(image)
        scikit_skel = skeletonize(image)

        assert (
            vesskel_skel.shape == scikit_skel.shape
        ), f"shape mismatch: vesskel {vesskel_skel.shape} vs scikit {scikit_skel.shape}"
        assert np.array_equal(
            vesskel_skel, scikit_skel
        ), "skeleton mismatch: algorithms produce different results"
