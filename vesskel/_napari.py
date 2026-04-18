import time

import numpy as np
from napari.layers import Image
from napari.types import LayerDataTuple
from napari.utils.notifications import show_info

from vesskel.thin import lee94_thin


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
