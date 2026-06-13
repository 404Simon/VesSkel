"""3D Benchmark: vesskel vs skimage vs VesselVio"""

import sys
import time

import numpy as np
import tifffile
from skimage.morphology import skeletonize as skimage_thin

sys.path.append("/home/simon/dev/VesselVio")
from library.lee94 import skeletonize as vesselvio_lee94_thin

from vesskel.thin import lee94_thin

VOLUME_PATH = "/home/simon/dev/VesSkel/tests/vessap_preprocessed.tif"


def warmup():
    small = np.zeros((10, 10, 10), dtype=np.uint8)
    small[2:8, 2:8, 2:8] = 1
    lee94_thin(small)


def print_row(name, lee_t, sk_lee_t, vv_t, speedup_sk=None, speedup_vv=None):
    if speedup_sk is None and speedup_vv is None:
        print(f"{name:<10} {lee_t:<16} {sk_lee_t:<16} {vv_t:<16}")
    elif all(isinstance(x, (int, float)) for x in (speedup_sk, speedup_vv)):
        print(
            f"{name:<10} {lee_t:<16.3f} {sk_lee_t:<16.3f} {vv_t:<16.3f} {speedup_sk:<12.2f} {speedup_vv:<12.2f}"
        )
    else:
        print(
            f"{name:<10} {lee_t:<16} {sk_lee_t:<16} {vv_t:<16} {str(speedup_sk):<12} {str(speedup_vv):<12}"
        )


def main():
    print("Loading volume...")
    vol = tifffile.imread(VOLUME_PATH).astype(np.uint8)
    print(f"Volume shape: {vol.shape}, dtype: {vol.dtype}\n")

    print("Warming up numba JIT...")
    warmup()
    print("JIT warmup done.\n")

    print_row(
        "Run",
        "vesskel(s)",
        "sk lee(s)",
        "vv lee(s)",
        "Spdup sk",
        "Spdup vv",
    )
    print("-" * 85)

    lee_times, sk_lee_times, vv_times = [], [], []

    for i in range(10):
        # vesskel lee94
        t0 = time.perf_counter()
        lee94_thin(vol)
        lee_t = time.perf_counter() - t0

        # skimage lee
        t0 = time.perf_counter()
        skimage_thin(vol, method="lee")
        sk_lee_t = time.perf_counter() - t0

        # VesselVio lee94 - needs uint8, padded by 1, C-contiguous
        vv_input = np.ascontiguousarray(np.pad(vol, 1).copy())
        t0 = time.perf_counter()
        vesselvio_lee94_thin(vv_input)
        vv_t = time.perf_counter() - t0

        lee_times.append(lee_t)
        sk_lee_times.append(sk_lee_t)
        vv_times.append(vv_t)

        speedup_sk = sk_lee_t / lee_t if lee_t > 0 else float("inf")
        speedup_vv = vv_t / lee_t if lee_t > 0 else float("inf")
        print_row(i, lee_t, sk_lee_t, vv_t, speedup_sk, speedup_vv)

    print("-" * 85)

    stats = [
        ("TOTAL", sum(lee_times), sum(sk_lee_times), sum(vv_times)),
        ("MEAN", np.mean(lee_times), np.mean(sk_lee_times), np.mean(vv_times)),
        ("MEDIAN", np.median(lee_times), np.median(sk_lee_times), np.median(vv_times)),
    ]

    for name, lt, sklt, vvt in stats:
        speedup_sk = sklt / lt
        speedup_vv = vvt / lt
        print_row(name, lt, sklt, vvt, speedup_sk, speedup_vv)


if __name__ == "__main__":
    main()
