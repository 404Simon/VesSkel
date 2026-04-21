"""Benchmark: vesskel vs skimage.morphology.skeletonize on HRF dataset."""

import time

import numpy as np
from skimage.morphology import skeletonize as skimage_thin

from vesskel.hrf import HRFDataset, preprocess_segmentation
from vesskel.thin import lee94_thin


def warmup():
    small = np.zeros((10, 10), dtype=np.uint8)
    small[2:8, 2:8] = 1
    lee94_thin(small)


def print_row(name, lee_t, sk_zhang_t, sk_lee_t, speedup_zhang=None, speedup_lee=None):
    if speedup_zhang is None and speedup_lee is None:
        print(f"{name:<10} {lee_t:<12} {sk_zhang_t:<12} {sk_lee_t:<12}")
    elif isinstance(speedup_zhang, (int, float)) and isinstance(
        speedup_lee, (int, float)
    ):
        print(
            f"{name:<10} {lee_t:<12.3f} {sk_zhang_t:<12.3f} {sk_lee_t:<12.3f} {speedup_zhang:<12.2f} {speedup_lee:<12.2f}"
        )
    else:
        print(
            f"{name:<10} {lee_t:<12} {sk_zhang_t:<12} {sk_lee_t:<12} {str(speedup_zhang):<12} {str(speedup_lee):<12}"
        )


def main():
    ds = HRFDataset("HRF")
    print("Warming up numba JIT...")
    warmup()
    print("JIT warmup done.\n")

    print_row(
        "Sample",
        "vesskel(s)",
        "sk zhang(s)",
        "sk lee(s)",
        "Spdup zhang",
        "Spdup lee",
    )
    print("-" * 75)

    lee_times, sk_zhang_times, sk_lee_times = [], [], []

    for i in range(len(ds)):
        img, seg, mask, info = ds.load_sample(i)
        cleaned = preprocess_segmentation(seg, mask)

        t0 = time.perf_counter()
        lee94_thin(cleaned)
        lee_t = time.perf_counter() - t0

        t0 = time.perf_counter()
        skimage_thin(cleaned > 0, method="zhang")
        sk_zhang_t = time.perf_counter() - t0

        t0 = time.perf_counter()
        skimage_thin(cleaned > 0, method="lee")
        sk_lee_t = time.perf_counter() - t0

        lee_times.append(lee_t)
        sk_zhang_times.append(sk_zhang_t)
        sk_lee_times.append(sk_lee_t)

        speedup_zhang = sk_zhang_t / lee_t if lee_t > 0 else float("inf")
        speedup_lee = sk_lee_t / lee_t if lee_t > 0 else float("inf")
        print_row(info["name"], lee_t, sk_zhang_t, sk_lee_t, speedup_zhang, speedup_lee)

    print("-" * 75)

    stats = [
        ("TOTAL", sum(lee_times), sum(sk_zhang_times), sum(sk_lee_times)),
        ("MEAN", np.mean(lee_times), np.mean(sk_zhang_times), np.mean(sk_lee_times)),
        (
            "MEDIAN",
            np.median(lee_times),
            np.median(sk_zhang_times),
            np.median(sk_lee_times),
        ),
    ]

    for name, lt, szt, slt in stats:
        speedup_zhang = szt / lt
        speedup_lee = slt / lt
        print_row(name, lt, szt, slt, speedup_zhang, speedup_lee)


if __name__ == "__main__":
    main()
