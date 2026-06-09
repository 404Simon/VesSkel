"""Lee94 thinning algorithm for 2D binary images.

This is a 2D numba-parallelized pure-Python implementation of the
thinning algorithm from [Lee94], based on the scikit-image Cython
implementation in `skimage.morphology._skeletonize_lee_cy` (`_compute_thin_image`)
[SKIMAGE], which itself is a port of the Skeletonize3D ImageJ plugin by
Ignacio Arganda-Carreras [IAC15].

References
----------
- [Lee94] T.-C. Lee, R.L. Kashyap and C.-N. Chu, Building skeleton models
          via 3-D medial surface/axis thinning algorithms.
          Computer Vision, Graphics, and Image Processing, 56(6):462-478, 1994.

- [IAC15] Ignacio Arganda-Carreras, 2015. Skeletonize3D plugin for ImageJ(C).
           https://imagej.net/Skeletonize3D

- [SKIMAGE] scikit-image, `skimage.morphology._skeletonize_lee_cy`.
           https://github.com/scikit-image/scikit-image/blob/main/src/skimage/morphology/_skeletonize_lee_cy.pyx.in
"""

import numpy as np
from numba import njit, prange

# 8-neighbor offsets (row, col)
_NEIGHBORS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

# 4-neighbor offsets for border detection
_BORDERS = ((-1, 0), (1, 0), (0, 1), (0, -1))  # N, S, E, W


def _build_simple_lut():
    lut = np.zeros(256, dtype=np.bool_)
    nr = np.empty(8, dtype=np.int64)
    nc = np.empty(8, dtype=np.int64)
    visited = np.empty(8, dtype=np.bool_)
    stack = np.empty(8, dtype=np.int64)
    for pattern in range(256):
        fg_idx = [i for i in range(8) if (pattern >> i) & 1]
        n = len(fg_idx)
        if n == 0:
            continue
        for j in range(n):
            nr[j] = _NEIGHBORS[fg_idx[j]][0]
            nc[j] = _NEIGHBORS[fg_idx[j]][1]
        visited[:n] = False
        components = 0
        for start in range(n):
            if visited[start]:
                continue
            components += 1
            sp = 0
            stack[sp] = start
            sp += 1
            visited[start] = True
            while sp > 0:
                sp -= 1
                ci = stack[sp]
                cr, cc = nr[ci], nc[ci]
                for ni in range(n):
                    if visited[ni]:
                        continue
                    if abs(nr[ni] - cr) <= 1 and abs(nc[ni] - cc) <= 1:
                        visited[ni] = True
                        stack[sp] = ni
                        sp += 1
        lut[pattern] = components == 1
    return lut


_SIMPLE_LUT = _build_simple_lut()


@njit(parallel=True, cache=True)
def thin_2d(img):
    """Lee94 thinning algorithm for a 2D binary image.

    Parameters
    ----------
    img : ndarray
        2D binary image (0=background, 1=foreground).

    Returns
    -------
    ndarray
        Thinned binary image with the same shape as img.
    """
    img = img.astype(np.uint8).copy()
    h, w = img.shape

    padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = img

    # Per-row candidate storage for parallel collection
    row_candidates = np.empty((h, w, 2), dtype=np.int64)
    row_counts = np.zeros(h, dtype=np.int64)
    # Flattened candidates for sequential recheck
    candidates = np.empty((h * w, 2), dtype=np.int64)
    group_candidates = np.empty((4, h * w, 2), dtype=np.int64)
    group_counts = np.zeros(4, dtype=np.int64)
    can_remove = np.zeros(h * w, dtype=np.bool_)

    b0r, b0c = _BORDERS[0]
    b1r, b1c = _BORDERS[1]
    b2r, b2c = _BORDERS[2]
    b3r, b3c = _BORDERS[3]

    while True:
        total_removed = 0

        # Phase 1: single combined scan -all 4 borders
        row_counts[:] = 0
        for r in prange(1, h + 1):
            ri = r - 1
            rc = row_candidates[ri]
            cnt = 0
            for c in range(1, w + 1):
                if padded[r, c] != 1:
                    continue
                if (
                    padded[r + b0r, c + b0c] != 0
                    and padded[r + b1r, c + b1c] != 0
                    and padded[r + b2r, c + b2c] != 0
                    and padded[r + b3r, c + b3c] != 0
                ):
                    continue
                rc[cnt, 0] = r
                rc[cnt, 1] = c
                cnt += 1
            row_counts[ri] = cnt

        count = 0
        for ri in range(h):
            n = row_counts[ri]
            rc = row_candidates[ri]
            for j in range(n):
                candidates[count, 0] = rc[j, 0]
                candidates[count, 1] = rc[j, 1]
                count += 1

        if count == 0:
            break

        # Bucket into 4 parity groups
        group_counts[:] = 0
        for i in range(count):
            r = candidates[i, 0]
            c = candidates[i, 1]
            g = ((r & 1) << 1) | (c & 1)
            gc = group_counts[g]
            group_candidates[g, gc, 0] = r
            group_candidates[g, gc, 1] = c
            group_counts[g] = gc + 1

        # Phase 2: Wavefront -4 groups
        for g in range(4):
            gc = group_counts[g]
            if gc == 0:
                continue
            gca = group_candidates[g]

            for i in prange(gc):
                r = gca[i, 0]
                c = gca[i, 1]
                if padded[r, c] != 1:
                    can_remove[i] = False
                    continue

                pat = np.uint8(0)
                if padded[r - 1, c - 1]:
                    pat |= np.uint8(1)
                if padded[r - 1, c]:
                    pat |= np.uint8(2)
                if padded[r - 1, c + 1]:
                    pat |= np.uint8(4)
                if padded[r, c - 1]:
                    pat |= np.uint8(8)
                if padded[r, c + 1]:
                    pat |= np.uint8(16)
                if padded[r + 1, c - 1]:
                    pat |= np.uint8(32)
                if padded[r + 1, c]:
                    pat |= np.uint8(64)
                if padded[r + 1, c + 1]:
                    pat |= np.uint8(128)

                can_remove[i] = _SIMPLE_LUT[pat]

            for i in range(gc):
                if can_remove[i]:
                    padded[gca[i, 0], gca[i, 1]] = 0
                    total_removed += 1

        if total_removed == 0:
            break

    return padded[1:-1, 1:-1].copy()
