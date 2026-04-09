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


@njit(cache=True)
def _is_endpoint(img, r, c):
    """Check if point is an endpoint (exactly 1 foreground neighbor)."""
    count = 0
    for dr, dc in _NEIGHBORS:
        if img[r + dr, c + dc] == 1:
            count += 1
            if count > 1:
                return False
    return True


@njit(cache=True)
def _count_8connected_components(img, r, c):
    """Count 8-connected components among the 8-neighbors of (r, c).

    Computed via flood-fill DFS over the 8-neighbor foreground pixels.
    Uses fixed-size arrays instead of Python sets for numba compatibility.
    """
    nr_arr = np.empty(8, dtype=np.int64)
    nc_arr = np.empty(8, dtype=np.int64)
    is_fg = np.zeros(8, dtype=np.bool_)
    n = 0

    for i in range(8):
        dr, dc = _NEIGHBORS[i]
        if img[r + dr, c + dc] == 1:
            nr_arr[n] = r + dr
            nc_arr[n] = c + dc
            is_fg[n] = True
            n += 1

    if n == 0:
        return 0

    visited = np.zeros(8, dtype=np.bool_)
    stack = np.empty(8, dtype=np.int64)
    components = 0

    for start in range(n):
        if visited[start] or not is_fg[start]:
            continue
        components += 1
        sp = 0
        stack[sp] = start
        sp += 1
        visited[start] = True
        while sp > 0:
            sp -= 1
            ci = stack[sp]
            cr, cc = nr_arr[ci], nc_arr[ci]
            for ni in range(n):
                if not is_fg[ni] or visited[ni]:
                    continue
                if abs(nr_arr[ni] - cr) <= 1 and abs(nc_arr[ni] - cc) <= 1:
                    visited[ni] = True
                    stack[sp] = ni
                    sp += 1

    return components


@njit(cache=True)
def _is_simple_point(img, r, c):
    """Check if removing the point preserves connectivity.

    A point is simple if its 8-neighbors form exactly one 8-connected component.
    """
    return _count_8connected_components(img, r, c) == 1


@njit(parallel=True, cache=True)
def lee94_thin(img):
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

    while True:
        total_removed = 0

        for border_idx in range(4):
            br, bc = _BORDERS[border_idx]

            # Step 1: collect candidates in parallel (per-row)
            row_counts[:] = 0
            for r in prange(1, h + 1):
                for c in range(1, w + 1):
                    if padded[r, c] != 1:
                        continue
                    if padded[r + br, c + bc] != 0:
                        continue
                    if _is_endpoint(padded, r, c):
                        continue
                    if not _is_simple_point(padded, r, c):
                        continue
                    idx = row_counts[r - 1]
                    row_candidates[r - 1, idx, 0] = r
                    row_candidates[r - 1, idx, 1] = c
                    row_counts[r - 1] = idx + 1

            # Merge per-row results into flat candidates array
            count = 0
            for r in range(h):
                for i in range(row_counts[r]):
                    candidates[count, 0] = row_candidates[r, i, 0]
                    candidates[count, 1] = row_candidates[r, i, 1]
                    count += 1

            # Step 2: sequential rechecking (data dependency - cannot parallelize)
            for i in range(count):
                r = candidates[i, 0]
                c = candidates[i, 1]
                if padded[r, c] != 1:
                    continue
                if _is_simple_point(padded, r, c):
                    padded[r, c] = 0
                    total_removed += 1

        if total_removed == 0:
            break

    return padded[1:-1, 1:-1].copy()
