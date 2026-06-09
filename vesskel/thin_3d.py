"""Lee94 thinning algorithm for 3D binary images.

This is a pure Python implementation of the
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

_EULER_ARR = np.array(
    [
        1,
        -1,
        -1,
        1,
        -3,
        -1,
        -1,
        1,
        -1,
        1,
        1,
        -1,
        3,
        1,
        1,
        -1,
        -3,
        -1,
        3,
        1,
        1,
        -1,
        3,
        1,
        -1,
        1,
        1,
        -1,
        3,
        1,
        1,
        -1,
        -3,
        3,
        -1,
        1,
        1,
        3,
        -1,
        1,
        -1,
        1,
        1,
        -1,
        3,
        1,
        1,
        -1,
        1,
        3,
        3,
        1,
        5,
        3,
        3,
        1,
        -1,
        1,
        1,
        -1,
        3,
        1,
        1,
        -1,
        -7,
        -1,
        -1,
        1,
        -3,
        -1,
        -1,
        1,
        -1,
        1,
        1,
        -1,
        3,
        1,
        1,
        -1,
        -3,
        -1,
        3,
        1,
        1,
        -1,
        3,
        1,
        -1,
        1,
        1,
        -1,
        3,
        1,
        1,
        -1,
        -3,
        3,
        -1,
        1,
        1,
        3,
        -1,
        1,
        -1,
        1,
        1,
        -1,
        3,
        1,
        1,
        -1,
        1,
        3,
        3,
        1,
        5,
        3,
        3,
        1,
        -1,
        1,
        1,
        -1,
        3,
        1,
        1,
        -1,
    ],
    dtype=np.int32,
)

_EULER_LUT = np.zeros(256, dtype=np.int32)
_EULER_LUT[1::2] = _EULER_ARR

_OCTANTS = np.array(
    [
        [2, 1, 11, 10, 5, 4, 14],
        [0, 9, 3, 12, 1, 10, 4],
        [8, 7, 17, 16, 5, 4, 14],
        [6, 15, 7, 16, 3, 12, 4],
        [20, 23, 19, 22, 11, 14, 10],
        [18, 21, 9, 12, 19, 22, 10],
        [26, 23, 17, 14, 25, 22, 16],
        [24, 25, 15, 16, 21, 22, 12],
    ],
    dtype=np.int64,
)

_BORDERS = np.array([4, 3, 2, 1, 5, 6], dtype=np.int64)

_OFFSETS_26 = np.array(
    [
        (-1, -1, -1),
        (-1, -1, 0),
        (-1, -1, 1),
        (-1, 0, -1),
        (-1, 0, 0),
        (-1, 0, 1),
        (-1, 1, -1),
        (-1, 1, 0),
        (-1, 1, 1),
        (0, -1, -1),
        (0, -1, 0),
        (0, -1, 1),
        (0, 0, -1),
        (0, 0, 1),
        (0, 1, -1),
        (0, 1, 0),
        (0, 1, 1),
        (1, -1, -1),
        (1, -1, 0),
        (1, -1, 1),
        (1, 0, -1),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, -1),
        (1, 1, 0),
        (1, 1, 1),
    ],
    dtype=np.int8,
)

_ADJ26 = np.zeros((26, 26), dtype=np.uint8)
for _i in range(26):
    for _j in range(26):
        if _i == _j:
            continue
        dp = int(_OFFSETS_26[_i, 0]) - int(_OFFSETS_26[_j, 0])
        dr = int(_OFFSETS_26[_i, 1]) - int(_OFFSETS_26[_j, 1])
        dc = int(_OFFSETS_26[_i, 2]) - int(_OFFSETS_26[_j, 2])
        if abs(dp) <= 1 and abs(dr) <= 1 and abs(dc) <= 1:
            _ADJ26[_i, _j] = 1

_ADJ26_LIST = np.full((26, 26), -1, dtype=np.int8)
_ADJ26_COUNT = np.zeros(26, dtype=np.uint8)
for _i in range(26):
    _count = 0
    for _j in range(26):
        if _ADJ26[_i, _j] == 1:
            _ADJ26_LIST[_i, _count] = _j
            _count += 1
    _ADJ26_COUNT[_i] = _count

_BORDER_OFFSETS = np.array(
    [
        (0, 0, 0),
        (0, 0, -1),
        (0, 0, 1),
        (0, 1, 0),
        (0, -1, 0),
        (1, 0, 0),
        (-1, 0, 0),
    ],
    dtype=np.int8,
)


@njit(cache=True)
def _get_neighborhood(img, p, r, c, neighborhood):
    idx = 0
    for dp in range(-1, 2):
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                neighborhood[idx] = img[p + dp, r + dr, c + dc]
                idx += 1


@njit(cache=True)
def _is_endpoint(neighbors):
    s = 0
    for j in range(27):
        s += neighbors[j]
    return s == 2


@njit(cache=True)
def _is_euler_invariant(neighbors):
    euler_char = 0
    for octant in range(8):
        n = 1
        for j in range(7):
            idx = _OCTANTS[octant, j]
            if neighbors[idx] == 1:
                n |= 1 << (7 - j)
        euler_char += _EULER_LUT[n]
    return euler_char == 0


@njit(cache=True)
def _is_simple_point(neighbors):
    cube = np.empty(26, dtype=np.uint8)
    j = 0
    for i in range(27):
        if i == 13:
            continue
        cube[j] = neighbors[i]
        j += 1

    visited = np.zeros(26, dtype=np.uint8)
    stack = np.empty(26, dtype=np.int64)
    components = 0

    for i in range(26):
        if cube[i] != 1 or visited[i] == 1:
            continue

        components += 1
        if components >= 2:
            return False

        sp = 0
        stack[sp] = i
        sp += 1
        visited[i] = 1

        while sp > 0:
            sp -= 1
            cur = stack[sp]
            for k in range(_ADJ26_COUNT[cur]):
                nxt = _ADJ26_LIST[cur, k]
                if cube[nxt] != 1 or visited[nxt] == 1:
                    continue
                visited[nxt] = 1
                stack[sp] = nxt
                sp += 1

    return True


@njit(cache=True)
def _check_voxel(neighborhood):
    return (
        not _is_endpoint(neighborhood)
        and _is_euler_invariant(neighborhood)
        and _is_simple_point(neighborhood)
    )


@njit(parallel=True, cache=True)
def thin_3d(img):
    """Lee94 thinning algorithm for a 3D binary volume.

    Parameters
    ----------
    img : ndarray
        3D binary volume (0=background, 1=foreground).

    Returns
    -------
    ndarray
        Thinned binary volume with the same shape as img.

    Raises
    ------
    ValueError
        If input is not 3-dimensional.
    """
    img = img.astype(np.uint8).copy()
    d, h, w = img.shape

    padded = np.zeros((d + 2, h + 2, w + 2), dtype=np.uint8)
    padded[1:-1, 1:-1, 1:-1] = img

    max_candidates = d * h * w
    candidates = np.empty((max_candidates, 3), dtype=np.int64)
    can_remove = np.zeros(max_candidates, dtype=np.bool_)
    group_indices = np.empty((8, max_candidates), dtype=np.int64)
    group_counts = np.zeros(8, dtype=np.int64)

    # Pre-load border offsets for combined scan
    b1 = _BORDER_OFFSETS[1]
    b2 = _BORDER_OFFSETS[2]
    b3 = _BORDER_OFFSETS[3]
    b4 = _BORDER_OFFSETS[4]
    b5 = _BORDER_OFFSETS[5]
    b6 = _BORDER_OFFSETS[6]

    while True:
        total_removed = 0

        # ---------------------------------------------------------------
        # Pass 1: count candidates per (p, r) pair
        # ---------------------------------------------------------------
        pr_counts = np.zeros(d * h, dtype=np.int64)
        for p in prange(0, d):
            pp = p + 1
            for r in range(h):
                rr = r + 1
                pr_idx = p * h + r
                cnt = 0
                for c in range(w):
                    cc = c + 1
                    if padded[pp, rr, cc] != 1:
                        continue
                    if (
                        padded[pp + b1[0], rr + b1[1], cc + b1[2]] != 0
                        and padded[pp + b2[0], rr + b2[1], cc + b2[2]] != 0
                        and padded[pp + b3[0], rr + b3[1], cc + b3[2]] != 0
                        and padded[pp + b4[0], rr + b4[1], cc + b4[2]] != 0
                        and padded[pp + b5[0], rr + b5[1], cc + b5[2]] != 0
                        and padded[pp + b6[0], rr + b6[1], cc + b6[2]] != 0
                    ):
                        continue
                    cnt += 1
                pr_counts[pr_idx] = cnt

        # Prefix sum
        offsets = np.empty(d * h + 1, dtype=np.int64)
        offsets[0] = 0
        for i in range(d * h):
            offsets[i + 1] = offsets[i] + pr_counts[i]
        total_count = offsets[d * h]

        if total_count == 0:
            break

        # ---------------------------------------------------------------
        # Pass 2: fill candidates using offset indices
        # ---------------------------------------------------------------
        for p in prange(0, d):
            pp = p + 1
            for r in range(h):
                rr = r + 1
                pr_idx = p * h + r
                base = offsets[pr_idx]
                cnt = 0
                nc = pr_counts[pr_idx]
                for c in range(w):
                    cc = c + 1
                    if padded[pp, rr, cc] != 1:
                        continue
                    if (
                        padded[pp + b1[0], rr + b1[1], cc + b1[2]] != 0
                        and padded[pp + b2[0], rr + b2[1], cc + b2[2]] != 0
                        and padded[pp + b3[0], rr + b3[1], cc + b3[2]] != 0
                        and padded[pp + b4[0], rr + b4[1], cc + b4[2]] != 0
                        and padded[pp + b5[0], rr + b5[1], cc + b5[2]] != 0
                        and padded[pp + b6[0], rr + b6[1], cc + b6[2]] != 0
                    ):
                        continue
                    idx = base + cnt
                    candidates[idx, 0] = pp
                    candidates[idx, 1] = rr
                    candidates[idx, 2] = cc
                    cnt += 1

        # ---------------------------------------------------------------
        # Bucket into 8 parity groups
        # ---------------------------------------------------------------
        group_counts[:] = 0
        for i in range(total_count):
            p = candidates[i, 0]
            r = candidates[i, 1]
            c = candidates[i, 2]
            g = ((p & 1) << 2) | ((r & 1) << 1) | (c & 1)
            gc = group_counts[g]
            group_indices[g, gc] = i
            group_counts[g] = gc + 1

        # ---------------------------------------------------------------
        # Phase 2: Wavefront -8 parity groups
        # ---------------------------------------------------------------
        neighborhood = np.empty(27, dtype=np.uint8)

        for g in range(8):
            gc = group_counts[g]
            if gc == 0:
                continue

            # Parallel check: endpoint + Euler + 26-connectivity
            for j in prange(gc):
                idx = group_indices[g, j]
                p = candidates[idx, 0]
                r = candidates[idx, 1]
                c = candidates[idx, 2]

                if padded[p, r, c] != 1:
                    can_remove[idx] = False
                    continue

                _get_neighborhood(padded, p, r, c, neighborhood)
                can_remove[idx] = _check_voxel(neighborhood)

            # Sequential apply
            for j in range(gc):
                idx = group_indices[g, j]
                if can_remove[idx]:
                    p = candidates[idx, 0]
                    r = candidates[idx, 1]
                    c = candidates[idx, 2]
                    padded[p, r, c] = 0
                    total_removed += 1

        if total_removed == 0:
            break

    return padded[1:-1, 1:-1, 1:-1].copy()
