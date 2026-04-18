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
def _find_simple_point_candidates(img, curr_border, candidates):
    count = 0
    dp = int(_BORDER_OFFSETS[curr_border, 0])
    dr = int(_BORDER_OFFSETS[curr_border, 1])
    dc = int(_BORDER_OFFSETS[curr_border, 2])

    for p in range(1, img.shape[0] - 1):
        for r in range(1, img.shape[1] - 1):
            for c in range(1, img.shape[2] - 1):
                if img[p, r, c] != 1:
                    continue
                if img[p + dp, r + dr, c + dc] != 0:
                    continue

                candidates[count, 0] = p
                candidates[count, 1] = r
                candidates[count, 2] = c
                count += 1

    return count


@njit(cache=True, parallel=True)
def _mark_removable_candidates(img, candidates, num_candidates, removable):
    for i in prange(num_candidates):
        p = candidates[i, 0]
        r = candidates[i, 1]
        c = candidates[i, 2]

        neighborhood = np.empty(27, dtype=np.uint8)
        _get_neighborhood(img, p, r, c, neighborhood)

        can_remove = (
            (not _is_endpoint(neighborhood))
            and _is_euler_invariant(neighborhood)
            and _is_simple_point(neighborhood)
        )
        removable[i] = 1 if can_remove else 0


@njit(cache=True)
def _apply_removals(img, candidates, num_candidates, removable):
    removed = 0
    neighborhood = np.empty(27, dtype=np.uint8)
    for i in range(num_candidates):
        if removable[i] == 0:
            continue
        p = candidates[i, 0]
        r = candidates[i, 1]
        c = candidates[i, 2]
        if img[p, r, c] != 1:
            continue
        _get_neighborhood(img, p, r, c, neighborhood)
        if _is_simple_point(neighborhood):
            img[p, r, c] = 0
            removed += 1
    return removed


@njit(cache=True)
def _compute_thin_image(img):
    num_borders = 6
    unchanged_borders = 0

    candidates = np.empty((img.size, 3), dtype=np.int32)
    removable = np.empty(img.size, dtype=np.uint8)

    while unchanged_borders < num_borders:
        unchanged_borders = 0
        for j in range(num_borders):
            curr_border = _BORDERS[j]
            num_candidates = _find_simple_point_candidates(img, curr_border, candidates)

            if num_candidates == 0:
                unchanged_borders += 1
                continue

            _mark_removable_candidates(img, candidates, num_candidates, removable)
            removed = _apply_removals(img, candidates, num_candidates, removable)

            if removed == 0:
                unchanged_borders += 1

    return img


def thin_3d(img):
    """Lee94 thinning algorithm for a 3D binary volume."""
    if img.ndim != 3:
        raise ValueError(f"Expected 3D input, got {img.ndim}D")

    work = (img > 0).astype(np.uint8, copy=False)
    padded = np.zeros(
        (work.shape[0] + 2, work.shape[1] + 2, work.shape[2] + 2), dtype=np.uint8
    )
    padded[1:-1, 1:-1, 1:-1] = work

    out = _compute_thin_image(padded)
    return out[1:-1, 1:-1, 1:-1].copy()
