"""Lee94 thinning algorithm for 2D binary images.

This is a pure-Python (unoptimized) implementation of the thinning
algorithm from [Lee94], closely based on the scikit-image Cython
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


def _pad_2d_to_3d(img_2d):
    """Embed a 2D binary image into a padded 3D array (3, H+2, W+2)."""
    h, w = img_2d.shape
    vol = np.zeros((3, h + 2, w + 2), dtype=np.uint8)
    vol[1, 1:-1, 1:-1] = img_2d
    return vol


def _get_neighborhood(vol, p, r, c):
    """Return the 27-neighborhood as a flat array."""
    n = np.empty(27, dtype=np.uint8)
    idx = 0
    for dp in (-1, 0, 1):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                n[idx] = vol[p + dp, r + dr, c + dc]
                idx += 1
    return n


def _is_endpoint(neighbors):
    """Endpoint has exactly 1 foreground neighbor (+ itself = 2)."""
    return np.sum(neighbors) == 2


# Euler LUT from Lee94 Table 2, column δG_26
_EULER_LUT = np.zeros(256, dtype=np.int32)
_arr = [
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
]
_EULER_LUT[1::2] = _arr

# Octant index tables for euler characteristic
_OCTANT_INDICES = [
    [2, 1, 11, 10, 5, 4, 14],  # NEB
    [0, 9, 3, 12, 1, 10, 4],  # NWB
    [8, 7, 17, 16, 5, 4, 14],  # SEB
    [6, 15, 7, 16, 3, 12, 4],  # SWB
    [20, 23, 19, 22, 11, 14, 10],  # NEU
    [18, 21, 9, 12, 19, 22, 10],  # NWU
    [26, 23, 17, 14, 25, 22, 16],  # SEU
    [24, 25, 15, 16, 21, 22, 12],  # SWU
]


def _is_euler_invariant(neighbors):
    """Check Euler characteristic preservation across all 8 octants."""
    euler_sum = 0
    for octant in range(8):
        n = 1
        for j, idx in enumerate(_OCTANT_INDICES[octant]):
            if neighbors[idx] == 1:
                n |= 1 << (7 - j)
        euler_sum += _EULER_LUT[n]
    return euler_sum == 0


# Octree structure for simple point labeling
#   Each entry corresponds to one of the 8 octants and contains:
#     - list of 7 neighbor indices within that octant
#     - list of 7 sub-octant indices to recurse into for connectivity
_OCTREE = [
    ([0, 1, 3, 4, 9, 10, 12], [[], [2], [3], [2, 3, 4], [5], [2, 5, 6], [3, 5, 7]]),
    ([1, 4, 10, 2, 5, 11, 13], [[1], [1, 3, 4], [1, 5, 6], [], [4], [6], [4, 6, 8]]),
    ([3, 4, 12, 6, 7, 14, 15], [[1], [1, 2, 4], [1, 5, 7], [], [4], [7], [4, 7, 8]]),
    ([4, 5, 13, 7, 15, 8, 16], [[1, 2, 3], [2], [2, 6, 8], [3], [3, 7, 8], [], [8]]),
    ([9, 10, 12, 17, 18, 20, 21], [[1], [1, 2, 6], [1, 3, 7], [], [6], [7], [6, 7, 8]]),
    (
        [10, 11, 13, 18, 21, 19, 22],
        [[1, 2, 5], [2], [2, 4, 8], [5], [5, 7, 8], [], [8]],
    ),
    (
        [12, 14, 15, 20, 21, 23, 24],
        [[1, 3, 5], [3], [3, 4, 8], [5], [5, 6, 8], [], [8]],
    ),
    (
        [13, 15, 16, 21, 22, 24, 25],
        [[2, 4, 6], [3, 4, 7], [4], [5, 6, 7], [6], [7], []],
    ),
]

# Maps each neighbor to the octant where it serves as the starting point for connectivity labeling.
_OCTANT_START = {
    0: 1,
    1: 1,
    2: 2,
    3: 1,
    4: 1,
    5: 2,
    6: 3,
    7: 3,
    8: 4,
    9: 1,
    10: 1,
    11: 2,
    12: 1,
    13: 2,
    14: 3,
    15: 3,
    16: 4,
    17: 5,
    18: 5,
    19: 6,
    20: 5,
    21: 5,
    22: 6,
    23: 7,
    24: 7,
    25: 8,
}


def _octree_labeling(octant, label, cube):
    """Recursively label connected components in the 26-neighborhood."""
    indices, adj_octants = _OCTREE[octant - 1]
    for i, idx in enumerate(indices):
        if cube[idx] == 1:
            cube[idx] = label
            for new_oct in adj_octants[i]:
                _octree_labeling(new_oct, label, cube)


def _is_simple_point(neighbors):
    """Check if removing the center point preserves connectivity."""
    cube = np.concatenate([neighbors[:13], neighbors[14:]])  # skip center
    label = 2
    for i in range(26):
        if cube[i] == 1:
            _octree_labeling(_OCTANT_START[i], label, cube)
            label += 1
            if label - 2 >= 2:
                return False
    return True


def _is_border_point(vol, p, r, c, border):
    """Check if point is on the specified border."""
    if border == 1:
        return vol[p, r, c - 1] == 0  # N
    if border == 2:
        return vol[p, r, c + 1] == 0  # S
    if border == 3:
        return vol[p, r + 1, c] == 0  # E
    if border == 4:
        return vol[p, r - 1, c] == 0  # W
    if border == 5:
        return vol[p + 1, r, c] == 0  # U
    if border == 6:
        return vol[p - 1, r, c] == 0  # B
    return False


def lee94_thin(img_2d):
    """Lee94 thinning algorithm for a 2D binary image.

    Parameters
    ----------
    img_2d : ndarray
        2D binary image (0=background, 1=foreground).

    Returns
    -------
    ndarray
        Thinned binary image with the same shape as img_2d.
    """
    vol = _pad_2d_to_3d(img_2d)
    borders = [4, 3, 2, 1]  # 2D: only W, E, S, N
    num_borders = 4

    unchanged = 0
    while unchanged < num_borders:
        unchanged = 0
        for border in borders:
            # Step 1: collect candidates
            candidates = []
            for r in range(1, vol.shape[1] - 1):
                for c in range(1, vol.shape[2] - 1):
                    if vol[1, r, c] != 1:
                        continue
                    if not _is_border_point(vol, 1, r, c, border):
                        continue
                    nb = _get_neighborhood(vol, 1, r, c)
                    if _is_endpoint(nb):
                        continue
                    if not _is_euler_invariant(nb):
                        continue
                    if not _is_simple_point(nb):
                        continue
                    candidates.append((1, r, c))

            # Step 2: sequential rechecking
            changed = False
            for p, r, c in candidates:
                nb = _get_neighborhood(vol, p, r, c)
                if _is_simple_point(nb):
                    vol[p, r, c] = 0
                    changed = True

            if not changed:
                unchanged += 1

    return vol[1, 1:-1, 1:-1].copy()
