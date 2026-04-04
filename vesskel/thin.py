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

# 8-neighbor offsets (row, col)
_NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

# 4-neighbor offsets for border detection
_BORDERS = [(-1, 0), (1, 0), (0, 1), (0, -1)]  # N, S, E, W


def _is_endpoint(img, r, c):
    """Check if point is an endpoint (exactly 1 foreground neighbor)."""
    count = 0
    for dr, dc in _NEIGHBORS:
        if img[r + dr, c + dc] == 1:
            count += 1
            if count > 1:
                return False
    return True


def _count_8connected_components(img, r, c):
    """Count 8-connected components among the 8-neighbors of (r, c).

    Computed via flood-fill DFS over the 8-neighbor foreground pixels.
    """
    neighbors = []
    for dr, dc in _NEIGHBORS:
        if img[r + dr, c + dc] == 1:
            neighbors.append((r + dr, c + dc))

    if not neighbors:
        return 0

    neighbor_set = set(neighbors)
    visited = set()
    components = 0

    for start in neighbors:
        if start in visited:
            continue
        components += 1
        stack = [start]
        visited.add(start)
        while stack:
            cr, cc = stack.pop()
            for dr, dc in _NEIGHBORS:
                nr, nc = cr + dr, cc + dc
                if (nr, nc) in neighbor_set and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    stack.append((nr, nc))

    return components


def _is_simple_point(img, r, c):
    """Check if removing the point preserves connectivity.

    A point is simple if its 8-neighbors form exactly one 8-connected component.
    """
    return _count_8connected_components(img, r, c) == 1


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

    while True:
        total_removed = 0

        for border_idx in range(4):
            br, bc = _BORDERS[border_idx]

            # Step 1: collect candidates
            candidates = []
            for r in range(1, h + 1):
                for c in range(1, w + 1):
                    if padded[r, c] != 1:
                        continue
                    # Check if point is on the current border direction
                    if padded[r + br, c + bc] != 0:
                        continue
                    if _is_endpoint(padded, r, c):
                        continue
                    if not _is_simple_point(padded, r, c):
                        continue
                    candidates.append((r, c))

            # Step 2: sequential rechecking
            for r, c in candidates:
                if padded[r, c] != 1:
                    continue
                if _is_simple_point(padded, r, c):
                    padded[r, c] = 0
                    total_removed += 1

        if total_removed == 0:
            break

    return padded[1:-1, 1:-1].copy()
