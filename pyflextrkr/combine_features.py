"""
Combine features that share a large part of their boundary.

The detect-and-spread step (label_and_grow_features) gives every core its own piece of the contiguous echo
around it. When several intense cores sit inside one large storm, the storm is cut into several features by
straight or diagonal lines, and only a part of it can become an MCS. The cuts are long compared with the size of
the pieces. Two features that only touch over a short stretch are separate storms.

combine_features_with_share_boundary merges the features whose shared boundary is a large fraction of the
perimeter of the smaller one. The boundary is counted in unit pixel edges (4-neighbor pixel pairs with different
labels), so a diagonal interface counts (about 2 edges per diagonal step) and two features that meet at a single
corner point do not. Pairs are found for the whole label array in one pass, chains and clusters of flagged
pairs are merged at once (scipy connected components), and the test is repeated on the merged features until
nothing is flagged.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

# A Euclidean length measured in unit pixel edges (staircase length) is 1.0 x (axis-aligned line) to 1.41 x (45
# degree line) longer, 4/pi = 1.27 x on average over all orientations. Used to put the perimeter of the circle or
# ellipse that is equivalent to a feature in the same units as the edge counts.
EDGE_PER_EUCLID = 4.0 / np.pi


def boundary_lengths(labels):
    """
    Perimeter of every label and shared boundary length of every pair of touching labels, in unit pixel edges.

    Every pixel is compared with its right and its lower neighbor. A pair with different labels is one unit
    edge of the interface between those two labels. The outside of the array counts as background, so the edge
    of the domain is an exposed boundary and belongs to the perimeter.

    Args:
        labels: np.ndarray, 2D, non-negative integers
            Label array, 0 = background.

    Returns:
        perimeter: np.ndarray, 1D, int
            Index = label (length max(labels)+1). Number of unit edges of the label to anything else: other
            labels, background and the domain edge. perimeter[0] = 0.
        pairs: np.ndarray, (npair, 2), int
            Labels (a, b), a < b, that share at least one unit edge. Neither is background.
        shared: np.ndarray, (npair,), int
            Number of unit edges shared by each pair.
    """
    labels = np.asarray(labels)
    nlabel = int(labels.max()) if labels.size > 0 else 0

    # Neighbor pairs in both directions, with a zero border so that the domain edge is an exposed boundary
    padded = np.pad(labels, 1)
    left, right = padded[:, :-1], padded[:, 1:]
    top, bottom = padded[:-1, :], padded[1:, :]
    diff_h = left != right
    diff_v = top != bottom
    edge_a = np.concatenate((left[diff_h], top[diff_v]))
    edge_b = np.concatenate((right[diff_h], bottom[diff_v]))

    # Perimeter: every edge counts for both of its labels (the background, label 0, is dropped)
    perimeter = np.bincount(edge_a, minlength=nlabel + 1) + np.bincount(edge_b, minlength=nlabel + 1)
    perimeter[0] = 0

    # Shared boundary: edges between two labels, counted per pair through one integer key per pair
    both = (edge_a > 0) & (edge_b > 0)
    low = np.minimum(edge_a[both], edge_b[both]).astype(np.int64)
    high = np.maximum(edge_a[both], edge_b[both]).astype(np.int64)
    key, shared = np.unique(low * (nlabel + 1) + high, return_counts=True)
    pairs = np.stack((key // (nlabel + 1), key % (nlabel + 1)), axis=1)
    return perimeter, pairs, shared


def equivalent_perimeter(labels, shape="ellipse"):
    """
    Perimeter of the smooth shape that has the same area as each label, in unit pixel edges.

    The raw perimeter of a feature is inflated by ragged edges (1.7-1.9 x the smooth one for the 10 dBZ outlines of
    radar echo). The equivalent shape ignores the raggedness.

    Args:
        labels: np.ndarray, 2D, non-negative integers
            Label array, 0 = background. Pixels are assumed square.
        shape: string (optional, default="ellipse")
            "ellipse": ellipse with the area of the feature and the axis ratio of its second moments (keeps the
                       elongation, so lines are not treated as circles). Ramanujan's approximation of the perimeter.
            "circle": circle with the area of the feature (the shortest outline possible for that area).

    Returns:
        perimeter: np.ndarray, 1D, float
            Index = label (length max(labels)+1), 0 for a label without pixels.
    """
    if shape not in ("ellipse", "circle"):
        raise ValueError(f"shape must be 'ellipse' or 'circle', got '{shape}'")
    labels = np.asarray(labels)
    nlabel = int(labels.max()) if labels.size > 0 else 0

    # Only the labeled pixels are needed
    iy, ix = np.nonzero(labels)
    lab = labels[iy, ix]
    area = np.bincount(lab, minlength=nlabel + 1).astype(float)

    if shape == "circle":
        euclid = 2.0 * np.sqrt(np.pi * area)
    else:
        # Second moments of the pixel centers per label (+1/12: the variance of a pixel itself)
        safe_area = np.maximum(area, 1.0)
        y = iy.astype(float)
        x = ix.astype(float)
        dx = x - (np.bincount(lab, weights=x, minlength=nlabel + 1) / safe_area)[lab]
        dy = y - (np.bincount(lab, weights=y, minlength=nlabel + 1) / safe_area)[lab]
        var_xx = np.bincount(lab, weights=dx * dx, minlength=nlabel + 1) / safe_area + 1.0 / 12.0
        var_yy = np.bincount(lab, weights=dy * dy, minlength=nlabel + 1) / safe_area + 1.0 / 12.0
        var_xy = np.bincount(lab, weights=dx * dy, minlength=nlabel + 1) / safe_area
        # Eigenvalues of the covariance matrix -> axis ratio (minor / major)
        half_trace = 0.5 * (var_xx + var_yy)
        radius = np.sqrt(0.25 * (var_xx - var_yy) ** 2 + var_xy ** 2)
        ratio = np.sqrt(np.maximum(half_trace - radius, 1e-12) / (half_trace + radius))
        # Semi-axes of the ellipse with this ratio and the area of the feature
        semi_major = np.sqrt(area / (np.pi * ratio))
        semi_minor = semi_major * ratio
        # Ramanujan's second approximation of the perimeter of an ellipse
        h = ((semi_major - semi_minor) / np.maximum(semi_major + semi_minor, 1e-12)) ** 2
        euclid = np.pi * (semi_major + semi_minor) * (1.0 + 3.0 * h / (10.0 + np.sqrt(4.0 - 3.0 * h)))

    perimeter = euclid * EDGE_PER_EUCLID
    perimeter[area == 0] = 0.0
    return perimeter


def touching_pairs(labels):
    """
    Table of the touching pairs of a label array: sizes, shared boundary and the fractions that combine_features_with_share_boundary
    tests (the shared boundary as a fraction of the perimeter of the smaller feature of the pair).

    Args:
        labels: np.ndarray, 2D, non-negative integers
            Label array, 0 = background.

    Returns:
        pairs: dictionary of 1D np.ndarray, one entry per touching pair (sorted by label)
            small, large: the labels of the smaller and the larger feature (equal sizes: the lower label is the smaller)
            area_small, area_large: pixels
            shared: unit pixel edges shared by the two features
            perim_small: raw perimeter of the smaller feature [unit pixel edges]
            equiv_ellipse, equiv_circle: perimeter of the equal-area ellipse / circle of the smaller feature [unit pixel edges]
            frac_raw, frac_ellipse, frac_circle: shared / perim_small, / equiv_ellipse, / equiv_circle
    """
    labels = np.asarray(labels)
    perimeter, pairs, shared = boundary_lengths(labels)
    area = np.bincount(labels.ravel(), minlength=perimeter.size)
    # The smaller feature of each pair (equal sizes: the lower label)
    a_is_small = area[pairs[:, 0]] <= area[pairs[:, 1]]
    small = np.where(a_is_small, pairs[:, 0], pairs[:, 1])
    large = np.where(a_is_small, pairs[:, 1], pairs[:, 0])
    equiv_ellipse = equivalent_perimeter(labels, "ellipse")[small]
    equiv_circle = equivalent_perimeter(labels, "circle")[small]
    perim_small = perimeter[small]
    return {
        "small": small, "large": large, "area_small": area[small], "area_large": area[large], "shared": shared,
        "perim_small": perim_small, "equiv_ellipse": equiv_ellipse, "equiv_circle": equiv_circle,
        "frac_raw": shared / perim_small, "frac_ellipse": shared / equiv_ellipse, "frac_circle": shared / equiv_circle,
    }


def renumber_by_size(labels):
    """
    Renumber the labels 1..N from the largest to the smallest feature (the convention of label_and_grow_features).

    Args:
        labels: np.ndarray, 2D, non-negative integers
            Label array, 0 = background. The numbers may have gaps.

    Returns:
        renumbered: np.ndarray, 2D, same dtype
            Label array with the numbers 1..N; equal sizes keep the order of their old numbers.
        old_to_new: np.ndarray, 1D, int
            Index = old label (length max(labels)+1), value = new label; 0 for a number that is not in labels.
    """
    labels = np.asarray(labels)
    area = np.bincount(labels.ravel(), minlength=int(labels.max()) + 1 if labels.size > 0 else 1)
    present = np.nonzero(area[1:])[0] + 1
    order = present[np.argsort(-area[present], kind="stable")]
    old_to_new = np.zeros(area.size, dtype=np.int64)
    old_to_new[order] = np.arange(1, order.size + 1)
    return old_to_new[labels].astype(labels.dtype), old_to_new


def combine_features_with_share_boundary(
    feature_number,
    perim_frac_thresh=0.2,
    equiv_frac_thresh=0.3,
    equiv_shape="ellipse",
    max_iterations=10,
):
    """
    Combine features that share a large part of their boundary (optional step after detect-and-spread).

    A pair of touching features is flagged when the boundary they share is at least
      perim_frac_thresh x (raw perimeter of the smaller feature)   OR
      equiv_frac_thresh x (perimeter of the smooth shape equivalent to the smaller feature).
    The smaller feature is the one with fewer pixels. All flagged pairs are merged at once (a feature with several
    flagged neighbors joins them all, chains of flagged pairs form one feature). The test is repeated on the merged
    features until nothing is flagged, which also merges a small piece that is wedged between two larger ones and
    shares only a part of its boundary with each. Features that only touch over a short stretch, or at a corner
    point, are not merged.

    The boundary and the perimeters are counted in unit pixel edges (see boundary_lengths), so the two thresholds
    are not interchangeable: the raw perimeter is larger than the equivalent one, so it needs the smaller
    fraction. Only the pixels of the features are relabeled; background pixels and the set of labeled pixels do not
    change. Pixels are assumed square (a lat/lon grid is fine away from the poles for this purpose).

    Args:
        feature_number: np.ndarray, 2D, integer
            Feature labels, 0 = background (e.g. final_Feature_Number of label_and_grow_features).
        perim_frac_thresh: float or None (optional, default=0.2)
            Flag a pair when shared boundary / raw perimeter of the smaller feature >= this. None: no raw test.
        equiv_frac_thresh: float or None (optional, default=0.3)
            Flag a pair when shared boundary / perimeter of the equivalent shape of the smaller feature >= this.
            None: no equivalent-shape test.
        equiv_shape: string (optional, default="ellipse")
            "ellipse" or "circle", see equivalent_perimeter.
        max_iterations: int (optional, default=10)
            Maximum number of merge passes.

    Returns:
        labels: np.ndarray, 2D, same dtype as feature_number
            Combined feature labels, numbered 1..N from the largest to the smallest feature. A copy of
            feature_number when no pair is flagged.
        nfeatures: int
            Number of features N.
        info: dictionary
            nfeatures_before: number of features of the input.
            npasses: number of merge passes that merged something.
            passes: list of dictionaries, one per merge pass: nflagged, nfeatures (after the pass), and pairs, the flagged
                pairs as an (nflagged, 2) array of the labels of that pass (pass 1: the labels of feature_number).
            converged: True when a pass flagged nothing (False when max_iterations was reached first).
            old_to_new: np.ndarray, index = input label, value = new label (0 = label not in the input).
            groups: dictionary, new label -> input labels that were combined into it (merged features only).
    """
    # Check the inputs
    feature_number = np.asarray(feature_number)
    if feature_number.ndim != 2:
        raise ValueError(f"feature_number must be a 2D array, got {feature_number.ndim} dimensions")
    if equiv_shape not in ("ellipse", "circle"):
        raise ValueError(f"equiv_shape must be 'ellipse' or 'circle', got '{equiv_shape}'")
    for name, value in (("perim_frac_thresh", perim_frac_thresh), ("equiv_frac_thresh", equiv_frac_thresh)):
        if value is not None and not value > 0:
            raise ValueError(f"{name} must be None or > 0, got {value}")
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")
    if feature_number.size > 0 and feature_number.min() < 0:
        raise ValueError("feature_number must not have negative labels")

    # Labels as integers (the original dtype is restored in the output)
    labels_in = feature_number.astype(np.int64) if feature_number.dtype.kind not in "iu" else feature_number
    nlabel = int(labels_in.max()) if labels_in.size > 0 else 0
    area0 = np.bincount(labels_in.ravel(), minlength=nlabel + 1)
    present0 = np.nonzero(area0[1:])[0] + 1
    info = {
        "nfeatures_before": int(present0.size), "npasses": 0, "passes": [], "converged": True,
        "old_to_new": np.arange(nlabel + 1), "groups": {},
    }
    # Nothing to combine
    if present0.size < 2 or (perim_frac_thresh is None and equiv_frac_thresh is None):
        return feature_number.copy(), int(present0.size), info

    # Merge passes: flag the pairs, merge the flagged pairs at once, repeat on the merged features
    labels = labels_in.astype(np.int32) if nlabel < np.iinfo(np.int32).max else labels_in
    to_current = np.arange(nlabel + 1)
    info["converged"] = False
    for _ in range(max_iterations):
        perimeter, pairs, shared = boundary_lengths(labels)
        if pairs.shape[0] == 0:
            info["converged"] = True
            break

        # The smaller feature of each pair decides (equal sizes: the lower label)
        area = np.bincount(labels.ravel(), minlength=nlabel + 1)
        smaller = np.where(area[pairs[:, 0]] <= area[pairs[:, 1]], pairs[:, 0], pairs[:, 1])
        flagged = np.zeros(pairs.shape[0], dtype=bool)
        if perim_frac_thresh is not None:
            flagged |= shared >= perim_frac_thresh * perimeter[smaller]
        if equiv_frac_thresh is not None:
            flagged |= shared >= equiv_frac_thresh * equivalent_perimeter(labels, equiv_shape)[smaller]
        if not flagged.any():
            info["converged"] = True
            break

        # Chains and clusters of flagged pairs become one feature (connected components of the pair graph);
        # label 0 stays background
        graph = sparse.coo_matrix(
            (np.ones(int(flagged.sum())), (pairs[flagged, 0], pairs[flagged, 1])),
            shape=(nlabel + 1, nlabel + 1),
        )
        _, component = connected_components(graph, directed=False)
        lut = component + 1
        lut[0] = 0
        labels = lut[labels].astype(labels.dtype)
        to_current = lut[to_current]
        info["npasses"] += 1
        info["passes"].append({
            "nflagged": int(flagged.sum()), "nfeatures": int(np.unique(labels).size - 1),
            "pairs": pairs[flagged].copy(),
        })

    # No merge: the input labels are returned as they are
    if info["npasses"] == 0:
        return feature_number.copy(), int(present0.size), info

    # Number the combined features by size, and record which input labels formed each one
    labels, renumber = renumber_by_size(labels)
    old_to_new = np.zeros(nlabel + 1, dtype=np.int64)       # only the labels that exist in the input get a number
    old_to_new[present0] = renumber[to_current[present0]]
    new_of_present = old_to_new[present0]
    new_labels, count = np.unique(new_of_present, return_counts=True)
    info["old_to_new"] = old_to_new
    info["groups"] = {int(n): present0[new_of_present == n].tolist() for n in new_labels[count > 1]}
    return labels.astype(feature_number.dtype), int(new_labels.size), info
