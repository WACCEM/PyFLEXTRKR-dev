"""
Links between the features of two consecutive files from their overlap (the core of tracksingle_drift.trackclouds).

For every feature of the reference file the features of the new file that overlap it are listed, in ascending order of their number,
if the overlap is larger than `othresh` times the size of the reference feature (forward links); the same from the new file to the
reference file (backward links). The original code scanned the whole grid for every feature, 3 times in each direction
(about 10 core-seconds per pair of files for 250 features on the 1059 x 1799 grid). Here the overlap of all pairs of features is counted in
one pass over the pixels where both files have a feature (a histogram of the pairs of feature numbers), and the links are read
from it. The links, their order and the stored sizes are those of the loops (tests/tracksingle_reference.py is a frozen copy).

Author: Zhe Feng | zhe.feng@pnnl.gov
"""
import sys

import numpy as np


def _rank_in_group(group):
    """
    Position of every element within its group of equal values in an ascending array.

    Args:
        group: numpy array of int, ascending
            Group number of each element.

    Returns:
        rank: numpy array of int
            0 for the first element of each group, 1 for the second, ...
    """
    if len(group) == 0:
        return np.zeros(0, dtype=np.int64)
    starts = np.flatnonzero(np.r_[True, group[1:] != group[:-1]])
    sizes = np.diff(np.r_[starts, len(group)])
    return np.arange(len(group)) - np.repeat(starts, sizes)


def overlap_links(reference_labels, new_labels, nreference, nnew, nmaxlinks, othresh, fillval):
    """
    Forward and backward links between the features of a reference and a new file.

    Arguments:
        reference_labels, new_labels: numpy arrays of int, same shape
            Feature number of every pixel (0: no feature) in the reference file and in the new file.
        nreference, nnew: int
            Number of features + 1 of the reference and the new file; the links are made for the feature numbers 1 .. nreference
            of the reference file and 1 .. nnew of the new file (the arrays have that many rows).
        nmaxlinks: int
            Maximum number of links of a feature; more is an error (the run stops, as in trackclouds).
        othresh: float
            A link needs an overlap larger than this fraction of the size of the feature the row belongs to.
        fillval: int
            Value of the elements without a link.

    Returns:
        forward_index, forward_size, backward_index, backward_size: numpy arrays of int
            forward_index [1, nreference, nmaxlinks]: row r holds the numbers of the new features linked to reference feature r + 1
            (ascending); forward_size the size of those new features [pixels]. backward_index [1, nnew, nmaxlinks]: row q holds the
            numbers of the reference features linked to new feature q + 1; backward_size the size of those reference features.
    """
    ref = np.asarray(reference_labels).ravel()
    new = np.asarray(new_labels).ravel()
    forward_index = np.ones((1, int(nreference), int(nmaxlinks)), dtype=int) * fillval
    forward_size = np.ones((1, int(nreference), int(nmaxlinks)), dtype=int) * fillval
    backward_index = np.ones((1, int(nnew), int(nmaxlinks)), dtype=int) * fillval
    backward_size = np.ones((1, int(nnew), int(nmaxlinks)), dtype=int) * fillval

    # Pixels where both files have a feature: the pairs of feature numbers and how many pixels each pair has
    both = (ref > 0) & (new > 0)
    if not both.any():
        return forward_index, forward_size, backward_index, backward_size
    pair_ref = ref[both].astype(np.int64)
    pair_new = new[both].astype(np.int64)
    base = int(max(pair_ref.max(), pair_new.max())) + 1
    keys, overlap = np.unique(pair_ref * base + pair_new, return_counts=True)       # ascending: by reference, then by new feature
    ref_of_pair, new_of_pair = keys // base, keys % base

    # Sizes of the features of each file (whole features, also where the other file has none)
    size_ref = np.bincount(ref[ref > 0].astype(np.int64), minlength=base)
    size_new = np.bincount(new[new > 0].astype(np.int64), minlength=base)

    # Forward links: for each reference feature the new features whose overlap is larger than othresh of its size
    keep = (ref_of_pair <= nreference) & (overlap / size_ref[ref_of_pair].astype(float) > othresh)
    r, m = ref_of_pair[keep], new_of_pair[keep]
    slot = _rank_in_group(r)
    if len(slot) and slot.max() + 1 > nmaxlinks:
        sys.exit("More than " + str(int(nmaxlinks)) + " clouds in new file match with reference cloud?!")
    forward_index[0, r - 1, slot] = m
    forward_size[0, r - 1, slot] = size_new[m]

    # Backward links: the same pairs, ordered by new feature, then by reference feature
    order = np.lexsort((ref_of_pair, new_of_pair))
    q_all, r_all, count = new_of_pair[order], ref_of_pair[order], overlap[order]
    keep = (q_all <= nnew) & (count / size_new[q_all].astype(float) > othresh)
    q, r = q_all[keep], r_all[keep]
    slot = _rank_in_group(q)
    if len(slot) and slot.max() + 1 > nmaxlinks:
        sys.exit("More than " + str(int(nmaxlinks)) + " clouds in reference file match with new cloud?!")
    backward_index[0, q - 1, slot] = r
    backward_size[0, q - 1, slot] = size_ref[r]

    return forward_index, forward_size, backward_index, backward_size
