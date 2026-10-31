"""
Tests of pyflextrkr.overlap_links against the frozen original loops (tests/tracksingle_reference.py): the same links, in the same
order, with the same sizes, for random label images with large and small features that merge and split between the two files.
"""
import numpy as np
import pytest
import scipy.ndimage as ndi

import tracksingle_reference as ref
from pyflextrkr.overlap_links import overlap_links

FILL = -9999


def _label_images(seed, shape=(60, 80), threshold=0.0, shift=(2, 3), noise=0.35):
    """
    Two label images of smooth random fields; the second field is the first one moved and perturbed, so that the features overlap
    partly, some of them with several features of the other file (mergers and splits).

    Returns:
        reference, new: numpy arrays [1, ny, nx] of int
        nreference, nnew: int (number of features + 1, as trackclouds uses them)
    """
    rng = np.random.default_rng(seed)
    base = ndi.gaussian_filter(rng.standard_normal(shape), 2.5)
    other = np.roll(base, shift, axis=(0, 1)) + noise * ndi.gaussian_filter(rng.standard_normal(shape), 2.0)
    scale = base.std()
    labels_ref, n_ref = ndi.label(base > threshold * scale)
    labels_new, n_new = ndi.label(other > threshold * scale)
    return labels_ref[None].astype(int), labels_new[None].astype(int), n_ref + 1, n_new + 1


@pytest.mark.parametrize("othresh", [0.5, 0.2, 0.0])
@pytest.mark.parametrize("seed", range(12))
def test_same_links_as_the_original_loops(seed, othresh):
    reference, new, nreference, nnew = _label_images(seed, threshold=[0.3, 0.0, 0.6][seed % 3])
    got = overlap_links(reference, new, nreference, nnew, 50, othresh, FILL)
    want = ref.overlap_links_reference(reference, new, nreference, nnew, 50, othresh, FILL)
    for g, w in zip(got, want):
        assert g.shape == w.shape and g.dtype == w.dtype
        assert np.array_equal(g, w)
    assert (got[0] > 0).sum() > 0 and (got[2] > 0).sum() > 0               # there are links


def test_the_random_images_have_features_linked_to_several_features():
    # splits and mergers: with any overlap counted, some feature has two or more links in each direction
    most = [0, 0]
    for seed in range(12):
        reference, new, nreference, nnew = _label_images(seed, threshold=[0.3, 0.0, 0.6][seed % 3])
        got = overlap_links(reference, new, nreference, nnew, 50, 0.0, FILL)
        most = [max(most[0], int((got[0] > 0).sum(axis=2).max())), max(most[1], int((got[2] > 0).sum(axis=2).max()))]
    assert most[0] >= 2 and most[1] >= 2


def test_no_overlap_gives_only_fill_values():
    reference = np.zeros((1, 10, 10), dtype=int)
    new = np.zeros((1, 10, 10), dtype=int)
    reference[0, :3, :3] = 1
    new[0, 6:, 6:] = 1
    got = overlap_links(reference, new, 2, 2, 5, 0.5, FILL)
    want = ref.overlap_links_reference(reference, new, 2, 2, 5, 0.5, FILL)
    assert all(np.array_equal(g, w) for g, w in zip(got, want))
    assert all((g == FILL).all() for g in got)


def test_too_many_links_stop_like_the_original():
    # one reference feature that overlaps 6 new features, at most 5 links allowed
    reference = np.ones((1, 4, 12), dtype=int)
    new = np.repeat(np.arange(1, 7), 2)[None, None, :].repeat(4, axis=1)
    with pytest.raises(SystemExit):
        overlap_links(reference, new, 2, 7, 5, 0.0, FILL)
    # (the original stops with the message at 7 or more links and with an IndexError at exactly nmaxlinks + 1 = 6: both stop)
    with pytest.raises((SystemExit, IndexError)):
        ref.overlap_links_reference(reference, new, 2, 7, 5, 0.0, FILL)
    assert overlap_links(reference, new, 2, 7, 6, 0.0, FILL)[0][0, 0].tolist() == [1, 2, 3, 4, 5, 6]


def test_features_beyond_the_counted_ones_are_ignored_like_in_the_original():
    # a label larger than the number of features given (nreference, nnew): the loops never visit it
    reference = np.zeros((1, 6, 6), dtype=int)
    new = np.zeros((1, 6, 6), dtype=int)
    reference[0, :3] = 1
    reference[0, 3:] = 4                       # label 4, but nreference = 3: not visited
    new[0, :, :3] = 1
    new[0, :, 3:] = 5                          # label 5, nnew = 3: not visited as a row, but a valid partner of reference feature 1
    got = overlap_links(reference, new, 3, 3, 5, 0.2, FILL)
    want = ref.overlap_links_reference(reference, new, 3, 3, 5, 0.2, FILL)
    assert all(np.array_equal(g, w) for g, w in zip(got, want))
