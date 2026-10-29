"""
Tests of pyflextrkr.combine_features (combining features that share a large part of their boundary).

Label arrays are built by hand (rectangles, a diagonal cut, a disc) so that the shared boundary and the
perimeters are known exactly; the last tests run detect-and-spread (label_and_grow_features) on synthetic
reflectivity frames (tests/combine_features_synthetic.py). Numpy only, no data files.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pyflextrkr.combine_features import (
    boundary_lengths,
    combine_features_with_share_boundary,
    equivalent_perimeter,
    renumber_by_size,
)
from pyflextrkr.label_and_grow_features import label_and_grow_features
from combine_features_synthetic import make_frame


def _blocks(shape, *blocks):
    """Label array with rectangles: each block is (label, row0, row1, col0, col1), end points excluded."""
    labels = np.zeros(shape, dtype=int)
    for label, r0, r1, c0, c1 in blocks:
        labels[r0:r1, c0:c1] = label
    return labels


# ---------------------------------------------------------------------------
# Boundary lengths
# ---------------------------------------------------------------------------
class TestBoundaryLengths:

    def test_straight_cut_counts_its_length(self):
        labels = np.ones((10, 10), dtype=int)
        labels[:, 5:] = 2
        perimeter, pairs, shared = boundary_lengths(labels)
        assert pairs.tolist() == [[1, 2]] and shared.tolist() == [10]
        # a 10 x 5 rectangle: 2 x (10 + 5) edges, of which 10 are shared
        assert perimeter[1] == 30 and perimeter[2] == 30 and perimeter[0] == 0

    def test_diagonal_interface_counts_two_edges_per_step(self):
        n = 10
        ii, jj = np.indices((n, n))
        labels = np.where(jj <= ii, 1, 2)
        _, pairs, shared = boundary_lengths(labels)
        assert pairs.tolist() == [[1, 2]]
        assert shared.tolist() == [2 * (n - 1)]

    def test_features_touching_at_a_corner_point_share_nothing(self):
        labels = _blocks((8, 8), (1, 0, 4, 0, 4), (2, 4, 8, 4, 8))
        _, pairs, shared = boundary_lengths(labels)
        assert pairs.shape == (0, 2) and shared.size == 0

    def test_background_and_domain_edge_belong_to_the_perimeter_only(self):
        labels = _blocks((6, 6), (1, 0, 3, 0, 3))                 # in the corner of the domain
        perimeter, pairs, shared = boundary_lengths(labels)
        assert perimeter[1] == 12 and pairs.shape[0] == 0

    def test_three_features_in_a_row_give_two_pairs(self):
        labels = _blocks((10, 30), (1, 0, 10, 0, 10), (2, 0, 10, 10, 20), (3, 0, 10, 20, 30))
        _, pairs, shared = boundary_lengths(labels)
        assert pairs.tolist() == [[1, 2], [2, 3]] and shared.tolist() == [10, 10]


# ---------------------------------------------------------------------------
# Equivalent perimeter
# ---------------------------------------------------------------------------
class TestEquivalentPerimeter:

    def test_circle_follows_the_area(self):
        labels = _blocks((40, 40), (1, 5, 25, 5, 25))             # 400 px
        expected = 2.0 * np.sqrt(np.pi * 400) * 4.0 / np.pi
        assert equivalent_perimeter(labels, "circle")[1] == pytest.approx(expected)

    def test_ellipse_of_a_square_is_the_circle(self):
        labels = _blocks((40, 40), (1, 5, 25, 5, 25))
        assert equivalent_perimeter(labels, "ellipse")[1] == pytest.approx(equivalent_perimeter(labels, "circle")[1], rel=1e-3)

    def test_ellipse_of_a_line_is_longer_than_the_circle(self):
        labels = _blocks((20, 100), (1, 8, 11, 10, 70))           # 3 x 60 px
        circle = equivalent_perimeter(labels, "circle")[1]
        ellipse = equivalent_perimeter(labels, "ellipse")[1]
        assert ellipse > 2.0 * circle

    def test_label_without_pixels_has_no_perimeter(self):
        labels = _blocks((20, 20), (1, 2, 8, 2, 8), (3, 10, 15, 10, 15))
        assert equivalent_perimeter(labels)[2] == 0.0

    def test_invalid_shape_is_an_error(self):
        with pytest.raises(ValueError):
            equivalent_perimeter(np.ones((3, 3), dtype=int), "square")


# ---------------------------------------------------------------------------
# Which features are combined
# ---------------------------------------------------------------------------
class TestCombineDecisions:

    def test_two_squares_sharing_a_whole_side_merge(self):
        # shared 20 of a perimeter of 80 = 0.25 (raw), 0.22 (circle)
        labels = _blocks((30, 60), (1, 5, 25, 5, 25), (2, 5, 25, 25, 45))
        out, n, info = combine_features_with_share_boundary(labels)
        assert n == 1 and np.array_equal(out > 0, labels > 0) and set(np.unique(out)) == {0, 1}
        assert info["groups"] == {1: [1, 2]} and info["nfeatures_before"] == 2 and info["converged"]

    def test_squares_touching_over_a_short_stretch_stay_apart(self):
        labels = _blocks((50, 60), (1, 5, 25, 5, 25), (2, 23, 43, 25, 45))     # 2 shared edges
        out, n, _ = combine_features_with_share_boundary(labels)
        assert n == 2 and np.array_equal(out, labels)

    def test_squares_touching_at_a_corner_point_stay_apart(self):
        labels = _blocks((30, 30), (1, 2, 12, 2, 12), (2, 12, 22, 12, 22))
        assert combine_features_with_share_boundary(labels)[1] == 2

    def test_a_disc_cut_along_its_diagonal_is_combined(self):
        yy, xx = np.indices((60, 60))
        disc = np.hypot(yy - 30, xx - 30) <= 20
        labels = np.where(disc, np.where(xx + yy < 60, 1, 2), 0)
        out, n, _ = combine_features_with_share_boundary(labels)
        assert n == 1 and np.array_equal(out > 0, disc)

    def test_chain_of_features_forms_one_feature(self):
        labels = _blocks((30, 80), (1, 5, 25, 5, 25), (2, 5, 25, 25, 45), (3, 5, 25, 45, 65))
        out, n, info = combine_features_with_share_boundary(labels)
        assert n == 1 and info["groups"] == {1: [1, 2, 3]}

    def test_feature_with_several_neighbors_joins_them_all(self):
        # a center square touched on 4 sides by squares that do not touch each other
        labels = _blocks((70, 70), (1, 25, 45, 25, 45), (2, 5, 25, 25, 45), (3, 45, 65, 25, 45),
                         (4, 25, 45, 5, 25), (5, 25, 45, 45, 65))
        out, n, info = combine_features_with_share_boundary(labels)
        assert n == 1 and info["groups"] == {1: [1, 2, 3, 4, 5]}

    def test_wedged_piece_needs_the_second_pass(self):
        # A and B share a whole side (merge in pass 1); W touches each of them over 10 edges (0.17 of its
        # perimeter of 60: below the threshold alone) and the merged feature over 20 (0.33)
        labels = _blocks((50, 60), (1, 5, 25, 5, 25), (2, 5, 25, 25, 45), (3, 25, 35, 15, 35))
        out, n, info = combine_features_with_share_boundary(labels)
        assert n == 1 and info["npasses"] == 2 and info["converged"]
        out1, n1, info1 = combine_features_with_share_boundary(labels, max_iterations=1)
        assert n1 == 2 and info1["npasses"] == 1 and not info1["converged"]

    def test_elongated_feature_is_not_treated_as_a_circle(self):
        # a 3 x 60 line with a 10 x 20 block on top: shared 20 of the line's 126 edges
        labels = _blocks((40, 100), (1, 10, 13, 5, 65), (2, 0, 10, 5, 25))
        by_circle = combine_features_with_share_boundary(labels, perim_frac_thresh=None, equiv_shape="circle")[1]
        by_ellipse = combine_features_with_share_boundary(labels, perim_frac_thresh=None, equiv_shape="ellipse")[1]
        assert by_circle == 1 and by_ellipse == 2

    def test_the_raw_and_the_equivalent_test_each_decide_alone(self):
        labels = _blocks((30, 60), (1, 5, 25, 5, 25), (2, 5, 25, 25, 45))      # fractions 0.25 raw, 0.22 circle
        assert combine_features_with_share_boundary(labels, perim_frac_thresh=0.2, equiv_frac_thresh=None)[1] == 1
        assert combine_features_with_share_boundary(labels, perim_frac_thresh=0.3, equiv_frac_thresh=None)[1] == 2
        assert combine_features_with_share_boundary(labels, perim_frac_thresh=None, equiv_frac_thresh=0.2)[1] == 1
        assert combine_features_with_share_boundary(labels, perim_frac_thresh=None, equiv_frac_thresh=0.3)[1] == 2


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
class TestOutput:

    @staticmethod
    def _example():
        # features 1 and 4 merge; 2 is a large separate feature; 3 touches nothing
        return _blocks((60, 90), (1, 5, 25, 5, 25), (4, 5, 25, 25, 45), (2, 30, 58, 50, 85), (3, 40, 44, 5, 9))

    def test_labels_are_contiguous_sorted_by_size_and_pixels_do_not_change(self):
        labels = self._example()
        out, n, _ = combine_features_with_share_boundary(labels)
        assert n == 3 and sorted(np.unique(out)) == [0, 1, 2, 3]
        sizes = np.bincount(out.ravel())[1:]
        assert sizes.tolist() == sorted(sizes.tolist(), reverse=True)
        assert np.array_equal(out > 0, labels > 0)

    def test_old_to_new_and_groups(self):
        labels = self._example()
        out, _, info = combine_features_with_share_boundary(labels)
        old_to_new = info["old_to_new"]
        assert old_to_new[0] == 0 and old_to_new[1] == old_to_new[4] and len({old_to_new[i] for i in (1, 2, 3)}) == 3
        assert np.array_equal(old_to_new[labels], out)
        assert info["groups"] == {int(old_to_new[1]): [1, 4]}

    def test_nothing_flagged_returns_a_copy_of_the_input(self):
        labels = _blocks((50, 60), (1, 5, 25, 5, 25), (2, 23, 43, 25, 45))
        out, n, info = combine_features_with_share_boundary(labels)
        assert out is not labels and np.array_equal(out, labels) and n == 2 and info["npasses"] == 0

    def test_both_tests_off_returns_the_input(self):
        labels = _blocks((30, 60), (1, 5, 25, 5, 25), (2, 5, 25, 25, 45))
        out, n, _ = combine_features_with_share_boundary(labels, perim_frac_thresh=None, equiv_frac_thresh=None)
        assert n == 2 and np.array_equal(out, labels)

    def test_empty_and_single_feature_frames(self):
        assert combine_features_with_share_boundary(np.zeros((10, 10), dtype=int))[1] == 0
        single = _blocks((10, 10), (1, 2, 6, 2, 6))
        assert combine_features_with_share_boundary(single)[1] == 1

    def test_dtype_is_kept(self):
        labels = self._example()
        assert combine_features_with_share_boundary(labels.astype(np.int32))[0].dtype == np.int32
        assert combine_features_with_share_boundary(labels.astype(float))[0].dtype == np.float64

    def test_gaps_in_the_label_numbers_are_fine(self):
        labels = _blocks((30, 60), (3, 5, 25, 5, 25), (9, 5, 25, 25, 45))
        out, n, info = combine_features_with_share_boundary(labels)
        assert n == 1 and info["groups"] == {1: [3, 9]} and out.max() == 1

    def test_invalid_arguments_are_errors(self):
        labels = self._example()
        for kwargs in ({"perim_frac_thresh": 0.0}, {"equiv_frac_thresh": -1.0}, {"equiv_shape": "box"}, {"max_iterations": 0}):
            with pytest.raises(ValueError):
                combine_features_with_share_boundary(labels, **kwargs)
        with pytest.raises(ValueError):
            combine_features_with_share_boundary(labels[0])
        with pytest.raises(ValueError):
            combine_features_with_share_boundary(-labels)

    def test_renumber_by_size(self):
        labels = _blocks((20, 40), (5, 2, 4, 2, 4), (2, 6, 18, 6, 36))
        out, old_to_new = renumber_by_size(labels)
        assert old_to_new[2] == 1 and old_to_new[5] == 2 and out.max() == 2


# ---------------------------------------------------------------------------
# Synthetic reflectivity: detect-and-spread, then combine
# ---------------------------------------------------------------------------
def _detect_and_spread(dbz):
    """label_and_grow_features with the [25, 10] settings of the MRMS configs (smooth 10, 4 core pixels, 81 km2, bfs)."""
    result = label_and_grow_features(
        np.maximum(dbz, 0.0), 3.0, [25.0, 10.0, 10.0, 10.0], 81.0, 4, 10, 0, {"pbc_direction": "none"},
        core_operator="gt", growth_method="bfs",
    )
    return result["final_Feature_Number"], int(result["final_nFeature"])


class TestSyntheticReflectivity:

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_oval_storm_is_one_feature_and_the_barely_touching_pair_is_not(self, seed):
        dbz, storms = make_frame(seed, "cluster")
        labels, n_before = _detect_and_spread(dbz)
        # detect-and-spread: one feature per core
        assert n_before == sum(s["ncores"] for s in storms)
        out, n_after, info = combine_features_with_share_boundary(labels)
        oval = next(s for s in storms if s["name"] == "oval")
        assert len(np.unique(labels[oval["mask"] & (labels > 0)])) >= 4
        assert len(np.unique(out[oval["mask"] & (out > 0)])) == 1
        # the two touching storms and the single storms are separate features: one per storm, the oval is one
        assert n_after == len(storms)
        west, east = (next(s for s in storms if s["name"] == name)["mask"] for name in ("pair_west", "pair_east"))
        west_only, east_only = west & ~east, east & ~west
        assert np.bincount(out[west_only]).argmax() != np.bincount(out[east_only]).argmax()

    @pytest.mark.parametrize("seed", [0, 1])
    def test_spread_storms_are_left_alone(self, seed):
        dbz, storms = make_frame(seed, "spread")
        labels, n_before = _detect_and_spread(dbz)
        out, n_after, info = combine_features_with_share_boundary(labels)
        assert n_before == n_after == len(storms) and info["npasses"] == 0 and np.array_equal(out, labels)
