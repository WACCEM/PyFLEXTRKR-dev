"""
Synthetic tests for pyflextrkr/identifymcs_radar.py's pure-function core:
MCS criteria testing, merge/split fragment folding and feature classification
(the lifecycle staging is tested in test_mcs_lifecycle_synthetic.py).

No dask import anywhere in identifymcs_radar.py, so all of this runs on CI.
"""

import os
import numpy as np
import pytest
import xarray as xr

from pyflextrkr.identifymcs_radar import identifymcs_radar
from pyflextrkr.identifymcs_radar import (
    group_contiguous_indices,
    identify_mcs_periods,
    find_mergesplit_parents,
    resolve_fragment_roots,
    release_noncoexisting_fragments,
    attach_fragments,
    classify_feature_class,
    classify_track_class,
    CLASS_MCS, CLASS_MULTICELL, CLASS_ISOLATED_CELL, CLASS_NON_CONVECTIVE,
)

DT = 1.0 / 12.0  # 5-minute time resolution [hour]

MCS_PARAMS = {
    'mcs_majoraxis_thresh': 100.0,
    'mcs_dbz45_area_thresh': 0,
    'mcs_duration_thresh': 4.0,
    'mcs_gap_hour': 0.25,
}


# ---------------------------------------------------------------------------
# group_contiguous_indices
# ---------------------------------------------------------------------------

class TestGroupContiguousIndices:
    def test_empty(self):
        assert group_contiguous_indices(np.array([], dtype=int), 0) == []

    def test_single_group_no_gap(self):
        groups = group_contiguous_indices(np.array([0, 1, 2, 3]), 0)
        assert len(groups) == 1
        np.testing.assert_array_equal(groups[0], [0, 1, 2, 3])

    def test_gap_bridged(self):
        # Indices 0,1,3,4: one missing index (2) - bridged with max_gap_steps=1
        groups = group_contiguous_indices(np.array([0, 1, 3, 4]), 1)
        assert len(groups) == 1

    def test_gap_too_large_splits(self):
        groups = group_contiguous_indices(np.array([0, 1, 5, 6]), 1)
        assert len(groups) == 2


# ---------------------------------------------------------------------------
# identify_mcs_periods
# ---------------------------------------------------------------------------

class TestIdentifyMcsPeriods:
    def _make_track(self, n, majoraxis_val=150.0, dbz45_val=10.0):
        majoraxis = np.full(n, majoraxis_val)
        dbz45 = np.full(n, dbz45_val)
        basetime = np.arange(n) * DT * 3600.0
        return majoraxis, dbz45, basetime

    def test_sustained_mcs(self):
        n = 60  # 5 hours at 5-min resolution
        majoraxis, dbz45, basetime = self._make_track(n)
        mcs_status, is_mcs = identify_mcs_periods(majoraxis, dbz45, basetime, DT, MCS_PARAMS)
        assert is_mcs
        assert np.all(mcs_status == 1)

    def test_too_short_not_mcs(self):
        n = 20  # ~1.67 hours, below the 4-hour duration threshold
        majoraxis, dbz45, basetime = self._make_track(n)
        mcs_status, is_mcs = identify_mcs_periods(majoraxis, dbz45, basetime, DT, MCS_PARAMS)
        assert not is_mcs
        assert np.all(mcs_status == 0)

    def test_small_feature_never_mcs(self):
        n = 60
        majoraxis, dbz45, basetime = self._make_track(n, majoraxis_val=50.0)
        mcs_status, is_mcs = identify_mcs_periods(majoraxis, dbz45, basetime, DT, MCS_PARAMS)
        assert not is_mcs

    def test_no_45dbz_area_never_mcs(self):
        n = 60
        majoraxis, dbz45, basetime = self._make_track(n, dbz45_val=0.0)
        mcs_status, is_mcs = identify_mcs_periods(majoraxis, dbz45, basetime, DT, MCS_PARAMS)
        assert not is_mcs

    def test_gap_bridged_still_mcs(self):
        n = 60
        majoraxis, dbz45, basetime = self._make_track(n)
        # Drop criteria for 2 steps in the middle (10 min gap, within mcs_gap_hour=0.25h=3 steps)
        majoraxis[30:32] = 0.0
        mcs_status, is_mcs = identify_mcs_periods(majoraxis, dbz45, basetime, DT, MCS_PARAMS)
        assert is_mcs

    def test_nan_treated_as_not_meeting_criteria(self):
        n = 60
        majoraxis, dbz45, basetime = self._make_track(n)
        majoraxis[:] = np.nan
        mcs_status, is_mcs = identify_mcs_periods(majoraxis, dbz45, basetime, DT, MCS_PARAMS)
        assert not is_mcs
        assert np.all(mcs_status == 0)


# ---------------------------------------------------------------------------
# find_mergesplit_parents / resolve_fragment_roots / attach_fragments
# ---------------------------------------------------------------------------

class TestFragmentFolding:
    def test_simple_merge_fragment(self):
        # Track 0 (short) merges into track 1 (the main track)
        lifetime = np.array([0.5, 5.0])
        end_merge = np.array([2.0, 0.0])  # track 0 ends merging into track 2 (1-based)
        start_split = np.array([0.0, 0.0])
        is_protected = np.array([False, False])
        parent_idx, link_type, is_fragment = find_mergesplit_parents(
            lifetime, end_merge, start_split, is_protected, 1.0, 1.0,
        )
        assert is_fragment[0] and not is_fragment[1]
        assert parent_idx[0] == 1
        assert link_type[0] == 'merge'

    def test_simple_split_fragment(self):
        lifetime = np.array([5.0, 0.5])
        end_merge = np.array([0.0, 0.0])
        start_split = np.array([0.0, 1.0])  # track 1 splits from track 1 (1-based) = track 0
        is_protected = np.array([False, False])
        parent_idx, link_type, is_fragment = find_mergesplit_parents(
            lifetime, end_merge, start_split, is_protected, 1.0, 1.0,
        )
        assert is_fragment[1] and not is_fragment[0]
        assert parent_idx[1] == 0
        assert link_type[1] == 'split'

    def test_protected_track_never_a_fragment(self):
        # Even though short and merging, an MCS-protected track is never a fragment
        lifetime = np.array([0.5, 5.0])
        end_merge = np.array([2.0, 0.0])
        start_split = np.array([0.0, 0.0])
        is_protected = np.array([True, False])
        parent_idx, link_type, is_fragment = find_mergesplit_parents(
            lifetime, end_merge, start_split, is_protected, 1.0, 1.0,
        )
        assert not is_fragment[0]

    def test_long_track_not_a_fragment_despite_merge_link(self):
        lifetime = np.array([5.0, 10.0])  # track 0 too long to be a merge fragment
        end_merge = np.array([2.0, 0.0])
        start_split = np.array([0.0, 0.0])
        is_protected = np.array([False, False])
        parent_idx, link_type, is_fragment = find_mergesplit_parents(
            lifetime, end_merge, start_split, is_protected, 1.0, 1.0,
        )
        assert not is_fragment[0]

    def test_merge_precedence_over_split(self):
        # A short track that both ends by merging and starts by splitting:
        # merge takes precedence.
        lifetime = np.array([0.5, 5.0, 5.0])
        end_merge = np.array([2.0, 0.0, 0.0])
        start_split = np.array([3.0, 0.0, 0.0])
        is_protected = np.array([False, False, False])
        parent_idx, link_type, is_fragment = find_mergesplit_parents(
            lifetime, end_merge, start_split, is_protected, 1.0, 1.0,
        )
        assert is_fragment[0]
        assert link_type[0] == 'merge'
        assert parent_idx[0] == 1

    def test_chain_resolves_to_root(self):
        # A -> B -> C (C is the main, non-fragment track)
        parent_idx = np.array([1, 2, -1])
        is_fragment = np.array([True, True, False])
        root_idx, resolved = resolve_fragment_roots(parent_idx, is_fragment)
        assert root_idx[0] == 2
        assert root_idx[1] == 2
        assert root_idx[2] == 2
        assert not resolved[2]

    def test_cycle_breaks_deterministically_without_crash(self):
        # A <-> B mutual cycle, no genuine anchor
        parent_idx = np.array([1, 0])
        is_fragment = np.array([True, True])
        root_idx, resolved = resolve_fragment_roots(parent_idx, is_fragment)
        # Exactly one of the two must become the (un-fragmented) anchor
        assert resolved.sum() == 1
        anchor = int(np.where(~resolved)[0][0])
        assert root_idx[0] == anchor
        assert root_idx[1] == anchor

    def test_attach_fragments_matches_basetime_and_paints_pixels(self):
        # 2 tracks, track 0 is a merge fragment into track 1
        max_dur = 3
        basetime_full = np.array([
            [100.0, 200.0, 0.0],
            [100.0, 200.0, 300.0],
        ])
        cloudnumber_full = np.array([
            [5, 6, 0],
            [10, 11, 12],
        ], dtype=np.int32)
        track_status_full = np.array([
            [1, 1, 0],
            [1, 1, 1],
        ], dtype=np.int32)
        area_full = np.array([
            [50.0, 60.0, 0.0],
            [500.0, 600.0, 700.0],
        ])
        root_idx = np.array([1, 1])
        link_type = np.array(['merge', ''])
        keep_track_indices = np.array([1])

        (merge_cn, merge_stat, merge_ar,
         split_cn, split_stat, split_ar,
         n_unmatched, n_truncated, fragment_sums) = attach_fragments(
            root_idx, link_type, basetime_full, cloudnumber_full,
            track_status_full, area_full, keep_track_indices,
            nmaxmerge=3, fillval=-9999, fillval_f=np.nan,
        )
        assert n_unmatched == 0
        assert n_truncated == 0
        assert fragment_sums == {}   # no sum_vars requested
        # new_idx=0 (track 1 is the only kept track); matched at times 0 and 1
        assert merge_cn[0, 0, 0] == 5
        assert merge_cn[0, 1, 0] == 6
        assert merge_ar[0, 0, 0] == 50.0
        assert np.all(split_cn == -9999)  # no split fragments

    def test_more_than_nmaxmerge_fragments_truncated_not_crashed(self):
        max_dur = 1
        n_frag = 4
        basetime_full = np.zeros((n_frag + 1, max_dur))
        basetime_full[:, 0] = 100.0
        cloudnumber_full = np.arange(1, (n_frag + 1) * max_dur + 1).reshape(n_frag + 1, max_dur).astype(np.int32)
        track_status_full = np.ones((n_frag + 1, max_dur), dtype=np.int32)
        area_full = np.full((n_frag + 1, max_dur), 10.0)
        root_idx = np.array([n_frag] * n_frag + [n_frag])
        link_type = np.array(['merge'] * n_frag + [''])
        keep_track_indices = np.array([n_frag])

        result = attach_fragments(
            root_idx, link_type, basetime_full, cloudnumber_full,
            track_status_full, area_full, keep_track_indices,
            nmaxmerge=2, fillval=-9999, fillval_f=np.nan,
        )
        n_unmatched, n_truncated = result[6], result[7]
        assert n_truncated == n_frag - 2  # only 2 of 4 fit


class TestReleaseNoncoexistingFragments:
    """A fragment is folded into its root only if every time point exists on the root."""

    @staticmethod
    def _run(parent_idx, basetime_full):
        parent_idx = np.array(parent_idx)
        is_fragment = parent_idx >= 0
        link_type = np.where(is_fragment, 'merge', '').astype('<U6')
        return release_noncoexisting_fragments(
            parent_idx, link_type, is_fragment, np.array(basetime_full, dtype=float),
        )

    def test_fully_coexisting_fragment_stays_attached(self):
        bt = [[100, 200, 0], [100, 200, 300]]
        parent, link, root, resolved, n_rel = self._run([1, -1], bt)
        assert n_rel == 0
        assert resolved[0] and not resolved[1]
        assert root[0] == 1
        assert link[0] == 'merge'

    def test_fragment_before_root_start_is_released(self):
        # Root only starts at 300, the fragment lived at 100-200 (pre-merger history)
        bt = [[100, 200], [300, 400]]
        parent, link, root, resolved, n_rel = self._run([1, -1], bt)
        assert n_rel == 1
        assert not resolved[0]
        assert root[0] == 0                 # its own anchor
        assert parent[0] == -1 and link[0] == ''

    def test_partial_overlap_is_released_not_split(self):
        # Overlaps the root at only one of its three times -> whole fragment stays standalone
        bt = [[100, 200, 300], [300, 400, 0]]
        parent, link, root, resolved, n_rel = self._run([1, -1], bt)
        assert n_rel == 1
        assert not resolved[0]

    def test_chain_releases_only_the_member_that_does_not_coexist(self):
        # A (0) -> B (1) -> R (2); R starts late. A lived before R, B is inside R's lifetime.
        bt = [[100, 200, 0], [300, 350, 0], [300, 350, 400]]
        parent, link, root, resolved, n_rel = self._run([1, 2, -1], bt)
        assert n_rel == 1
        assert not resolved[0] and resolved[1] and not resolved[2]
        assert root[1] == 2

    def test_release_reroots_downstream_fragment_until_fixed_point(self):
        # A (0) -> B (1) -> R (2). B does not coexist with R, so it is released; A then
        # resolves to B (not R) and does not coexist with B either -> released in 2nd pass.
        bt = [[600, 700, 0], [100, 200, 600], [600, 700, 0]]
        parent, link, root, resolved, n_rel = self._run([1, 2, -1], bt)
        assert n_rel == 2
        assert not resolved.any()
        assert np.all(parent == -1) and np.all(link == '')

    def test_result_is_idempotent(self):
        bt = [[100, 200, 0], [100, 200, 300], [500, 600, 0]]
        parent, link, root, resolved, n_rel = self._run([1, -1, 1], bt)
        assert n_rel == 1                   # track 2 never overlaps root 1
        parent2, link2, root2, resolved2, n_rel2 = release_noncoexisting_fragments(
            parent, link, resolved, np.array(bt, dtype=float),
        )
        assert n_rel2 == 0
        assert np.array_equal(resolved, resolved2) and np.array_equal(root, root2)

    def test_no_unmatched_time_points_after_release(self):
        # track 0 coexists with root 2, track 1 does not: after release attach_fragments loses nothing
        bt = np.array([[100, 200, 0], [10, 20, 30], [100, 200, 300]], dtype=float)
        cloudnumber = np.array([[5, 6, 0], [1, 2, 3], [10, 11, 12]], dtype=np.int32)
        status = np.ones((3, 3), dtype=np.int32)
        area = np.full((3, 3), 10.0)
        parent, link, root, resolved, n_rel = self._run([2, 2, -1], bt)
        assert n_rel == 1
        keep = np.where(~resolved)[0]
        assert list(keep) == [1, 2]
        result = attach_fragments(
            root, link, bt, cloudnumber, status, area, keep,
            nmaxmerge=3, fillval=-9999, fillval_f=np.nan,
        )
        merge_cn, n_unmatched = result[0], result[6]
        assert n_unmatched == 0
        assert merge_cn[1, 0, 0] == 5 and merge_cn[1, 1, 0] == 6   # track 0 painted into kept track 2

    def test_cycle_anchor_keeps_no_link(self):
        # A <-> B with identical times: one becomes the anchor and must not keep a link to itself
        bt = [[100, 200], [100, 200]]
        parent, link, root, resolved, n_rel = self._run([1, 0], bt)
        assert n_rel == 0
        assert resolved.sum() == 1
        anchor = int(np.where(~resolved)[0][0])
        assert parent[anchor] == -1 and link[anchor] == ''

    def test_empty_input(self):
        parent, link, root, resolved, n_rel = self._run(np.array([], dtype=int), np.zeros((0, 4)))
        assert n_rel == 0 and len(resolved) == 0


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

class TestFragmentSums:
    """attach_fragments(sum_vars=...) adds up the fragments that are stored in the merge/split slots."""

    @staticmethod
    def _inputs():
        # tracks 0 (merge fragment) and 1 (split fragment) are folded into root track 2;
        # track 3 is a kept track without fragments
        basetime = np.array([[100.0, 200.0, 0.0],
                             [200.0, 300.0, 0.0],
                             [100.0, 200.0, 300.0],
                             [100.0, 200.0, 300.0]])
        cloudnumber = np.array([[5, 6, 0], [7, 8, 0], [10, 11, 12], [20, 21, 22]], dtype=np.int32)
        status = np.ones((4, 3), dtype=np.int32)
        area = np.array([[50.0, 60.0, 0.0], [70.0, 80.0, 0.0],
                         [500.0, 600.0, 700.0], [900.0, 900.0, 900.0]])
        conv = np.array([[5.0, 6.0, 0.0], [7.0, 8.0, 0.0],
                         [50.0, 60.0, 70.0], [90.0, 90.0, 90.0]])
        root_idx = np.array([2, 2, 2, 3])
        link_type = np.array(['merge', 'split', '', ''])
        return basetime, cloudnumber, status, area, conv, root_idx, link_type

    def test_sums_merge_and_split_fragments_at_their_times(self):
        basetime, cloudnumber, status, area, conv, root_idx, link_type = self._inputs()
        result = attach_fragments(
            root_idx, link_type, basetime, cloudnumber, status, area, np.array([2, 3]),
            nmaxmerge=3, fillval=-9999, fillval_f=np.nan,
            sum_vars={"area": area, "conv_area": conv},
        )
        assert result[7] == 0                       # nothing truncated
        sums = result[8]
        assert set(sums) == {"area", "conv_area"}
        assert sums["area"].shape == (2, 3) and sums["area"].dtype == np.float64
        # root 2 (new index 0): the merge fragment sits at its times 0, 1; the split fragment at 1, 2
        np.testing.assert_allclose(sums["area"][0], [50.0, 60.0 + 70.0, 80.0])
        np.testing.assert_allclose(sums["conv_area"][0], [5.0, 6.0 + 7.0, 8.0])
        # track 3 (new index 1) has no fragment: 0, never NaN
        np.testing.assert_array_equal(sums["area"][1], 0.0)

    def test_sums_equal_the_merge_and_split_area_arrays(self):
        basetime, cloudnumber, status, area, conv, root_idx, link_type = self._inputs()
        result = attach_fragments(
            root_idx, link_type, basetime, cloudnumber, status, area, np.array([2, 3]),
            nmaxmerge=3, fillval=-9999, fillval_f=np.nan, sum_vars={"area": area},
        )
        merge_ar, split_ar, sums = result[2], result[5], result[8]
        stored = np.nansum(merge_ar, axis=2) + np.nansum(split_ar, axis=2)
        np.testing.assert_allclose(sums["area"], stored)

    def test_nan_values_are_skipped(self):
        basetime, cloudnumber, status, area, conv, root_idx, link_type = self._inputs()
        conv = conv.copy()
        conv[0, 0] = np.nan                         # the merge fragment has no value at its first time
        result = attach_fragments(
            root_idx, link_type, basetime, cloudnumber, status, area, np.array([2, 3]),
            nmaxmerge=3, fillval=-9999, fillval_f=np.nan, sum_vars={"conv_area": conv},
        )
        sums = result[8]["conv_area"]
        assert not np.isnan(sums).any()
        np.testing.assert_allclose(sums[0], [0.0, 6.0 + 7.0, 8.0])

    def test_fragments_dropped_by_the_slot_limit_are_not_counted(self):
        basetime, cloudnumber, status, area, conv, root_idx, link_type = self._inputs()
        link_type = np.array(['merge', 'merge', '', ''])    # two merge fragments, only one slot
        result = attach_fragments(
            root_idx, link_type, basetime, cloudnumber, status, area, np.array([2, 3]),
            nmaxmerge=1, fillval=-9999, fillval_f=np.nan, sum_vars={"area": area},
        )
        merge_ar, n_truncated, sums = result[2], result[7], result[8]
        assert n_truncated == 1                              # both fragments sit at root time 1
        np.testing.assert_allclose(sums["area"][0], [50.0, 60.0, 80.0])   # the second fragment's 70 at time 1 is dropped
        np.testing.assert_allclose(sums["area"], np.nansum(merge_ar, axis=2))

    def test_without_sum_vars_the_dict_is_empty(self):
        basetime, cloudnumber, status, area, conv, root_idx, link_type = self._inputs()
        result = attach_fragments(
            root_idx, link_type, basetime, cloudnumber, status, area, np.array([2, 3]),
            nmaxmerge=3, fillval=-9999, fillval_f=np.nan,
        )
        assert result[8] == {}


class TestClassification:
    def test_mcs_takes_precedence_over_multicell(self):
        mcs_status = np.array([1, 0, 0, 0])
        ncells = np.array([3, 3, 1, 0])
        fc = classify_feature_class(mcs_status, ncells)
        assert fc[0] == CLASS_MCS
        assert fc[1] == CLASS_MULTICELL
        assert fc[2] == CLASS_ISOLATED_CELL
        assert fc[3] == CLASS_NON_CONVECTIVE

    def test_fillval_passthrough(self):
        mcs_status = np.array([0])
        ncells = np.array([-9999])
        fc = classify_feature_class(mcs_status, ncells, fillval=-9999)
        assert fc[0] == -9999

    def test_track_class_is_most_organized(self):
        fc = np.array([-9999, CLASS_NON_CONVECTIVE, CLASS_ISOLATED_CELL, CLASS_MCS, CLASS_MULTICELL])
        assert classify_track_class(fc) == CLASS_MCS

    def test_track_class_all_fill_returns_fill(self):
        fc = np.array([-9999, -9999])
        assert classify_track_class(fc, fillval=-9999) == -9999


# ---------------------------------------------------------------------------
# End-to-end driver test with a hand-built sparse trackstats file
# ---------------------------------------------------------------------------

def _build_sparse_trackstats_file(tmp_path):
    """
    Build a minimal sparse trackstats_*.nc file directly (matching the
    structure load_sparse_trackstats expects), with 3 tracks:
      - Track 0: a short (2-step) merge fragment, ending by merging into
        track 2.
      - Track 1: a short-lived, never-MCS, single-cell track (no merge/split
        links) - kept, classified isolated-cell.
      - Track 2: a long (70-step, ~5.8h) MCS-qualifying track for its whole
        duration, with an embedded full lifecycle (unorganized -> genesis ->
        mature -> decay), receiving track 0 as a merge fragment.
    """
    dt_hr = 1.0 / 12.0  # 5-minute resolution
    max_trackduration = 70
    ntracks = 3
    t0_epoch = 1746200000.0
    dt_sec = dt_hr * 3600.0

    track_duration = np.array([2, 10, 70], dtype=np.int32)
    end_merge_tracknumber = np.array([3.0, 0.0, 0.0])  # track 0 merges into track index 2 (1-based: 3)
    start_split_tracknumber = np.array([0.0, 0.0, 0.0])

    majoraxis = np.zeros((ntracks, max_trackduration))
    area_45dbz = np.zeros((ntracks, max_trackduration))
    base_time = np.zeros((ntracks, max_trackduration))
    ncells = np.zeros((ntracks, max_trackduration))
    conv_area = np.zeros((ntracks, max_trackduration))
    conv_majoraxis = np.zeros((ntracks, max_trackduration))
    strat_area = np.zeros((ntracks, max_trackduration))
    cloudnumber = np.zeros((ntracks, max_trackduration), dtype=np.int32)
    track_status = np.zeros((ntracks, max_trackduration), dtype=np.int32)
    area = np.zeros((ntracks, max_trackduration))
    meanlat = np.zeros((ntracks, max_trackduration))
    meanlon = np.zeros((ntracks, max_trackduration))

    # Track 0: merge fragment, same basetime as track 2's first 2 steps
    d0 = track_duration[0]
    base_time[0, 0:d0] = t0_epoch + np.arange(d0) * dt_sec
    majoraxis[0, 0:d0] = 10.0
    cloudnumber[0, 0:d0] = [101, 102]
    track_status[0, 0:d0] = 32
    area[0, 0:d0] = 50.0
    conv_area[0, 0:d0] = 20.0
    strat_area[0, 0:d0] = 30.0
    meanlat[0, 0:d0] = 35.0
    meanlon[0, 0:d0] = -100.0

    # Track 1: short-lived, isolated-cell, never MCS
    d1 = track_duration[1]
    base_time[1, 0:d1] = t0_epoch + 10000.0 + np.arange(d1) * dt_sec
    majoraxis[1, 0:d1] = 30.0
    ncells[1, 0:d1] = 1
    area[1, 0:d1] = 80.0
    meanlat[1, 0:d1] = 36.0
    meanlon[1, 0:d1] = -99.0

    # Track 2: long MCS track with a full embedded lifecycle
    d2 = track_duration[2]
    base_time[2, 0:d2] = t0_epoch + np.arange(d2) * dt_sec
    majoraxis[2, 0:d2] = 150.0
    area_45dbz[2, 0:d2] = 50.0
    conv_area[2, 0:d2] = 50.0
    ncells[2, 0:d2] = 3
    conv_majoraxis[2, 0:10] = 20.0       # not yet organized
    conv_majoraxis[2, 10:d2] = 150.0     # organized from step 10 (50 min in)
    strat_area[2, 0:40] = 0.0
    strat_area[2, 40:55] = 500.0         # stratiform appears, then disappears
    cloudnumber[2, 0:d2] = np.arange(200, 200 + d2)
    track_status[2, 0:d2] = 1
    area[2, 0:d2] = 40000.0
    meanlat[2, 0:d2] = 37.0
    meanlon[2, 0:d2] = -98.0

    # Build sparse (row, col) pairs covering every track's valid (0:duration) entries
    rows, cols = [], []
    for it, dur in enumerate(track_duration):
        rows.extend([it] * int(dur))
        cols.extend(range(int(dur)))
    rows = np.array(rows, dtype=np.int64)
    cols = np.array(cols, dtype=np.int64)

    def _flatten(arr2d):
        return arr2d[rows, cols]

    sparse_vars = {
        "majoraxis": _flatten(majoraxis),
        "area_45dbz": _flatten(area_45dbz),
        "base_time": _flatten(base_time),
        "ncells": _flatten(ncells).astype(np.int32),
        "conv_area": _flatten(conv_area),
        "conv_majoraxis": _flatten(conv_majoraxis),
        "strat_area": _flatten(strat_area),
        "cloudnumber": _flatten(cloudnumber).astype(np.int32),
        "track_status": _flatten(track_status).astype(np.int32),
        "area": _flatten(area),
        "meanlat": _flatten(meanlat),
        "meanlon": _flatten(meanlon),
        "tracks_indices": rows,
        "times_indices": cols,
    }

    varlist = {}
    for key, value in sparse_vars.items():
        varlist[key] = (["sparse_index"], value)
    varlist["track_duration"] = (["tracks"], track_duration)
    varlist["end_merge_tracknumber"] = (["tracks"], end_merge_tracknumber)
    varlist["start_split_tracknumber"] = (["tracks"], start_split_tracknumber)

    coordlist = {
        "tracks": (["tracks"], np.arange(ntracks)),
        "sparse_index": (["sparse_index"], np.arange(len(rows))),
    }
    ds = xr.Dataset(varlist, coords=coordlist)

    stats_path = str(tmp_path) + "/"
    os.makedirs(stats_path, exist_ok=True)
    outfile = os.path.join(stats_path, "trackstats_sparse_20250101.0000_20250101.1200.nc")
    ds.to_netcdf(outfile)
    return stats_path


def _make_driver_config(stats_path):
    return {
        "mcsrobust_filebase": "mcs_tracks_robust_",
        "trackstats_sparse_filebase": "trackstats_sparse_",
        "stats_outpath": stats_path,
        "startdate": "20250101.0000",
        "enddate": "20250101.1200",
        "datatimeresolution": 1.0 / 12.0,
        "duration_range": [2, 100],
        "tracks_dimname": "tracks",
        "times_dimname": "times",
        "nmaxlinks": 10,
        "fillval": -9999,
        "max_speed_for_trajectory_smoothing": 40,
        "mcs_majoraxis_thresh": 100,
        "mcs_dbz45_area_thresh": 0,
        "mcs_duration_thresh": 4,
        "mcs_gap_hour": 0.25,
        "mcs_lifecycle_line_thresh": 100,
        "mcs_lifecycle_gap_hour": 1.0,
        "mcs_lifecycle_genesis_lag_hour": 1.0,
        "mcs_lifecycle_mature_lag_hour": 2.0,
        "mcs_lifecycle_sf_area_min": 0,
        "mcs_lifecycle_thresh": 8,
        "mcs_merge_duration": 1.0,
        "mcs_split_duration": 1.0,
        "mcs_dbz45_area_varname": "area_45dbz",
        "finalstats_min_area": 0,
        "finalstats_min_duration": None,
    }


class TestIdentifymcsRadarEndToEnd:

    def test_driver_runs_and_drops_fragment(self, tmp_path):
        stats_path = _build_sparse_trackstats_file(tmp_path)
        config = _make_driver_config(stats_path)

        outfile = identifymcs_radar(config)
        assert os.path.isfile(outfile)
        # Named like the MCS Tb stats of the Tb+PF pipelines (config mcsradarstats_filebase, default mcs_tracks_);
        # regionprops_driver writes the mcsrobust_filebase file from it
        assert os.path.basename(outfile) == "mcs_tracks_20250101.0000_20250101.1200.nc"
        config["mcsradarstats_filebase"] = "radar_stats_"
        assert os.path.basename(identifymcs_radar(config)) == "radar_stats_20250101.0000_20250101.1200.nc"

        ds = xr.open_dataset(outfile, decode_times=False, mask_and_scale=False)
        # Track 0 (the merge fragment) should have been dropped; 2 tracks remain.
        assert ds.sizes["tracks"] == 2

        # Among the 2 kept tracks, find the long MCS track (track_duration==70)
        track_duration = ds["track_duration"].values
        mcs_track = int(np.where(track_duration == 70)[0][0])
        isolated_track = int(np.where(track_duration == 10)[0][0])

        assert ds["is_mcs"].values[mcs_track] == 1
        assert ds["track_class"].values[mcs_track] == CLASS_MCS
        assert ds["track_class"].values[isolated_track] == CLASS_ISOLATED_CELL

        # The fragment's cloudnumbers (101, 102) should appear in the MCS
        # track's merge_cloudnumber at the matching times.
        merge_cn = ds["merge_cloudnumber"].values[mcs_track, 0:2, :]
        assert 101 in merge_cn
        assert 102 in merge_cn

        # the lifecycle is staged by the second pass (regionprops_driver), not here
        assert not any(v.startswith("lifecycle_") for v in ds.data_vars)

    def test_total_area_includes_folded_fragments(self, tmp_path):
        stats_path = _build_sparse_trackstats_file(tmp_path)
        config = _make_driver_config(stats_path)
        ds = xr.open_dataset(identifymcs_radar(config), decode_times=False, mask_and_scale=False)
        mcs = int(np.where(ds["track_duration"].values == 70)[0][0])
        area = ds["area"].values
        total = ds["total_area"].values
        assert ds["total_area"].dims == ("tracks", "times")
        # the 2-step fragment (area 50 at each step) is folded into the MCS at its first two steps
        np.testing.assert_allclose(total[mcs, 0:2], area[mcs, 0:2] + 50.0)
        np.testing.assert_allclose(total[mcs, 2:70], area[mcs, 2:70])
        assert np.all(np.isnan(total[mcs, 70:]))
        # identity for every kept track, NaN where the track has no data
        fragments = np.nansum(ds["merge_area"].values, axis=2) + np.nansum(ds["split_area"].values, axis=2)
        valid = ~np.isnan(area)
        np.testing.assert_allclose(total[valid], (area + fragments)[valid], rtol=1e-6)
        assert np.all(np.isnan(total[~valid]))
        # a track that received no fragment keeps its own area
        other = int(np.where(ds["track_duration"].values == 10)[0][0])
        np.testing.assert_allclose(total[other, :10], area[other, :10])

    def test_total_conv_and_strat_area_include_folded_fragments(self, tmp_path):
        stats_path = _build_sparse_trackstats_file(tmp_path)
        config = _make_driver_config(stats_path)
        ds = xr.open_dataset(identifymcs_radar(config), decode_times=False, mask_and_scale=False)
        mcs = int(np.where(ds["track_duration"].values == 70)[0][0])
        other = int(np.where(ds["track_duration"].values == 10)[0][0])
        for name, frag in (("conv_area", 20.0), ("strat_area", 30.0)):
            own = ds[name].values
            total = ds[f"total_{name}"].values
            assert ds[f"total_{name}"].dims == ("tracks", "times")
            assert ds[f"total_{name}"].attrs["units"] == "km^2"
            # the fragment (conv 20, strat 30 at each of its 2 steps) is added to the MCS at its first two steps
            np.testing.assert_allclose(total[mcs, 0:2], own[mcs, 0:2] + frag)
            np.testing.assert_allclose(total[mcs, 2:70], own[mcs, 2:70])
            assert np.all(np.isnan(total[mcs, 70:]))
            # a track without fragments keeps its own value, and the NaN pattern is that of the data
            np.testing.assert_allclose(total[other, :10], own[other, :10])
            assert np.all(np.isnan(total[other, 10:]))
        # the stratiform episode of the MCS (500 km2 from step 40 to 54) is untouched by the fragment
        np.testing.assert_allclose(ds["total_strat_area"].values[mcs, 40:55], 500.0)

    def test_totals_stay_aligned_when_tracks_are_filtered(self, tmp_path):
        # finalstats_min_area drops the 80 km2 isolated track after the fragments were attached:
        # the per-track sums must be filtered the same way as the merge/split arrays
        stats_path = _build_sparse_trackstats_file(tmp_path)
        config = _make_driver_config(stats_path)
        config["finalstats_min_area"] = 1000
        ds = xr.open_dataset(identifymcs_radar(config), decode_times=False, mask_and_scale=False)
        assert ds.sizes["tracks"] == 1
        assert ds["track_duration"].values[0] == 70
        np.testing.assert_allclose(ds["total_conv_area"].values[0, 0:2], 50.0 + 20.0)
        np.testing.assert_allclose(ds["total_strat_area"].values[0, 0:2], 0.0 + 30.0)
        np.testing.assert_allclose(ds["total_area"].values[0, 0:2], 40000.0 + 50.0)

    def test_driver_no_mcs_does_not_crash(self, tmp_path):
        """
        Raise the majoraxis threshold requirement to an unreachable value so
        NO track qualifies as MCS - the driver must still run to completion
        (this is the scenario that raised a NameError in robustmcs_radar.py).
        """
        stats_path = _build_sparse_trackstats_file(tmp_path)
        config = _make_driver_config(stats_path)
        config["mcs_majoraxis_thresh"] = 100000  # unreachable

        outfile = identifymcs_radar(config)
        assert os.path.isfile(outfile)
        ds = xr.open_dataset(outfile, decode_times=False, mask_and_scale=False)
        assert np.all(ds["is_mcs"].values == 0)

    def test_driver_alone_with_stale_duration_range_auto_update(self, tmp_path):
        """
        Running identifymcs_radar without run_trackstats in the same run: the config's
        duration_range was never expanded by trackstats_driver and is shorter than the
        longest track (70 steps) in the sparse file. With duration_range_auto_update the
        driver must size itself from the file instead of failing in the sparse loader.
        """
        stats_path = _build_sparse_trackstats_file(tmp_path)
        reference = _make_driver_config(stats_path)  # duration_range [2, 100]
        ref_ds = xr.open_dataset(identifymcs_radar(reference), decode_times=False, mask_and_scale=False)

        config = _make_driver_config(stats_path)
        config["duration_range"] = [2, 10]
        config["duration_range_auto_update"] = True
        config["duration_range_round_base"] = 10
        outfile = identifymcs_radar(config)

        ds = xr.open_dataset(outfile, decode_times=False, mask_and_scale=False)
        # 70 time steps in the file -> already a multiple of 10
        assert ds.sizes["times"] == 70
        assert ds.sizes["tracks"] == ref_ds.sizes["tracks"] == 2
        mcs_track = int(np.where(ds["track_duration"].values == 70)[0][0])
        assert ds["is_mcs"].values[mcs_track] == 1
        assert ds["track_class"].values[mcs_track] == CLASS_MCS
        # Same results as with a duration_range that was already large enough
        np.testing.assert_array_equal(ds["track_duration"].values, ref_ds["track_duration"].values)
        np.testing.assert_array_equal(ds["track_class"].values, ref_ds["track_class"].values)
        np.testing.assert_array_equal(
            ds["merge_cloudnumber"].values[:, 0:70, :], ref_ds["merge_cloudnumber"].values[:, 0:70, :]
        )

    def test_driver_alone_with_stale_duration_range_no_auto_update_exits(self, tmp_path):
        stats_path = _build_sparse_trackstats_file(tmp_path)
        config = _make_driver_config(stats_path)
        config["duration_range"] = [2, 10]
        config["duration_range_auto_update"] = False
        with pytest.raises(SystemExit):
            identifymcs_radar(config)
