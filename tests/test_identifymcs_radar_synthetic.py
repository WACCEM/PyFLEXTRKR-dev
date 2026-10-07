"""
Synthetic tests for pyflextrkr/identifymcs_radar.py's pure-function core:
MCS criteria testing, merge/split fragment folding, feature classification,
and - the specific focus the task asked to harden - multi-cycle-aware
lifecycle staging.

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
    classify_lifecycle,
    STAGE_CLOUD_ONLY, STAGE_UNORGANIZED, STAGE_GENESIS, STAGE_MATURE, STAGE_DECAY,
    CLASS_MCS, CLASS_MULTICELL, CLASS_ISOLATED_CELL, CLASS_NON_CONVECTIVE,
)

DT = 1.0 / 12.0  # 5-minute time resolution [hour]

LIFECYCLE_PARAMS = {
    'mcs_lifecycle_line_thresh': 100.0,  # km
    'mcs_lifecycle_gap_hour': 1.0,
    'mcs_lifecycle_genesis_lag_hour': 1.0,
    'mcs_lifecycle_mature_lag_hour': 2.0,
    'mcs_lifecycle_sf_area_min': 0,
}

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
         n_unmatched, n_truncated) = attach_fragments(
            root_idx, link_type, basetime_full, cloudnumber_full,
            track_status_full, area_full, keep_track_indices,
            nmaxmerge=3, fillval=-9999, fillval_f=np.nan,
        )
        assert n_unmatched == 0
        assert n_truncated == 0
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
# Lifecycle staging - the scenarios the task explicitly asked to harden
# ---------------------------------------------------------------------------

def _hours_to_steps(hours):
    return int(round(hours / DT))


class TestLifecycleStaging:

    def test_s1_full_monotonic_cycle(self):
        """
        Full textbook lifecycle: cloud-only -> unorganized convection ->
        genesis (line organizes) -> mature (stratiform develops) -> decay.
        """
        n = _hours_to_steps(12)
        conv_area = np.zeros(n)
        conv_majoraxis = np.zeros(n)
        strat_area = np.zeros(n)

        # Unorganized convection starts at hour 1
        t_conv_start = _hours_to_steps(1)
        conv_area[t_conv_start:] = 50.0
        conv_majoraxis[t_conv_start:] = 20.0  # not yet organized (<100km)

        # Line organizes (>=100km) at hour 3, for the rest of the track
        t_line_start = _hours_to_steps(3)
        conv_majoraxis[t_line_start:] = 150.0

        # Stratiform develops starting hour 6 (2h after line = mature_lag), ends hour 9
        t_sf_start = _hours_to_steps(6)
        t_sf_end = _hours_to_steps(9)
        strat_area[t_sf_start:t_sf_end] = 500.0

        stage, cycle, lc_idx, ncyc, complete = classify_lifecycle(
            conv_area, conv_majoraxis, strat_area, DT, LIFECYCLE_PARAMS,
        )
        assert ncyc == 1
        assert lc_idx[0] == 0  # stage 1 (cloud only) starts at t=0
        assert lc_idx[1] == t_conv_start  # stage 2 starts when convection appears
        # Genesis starts at line_start + genesis_lag (1h)
        expected_genesis_start = t_line_start + _hours_to_steps(1)
        assert lc_idx[2] == expected_genesis_start
        assert lc_idx[3] == t_sf_start  # mature starts when stratiform appears (after the lag search window)
        assert lc_idx[4] == t_sf_end  # decay starts right after stratiform disappears
        assert complete

    def test_s2_no_mature_stage(self):
        """Convective line organizes but stratiform never develops -> no mature stage."""
        n = _hours_to_steps(10)
        conv_area = np.full(n, 50.0)
        conv_majoraxis = np.zeros(n)
        conv_majoraxis[_hours_to_steps(1):] = 150.0
        strat_area = np.zeros(n)  # never any stratiform

        stage, cycle, lc_idx, ncyc, complete = classify_lifecycle(
            conv_area, conv_majoraxis, strat_area, DT, LIFECYCLE_PARAMS,
        )
        assert lc_idx[3] == -1  # mature never observed
        # Genesis extends all the way to the track's end (the organized line
        # never stops within the track), so there is no time left for decay
        # to be observed either - correctly distinct from "mature lasting to
        # the end" (test_mature_lasting_to_track_end_no_decay_observed).
        assert lc_idx[4] == -1
        assert not complete

    def test_s3_two_separate_convective_episodes(self):
        """
        The key multi-cycle robustness case: two separate organized-line
        episodes (e.g. re-intensification after a lull), separated by a gap
        longer than mcs_lifecycle_gap_hour. Both episodes must be staged
        (genesis/decay at minimum for each), not just the larger one.
        """
        n = _hours_to_steps(20)
        conv_area = np.full(n, 50.0)
        conv_majoraxis = np.zeros(n)
        strat_area = np.zeros(n)

        # Episode 1: hours 1-4 (organized line)
        e1_start, e1_end = _hours_to_steps(1), _hours_to_steps(4)
        conv_majoraxis[e1_start:e1_end] = 150.0

        # Gap: hours 4-9 (5 hours, > mcs_lifecycle_gap_hour=1h) - line disorganizes
        # Episode 2: hours 9-14
        e2_start, e2_end = _hours_to_steps(9), _hours_to_steps(14)
        conv_majoraxis[e2_start:e2_end] = 150.0

        stage, cycle, lc_idx, ncyc, complete = classify_lifecycle(
            conv_area, conv_majoraxis, strat_area, DT, LIFECYCLE_PARAMS,
        )
        assert ncyc == 2, f"Expected 2 separate episodes, got {ncyc}"
        # Both episodes should have contributed a genesis stage (3) at some point
        assert np.any(stage[e1_start:e1_end] == STAGE_GENESIS)
        assert np.any(stage[e2_start:e2_end] == STAGE_GENESIS)
        # cycle should distinguish episode 1 (cycle==1) from episode 2 (cycle==2)
        assert np.any(cycle == 1)
        assert np.any(cycle == 2)
        # Between the two episodes, the track should show decay (5) from episode 1
        # Also confirm this DIFFERS from the "largest-group-only" behavior: the
        # first episode must leave a trace (not be silently overwritten/ignored).
        assert lc_idx[2] != -1  # genesis was observed (from episode 1, the first)

    def test_s4_convection_never_organizes_into_line(self):
        """Convection present throughout, but majoraxis never reaches the line threshold."""
        n = _hours_to_steps(10)
        conv_area = np.full(n, 50.0)
        conv_majoraxis = np.full(n, 20.0)  # always well below 100km threshold
        strat_area = np.zeros(n)

        stage, cycle, lc_idx, ncyc, complete = classify_lifecycle(
            conv_area, conv_majoraxis, strat_area, DT, LIFECYCLE_PARAMS,
        )
        assert ncyc == 0
        assert np.all(stage == STAGE_UNORGANIZED)
        assert lc_idx[2] == -1 and lc_idx[3] == -1 and lc_idx[4] == -1
        assert not complete

    def test_s5_episode_too_short_for_genesis(self):
        """A brief line-organization blip shorter than genesis_lag_hour never reaches genesis."""
        n = _hours_to_steps(5)
        conv_area = np.full(n, 50.0)
        conv_majoraxis = np.zeros(n)
        # Line organized for only 2 time steps (10 min, << genesis_lag_hour=1h)
        conv_majoraxis[2:4] = 150.0
        strat_area = np.zeros(n)

        stage, cycle, lc_idx, ncyc, complete = classify_lifecycle(
            conv_area, conv_majoraxis, strat_area, DT, LIFECYCLE_PARAMS,
        )
        assert ncyc == 1  # episode detected...
        assert lc_idx[2] == -1  # ...but too short to reach genesis
        assert np.all(stage == STAGE_UNORGANIZED)  # stays at baseline throughout

    def test_s6_no_track_meets_lifecycle_thresh_no_crash(self):
        """
        The identifymcs_radar() driver only calls classify_lifecycle for
        tracks at/above mcs_lifecycle_thresh - but classify_lifecycle itself
        must never crash regardless of input, including an all-zero/short
        track (this is the scenario that caused a NameError in
        robustmcs_radar.py when NO track met the threshold at all).
        """
        for n in [0, 1, 3]:
            conv_area = np.zeros(n)
            conv_majoraxis = np.zeros(n)
            strat_area = np.zeros(n)
            stage, cycle, lc_idx, ncyc, complete = classify_lifecycle(
                conv_area, conv_majoraxis, strat_area, DT, LIFECYCLE_PARAMS,
            )
            assert len(stage) == n
            if n == 0:
                assert list(lc_idx) == [-1, -1, -1, -1, -1]
            else:
                # Stage 1 (cloud only) is the R0 baseline, always observed
                # starting at t=0 for any non-empty track with no convection;
                # only stages 2-5 (which require actual convection/
                # organization) stay unobserved.
                assert lc_idx[0] == 0
                assert list(lc_idx[1:]) == [-1, -1, -1, -1]
            assert not complete

    def test_nan_inputs_treated_as_zero_absent(self):
        n = _hours_to_steps(5)
        conv_area = np.full(n, np.nan)
        conv_majoraxis = np.full(n, np.nan)
        strat_area = np.full(n, np.nan)
        stage, cycle, lc_idx, ncyc, complete = classify_lifecycle(
            conv_area, conv_majoraxis, strat_area, DT, LIFECYCLE_PARAMS,
        )
        assert np.all(stage == STAGE_CLOUD_ONLY)
        assert ncyc == 0

    def test_line_present_from_track_start(self):
        """Edge case: the convective line is already organized at t=0 (a=0)."""
        n = _hours_to_steps(8)
        conv_area = np.full(n, 50.0)
        conv_majoraxis = np.full(n, 150.0)  # organized from t=0
        strat_area = np.zeros(n)
        strat_area[_hours_to_steps(3):] = 500.0

        stage, cycle, lc_idx, ncyc, complete = classify_lifecycle(
            conv_area, conv_majoraxis, strat_area, DT, LIFECYCLE_PARAMS,
        )
        assert ncyc == 1
        # Genesis should start at genesis_lag_hour (1h) from track start (a=0)
        assert lc_idx[2] == _hours_to_steps(1)

    def test_mature_lasting_to_track_end_no_decay_observed(self):
        """If the mature stage never ends within the track, decay is never observed."""
        n = _hours_to_steps(10)
        conv_area = np.full(n, 50.0)
        conv_majoraxis = np.full(n, 150.0)
        strat_area = np.full(n, 500.0)  # stratiform present (and sustained) throughout

        stage, cycle, lc_idx, ncyc, complete = classify_lifecycle(
            conv_area, conv_majoraxis, strat_area, DT, LIFECYCLE_PARAMS,
        )
        assert lc_idx[3] != -1  # mature observed
        assert lc_idx[4] == -1  # decay never observed (track ends while still mature)
        assert not complete  # "complete" requires decay too

    def test_gap_within_tolerance_keeps_one_episode(self):
        """A short dropout within mcs_lifecycle_gap_hour stays one episode, not two."""
        n = _hours_to_steps(10)
        conv_area = np.full(n, 50.0)
        conv_majoraxis = np.full(n, 150.0)
        gap_steps = _hours_to_steps(LIFECYCLE_PARAMS['mcs_lifecycle_gap_hour'])
        dropout_start = _hours_to_steps(4)
        conv_majoraxis[dropout_start:dropout_start + gap_steps] = 0.0  # dropout <= gap tolerance
        strat_area = np.zeros(n)

        stage, cycle, lc_idx, ncyc, complete = classify_lifecycle(
            conv_area, conv_majoraxis, strat_area, DT, LIFECYCLE_PARAMS,
        )
        assert ncyc == 1

    def test_gap_beyond_tolerance_splits_into_two_episodes(self):
        n = _hours_to_steps(15)
        conv_area = np.full(n, 50.0)
        conv_majoraxis = np.full(n, 150.0)
        gap_steps = _hours_to_steps(LIFECYCLE_PARAMS['mcs_lifecycle_gap_hour'])
        dropout_start = _hours_to_steps(4)
        conv_majoraxis[dropout_start:dropout_start + gap_steps + 2] = 0.0  # dropout > gap tolerance
        strat_area = np.zeros(n)

        stage, cycle, lc_idx, ncyc, complete = classify_lifecycle(
            conv_area, conv_majoraxis, strat_area, DT, LIFECYCLE_PARAMS,
        )
        assert ncyc == 2


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
        "finalstats_filebase": "trackstats_final_",
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

        # 70 steps * 5 min = 5.83h, below the default mcs_lifecycle_thresh of
        # 8h, so lifecycle staging should NOT have been evaluated here.
        lifecycle_evaluated = ds["lifecycle_evaluated"].values[mcs_track]
        expected_evaluated = int((70 * (1.0 / 12.0)) >= 8)
        assert lifecycle_evaluated == expected_evaluated

    def test_driver_lifecycle_evaluated_when_long_enough(self, tmp_path):
        stats_path = _build_sparse_trackstats_file(tmp_path)
        config = _make_driver_config(stats_path)
        config["mcs_lifecycle_thresh"] = 4  # lower threshold so the 5.83h track qualifies

        outfile = identifymcs_radar(config)
        ds = xr.open_dataset(outfile, decode_times=False, mask_and_scale=False)
        track_duration = ds["track_duration"].values
        mcs_track = int(np.where(track_duration == 70)[0][0])

        assert ds["lifecycle_evaluated"].values[mcs_track] == 1
        lc_idx = ds["lifecycle_index"].values[mcs_track, :]
        assert lc_idx[2] != -9999  # genesis observed
        assert lc_idx[3] != -9999  # mature observed
        assert lc_idx[4] != -9999  # decay observed

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
