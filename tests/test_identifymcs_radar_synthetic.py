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
    identify_mcs_periods_conv_diameter,
    identify_mcs_periods_by_method,
    mcs_size_varname,
    mcs_params_from_config,
    find_mergesplit_parents,
    resolve_fragment_roots,
    release_noncoexisting_fragments,
    attach_fragments,
    classify_feature_class,
    classify_track_class,
    number_mcs_tracks,
    CLASS_MCS, CLASS_MULTICELL, CLASS_ISOLATED_CELL, CLASS_NON_CONVECTIVE,
    CLASS_SINGLE_CELL, CLASS_STRATIFORM,
)

DT = 1.0 / 12.0  # 5-minute time resolution [hour]

MCS_PARAMS = {
    'mcs_majoraxis_thresh': 100.0,
    'mcs_dbz45_area_thresh': 0,
    'mcs_duration_thresh': 4.0,
    'mcs_gap_hour': 0.25,
}

# The same criteria with the convective-cell size test
MCS_PARAMS_DIAMETER = dict(MCS_PARAMS, mcs_size_method='conv_diameter_sum', mcs_size_thresh=100.0)


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
# identify_mcs_periods_conv_diameter / identify_mcs_periods_by_method / parameters
# ---------------------------------------------------------------------------

class TestIdentifyMcsPeriodsConvDiameter:
    def _make_track(self, n, diameter_val=150.0, dbz45_val=10.0):
        return np.full(n, diameter_val), np.full(n, dbz45_val), np.arange(n) * DT * 3600.0

    def test_sustained_mcs(self):
        diameter, dbz45, basetime = self._make_track(60)
        mcs_status, is_mcs = identify_mcs_periods_conv_diameter(diameter, dbz45, basetime, DT, MCS_PARAMS_DIAMETER)
        assert is_mcs
        assert np.all(mcs_status == 1)

    def test_too_short_not_mcs(self):
        diameter, dbz45, basetime = self._make_track(20)
        _, is_mcs = identify_mcs_periods_conv_diameter(diameter, dbz45, basetime, DT, MCS_PARAMS_DIAMETER)
        assert not is_mcs

    def test_threshold_is_inclusive_and_below_is_not_mcs(self):
        diameter, dbz45, basetime = self._make_track(60, diameter_val=100.0)
        assert identify_mcs_periods_conv_diameter(diameter, dbz45, basetime, DT, MCS_PARAMS_DIAMETER)[1]
        diameter, dbz45, basetime = self._make_track(60, diameter_val=99.9)
        assert not identify_mcs_periods_conv_diameter(diameter, dbz45, basetime, DT, MCS_PARAMS_DIAMETER)[1]

    def test_no_45dbz_area_never_mcs(self):
        diameter, dbz45, basetime = self._make_track(60, dbz45_val=0.0)
        assert not identify_mcs_periods_conv_diameter(diameter, dbz45, basetime, DT, MCS_PARAMS_DIAMETER)[1]

    def test_gap_bridged_and_long_gap_breaks(self):
        diameter, dbz45, basetime = self._make_track(60)
        diameter[30:32] = 0.0           # 10 min, within mcs_gap_hour
        assert identify_mcs_periods_conv_diameter(diameter, dbz45, basetime, DT, MCS_PARAMS_DIAMETER)[1]
        diameter[30:40] = 0.0           # 50 min: two runs of 2.5 h and 1.7 h, neither reaches 4 h
        assert not identify_mcs_periods_conv_diameter(diameter, dbz45, basetime, DT, MCS_PARAMS_DIAMETER)[1]

    def test_nan_and_empty(self):
        diameter, dbz45, basetime = self._make_track(60)
        diameter[:] = np.nan
        mcs_status, is_mcs = identify_mcs_periods_conv_diameter(diameter, dbz45, basetime, DT, MCS_PARAMS_DIAMETER)
        assert not is_mcs and np.all(mcs_status == 0)
        mcs_status, is_mcs = identify_mcs_periods_conv_diameter(np.array([]), np.array([]), np.array([]), DT, MCS_PARAMS_DIAMETER)
        assert len(mcs_status) == 0 and not is_mcs

    def test_same_logic_as_the_major_axis_test(self):
        """Both tests share the run / gap / duration logic: the same series gives the same periods."""
        rng = np.random.default_rng(3)
        series = np.where(rng.random(120) < 0.8, 150.0, 20.0)
        dbz45 = np.full(120, 5.0)
        basetime = np.arange(120) * DT * 3600.0
        a = identify_mcs_periods(series, dbz45, basetime, DT, MCS_PARAMS)
        b = identify_mcs_periods_conv_diameter(series, dbz45, basetime, DT, MCS_PARAMS_DIAMETER)
        np.testing.assert_array_equal(a[0], b[0])
        assert a[1] == b[1]


class TestMcsParamsOfEarlierAndCurrentConfigs:
    def test_current_config_gets_the_current_definitions(self):
        p = mcs_params_from_config({})
        assert p['mcs_dbz45_area_varname'] == 'conv_area_45dbz'
        assert (p['mcs_lifecycle_thresh'], p['mcs_lifecycle_classes']) == (0, [1])
        assert (p['mcs_lifecycle_genesis_lag_hour'], p['mcs_lifecycle_mature_lag_hour']) == (0.0, 0.0)
        # the single-pass lifecycle with the diameter-sum line and the median stratiform floor
        assert (p['mcs_lifecycle_method'], p['mcs_lifecycle_line_method'], p['mcs_lifecycle_sf_stat']) == ('single', 'conv_diameter_sum', 'median')

    def test_earlier_config_keeps_the_earlier_definitions_for_the_keys_it_does_not_set(self):
        p = mcs_params_from_config({'mcs_majoraxis_thresh': 100})
        assert p['mcs_dbz45_area_varname'] == 'area_45dbz'
        assert (p['mcs_lifecycle_thresh'], p['mcs_lifecycle_classes']) == (8, None)
        assert (p['mcs_lifecycle_genesis_lag_hour'], p['mcs_lifecycle_mature_lag_hour']) == (1.0, 2.0)
        # the earlier lifecycle: every line episode with its lags, the major axis, the mean floor
        assert (p['mcs_lifecycle_method'], p['mcs_lifecycle_line_method'], p['mcs_lifecycle_sf_stat']) == ('episodes', 'conv_majoraxis', 'mean')

    def test_explicit_keys_win_in_both(self):
        for config in ({}, {'mcs_majoraxis_thresh': 100}):
            p = mcs_params_from_config(dict(config, mcs_dbz45_area_varname='conv_area_50dbz', mcs_lifecycle_thresh=3, mcs_lifecycle_classes=[1, 2],
                                            mcs_lifecycle_genesis_lag_hour=0.5, mcs_lifecycle_mature_lag_hour=1.0, mcs_lifecycle_sf_stat='median',
                                            mcs_lifecycle_method='episodes', mcs_lifecycle_line_method='conv_diameter_sum'))
            assert (p['mcs_dbz45_area_varname'], p['mcs_lifecycle_thresh'], p['mcs_lifecycle_classes']) == ('conv_area_50dbz', 3, [1, 2])
            assert (p['mcs_lifecycle_genesis_lag_hour'], p['mcs_lifecycle_mature_lag_hour'], p['mcs_lifecycle_sf_stat']) == (0.5, 1.0, 'median')
            assert (p['mcs_lifecycle_method'], p['mcs_lifecycle_line_method']) == ('episodes', 'conv_diameter_sum')
        assert mcs_params_from_config({'mcs_lifecycle_classes': 1})['mcs_lifecycle_classes'] == [1]        # a single value is a list of one
        # the single-pass rules on an earlier config, set explicitly
        assert mcs_params_from_config({'mcs_majoraxis_thresh': 100, 'mcs_lifecycle_method': 'single'})['mcs_lifecycle_method'] == 'single'

    def test_the_minimum_time_to_mature_of_the_upscale_flag(self):
        assert mcs_params_from_config({})['mcs_upscale_min_mature_hour'] == 1.0
        assert mcs_params_from_config({'mcs_majoraxis_thresh': 100})['mcs_upscale_min_mature_hour'] == 1.0     # an earlier config too
        assert mcs_params_from_config({'mcs_upscale_min_mature_hour': 2})['mcs_upscale_min_mature_hour'] == 2.0
        assert mcs_params_from_config({'mcs_upscale_min_mature_hour': 0})['mcs_upscale_min_mature_hour'] == 0.0
        for bad in (-1, 'one hour', True, None):
            with pytest.raises(ValueError, match="mcs_upscale_min_mature_hour"):
                mcs_params_from_config({'mcs_upscale_min_mature_hour': bad})

    def test_unknown_lifecycle_choices_stop_with_a_message(self):
        with pytest.raises(ValueError, match="mcs_lifecycle_method 'cycles'"):
            mcs_params_from_config({'mcs_lifecycle_method': 'cycles'})
        with pytest.raises(ValueError, match="mcs_lifecycle_line_method 'majoraxis'"):
            mcs_params_from_config({'mcs_lifecycle_line_method': 'majoraxis'})
        with pytest.raises(ValueError, match="mcs_lifecycle_sf_stat 'max'"):
            mcs_params_from_config({'mcs_lifecycle_sf_stat': 'max'})


class TestNumberMcsTracks:
    def test_the_mcs_tracks_are_numbered_in_the_order_of_the_file_and_the_others_are_zero(self):
        n = number_mcs_tracks([0, 1, 0, 1, 1, 0])
        assert n.tolist() == [0, 1, 0, 2, 3, 0] and n.dtype == np.int32
        assert number_mcs_tracks(np.array([True, True, False])).tolist() == [1, 2, 0]
        assert number_mcs_tracks([0, 0]).tolist() == [0, 0] and number_mcs_tracks([]).tolist() == []


class TestMcsSizeMethod:
    def test_no_size_key_gives_the_diameter_sum(self):
        """The default definition: the sum of the circular-equivalent diameters of the convective cells, 100 km."""
        params = mcs_params_from_config({})
        assert params['mcs_size_method'] == 'conv_diameter_sum'
        assert params['mcs_size_thresh'] == 100
        assert mcs_size_varname(params) == 'conv_diameter_sum'

    def test_legacy_key_without_method_keeps_the_major_axis(self):
        """A config from before mcs_size_method existed (mcs_majoraxis_thresh only): the major axis, as it always gave."""
        params = mcs_params_from_config({'mcs_majoraxis_thresh': 80})
        assert params['mcs_size_method'] == 'majoraxis'
        assert params['mcs_size_thresh'] == 80
        assert params['mcs_majoraxis_thresh'] == 80
        assert mcs_size_varname(params) == 'majoraxis'
        # an explicit method wins over the legacy key
        assert mcs_params_from_config({'mcs_size_method': 'conv_diameter_sum', 'mcs_majoraxis_thresh': 80})['mcs_size_method'] == 'conv_diameter_sum'

    def test_the_first_names_of_the_keys_stop_with_a_message(self):
        """mcs_length_method / mcs_length_thresh were renamed: a config that still has them must not be ignored silently."""
        with pytest.raises(ValueError, match="mcs_length_method.*mcs_size_method"):
            mcs_params_from_config({'mcs_length_method': 'majoraxis'})
        with pytest.raises(ValueError, match="mcs_length_thresh.*mcs_size_thresh"):
            mcs_params_from_config({'mcs_length_thresh': 100})

    def test_log_line_says_which_test_runs_and_why(self):
        from pyflextrkr.identifymcs_radar import mcs_size_test_text
        text = mcs_size_test_text({}, mcs_params_from_config({}))
        assert 'conv_diameter_sum >= 100 km' in text and 'the default' in text and '45 dBZ area conv_area_45dbz > 0' in text
        legacy = {'mcs_majoraxis_thresh': 100}
        assert 'major axis' in mcs_size_test_text(legacy, mcs_params_from_config(legacy))
        explicit = {'mcs_size_method': 'majoraxis', 'mcs_size_thresh': 90}
        assert mcs_size_test_text(explicit, mcs_params_from_config(explicit)) == 'MCS size test: majoraxis >= 90 km, 45 dBZ area conv_area_45dbz > 0'

    def test_conv_diameter_sum_config(self):
        params = mcs_params_from_config({'mcs_size_method': 'conv_diameter_sum', 'mcs_size_thresh': 120})
        assert params['mcs_size_thresh'] == 120
        assert mcs_size_varname(params) == 'conv_diameter_sum'
        # the old key is not the threshold of the new method
        assert mcs_params_from_config({'mcs_size_method': 'conv_diameter_sum', 'mcs_majoraxis_thresh': 50})['mcs_size_thresh'] == 50

    def test_mcs_size_thresh_wins_for_majoraxis_too(self):
        params = mcs_params_from_config({'mcs_size_thresh': 90, 'mcs_majoraxis_thresh': 100})
        assert params['mcs_size_thresh'] == params['mcs_majoraxis_thresh'] == 90

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            mcs_params_from_config({'mcs_size_method': 'diameter'})
        with pytest.raises(ValueError):
            mcs_size_varname({'mcs_size_method': 'diameter'})

    def test_dispatch(self):
        """A stratiform shield with small cores: 150 km major axis, 40 km of cell diameters."""
        n = 60
        length_major, length_diam = np.full(n, 150.0), np.full(n, 40.0)
        dbz45, basetime = np.full(n, 5.0), np.arange(n) * DT * 3600.0
        major = mcs_params_from_config({'mcs_size_method': 'majoraxis'})
        diam = mcs_params_from_config({'mcs_size_method': 'conv_diameter_sum', 'mcs_size_thresh': 100})
        assert identify_mcs_periods_by_method(length_major, dbz45, basetime, DT, major)[1]
        assert not identify_mcs_periods_by_method(length_diam, dbz45, basetime, DT, diam)[1]
        assert identify_mcs_periods_by_method(length_diam, dbz45, basetime, DT, dict(diam, mcs_size_thresh=30))[1]
        # params dicts written before mcs_size_method existed (like MCS_PARAMS) keep working
        assert identify_mcs_periods_by_method(length_major, dbz45, basetime, DT, MCS_PARAMS)[1]
        # the major-axis method honours mcs_size_thresh
        assert not identify_mcs_periods_by_method(length_major, dbz45, basetime, DT, dict(major, mcs_size_thresh=200))[1]


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

    def test_names_of_the_classes(self):
        """Stratiform / single cell / multi cell are the former non-convective / isolated cell / multi-cell (same codes)."""
        assert (CLASS_MCS, CLASS_MULTICELL, CLASS_SINGLE_CELL, CLASS_STRATIFORM) == (1, 2, 3, 4)
        assert CLASS_ISOLATED_CELL == CLASS_SINGLE_CELL and CLASS_NON_CONVECTIVE == CLASS_STRATIFORM

    def test_classes_are_exclusive_and_cover_every_step(self):
        """MCS first, then stratiform (0 cells), single cell (1 cell), multi cell (2+): one class for every ncells."""
        ncells = np.arange(0, 12)
        for mcs_flag in (0, 1):
            fc = classify_feature_class(np.full(len(ncells), mcs_flag), ncells)
            assert np.all(np.isin(fc, [CLASS_MCS, CLASS_MULTICELL, CLASS_SINGLE_CELL, CLASS_STRATIFORM]))
            if mcs_flag:
                assert np.all(fc == CLASS_MCS)
            else:
                np.testing.assert_array_equal(fc, [CLASS_STRATIFORM, CLASS_SINGLE_CELL] + [CLASS_MULTICELL] * 10)

    def test_track_class_non_mcs_by_the_largest_cell_count(self):
        """A non-MCS track is stratiform with no cell at any step, single cell with at most one, multi cell with two or more."""
        zeros = np.zeros(6, dtype=int)
        no_mcs = np.zeros(6, dtype=int)
        stratiform = classify_feature_class(no_mcs, zeros)
        single = classify_feature_class(no_mcs, np.array([0, 0, 1, 1, 0, 0]))
        multi = classify_feature_class(no_mcs, np.array([0, 1, 1, 2, 1, 0]))
        assert classify_track_class(stratiform) == CLASS_STRATIFORM
        assert classify_track_class(single) == CLASS_SINGLE_CELL
        assert classify_track_class(multi) == CLASS_MULTICELL

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

def _build_sparse_trackstats_file(tmp_path, conv_area_45dbz=0.0):
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
    conv_area_45dbz_2d = np.zeros((ntracks, max_trackduration))
    base_time = np.zeros((ntracks, max_trackduration))
    ncells = np.zeros((ntracks, max_trackduration))
    conv_area = np.zeros((ntracks, max_trackduration))
    conv_diameter_sum = np.zeros((ntracks, max_trackduration))
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
    conv_diameter_sum[1, 0:d1] = 11.3
    area[1, 0:d1] = 80.0
    meanlat[1, 0:d1] = 36.0
    meanlon[1, 0:d1] = -99.0

    # Track 2: long MCS track with a full embedded lifecycle
    d2 = track_duration[2]
    base_time[2, 0:d2] = t0_epoch + np.arange(d2) * dt_sec
    majoraxis[2, 0:d2] = 150.0
    area_45dbz[2, 0:d2] = 50.0
    conv_area_45dbz_2d[2, 0:d2] = conv_area_45dbz       # the 45 dBZ area of the convective pixels (0: only stratiform echo above 45 dBZ)
    conv_area[2, 0:d2] = 50.0
    ncells[2, 0:d2] = 3
    conv_diameter_sum[2, 0:d2] = 40.0    # three small cells in a 150 km feature: long, but not 100 km of cells
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
        "conv_area_45dbz": _flatten(conv_area_45dbz_2d),
        "base_time": _flatten(base_time),
        "ncells": _flatten(ncells).astype(np.int32),
        "conv_area": _flatten(conv_area),
        "conv_diameter_sum": _flatten(conv_diameter_sum),
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

        # the MCS number: 1 for the only MCS track, 0 (the _FillValue) for the others
        assert ds["mcs_number"].values[mcs_track] == 1 and ds["mcs_number"].values[isolated_track] == 0
        assert ds["mcs_number"].dims == ("tracks",) and ds["mcs_number"].dtype == np.int32
        assert ds["mcs_number"].attrs["_FillValue"] == 0 and "MCS" in ds["mcs_number"].attrs["long_name"]

    def test_mcs_number_counts_the_kept_mcs_tracks_after_the_filter(self, tmp_path):
        # the track dropped by the safety-valve filter does not take a number
        stats_path = _build_sparse_trackstats_file(tmp_path)
        config = _make_driver_config(stats_path)
        config["finalstats_min_area"] = 1000
        ds = xr.open_dataset(identifymcs_radar(config), decode_times=False, mask_and_scale=False)
        assert ds["is_mcs"].values.tolist() == [1] and ds["mcs_number"].values.tolist() == [1]

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


class TestIdentifymcsRadarConvDiameter:
    """The driver with mcs_size_method conv_diameter_sum (the sparse file's long track has 150 km major axis, 40 km of cells)."""

    def test_a_config_with_the_legacy_key_keeps_the_major_axis_test(self, tmp_path):
        stats_path = _build_sparse_trackstats_file(tmp_path)
        config = _make_driver_config(stats_path)
        ds = xr.open_dataset(identifymcs_radar(config), decode_times=False, mask_and_scale=False)
        mcs = int(np.where(ds["track_duration"].values == 70)[0][0])
        assert ds["is_mcs"].values[mcs] == 1
        assert ds.attrs["mcs_size_method"] == "majoraxis"
        assert ds.attrs["mcs_size_thresh_km"] == 100
        assert ds.attrs["mcs_majoraxis_thresh_km"] == 100

    def test_a_config_without_a_size_key_runs_the_diameter_sum_test(self, tmp_path):
        """The default definition: no mcs_size_method and no legacy key, the same file as the small-cells test."""
        stats_path = _build_sparse_trackstats_file(tmp_path)
        config = _make_driver_config(stats_path)
        del config["mcs_majoraxis_thresh"]
        ds = xr.open_dataset(identifymcs_radar(config), decode_times=False, mask_and_scale=False)
        assert ds.attrs["mcs_size_method"] == "conv_diameter_sum"
        assert ds.attrs["mcs_size_thresh_km"] == 100
        assert np.all(ds["is_mcs"].values == 0)                 # 150 km major axis but 40 km of cell diameters

    def test_the_45_dbz_area_of_the_convective_pixels_is_the_default(self, tmp_path):
        """No mcs_dbz45_area_varname: conv_area_45dbz. The long feature has 45 dBZ echo, but none in its convective pixels: not an MCS."""
        stats_path = _build_sparse_trackstats_file(tmp_path)
        config = _make_driver_config(stats_path)
        config.update(mcs_size_method="majoraxis")                          # not an earlier config: the current defaults
        config.pop("mcs_dbz45_area_varname")                                 # the fixture names it: the default is what is tested
        ds = xr.open_dataset(identifymcs_radar(config), decode_times=False, mask_and_scale=False)
        assert np.all(ds["is_mcs"].values == 0) and ds.attrs["mcs_dbz45_area_varname"] == "conv_area_45dbz"
        # the whole-feature area is still there when a config asks for it
        config.update(mcs_dbz45_area_varname="area_45dbz")
        ds = xr.open_dataset(identifymcs_radar(config), decode_times=False, mask_and_scale=False)
        assert int(ds["is_mcs"].values.sum()) == 1 and ds.attrs["mcs_dbz45_area_varname"] == "area_45dbz"

    def test_convective_pixels_above_45_dbz_make_the_mcs(self, tmp_path):
        stats_path = _build_sparse_trackstats_file(tmp_path, conv_area_45dbz=20.0)
        config = _make_driver_config(stats_path)
        config.update(mcs_size_method="majoraxis")
        config.pop("mcs_dbz45_area_varname")
        ds = xr.open_dataset(identifymcs_radar(config), decode_times=False, mask_and_scale=False)
        assert int(ds["is_mcs"].values.sum()) == 1

    def test_small_cells_in_a_long_feature_are_not_mcs(self, tmp_path):
        stats_path = _build_sparse_trackstats_file(tmp_path)
        config = _make_driver_config(stats_path)
        config.update(mcs_size_method="conv_diameter_sum", mcs_size_thresh=100)
        ds = xr.open_dataset(identifymcs_radar(config), decode_times=False, mask_and_scale=False)
        mcs = int(np.where(ds["track_duration"].values == 70)[0][0])
        assert np.all(ds["is_mcs"].values == 0)
        # three cells at every step: multi-cell; the one-cell track: single cell
        assert ds["track_class"].values[mcs] == CLASS_MULTICELL
        assert ds["track_class"].values[int(np.where(ds["track_duration"].values == 10)[0][0])] == CLASS_SINGLE_CELL
        assert ds.attrs["mcs_size_method"] == "conv_diameter_sum"
        assert ds.attrs["mcs_size_thresh_km"] == 100
        assert "mcs_majoraxis_thresh_km" not in ds.attrs

    def test_lower_threshold_gives_the_mcs(self, tmp_path):
        stats_path = _build_sparse_trackstats_file(tmp_path)
        config = _make_driver_config(stats_path)
        config.update(mcs_size_method="conv_diameter_sum", mcs_size_thresh=30)
        ds = xr.open_dataset(identifymcs_radar(config), decode_times=False, mask_and_scale=False)
        mcs = int(np.where(ds["track_duration"].values == 70)[0][0])
        assert ds["is_mcs"].values[mcs] == 1
        assert ds["track_class"].values[mcs] == CLASS_MCS
        assert ds.sizes["tracks"] == 2      # the fragment is still folded into the MCS

    def test_missing_variable_exits_with_a_hint(self, tmp_path):
        stats_path = _build_sparse_trackstats_file(tmp_path)
        sparse_file = os.path.join(stats_path, "trackstats_sparse_20250101.0000_20250101.1200.nc")
        ds = xr.open_dataset(sparse_file).load()
        ds.close()
        ds.drop_vars("conv_diameter_sum").to_netcdf(sparse_file)
        config = _make_driver_config(stats_path)
        config["mcs_size_method"] = "conv_diameter_sum"
        with pytest.raises(SystemExit):
            identifymcs_radar(config)
        # the major-axis method does not need the variable
        config["mcs_size_method"] = "majoraxis"
        assert os.path.isfile(identifymcs_radar(config))
