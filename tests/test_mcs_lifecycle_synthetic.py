"""
Synthetic tests for pyflextrkr/mcs_lifecycle.py, the lifecycle staging of one MCS track:

- the scenarios the staging was hardened for (multi-cycle tracks, no mature stage, line present from the start, ...)
- the time-based staging: results do not depend on the time resolution (5 min ... 1 h), irregular steps (the MRMS
  4/6-minute cadence) or missing samples; with smoothing off they equal the frozen pre-hardening classifier
- the running-median smoothing of the input series
- Monte-Carlo robustness on noisy synthetic MCS with a known lifecycle (tests/lifecycle_synthetic.py)

No dask import and no data needed, so all of this runs on CI.
"""
import numpy as np
import pytest

import lifecycle_synthetic as L
from pyflextrkr.mcs_lifecycle import (
    classify_lifecycle,
    running_median,
    smooth_inputs,
    stratiform_floor,
    STAGE_CLOUD_ONLY, STAGE_UNORGANIZED, STAGE_GENESIS, STAGE_MATURE, STAGE_DECAY,
)

DT = 1.0 / 12.0  # 5-minute time resolution [hour]

LIFECYCLE_PARAMS = dict(L.PARAMS)


# ---------------------------------------------------------------------------
# Lifecycle staging - the scenarios the task explicitly asked to harden (regular time axis, no smoothing)
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
# Time-based staging
# ---------------------------------------------------------------------------

def _random_runs(rng, n, p_on=0.5, max_run=40):
    """Boolean series made of random-length runs."""
    out = np.empty(0, dtype=bool)
    while out.size < n:
        out = np.concatenate([out, np.full(int(rng.integers(1, max_run)), rng.random() < p_on)])
    return out[:n]


class TestTimeBasedStaging:

    def test_explicit_regular_time_axis_equals_the_default(self):
        t, conv, length, strat = L.sample_case('two_cycles', 5, 'clean', np.random.default_rng(0))
        a = classify_lifecycle(conv, length, strat, DT, LIFECYCLE_PARAMS)
        b = classify_lifecycle(conv, length, strat, DT, LIFECYCLE_PARAMS, time_hours=np.arange(len(t)) * DT)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)

    @pytest.mark.parametrize("dt_hour", [1.0 / 12.0, 0.25, 0.5, 1.0])
    def test_equals_the_frozen_classifier_without_smoothing_on_a_regular_axis(self, dt_hour):
        """Smoothing off and a regular axis: the same stages, cycles, indices and flags as before the change."""
        rng = np.random.default_rng(int(dt_hour * 1000))
        for _ in range(150):
            n = int(rng.integers(1, 400 if dt_hour < 0.2 else 60))
            line = _random_runs(rng, n, max_run=40 if dt_hour < 0.2 else 6)
            conv = np.where(_random_runs(rng, n, p_on=0.8), rng.uniform(10, 100, n), 0.0)
            length = np.where(line, rng.uniform(100, 300, n), rng.uniform(0, 99, n))
            strat = np.where(_random_runs(rng, n, p_on=0.5), rng.uniform(100, 1000, n), 0.0)
            new = classify_lifecycle(conv, length, strat, dt_hour, LIFECYCLE_PARAMS)
            old = L.classify_lifecycle_v1(conv, length, strat, dt_hour, LIFECYCLE_PARAMS)
            for x, y in zip(new, old):
                np.testing.assert_array_equal(x, y)

    @pytest.mark.parametrize("resolution", [5, 15, 30, 60, 'irregular'])
    def test_stage_onsets_do_not_depend_on_the_resolution(self, resolution):
        t, conv, length, strat = L.sample_case('textbook', resolution, 'clean', np.random.default_rng(1))
        dt = L.nominal_step_hour(resolution)
        _, _, idx, ncycles, complete = classify_lifecycle(
            conv, length, strat, dt, LIFECYCLE_PARAMS, time_hours=t)
        true_onset, true_ncycles = L.truth('textbook')
        assert ncycles == true_ncycles and complete
        step = 6.0 / 60.0 if resolution == 'irregular' else dt
        np.testing.assert_allclose(t[idx[2:5]], true_onset, atol=2 * step)

    def test_missing_samples_do_not_shorten_the_lags(self):
        """A quarter of the samples missing: the 1 h genesis lag is still 1 h, not 12 remaining samples."""
        rng = np.random.default_rng(2)
        t, conv, length, strat = L.sample_case('textbook', 5, 'clean', rng)
        keep = rng.random(len(t)) >= 0.25
        keep[0] = True
        t, conv, length, strat = t[keep], conv[keep], length[keep], strat[keep]
        _, _, idx, ncycles, _ = classify_lifecycle(conv, length, strat, DT, LIFECYCLE_PARAMS, time_hours=t)
        true_onset, _ = L.truth('textbook')
        assert ncycles == 1
        np.testing.assert_allclose(t[idx[2:5]], true_onset, atol=0.25)

    def test_real_time_not_the_nominal_step_sets_the_gap(self):
        """Samples 10 min apart although the nominal step says 5 min: an 80 min dropout is not bridged."""
        n = 60
        t = np.arange(n) * (10.0 / 60.0)
        conv = np.full(n, 50.0)
        length = np.full(n, 150.0)
        length[20:28] = 0.0                       # 8 samples = 80 min > mcs_lifecycle_gap_hour (1 h)
        strat = np.zeros(n)
        assert classify_lifecycle(conv, length, strat, DT, LIFECYCLE_PARAMS, time_hours=t)[3] == 2
        length[20:28] = 150.0
        length[20:25] = 0.0                       # 5 samples = 50 min <= 1 h: bridged
        assert classify_lifecycle(conv, length, strat, DT, LIFECYCLE_PARAMS, time_hours=t)[3] == 1

    def test_gap_bridging_in_hours_at_30_min_resolution(self):
        n = 30
        conv = np.full(n, 50.0)
        strat = np.zeros(n)
        for n_missing, expected in [(1, 1), (2, 1), (3, 2)]:    # 0.5 h, 1.0 h bridged; 1.5 h splits
            length = np.full(n, 150.0)
            length[10:10 + n_missing] = 0.0
            assert classify_lifecycle(conv, length, strat, 0.5, LIFECYCLE_PARAMS)[3] == expected

    def test_lags_are_in_hours_at_every_resolution(self):
        """Line organized from hour 3 for 10 h: genesis starts 1 h later at any resolution."""
        for dt in [1.0 / 12.0, 0.25, 0.5, 1.0]:
            t = np.arange(0.0, 14.0, dt)
            conv = np.full(len(t), 50.0)
            length = np.where(t >= 3.0, 150.0, 20.0)
            strat = np.zeros(len(t))
            _, _, idx, _, _ = classify_lifecycle(conv, length, strat, dt, LIFECYCLE_PARAMS, time_hours=t)
            assert t[idx[2]] == pytest.approx(4.0, abs=1e-9)

    def test_invalid_time_axis_falls_back_to_the_regular_axis(self, caplog):
        t, conv, length, strat = L.sample_case('textbook', 5, 'clean', np.random.default_rng(3))
        reference = classify_lifecycle(conv, length, strat, DT, LIFECYCLE_PARAMS)
        bad = t.copy()
        bad[10] = bad[9]                          # not strictly increasing
        result = classify_lifecycle(conv, length, strat, DT, LIFECYCLE_PARAMS, time_hours=bad)
        for x, y in zip(reference, result):
            np.testing.assert_array_equal(x, y)
        assert "strictly increasing" in caplog.text


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

class TestRunningMedian:

    def test_spike_removed_and_step_edge_kept(self):
        t = np.arange(40) * DT
        x = np.concatenate([np.zeros(20), np.full(20, 10.0)])
        x[5] = 100.0
        y = running_median(x, t, 0.5)             # +-15 min = 7 samples
        assert y[5] == 0.0
        assert y[19] == 0.0 and y[20] == 10.0     # the step does not move
        np.testing.assert_array_equal(y[:20], 0.0)
        np.testing.assert_array_equal(y[20:], 10.0)

    def test_short_dropout_is_filled(self):
        t = np.arange(40) * DT
        x = np.full(40, 10.0)
        x[10:12] = 0.0                            # 2 samples (10 min) lost
        np.testing.assert_array_equal(running_median(x, t, 0.5), 10.0)

    def test_zero_window_is_the_identity_and_returns_a_copy(self):
        t = np.arange(10) * DT
        x = np.arange(10, dtype=float)
        y = running_median(x, t, 0.0)
        np.testing.assert_array_equal(y, x)
        y[0] = -1.0
        assert x[0] == 0.0

    def test_window_is_in_hours_not_samples(self):
        x = np.array([0.0, 0.0, 100.0, 0.0, 0.0])
        # hourly data: a 0.5 h window holds one sample, nothing is smoothed
        np.testing.assert_array_equal(running_median(x, np.arange(5) * 1.0, 0.5), x)
        # 5-min data: the same window holds 7 samples, the spike goes
        assert running_median(x, np.arange(5) * DT, 0.5)[2] == 0.0

    def test_irregular_axis_uses_time_not_index(self):
        t = np.array([0.0, 0.01, 0.02, 1.0, 1.01, 1.02])
        x = np.array([1.0, 100.0, 1.0, 5.0, 500.0, 5.0])
        np.testing.assert_array_equal(running_median(x, t, 0.1), [1, 1, 1, 5, 5, 5])

    def test_window_is_truncated_at_the_ends(self):
        t = np.arange(20) * DT
        x = np.zeros(20)
        x[0] = 100.0
        x[-1] = 100.0
        y = running_median(x, t, 0.5)
        assert y[0] == 0.0 and y[-1] == 0.0

    def test_short_series_are_returned_unchanged(self):
        np.testing.assert_array_equal(running_median(np.array([1.0, 50.0]), np.array([0.0, 0.1]), 0.5), [1.0, 50.0])
        assert len(running_median(np.array([]), np.array([]), 0.5)) == 0


class TestRunningMedianSkipna:

    def test_nan_are_left_out_of_the_median(self):
        t = np.arange(10) * DT
        x = np.full(10, 5.0)
        x[4] = np.nan
        np.testing.assert_array_equal(running_median(x, t, 0.5, skipna=True), 5.0)
        assert np.isnan(running_median(x, t, 0.5)[4])             # default: a NaN in the window gives NaN

    def test_a_window_of_only_nan_stays_nan_and_the_gap_edges_are_filled(self):
        t = np.arange(20) * DT                                    # 0.5 h = 7 samples
        x = np.full(20, 5.0)
        x[5:16] = np.nan                                          # 11 missing samples
        y = running_median(x, t, 0.5, skipna=True)
        assert np.isnan(y[10])                                    # window 7..13 holds no value
        assert y[5] == 5.0 and y[15] == 5.0                       # the window still reaches a value
        np.testing.assert_array_equal(y[:5], 5.0)

    def test_skipna_without_a_window_returns_the_input(self):
        x = np.array([1.0, np.nan, 3.0, 4.0])
        np.testing.assert_array_equal(running_median(x, np.arange(4) * DT, 0.0, skipna=True), x)


class TestStagingHelpers:
    PARAMS = dict(LIFECYCLE_PARAMS, mcs_lifecycle_smooth_hour=0.5)

    def test_smooth_inputs_replaces_nan_by_zero_then_takes_the_median(self):
        t = np.arange(40) * DT
        conv = np.full(40, 10.0)
        conv[5] = np.nan                                          # one lost sample: 0, then removed by the median
        length = np.full(40, 7.0)
        length[20] = 500.0                                        # one spike
        strat = np.zeros(40)
        conv_s, length_s, strat_s = smooth_inputs(conv, length, strat, t, self.PARAMS)
        np.testing.assert_array_equal(conv_s, 10.0)
        np.testing.assert_array_equal(length_s, 7.0)
        np.testing.assert_array_equal(strat_s, 0.0)

    def test_smooth_inputs_without_a_window_only_replaces_nan(self):
        t = np.arange(6) * DT
        conv = np.array([1.0, np.nan, 3.0, 4.0, 5.0, 6.0])
        out = smooth_inputs(conv, conv, conv, t, LIFECYCLE_PARAMS)
        for series in out:
            np.testing.assert_array_equal(series, [1.0, 0.0, 3.0, 4.0, 5.0, 6.0])

    def test_stratiform_floor_is_the_mean_of_the_positive_values(self):
        assert stratiform_floor(np.array([0.0, 0.0, 10.0, 30.0]), LIFECYCLE_PARAMS) == 20.0
        assert stratiform_floor(np.array([np.nan, 10.0, 30.0]), LIFECYCLE_PARAMS) == 20.0
        assert stratiform_floor(np.zeros(5), LIFECYCLE_PARAMS) == 0.0
        assert stratiform_floor(np.array([10.0, 30.0]), dict(LIFECYCLE_PARAMS, mcs_lifecycle_sf_area_min=25)) == 25

    @pytest.mark.parametrize("case", ['textbook', 'two_cycles'])
    def test_first_and_last_mature_sample_of_every_cycle_meet_the_helper_thresholds(self, case):
        """The plot draws the line threshold and this floor: the mature stage of each episode must start and end
        on a sample that is on a line and has a stratiform area of at least that floor."""
        t, conv, length, strat = L.sample_case(case, 5, 'moderate', np.random.default_rng(7))
        stage, cycle = classify_lifecycle(conv, length, strat, DT, self.PARAMS, time_hours=t)[:2]
        _, length_s, strat_s = smooth_inputs(conv, length, strat, t, self.PARAMS)
        floor = stratiform_floor(strat_s, self.PARAMS)
        satisfied = (length_s >= self.PARAMS['mcs_lifecycle_line_thresh']) & (strat_s > 0) & (strat_s >= floor)
        mature = stage == STAGE_MATURE
        assert mature.any()
        for c in np.unique(cycle[mature]):
            idx = np.where(mature & (cycle == c))[0]
            assert satisfied[idx[0]] and satisfied[idx[-1]]


class TestSmoothingInStaging:
    PARAMS_SMOOTH = dict(LIFECYCLE_PARAMS, mcs_lifecycle_smooth_hour=0.5)

    @staticmethod
    def _spiky_series():
        """10 h of 5-min data with a weak convective line (40 km) and three 150 km spikes 40 min apart."""
        n = 120
        conv = np.full(n, 50.0)
        length = np.full(n, 40.0)
        length[[24, 32, 40]] = 150.0              # 2.0 h, 2.67 h, 3.33 h: bridged by the 1 h gap rule
        return conv, length, np.zeros(n)

    def test_isolated_spikes_start_an_episode_without_smoothing(self):
        stage, _, _, ncycles, _ = classify_lifecycle(*self._spiky_series(), DT, LIFECYCLE_PARAMS)
        assert ncycles == 1 and np.any(stage == STAGE_GENESIS)        # the failure the smoothing is for

    def test_isolated_spikes_do_not_start_an_episode_with_smoothing(self):
        stage, _, lc_idx, ncycles, complete = classify_lifecycle(*self._spiky_series(), DT, self.PARAMS_SMOOTH)
        assert ncycles == 0 and not np.any(stage == STAGE_GENESIS)
        assert np.all(stage == STAGE_UNORGANIZED) and not complete

    def test_stray_stratiform_sample_does_not_stretch_the_mature_stage_with_smoothing(self):
        n = 150
        conv = np.full(n, 50.0)
        length = np.full(n, 150.0)               # the line exists from t = 0
        strat = np.zeros(n)
        strat[36:81] = 500.0                      # mature stage: steps 36-80
        strat[120] = 500.0                        # one stray sample 3 h after the stratiform area is gone
        plain = classify_lifecycle(conv, length, strat, DT, LIFECYCLE_PARAMS)[0]
        smooth = classify_lifecycle(conv, length, strat, DT, self.PARAMS_SMOOTH)[0]
        assert np.all(plain[36:121] == STAGE_MATURE)                  # the mature stage runs on to the stray sample
        assert np.all(smooth[36:81] == STAGE_MATURE)
        assert np.all(smooth[81:] == STAGE_DECAY)

    def test_smoothing_keeps_the_boundaries_of_a_clean_series(self):
        t, conv, length, strat = L.sample_case('textbook', 5, 'clean', np.random.default_rng(4))
        plain = classify_lifecycle(conv, length, strat, DT, LIFECYCLE_PARAMS, time_hours=t)
        smooth = classify_lifecycle(conv, length, strat, DT, self.PARAMS_SMOOTH, time_hours=t)
        assert smooth[3] == plain[3] == 1
        np.testing.assert_allclose(smooth[2][2:5], plain[2][2:5], atol=2)     # within 2 samples (10 min)

    def test_smoothing_off_by_default(self):
        args = (*self._spiky_series(), DT)
        a = classify_lifecycle(*args, LIFECYCLE_PARAMS)
        b = classify_lifecycle(*args, dict(LIFECYCLE_PARAMS, mcs_lifecycle_smooth_hour=0.0))
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)


# ---------------------------------------------------------------------------
# Monte-Carlo robustness on noisy synthetic MCS with a known lifecycle
# ---------------------------------------------------------------------------

def _new(smooth_hour):
    params = dict(LIFECYCLE_PARAMS, mcs_lifecycle_smooth_hour=smooth_hour)
    return lambda conv, length, strat, dt, p, t: classify_lifecycle(
        conv, length, strat, dt, dict(p, **params), time_hours=t)


def _old(conv, length, strat, dt, p, t):
    return L.classify_lifecycle_v1(conv, length, strat, dt, p)


CASE_NAMES = list(L.CASES)


class TestLifecycleRobustness:
    """Fixed seeds. The thresholds leave a wide margin to the values observed with NumPy 2.x streams (pass rate >= 0.97,
    invented <= 0.04, severe-noise mean 0.94 against 0.35 before the change), since the random streams of Generator
    methods are not guaranteed to be the same in every NumPy version."""
    SMOOTH_HOUR = 0.5

    @pytest.mark.parametrize("resolution", L.RESOLUTIONS)
    @pytest.mark.parametrize("case", CASE_NAMES)
    def test_clean_series_reproduce_the_truth_at_every_resolution(self, case, resolution):
        result = L.run_case(_new(self.SMOOTH_HOUR), case, resolution, 'clean', 4, seed=11)
        assert L.pass_rate(result) == 1.0
        assert not result['invented'].any() and not result['missed'].any()

    @pytest.mark.parametrize("resolution", [5, 'irregular'])
    @pytest.mark.parametrize("case", CASE_NAMES)
    def test_moderate_noise_at_5_min_resolution(self, case, resolution):
        """Onsets within max(1 h, 2 samples) and the right cycle count in >= 85 % of the 5-min realizations
        (the staging before the change reached 15-95 % in these cases)."""
        result = L.run_case(_new(self.SMOOTH_HOUR), case, resolution, 'moderate', 60, seed=21)
        assert L.pass_rate(result) >= 0.85

    @pytest.mark.parametrize("resolution", [5, 'irregular'])
    @pytest.mark.parametrize("case", CASE_NAMES)
    def test_stages_missing_in_the_truth_are_rarely_invented(self, case, resolution):
        result = L.run_case(_new(self.SMOOTH_HOUR), case, resolution, 'moderate', 60, seed=31)
        assert result['invented'].mean() <= 0.12

    @pytest.mark.parametrize("resolution", [5, 'irregular'])
    def test_smoothing_helps_under_severe_noise(self, resolution):
        smooth, plain = [], []
        for i, case in enumerate(CASE_NAMES):
            smooth.append(L.pass_rate(L.run_case(_new(self.SMOOTH_HOUR), case, resolution, 'severe', 40, seed=41 + i)))
            plain.append(L.pass_rate(L.run_case(_old, case, resolution, 'severe', 40, seed=41 + i)))
        assert np.mean(smooth) >= np.mean(plain) + 0.2
        assert np.mean(smooth) >= 0.8

    @pytest.mark.parametrize("resolution", L.RESOLUTIONS)
    def test_never_crashes_or_returns_invalid_stages(self, resolution):
        """Severe noise, spikes, dropped samples and blocks, any case: valid stage codes and indices."""
        rng = np.random.default_rng(51)
        for case in CASE_NAMES:
            for _ in range(5):
                t, conv, length, strat = L.sample_case(case, resolution, 'severe', rng)
                stage, cycle, idx, ncycles, complete = classify_lifecycle(
                    conv, length, strat, L.nominal_step_hour(resolution),
                    dict(LIFECYCLE_PARAMS, mcs_lifecycle_smooth_hour=self.SMOOTH_HOUR), time_hours=t)
                assert len(stage) == len(t) and len(cycle) == len(t)
                assert set(np.unique(stage)) <= {1, 2, 3, 4, 5}
                assert np.all((idx >= -1) & (idx < len(t)))
                assert ncycles >= 0 and isinstance(complete, bool)
