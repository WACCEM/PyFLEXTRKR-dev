"""
Tests for ft_utilities.match_pairs_to_files: it must return, for every file, exactly what
np.where(np.abs(basetime - file_basetime) < dt_thresh) returns (same entries, same order, same dtype).
"""
import numpy as np
import pytest

from pyflextrkr.ft_utilities import match_pairs_to_files


def reference(basetime, file_basetimes, dt_thresh):
    return [np.where(np.abs(basetime - t) < dt_thresh) for t in file_basetimes]


def assert_same(got, expected):
    assert len(got) == len(expected)
    for (gi, gj), (ei, ej) in zip(got, expected):
        np.testing.assert_array_equal(gi, ei)
        np.testing.assert_array_equal(gj, ej)
        assert gi.dtype == ei.dtype and gj.dtype == ej.dtype


def make_basetime(rng, ntracks=300, ntimes=40, step=300.0, jitter=20.0):
    """Tracks start at a random step of a 5-minute grid, last a random number of steps, NaN elsewhere."""
    basetime = np.full((ntracks, ntimes), np.nan)
    start = rng.integers(0, 200, ntracks)
    dur = rng.integers(0, ntimes + 1, ntracks)
    for k in range(ntracks):
        t = 1.7e9 + (start[k] + np.arange(dur[k])) * step + np.round(rng.uniform(-jitter, jitter, dur[k]))
        basetime[k, :dur[k]] = t
    return basetime


@pytest.mark.parametrize("dt_thresh", [1.0, 60.0, 150.0, 301.0, 1000.0])
def test_equals_np_where_on_random_data(dt_thresh):
    rng = np.random.default_rng(7)
    basetime = make_basetime(rng)
    file_basetimes = 1.7e9 + np.arange(0, 260) * 300.0 + rng.integers(-10, 10, 260)
    assert_same(match_pairs_to_files(basetime, file_basetimes, dt_thresh),
                reference(basetime, file_basetimes, dt_thresh))


def test_an_entry_can_belong_to_several_files_when_the_threshold_is_large():
    basetime = np.array([[100.0, 400.0], [250.0, np.nan]])
    files = [100.0, 250.0, 400.0]
    got = match_pairs_to_files(basetime, files, 200.0)
    assert_same(got, reference(basetime, files, 200.0))
    assert sum(len(i) for i, _ in got) > np.isfinite(basetime).sum()


def test_exactly_the_threshold_apart_is_excluded():
    basetime = np.array([[1000.0, 1060.0, 1059.0, 940.0, 941.0]])
    got = match_pairs_to_files(basetime, [1000.0], 60.0)
    assert_same(got, reference(basetime, [1000.0], 60.0))
    np.testing.assert_array_equal(got[0][1], [0, 2, 4])


def test_order_is_by_track_then_time():
    # two tracks with times that interleave in time: the order must still be track-major
    basetime = np.array([[10.0, 20.0, 30.0], [11.0, 21.0, np.nan], [9.0, np.nan, np.nan]])
    got = match_pairs_to_files(basetime, [10.0, 20.0, 30.0], 5.0)
    assert_same(got, reference(basetime, [10.0, 20.0, 30.0], 5.0))
    np.testing.assert_array_equal(got[0][0], [0, 1, 2])
    np.testing.assert_array_equal(got[0][1], [0, 0, 0])


def test_files_without_matches_and_empty_inputs():
    basetime = np.array([[100.0, 200.0]])
    got = match_pairs_to_files(basetime, [5000.0, 100.0], 60.0)
    assert_same(got, reference(basetime, [5000.0, 100.0], 60.0))
    assert len(got[0][0]) == 0 and len(got[1][0]) == 1
    all_nan = np.full((4, 3), np.nan)
    assert_same(match_pairs_to_files(all_nan, [1.0, 2.0], 60.0), reference(all_nan, [1.0, 2.0], 60.0))
    assert match_pairs_to_files(basetime, [], 60.0) == []
    none = np.zeros((0, 5))
    assert_same(match_pairs_to_files(none, [1.0], 60.0), reference(none, [1.0], 60.0))
