"""
Tests for ft_utilities.data_time_windows / match_pairs_to_windows: the windows of data records
(prior / after / nearest) must select exactly the (track, time) entries a brute-force comparison selects.
"""
import numpy as np
import pytest

from pyflextrkr.ft_utilities import data_time_windows, match_pairs_to_windows


def brute(basetime, start, end, closed):
    out = []
    for s, e in zip(start, end):
        if closed == "right":
            m = (basetime > s) & (basetime <= e)
        else:
            m = (basetime >= s) & (basetime < e)
        out.append(np.where(m))
    return out


def make_basetime(rng, ntracks=200, ntimes=50, step=300.0):
    bt = np.full((ntracks, ntimes), np.nan)
    start = rng.integers(0, 150, ntracks)
    dur = rng.integers(0, ntimes + 1, ntracks)
    for k in range(ntracks):
        bt[k, :dur[k]] = 1.7e9 + (start[k] + np.arange(dur[k])) * step + rng.integers(-20, 20)
    return bt


@pytest.mark.parametrize("mode", ["prior", "after", "nearest"])
def test_windows_equal_brute_force(mode):
    rng = np.random.default_rng(3)
    bt = make_basetime(rng)
    data_t = 1.7e9 + np.arange(0, 60) * 3600.0 / 4 + 7.0
    start, end, closed = data_time_windows(data_t, mode, 3600.0 / 2)
    got = match_pairs_to_windows(bt, start, end, closed)
    exp = brute(bt, start, end, closed)
    for (gi, gj), (ei, ej) in zip(got, exp):
        np.testing.assert_array_equal(gi, ei)
        np.testing.assert_array_equal(gj, ej)


def test_prior_includes_the_data_time_and_excludes_the_start():
    bt = np.array([[100.0, 3700.0, 3701.0, 7300.0]])
    start, end, closed = data_time_windows([3700.0], "prior", 3600.0)
    ((i, j),) = match_pairs_to_windows(bt, start, end, closed)
    np.testing.assert_array_equal(j, [1])            # (100, 3700]: 3700 yes, 100 is exactly start: excluded
    start, end, closed = data_time_windows([3700.0], "after", 3600.0)
    ((i, j),) = match_pairs_to_windows(bt, start, end, closed)
    np.testing.assert_array_equal(j, [1, 2])         # [3700, 7300): 7300 excluded


def test_bad_mode():
    with pytest.raises(ValueError):
        data_time_windows([0.0], "sideways")
