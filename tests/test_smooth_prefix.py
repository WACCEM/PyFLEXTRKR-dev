"""
regionprops_driver smooths the trajectory of every track. A track fills only the first `track_duration` of the `ntimes` columns of the
dense arrays; the rest is NaN. Smoothing only the first max(duration, 3) entries (not all ntimes) must give the same trajectory:
trailing NaN never change the outlier test, the choice between the variants or the interpolation, but a trajectory of 1 or 2
points takes other branches than a longer one, hence the 3 (smoothing exactly `duration` entries is not always the same for
duration 1 and 2). This checks the equivalence on random trajectories with jumps and gaps.
"""
import numpy as np
import pytest

from pyflextrkr.smooth_trajectory import smooth_trajectory

NTIMES = 24


def _trajectory(rng, duration):
    """
    A random walk of `duration` positions (degrees), NaN after that, with jumps (outliers) and NaN gaps inside.
    """
    lon = np.full(NTIMES, np.nan)
    lat = np.full(NTIMES, np.nan)
    lon[:duration] = -95 + np.cumsum(rng.normal(0.0, 0.08, duration))
    lat[:duration] = 35 + np.cumsum(rng.normal(0.0, 0.05, duration))
    for i in rng.choice(duration, size=min(duration, rng.integers(0, 4)), replace=False):
        lon[i] += rng.choice([-1.0, 1.0]) * rng.uniform(1, 4)           # a jump
    for i in rng.choice(duration, size=min(duration, rng.integers(0, 3)), replace=False):
        lon[i] = lat[i] = np.nan                                         # a gap
    return lon, lat


@pytest.mark.parametrize("seed", range(10))
def test_prefix_gives_the_same_trajectory_as_the_full_row(seed):
    rng = np.random.default_rng(seed)
    for duration in list(range(1, NTIMES + 1)) * 8:
        lon, lat = _trajectory(rng, duration)
        full = smooth_trajectory(lon, lat, max_speed_kmh=180.0, time_step_h=5 / 60.0)
        n = min(NTIMES, max(duration, 3))
        part = smooth_trajectory(lon[:n], lat[:n], max_speed_kmh=180.0, time_step_h=5 / 60.0)
        got_lon, got_lat = np.full(NTIMES, np.nan), np.full(NTIMES, np.nan)
        got_lon[:n], got_lat[:n] = part
        got_lon[n:], got_lat[n:] = lon[n:], lat[n:]
        assert np.array_equal(full[0], got_lon, equal_nan=True) and np.array_equal(full[1], got_lat, equal_nan=True), \
            f"duration {duration}"
