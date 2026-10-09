"""
Tests of the memory-lean movement_speed.py against the dense implementation it replaced (tests/movement_speed_reference.py,
a frozen copy of the version of commit b06ae17).

The old version returned arrays over all tracks for every file pair and stacked them into [file pairs x tracks] arrays
(17 GB each for a month of 240,000 tracks and 8,900 pairs); the new one keeps only the measured tracks of each pair and
writes them straight into [tracks x times] arrays. The results must be identical:
  - offset_to_speed, the start-pair rule, the alignment to the track times and filter_interp_speed on random data,
  - the worker on synthetic pixel files, dense and sparse return,
  - movement_speed(config) end to end on synthetic pixel files and statistics.
"""
import os
import sys
import copy
import tracemalloc
import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, os.path.dirname(__file__))
import movement_speed_reference as ref
from pyflextrkr import movement_speed as new

BASETIME0 = 1746212400.0
DT = 300.0


def assert_same(a, b, err_msg=""):
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=err_msg)      # NaN == NaN


# ---------------------------------------------------------------------------
# offset_to_speed
# ---------------------------------------------------------------------------

class TestOffsetToSpeed:

    def test_2d_equals_the_old_function(self):
        rng = np.random.default_rng(1)
        x = rng.integers(-6, 7, (40, 25)).astype(float)
        y = rng.integers(-6, 7, (40, 25)).astype(float)
        x[rng.random(x.shape) < 0.3] = np.nan
        y[np.isnan(x)] = np.nan
        time_lag = rng.choice([290.0, 300.0, 310.0], 40)
        for a, b in zip(new.offset_to_speed(x, y, time_lag), ref.offset_to_speed(x, y, time_lag)):
            assert_same(a, b)

    def test_1d_entries_equal_the_2d_result(self):
        rng = np.random.default_rng(2)
        x = rng.integers(-6, 7, (30, 12)).astype(float)
        y = rng.integers(-6, 7, (30, 12)).astype(float)
        time_lag = rng.choice([290.0, 300.0, 310.0], 30)
        two = new.offset_to_speed(x, y, time_lag)
        one = new.offset_to_speed(x.ravel(), y.ravel(), np.repeat(time_lag, 12))
        for a, b in zip(one, two):
            assert_same(a, b.ravel())

    def test_no_movement_has_no_direction(self):
        _, r_dir, r_speed = new.offset_to_speed(np.array([0.0, 1.0]), np.array([0.0, 0.0]), np.array([300.0, 300.0]))
        assert np.isnan(r_dir[0])                                    # no movement: no direction
        assert r_dir[1] == pytest.approx(90.0)                       # 1 pixel to the east
        assert r_speed[0] == 0.0 and r_speed[1] == pytest.approx(1.0 / 300.0)


# ---------------------------------------------------------------------------
# the start pair of a track
# ---------------------------------------------------------------------------

class TestNearestPairIndex:

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_equals_nanargmin(self, seed):
        rng = np.random.default_rng(seed)
        pairs = BASETIME0 + np.arange(200) * DT + rng.integers(-20, 20, 200)
        # starts: on the grid, between two pairs (exactly halfway included), before and after the first and last pair
        starts = np.concatenate([pairs[rng.integers(0, 200, 300)] + rng.integers(-300, 300, 300),
                                 (pairs[:-1] + pairs[1:])[:20] / 2.0, [pairs[0] - 1e4, pairs[-1] + 1e4]])
        expected = np.array([np.nanargmin(np.abs(pairs - s)) for s in starts])
        assert_same(new.nearest_pair_index(pairs, starts), expected)

    def test_nan_start_and_no_pairs(self):
        assert_same(new.nearest_pair_index(np.array([1.0, 2.0]), np.array([np.nan, 1.1])), [-1, 0])
        assert_same(new.nearest_pair_index(np.array([]), np.array([1.0])), [-1])


# ---------------------------------------------------------------------------
# alignment of the movement of the file pairs to the track times
# ---------------------------------------------------------------------------

def random_results(rng, ntracks, ntimes, npairs, lag=1, extra=True):
    """Sparse per-pair results of tracks that start at random pairs, last a random time, with some pairs missing and a few
    entries before the start, beyond ntimes, and with huge values; plus the dense arrays the old code stacked."""
    pair_base = BASETIME0 + np.arange(npairs) * DT + rng.integers(-10, 10, npairs)
    start_pair = rng.integers(0, npairs, ntracks)
    start_time = pair_base[start_pair] + rng.integers(-30, 30, ntracks)       # nearest pair is start_pair
    start_time[rng.random(ntracks) < 0.2] += DT / 2.0                          # some halfway between two pairs
    dur = rng.integers(1, ntimes + 8, ntracks)                                 # some longer than ntimes
    time_lag = rng.choice([290.0, 300.0, 310.0], npairs)
    pixel_length = np.full(ntracks, 3.0)
    dense = {k: np.full((npairs, ntracks), np.nan) for k in ("y", "x")}
    real = [[t for t in range(ntracks) if start_pair[t] <= p < start_pair[t] + dur[t] and rng.random() < 0.7]
            for p in range(npairs)]
    has_real = np.array(sorted({t for tr in real for t in tr}), dtype=np.int64)
    results = []
    for p in range(npairs):
        tr = list(real[p])
        if extra and len(has_real):                                            # a few before the start (dropped); the old
            # code fails for a track that only has values before its start pair, so only tracks that also have real ones
            tr += [t for t in rng.choice(has_real, 2) if p < start_pair[t] and t not in tr]
        tr = np.array(sorted(set(tr)), dtype=np.int64)
        y = rng.integers(-5, 6, len(tr)).astype(float)
        x = rng.integers(-5, 6, len(tr)).astype(float)
        if extra and len(tr) and rng.random() < 0.1:
            y[0] = 1e9                                                         # huge value, filtered later
        dense["y"][p, tr] = y
        dense["x"][p, tr] = x
        results.append((tr, y, x, time_lag[p], pair_base[p], pixel_length[tr]))
    return results, dense, pair_base, start_time, time_lag, pixel_length


def old_dense_dataset(dense, pair_base, start_time, time_lag, pixel_length, ntimes, lag):
    """The old pipeline on the dense arrays: offset_to_speed, conversion, define_movement_dataset (the alignment)."""
    npairs, ntracks = dense["y"].shape
    r_mag, r_dir, r_speed = ref.offset_to_speed(dense["x"], dense["y"], time_lag)
    pl = np.tile(pixel_length, (npairs, 1))
    mag = r_mag * pl / lag
    mx = dense["x"] * pl / lag
    my = dense["y"] * pl / lag
    spd = r_speed * pl * 1000.
    stats_basetime = np.full((ntracks, ntimes), np.nan)
    stats_basetime[:, 0] = start_time
    return ref.define_movement_dataset(pair_base, lag, r_dir, mag, spd, mx, my, ntimes, ntracks, stats_basetime,
                                       np.arange(ntimes), "times", np.arange(ntracks), "tracks")


class TestAlignMovementToTracks:

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_equals_the_dense_alignment(self, seed):
        rng = np.random.default_rng(seed)
        ntracks, ntimes, npairs = 60, 14, 40
        results, dense, pair_base, start_time, time_lag, pixel_length = random_results(rng, ntracks, ntimes, npairs)
        expected = old_dense_dataset(dense, pair_base, start_time, time_lag, pixel_length, ntimes, 1)
        got = new.align_movement_to_tracks(results, start_time, ntimes, 1)
        for name in got:
            assert got[name].dtype == np.float32 and got[name].shape == (ntracks, ntimes)
            assert_same(got[name], expected[name].values, err_msg=name)
        assert np.isfinite(got["movement_speed"]).sum() > 100                  # not a test of NaN == NaN

    def test_lag_scales_the_distance_not_the_speed(self):
        rng = np.random.default_rng(5)
        results, dense, pair_base, start_time, time_lag, pixel_length = random_results(rng, 20, 10, 30, extra=False)
        expected = old_dense_dataset(dense, pair_base, start_time, time_lag, pixel_length, 10, 2)
        got = new.align_movement_to_tracks(results, start_time, 10, 2)
        for name in got:
            assert_same(got[name], expected[name].values, err_msg=name)

    def test_no_measurements(self):
        out = new.align_movement_to_tracks([(np.array([], dtype=np.int64), np.array([]), np.array([]), 300.0, BASETIME0, np.array([]))],
                                           np.array([BASETIME0, np.nan]), 5, 1)
        assert all(a.shape == (2, 5) and np.all(np.isnan(a)) for a in out.values())
        assert all(np.all(np.isnan(a)) for a in new.align_movement_to_tracks([], np.array([BASETIME0]), 5, 1).values())

    def test_memory_is_the_size_of_the_output(self):
        """A month would need [pairs x tracks] float64 arrays of 17 GB each; here 2,000 pairs x 100,000 tracks would be
        1.6 GB per array in the old code, the new one must stay near the 5 output arrays (80 MB) plus the entries."""
        rng = np.random.default_rng(6)
        npairs, ntracks, ntimes, per_pair = 2000, 100_000, 40, 150
        pair_base = BASETIME0 + np.arange(npairs) * DT
        start_time = pair_base[rng.integers(0, npairs - ntimes, ntracks)]
        results = []
        for p in range(npairs):
            tr = np.sort(rng.choice(ntracks, per_pair, replace=False))
            results.append((tr, rng.integers(-4, 5, per_pair).astype(float), rng.integers(-4, 5, per_pair).astype(float),
                            DT, pair_base[p], np.full(per_pair, 3.0)))
        tracemalloc.start()
        out = new.align_movement_to_tracks(results, start_time, ntimes, 1)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        assert peak < 400e6, f"peak {peak / 1e6:.0f} MB"
        assert np.isfinite(out["movement_speed"]).sum() > 0


# ---------------------------------------------------------------------------
# filter_interp_speed
# ---------------------------------------------------------------------------

def random_movement_dataset(rng, ntracks=200, ntimes=30):
    arrays = {}
    speed = rng.uniform(0, 80, (ntracks, ntimes)).astype(np.float32)             # above and below the 50 m/s threshold
    speed[rng.random(speed.shape) < 0.5] = np.nan
    speed[rng.random(ntracks) < 0.3] = np.nan                                     # tracks without any value
    few = rng.random(ntracks) < 0.15                                              # tracks with only 1-2 values
    for t in np.where(few)[0]:
        keep = rng.choice(ntimes, 2, replace=False)
        row = np.full(ntimes, np.nan, dtype=np.float32); row[keep] = speed[t, keep] if np.isfinite(speed[t, keep]).all() else 10.0
        speed[t] = row
    nan = np.isnan(speed)
    arrays["movement_speed"] = speed
    arrays["movement_distance"] = np.where(nan, np.nan, rng.uniform(0, 30, speed.shape)).astype(np.float32)
    arrays["movement_theta"] = np.where(nan, np.nan, rng.uniform(0, 360, speed.shape)).astype(np.float32)
    arrays["movement_distance_x"] = np.where(nan, np.nan, rng.normal(0, 5, speed.shape)).astype(np.float32)
    arrays["movement_distance_y"] = np.where(nan, np.nan, rng.normal(0, 5, speed.shape)).astype(np.float32)
    return xr.Dataset({k: (["tracks", "times"], v, {"units": "u"}) for k, v in arrays.items()},
                      coords={"tracks": np.arange(ntracks), "times": np.arange(ntimes)})


class TestFilterInterpSpeed:
    CONFIG = {"tracks_dimname": "tracks", "times_dimname": "times", "max_speed_thresh": 50}

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_equals_the_old_function(self, seed):
        ds = random_movement_dataset(np.random.default_rng(seed))
        expected = ref.filter_interp_speed(copy.deepcopy(ds), self.CONFIG)
        got = new.filter_interp_speed(copy.deepcopy(ds), self.CONFIG)
        for name in ds.data_vars:
            assert_same(got[name].values, expected[name].values, err_msg=name)
            assert got[name].attrs == expected[name].attrs
        assert not np.array_equal(got["movement_speed"].values, ds["movement_speed"].values, equal_nan=True)   # it did filter

    def test_too_few_values_become_nan(self):
        ds = random_movement_dataset(np.random.default_rng(3), ntracks=6, ntimes=8)
        ds["movement_speed"].values[:] = np.nan
        ds["movement_speed"].values[0, [1, 2]] = 10.0
        ds["movement_speed"].values[1, [1, 2, 4]] = 10.0
        out = new.filter_interp_speed(ds, self.CONFIG)
        assert np.all(np.isnan(out["movement_speed"].values[0]))


# ---------------------------------------------------------------------------
# the worker, and movement_speed(config), on synthetic pixel files
# ---------------------------------------------------------------------------

NY = NX = 80
NFILES = 12


def texture(rng, n):
    return rng.uniform(5, 60, (n, n))


def write_pixel_files(path):
    """12 files with three tracks: 1 (8x8, starts at file 2, +2 px east per step, one jump of 6 px), 2 (8x8, 1 px north per
    step, all files), 3 (3x3, too small to be measured). Pixel track number = track index + 1."""
    rng = np.random.default_rng(10)
    tex = {1: texture(rng, 8), 2: texture(rng, 8), 3: texture(rng, 3)}
    x1 = 10
    for i in range(NFILES):
        track = np.zeros((NY, NX), dtype=np.int32)
        field = np.full((NY, NX), np.nan, dtype=np.float32)
        if 2 <= i <= 9:
            if i > 2:
                x1 += 6 if i == 6 else 2
            track[30:38, x1:x1 + 8] = 1
            field[30:38, x1:x1 + 8] = tex[1]
        y2 = 50 - i
        track[y2:y2 + 8, 60:68] = 2
        field[y2:y2 + 8, 60:68] = tex[2]
        track[5:8, 5:8] = 3
        field[5:8, 5:8] = tex[3]
        t = BASETIME0 + i * DT
        stamp = f"20250502_{19 + (5 * i) // 60:02d}{(5 * i) % 60:02d}00"
        xr.Dataset(
            {"time": (["time"], np.array([t])), "dbztracknumber": (["time", "y", "x"], track[None]),
             "dbz_comp": (["time", "y", "x"], field[None])}
        ).to_netcdf(os.path.join(path, f"mcstrack_{stamp}.nc"))
    return sorted(os.listdir(path))


def stats_start_times():
    start = np.array([BASETIME0 + 2 * DT, BASETIME0, BASETIME0])
    return start


def make_config(root, name):
    out = os.path.join(root, name)
    os.makedirs(os.path.join(out, "stats"), exist_ok=True)
    config = {
        "stats_outpath": os.path.join(out, "stats") + "/", "startdate": "20250502.1900", "enddate": "20250502.2000",
        "start_basetime": BASETIME0 - 10, "end_basetime": BASETIME0 + NFILES * DT, "tracks_dimname": "tracks",
        "times_dimname": "times", "run_parallel": 0, "feature_type": "radar_feature", "pixel_radius": 3.0,
        "lag_for_speed": 1, "max_speed_thresh": 50, "mcsrobust_filebase": "robust_", "mcsfinal_filebase": "final_",
        "pixeltracking_outpath": os.path.join(root, "pixel") + "/", "pixeltracking_filebase": "mcstrack_",
        "track_number_for_speed": "dbztracknumber", "track_field_for_speed": "dbz_comp", "min_size_thresh_for_speed": 20,
    }
    base = np.full((3, NFILES), np.nan)
    start = stats_start_times()
    base[0, :8] = start[0] + np.arange(8) * DT
    base[1, :] = start[1] + np.arange(NFILES) * DT
    base[2, :] = start[2] + np.arange(NFILES) * DT
    xr.Dataset({"base_time": (["tracks", "times"], base), "track_duration": (["tracks"], np.array([8, 12, 12]))},
               coords={"tracks": np.arange(3), "times": np.arange(NFILES)}).to_netcdf(
        config["stats_outpath"] + "robust_20250502.1900_20250502.2000.nc")
    return config


@pytest.fixture(scope="module")
def pixel_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("speed")
    os.makedirs(root / "pixel")
    write_pixel_files(str(root / "pixel"))
    return str(root)


class TestWorker:

    def test_sparse_is_the_dense_result_of_the_measured_tracks(self, pixel_dir):
        config = make_config(pixel_dir, "w")
        files = sorted(os.listdir(os.path.join(pixel_dir, "pixel")))
        pair = (os.path.join(pixel_dir, "pixel", files[3]), os.path.join(pixel_dir, "pixel", files[4]))
        dense = new.movement_of_feature_fft(pair, 3, config)
        old = ref.movement_of_feature_fft(pair, 3, config)
        for a, b in zip(dense, old):
            assert_same(a, b)                                        # default return unchanged
        track, y, x, time_lag, base_time, pixel_length = new.movement_of_feature_fft(pair, 3, config, sparse=True)
        assert_same(track, [0, 1])                                   # track 3 (3x3) is not measured
        assert_same(y, dense[0][[0, 1]]) and assert_same(x, dense[1][[0, 1]])
        assert time_lag == dense[2] == DT and base_time == dense[3]
        assert np.isnan(dense[0][2]) and np.isnan(dense[1][2])
        assert (y[0], x[0]) == (0.0, 2.0)                            # track 1 moves 2 pixels in x per step
        assert (y[1], x[1]) == (-1.0, 0.0)                           # track 2 moves 1 pixel in y per step

    def test_a_track_that_is_in_one_file_only_is_not_measured(self, pixel_dir):
        config = make_config(pixel_dir, "w2")
        files = sorted(os.listdir(os.path.join(pixel_dir, "pixel")))
        pair = (os.path.join(pixel_dir, "pixel", files[1]), os.path.join(pixel_dir, "pixel", files[2]))   # track 1 starts at file 2
        track = new.movement_of_feature_fft(pair, 3, config, sparse=True)[0]
        assert_same(track, [1])


class TestWorkerSizeThreshold:

    def test_a_track_with_exactly_min_size_thresh_pixels_is_measured(self, tmp_path):
        rng = np.random.default_rng(11)
        tex = rng.uniform(5, 60, (4, 5))                                       # 20 pixels
        tex19 = rng.uniform(5, 60, (19,))
        files = []
        for i in range(2):
            track = np.zeros((NY, NX), dtype=np.int32)
            field = np.full((NY, NX), np.nan, dtype=np.float32)
            track[10:14, 10 + i:15 + i] = 1                                    # 4 x 5 = 20 pixels
            field[10:14, 10 + i:15 + i] = tex
            flat = np.zeros(NY * NX, dtype=bool); flat[3000 + i:3019 + i] = True      # 19 pixels in a row
            track[flat.reshape(NY, NX)] = 2
            field[flat.reshape(NY, NX)] = tex19
            f = str(tmp_path / f"mcstrack_20250502_19{5 * i:02d}00.nc")
            xr.Dataset({"time": (["time"], np.array([BASETIME0 + i * DT])), "dbztracknumber": (["time", "y", "x"], track[None]),
                        "dbz_comp": (["time", "y", "x"], field[None])}).to_netcdf(f)
            files.append(f)
        config = {"track_number_for_speed": "dbztracknumber", "track_field_for_speed": "dbz_comp",
                  "min_size_thresh_for_speed": 20, "pixel_radius": 3.0}
        measured = new.movement_of_feature_fft(tuple(files), 2, config, sparse=True)[0]
        assert_same(measured, [0])
        dense = new.movement_of_feature_fft(tuple(files), 2, config)
        assert np.isfinite(dense[1][0]) and np.isnan(dense[1][1])


class TestMovementSpeedEndToEnd:

    def test_final_file_equals_the_old_implementation(self, pixel_dir):
        c_old, c_new = make_config(pixel_dir, "old"), make_config(pixel_dir, "new")
        old_file = ref.movement_speed(c_old)
        new_file = new.movement_speed(c_new)
        a = xr.open_dataset(old_file, decode_times=False, mask_and_scale=False)
        b = xr.open_dataset(new_file, decode_times=False, mask_and_scale=False)
        assert set(a.data_vars) == set(b.data_vars)
        assert b.attrs["max_speed_thresh"] == a.attrs["max_speed_thresh"] == 50
        assert set(b.attrs) == set(a.attrs) and set(b.coords) == set(a.coords)
        for name in a.data_vars:
            assert_same(b[name].values, a[name].values, err_msg=name)
            assert b[name].dtype == a[name].dtype and b[name].dims == a[name].dims
        speed, theta, dist = b["movement_speed"].values, b["movement_theta"].values, b["movement_distance"].values
        # track 1: 2 pixels (6 km) per 300 s = 20 m/s towards the east, for the 7 pairs of its 8 files; the 6-pixel jump
        # (60 m/s, above max_speed_thresh) is interpolated back to 20 m/s
        np.testing.assert_allclose(speed[0, :7], 20.0, rtol=1e-5)
        np.testing.assert_allclose(theta[0, :7], 90.0, rtol=1e-5)
        np.testing.assert_allclose(dist[0, :7], 6.0, rtol=1e-5)
        assert np.all(np.isnan(speed[0, 7:]))
        # track 2: 1 pixel (3 km) per step = 10 m/s, 11 pairs
        np.testing.assert_allclose(speed[1, :11], 10.0, rtol=1e-5)
        assert np.isnan(speed[1, 11])
        assert np.all(np.isnan(speed[2]))                                                  # track 3 (3x3 pixels) is too small
