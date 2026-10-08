"""
Tests for the second pass of the radar_feature MCS statistics (regionprops_func.py, regionprops_driver.py):
the shape and extreme statistics of every track recomputed on its merged mask (own feature + folded merge/split
fragments), the convective length of groups of nearby convective cells (conv_feature_number), and the lifecycle staging.

Synthetic cloudid files are written with netcdf_io.write_radar_featureid, the writer of idfeature_radar3d, and the
expected values are derived from the pixel coordinates with numpy, so this needs no data and runs on CI.

Geometry (60 x 80 grid, 3 km):
  feature 1 (track A): rows 10-14, cols 5-24, with convective blocks C1 (cols 5-9, 50 dBZ) and C2 (cols 15-19, 46 dBZ)
  feature 2 (fragment merged into track A): rows 10-14, cols 30-39, 47 dBZ, with convective block C3 (cols 30-34, 52 dBZ)
  feature 3 (track B): rows 25-29, cols 5-14, with convective block C4 (cols 5-9, 48 dBZ)
Nearest convective pixels are 6 px (18 km) apart for C1-C2, 11 px (33 km) for C2-C3 and for A-B.
"""
import os
import logging
import numpy as np
import pytest
import xarray as xr
from netCDF4 import stringtochar

from pyflextrkr.netcdf_io import write_radar_featureid
from pyflextrkr.steiner_func import dilate_label_conv_mask
from pyflextrkr.trackstats_func import calc_stats_singlefile
from pyflextrkr.regionprops_func import (
    regionprops_singlefile, recomputed_variable_names, conv_dilate_overwrite_km, redilate_conv_feature_number,
)
from pyflextrkr.regionprops_driver import compute_regionprops

NY, NX = 60, 80
DX = 3000.0
BASETIME0 = 1746212400.0                 # 2025-05-02 19:00:00 UTC
DBZ_THRESHS = [45, 50]


def _rect(rows, cols):
    return np.ix_(range(rows[0], rows[1] + 1), range(cols[0], cols[1] + 1))


def build_arrays(radius_km=0.0):
    """Arrays of the synthetic cloudid file."""
    a = {name: np.zeros((NY, NX), dtype=np.int32) for name in ("feature_number", "convsf", "cell_number")}
    a["dbz"] = np.full((NY, NX), np.nan, dtype=np.float32)
    for t in (10, 20, 30, 40, 50):
        a[f"echotop{t}"] = np.full((NY, NX), np.nan, dtype=np.float32)

    def feature(number, rows, cols, dbz):
        a["feature_number"][_rect(rows, cols)] = number
        a["convsf"][_rect(rows, cols)] = 3
        a["dbz"][_rect(rows, cols)] = dbz

    def convective(rows, cols, dbz, cell, tops):
        a["convsf"][_rect(rows, cols)] = 4
        a["dbz"][_rect(rows, cols)] = dbz
        a["cell_number"][_rect(rows, cols)] = cell
        for t, h in tops.items():
            a[f"echotop{t}"][_rect(rows, cols)] = h

    feature(1, (10, 14), (5, 24), 20.0)
    convective((10, 14), (5, 9), 50.0, 1, {10: 10000., 20: 8000., 30: 6000., 40: 4000., 50: 2000.})
    convective((10, 14), (15, 19), 46.0, 2, {10: 9000., 20: 7000., 30: 5000., 40: 3000.})
    feature(2, (10, 14), (30, 39), 47.0)
    convective((10, 14), (30, 34), 52.0, 3, {10: 12000., 20: 9000., 30: 7000., 40: 6000., 50: 5000.})
    feature(3, (25, 29), (5, 14), 20.0)
    convective((25, 29), (5, 9), 48.0, 4, {10: 7000., 20: 6000., 30: 5000., 40: 3000.})
    a["conv_feature_number"] = dilate_label_conv_mask(a["convsf"] == 4, radius_km * 1000.0, DX, DX)
    return a


def write_cloudid(path, a, radius_km=0.0, with_conv_feature_number=True, basetime=BASETIME0):
    nfeatures = int(a["feature_number"].max())
    x = np.arange(NX, dtype=np.float64) * DX
    y = np.arange(NY, dtype=np.float64) * DX
    lon2d, lat2d = np.meshgrid(-100.0 + x / 111000.0, 35.0 + y / 111000.0)
    dbz = xr.DataArray(a["dbz"], dims=['lat', 'lon'])
    stamp = path.rsplit("cloudid_", 1)[1][:-3]
    kwargs = {"conv_feature_number": a["conv_feature_number"]} if with_conv_feature_number else {}
    write_radar_featureid(
        path, np.array([basetime]), stamp[:8], stamp[9:15],
        xr.DataArray(lon2d, dims=['lat', 'lon']), xr.DataArray(lat2d, dims=['lat', 'lon']), x, y,
        dbz, dbz.copy(), a["convsf"], (a["convsf"] == 4).astype(np.int32), a["cell_number"],
        (a["feature_number"] > 0).astype(np.int32),
        a["echotop10"], a["echotop20"], a["echotop30"], a["echotop40"], a["echotop50"],
        a["feature_number"], np.bincount(a["feature_number"].ravel())[1:nfeatures + 1].astype(np.int32), nfeatures,
        "synthetic.nc",
        {'feature_varname': 'feature_number', 'nfeature_varname': 'nfeatures', 'featuresize_varname': 'npix_feature',
         'conv_dilate_radius_km': radius_km},
        feature_params={'conv_dilate_radius_km': radius_km},
        **kwargs,
    )
    return path


def worker_config(**extra):
    config = {
        'pixel_radius': DX / 1000.0, 'area_method': 'fixed', 'feature_varname': 'feature_number',
        'ref_varname': 'dbz_comp', 'pbc_direction': 'none', 'trackstats_dbz_area_threshs': DBZ_THRESHS,
    }
    config.update(extra)
    return config


def major_axis_km(rows, cols):
    """4 x sqrt(largest eigenvalue of the coordinate covariance), the major axis of the pixel set [km]."""
    cov = np.cov(np.vstack([rows, cols]), bias=True)
    return 4.0 * np.sqrt(np.linalg.eigvalsh(cov)[-1]) * DX / 1000.0


def pixels(a, *numbers):
    mask = np.isin(a["feature_number"], numbers)
    return np.nonzero(mask)


@pytest.fixture(scope="module")
def cloudid_r0(tmp_path_factory):
    d = tmp_path_factory.mktemp("rp_r0")
    a = build_arrays(0.0)
    return write_cloudid(str(d / "cloudid_20250502_190000.nc"), a, 0.0), a


def run_worker(cloudid, a, tracks, config=None):
    """tracks: list of (own cloudnumber, [merge fragments], [split fragments])."""
    n = len(tracks)
    nmax = max([len(t[1]) for t in tracks] + [len(t[2]) for t in tracks] + [1])
    own = np.array([t[0] for t in tracks])
    merge = np.full((n, nmax), -9999)
    split = np.full((n, nmax), -9999)
    for i, t in enumerate(tracks):
        merge[i, :len(t[1])] = t[1]
        split[i, :len(t[2])] = t[2]
    return regionprops_singlefile(cloudid, own, merge, split, config or worker_config())


class TestWorkerAgainstTrackstats:

    def test_tracks_without_fragments_reproduce_the_trackstats_values(self, cloudid_r0):
        """Same definitions as trackstats_func: every recomputed variable equals its trackstats value."""
        cloudid, a = cloudid_r0
        out, diag = run_worker(cloudid, a, [(1, [], []), (2, [], []), (3, [], [])])
        assert diag == {"n_conflict": 0, "n_out_of_range": 0}
        config = worker_config(tracking_outpath=os.path.dirname(cloudid) + "/", feature_type='radar_feature',
                               terrain_file=None, rangemask_varname=None)
        ts, _ = calc_stats_singlefile(
            np.array([1, 2, 3], dtype=np.int32), stringtochar(np.array([os.path.basename(cloudid)], dtype='S'))[0],
            np.ones(3, dtype=np.int32), np.zeros(3, dtype=np.int32), np.zeros(3, dtype=np.int32),
            np.zeros(3, dtype=np.int32), config,
        )
        for name in recomputed_variable_names(DBZ_THRESHS):
            np.testing.assert_allclose(out[name], ts[name], rtol=1e-5, equal_nan=True, err_msg=name)
        np.testing.assert_allclose(out["merged_area"], ts["area"], rtol=1e-6)


class TestWorkerMergedMask:

    def test_area_and_position_of_the_merged_mask(self, cloudid_r0):
        cloudid, a = cloudid_r0
        out, _ = run_worker(cloudid, a, [(1, [2], []), (3, [], [])])
        rows, cols = pixels(a, 1, 2)
        assert out["merged_area"][0] == pytest.approx(rows.size * 9.0)             # 150 px
        assert out["merged_area"][1] == pytest.approx(50 * 9.0)
        lon = -100.0 + cols * DX / 111000.0
        lat = 35.0 + rows * DX / 111000.0
        assert out["meanlon"][0] == pytest.approx(lon.mean(), abs=1e-4)
        assert out["meanlat"][0] == pytest.approx(lat.mean(), abs=1e-4)
        assert out["mean_x"][0] == pytest.approx((cols * DX / 1000.).mean(), rel=1e-5)
        assert out["mean_y"][0] == pytest.approx((rows * DX / 1000.).mean(), rel=1e-5)

    def test_shape_of_the_merged_mask(self, cloudid_r0):
        cloudid, a = cloudid_r0
        out, _ = run_worker(cloudid, a, [(1, [2], [])])
        rows, cols = pixels(a, 1, 2)
        cov = np.cov(np.vstack([rows, cols]), bias=True)
        lam = np.linalg.eigvalsh(cov)
        assert out["majoraxis"][0] == pytest.approx(4 * np.sqrt(lam[-1]) * 3.0, rel=1e-4)
        assert out["minoraxis"][0] == pytest.approx(4 * np.sqrt(lam[0]) * 3.0, rel=1e-4)
        assert out["aspectratio"][0] == pytest.approx(out["majoraxis"][0] / out["minoraxis"][0], rel=1e-5)
        assert abs(abs(out["orientation"][0]) - 90.0) < 1.0          # long axis along the columns
        # the merged mask is much longer than the own feature
        own, _ = run_worker(cloudid, a, [(1, [], [])])
        assert out["majoraxis"][0] > 1.5 * own["majoraxis"][0]

    def test_extremes_include_the_fragment(self, cloudid_r0):
        cloudid, a = cloudid_r0
        merged, _ = run_worker(cloudid, a, [(1, [2], [])])
        own, _ = run_worker(cloudid, a, [(1, [], [])])
        assert own["max_dbz"][0] == pytest.approx(50.0) and merged["max_dbz"][0] == pytest.approx(52.0)
        assert own["maxETH_10dbz"][0] == pytest.approx(10.0) and merged["maxETH_10dbz"][0] == pytest.approx(12.0)
        assert own["maxETH_50dbz"][0] == pytest.approx(2.0) and merged["maxETH_50dbz"][0] == pytest.approx(5.0)
        # > 45 dBZ: C1 (25 px) + C2 (25 px) of feature 1; with the fragment all 50 px of feature 2 as well
        assert own["area_45dbz"][0] == pytest.approx(50 * 9.0)
        assert merged["area_45dbz"][0] == pytest.approx(100 * 9.0)
        assert merged["conv_area_45dbz"][0] == pytest.approx(75 * 9.0)            # C1 + C2 + C3
        assert own["area_50dbz"][0] == 0.0 and merged["area_50dbz"][0] == pytest.approx(25 * 9.0)   # only C3 > 50 dBZ
        assert merged["conv_area_50dbz"][0] == pytest.approx(25 * 9.0)

    def test_weighted_centroid_moves_toward_the_strong_fragment(self, cloudid_r0):
        cloudid, a = cloudid_r0
        merged, _ = run_worker(cloudid, a, [(1, [2], [])])
        # the reflectivity weights (47-52 dBZ in the fragment, 20 dBZ in the stratiform part of feature 1)
        # put the weighted centroid east of the plain mean
        assert merged["x_weightedcentroid"][0] > merged["mean_x"][0]
        assert merged["lon_weightedcentroid"][0] > merged["meanlon"][0]

    def test_split_fragments_are_merged_too(self, cloudid_r0):
        cloudid, a = cloudid_r0
        merged_via_split, _ = run_worker(cloudid, a, [(1, [], [2])])
        merged_via_merge, _ = run_worker(cloudid, a, [(1, [2], [])])
        for name in recomputed_variable_names(DBZ_THRESHS) + ["merged_area"]:
            np.testing.assert_allclose(merged_via_split[name], merged_via_merge[name], equal_nan=True)

    def test_a_track_without_convective_pixels(self, cloudid_r0):
        cloudid, a = cloudid_r0
        a2 = {k: v.copy() for k, v in a.items()}
        a2["convsf"][a2["convsf"] == 4] = 3
        a2["conv_feature_number"][:] = 0
        path = write_cloudid(os.path.join(os.path.dirname(cloudid), "cloudid_20250502_190500.nc"), a2, 0.0,
                             basetime=BASETIME0 + 300)
        out, _ = run_worker(path, a2, [(1, [], [])])
        assert out["conv_majoraxis"][0] == 0.0
        assert np.isnan(out["maxETH_10dbz"][0])
        assert out["conv_area_45dbz"][0] == 0.0
        assert out["majoraxis"][0] > 0

    def test_fill_values_in_the_fragment_slots_are_ignored(self, cloudid_r0):
        cloudid, a = cloudid_r0
        out, diag = run_worker(cloudid, a, [(1, [], [])])          # both slot arrays hold only -9999
        assert out["merged_area"][0] == pytest.approx(100 * 9.0)
        assert diag == {"n_conflict": 0, "n_out_of_range": 0}

    def test_unreadable_file_returns_none(self, tmp_path):
        assert regionprops_singlefile(str(tmp_path / "cloudid_20250502_190000.nc"), np.array([1]),
                                      np.array([[-9999]]), np.array([[-9999]]), worker_config()) is None

    def test_inconsistent_feature_numbers_are_counted_not_fatal(self, cloudid_r0):
        cloudid, a = cloudid_r0
        # track 0 also claims feature 99, which is not in the file, and feature 3, which track 1 owns:
        # the later track keeps it, and neither case stops the step
        out, diag = run_worker(cloudid, a, [(1, [99, 3], []), (3, [], [])])
        assert diag["n_out_of_range"] == 1
        assert diag["n_conflict"] == 1
        assert out["merged_area"][0] == pytest.approx(100 * 9.0)           # feature 1 only
        assert out["merged_area"][1] == pytest.approx(50 * 9.0)            # feature 3 stayed with track 1

    def test_a_feature_listed_twice_by_the_same_track_is_not_a_conflict(self, cloudid_r0):
        cloudid, a = cloudid_r0
        out, diag = run_worker(cloudid, a, [(1, [2], [2])])
        assert diag["n_conflict"] == 0
        assert out["merged_area"][0] == pytest.approx(150 * 9.0)

    def test_no_tracks_in_the_file(self, cloudid_r0):
        cloudid, a = cloudid_r0
        out, diag = regionprops_singlefile(cloudid, np.array([], dtype=int), np.zeros((0, 1), dtype=int),
                                           np.zeros((0, 1), dtype=int), worker_config())
        assert all(len(v) == 0 for v in out.values())


@pytest.mark.parametrize("radius_km", [0.0, 10.0, 20.0])
class TestConvectiveLengthOfDilatedGroups:
    """conv_majoraxis: the longest group of nearby convective cells, measured on the region inside the merged mask."""

    @staticmethod
    def _expected(a, mask_numbers, radius_km):
        """Major axis [km] of each group (dilated region of connected convective cells inside the mask)."""
        mask = np.isin(a["feature_number"], mask_numbers)
        conv_groups = a["conv_feature_number"]
        best = 0.0
        for g in np.unique(conv_groups[mask & (a["convsf"] == 4)]):
            rows, cols = np.nonzero(mask & (conv_groups == g))
            best = max(best, major_axis_km(rows, cols))
        return best

    def test_conv_majoraxis_of_the_merged_mask(self, tmp_path, radius_km):
        a = build_arrays(radius_km)
        cloudid = write_cloudid(str(tmp_path / "cloudid_20250502_190000.nc"), a, radius_km)
        out, _ = run_worker(cloudid, a, [(1, [2], []), (3, [], [])])
        np.testing.assert_allclose(out["conv_majoraxis"][0], self._expected(a, [1, 2], radius_km), rtol=1e-4)
        np.testing.assert_allclose(out["conv_majoraxis"][1], self._expected(a, [3], radius_km), rtol=1e-4)

    def test_values_per_radius(self, tmp_path, radius_km):
        """The analytic lengths: undilated cells are 5x5 squares; the dilation joins C1+C2 at 10 km and all three at 20 km."""
        a = build_arrays(radius_km)
        cloudid = write_cloudid(str(tmp_path / "cloudid_20250502_190000.nc"), a, radius_km)
        out, _ = run_worker(cloudid, a, [(1, [2], []), (3, [], [])])
        square = 4 * np.sqrt(2.0) * 3.0                                    # 5x5 block: 17.0 km
        cols_c1c2 = np.arange(5, 23)                                       # 10 km: cols 5-22 of feature 1
        cols_c3 = np.arange(30, 38)                                        # 10 km: cols 30-37 of feature 2
        cols_all = np.concatenate([np.arange(5, 25), np.arange(30, 40)])   # 20 km: the whole merged mask
        rows5 = np.arange(10, 15)

        def length(cols):
            return major_axis_km(np.repeat(rows5, len(cols)), np.tile(cols, 5))
        if radius_km == 0.0:
            assert out["conv_majoraxis"][0] == pytest.approx(square, rel=1e-4)
            assert out["conv_majoraxis"][1] == pytest.approx(square, rel=1e-4)
        elif radius_km == 10.0:
            assert out["conv_majoraxis"][0] == pytest.approx(max(length(cols_c1c2), length(cols_c3)), rel=1e-4)
            assert out["conv_majoraxis"][0] > 3 * square / 1.5          # much longer than a single cell
            assert out["conv_majoraxis"][1] == pytest.approx(length(np.arange(5, 13)), rel=1e-4)
        else:
            assert out["conv_majoraxis"][0] == pytest.approx(length(cols_all), rel=1e-4)
            assert out["conv_majoraxis"][1] == pytest.approx(length(np.arange(5, 15)), rel=1e-4)

    def test_a_group_shared_by_two_tracks_is_split_by_their_masks(self, tmp_path, radius_km):
        """At 20 km the dilated regions of A and B touch (same conv_feature_number); each track keeps its own part."""
        a = build_arrays(radius_km)
        if radius_km == 20.0:
            assert a["conv_feature_number"][12, 7] == a["conv_feature_number"][27, 7] > 0
        cloudid = write_cloudid(str(tmp_path / "cloudid_20250502_190000.nc"), a, radius_km)
        out, _ = run_worker(cloudid, a, [(1, [2], []), (3, [], [])])
        a_only, _ = run_worker(cloudid, a, [(1, [2], [])])
        assert out["conv_majoraxis"][0] == pytest.approx(a_only["conv_majoraxis"][0], rel=1e-6)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

NSTEPS = 40
DT_HOUR = 1.0 / 12.0


def _stamp(t):
    return f"20250502_{19 + (5 * t) // 60:02d}{(5 * t) % 60:02d}00"


def build_run(tmp_path, radius_km=20.0, nfiles=NSTEPS, with_conv_feature_number=True):
    """Cloudid files of NSTEPS steps and a hand-made first-pass stats file of 3 tracks:
       0 = merge fragment (feature 2 -> into track 1) is dropped from the stats, 1 = track A (feature 1, with feature 2
       folded in), 2 = track B (feature 3)."""
    tracking = tmp_path / "tracking"
    stats = tmp_path / "stats"
    tracking.mkdir()
    stats.mkdir()
    a = build_arrays(radius_km)
    for t in range(nfiles):
        write_cloudid(str(tracking / f"cloudid_{_stamp(t)}.nc"), a, radius_km, with_conv_feature_number,
                      basetime=BASETIME0 + 300.0 * t)
    ntracks, ntimes, nmerge = 2, NSTEPS + 2, 3
    shape = (ntracks, ntimes)
    nan2d = lambda: np.full(shape, np.nan, dtype=np.float32)
    cn = np.full(shape, -9999, dtype=np.int32)
    cn[0, :NSTEPS], cn[1, :NSTEPS] = 1, 3
    mergecn = np.full((ntracks, ntimes, nmerge), -9999, dtype=np.int32)
    mergecn[0, :NSTEPS, 0] = 2
    splitcn = np.full((ntracks, ntimes, nmerge), -9999, dtype=np.int32)
    basetime = np.full(shape, np.nan)
    basetime[:, :NSTEPS] = BASETIME0 + 300.0 * np.arange(NSTEPS)
    total_area = nan2d()
    total_area[0, :NSTEPS], total_area[1, :NSTEPS] = 150 * 9.0, 50 * 9.0
    total_conv = nan2d()
    total_conv[0, :NSTEPS], total_conv[1, :NSTEPS] = 75 * 9.0, 25 * 9.0
    total_strat = nan2d()
    total_strat[0, :NSTEPS], total_strat[1, :NSTEPS] = 75 * 9.0, 25 * 9.0
    first = {name: nan2d() for name in recomputed_variable_names(DBZ_THRESHS)}
    first["majoraxis"][:, :NSTEPS] = 120.0                    # first-pass values: replaced by the second pass
    first["area_45dbz"][:, :NSTEPS] = 1.0
    lon0 = np.linspace(-100.0, -99.8, NSTEPS)
    first["meanlon"][:, :NSTEPS] = lon0
    first["meanlat"][:, :NSTEPS] = 35.0
    smooth = nan2d()
    ds = xr.Dataset(
        {
            "track_duration": (["tracks"], np.array([NSTEPS, NSTEPS], dtype=np.int32)),
            "base_time": (["tracks", "times"], basetime),
            "cloudnumber": (["tracks", "times"], cn),
            "merge_cloudnumber": (["tracks", "times", "mergers"], mergecn),
            "split_cloudnumber": (["tracks", "times", "mergers"], splitcn),
            "total_area": (["tracks", "times"], total_area),
            "total_conv_area": (["tracks", "times"], total_conv),
            "total_strat_area": (["tracks", "times"], total_strat),
            "is_mcs": (["tracks"], np.array([1, 0], dtype=np.int16)),
            "meanlon_smooth": (["tracks", "times"], smooth.copy()),
            "meanlat_smooth": (["tracks", "times"], smooth.copy()),
            **{k: (["tracks", "times"], v, {"long_name": k, "units": "x"}) for k, v in first.items()},
        },
        coords={"tracks": np.arange(ntracks), "times": np.arange(ntimes), "mergers": np.arange(nmerge)},
    )
    ds.to_netcdf(stats / "mcs_tracks_20250502.1900_20250502.2000.nc")
    config = {
        'mcsrobust_filebase': 'mcs_tracks_robust_', 'stats_outpath': str(stats) + "/",
        'tracking_outpath': str(tracking) + "/", 'cloudid_filebase': 'cloudid_',
        'startdate': '20250502.1900', 'enddate': '20250502.2000',
        'start_basetime': BASETIME0 - 10, 'end_basetime': BASETIME0 + 300.0 * NSTEPS,
        'tracks_dimname': 'tracks', 'times_dimname': 'times', 'run_parallel': 0, 'match_pixel_dt_thresh': 60,
        'datatimeresolution': DT_HOUR, 'max_speed_for_trajectory_smoothing': 40, 'fillval': -9999,
        'pixel_radius': DX / 1000.0, 'area_method': 'fixed', 'feature_varname': 'feature_number',
        'ref_varname': 'dbz_comp', 'pbc_direction': 'none', 'trackstats_dbz_area_threshs': DBZ_THRESHS,
        'mcs_majoraxis_thresh': 100, 'mcs_dbz45_area_thresh': 0, 'mcs_duration_thresh': 0.5, 'mcs_gap_hour': 0.25,
        'mcs_lifecycle_thresh': 0.5, 'mcs_lifecycle_smooth_hour': 0.0,
    }
    return config, a


def open_out(config):
    return xr.open_dataset(
        config["stats_outpath"] + "mcs_tracks_robust_20250502.1900_20250502.2000.nc",
        decode_times=False, mask_and_scale=False)


class TestDriver:

    def test_recomputes_on_the_merged_mask_and_keeps_the_rest(self, tmp_path):
        config, a = build_run(tmp_path, radius_km=20.0)
        out = compute_regionprops(config)
        assert os.path.basename(out) == "mcs_tracks_robust_20250502.1900_20250502.2000.nc"
        ds = open_out(config)
        first = xr.open_dataset(config["stats_outpath"] + "mcs_tracks_20250502.1900_20250502.2000.nc",
                                decode_times=False, mask_and_scale=False)
        # same tracks, in the same order, same dimensions
        assert ds.sizes["tracks"] == 2 and ds.sizes["times"] == NSTEPS + 2
        np.testing.assert_array_equal(ds["cloudnumber"].values, first["cloudnumber"].values)
        np.testing.assert_array_equal(ds["is_mcs"].values, first["is_mcs"].values)
        np.testing.assert_array_equal(ds["total_area"].values, first["total_area"].values)
        # the shape and extremes are those of the merged mask (the first-pass 120 km / 1 km2 are gone)
        rows, cols = pixels(a, 1, 2)
        majoraxis = ds["majoraxis"].values
        assert majoraxis[0, 0] == pytest.approx(major_axis_km(rows, cols), rel=1e-4)
        assert not np.any(majoraxis[:, :NSTEPS] == 120.0)
        assert np.all(np.isnan(majoraxis[:, NSTEPS:]))                       # no data after the last step stays NaN
        assert ds["area_45dbz"].values[0, 0] == pytest.approx(100 * 9.0)
        assert ds["max_dbz"].values[0, 0] == pytest.approx(52.0)
        assert ds["conv_majoraxis"].values[0, 0] > 60.0                      # 20 km: one long convective group
        # attributes say where the values come from
        assert "merged mask" in ds["majoraxis"].attrs["comments"]
        assert "20 km" in ds["conv_majoraxis"].attrs["comments"]
        assert ds.attrs["conv_dilate_radius_km"] == 20.0
        assert ds.attrs["n_pairs_not_computed"] == 0 and ds.attrs["n_pairs_area_mismatch"] == 0

    def test_lifecycle_is_staged_from_the_totals_and_the_merged_conv_majoraxis(self, tmp_path):
        config, _ = build_run(tmp_path, radius_km=20.0)
        compute_regionprops(config)
        ds = open_out(config)
        for name in ("lifecycle_stage", "lifecycle_cycle", "lifecycle_index", "lifecycle_ncycles",
                     "lifecycle_complete_flag", "lifecycle_evaluated"):
            assert name in ds
        assert ds["lifecycle_index"].dims == ("tracks", "lifestages") and ds.sizes["lifestages"] == 5
        # 12 steps of 5 min = 1 h >= mcs_lifecycle_thresh (0.5 h): both tracks are evaluated
        np.testing.assert_array_equal(ds["lifecycle_evaluated"].values, [1, 1])
        # track A: 150 km long convective group from the first step: an organized line, so cloud-only is not the answer
        assert ds["lifecycle_ncycles"].values[0] == 1
        assert np.any(ds["lifecycle_stage"].values[0, :NSTEPS] > 2)
        # track B: a single 35 km group, never organized
        assert ds["lifecycle_ncycles"].values[1] == 0

    def test_lifecycle_not_evaluated_below_the_duration_threshold(self, tmp_path):
        config, _ = build_run(tmp_path)
        config["mcs_lifecycle_thresh"] = 5.0                                  # the tracks last 3.3 h
        compute_regionprops(config)
        ds = open_out(config)
        np.testing.assert_array_equal(ds["lifecycle_evaluated"].values, [0, 0])
        assert np.all(ds["lifecycle_stage"].values == -9999)

    def test_dilation_radius_changes_the_lifecycle_input(self, tmp_path):
        lengths = {}
        for radius in (0.0, 20.0):
            sub = tmp_path / f"run_{int(radius)}"
            sub.mkdir()
            config, _ = build_run(sub, radius_km=radius)
            compute_regionprops(config)
            lengths[radius] = open_out(config)["conv_majoraxis"].values[0, 0]
        assert lengths[20.0] > 3 * lengths[0.0]

    def test_smoothed_trajectory_is_recomputed_from_the_new_centroid(self, tmp_path):
        config, a = build_run(tmp_path)
        compute_regionprops(config)
        ds = open_out(config)
        rows, cols = pixels(a, 1, 2)
        meanlon = ds["meanlon"].values[0, :NSTEPS]
        np.testing.assert_allclose(meanlon, (-100.0 + cols * DX / 111000.0).mean(), atol=1e-4)
        smooth = ds["meanlon_smooth"].values[0, :NSTEPS]
        assert np.all(np.isfinite(smooth))
        np.testing.assert_allclose(smooth, meanlon, atol=1e-4)               # a stationary track stays where it is

    def test_a_missing_cloudid_file_gives_nan_and_is_counted(self, tmp_path):
        config, _ = build_run(tmp_path)
        os.remove(config["tracking_outpath"] + f"cloudid_{_stamp(5)}.nc")     # 2 of 80 pairs = 2.5 %
        compute_regionprops(config)
        ds = open_out(config)
        assert ds.attrs["n_pairs_not_computed"] == 2
        assert np.all(np.isnan(ds["majoraxis"].values[:, 5]))
        assert np.all(np.isfinite(ds["majoraxis"].values[:, [4, 6]]))

    def test_a_feature_that_is_not_in_its_file_counts_as_not_computed(self, tmp_path):
        config, _ = build_run(tmp_path)
        path = config["stats_outpath"] + "mcs_tracks_20250502.1900_20250502.2000.nc"
        ds = xr.open_dataset(path, decode_times=False, mask_and_scale=False).load()
        ds.close()
        ds["cloudnumber"][1, 7] = 99                                          # track B, step 7: feature 99 does not exist
        os.remove(path)
        ds.to_netcdf(path)
        compute_regionprops(config)                                           # 1 of 80 pairs = 1.25 %: tolerated
        out = open_out(config)
        assert out.attrs["n_pairs_not_computed"] == 1
        assert np.isnan(out["majoraxis"].values[1, 7]) and np.isfinite(out["majoraxis"].values[1, 6])

    def test_too_many_missing_cloudid_files_abort(self, tmp_path):
        config, _ = build_run(tmp_path)
        for t in range(5, 9):                                                  # 8 of 80 pairs = 10 % > the 5 % limit
            os.remove(config["tracking_outpath"] + f"cloudid_{_stamp(t)}.nc")
        with pytest.raises(SystemExit):
            compute_regionprops(config)

    def test_area_mismatch_between_stats_and_pixels_aborts(self, tmp_path):
        config, _ = build_run(tmp_path)
        path = config["stats_outpath"] + "mcs_tracks_20250502.1900_20250502.2000.nc"
        ds = xr.open_dataset(path, decode_times=False, mask_and_scale=False).load()
        ds.close()
        ds["total_area"][0, :NSTEPS] = 100 * 9.0                              # the fragment's pixels are missing
        os.remove(path)
        ds.to_netcdf(path)
        with pytest.raises(SystemExit):
            compute_regionprops(config)

    def test_cloudid_without_conv_feature_number_exits_with_a_message(self, tmp_path, caplog):
        config, _ = build_run(tmp_path, with_conv_feature_number=False)
        with pytest.raises(SystemExit):
            compute_regionprops(config)
        assert "conv_feature_number" in caplog.text
        assert "conv_dilate_overwrite_radius_km" in caplog.text            # the message names the other way out


# ---------------------------------------------------------------------------
# conv_dilate_overwrite_radius_km: regroup the convective pixels at another radius
# ---------------------------------------------------------------------------

OVERWRITE_TRACKS = [(1, [2], []), (3, [], [])]


def overwrite_config(radius_km, **extra):
    return worker_config(dx=DX, dy=DX, conv_dilate_overwrite_radius_km=radius_km, **extra)


class TestOverwriteOption:
    """How conv_dilate_overwrite_radius_km is read, and the groups it makes."""

    def test_off_when_the_key_is_missing_or_null(self):
        assert conv_dilate_overwrite_km({}) is None
        assert conv_dilate_overwrite_km({"conv_dilate_overwrite_radius_km": None}) is None

    def test_radius_is_a_float_and_zero_is_a_radius_not_off(self):
        grid = {"dx": DX, "dy": DX}
        assert conv_dilate_overwrite_km({**grid, "conv_dilate_overwrite_radius_km": 9}) == 9.0
        assert conv_dilate_overwrite_km({**grid, "conv_dilate_overwrite_radius_km": 0}) == 0.0

    @pytest.mark.parametrize("bad", [-1, "9", True, [9], float("nan"), float("inf")])
    def test_an_invalid_radius_raises(self, bad):
        with pytest.raises(ValueError, match="conv_dilate_overwrite_radius_km"):
            conv_dilate_overwrite_km({"dx": DX, "dy": DX, "conv_dilate_overwrite_radius_km": bad})

    def test_the_grid_spacing_is_required(self):
        with pytest.raises(ValueError, match="dx and dy"):
            conv_dilate_overwrite_km({"conv_dilate_overwrite_radius_km": 9})

    def test_nine_km_is_three_pixels(self):
        """C1 and C2 are 6 px (18 km) apart: their 9 km discs (3 px) just meet, a smaller radius leaves them apart."""
        convsf = build_arrays(0.0)["convsf"]
        grid = {"dx": DX, "dy": DX}
        joined = redilate_conv_feature_number(convsf, 9.0, grid)
        apart = redilate_conv_feature_number(convsf, 8.9, grid)
        assert joined[12, 7] == joined[12, 17] > 0               # C1 and C2: one group
        assert apart[12, 7] != apart[12, 17]
        assert joined[12, 32] != joined[12, 17]                  # C3 is 33 km from C2: not joined
        np.testing.assert_array_equal(joined, build_arrays(9.0)["conv_feature_number"])


@pytest.mark.parametrize("overwrite_km", [0.0, 9.0, 10.0, 20.0])
class TestOverwriteInTheWorker:
    """Groups recomputed from convsf give the statistics of groups that idfeature made at that radius."""

    @staticmethod
    def _reference(tmp_path, radius_km):
        """The worker on a cloudid file whose groups were made at radius_km."""
        (tmp_path / "reference").mkdir()
        a = build_arrays(radius_km)
        cloudid = write_cloudid(str(tmp_path / "reference" / "cloudid_20250502_190000.nc"), a, radius_km)
        return run_worker(cloudid, a, OVERWRITE_TRACKS)[0]

    def test_replaces_the_groups_of_the_file_and_leaves_the_file_alone(self, tmp_path, overwrite_km):
        a = build_arrays(5.0)                                     # the file has 5 km groups
        cloudid = write_cloudid(str(tmp_path / "cloudid_20250502_190000.nc"), a, 5.0)
        stat = (os.stat(cloudid).st_size, os.stat(cloudid).st_mtime_ns)
        out, _ = run_worker(cloudid, a, OVERWRITE_TRACKS, overwrite_config(overwrite_km))
        reference = self._reference(tmp_path, overwrite_km)
        for name in reference:
            np.testing.assert_array_equal(out[name], reference[name], err_msg=name)
        own, _ = run_worker(cloudid, a, OVERWRITE_TRACKS)         # without the option: the 5 km groups
        assert out["conv_majoraxis"][0] != own["conv_majoraxis"][0]
        assert (os.stat(cloudid).st_size, os.stat(cloudid).st_mtime_ns) == stat
        with xr.open_dataset(cloudid, decode_times=False, mask_and_scale=False) as ds:
            np.testing.assert_array_equal(ds["conv_feature_number"].values.squeeze(), a["conv_feature_number"])

    def test_a_cloudid_file_without_conv_feature_number_works(self, tmp_path, overwrite_km):
        a = build_arrays(5.0)
        cloudid = write_cloudid(str(tmp_path / "cloudid_20250502_190000.nc"), a, 5.0, with_conv_feature_number=False)
        out, _ = run_worker(cloudid, a, OVERWRITE_TRACKS, overwrite_config(overwrite_km))
        reference = self._reference(tmp_path, overwrite_km)
        for name in reference:
            np.testing.assert_array_equal(out[name], reference[name], err_msg=name)


class TestOverwriteInTheDriver:

    @staticmethod
    def _run(tmp_path, name, file_radius_km, overwrite_km=None, with_conv_feature_number=True):
        (tmp_path / name).mkdir()
        config, _ = build_run(tmp_path / name, radius_km=file_radius_km, with_conv_feature_number=with_conv_feature_number)
        config.update(dx=DX, dy=DX, conv_dilate_overwrite_radius_km=overwrite_km)
        compute_regionprops(config)
        return open_out(config)

    def test_equals_a_run_on_files_made_at_that_radius(self, tmp_path):
        reference = self._run(tmp_path, "reference", 9.0)
        plain = self._run(tmp_path, "plain", 5.0)
        assert not np.array_equal(plain["conv_majoraxis"].values, reference["conv_majoraxis"].values, equal_nan=True)
        for name, with_groups in (("with_groups", True), ("without_groups", False)):
            ds = self._run(tmp_path, name, 5.0, overwrite_km=9.0, with_conv_feature_number=with_groups)
            assert set(ds.data_vars) == set(reference.data_vars)
            for var in reference.data_vars:
                np.testing.assert_array_equal(ds[var].values, reference[var].values, err_msg=f"{name}: {var}")

    def test_attributes_and_log_record_the_radius_used(self, tmp_path, caplog):
        caplog.set_level(logging.INFO)
        ds = self._run(tmp_path, "overwritten", 5.0, overwrite_km=9.0)
        assert ds.attrs["conv_dilate_radius_km"] == 9.0
        assert ds.attrs["conv_dilate_radius_km_idfeature"] == 5.0
        assert "9 km" in ds["conv_majoraxis"].attrs["comments"]
        assert "recomputed" in ds["conv_majoraxis"].attrs["comments"]
        assert "recomputed from convsf at 9 km" in caplog.text and "cloudid files have 5 km" in caplog.text
        plain = self._run(tmp_path, "plain", 5.0)
        assert plain.attrs["conv_dilate_radius_km"] == 5.0
        assert "conv_dilate_radius_km_idfeature" not in plain.attrs
        assert "recomputed" not in plain["conv_majoraxis"].attrs["comments"]

    def test_an_invalid_radius_exits_with_a_message(self, tmp_path, caplog):
        config, _ = build_run(tmp_path)
        config.update(dx=DX, dy=DX, conv_dilate_overwrite_radius_km=-3)
        with pytest.raises(SystemExit):
            compute_regionprops(config)
        assert "conv_dilate_overwrite_radius_km" in caplog.text
