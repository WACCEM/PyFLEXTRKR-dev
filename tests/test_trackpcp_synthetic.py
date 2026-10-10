"""
Synthetic tests of the track post-processing drivers (trackpcp_driver): statistics of an hourly variable inside
5-minute tracks, per-pixel accumulation, class maps. A small grid with four tracks that overlap, move and have
convective pixels; the expected values are computed with plain loops, independent of the vectorized code.
"""
import os
import calendar
import datetime
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pyflextrkr import trackpcp_driver as D
from pyflextrkr import trackpcp_func as F
from pyflextrkr.trackvar_io import parse_class_names, class_info

NY, NX, NSTEP, NHOUR = 30, 40, 36, 3
T0 = calendar.timegm(datetime.datetime(2025, 5, 1, 0, 0, 0).timetuple())
CLASS_LONG_NAME = "Most-organized feature class observed on this track: 1=MCS, 2=multi-cell, 3=isolated-cell, 4=non-convective"
TRACK_CLASS = np.array([1, 2, 3, 4], dtype=np.int16)


def step_time(s):
    return T0 + 300 * s + 20


def blob(track, s):
    """(rows, cols) slice of the track at step s, or None."""
    if track == 1 and 0 <= s <= 35:
        c0 = 2 + s // 2
        return slice(10, 14), slice(c0, c0 + 4)
    if track == 2 and 6 <= s <= 20:
        return slice(12, 15), slice(8, 11)
    if track == 3 and 10 <= s <= 14:
        return slice(20, 22), slice(30, 32)
    if track == 4 and 0 <= s <= 35:
        return slice(2, 7), slice(25, 31)
    return None


def make_masks():
    masks = np.zeros((NSTEP, NY, NX), dtype=np.int32)
    conv = np.zeros((NSTEP, NY, NX), dtype=np.int8)
    for s in range(NSTEP):
        for tr in (4, 2, 1, 3):          # later tracks overwrite: 1 over 2 over 4 where they overlap
            b = blob(tr, s)
            if b is None:
                continue
            masks[s][b] = tr
            conv[s][b] = 3
        if s % 3 == 0:
            conv[s, 11:13, 3 + s // 2:5 + s // 2] = 4          # convective core of track 1
        b3 = blob(3, s)
        if b3 is not None:
            conv[s][b3] = 4
    return masks, conv


def make_rain():
    rain = np.zeros((NHOUR, NY, NX), dtype=np.float32)
    yy, xx = np.mgrid[0:NY, 0:NX]
    for h in range(NHOUR):
        rain[h] = 2.0 + 0.2 * xx + 0.1 * yy + 3.0 * h
    rain[:, 0:2, 0:2] = np.nan
    rain[1, 12:14, 10:12] = np.nan                               # missing data inside the swath of track 1/2
    return rain


def time_string(t, fmt):
    return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime(fmt)


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    root = tmp_path_factory.mktemp("trackpcp")
    pix_dir, qpe_dir, stats_dir = root / "pixel", root / "qpe", root / "stats"
    for d in (pix_dir, qpe_dir, stats_dir):
        d.mkdir()
    masks, conv = make_masks()
    rain = make_rain()
    lon = 250.0 + 0.03 * np.arange(NX)[None, :] * np.ones((NY, 1))
    lat = 35.0 + 0.03 * np.arange(NY)[:, None] * np.ones((1, NX))
    dbz = np.full((NY, NX), 20.0, dtype=np.float32)
    dbz[:, 36:] = np.nan                                         # the right edge is outside the reflectivity coverage
    for s in range(NSTEP):
        xr.Dataset({"cloudtracknumber": (("time", "lat", "lon"), masks[s][None].astype(np.float64)),
                    "convsf": (("time", "lat", "lon"), conv[s][None].astype(np.int64)),
                    "dbz_comp": (("time", "lat", "lon"), dbz[None]),
                    "longitude": (("lat", "lon"), lon), "latitude": (("lat", "lon"), lat)}).to_netcdf(
            pix_dir / f"mcstrack_{time_string(step_time(s), '%Y%m%d_%H%M%S')}.nc")
    for h in range(NHOUR):
        t = T0 + 3600 * (h + 1)
        xr.Dataset({"precip": (("time", "y", "x"), rain[h][None], {"units": "mm/hr"})},
                   coords={"time": [0]}).to_netcdf(qpe_dir / f"QPE_{time_string(t, '%Y%m%d-%H%M%S')}.nc")
    # track statistics
    ntr = 4
    bt = np.full((ntr, NSTEP), np.nan)
    for tr in range(1, 5):
        for s in range(NSTEP):
            if blob(tr, s) is not None:
                bt[tr - 1, s] = step_time(s)
    ds = xr.Dataset({"base_time": (("tracks", "times"), bt),
                     "track_class": (("tracks",), TRACK_CLASS, {"long_name": CLASS_LONG_NAME, "_FillValue": np.int16(-9999)}),
                     "track_duration": (("tracks",), np.isfinite(bt).sum(axis=1).astype(np.int32))})
    startdate, enddate = "20250501.0000", "20250501.0300"
    ds.to_netcdf(stats_dir / f"mcs_tracks_final_{startdate}_{enddate}.nc")
    config = {
        "stats_outpath": str(stats_dir) + "/", "startdate": startdate, "enddate": enddate,
        "start_basetime": T0, "end_basetime": T0 + 3 * 3600, "pixeltracking_outpath": str(pix_dir) + "/",
        "pixeltracking_filebase": "mcstrack_", "pixel_radius": 3.0, "area_method": "fixed", "run_parallel": 0,
        "datatimeresolution": 5.0 / 60,
        "trackpcp_data_path": str(qpe_dir) + "/", "trackpcp_data_filebase": "QPE_", "trackpcp_data_time_format": "yyyymodd-hhmmss",
        "trackpcp_data_varname": "precip", "trackpcp_time_match": "prior", "trackpcp_window_sec": 3600,
        "trackpcp_stats_filebase": "mcs_tracks_final_", "trackpcp_out_filebase": "out_", "trackpcp_outdir": str(root / "out"),
        "trackpcp_stats": ["count", "mean", "median", "max", "p90"], "trackpcp_nmaxpf": 2, "trackpcp_rain_thresh": 6.0,
        "trackpcp_heavy_thresh": 9.0, "trackpcp_box_km": [9, 15], "trackpcp_accum_thresholds_mm": [20], "trackpcp_cluster_thresh_mm": 10,
    }
    return {"config": config, "masks": masks, "conv": conv, "rain": rain, "bt": bt, "root": root, "lon": lon, "lat": lat}


def expected_windows(syn, classes=None, use_class=True):
    """Brute force owner pixels per window: {hour: {(y, x): (track, convective, nsteps)}}."""
    out = {}
    cls = {1: 1, 2: 2, 3: 3, 4: 4}
    for h in range(NHOUR):
        steps = [s for s in range(NSTEP) if T0 + 3600 * h < step_time(s) <= T0 + 3600 * (h + 1)]
        claims = {}
        for s in steps:
            for y in range(NY):
                for x in range(NX):
                    tr = int(syn["masks"][s, y, x])
                    if tr == 0 or (classes is not None and cls[tr] not in classes):
                        continue
                    e = claims.setdefault((y, x), {}).setdefault(tr, [0, False])
                    e[0] += 1
                    e[1] = e[1] or (syn["conv"][s, y, x] == 4)
        owned = {}
        for pxl, d in claims.items():
            best = min(d, key=lambda t: ((cls[t] if use_class else 0), -d[t][0], t))
            owned[pxl] = (best, d[best][1], d[best][0])
        out[h] = owned
    return out


def test_parse_class_names():
    assert parse_class_names(CLASS_LONG_NAME) == {1: "MCS", 2: "multi-cell", 3: "isolated-cell", 4: "non-convective"}
    assert parse_class_names("no list here") == {}
    values, names = class_info([1, 2, 5], CLASS_LONG_NAME, None)
    assert values == [1, 2, 5] and names == ["MCS", "multi-cell", "class_5"]


@pytest.mark.parametrize("classes", [None, [1, 2]])
def test_step1_matches_brute_force(synthetic, classes):
    config = dict(synthetic["config"])
    config["trackpcp_classes"] = classes
    outfile = D.compute_track_precip(config)
    ds = xr.open_dataset(outfile)
    exp = expected_windows(synthetic, classes)
    rain = synthetic["rain"]
    area = 9.0
    checked = 0
    for h in range(NHOUR):
        steps = [s for s in range(NSTEP) if T0 + 3600 * h < step_time(s) <= T0 + 3600 * (h + 1)]
        for tr in range(1, 5):
            if classes is not None and tr not in classes:
                assert np.isnan(ds["total_rain"].values[tr - 1]).all()
                continue
            tsteps = [s for s in steps if blob(tr, s) is not None]
            if not tsteps:
                continue
            place = tsteps[-1]
            pix = [p for p, (t, c, n) in exp[h].items() if t == tr]
            vals = np.array([rain[h][p] for p in pix], dtype=float)
            conv = np.array([exp[h][p][1] for p in pix])
            fin = np.isfinite(vals)
            row = ds.isel(tracks=tr - 1, times=place)
            assert float(row["total_rain"]) == pytest.approx(np.nansum(vals), rel=1e-5)
            assert float(row["total_volrain"]) == pytest.approx(np.nansum(vals) * area, rel=1e-5)
            assert float(row["total_rainvolume"]) == pytest.approx(np.nansum(vals) * area * 1e-3, rel=1e-5)
            assert float(row["pcp_area"]) == pytest.approx(len(pix) * area)
            assert float(row["pcp_nsteps"]) == len(tsteps)
            assert float(row["precip_count"]) == fin.sum()
            assert float(row["precip_mean"]) == pytest.approx(np.nanmean(vals), rel=1e-5)
            assert float(row["precip_median"]) == pytest.approx(np.median(vals[fin]), rel=1e-5)
            assert float(row["precip_p90"]) == pytest.approx(np.percentile(vals[fin], 90), rel=1e-5)
            assert float(row["precip_max"]) == pytest.approx(np.nanmax(vals), rel=1e-5)
            if conv.any():
                assert float(row["conv_rain"]) == pytest.approx(np.nansum(vals[conv]), rel=1e-5)
                assert float(row["pcp_conv_area"]) == pytest.approx(conv.sum() * area)
            if (~conv).any():
                assert float(row["strat_rain"]) == pytest.approx(np.nansum(vals[~conv]), rel=1e-5)
            # the other steps of the window and everything outside it stay NaN
            others = [s for s in range(NSTEP) if s != place]
            assert np.isnan(ds["total_rain"].values[tr - 1, others][np.isin(others, steps)]).all()
            assert row["pcp_time"].values == np.datetime64(T0 + 3600 * (h + 1), "s")   # decoded from "seconds since 1970-01-01"
            checked += 1
    assert checked >= (6 if classes is None else 4)
    # per-track total equals the sum over the windows
    for tr in range(1, 5):
        if classes is not None and tr not in classes:
            continue
        col = ds["total_rainvolume"].values[tr - 1]
        assert float(ds["track_total_rainvolume"].values[tr - 1]) == pytest.approx(np.nansum(col), rel=1e-5)
    # the Parquet table has the same rows
    df = pd.read_parquet(outfile[:-3] + ".parquet")
    assert len(df) == int(np.isfinite(ds["total_rain"].values).sum())
    ds.close()


def test_owner_priority_and_no_class_variable(synthetic):
    """Track 1 (MCS) wins the pixels it shares with track 2 and 4; without a class variable the track with most steps wins."""
    masks, conv = synthetic["masks"], synthetic["conv"]
    nlut = 5
    cls = np.array([0, 1, 2, 3, 4], dtype=np.int16)
    static = {"class_lut": cls, "class_values": None, "class_priority": None}
    _, select, rank = F.class_luts(nlut, None, None, static)
    steps = [s for s in range(NSTEP) if T0 < step_time(s) <= T0 + 3600]
    owner = F.window_owner([masks[s] for s in steps], [conv[s] for s in steps], select, rank, 4)
    exp = expected_windows(synthetic)[0]
    got = {(int(p // NX), int(p % NX)): (int(t), bool(c), int(n)) for p, t, c, n in
           zip(owner["pix"], owner["trk"], owner["conv"], owner["nsteps"])}
    assert got == exp
    assert owner["n_pixels_contested"] > 0
    # no class variable: priority by steps only
    static_nc = {"class_lut": None, "class_values": None, "class_priority": None}
    _, select2, rank2 = F.class_luts(nlut, None, None, static_nc)
    owner2 = F.window_owner([masks[s] for s in steps], [conv[s] for s in steps], select2, rank2, 4)
    exp2 = expected_windows(synthetic, use_class=False)[0]
    got2 = {(int(p // NX), int(p % NX)): (int(t), bool(c), int(n)) for p, t, c, n in
            zip(owner2["pix"], owner2["trk"], owner2["conv"], owner2["nsteps"])}
    assert got2 == exp2


def test_step1_without_class_variable(synthetic):
    config = dict(synthetic["config"])
    config["trackpcp_class_varname"] = None
    config["trackpcp_out_filebase"] = "noclass_"
    outfile = D.compute_track_precip(config)
    ds = xr.open_dataset(outfile)
    assert np.isfinite(ds["total_rain"].values).sum() > 0
    ds.close()


def test_step2_pixel_accumulation_and_maps(synthetic):
    config = dict(synthetic["config"])
    config["trackpcp_classes"] = [1, 2]
    pix_file, met_file = D.compute_track_pixel_accum(config)
    df = pd.read_parquet(pix_file)
    exp = expected_windows(synthetic, [1, 2])
    rain = synthetic["rain"]
    for tr in (1, 2):
        truth = np.zeros((NY, NX))
        for h in range(NHOUR):
            for (y, x), (t, c, n) in exp[h].items():
                if t == tr and np.isfinite(rain[h][y, x]):
                    truth[y, x] += rain[h][y, x]            # window of 1 h: depth = rate
        m = D.track_pixels_to_map(df, tr, NY, NX)
        got = np.nan_to_num(m)
        np.testing.assert_allclose(got, truth, rtol=1e-5, atol=1e-5)
    assert set(df["track_number"].unique()) == {1, 2}
    met = pd.read_parquet(met_file).set_index("track_number")
    dur_h = (np.nanmax(synthetic["bt"], axis=1) - np.nanmin(synthetic["bt"], axis=1)) / 3600 + 5 / 60
    for tr in (1, 2):
        rows = df[df["track_number"] == tr]
        assert met.loc[tr, "accum_max_mm"] == pytest.approx(rows["accum_mm"].max(), rel=1e-5)
        assert met.loc[tr, "accum_vol_1e6m3"] == pytest.approx((rows["accum_mm"] * 9.0).sum() * 1e-3, rel=1e-4)
        np.testing.assert_allclose(rows["rate_mm_day"].values, rows["accum_mm"].values / (dur_h[tr - 1] / 24), rtol=1e-4)
        assert met.loc[tr, "local_depth_mm_9km"] >= met.loc[tr, "local_depth_mm_15km"] - 1e-6   # larger box = lower mean
    assert (met["local_rate_mm_h_9km"] > 0).all()

    # step 3: class maps; the per-class sum equals the pixel table of the same classes
    config["trackpcp_classes"] = None
    config["trackpcp_pcp_thresh"] = 5.0
    config["trackpcp_coverage_varname"] = "dbz_comp"
    nc = D.compute_class_maps(config)
    cm = xr.open_dataset(nc)
    assert list(cm["class_value"].values) == [1, 2, 3, 4]
    assert list(cm["class_name"].values) == ["MCS", "multi-cell", "isolated-cell", "non-convective"]
    exp_all = expected_windows(synthetic)
    for ci, tr_class in enumerate([1, 2, 3, 4]):
        truth = np.zeros((NY, NX))
        cnt = np.zeros((NY, NX))
        for h in range(NHOUR):
            for (y, x), (t, c, n) in exp_all[h].items():
                if t == tr_class:
                    cnt[y, x] += 1
                    if np.isfinite(rain[h][y, x]):
                        truth[y, x] += rain[h][y, x]
        np.testing.assert_allclose(cm["precip"].values[ci], truth, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(cm["cloud_count"].values[ci], cnt)
    # classes 1 and 2 of the map equal the accumulations of step 2 (same owner rule: step 2 had only classes 1 and 2,
    # so shared pixels with class 3 / 4 are the same because the best class wins)
    for ci, tr in ((0, 1), (1, 2)):
        np.testing.assert_allclose(cm["precip"].values[ci], np.nan_to_num(D.track_pixels_to_map(df, tr, NY, NX)),
                                   rtol=1e-5, atol=1e-5)
    total = np.nansum(rain, axis=0)
    np.testing.assert_allclose(cm["total_precip"].values, total, rtol=1e-5, atol=1e-5)
    assert int(cm["ntimes"]) == NHOUR
    # precipitation where the coverage variable is missing in every step of the window: the right edge
    unc = np.where(np.isfinite(rain), rain, 0.0).sum(axis=0)
    unc[:, :36] = 0.0
    np.testing.assert_allclose(cm["precip_uncovered"].values, unc, rtol=1e-5, atol=1e-5)
    assert (cm["precip"].sum("class").values <= total + 1e-4).all()
    cm.close()


def test_zarr_source_equals_netcdf(synthetic):
    pytest.importorskip("zarr")
    root = synthetic["root"]
    rain = synthetic["rain"]
    times = np.array([np.datetime64(T0 + 3600 * (h + 1), "s").astype("datetime64[ns]") for h in range(NHOUR)])
    zpath = str(root / "qpe.zarr")
    xr.Dataset({"precip": (("time", "y", "x"), rain)}, coords={"time": times}).to_zarr(zpath, mode="w")
    config = dict(synthetic["config"])
    config.update({"trackpcp_data_path": zpath, "trackpcp_data_format": None, "trackpcp_out_filebase": "zarr_",
                   "trackpcp_stats": ["mean"]})
    a = xr.open_dataset(D.compute_track_precip(config))
    config2 = dict(synthetic["config"])
    config2.update({"trackpcp_out_filebase": "nc_", "trackpcp_stats": ["mean"]})
    b = xr.open_dataset(D.compute_track_precip(config2))
    np.testing.assert_allclose(a["total_rain"].values, b["total_rain"].values, equal_nan=True)
    a.close()
    b.close()


def test_nearest_mode_with_5min_data(synthetic):
    """5-min data and 5-min tracks: every step is its own window and the value is stored at that step."""
    root = synthetic["root"]
    qdir = root / "qpe5"
    qdir.mkdir(exist_ok=True)
    rng = np.random.default_rng(5)
    rain5 = rng.uniform(0.0, 20.0, size=(NSTEP, NY, NX)).astype(np.float32)
    for s in range(NSTEP):
        xr.Dataset({"precip": (("time", "y", "x"), rain5[s][None])}, coords={"time": [0]}).to_netcdf(
            qdir / f"R5_{time_string(step_time(s), '%Y%m%d-%H%M%S')}.nc")
    config = dict(synthetic["config"])
    config.update({"trackpcp_data_path": str(qdir) + "/", "trackpcp_data_filebase": "R5_", "trackpcp_time_match": "nearest",
                   "trackpcp_window_sec": 300, "trackpcp_out_filebase": "nearest_", "trackpcp_stats": ["max"],
                   "trackpcp_precip": True, "start_basetime": T0, "end_basetime": T0 + 3 * 3600})
    ds = xr.open_dataset(D.compute_track_precip(config))
    masks = synthetic["masks"]
    for s in (3, 11, 25):
        for tr in (1, 4):
            m = masks[s] == tr
            if not m.any():
                continue
            got = float(ds["total_rain"].values[tr - 1, s])
            assert got == pytest.approx(float(rain5[s][m].sum()), rel=1e-5)
            assert float(ds["pcp_nsteps"].values[tr - 1, s]) == 1.0
            # a 5-minute record of mm/h over 300 s is a depth of rate / 12 mm
            assert float(ds["total_rainvolume"].values[tr - 1, s]) == pytest.approx(float(rain5[s][m].sum()) * 9.0 * (300 / 3600) * 1e-3, rel=1e-5)
    # every step of every track has a value
    assert np.isfinite(ds["total_rain"].values[0, :NSTEP]).all()
    ds.close()
