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


def write_combined_tree(root, rain, nhour=NHOUR, name="combined"):
    """
    A 5-minute data set in one folder per day, like the combined MRMS input: a file every 5 minutes, only the sharp-hour files
    carry the hourly variable QPE1h (the record at T covers (T - 1 h, T]), the others another variable.

    Args:
        root: Path
            Folder to make the tree in.
        rain: np.ndarray [nhour, ny, nx]
            Hourly precipitation, hour h is the record valid at T0 + (h + 1) h.
        nhour: int (default: NHOUR)
            Number of hourly records.
        name: string (default: 'combined')
            Name of the folder of the tree.

    Returns:
        path: string, the folder (with a trailing slash)
    """
    path = root / name
    for step in range(12 * nhour + 1):
        t = T0 + 300 * step
        day = path / time_string(t, "%Y%m%d")
        day.mkdir(parents=True, exist_ok=True)
        if step % 12 == 0:                                        # every sharp-hour file has the variable (the first: the hour before)
            data = {"QPE1h": (("time", "y", "x"), rain[step // 12 - 1][None] if step > 0 else np.zeros((1, NY, NX), dtype=np.float32))}
        else:
            data = {"CompositeReflectivity": (("time", "y", "x"), np.zeros((1, NY, NX), dtype=np.float32))}
        xr.Dataset(data, coords={"time": [0]}).to_netcdf(day / f"MRMS_HRRR_{time_string(t, '%Y%m%d_%H%M%S')}.nc")
    return str(path) + "/"


def test_file_source_in_day_folders_with_the_sharp_hour_files_only(tmp_path):
    from pyflextrkr.trackvar_io import NetCDFFileSource, open_data_source
    rain = np.arange(6, dtype=np.float32)[:, None, None] * np.ones((6, NY, NX), dtype=np.float32)
    t_start = T0 + 22 * 3600                                   # the tree crosses midnight: 2025-05-01 22:00 ... 2025-05-02 04:00
    shifted = np.array(rain)
    path = tmp_path / "tree"
    for step in range(12 * 6 + 1):
        t = t_start + 300 * step
        day = path / time_string(t, "%Y%m%d")
        day.mkdir(parents=True, exist_ok=True)
        var = {"QPE1h": (("time", "y", "x"), (shifted[step // 12 - 1] if step > 0 else -np.ones((NY, NX), dtype=np.float32))[None])} \
            if step % 12 == 0 else {"Other": (("time", "y", "x"), np.zeros((1, NY, NX), dtype=np.float32))}
        xr.Dataset(var, coords={"time": [0]}).to_netcdf(day / f"MRMS_HRRR_{time_string(t, '%Y%m%d_%H%M%S')}.nc")
    kwargs = dict(time_format="yyyymodd_hhmmss", varname="QPE1h", subdir_format="%Y%m%d", every_sec=3600)
    src = NetCDFFileSource(str(path) + "/", "MRMS_HRRR_", start_basetime=t_start, end_basetime=t_start + 6 * 3600, **kwargs)
    # the records are the sharp hours of both days (not the 5-minute files): 22:00 of the first day to 04:00 of the second
    expected = [t_start + 3600 * h for h in range(7)]
    np.testing.assert_array_equal(src.times, expected)
    assert src.nrecords == 7 and src.grid_shape == (NY, NX)
    for i in range(7):
        assert os.path.basename(src.filename(i)).startswith("MRMS_HRRR_") and float(src.read(i)[0, 0]) == float(i - 1)
    # a time range inside one day lists only the records of the range
    part = NetCDFFileSource(str(path) + "/", "MRMS_HRRR_", start_basetime=t_start + 3600 * 3, end_basetime=t_start + 3600 * 4, **kwargs)
    np.testing.assert_array_equal(part.times, expected[3:5])
    # the same source from the config keys
    config = {"trackpcp_data_path": str(path), "trackpcp_data_filebase": "MRMS_HRRR_", "trackpcp_data_time_format": "yyyymodd_hhmmss",
              "trackpcp_data_varname": "QPE1h", "trackpcp_data_subdir_format": "%Y%m%d", "trackpcp_data_every_sec": 3600}
    cfg_src = open_data_source(config, t_start, t_start + 6 * 3600)
    np.testing.assert_array_equal(cfg_src.times, expected)
    # without the options every file is a record (the default is unchanged): a flat folder is read as before
    with pytest.raises(Exception):
        NetCDFFileSource(str(path) + "/", "MRMS_HRRR_", start_basetime=t_start, end_basetime=t_start + 7200, time_format="yyyymodd_hhmmss",
                         varname="QPE1h").read(0)
    # wrong combinations stop with a message
    with pytest.raises(ValueError, match="time from the file name"):
        NetCDFFileSource(str(path) + "/", "MRMS_HRRR_", time_varname="time", subdir_format="%Y%m%d")
    with pytest.raises(ValueError, match="positive"):
        NetCDFFileSource(str(path) + "/", "MRMS_HRRR_", every_sec=0)


def test_class_maps_from_combined_files_equal_the_flat_source(synthetic):
    """The same hourly precipitation as one file per record and as the sharp-hour files of 5-minute files in day folders."""
    config = dict(synthetic["config"])
    config["trackpcp_pcp_thresh"] = 5.0
    flat = xr.open_dataset(D.compute_class_maps(config, outfile=str(synthetic["root"] / "out" / "flat_maps.nc")))
    combined = dict(config)
    combined.update({"trackpcp_data_path": write_combined_tree(synthetic["root"], synthetic["rain"]), "trackpcp_data_filebase": "MRMS_HRRR_",
                     "trackpcp_data_time_format": "yyyymodd_hhmmss", "trackpcp_data_varname": "QPE1h",
                     "trackpcp_data_subdir_format": "%Y%m%d", "trackpcp_data_every_sec": 3600})
    got = xr.open_dataset(D.compute_class_maps(combined, outfile=str(synthetic["root"] / "out" / "combined_maps.nc")))
    for name in ("precip", "cloud_count", "precip_count", "precip_conv", "precip_strat", "total_precip", "total_precip_count", "ntimes"):
        np.testing.assert_array_equal(flat[name].values, got[name].values, err_msg=name)
    assert int(got["ntimes"]) == NHOUR
    flat.close()
    got.close()
    # one calendar month of the run: the data source is made for that month and the maps are the same
    month = D.compute_class_maps(combined, year=2025, month=5, outfile=str(synthetic["root"] / "out" / "month_maps.nc"))
    with xr.open_dataset(month) as m, xr.open_dataset(str(synthetic["root"] / "out" / "flat_maps.nc")) as ref:
        assert int(m["ntimes"]) == NHOUR
        np.testing.assert_array_equal(m["precip"].values, ref["precip"].values)


def test_a_month_is_cut_to_the_tracking_period_and_a_month_outside_it_is_an_error(synthetic):
    """A run of a few hours has the windows of those hours in its month maps, a month that it does not touch stops with a message."""
    config = dict(synthetic["config"])
    config["trackpcp_pcp_thresh"] = 5.0
    out = synthetic["root"] / "out"
    full = D.compute_class_maps(config, year=2025, month=5, outfile=str(out / "cut_full.nc"))
    short = dict(config)
    short["end_basetime"] = T0 + 2 * 3600                              # the first two hours of the three of the run
    part = D.compute_class_maps(short, year=2025, month=5, outfile=str(out / "cut_short.nc"))
    with xr.open_dataset(full) as a, xr.open_dataset(part) as b:
        assert int(a["ntimes"]) == NHOUR and int(b["ntimes"]) == NHOUR - 1
        assert float(b["total_precip"].sum()) < float(a["total_precip"].sum())
    with pytest.raises(ValueError, match="outside the tracking period"):
        D.compute_class_maps(config, year=2025, month=6, outfile=str(out / "cut_june.nc"))


def test_step_1_with_row_statistics_makes_the_parquet_table_only(synthetic):
    """stats_format: rows has no dense netCDF to copy: step 1 writes the Parquet table, the same rows as with the dense file."""
    from pyflextrkr.rowstats import dense_to_rows, write_rows
    from pyflextrkr.step_outputs import check_step_outputs
    config = dict(synthetic["config"])
    config.update({"trackpcp_outdir": str(synthetic["root"] / "dense_out"), "trackpcp_steps": [1]})
    dense_file = D.stats_file_name(config, D.settings(config))
    parquet = lambda c: D.track_precip_file(c, D.settings(c))[:-3] + ".parquet"
    D.trackpcp_driver(config)
    reference = pd.read_parquet(parquet(config))
    # the same statistics as a row folder only (a row table has the entries of a track one after the other: the times are left-packed)
    raw = xr.open_dataset(dense_file, mask_and_scale=False, decode_times=False).load()
    packed = np.full(raw["base_time"].shape, np.nan)
    for k, row in enumerate(raw["base_time"].values):
        packed[k, :np.isfinite(row).sum()] = row[np.isfinite(row)]
    raw["base_time"] = (("tracks", "times"), packed)
    rows = dict(config)
    rows["stats_outpath"] = str(synthetic["root"] / "rows_stats") + "/"
    rows["trackpcp_outdir"] = str(synthetic["root"] / "rows_out")
    write_rows(dense_to_rows(raw), D.stats_file_name(rows, D.settings(rows))[:-3] + ".rows")
    assert not os.path.isfile(D.stats_file_name(rows, D.settings(rows)))
    files = D.trackpcp_driver(rows)
    assert files == [parquet(rows)] and os.listdir(rows["trackpcp_outdir"]) == [os.path.basename(parquet(rows))]
    ok, message = check_step_outputs("trackpcp", rows)
    assert ok and message == "1 files"
    # the records are the same, the time index is the position in the left-packed row
    got = pd.read_parquet(files[0])
    assert len(got) == len(reference) and (got["track_number"].values == reference["track_number"].values).all()
    pd.testing.assert_frame_equal(got.drop(columns="time_index"), reference.drop(columns="time_index"))


def test_months_of_period_steps_and_settings():
    t = lambda *a: calendar.timegm(datetime.datetime(*a).timetuple())
    # the end of the period is not part of it: a year from January 1st to January 1st is 12 months
    assert D.months_of_period({"start_basetime": t(2025, 1, 1), "end_basetime": t(2026, 1, 1)}) == [(2025, m) for m in range(1, 13)]
    assert D.months_of_period({"start_basetime": t(2025, 11, 20, 6), "end_basetime": t(2026, 2, 1)}) == [(2025, 11), (2025, 12), (2026, 1)]
    assert D.months_of_period({"start_basetime": t(2025, 5, 1), "end_basetime": t(2025, 5, 1, 3)}) == [(2025, 5)]
    assert D.trackpcp_step_list({}) == [1, 2, 3] and D.trackpcp_step_list({"trackpcp_steps": [3, 1, 3]}) == [1, 3]
    for bad in ([4], [0, 1], ["x"]):
        with pytest.raises((ValueError, TypeError)):
            D.trackpcp_step_list({"trackpcp_steps": bad})
    assert D.classmaps_by({}) == "period" and D.classmaps_by({"trackpcp_classmaps_by": "month"}) == "month"
    with pytest.raises(ValueError, match="period"):
        D.classmaps_by({"trackpcp_classmaps_by": "year"})


def test_the_pipeline_step_makes_its_outputs_by_month_and_the_check_sees_them(synthetic, monkeypatch):
    from pyflextrkr.step_outputs import check_step_outputs
    config = dict(synthetic["config"])
    config.update({"trackpcp_outdir": str(synthetic["root"] / "step_out"), "trackpcp_out_filebase": "step_", "trackpcp_steps": [1, 2, 3],
                   "trackpcp_classmaps_by": "month", "trackpcp_pcp_thresh": 5.0})
    ok, message = check_step_outputs("trackpcp", config)
    assert not ok and "missing" in message                                       # nothing made yet
    files = D.trackpcp_driver(config)
    names = [os.path.basename(f) for f in files]
    assert names == ["step_20250501.0000_20250501.0300.nc", "step_20250501.0000_20250501.0300.parquet",
                     "trackpcp_pixels_20250501.0000_20250501.0300.parquet", "trackpcp_metrics_20250501.0000_20250501.0300.parquet",
                     "trackpcp_grid.nc", "trackpcp_classmaps_202505.nc"]
    ok, message = check_step_outputs("trackpcp", config)
    assert ok and message == "6 files"
    # the month file is the file of the period (the whole run is inside May)
    period = xr.open_dataset(D.compute_class_maps(config, outfile=str(synthetic["root"] / "step_out" / "period.nc")))
    month = xr.open_dataset(files[-1])
    for name in ("precip", "cloud_count", "precip_count", "total_precip", "ntimes"):
        np.testing.assert_array_equal(period[name].values, month[name].values, err_msg=name)
    period.close()
    month.close()
    os.remove(files[2])
    ok, message = check_step_outputs("trackpcp", config)
    assert not ok and "1 of 6" in message and "trackpcp_pixels_" in message
    # step 3 alone, one file for the whole period (the default)
    only3 = dict(config)
    only3.update({"trackpcp_steps": [3], "trackpcp_classmaps_by": "period", "trackpcp_outdir": str(synthetic["root"] / "step_out3")})
    assert [os.path.basename(f) for f in D.trackpcp_driver(only3)] == ["trackpcp_classmaps_20250501.0000_20250501.0300.nc"]
    assert check_step_outputs("trackpcp", only3) == (True, "1 files")
    # the statistics of the tracks are read once for all the months of a run
    calls = []
    real = D.open_tracks
    monkeypatch.setattr(D, "open_tracks", lambda cfg, S: (calls.append(1), real(cfg, S))[1])
    monkeypatch.setattr(D, "months_of_period", lambda cfg: [(2025, 5), (2025, 5), (2025, 5)])
    D.compute_class_maps_monthly(only3)
    assert len(calls) == 1
    # a typo in the settings stops before any step runs
    with pytest.raises(ValueError):
        D.trackpcp_driver(dict(config, trackpcp_classmaps_by="week"))
