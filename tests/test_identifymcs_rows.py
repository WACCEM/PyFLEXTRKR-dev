"""
Tests of the rows mode of identifymcs (pyflextrkr/identifymcs_rows.py) and of the table helpers it uses (rowstats.TableBuilder,
load_sparse_entries, gather_index, table_differences).

The central check: for the same sparse trackstats file, the row tables of identifymcs_radar_rows are identical (every column,
dtype and value, and the meta data: dims, dtypes, fill values, attributes) to what dense_to_rows makes of the dense file of
identifymcs_radar. It is run on the 3-track file of the dense tests and on random files with merge / split fragments that coexist
with their root or not, chains of fragments, a full slot dimension, tracks with entries missing and the final-statistics filters.
Synthetic data only.
"""
import os

import numpy as np
import pytest
import xarray as xr

from pyflextrkr.ft_utilities import load_sparse_trackstats
from pyflextrkr.identifymcs_radar import identifymcs_radar
from pyflextrkr.identifymcs_rows import identifymcs_radar_rows
from pyflextrkr.rowstats import (RowStats, TableBuilder, dense_to_rows, gather_index, load_sparse_entries, read_tables,
                                 rows_to_dense, table_differences)
from test_identifymcs_radar_synthetic import _build_sparse_trackstats_file, _make_driver_config
from test_rowstats import _dense, _same_attrs, _write

DT_HOUR = 1.0 / 12.0


def random_sparse_file(tmp_path, seed, ntracks=70, nmissing=3, ntiny=0):
    """
    Write a sparse trackstats file of random tracks: long roots (some of them MCS-sized), short fragments that merge into / split
    from a root (most of them inside the time span of the root, some not), fragments of fragments, and a few valid entries that
    the sparse file does not have (base_time is then 0 there).

    Args:
        tmp_path: pathlib.Path
            Folder to write to.
        seed: int
            Random seed.
        ntracks: int (default: 70)
            Number of tracks.
        nmissing: int (default: 3)
            Valid entries to leave out of the sparse file.
        ntiny: int (default: 0)
            Number of the last tracks that are standalone tracks of one or two entries whose positions are NaN or jump (the
            trajectory smoothing takes other branches for such short trajectories).

    Returns:
        stats_path: string
            Folder with trackstats_sparse_20250101.0000_20250101.1200.nc (with trailing slash).
    """
    rng = np.random.default_rng(seed)
    t0, dt = 1746200000.0, DT_HOUR * 3600.0
    duration = np.zeros(ntracks, dtype=np.int32)
    start = np.zeros(ntracks, dtype=np.int64)
    merge_into = np.zeros(ntracks)
    split_from = np.zeros(ntracks)
    nroots = ntracks // 2
    is_big = np.zeros(ntracks, dtype=bool)
    for i in range(nroots):                                                        # long tracks, a quarter of them MCS-sized
        is_big[i] = rng.random() < 0.25
        duration[i] = rng.integers(50, 70) if is_big[i] else rng.integers(14, 60)
        start[i] = rng.integers(0, 40)
    for i in range(nroots, ntracks):                                               # short tracks that link to another track
        parent = int(rng.integers(0, i))                                           # a root or an earlier fragment
        duration[i] = rng.integers(1, 12)
        inside = rng.random() < 0.75
        if inside and duration[parent] > duration[i]:
            start[i] = start[parent] + rng.integers(0, duration[parent] - duration[i] + 1)
        else:
            start[i] = rng.integers(0, 90)
        kind = rng.integers(0, 4)
        if kind in (0, 2):
            merge_into[i] = parent + 1
        if kind in (1, 2):
            split_from[i] = parent + 1
    tiny = np.zeros(ntracks, dtype=bool)
    if ntiny:
        tiny[ntracks - ntiny:] = True
        duration[tiny] = rng.integers(1, 3, ntiny)
        start[tiny] = rng.integers(0, 90, ntiny)
        merge_into[tiny] = split_from[tiny] = 0
    width = int(duration.max())
    shape = (ntracks, width)
    valid = np.arange(width)[None, :] < duration[:, None]

    def field(values):
        out = np.zeros(shape, dtype=values.dtype)
        out[valid] = values[valid]
        return out

    random_f = lambda lo, hi, dtype=np.float64: field(rng.uniform(lo, hi, shape).astype(dtype))
    base_time = field(t0 + (start[:, None] + np.arange(width)[None, :]) * dt)
    majoraxis = random_f(20, 160)
    majoraxis[is_big] = np.where(valid[is_big], 150.0, 0.0)                         # above the MCS threshold all the time
    area = random_f(100, 30000, np.float32)
    area[rng.random(shape) < 0.05] = np.nan                                         # NaN data inside a track
    area[~valid] = 0
    conv_area = random_f(0, 5000)
    strat_area = random_f(0, 9000)
    lon = field(-98 + np.cumsum(rng.normal(0, 0.05, shape), axis=1))
    lat = field(35 + np.cumsum(rng.normal(0, 0.03, shape), axis=1))
    jump = rng.random(shape) < 0.02
    lon[jump & valid] += 3.0                                                        # outliers for the smoothing
    lon[tiny[:, None] & valid & (rng.random(shape) < 0.3)] += 3.0                   # short tracks: a jump, a gap
    lon[tiny[:, None] & valid & (rng.random(shape) < 0.3)] = np.nan
    lat[tiny[:, None] & valid & (rng.random(shape) < 0.3)] = np.nan
    cloudnumber = field(rng.integers(1, 500, shape).astype(np.int32))
    track_status = field(rng.integers(0, 40, shape).astype(np.int32))
    ncells = field(rng.integers(0, 5, shape).astype(np.int32))
    # conv_diameter_sum from its own random stream (the streams of the other variables stay as they were): 140 km all the time on
    # every second MCS-sized track, 40 km on the others (a long feature with small cells), so that the two MCS size tests differ
    diameter_rng = np.random.default_rng(seed + 1000)
    conv_diameter_sum = field(diameter_rng.uniform(0, 160, shape))
    cells_big = is_big & (np.arange(ntracks) % 2 == 1)
    conv_diameter_sum[is_big] = np.where(valid[is_big], 40.0, 0.0)
    conv_diameter_sum[cells_big] = np.where(valid[cells_big], 140.0, 0.0)
    sparse_values = {"majoraxis": majoraxis, "area_45dbz": random_f(0, 4000), "base_time": base_time, "ncells": ncells,
                     "conv_area": conv_area, "strat_area": strat_area, "cloudnumber": cloudnumber,
                     "track_status": track_status, "area": area, "meanlat": lat, "meanlon": lon,
                     "conv_diameter_sum": conv_diameter_sum}

    # Sparse entries: the valid ones, except a few that are left out
    rows, cols = np.nonzero(valid)
    keep = np.ones(len(rows), dtype=bool)
    keep[rng.choice(len(rows), size=nmissing, replace=False)] = False
    rows, cols = rows[keep], cols[keep]
    variables = {name: (["sparse_index"], arr[rows, cols]) for name, arr in sparse_values.items()}
    variables["tracks_indices"] = (["sparse_index"], rows.astype(np.int64))
    variables["times_indices"] = (["sparse_index"], cols.astype(np.int64))
    variables["track_duration"] = (["tracks"], duration, {"long_name": "duration"})
    variables["end_merge_tracknumber"] = (["tracks"], merge_into)
    variables["start_split_tracknumber"] = (["tracks"], split_from)
    variables["start_basetime"] = (["tracks"], t0 + start * dt, {"units": "seconds since 1970-01-01"})
    ds = xr.Dataset(variables, coords={"tracks": np.arange(ntracks), "sparse_index": np.arange(len(rows))},
                    attrs={"Title": "random", "n_files": np.int32(5)})
    for name in ("area", "majoraxis", "cloudnumber"):
        ds[name].attrs["long_name"] = name
    stats_path = str(tmp_path) + "/"
    ds.to_netcdf(os.path.join(stats_path, "trackstats_sparse_20250101.0000_20250101.1200.nc"))
    return stats_path


def run_both(stats_path, **overrides):
    """
    Run the dense and the rows step on the same sparse file.

    Args:
        stats_path: string
            Folder with the sparse trackstats file.
        **overrides:
            Config keys to change (nmaxlinks, finalstats_min_area, ...).

    Returns:
        dense_tables, rows_tables: RowTables
            dense_to_rows of the dense file, and the tables the rows step wrote.
    """
    config = dict(_make_driver_config(stats_path), **overrides)
    dense_file = identifymcs_radar(config)
    rows_folder = identifymcs_radar_rows(config)
    assert rows_folder.endswith("mcs_tracks_20250101.0000_20250101.1200.rows") and os.path.isdir(rows_folder)
    return dense_to_rows(dense_file), read_tables(rows_folder)


def test_builder_makes_the_tables_of_dense_to_rows(tmp_path):
    want = dense_to_rows(_write(tmp_path, _dense()))
    meta = want.meta
    builder = TableBuilder(want.duration, meta["dims"]["times"], nslots=meta["dims"]["mergers"], global_attrs=meta["global_attrs"])
    for name in meta["vars_1d"]:
        if name != "tracks":
            builder.add_track_var(name, want.tracks[name], meta["attrs"][name])
    for name, ncol in meta["vars_tracks2d"].items():
        builder.add_track2d_var(name, np.stack([want.tracks[f"{name}__{j}"] for j in range(ncol)], axis=1),
                                meta["dims_of"][name][1], meta["attrs"][name])
    for name, c in meta["coords"].items():
        if name not in ("times", "mergers"):
            builder.add_coord(name, np.asarray(c["values"]), meta["attrs"][name], c["dims"])
    for name in meta["vars_2d"]:
        builder.add_entry_var(name, want.entries[name], meta["attrs"][name], fill=meta["fill"][name])
    columns = {c: v[::-1] for c, v in want.fragments.items()}                      # any order: build() sorts
    builder.add_fragment_vars(columns, {n: (meta["attrs"][n], meta["fill"][n]) for n in meta["vars_3d"]})
    assert table_differences(want, builder.build()) == []


def test_table_differences_names_what_differs(tmp_path):
    a = dense_to_rows(_write(tmp_path, _dense()))
    b = dense_to_rows(_write(tmp_path, _dense()))
    assert table_differences(a, b) == []
    b.entries["area"] = b.entries["area"] + 1
    b.tracks["is_mcs"] = b.tracks["is_mcs"].astype(np.int32)
    b.meta["attrs"]["area"] = dict(b.meta["attrs"]["area"], units="m^2")
    b.meta["global_attrs"]["Created_on"] = "another time"
    got = table_differences(a, b)
    assert "entries.area: values differ" in got and "tracks.is_mcs: dtype int16 / int32" in got
    assert any(g.startswith("meta.attrs: differs in ['area']") for g in got) and len(got) == 3


def test_gather_index_selects_whole_tracks_in_the_given_order():
    offsets = np.array([0, 3, 3, 7, 9])
    index, new_offsets = gather_index(offsets, [3, 0, 2])
    assert index.tolist() == [7, 8, 0, 1, 2, 3, 4, 5, 6] and new_offsets.tolist() == [0, 2, 5, 9]
    index, new_offsets = gather_index(offsets, [])
    assert len(index) == 0 and new_offsets.tolist() == [0]


def test_flat_entries_are_the_valid_part_of_the_sparse_matrices(tmp_path):
    stats_path = random_sparse_file(tmp_path, seed=3)
    file = stats_path + "trackstats_sparse_20250101.0000_20250101.1200.nc"
    ds_1d, flat, attrs, offsets, longest = load_sparse_entries(file)
    _, _, sparse, width = load_sparse_trackstats(100, file, "times_indices", "tracks", "tracks_indices")
    duration = ds_1d["track_duration"].values
    valid = np.arange(sparse["area"].shape[1])[None, :] < duration[:, None]
    assert offsets[-1] == duration.sum() and longest == duration.max()
    for name in flat:
        assert flat[name].dtype == sparse[name].dtype and np.array_equal(flat[name], sparse[name].toarray()[valid], equal_nan=True), name
        assert _same_attrs(attrs[name], xr.open_dataset(file, mask_and_scale=False, decode_times=False)[name].attrs), name
    assert "tracks_indices" not in flat and "times_indices" not in flat


def test_entries_outside_the_duration_or_twice_are_refused(tmp_path):
    stats_path = random_sparse_file(tmp_path, seed=3)
    file = stats_path + "trackstats_sparse_20250101.0000_20250101.1200.nc"
    ds = xr.open_dataset(file, mask_and_scale=False, decode_times=False).load()
    bad = ds.copy(deep=True)
    bad["track_duration"] = bad["track_duration"] - 1                              # entries now lie behind the duration
    bad.to_netcdf(stats_path + "outside.nc")
    with pytest.raises(ValueError, match="cannot be stored as rows"):
        load_sparse_entries(stats_path + "outside.nc")
    dup = ds.copy(deep=True)
    dup["times_indices"] = dup["times_indices"].copy()
    dup["times_indices"].values[1] = dup["times_indices"].values[0]
    dup["tracks_indices"].values[1] = dup["tracks_indices"].values[0]
    dup.to_netcdf(stats_path + "twice.nc")
    with pytest.raises(ValueError, match="more than once"):
        load_sparse_entries(stats_path + "twice.nc")


def test_rows_step_equals_dense_step_on_the_three_track_file(tmp_path):
    stats_path = _build_sparse_trackstats_file(tmp_path)
    dense, rows = run_both(stats_path)
    assert table_differences(dense, rows) == []
    assert len(rows.duration) == 2 and rows.fragments["cloudnumber"].tolist() == [101, 102]
    # the MCS number is a track variable of the rows (equal to the dense file by table_differences above)
    assert sorted(rows.tracks["mcs_number"].tolist()) == [0, 1] and rows.tracks["is_mcs"].tolist() == [int(n > 0) for n in rows.tracks["mcs_number"]]


@pytest.mark.parametrize("seed", range(12))
def test_rows_step_equals_dense_step_on_random_tracks(tmp_path, seed):
    stats_path = random_sparse_file(tmp_path, seed)
    dense, rows = run_both(stats_path, nmaxlinks=1 + seed % 4)                    # a full slot dimension in some of the files
    assert table_differences(dense, rows) == []
    assert len(rows.fragments["track"]) > 0 and rows.tracks["is_mcs"].any()


@pytest.mark.parametrize("seed", range(6))
def test_rows_step_equals_dense_step_with_the_conv_diameter_sum_test(tmp_path, seed):
    """mcs_size_method conv_diameter_sum: the same tracks are MCS in both modes, and not those of the major-axis test."""
    stats_path = random_sparse_file(tmp_path, seed)
    overrides = dict(mcs_size_method="conv_diameter_sum", mcs_size_thresh=100, nmaxlinks=1 + seed % 4)
    dense, rows = run_both(stats_path, **overrides)
    assert table_differences(dense, rows) == []
    major_axis_rows = read_tables(identifymcs_radar_rows(dict(_make_driver_config(stats_path))))
    n_new, n_old = rows.tracks["is_mcs"].sum(), major_axis_rows.tracks["is_mcs"].sum()
    assert 0 < n_new <= n_old
    if seed == 0:                                   # this file has MCS-sized tracks of both kinds
        assert n_new < n_old


@pytest.mark.parametrize("seed", range(5))
def test_rows_step_equals_dense_step_with_tracks_of_one_and_two_entries(tmp_path, seed):
    stats_path = random_sparse_file(tmp_path, seed, ntiny=40)
    dense, rows = run_both(stats_path)
    assert table_differences(dense, rows) == []
    assert (rows.duration <= 2).sum() >= 30
    assert np.isnan(rows.entries["meanlon_smooth"]).any() and np.isnan(rows.entries["meanlon"]).any()


@pytest.mark.parametrize("seed", [1, 5, 8])
def test_rows_step_equals_dense_step_with_the_final_statistics_filters(tmp_path, seed):
    stats_path = random_sparse_file(tmp_path, seed)
    dense, rows = run_both(stats_path, finalstats_min_area=8000, finalstats_min_duration=0.9)
    assert table_differences(dense, rows) == []
    unfiltered = identifymcs_radar_rows(dict(_make_driver_config(stats_path)))
    assert len(rows.duration) < len(read_tables(unfiltered).duration)


def test_rows_back_to_dense_equal_the_dense_file(tmp_path):
    stats_path = random_sparse_file(tmp_path, 4)
    config = _make_driver_config(stats_path)
    dense_file, rows_folder = identifymcs_radar(config), identifymcs_radar_rows(config)
    want = xr.open_dataset(dense_file, mask_and_scale=False, decode_times=False)
    got = RowStats.open(rows_folder).to_xarray()
    assert set(got.variables) == set(want.variables)
    for name in want.variables:
        assert got[name].dims == want[name].dims and got[name].dtype == want[name].dtype, name
        assert np.array_equal(got[name].values, want[name].values, equal_nan=want[name].dtype.kind == "f"), name
    assert got.sizes["times"] == want.sizes["times"] == 100


def test_a_track_longer_than_duration_range_does_not_stop_the_rows_step(tmp_path):
    stats_path = random_sparse_file(tmp_path, 2)
    longest = int(xr.open_dataset(stats_path + "trackstats_sparse_20250101.0000_20250101.1200.nc")["times_indices"].max()) + 1
    rows = read_tables(identifymcs_radar_rows(_make_driver_config(stats_path) | {"duration_range": [2, 10]}))
    assert longest > 10 and rows.meta["dims"]["times"] == int(np.ceil(longest / 10) * 10)


def test_no_tracks_give_empty_tables(tmp_path):
    ds = xr.Dataset(
        {"track_duration": (["tracks"], np.zeros(0, np.int32)), "end_merge_tracknumber": (["tracks"], np.zeros(0)),
         "start_split_tracknumber": (["tracks"], np.zeros(0)), "tracks_indices": (["sparse_index"], np.zeros(0, np.int64)),
         "times_indices": (["sparse_index"], np.zeros(0, np.int64)),
         **{n: (["sparse_index"], np.zeros(0)) for n in ("majoraxis", "area_45dbz", "base_time", "ncells", "conv_area", "strat_area",
                                                         "cloudnumber", "track_status", "area", "meanlat", "meanlon")}},
        coords={"tracks": np.zeros(0, np.int64), "sparse_index": np.zeros(0, np.int64)})
    ds.to_netcdf(str(tmp_path) + "/trackstats_sparse_20250101.0000_20250101.1200.nc", unlimited_dims=["tracks", "sparse_index"])
    rows = read_tables(identifymcs_radar_rows(_make_driver_config(str(tmp_path) + "/")))
    assert len(rows.duration) == 0 and len(rows.entries["track"]) == 0 and len(rows.fragments["track"]) == 0
    assert rows_to_dense(rows).sizes["tracks"] == 0
