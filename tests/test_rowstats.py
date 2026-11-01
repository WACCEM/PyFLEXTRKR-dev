"""
Tests of pyflextrkr.rowstats (row-based storage of the dense track statistics): the conversion dense -> rows -> dense is exact
(values, dtypes, fill values, attributes), independent of the block size, a subset of tracks can be rebuilt, and RaggedArray
gives the access of the dense array. Synthetic data only.
"""
import os

import numpy as np
import pytest
import xarray as xr

from pyflextrkr.rowstats import RaggedArray, RowStats, dense_to_rows, rows_to_dense, write_rows

NTRACKS, NTIMES, NSLOTS = 7, 6, 4


def _dense(seed=1):
    """
    A small dense stats dataset like the robust file: 1-D variables, [tracks, times] variables (fill after the end of each
    track, NaN also inside a track as data), [tracks, times, mergers] merge / split fragment variables, a [tracks, lifestages]
    variable, coordinates and attributes.
    """
    rng = np.random.default_rng(seed)
    dur = rng.integers(1, NTIMES + 1, NTRACKS).astype(np.int32)
    dur[0] = NTIMES                                                   # one track with the full width
    valid = np.arange(NTIMES)[None, :] < dur[:, None]

    def float2d(dtype):
        a = rng.random((NTRACKS, NTIMES)).astype(dtype)
        a[~valid] = np.nan
        return a

    area = float2d(np.float32)
    area[1, 0] = np.nan                                               # NaN inside the valid entries is data, not fill
    cloudnumber = np.where(valid, rng.integers(1, 100, (NTRACKS, NTIMES)), -9999).astype(np.int32)
    ints = lambda: np.full((NTRACKS, NTIMES, NSLOTS), -9999, np.int32)
    floats = lambda: np.full((NTRACKS, NTIMES, NSLOTS), np.nan, np.float32)
    merge_cn, merge_st, merge_ar, split_cn, split_st, split_ar = ints(), ints(), floats(), ints(), ints(), floats()
    merge_cn[2, 1, :2], merge_st[2, 1, :2], merge_ar[2, 1, :2] = [5, 6], [3, 4], [1.5, np.nan]
    merge_cn[5, 0, 0], merge_st[5, 0, 0], merge_ar[5, 0, 0] = 8, 1, 0.25
    split_cn[0, 2, 0], split_st[0, 2, 0], split_ar[0, 2, 0] = 9, 7, 2.5
    life = np.full((NTRACKS, 5), -9999, np.int32)
    life[:3] = rng.integers(0, 9, (3, 5))
    fill_i, fill_f = {"_FillValue": -9999}, {"_FillValue": np.nan}
    t, k, m = ["tracks"], ["tracks", "times"], ["tracks", "times", "mergers"]
    return xr.Dataset(
        {
            "track_duration": (t, dur, {"long_name": "duration"}),
            "is_mcs": (t, (dur > 3).astype(np.int16), {}),
            "area": (k, area, {"units": "km^2", **fill_f}),
            "cloudnumber": (k, cloudnumber, fill_i),
            "base_time": (k, float2d(np.float64), fill_f),
            "merge_cloudnumber": (m, merge_cn, fill_i), "merge_status": (m, merge_st, fill_i), "merge_area": (m, merge_ar, fill_f),
            "split_cloudnumber": (m, split_cn, fill_i), "split_status": (m, split_st, fill_i), "split_area": (m, split_ar, fill_f),
            "lifecycle_index": (["tracks", "lifestages"], life, fill_i),
        },
        coords={"tracks": np.arange(NTRACKS), "times": np.arange(NTIMES), "mergers": np.arange(NSLOTS),
                "lifestages": np.arange(5)},
        attrs={"Title": "test", "n_pairs": np.int32(3), "classes": np.array([1, 2], dtype=np.int32)},
    )


def _write(tmp_path, ds):
    path = str(tmp_path / "dense.nc")
    ds.to_netcdf(path, format="NETCDF4", unlimited_dims="tracks")
    return path


def _same_attrs(a, b):
    """
    Attribute dictionaries equal (arrays by value, NaN equal NaN).
    """
    if set(a) != set(b):
        return False
    return all(np.array_equal(np.asarray(a[k]), np.asarray(b[k]), equal_nan=np.asarray(a[k]).dtype.kind == "f") for k in a)


def _assert_same_dataset(got, want):
    assert set(got.variables) == set(want.variables)
    for n in want.variables:
        assert got[n].dims == want[n].dims, n
        assert got[n].dtype == want[n].dtype, n
        assert np.array_equal(got[n].values, want[n].values, equal_nan=want[n].dtype.kind == "f"), n
        assert _same_attrs(got[n].attrs, want[n].attrs), n
    assert _same_attrs(got.attrs, want.attrs)


def test_roundtrip_is_exact_and_independent_of_the_block_size(tmp_path):
    path = _write(tmp_path, _dense())
    want = xr.open_dataset(path, mask_and_scale=False, decode_times=False)
    first = dense_to_rows(path, block_tracks=3)
    _assert_same_dataset(rows_to_dense(first), want)
    other = dense_to_rows(path, block_tracks=100)
    for table in ("tracks", "entries", "fragments"):
        a, b = getattr(first, table), getattr(other, table)
        assert list(a) == list(b)
        for column in a:
            assert np.array_equal(a[column], b[column], equal_nan=a[column].dtype.kind == "f"), (table, column)
    # sizes: valid entries and fragments only
    assert len(first.entries["track"]) == int(want["track_duration"].sum())
    assert len(first.fragments["track"]) == 4 and list(first.fragments["kind"]) == [1, 0, 0, 0]


def test_fragments_table_content_and_order(tmp_path):
    rt = dense_to_rows(_write(tmp_path, _dense()))
    f = rt.fragments
    # sorted by track, k, kind (0 merge, 1 split), slot
    rows = list(zip(f["track"], f["k"], f["kind"], f["slot"], f["cloudnumber"], f["status"]))
    assert rows == [(0, 2, 1, 0, 9, 7), (2, 1, 0, 0, 5, 3), (2, 1, 0, 1, 6, 4), (5, 0, 0, 0, 8, 1)]
    assert np.isnan(f["area"][2]) and f["area"][3] == np.float32(0.25)


def test_parquet_folder_ragged_access_and_subset(tmp_path):
    pytest.importorskip("pyarrow")
    ds = _dense()
    path = _write(tmp_path, ds)
    write_rows(dense_to_rows(path), str(tmp_path / "rows"))
    rs = RowStats.open(str(tmp_path / "rows"))
    dur = ds["track_duration"].values
    # dense-like access
    area = rs["area"]
    assert len(area) == NTRACKS and np.array_equal(area.lengths, dur)
    for i in range(NTRACKS):
        assert np.array_equal(area[i], ds["area"].values[i, :dur[i]], equal_nan=True)
    assert area[0, 2] == ds["area"].values[0, 2]
    with np.errstate(all="ignore"):
        want_max = np.array([np.max(ds["area"].values[i, :dur[i]]) for i in range(NTRACKS)])
    assert np.array_equal(area.max(), want_max, equal_nan=True)
    assert np.array_equal(area.dense(NTIMES), ds["area"].values, equal_nan=True)
    assert np.array_equal(rs["is_mcs"], ds["is_mcs"].values)
    # a subset of tracks, in the order asked for, rebuilt as a dense dataset
    sub = rs.to_xarray(tracks=[4, 1, 0])
    width = int(dur[[4, 1, 0]].max())
    want = ds.isel(tracks=[4, 1, 0], times=slice(0, width))
    for n in ("area", "cloudnumber", "base_time", "track_duration", "merge_cloudnumber", "merge_area", "split_status", "lifecycle_index"):
        assert np.array_equal(sub[n].values, want[n].values, equal_nan=want[n].dtype.kind == "f"), n
        assert sub[n].dtype == want[n].dtype
    assert list(sub["tracks"].values) == [4, 1, 0] and sub.sizes["times"] == width
    # only the asked [tracks, times] variables are loaded
    assert "base_time" not in rs.to_xarray(variables=["area"])


def test_ragged_array_subsets_and_empty_tracks():
    flat = np.array([3.0, 1.0, 2.0, 7.0, 5.0])
    ra = RaggedArray(flat, [0, 2, 2, 5, 5])                           # tracks of length 2, 0, 3, 0
    assert list(ra.lengths) == [2, 0, 3, 0]
    mx = ra.max()
    assert mx[0] == 3.0 and np.isnan(mx[1]) and mx[2] == 7.0 and np.isnan(mx[3])
    assert ra.sum()[2] == 14.0 and ra.min()[0] == 1.0 and ra.mean()[0] == 2.0 and np.isnan(ra.mean()[1])
    sub = ra[[2, 0]]
    assert list(sub.lengths) == [3, 2] and list(sub[0]) == [2.0, 7.0, 5.0]
    assert list(ra[np.array([True, False, True, False])].lengths) == [2, 3]
    dense = ra.dense()
    assert dense.shape == (4, 3) and np.isnan(dense[1]).all() and list(dense[0, :2]) == [3.0, 1.0]
    assert ra[-1].shape == (0,) and ra[0, 1] == 1.0


def test_fill_that_is_not_constant_is_refused(tmp_path):
    ds = _dense()
    area = ds["area"].values.copy()
    # one value after the end of a track that is not the fill value (NaN)
    outside = np.argwhere(np.arange(NTIMES)[None, :] >= ds["track_duration"].values[:, None])
    assert len(outside) > 0
    i, j = outside[0]
    area[i, j] = 12.0
    ds["area"] = (["tracks", "times"], area, ds["area"].attrs)
    with pytest.raises(ValueError, match="constant fill"):
        dense_to_rows(_write(tmp_path, ds))


def test_load_trackstats_reads_both_formats_alike(tmp_path):
    from pyflextrkr.rowstats import load_trackstats
    path = _write(tmp_path, _dense())
    write_rows(dense_to_rows(path), str(tmp_path / "dense.rows"))
    want = xr.open_dataset(path)                                           # the defaults of xarray: decoded
    for given in (path, str(tmp_path / "dense.rows")):
        xr.testing.assert_identical(load_trackstats(given), want)
    # the .nc name of a file that only exists as a row folder
    os.remove(path)
    xr.testing.assert_identical(load_trackstats(path), want)
    # raw contents, a subset of tracks (the width follows the subset for rows), a subset of variables
    raw = load_trackstats(str(tmp_path / "dense.rows"), mask_and_scale=False, decode_times=False)
    assert raw["cloudnumber"].dtype == np.int32 and raw["cloudnumber"].attrs["_FillValue"] == -9999
    masked = load_trackstats(str(tmp_path / "dense.rows"), mask_and_scale=True, decode_times=False)   # as the consumers of the files do
    assert masked["cloudnumber"].dtype == np.float64 and np.isnan(masked["cloudnumber"].values).any()
    sub = load_trackstats(str(tmp_path / "dense.rows"), tracks=[5, 2], variables=["area"])
    assert sub.sizes["tracks"] == 2 and "cloudnumber" not in sub and "area" in sub and "merge_cloudnumber" in sub
    np.testing.assert_array_equal(sub["track_duration"].values, want["track_duration"].values[[5, 2]])


def test_load_trackstats_without_fragments_gives_the_per_track_variables_in_either_format(tmp_path):
    from pyflextrkr.rowstats import load_trackstats
    path = _write(tmp_path, _dense())
    write_rows(dense_to_rows(path), str(tmp_path / "dense.rows"))
    for given in (path, str(tmp_path / "dense.rows")):
        first = load_trackstats(given, variables=[], fragments=False)
        assert set(first.data_vars) == {"track_duration", "is_mcs", "lifecycle_index"}          # no [tracks, times] and no slot variables
        with_slots = load_trackstats(given, variables=["area"], fragments=False)
        assert set(with_slots.data_vars) == {"track_duration", "is_mcs", "lifecycle_index", "area"}
        full = load_trackstats(given, variables=[])
        assert "merge_area" in full and full["merge_area"].dims == ("tracks", "times", "mergers")
    a, b = (load_trackstats(g, variables=[], fragments=False) for g in (path, str(tmp_path / "dense.rows")))
    xr.testing.assert_identical(a, b)
    # a dense file with another three-dimensional variable (Tb+PF: [tracks, times, nmaxpf]) keeps it
    tbpf = _dense()
    tbpf["pf_area"] = (["tracks", "times", "nmaxpf"], np.zeros((NTRACKS, NTIMES, 2), dtype=np.float32))
    other = str(tmp_path / "tbpf.nc")
    tbpf.to_netcdf(other, format="NETCDF4", unlimited_dims="tracks")
    kept = load_trackstats(other, variables=["area"], fragments=False)
    assert "pf_area" in kept and "merge_area" not in kept


def test_stats_exists_finds_either_format(tmp_path):
    from pyflextrkr.rowstats import stats_exists
    path = _write(tmp_path, _dense())
    assert stats_exists(path) and not stats_exists(str(tmp_path / "other.nc"))
    write_rows(dense_to_rows(path), str(tmp_path / "dense.rows"))
    assert stats_exists(str(tmp_path / "dense.rows"))
    os.remove(path)
    assert stats_exists(path)                                              # the .nc name of a file that only exists as a row folder
    assert not stats_exists(str(tmp_path / "other.nc")) and not stats_exists(str(tmp_path / "missing.rows"))


def _rows_in_folder(tmp_path, ds=None):
    """
    A dense file (dense.nc) and its row folder under the name of a step output (r/mcs_tracks_a_b.rows, so that the default dense name
    r/mcs_tracks_a_b.nc is free).
    """
    path = _write(tmp_path, _dense() if ds is None else ds)
    folder = tmp_path / "r"
    folder.mkdir()
    rows = str(folder / "mcs_tracks_a_b.rows")
    write_rows(dense_to_rows(path), rows)
    return path, rows


def _raw(path):
    return xr.open_dataset(path, mask_and_scale=False, decode_times=False)


def test_write_dense_netcdf_gives_back_the_dense_file_under_the_old_name(tmp_path):
    from pyflextrkr.rowstats import dense_netcdf_name, write_dense_netcdf
    path, rows = _rows_in_folder(tmp_path)
    assert dense_netcdf_name(rows) == rows[:-5] + ".nc" and dense_netcdf_name("a/b.nc") == "a/b.nc"
    out = write_dense_netcdf(rows)
    assert out == rows[:-5] + ".nc"                                         # mcs_tracks_a_b.rows -> mcs_tracks_a_b.nc, next to it
    _assert_same_dataset(_raw(out), _raw(path))                              # every variable, dtype, fill value and attribute
    import netCDF4
    with netCDF4.Dataset(out) as nc:
        assert nc.dimensions["tracks"].isunlimited() and nc["area"].filters()["zlib"]
    with pytest.raises(FileExistsError):
        write_dense_netcdf(rows)                                             # an existing file is not replaced silently
    assert write_dense_netcdf(rows, overwrite=True) == out


@pytest.mark.parametrize("block", [1, 2, 3, 7, 100])
def test_write_dense_netcdf_does_not_depend_on_the_block_size(tmp_path, block):
    from pyflextrkr.rowstats import write_dense_netcdf
    path, rows = _rows_in_folder(tmp_path)
    out = write_dense_netcdf(rows, str(tmp_path / f"blocks_{block}.nc"), block_tracks=block)
    _assert_same_dataset(_raw(out), _raw(path))


def test_write_dense_netcdf_selection_and_the_nc_name_of_a_rows_only_file(tmp_path):
    from pyflextrkr.rowstats import load_trackstats, write_dense_netcdf
    path, rows = _rows_in_folder(tmp_path)
    # the .nc name of a statistics file that exists only as a row folder (no .nc file next to it yet)
    one = write_dense_netcdf(rows[:-5] + ".nc", str(tmp_path / "one.nc"), tracks=[5, 2, 0], variables=["area"])
    blocks = write_dense_netcdf(rows, str(tmp_path / "blocks.nc"), tracks=[5, 2, 0], variables=["area"], block_tracks=2)
    _assert_same_dataset(_raw(blocks), _raw(one))
    want = load_trackstats(rows, tracks=[5, 2, 0], variables=["area"], mask_and_scale=False, decode_times=False)
    _assert_same_dataset(_raw(one), want)
    assert _raw(one).sizes["tracks"] == 3 and "cloudnumber" not in _raw(one) and "area" in _raw(one)
    with pytest.raises(FileNotFoundError):
        write_dense_netcdf(str(tmp_path / "missing.rows"))


def test_convert_stats_to_dense_without_a_target_uses_the_old_name(tmp_path):
    convert = _convert_stats()
    path, rows = _rows_in_folder(tmp_path)
    assert convert.main(["to-dense", rows, "--block-tracks", "3"]) == 0
    _assert_same_dataset(_raw(rows[:-5] + ".nc"), _raw(path))
    with pytest.raises(SystemExit):                                          # exists now: an error without --overwrite
        convert.main(["to-dense", rows])
    assert convert.main(["to-dense", rows, "--overwrite"]) == 0
    assert convert.main(["to-dense", rows[:-5] + ".nc", str(tmp_path / "from_nc_name.nc")]) == 0     # the .nc name of the folder works too


def _convert_stats():
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runscripts", "convert_stats.py")
    spec = importlib.util.spec_from_file_location("convert_stats", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_convert_stats_goes_both_ways_and_selects_tracks(tmp_path):
    convert = _convert_stats()
    path = _write(tmp_path, _dense())
    rows, back, subset = str(tmp_path / "x.rows"), str(tmp_path / "back.nc"), str(tmp_path / "subset.nc")
    assert convert.main(["to-rows", path, rows, "--block-tracks", "3"]) == 0
    assert convert.main(["to-dense", rows, back]) == 0
    _assert_same_dataset(xr.open_dataset(back, mask_and_scale=False, decode_times=False),
                         xr.open_dataset(path, mask_and_scale=False, decode_times=False))
    assert convert.main(["to-dense", rows, subset, "--select", "is_mcs=1", "--variables", "area"]) == 0
    sub = xr.open_dataset(subset, mask_and_scale=False, decode_times=False)
    want = np.nonzero(xr.open_dataset(path)["is_mcs"].values == 1)[0]
    assert sub["tracks"].values.tolist() == want.tolist() and "cloudnumber" not in sub and "area" in sub
    with pytest.raises(SystemExit):
        convert.main(["to-dense", rows, subset, "--select", "not_a_variable=1"])
