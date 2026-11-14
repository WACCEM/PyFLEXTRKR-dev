"""
Tests of the small [tracks, times, n] variables of the row tables (pyflextrkr.rowstats): the precipitation-feature statistics of
the Tb+PF files (pf_area, pf_majoraxis, ... with n = nmaxpf) are stored as n columns name__0 ... name__{n-1} of the entries table
(meta "vars_entry3d"). The conversion dense -> rows -> dense stays exact, RowStats / load_trackstats / write_dense_netcdf /
TableBuilder / subset_tracks handle them, and files without such variables are unchanged. Synthetic data only.
"""
import numpy as np
import pytest
import xarray as xr

from pyflextrkr.rowstats import (RaggedArray, RowStats, TableBuilder, dense_to_rows, load_trackstats, read_tables, rows_to_dense,
                                 subset_tracks, table_differences, write_dense_netcdf, write_rows)
from test_rowstats import NTIMES, NTRACKS, _assert_same_dataset, _dense, _write

NPF = 3


def _dense_pf(seed=1):
    """
    The dense fixture of test_rowstats plus the statistics of the PF matching of a Tb+PF file: two [tracks, times, nmaxpf] float32
    variables (NaN outside the valid entries as fill, NaN inside them as data), one [tracks, times] int16 PF count and the nmaxpf
    coordinate.

    Args:
        seed: int (default: 1)
            Random seed.

    Returns:
        ds: xarray.Dataset
    """
    ds = _dense(seed)
    rng = np.random.default_rng(seed + 100)
    # The valid entries of every track (the first track_duration times)
    valid = np.arange(NTIMES)[None, :] < ds["track_duration"].values[:, None]

    def pf3d():
        """A [tracks, times, nmaxpf] float32 array: fill (NaN) outside the entries, some NaN as data inside them."""
        a = rng.random((NTRACKS, NTIMES, NPF)).astype(np.float32)
        a[~valid] = np.nan
        a[(rng.random(a.shape) < 0.2) & valid[:, :, None]] = np.nan
        return a

    pf_npf = np.where(valid, rng.integers(0, NPF + 1, (NTRACKS, NTIMES)), -9999).astype(np.int16)
    k3 = ["tracks", "times", "nmaxpf"]
    ds["pf_area"] = (k3, pf3d(), {"long_name": "PF area", "units": "km^2", "_FillValue": np.nan})
    ds["pf_majoraxis"] = (k3, pf3d(), {"long_name": "PF major axis", "units": "km", "_FillValue": np.nan})
    ds["pf_npf"] = (["tracks", "times"], pf_npf, {"_FillValue": np.int16(-9999)})
    return ds.assign_coords(nmaxpf=np.arange(NPF))


def _folder(tmp_path, ds=None):
    """
    Write a dense file and its row folder; return (dense path, row folder).
    """
    path = _write(tmp_path, _dense_pf() if ds is None else ds)
    folder = str(tmp_path / "pf.rows")
    write_rows(dense_to_rows(path), folder)
    return path, folder


def _raw(path):
    return xr.open_dataset(path, mask_and_scale=False, decode_times=False)


@pytest.mark.parametrize("block", [1, 3, 100])
def test_roundtrip_with_pf_variables_is_exact_and_independent_of_the_block_size(tmp_path, block):
    path = _write(tmp_path, _dense_pf())
    want = _raw(path)
    rt = dense_to_rows(path, block_tracks=block)
    _assert_same_dataset(rows_to_dense(rt), want)
    assert rt.meta["vars_entry3d"] == {"pf_area": NPF, "pf_majoraxis": NPF}
    nentries = int(want["track_duration"].sum())
    assert all(len(rt.entries[f"pf_area__{j}"]) == nentries for j in range(NPF)) and "pf_area" not in rt.entries
    assert "pf_npf" in rt.meta["vars_2d"] and rt.meta["dims_of"]["pf_area"] == ["tracks", "times", "nmaxpf"]


def test_a_file_without_pf_variables_has_no_entry3d_key(tmp_path):
    rt = dense_to_rows(_write(tmp_path, _dense()))
    assert "vars_entry3d" not in rt.meta
    folder = str(tmp_path / "plain.rows")
    write_rows(rt, folder)
    assert "vars_entry3d" not in RowStats.open(folder).tables().meta
    assert RowStats.open(folder)["area"].flat.ndim == 1                       # as before


def test_a_constant_fill_is_needed_outside_the_entries(tmp_path):
    ds = _dense_pf()
    bad = ds["pf_area"].values.copy()
    bad[1, NTIMES - 1, 0] = 5.0                  # track 1 is shorter than the full width in this fixture (or the value is data)
    ds["pf_area"] = (ds["pf_area"].dims, bad, ds["pf_area"].attrs)
    path = _write(tmp_path, ds)
    if ds["track_duration"].values[1] == NTIMES:
        pytest.skip("track 1 has the full width for this seed")
    with pytest.raises(ValueError, match="pf_area"):
        dense_to_rows(path)


def test_rowstats_gives_pf_variables_as_ragged_arrays_with_a_trailing_dimension(tmp_path):
    path, folder = _folder(tmp_path)
    want = _raw(path)
    duration = want["track_duration"].values
    rs = RowStats.open(folder)
    pf = rs["pf_area"]
    assert pf.flat.shape == (int(duration.sum()), NPF) and len(pf) == NTRACKS
    # One track: [duration, n]
    assert np.array_equal(pf[2], want["pf_area"].values[2, :duration[2], :], equal_nan=True)
    # The padded dense array, the reductions and a subset keep the trailing dimension
    assert np.array_equal(pf.dense(NTIMES), want["pf_area"].values, equal_nan=True)
    assert pf.max().shape == (NTRACKS, NPF) and pf.sum().shape == (NTRACKS, NPF) and pf.mean().shape == (NTRACKS, NPF)
    with np.errstate(invalid="ignore"):
        longest = [np.max(want["pf_area"].values[i, :duration[i], :], axis=0) for i in range(NTRACKS)]
    assert np.array_equal(pf.max(), np.array(longest), equal_nan=True)
    sub = pf[[3, 1]]
    assert isinstance(sub, RaggedArray) and np.array_equal(sub[0], pf[3], equal_nan=True) and sub.flat.shape[1] == NPF
    # to_xarray of a subset of the tracks and of one variable
    part = rs.to_xarray(tracks=[3, 1], variables=["pf_majoraxis"])
    assert "pf_majoraxis" in part and "pf_area" not in part and part["pf_majoraxis"].dims == ("tracks", "times", "nmaxpf")
    assert np.array_equal(part["pf_majoraxis"].values[0, :duration[3]], want["pf_majoraxis"].values[3, :duration[3]], equal_nan=True)


def test_load_trackstats_keeps_the_pf_variables_in_either_format_and_rowstats_loads_only_the_named_ones(tmp_path):
    path, folder = _folder(tmp_path)
    for source in (path, folder):
        # A list of [tracks, times] variables does not drop the small [tracks, times, n] ones (a dense file keeps them)
        part = load_trackstats(source, variables=["area"], mask_and_scale=False, decode_times=False)
        assert "area" in part and "pf_npf" not in part, source
        assert part["pf_area"].dims == ("tracks", "times", "nmaxpf") and "pf_majoraxis" in part, source
        everything = load_trackstats(source, mask_and_scale=False, decode_times=False)
        assert {"pf_area", "pf_majoraxis", "pf_npf"} <= set(everything.variables), source
    # the same values from both formats
    a = load_trackstats(path, variables=["area"], mask_and_scale=False, decode_times=False)
    b = load_trackstats(folder, variables=["area"], mask_and_scale=False, decode_times=False)
    xr.testing.assert_identical(a.drop_vars("tracks", errors="ignore"), b.drop_vars("tracks", errors="ignore"))
    # RowStats.tables reads only the columns of the variables that are named (the cheap path for a big file)
    lazy = RowStats.open(folder).tables(["area"])
    assert lazy.meta["vars_entry3d"] == {} and not any(c.startswith("pf_area") for c in lazy.entries)
    named = RowStats.open(folder).tables(["pf_majoraxis"])
    assert list(named.meta["vars_entry3d"]) == ["pf_majoraxis"] and "pf_majoraxis__2" in named.entries


@pytest.mark.parametrize("block", [None, 2])
def test_write_dense_netcdf_gives_back_the_file_with_pf_variables(tmp_path, block):
    path, folder = _folder(tmp_path)
    out = write_dense_netcdf(folder, outfile=str(tmp_path / "back.nc"), block_tracks=block)
    _assert_same_dataset(_raw(out), _raw(path))


def test_table_builder_makes_the_tables_of_dense_to_rows_with_pf_variables(tmp_path):
    want = dense_to_rows(_write(tmp_path, _dense_pf()))
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
    for name, ncol in meta["vars_entry3d"].items():
        values = np.stack([want.entries[f"{name}__{j}"] for j in range(ncol)], axis=1)
        builder.add_entry3d_var(name, values, "nmaxpf", meta["attrs"][name], fill=meta["fill"][name])
    columns = {c: v[::-1] for c, v in want.fragments.items()}                      # any order: build() sorts
    builder.add_fragment_vars(columns, {n: (meta["attrs"][n], meta["fill"][n]) for n in meta["vars_3d"]})
    assert table_differences(want, builder.build()) == []


def test_add_entry3d_var_checks_its_input_and_replaces_an_earlier_variable(tmp_path):
    rt = dense_to_rows(_write(tmp_path, _dense_pf()))
    builder = TableBuilder.from_tables(rt)
    nentries = int(rt.duration.sum())
    with pytest.raises(ValueError, match="expected shape"):
        builder.add_entry3d_var("pf_area", np.zeros(nentries), "nmaxpf", {"_FillValue": np.nan})
    with pytest.raises(ValueError, match="no fill value"):
        builder.add_entry3d_var("pf_area", np.zeros((nentries, 2)), "nmaxpf")
    # A variable with fewer elements replaces the old columns completely
    builder.add_entry3d_var("pf_area", np.zeros((nentries, 2), dtype=np.float32), "nmaxpf", {"_FillValue": np.nan})
    tables = builder.build()
    assert tables.meta["vars_entry3d"]["pf_area"] == 2 and "pf_area__2" not in tables.entries


def test_from_tables_keeps_the_pf_variables_and_a_new_variable_can_be_added(tmp_path):
    rt = dense_to_rows(_write(tmp_path, _dense_pf()))
    assert table_differences(rt, TableBuilder.from_tables(rt).build()) == []
    builder = TableBuilder.from_tables(rt)
    builder.add_track_var("pf_lifetime", np.arange(NTRACKS, dtype=np.float64), {"units": "hour"})
    out = builder.build()
    assert out.meta["vars_entry3d"] == rt.meta["vars_entry3d"] and "pf_lifetime" in out.tracks
    back = rows_to_dense(out)
    assert np.array_equal(back["pf_area"].values, rows_to_dense(rt)["pf_area"].values, equal_nan=True)


def test_subset_tracks_equals_the_selection_of_the_dense_file(tmp_path):
    ds = _dense_pf()
    path = _write(tmp_path, ds)
    rt = dense_to_rows(path)
    keep = [5, 0, 3]
    # The dense selection with a new tracks coordinate (what the robust-MCS step does with the PF statistics)
    dense_sub = ds.isel(tracks=keep)
    dense_sub["tracks"] = np.arange(len(keep))
    sub_path = str(tmp_path / "sub.nc")
    dense_sub.to_netcdf(sub_path, format="NETCDF4", unlimited_dims="tracks")
    assert table_differences(dense_to_rows(sub_path), subset_tracks(rt, keep)) == []
    # No tracks at all: empty tables of the right size
    empty = subset_tracks(rt, [])
    assert len(empty.duration) == 0 and empty.meta["dims"]["tracks"] == 0 and all(len(v) == 0 for v in empty.entries.values())
    assert all(len(v) == 0 for v in empty.fragments.values())


def test_ragged_array_with_a_trailing_dimension_and_empty_tracks():
    flat = np.arange(12, dtype=np.float64).reshape(6, 2)
    offsets = np.array([0, 2, 2, 6])                                       # lengths 2, 0, 4
    ra = RaggedArray(flat, offsets, fill=np.nan)
    assert ra.max().shape == (3, 2) and np.isnan(ra.max()[1]).all() and np.array_equal(ra.max()[0], [2, 3])
    assert np.array_equal(ra.sum()[2], flat[2:].sum(axis=0)) and np.allclose(ra.mean()[2], flat[2:].mean(axis=0))
    dense = ra.dense(5)
    assert dense.shape == (3, 5, 2) and np.isnan(dense[1]).all() and np.array_equal(dense[2, :4], flat[2:])
