"""
Tests of docs/rowstats_pandas.md (the Xarray -> Pandas cheat sheet and recipes for the row tables): the ```python blocks run in order on a synthetic
Tb+PF row folder (the blocks marked ```py need data of a real run and are not run), and every command of the cheat sheet is evaluated next to its
Xarray form on the dense file that the rows were made from and the two results are compared, so that the page cannot go stale or wrong. The
expressions are written here as they are in the page: a test fails if the page does not contain them.
"""
import os
import re

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pyflextrkr.rowstats import dense_to_rows, write_rows

DOC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "rowstats_pandas.md")
NTRACKS, NTIMES, NPF, NSLOTS = 14, 12, 3, 4


def python_blocks(path):
    """
    The runnable code blocks of a markdown file.

    Args:
        path: string
            Markdown file.

    Returns:
        blocks: list of strings
            The code of each ```python block, in order (a ```py block shows code that needs the files of a real run and is not run).
    """
    return re.findall(r"```python\n(.*?)```", open(path, encoding="utf-8").read(), flags=re.S)


def tbpf_dense(seed=3):
    """
    A small dense statistics file like the final Tb+PF file: 1-D variables, [tracks, times] variables (NaN or -9999 after the end of a track, NaN as data
    inside it), [tracks, times, nmaxpf] PF variables, [tracks, times, mergers] merge / split variables, and the attributes of the real file.

    Args:
        seed: int (default: 3)
            Random seed.

    Returns:
        ds: xarray.Dataset
            With the epoch times as seconds (the units of the real file make Xarray decode them).
    """
    rng = np.random.default_rng(seed)
    dur = rng.integers(2, NTIMES + 1, NTRACKS).astype(np.int32)
    dur[0], dur[1], dur[2] = NTIMES, 1, 2                                  # a full-width track, a one-step track, a short one
    valid = np.arange(NTIMES)[None, :] < dur[:, None]
    start = 1546300800.0 + np.sort(rng.integers(0, 20 * 86400, NTRACKS)).astype(np.float64) // 3600 * 3600
    fill_f, fill_i = {"_FillValue": np.nan}, {"_FillValue": -9999}

    def floats(low, high, dtype=np.float32, nan_inside=0.0):
        """A [tracks, times] float array: random values in the entries, NaN elsewhere (and some NaN as data inside the entries after the first step)."""
        a = rng.uniform(low, high, (NTRACKS, NTIMES)).astype(dtype)
        a[(rng.random(a.shape) < nan_inside) & (np.arange(NTIMES)[None, :] > 0)] = np.nan
        a[~valid] = np.nan
        return a

    base_time = np.where(valid, start[:, None] + 3600.0 * np.arange(NTIMES)[None, :], np.nan)
    pf = lambda: np.where(valid[:, :, None] & (rng.random((NTRACKS, NTIMES, NPF)) < 0.7), rng.uniform(10, 500, (NTRACKS, NTIMES, NPF)), np.nan).astype(np.float32)
    ints3 = lambda: np.full((NTRACKS, NTIMES, NSLOTS), -9999, np.int32)
    merge_cn, split_cn = ints3(), ints3()
    merge_ar, split_ar = np.full(merge_cn.shape, np.nan, np.float32), np.full(merge_cn.shape, np.nan, np.float32)
    merge_cn[0, 3, :2], merge_ar[0, 3, :2] = [11, 12], [100.0, 200.0]
    merge_cn[4, 1, 0], merge_ar[4, 1, 0] = 13, 50.0
    split_cn[0, 5, 0], split_ar[0, 5, 0] = 14, 75.0
    split_cn[6, 0, :3], split_ar[6, 0, :3] = [15, 16, 17], [1.0, 2.0, 3.0]
    t, k, m, p = ["tracks"], ["tracks", "times"], ["tracks", "times", "mergers"], ["tracks", "times", "nmaxpf"]
    time_units = {"units": "Seconds since 1970-1-1", "_FillValue": np.nan}
    return xr.Dataset(
        {
            "track_duration": (t, dur), "start_basetime": (t, start, time_units), "end_basetime": (t, start + 3600.0 * (dur - 1), time_units),
            "start_split_cloudnumber": (t, np.where(np.arange(NTRACKS) % 3 == 0, 7, -9999).astype(np.int32), fill_i),
            "end_merge_cloudnumber": (t, np.full(NTRACKS, -9999, np.int32), fill_i), "pbc_flag": (t, np.zeros(NTRACKS, np.int32)),
            "base_time": (k, base_time, time_units), "meanlat": (k, floats(-40, 40), fill_f), "meanlon": (k, floats(-180, 180), fill_f),
            "ccs_area": (k, floats(1e3, 1e6, nan_inside=0.15), fill_f), "total_rain": (k, floats(0, 5e4, nan_inside=0.1), fill_f),
            "corecold_mintb": (k, floats(170, 250), fill_f),
            "pf_area": (p, pf(), fill_f), "pf_rainrate": (p, pf(), fill_f),
            "merge_cloudnumber": (m, merge_cn, fill_i), "merge_ccs_area": (m, merge_ar, fill_f),
            "split_cloudnumber": (m, split_cn, fill_i), "split_ccs_area": (m, split_ar, fill_f),
        },
        coords={"tracks": np.arange(NTRACKS), "times": np.arange(NTIMES), "mergers": np.arange(NSLOTS), "nmaxpf": np.arange(NPF)},
        attrs={"Title": "test", "time_resolution_hour": 1.0},
    )


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    """The dense file and the row folder (named like the real ones: ...final_<dates>.nc and .rows) of the synthetic statistics."""
    folder = tmp_path_factory.mktemp("final")
    nc = str(folder / "mcs_tracks_final_a_b.nc")
    tbpf_dense().to_netcdf(nc, format="NETCDF4", unlimited_dims="tracks")
    rows = str(folder / "mcs_tracks_final_a_b.rows")
    write_rows(dense_to_rows(nc), rows)
    return nc, rows


@pytest.fixture(scope="module")
def namespace(files):
    """The state after the `# setup` block of the page: the tables, with the path of the synthetic folder."""
    nc, rows = files
    blocks = python_blocks(DOC)
    assert blocks[0].startswith("# setup")
    space = {}
    exec(compile(re.sub(r'^rows = ".*"', f'rows = "{rows}"', blocks[0], flags=re.M), "rowstats_pandas.md setup", "exec"), space)
    return space


@pytest.mark.filterwarnings("ignore:All-NaN slice")
def test_the_python_blocks_run_in_order_on_a_synthetic_file(files):
    nc, rows = files
    blocks = python_blocks(DOC)
    assert len(blocks) >= 9 and blocks[0].startswith("# setup")
    space = {}
    for number, code in enumerate(blocks):
        code = re.sub(r'^rows = ".*"', f'rows = "{rows}"', code, flags=re.M)
        try:
            exec(compile(code, f"rowstats_pandas.md block {number}", "exec"), space)
        except Exception as err:
            raise AssertionError(f"block {number} of docs/rowstats_pandas.md failed: {err!r}\n{code}") from err
    # a few results of the recipes against the dense file
    ds = xr.open_dataset(nc, mask_and_scale=True)
    valid = np.arange(NTIMES)[None, :] < ds.track_duration.values[:, None]
    assert np.allclose(space["summary"]["max_area"].to_numpy(), np.nanmax(ds.ccs_area.values, axis=1), equal_nan=True)
    assert np.allclose(space["largest_pf"].to_numpy(), np.nanmax(ds.pf_area.values, axis=2)[valid], equal_nan=True)
    assert space["n_pf"].to_numpy().tolist() == np.sum(~np.isnan(ds.pf_area.values), axis=2)[valid].tolist()
    assert space["n_merges"].to_numpy().tolist() == (ds.merge_cloudnumber.values > 0).sum(axis=(1, 2)).tolist()
    assert space["n_splits"].to_numpy().tolist() == (ds.split_cloudnumber.values > 0).sum(axis=(1, 2)).tolist()
    assert (space["split"] == -9999).sum() == int(np.isnan(ds.start_split_cloudnumber.values).sum())
    assert np.array_equal(space["split_nan"].to_numpy(), ds.start_split_cloudnumber.values, equal_nan=True)
    assert np.array_equal(space["dense"].values, ds.ccs_area.values[:, :NTIMES][valid.any(axis=1)][:, :space["dense"].shape[1]], equal_nan=True) \
        or space["dense"].shape[0] == int(valid.any(axis=1).sum())
    assert dict(space["ds_some"].sizes)["tracks"] == 3
    assert space["monthly"].sum() == NTRACKS
    # the entries are untouched by the recipes in their order (the offsets slices of the page rely on it)
    entries = space["entries"]
    assert (np.diff(entries["track"].to_numpy()) >= 0).all() and entries.groupby("track")["k"].apply(lambda s: (np.diff(s.to_numpy()) == 1).all()).all()


# The cheat sheet: (what, Xarray expression, Pandas expression, how to compare). The expressions are the ones of the page.
CASES = [
    ("number of tracks", 'ds.sizes["tracks"]', "len(tracks)", "equal"),
    ("number of entries", "int(ds.track_duration.sum())", "len(entries)", "equal"),
    ("a variable as numpy", "ds.ccs_area.values", 'entries["ccs_area"].to_numpy()', "entries"),
    ("one PF slot", "ds.pf_area.isel(nmaxpf=0).values", 'entries["pf_area__0"].to_numpy()', "entries"),
    ("all PF slots", "ds.pf_area.values", 'entries.filter(like="pf_area__")', "entries3"),
    ("describe a per-track variable", "ds.track_duration.to_series().describe()", 'tracks["track_duration"].describe()', "series"),
    ("tracks by a per-track variable", "ds.isel(tracks=ds.track_duration.values > 20)", 'tracks[tracks["track_duration"] > 20]', "track_ids"),
    ("tracks active at a time", "(ds.base_time.values == t.to_datetime64()).any(axis=1)", 'entries.loc[entries["time"] == t, "track"].unique()', "ids"),
    ("tracks with a step where a condition holds", "(ds.meanlat.values > -20).any(axis=1)", 'entries.loc[entries["meanlat"] > -20, "track"].unique()', "ids"),
    ("a variable where a condition holds", 'ds.ccs_area.where(ds.meanlat > -20)', 'entries["ccs_area"].where(entries["meanlat"] > -20)', "entries"),
    ("first time step", "ds.ccs_area.isel(times=0).values", 'entries[entries["k"] == 0]', "first"),
    ("last time step", "ds.ccs_area.values[np.arange(len(ds.tracks)), ds.track_duration.values - 1]", 'entries.groupby("track").tail(1)', "first"),
    ("mean of a per-track variable", "ds.track_duration.mean().item()", 'tracks["track_duration"].mean()', "scalar"),
    ("mean over the lifetime", 'ds.ccs_area.mean("times", skipna=True).values', 'entries.groupby("track")["ccs_area"].mean()', "per_track"),
    ("maximum over the lifetime", 'ds.ccs_area.max("times").values', 'entries.groupby("track")["ccs_area"].max()', "per_track"),
    ("sum over the lifetime", 'ds.total_rain.sum("times").values', 'entries.groupby("track")["total_rain"].sum()', "per_track"),
    ("running total", 'ds.total_rain.cumsum("times")', 'entries["total_rain"].fillna(0).groupby(entries["track"]).cumsum()', "entries"),
    ("change from the previous step", "ds.ccs_area - ds.ccs_area.shift(times=1)", 'entries.groupby("track")["ccs_area"].diff()', "entries"),
    ("running mean of 3 steps", "ds.ccs_area.rolling(times=3).mean()", 'entries.groupby("track")["ccs_area"].transform(lambda s: s.rolling(3).mean())', "entries"),
    ("total over everything", "float(ds.total_rain.sum())", 'entries["total_rain"].sum()', "scalar"),
    ("round to the second", 'ds.base_time.dt.round("s")', 'entries["time"].dt.round("s")', "entries_time"),
    ("hour of the day", "ds.base_time.dt.hour", 'entries["time"].dt.hour', "entries"),
    ("local solar time", 'ds.base_time + (ds.meanlon.astype("float64") / 15 * 3600e9).astype("timedelta64[ns]")',
     'entries["time"] + pd.to_timedelta(entries["meanlon"].astype("float64") / 15, unit="h")', "entries_time_close"),
    ("mcs started per month", 'ds.start_basetime.groupby(ds.start_basetime.dt.month).count()', 'tracks.groupby(tracks["start_time"].dt.month).size()', "month"),
    ("merges per track", '(ds.merge_cloudnumber > 0).sum(("times", "mergers"))', 'fragments[fragments["kind"] == 0].groupby("track").size()', "per_track0"),
]


def _prepare(files, namespace):
    """The namespace of the cheat sheet: the dense Dataset, the tables of the page's setup block, a time that exists and the helpers."""
    nc, rows = files
    space = dict(namespace)
    ds = xr.open_dataset(nc, mask_and_scale=True)
    space.update(ds=ds, np=np, pd=pd, xr=xr, t=pd.Timestamp(ds.base_time.values[3, 1]), valid=np.arange(NTIMES)[None, :] < ds.track_duration.values[:, None])
    return space


def _evaluate(expr, space):
    """Evaluate a cheat-sheet expression; the thresholds and the track number of the page are replaced by values that exist in the small file."""
    return eval(expr.replace("> 20", "> 5").replace("234", "3"), space)


def _as_ids(result):
    """Track numbers from a boolean mask over tracks, a DataFrame of tracks, a Dataset of tracks, or an array of track numbers."""
    if isinstance(result, np.ndarray) and result.dtype == bool:
        return np.nonzero(result)[0]
    if isinstance(result, xr.Dataset):
        return result.tracks.values
    if isinstance(result, pd.DataFrame):
        return result.index.to_numpy()
    return np.sort(np.asarray(result))


@pytest.mark.parametrize("what, xr_expr, pd_expr, how", CASES, ids=[c[0] for c in CASES])
def test_the_cheat_sheet_commands_agree_with_xarray(files, namespace, what, xr_expr, pd_expr, how):
    text = open(DOC, encoding="utf-8").read()
    # the page contains both commands as they are tested here
    assert f"`{xr_expr}`" in text, f"the Xarray command of '{what}' is not in the page"
    assert f"`{pd_expr}`" in text, f"the Pandas command of '{what}' is not in the page"
    space = _prepare(files, namespace)
    ds, valid = space["ds"], space["valid"]
    want, got = _evaluate(xr_expr, space), _evaluate(pd_expr, space)
    close = lambda a, b: np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), equal_nan=True, rtol=1e-5)
    if how == "equal":
        assert want == got
    elif how == "scalar":
        assert np.isclose(want, got, rtol=1e-5)
    elif how == "series":
        assert close(want.to_numpy(), got.to_numpy())
    elif how == "per_track":
        assert close(want.values if hasattr(want, "values") else want, got.reindex(space["tracks"].index).to_numpy())
    elif how == "per_track0":
        assert close(want.values, got.reindex(space["tracks"].index, fill_value=0).to_numpy())
    elif how == "entries":
        # the dense array holds the pandas column in its valid cells, in row-major order (a result as a DataArray: its values)
        assert close(np.asarray(want)[valid], np.asarray(got))
    elif how == "entries3":
        assert close(np.asarray(want)[valid], got.to_numpy())
    elif how == "entries_time":
        assert np.array_equal(np.asarray(want)[valid], got.to_numpy().astype("datetime64[ns]"))
    elif how == "entries_time_close":
        # the two ways to turn the hours into a time differ by the rounding of a float to whole nanoseconds (here: below a microsecond)
        difference = np.asarray(want)[valid] - got.to_numpy().astype("datetime64[ns]")
        assert (np.abs(difference) < np.timedelta64(1, "us")).all()
    elif how == "first":
        assert close(np.asarray(want), got["ccs_area"].to_numpy())
    elif how == "ids":
        assert np.array_equal(_as_ids(want), _as_ids(got))
    elif how == "track_ids":
        assert np.array_equal(_as_ids(want), _as_ids(got))
    elif how == "month":
        months = ds.start_basetime.dt.month.values
        assert close(np.bincount(months, minlength=13)[got.index.to_numpy()], got.to_numpy())
        assert dict(zip(want.month.values.tolist(), want.values.tolist())) == dict(zip(got.index.tolist(), got.tolist()))
    else:
        raise AssertionError(how)


def test_the_cheat_sheet_commands_that_need_more_than_one_expression(files, namespace):
    """The rows of the page whose two forms are not one expression each: the condition on every step, the entry of the peak, fragments, joins."""
    text = open(DOC, encoding="utf-8").read()
    space = _prepare(files, namespace)
    ds, tracks, entries, fragments, valid = space["ds"], space["tracks"], space["entries"], space["fragments"], space["valid"]
    # every step satisfies a condition: Xarray needs the padding masked first
    cmd = '(entries["meanlat"] > -20).groupby(entries["track"]).all()'
    assert f"`{cmd}`" in text
    dense_all = ((ds.meanlat.values > -20) | ~valid).all(axis=1)
    assert np.array_equal(eval(cmd, space).to_numpy(), dense_all[valid.any(axis=1)])
    # the whole entry where a variable peaks
    cmd = 'entries.loc[entries.groupby("track")["ccs_area"].idxmax()]'
    assert f"`{cmd}`" in text
    peak = eval(cmd, space)
    assert np.array_equal(peak["k"].to_numpy(), ds.ccs_area.fillna(-1).argmax("times").values)
    assert f"`ds.ccs_area.fillna(-1).argmax(\"times\").values`" in text
    # several statistics at once
    cmd = 'entries.groupby("track").agg(a=("ccs_area", "max"), b=("total_rain", "sum"))'
    assert f"`{cmd}`" in text
    agg = eval(cmd, space)
    assert list(agg.columns) == ["a", "b"] and np.allclose(agg["a"].to_numpy(), np.nanmax(ds.ccs_area.values, axis=1), equal_nan=True)
    # per-track variables on the entries, and the duration of the track on every entry
    cmd = 'entries["track"].map(tracks["track_duration"])'
    assert f"`{cmd}`" in text
    assert np.array_equal(eval(cmd, space).to_numpy(), np.repeat(ds.track_duration.values, ds.track_duration.values))
    cmd = 'entries.join(tracks, on="track")'
    assert f"`{cmd}`" in text
    joined = eval(cmd, space)
    assert len(joined) == len(entries) and "track_duration" in joined.columns
    # the number of tracks active at every time, against a count of the dense times
    cmd = 'entries.groupby("time").size()'
    assert f"`{cmd}`" in text
    count = eval(cmd, space)
    times = ds.base_time.values[valid]
    assert dict(zip(count.index.to_numpy().astype("datetime64[ns]"), count.to_numpy())) == {u: int(c) for u, c in zip(*np.unique(times, return_counts=True))}
    # the diurnal cycle
    cmd = 'entries.groupby(entries["time"].dt.hour)["ccs_area"].mean()'
    assert f"`{cmd}`" in text
    hours = pd.DatetimeIndex(times).hour
    diurnal = eval(cmd, space)
    for hour, value in diurnal.items():
        assert np.isclose(value, np.nanmean(ds.ccs_area.values[valid][hours == hour]), rtol=1e-5)
    # the merge fragments of a track: the rows are the non-fill cells of the dense array
    cmd = 'fragments[(fragments["track"] == i) & (fragments["kind"] == 0)]'
    assert f"`{cmd}`" in text
    i = 0
    space["i"] = i
    rows_i = eval(cmd, space)
    dense = ds.merge_cloudnumber.values[i, :int(ds.track_duration.values[i]), :]
    kk, ss = np.nonzero(~np.isnan(dense))
    assert sorted(zip(rows_i["k"], rows_i["slot"], rows_i["cloudnumber"])) == sorted(zip(kk, ss, dense[kk, ss].astype(int)))
    pivot = rows_i.pivot(index="k", columns="slot", values="cloudnumber").reindex(range(int(ds.track_duration.values[i])))
    assert np.array_equal(pivot.reindex(columns=range(NSLOTS)).to_numpy(), dense, equal_nan=True)
    # rows to a dense array, and the dense Dataset of some tracks
    cmd = 'entries.set_index(["track", "k"])["ccs_area"].to_xarray()'
    assert f"`{cmd}`" in text
    back = eval(cmd, space)
    assert np.array_equal(back.reindex(track=range(NTRACKS), k=range(NTIMES)).values[valid], ds.ccs_area.values[valid], equal_nan=True)
    assert 'load_trackstats(rows, tracks=ids, variables=["ccs_area"])' in text
    # the offsets slice of the page
    offsets = np.r_[0, tracks["track_duration"].cumsum().to_numpy()]
    for track in (0, 1, 5):
        assert entries.iloc[offsets[track]:offsets[track + 1]].equals(entries[entries["track"] == track])
