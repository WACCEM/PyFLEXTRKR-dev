# From Xarray to Pandas: analysing the row statistics

The Tb+PF workflow writes its track statistics as **row tables** (`stats_format: rows`, [rowstats.md](rowstats.md)) and, for the final file, also as the dense netCDF file
(`mcs_tracks_final_<dates>.nc`, [tbpf_mcs_workflow.md](tbpf_mcs_workflow.md)). If you know the dense file through Xarray (`ds.ccs_area.mean("times")`, `ds.sel(tracks=234)`), this page
is the translation to Pandas on the row tables, which is what you want for a year of tracks: the IMERG 2019 year is **26.4 GB as dense arrays and 0.30 GB as row tables**, loads in 0.5 s instead of
seconds per variable, and stays one line for the questions that are not a reduction along time.

Companions: the notebook [demo_mcs_track_stats_rows_pandas.ipynb](../Notebooks/demo_mcs_track_stats_rows_pandas.ipynb) repeats every analysis of
[demo_mcs_track_stats_healpix.ipynb](../Notebooks/demo_mcs_track_stats_healpix.ipynb) with Pandas and checks each result against Xarray; [rowstats.md](rowstats.md) describes the format and the other ways to read it.
Every `python` block of this page runs in the tests on a synthetic file, and every command of the [cheat sheet](#the-cheat-sheet) is evaluated there next to its Xarray form and compared.

## The idea

A dense file has one row per track and one column per time step up to the longest track (400 hours for the global year), so most cells are padding. The row tables keep only what exists:

```text
dense (Xarray)                              rows (Pandas)
ccs_area[tracks, times]                     entries: one row per (track, k) that exists
          k=0  1   2   3 ...  399                  track  k   ccs_area
track 0 [ 5.1 6.0 4.2 nan ...  nan ]               0      0   5.1
track 1 [ 2.2 nan nan nan ...  nan ]               0      1   6.0
track 2 [ 7.7 8.1 9.0 8.8 ...  nan ]               0      2   4.2
                                                   1      0   2.2
                                                   2      0   7.7 ...     (sorted by track, then k)
```

| dense netCDF (Xarray, `ds`) | row tables (Pandas) |
|---|---|
| 1-D `[tracks]` variables: `track_duration`, `start_basetime`, ... | columns of **`tracks`**, one row per track (row label = the 0-based track number) |
| `[tracks, times]` variables: `base_time`, `meanlat`, `ccs_area`, ... (NaN or a fill value after the end of the track) | columns of **`entries`**, one row per (track, time step) that exists: `track`, `k`, and the variables |
| `[tracks, times, nmaxpf]` PF variables: `pf_area`, `pf_rainrate`, ... | the columns `pf_area__0`, `pf_area__1`, `pf_area__2` of `entries` |
| `[tracks, times, mergers]` merge / split variables (-9999 = none) | rows of **`fragments`**: `track`, `k`, `kind` (0 merge, 1 split), `slot`, `cloudnumber`, `ccs_area` |
| the `tracks` coordinate (0-based; the number in the pixel files and in the trackpcp tables is `tracks + 1`) | the `track` column and the index of `tracks` |
| the `times` index | the `k` column |

Three things to hold on to: (1) a statistic of "the time steps of each track" is `entries.groupby("track")`, there is no axis to reduce and no padding to skip; (2) a condition on time steps
is a boolean column over `entries`, and "at least one step" / "every step" is `.groupby(entries["track"]).any()` / `.all()`; (3) `entries` is **sorted by track, then by `k`**, and
the rows of track `i` are `entries.iloc[offsets[i]:offsets[i + 1]]` with `offsets = cumsum(track_duration)`: keep that order.

## Reading the files

```python
# setup: the three tables, the track label, and the times (epoch seconds in the file) as timestamps
import numpy as np
import pandas as pd

rows = "mcs_tracks_final_20190101.0000_20200101.0000.rows"        # the row folder of the statistics
tracks = pd.read_parquet(f"{rows}/tracks.parquet").set_index("tracks")      # one row per track, labelled by the track number
entries = pd.read_parquet(f"{rows}/entries.parquet")                        # one row per (track, time step)
fragments = pd.read_parquet(f"{rows}/fragments.parquet")                    # one row per merge / split fragment
entries["time"] = pd.to_datetime(entries["base_time"], unit="s").dt.round("s")
tracks["start_time"] = pd.to_datetime(tracks["start_basetime"], unit="s").dt.round("s")
```

The same data read the usual ways, measured on the IMERG 2019 final statistics (41,501 tracks, 785,380 entries, 94 columns; one Perlmutter CPU node, files in the cache):

| what | Pandas on the row tables | Xarray on the dense file |
|---|---|---|
| open | `pd.read_parquet(f"{rows}/tracks.parquet")`: 0.006 s | `xr.open_dataset(nc)`: 0.03 s (lazy; 1.9 s the first time) |
| one variable | `pd.read_parquet(path, columns=["track", "k", "ccs_area"])`: 0.02 s, 13 MB for four columns | `ds.ccs_area.load()`: 0.4 s, 66 MB |
| four variables | the same with four columns: 0.02 s | `ds[["ccs_area", "meanlat", "meanlon", "base_time"]].load()`: 1.5 s |
| everything | `pd.read_parquet(f"{rows}/entries.parquet")`: 0.55 s for all 94 columns, 0.30 GB | 70 variables: 26.4 GB (`ds.nbytes`) |
| one track, all variables | `entries[entries["track"] == 2001]` (a scan, 0.3 ms) or `entries.iloc[offsets[2001]:offsets[2002]]` | `ds.sel(tracks=2001).load()`: 0.03 s |

* **Read only the columns you need** (`columns=[...]`). Parquet stores each column on its own, so the other ones are not touched.
* `filters=[("track", "==", 2001)]` only skips *row groups*; `entries.parquet` of a year is a single row group, so the filter reads as much as the whole file (0.46 s). Read the file once and slice it.
* Every column keeps the dtype of the file (`float32`, `int32`, ...), which is also why 94 columns of 785,380 entries are 0.30 GB: do not convert to `float64` unless you sum a lot.
* `RowStats` (`from pyflextrkr.rowstats import RowStats`) gives numpy access without a DataFrame: `rs = RowStats.open(rows)`, `rs["ccs_area"][2001]` is the series of track 2001, and
  `rs.to_pandas("entries", tracks=[1, 2], variables=["ccs_area"])` returns a DataFrame of some tracks and variables.
* DuckDB, polars, R and Spark read the same Parquet files (`entries.parquet`, `tracks.parquet`, `fragments.parquet`, plain columns); the column names are those of this page.

## The cheat sheet

The Pandas column assumes the `# setup` block above (`tracks`, `entries`, `fragments`, `entries["time"]`); the Xarray column assumes `ds = xr.open_dataset(nc)` of the dense file. `t` is a `pd.Timestamp`.
All of these are evaluated on a synthetic file and compared in `tests/test_rowstats_pandas_doc.py`.

### Look at the data

| what it does | Xarray (dense) | Pandas (rows) |
|---|---|---|
| number of tracks | `ds.sizes["tracks"]` | `len(tracks)` |
| number of (track, time step) entries | `int(ds.track_duration.sum())` | `len(entries)` |
| names of the variables | `list(ds.data_vars)` | `list(entries.columns)` |
| a variable as a numpy array (2-D padded, or 1-D without padding) | `ds.ccs_area.values` | `entries["ccs_area"].to_numpy()` |
| one PF slot of a `[tracks, times, nmaxpf]` variable | `ds.pf_area.isel(nmaxpf=0).values` | `entries["pf_area__0"].to_numpy()` |
| all the PF slots of a variable | `ds.pf_area.values` | `entries.filter(like="pf_area__")` |
| file attributes | `ds.attrs["time_resolution_hour"]` | `json.load(open(f"{rows}/meta.json"))["global_attrs"]["time_resolution_hour"]` |
| summary statistics of a per-track variable | `ds.track_duration.to_series().describe()` | `tracks["track_duration"].describe()` |

### Select tracks and time steps

| what it does | Xarray (dense) | Pandas (rows) |
|---|---|---|
| one track (per-track variables) | `ds.sel(tracks=234)` | `tracks.loc[234]` |
| one track (its time series) | `ds.sel(tracks=234).ccs_area.values[:duration]` (cut the padding) | `entries[entries["track"] == 234]` (no padding) |
| several tracks | `ds.sel(tracks=[1, 2, 3])` | `entries[entries["track"].isin([1, 2, 3])]` |
| tracks by a per-track variable | `ds.isel(tracks=ds.track_duration.values > 20)` | `tracks[tracks["track_duration"] > 20]` |
| tracks active at a time | `(ds.base_time.values == t.to_datetime64()).any(axis=1)` | `entries.loc[entries["time"] == t, "track"].unique()` |
| tracks with at least one step where a condition holds | `(ds.meanlat.values > -20).any(axis=1)` | `entries.loc[entries["meanlat"] > -20, "track"].unique()` |
| tracks where it holds at every step | `(ds.meanlat.values > -20).all(axis=1)` fails on the padding: mask it first | `(entries["meanlat"] > -20).groupby(entries["track"]).all()` |
| a variable where a condition holds (NaN elsewhere) | `ds.ccs_area.where(ds.meanlat > -20)` | `entries["ccs_area"].where(entries["meanlat"] > -20)` |
| the first time step of every track | `ds.ccs_area.isel(times=0).values` | `entries[entries["k"] == 0]` |
| the first six time steps | `ds.isel(times=slice(0, 6))` | `entries[entries["k"] < 6]` |
| the last time step of every track | `ds.ccs_area.values[np.arange(len(ds.tracks)), ds.track_duration.values - 1]` | `entries.groupby("track").tail(1)` |
| loop over tracks | `for i in ids: ds.sel(tracks=i)` | `for i, track in entries.groupby("track"):` (one pass) |

### One value per track, or per time step

| what it does | Xarray (dense) | Pandas (rows) |
|---|---|---|
| mean of a per-track variable | `ds.track_duration.mean().item()` | `tracks["track_duration"].mean()` |
| mean over the lifetime of each track | `ds.ccs_area.mean("times", skipna=True).values` | `entries.groupby("track")["ccs_area"].mean()` |
| maximum over the lifetime | `ds.ccs_area.max("times").values` | `entries.groupby("track")["ccs_area"].max()` |
| sum over the lifetime | `ds.total_rain.sum("times").values` | `entries.groupby("track")["total_rain"].sum()` |
| several statistics at once | one reduction per variable, then `xr.merge` | `entries.groupby("track").agg(a=("ccs_area", "max"), b=("total_rain", "sum"))` |
| the entry where a variable peaks | `ds.ccs_area.fillna(-1).argmax("times").values` (an index: take the other variables with it) | `entries.loc[entries.groupby("track")["ccs_area"].idxmax()]` (a whole row) |
| running total along the lifetime (a missing value adds 0) | `ds.total_rain.cumsum("times")` | `entries["total_rain"].fillna(0).groupby(entries["track"]).cumsum()` |
| change from the previous step (NaN at the first one) | `ds.ccs_area - ds.ccs_area.shift(times=1)` (`.diff("times")` drops the first step) | `entries.groupby("track")["ccs_area"].diff()` |
| running mean of 3 steps | `ds.ccs_area.rolling(times=3).mean()` | `entries.groupby("track")["ccs_area"].transform(lambda s: s.rolling(3).mean())` |
| a per-track value on every entry | `ds.track_duration` (broadcasts) | `entries["track"].map(tracks["track_duration"])` |
| total over everything | `float(ds.total_rain.sum())` | `entries["total_rain"].sum()` |
| number of tracks active at every time | (a loop over the times) | `entries.groupby("time").size()` |
| MCS started per month | `ds.start_basetime.groupby(ds.start_basetime.dt.month).count()` | `tracks.groupby(tracks["start_time"].dt.month).size()` |

### Times

| what it does | Xarray (dense) | Pandas (rows) |
|---|---|---|
| epoch seconds to timestamps | automatic (the netCDF units) | `pd.to_datetime(entries["base_time"], unit="s")` |
| round to the second | `ds.base_time.dt.round("s")` | `entries["time"].dt.round("s")` |
| hour of the day | `ds.base_time.dt.hour` | `entries["time"].dt.hour` |
| diurnal cycle of a variable | a mask over the padded array for each hour | `entries.groupby(entries["time"].dt.hour)["ccs_area"].mean()` |
| local solar time | `ds.base_time + (ds.meanlon.astype("float64") / 15 * 3600e9).astype("timedelta64[ns]")` | `entries["time"] + pd.to_timedelta(entries["meanlon"].astype("float64") / 15, unit="h")` |

### Merge / split fragments

| what it does | Xarray (dense) | Pandas (rows) |
|---|---|---|
| the merge fragments of a track | `ds.merge_cloudnumber.values[i, :duration, :]` ([time, 50 slots], -9999 = none) | `fragments[(fragments["track"] == i) & (fragments["kind"] == 0)]` |
| the dense `[time, slot]` array again | (it is the dense array) | `...pivot(index="k", columns="slot", values="cloudnumber")` |
| number of merges of each track | `(ds.merge_cloudnumber > 0).sum(("times", "mergers"))` | `fragments[fragments["kind"] == 0].groupby("track").size()` |

### Combine tables, and go back

| what it does | Xarray (dense) | Pandas (rows) |
|---|---|---|
| per-track variables next to a per-track result | `xr.merge([ds, other])` | `tracks.join(other)` |
| per-track variables on the entries | (broadcast) | `entries.join(tracks, on="track")` |
| rows to a dense array | (it is one) | `entries.set_index(["track", "k"])["ccs_area"].to_xarray()` |
| rows to a dense `Dataset` of some tracks | | `load_trackstats(rows, tracks=ids, variables=["ccs_area"])` |
| rows to the dense netCDF file | | `write_dense_netcdf(rows)` ([tbpf_mcs_workflow.md](tbpf_mcs_workflow.md)) |
| a dense variable to rows | `ds.ccs_area.to_dataframe().dropna()` | |

## Recipes

### A condition on time steps becomes a condition on tracks

```python
# one boolean per entry: the cold core of the cloud shield is colder than 200 K
cold = entries["corecold_mintb"] < 200

# one boolean per track: at least one step (the .any("times") of the dense file) and every step; the result is labelled by the track number
ever_cold = cold.groupby(entries["track"]).any()
always_cold = cold.groupby(entries["track"]).all()
cold_ids = ever_cold.index[ever_cold]

# the time series of those tracks, and a per-track value (the duration) on every entry
cold_entries = entries[entries["track"].isin(cold_ids)]
entries["duration"] = entries["track"].map(tracks["track_duration"])
```

### One row per track

```python
# several statistics of the lifetime variables at once; the result has one row per track, labelled by the track number
summary = entries.groupby("track").agg(max_area=("ccs_area", "max"), total_rain=("total_rain", "sum"), min_tb=("corecold_mintb", "min"))

# next to the per-track variables (both are labelled by the track number)
summary = tracks[["track_duration", "start_time"]].join(summary)

# the whole entry where the area is largest (every other variable of that time step comes with it), and the last entry of every track
at_peak = entries.loc[entries.groupby("track")["ccs_area"].idxmax()]
last = entries.groupby("track").tail(1)
```

`groupby("track").first()` and `.last()` return the first / last **non-missing value of each column**, which can come from different time steps if a variable has NaN. For the entry of the first or last
time step use `entries[entries["k"] == 0]` and `.tail(1)` as above.

### The lifecycle: time since genesis and a composite

```python
# the position in the life of each entry: 0 at the first step, 1 at the last (a one-step track has no position)
entries["phase"] = entries["k"] / (entries["duration"] - 1)

# a composite over the lifetime: the mean area in ten bins of the life, over all tracks
bins = np.linspace(0, 1, 11)
composite = entries.groupby(pd.cut(entries["phase"], bins, include_lowest=True), observed=True)["ccs_area"].mean()

# the hours since genesis are k itself (the data are hourly): the mean area at each hour of the life
by_hour = entries.groupby("k")["ccs_area"].mean()
```

### Times: counts, seasons, the diurnal cycle

```python
# MCS started per calendar month (the index is the first day of the month)
monthly = tracks.set_index("start_time").resample("MS").size()

# the diurnal cycle: the hour (UTC) of each entry, and the local solar time of each entry from its longitude
diurnal = entries.groupby(entries["time"].dt.hour)["ccs_area"].mean()
entries["local_time"] = entries["time"] + pd.to_timedelta(entries["meanlon"] / 15, unit="h")
local_diurnal = entries.groupby(entries["local_time"].dt.hour)["ccs_area"].mean()
```

### The precipitation features: the slots `__0`, `__1`, `__2`

```python
# the slots of one variable as a [entries, nmaxpf] table, and the largest of them at every step (NaN-aware, one value per entry)
pf_area = entries.filter(like="pf_area__")
largest_pf = pf_area.max(axis=1)

# the long form: one row per (track, k, slot); `slot` is the number of the precipitation feature (0 = the largest PF)
pf_long = pd.wide_to_long(entries[["track", "k", *pf_area.columns]], stubnames="pf_area", i=["track", "k"], j="slot", sep="__")

# the number of precipitation features at each step (the slots that are not NaN)
n_pf = pf_area.notna().sum(axis=1)
```

### Merge and split fragments

```python
# fragments are rows: how many merges and splits does each track have (tracks without one get 0)
n_merges = fragments[fragments["kind"] == 0].groupby("track").size().reindex(tracks.index, fill_value=0)
n_splits = fragments[fragments["kind"] == 1].groupby("track").size().reindex(tracks.index, fill_value=0)

# a fragment next to the track it is attached to at that time step (several fragments can share one (track, k))
attached = fragments.merge(entries[["track", "k", "ccs_area"]], on=["track", "k"], suffixes=("_fragment", "_track"))
attached["area_ratio"] = attached["ccs_area_fragment"] / attached["ccs_area_track"]
```

### Fill values and missing data

```python
# Xarray (mask_and_scale=True) turns the -9999 of an integer variable into NaN and the variable into float64; the row tables keep the integers of the file
split = tracks["start_split_cloudnumber"]
print((split == -9999).sum(), "tracks did not start from a split")
split_nan = split.where(split != -9999)                 # NaN where there is none, like Xarray (float64)

# NaN inside an entry is a missing value (for example a PF slot that does not exist); a row exists only where the track does
missing = entries.filter(like="pf_area__").isna().mean()
```

### Back to Xarray

```python
# a [track, k] array of a variable (NaN where a track is shorter), and the dense Dataset of some tracks with the names and fill values of the dense file
from pyflextrkr.rowstats import load_trackstats

dense = entries.set_index(["track", "k"])["ccs_area"].to_xarray()
ds_some = load_trackstats(rows, tracks=[1, 2, 3], variables=["ccs_area"], fragments=False)
```

### Join with the precipitation accumulated along the tracks

The tables of the `trackpcp` step ([tbpf_mcs_workflow.md](tbpf_mcs_workflow.md)) number the MCSs from 1 (`track_number`), the row tables from 0: `track = track_number - 1`.

```py
# the per-MCS metrics of the accumulation (one row per robust MCS) next to the per-track variables of the final statistics
metrics = pd.read_parquet(f"{stats}/trackpcp_metrics_20190101.0000_20200101.0000.parquet")
metrics = metrics.assign(track=metrics["track_number"] - 1).set_index("track")
tracks = tracks.join(metrics[["accum_vol_1e6m3", "swath_area_km2"]])
```

## Habits that keep it fast

| do | instead of | why |
|---|---|---|
| `entries.groupby("track")[col].mean()` | a Python loop over tracks with `entries[entries["track"] == i]` | one pass over the table: the loop reads the table once per track (2,000 single-track lookups: 0.68 s with a mask, 0.04 s with the offsets slice) |
| `entries.iloc[offsets[i]:offsets[i + 1]]` for one track out of many | `entries[entries["track"] == i]` | the entries of a track are one block of rows: a slice is a view, no scan |
| `pd.read_parquet(path, columns=[...])` | reading all 94 columns and dropping some | only the columns that are asked for are read from the disk |
| `.to_numpy()` for an array operation | `.apply(...)` / `.iterrows()` on a column or a table | vectorised operations run in compiled code, the row-by-row ones in Python |
| `tracks.join(summary)` / `entries["track"].map(...)` | a nested loop that matches track numbers by hand | the track number is the key of every table |
| a boolean mask, then `.groupby(entries["track"]).any()` | `.groupby("track").apply(lambda g: g["x"].any())` | the built-in reductions run in compiled code |

The offsets of the slice are `offsets = np.r_[0, tracks["track_duration"].cumsum().to_numpy()]`; they are valid as long as `entries` is **not re-sorted, filtered or reset**: work on copies
(`cold_entries = entries[...]`) and keep `entries` as read.

## Pitfalls

| symptom | cause | fix |
|---|---|---|
| a track is missing a step, or two tracks are mixed up after `entries.sort_values("ccs_area")` | the table is sorted by track and `k`, and the offsets slices depend on it | sort a copy, or sort by `["track", "k"]` again |
| `tracks.loc[234]` is the wrong track, or `KeyError` | the label is the 0-based track number, the pixel files and the trackpcp tables use the number + 1 | `track_number = track + 1` for the pixel files (`cloudtracknumber`) and the trackpcp tables |
| `-9999` shows up in a mean | integer variables keep the fill value of the file (Xarray turned it into NaN) | `col.where(col != -9999)` before the statistic |
| a sum differs in the last digits from Xarray | the columns are `float32`; the order of the additions differs | compare with `np.allclose`; `.astype("float64")` for a large sum |
| `groupby("track").first()` is not the first time step | `first` takes the first non-missing value of every column | `entries[entries["k"] == 0]` |
| the result of a `groupby` has fewer rows than `tracks` | a track that has no entry passing the filter has no group | `.reindex(tracks.index)` (and `fill_value=0` for counts) |
| `SettingWithCopyWarning`, or an edit does not stick | `entries.iloc[a:b]` and a boolean selection can be views or copies | `.copy()` before you write into a slice |
| the time of an entry is a float | `base_time` is epoch seconds | `pd.to_datetime(col, unit="s")` (the notebook rounds to the second: some inputs have a small offset) |
| a join of two statistics files matches the wrong tracks | `mcs_tracks_` and `mcs_tracks_pf_` number every Tb MCS, the robust and final files renumber the robust MCSs | join files of the same stage, or through `track_number` of the trackpcp tables |

## Learn more

* The notebook [demo_mcs_track_stats_rows_pandas.ipynb](../Notebooks/demo_mcs_track_stats_rows_pandas.ipynb): every analysis of the Xarray demo next to its Pandas version, with the timing on the IMERG year.
* The pandas user guide: [10 minutes to pandas](https://pandas.pydata.org/docs/user_guide/10min.html), [group by: split-apply-combine](https://pandas.pydata.org/docs/user_guide/groupby.html),
  [merge, join, concatenate](https://pandas.pydata.org/docs/user_guide/merging.html) and [time series](https://pandas.pydata.org/docs/user_guide/timeseries.html) are the four chapters this page uses.
* Xarray users: the dense array is `entries.set_index(["track", "k"])`, a `MultiIndex` that Pandas stacks and unstacks (`.unstack("k")` gives `[tracks, times]` back, `.to_xarray()` an Xarray array).
* [rowstats.md](rowstats.md): the tables, the `RowStats` and `load_trackstats` access and the conversion to the dense file.
