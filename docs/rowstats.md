# Row-based track statistics (`stats_format: rows`)

The statistics files of the radar MCS pipeline (`mcs_tracks_*`, `mcs_tracks_robust_*`, `mcs_tracks_final_*`) have `[tracks, times]`
and `[tracks, times, slots]` arrays. They are almost empty: a track lives for `track_duration` steps (7 on average) but every row
has the width of the longest track, and the merge / split fragments fill a few of the 50 slots. For the 4-day case c4 (31,137
tracks, 1,189 frames) the arrays are 21.7 GB uncompressed, for a month about 180 GB, for a year about 3 TB.

With `stats_format: rows` the same numbers are stored as three tables, one row per thing that exists: the file is a few MB for c4 (125 MB
for May 2025, whose dense file is 1.23 GB on disk), the steps need memory and time for the entries that exist (1.7 million per month, about
20 million for a year), and a year fits in **one** file. Measured: identifymcs of a month takes 3.3 min instead of 31 min, of six months
17.7 min with 5.5 GB; regionprops of a month 3.9 min instead of 36 min.
`stats_format: dense` (the default) writes the netCDF files as before; the two formats hold identical numbers, and a helper gives
the dense arrays back for any selection of tracks.

| | dense (default) | rows |
|---|---|---|
| config | `stats_format: dense` | `stats_format: rows` |
| files | `mcs_tracks_<dates>.nc`, `..._robust_...nc`, `..._final_...nc` | folders `mcs_tracks_<dates>.rows`, `..._robust_...rows`, `..._final_...rows` |
| steps | identifymcs, regionprops, mapfeature, speed read and write the netCDF files | the same steps read and write the folders |
| unchanged | `tracknumbers_*.nc`, `trackstats_sparse_*.nc`, cloudid, track and pixel files | the same files |
| limit | `duration_range` (width of the arrays), memory of the dense arrays | none of these |
| size, case c4 | 21.7 GB in memory, 0.155 GB compressed on disk | 0.045 GB in memory, 0.020 GB on disk |

## The tables

A folder `mcs_tracks_robust_<dates>.rows/` holds `tracks.parquet`, `entries.parquet`, `fragments.parquet` and `meta.json`.

| table | one row per | columns |
|---|---|---|
| tracks | track | the 1-D variables (`track_duration`, `is_mcs`, `track_class`, `lifecycle_ncycles`, ...) and the small 2-D ones as `name__0`, `name__1`, ... (`lifecycle_index`) |
| entries | (track, time index) that exists, sorted by track then time | `track`, `k` (time index), and every `[tracks, times]` variable (`area`, `base_time`, `cloudnumber`, `majoraxis`, `movement_speed`, ...) |
| fragments | merge or split fragment attached to a track at a time | `track`, `k`, `kind` (0 merge, 1 split), `slot`, `cloudnumber`, `status`, `area` |

A track's entries are one block of the entries table: `offsets[i]` to `offsets[i + 1]`, with `offsets = cumsum(track_duration)`. A
variable is therefore a flat array plus the offsets: that is what `RowStats` returns, with the indexing of the dense array.
`meta.json` has what is needed to rebuild the dense layout exactly: dimension sizes, dtypes, fill values, attributes. The tables
are Parquet files (columnar, compressed, readable by pandas, pyarrow, DuckDB, R, Spark).

## Reading the files

Three ways, from "my code was written for the dense files" to "I want to learn pandas":

1. **`load_trackstats`** returns the dense `xarray.Dataset` of any file, in either format, for all tracks or a selection. The result
   has the names, dimensions, dtypes, fill values and attributes of the dense file, so existing code works unchanged. A file name that
   ends in `.nc` but exists only as a `.rows` folder is found too.
2. **`RowStats`** gives numpy access that looks like `a[track, time]`: no dense array is built.
3. **pandas**: `rs.to_pandas("tracks" | "entries" | "fragments")`.

## Cookbook

The recipes run on any robust or final file (they use variables that every file has). `path` is the folder.

```python
# setup
import numpy as np
import pandas as pd
from pyflextrkr.rowstats import RowStats, load_trackstats

path = "mcs_tracks_robust_20250401.0000_20251001.0000.rows"
rs = RowStats.open(path)                  # lazy: only the tracks table is read
```

**How many tracks and entries?**

```python
ntracks = len(rs.duration)                # rs.duration = track_duration of every track
nentries = int(rs.duration.sum())
```

**One variable, indexed like the dense array** (`area` is read from the file the first time it is asked for):

```python
area = rs["area"]                         # RaggedArray: flat values + offsets
series = area[1]                          # the whole series of track 1 (a view, no copy)
value = area[1, 0]                        # element 0 of track 1, like a[1, 0] of the dense array
```

**One value per track, at numpy speed** (`max`, `min`, `sum` and `mean` of a track with a NaN are NaN; take `np.nanmax(series)` for one track
if you need to skip them):

```python
has = area.lengths > 0                                # tracks that have entries
first_area = area.flat[area.offsets[:-1][has]]        # the first value of each of them
last_area = area.flat[area.offsets[1:][has] - 1]      # the last value
total_per_track = area.sum()
```

**Pick tracks by a 1-D variable, build the dense array of only those**:

```python
selected = np.nonzero(rs["is_mcs"] == 1)[0]          # track numbers (0-based; pixel track number = index + 1)
dense_area = area[selected].dense()                  # [n selected, longest selected track], NaN padded
```

**The dense `Dataset` of a selection, for code written for the old files**:

```python
ds = load_trackstats(path, tracks=selected, variables=["area", "base_time"])
longest = ds["area"].max("times")                    # xarray as before; tracks coordinate keeps the numbers of the file
```

**pandas: the tracks table** (indexed by the track number):

```python
tracks = rs.to_pandas("tracks")
mcs = tracks[tracks["is_mcs"] == 1]
summary = mcs["track_duration"].describe()
```

**pandas: the entries as a long table**, then the usual groupby / join / resample:

```python
entries = rs.to_pandas("entries", tracks=selected, variables=["area", "base_time"])
entries["time"] = pd.to_datetime(entries["base_time"], unit="s")
peak = entries.groupby("track")["area"].max()                # maximum area of every selected track
with_class = entries.merge(tracks[["track_duration"]], left_on="track", right_index=True)
```

**pandas: the merge / split fragments**, joined to the entries they are attached to:

```python
fragments = rs.to_pandas("fragments", tracks=selected)
count = fragments.groupby("kind").size()                      # 0 = merge, 1 = split
attached = fragments.merge(entries[["track", "k", "area"]], on=["track", "k"], suffixes=("_fragment", "_track"))
```

**Back to a dense netCDF file for the tools that need one** (the selection keeps the track numbers in the `tracks` coordinate):

```bash
python runscripts/convert_stats.py to-dense mcs_tracks_robust_X.rows mcs_only.nc --select is_mcs=1 --variables area majoraxis
python runscripts/convert_stats.py to-rows  mcs_tracks_robust_X.nc   mcs_tracks_robust_X.rows        # an old dense file
```

## For developers: a step in rows mode

The steps keep the numbers of the dense steps and change only where they live: a `(track, time)` pair is the position
`offsets[track] + k` in a flat array, the per-track tests are called with slices, and the result is assembled with
`rowstats.TableBuilder` (`add_track_var`, `add_entry_var`, `add_fragment_vars`, `build`), which also records dtypes, fill values and
attributes so that `rows_to_dense` gives back what the dense step writes.

| step | module | function |
|---|---|---|
| identifymcs | `pyflextrkr/identifymcs_rows.py` | `identifymcs_radar_rows` (the per-track tests of `identifymcs_radar.py` are reused) |
| regionprops | `pyflextrkr/regionprops_rows.py` | `compute_regionprops_rows` (the worker `regionprops_singlefile` is the dense one) |
| mapfeature | `pyflextrkr/mapfeature_rows.py` | `mapfeature_driver_rows` (the worker `map_feature` is the dense one) |
| speed | `pyflextrkr/movement_speed_rows.py` | `movement_speed_rows` (the FFT worker, the filter and the alignment are those of `movement_speed.py`) |

`runscripts/run_mcs_radar3d.py` picks the functions from `stats_format`. How the equality with the dense steps is tested:
`tests/test_identifymcs_rows.py`, `test_regionprops_rows.py`, `test_mapfeature_rows.py`, `test_movement_speed_rows.py` run both
steps on the same inputs and compare every column, dtype and value of `dense_to_rows(dense output)` with the tables of the rows
step (random tracks with chains of fragments, a full slot dimension, tracks of one and two entries, missing files, final-statistics
filters). `conus_tracking/scripts/check_rows_step.py` does the same on finished tracking.
