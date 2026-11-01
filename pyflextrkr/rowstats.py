"""
Row-based (ragged-by-track) storage of track statistics.

The dense stats files of the radar MCS pipeline (mcs_tracks_*, mcs_tracks_robust_*, mcs_tracks_final_*) hold [tracks, times]
and [tracks, times, slots] arrays that are almost empty: a track lives for `track_duration` steps and fills only the first
`track_duration` columns, and merge/split fragments fill a few of the 50 slots. In a month that is 180 GB of mostly fill
values. This module stores the same data as three tables:

  tracks     one row per track: the 1-D variables (and small 2-D ones such as lifecycle_index as columns name__0, name__1, ...)
  entries    one row per (track, time index), sorted by track then time index: the [tracks, times] variables
  fragments  one row per attached merge/split fragment: track, k (time index), kind (0 merge, 1 split), slot, cloudnumber,
             status, area

A track's entries are the contiguous block flat[offsets[i]:offsets[i+1]] of every entries column, so a variable is a flat numpy
array plus one offsets array (a ragged array). RaggedArray gives it the feel of the dense array: a[i] is the series of track
i, a[i, j] one element, a.max() one value per track, a.dense() the padded [tracks, times] array. The tables go to Parquet
(one folder, three files and meta.json); dense_to_rows / RowStats.to_xarray convert from and to the dense layout exactly, with
the same variable names, dtypes, fill values and attributes.

    rt = dense_to_rows("mcs_tracks_robust_X.nc")        # dense netCDF -> RowTables (block by block, small memory)
    write_rows(rt, "mcs_tracks_robust_X.rows")          # Parquet folder
    rs = RowStats.open("mcs_tracks_robust_X.rows")      # lazy
    rs["area"][5]                                       # series of track 5
    ds = rs.to_xarray(tracks=mcs_indices)               # dense xarray Dataset of a subset (today's dims and names)

Author: Zhe Feng | zhe.feng@pnnl.gov
"""
import json
import os

import numpy as np
import xarray as xr

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # only needed to read / write the Parquet folder
    pa = pq = None

# Kinds of fragments: the 3-D variables are named <kind>_<field> (merge_cloudnumber, split_area, ...); the kind code in the
# fragments table is the index in this tuple
FRAGMENT_KINDS = ("merge", "split")
# A fragment slot is filled where <kind>_cloudnumber is larger than 0
FRAGMENT_PRESENCE_FIELD = "cloudnumber"
META_FILE = "meta.json"


def _json_value(value):
    """
    A netCDF attribute as a JSON-serializable value.

    Args:
        value: numpy scalar, numpy array, string, number
            Attribute value.

    Returns:
        value: int, float, str or list
            Python types only (arrays become lists).
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _netcdf_attr(value):
    """
    An attribute value as it is after a netCDF round trip: a one-element array is a scalar, everything else as _json_value.

    Args:
        value: numpy scalar, numpy array, list, string, number
            Attribute value.

    Returns:
        value: int, float, str or list
            JSON-serializable.
    """
    value = _json_value(np.asarray(value) if isinstance(value, (list, tuple)) else value)
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def _fill_of(values_outside, attrs, dtype):
    """
    The value that a dense variable holds where it has no data.

    Args:
        values_outside: numpy.ndarray
            Values of the variable outside the valid entries of one block (may be empty).
        attrs: dictionary
            Attributes of the variable (uses `_FillValue` if no value is outside).
        dtype: numpy.dtype
            dtype of the variable.

    Returns:
        fill: float or int
            NaN for floats and -9999 for integers if nothing else is known.
    """
    if values_outside.size:
        return values_outside.flat[0].item()
    if "_FillValue" in attrs:
        return _json_value(attrs["_FillValue"])
    return float("nan") if np.dtype(dtype).kind == "f" else -9999


def _is_fill(values, fill):
    """
    Boolean array: which values equal the fill value (NaN equals NaN).

    Args:
        values: numpy.ndarray
        fill: float or int

    Returns:
        mask: numpy.ndarray of bool
    """
    if isinstance(fill, float) and np.isnan(fill):
        return np.isnan(values)
    return values == fill


class RowTables:
    """
    The three tables of a stats file in memory, plus what is needed to rebuild the dense layout.

    Attributes:
        tracks: dictionary {column name: numpy array [ntracks]}
        entries: dictionary {column name: numpy array [nentries]}, sorted by track then time index
        fragments: dictionary {column name: numpy array [nfragments]}, sorted by track, k, kind, slot
        meta: dictionary
            dims (sizes of tracks / times / slots), the dense dimension names, variable lists, dtypes, fill values,
            attributes, small coordinates.
    """

    def __init__(self, tracks, entries, fragments, meta):
        self.tracks = tracks
        self.entries = entries
        self.fragments = fragments
        self.meta = meta

    @property
    def duration(self):
        """
        Number of entries of every track (int array [ntracks]).
        """
        return np.asarray(self.tracks[self.meta["duration_var"]]).astype(np.int64)

    @property
    def offsets(self):
        """
        Start of every track in the entries columns plus the end of the last (int array [ntracks + 1]).
        """
        return np.concatenate(([0], np.cumsum(self.duration)))


def dense_to_rows(source, tracks_dim="tracks", times_dim="times", slots_dim="mergers", duration_var="track_duration",
                  block_tracks=4000):
    """
    Convert a dense stats file to row tables.

    The file is read block by block along the tracks, so the memory is that of one block (block_tracks x times x slots for a
    fragment variable), not of the file. The conversion is checked while it runs: outside the valid entries every dense
    variable must hold one constant fill value, and every fragment field must hold its fill value where the slot is empty;
    otherwise the dense layout could not be rebuilt exactly and a ValueError is raised.

    Args:
        source: string or xarray.Dataset
            Dense stats file, or a Dataset opened with mask_and_scale=False, decode_times=False.
        tracks_dim, times_dim, slots_dim: string (default: 'tracks', 'times', 'mergers')
            Names of the dimensions (tracks_dimname, times_dimname and the fragment slot dimension of the stats files).
        duration_var: string (default: 'track_duration')
            1-D variable with the number of valid entries of each track.
        block_tracks: int (default: 4000)
            Tracks per block.

    Returns:
        tables: RowTables
    """
    ds = xr.open_dataset(source, mask_and_scale=False, decode_times=False) if isinstance(source, str) else source
    ntracks, ntimes = ds.sizes[tracks_dim], ds.sizes[times_dim]
    nslots = ds.sizes.get(slots_dim, 0)

    # Sort the variables by their dimensions
    names_1d, names_2d, names_3d, names_tracks2d, coords_small = [], [], [], [], {}
    for name, var in ds.variables.items():
        dims = var.dims
        if dims == (tracks_dim,):
            names_1d.append(name)
        elif dims == (tracks_dim, times_dim):
            names_2d.append(name)
        elif dims == (tracks_dim, times_dim, slots_dim):
            names_3d.append(name)
        elif len(dims) == 2 and dims[0] == tracks_dim:
            names_tracks2d.append(name)
        elif tracks_dim not in dims:
            coords_small[name] = {"dims": list(dims), "values": var.values.tolist()}
        else:
            raise ValueError(f"variable {name} with dimensions {dims} cannot be stored as rows")
    for name in names_3d:
        kind = name.split("_", 1)[0]
        if kind not in FRAGMENT_KINDS or not name.split("_", 1)[1:]:
            raise ValueError(f"3-D variable {name} is not a <kind>_<field> fragment variable of {FRAGMENT_KINDS}")
    fields = sorted({name.split("_", 1)[1] for name in names_3d})
    if names_3d and FRAGMENT_PRESENCE_FIELD not in fields:
        raise ValueError(f"fragment variables need a <kind>_{FRAGMENT_PRESENCE_FIELD} variable")

    duration = ds[duration_var].values.astype(np.int64)
    if duration.min() < 0 or duration.max() > ntimes:
        raise ValueError(f"{duration_var} is outside 0..{ntimes}")

    # Tables are filled block by block
    parts_1d = {n: [] for n in names_1d}
    parts_t2d = {n: [] for n in names_tracks2d}
    parts_2d = {n: [] for n in names_2d}
    parts_frag = []
    fill = {}
    columns = np.arange(ntimes)[None, :]
    for a in range(0, ntracks, block_tracks):
        b = min(a + block_tracks, ntracks)
        sel = {tracks_dim: slice(a, b)}
        valid = columns < duration[a:b, None]
        for n in names_1d:
            parts_1d[n].append(ds[n].isel(sel).values)
        for n in names_tracks2d:
            parts_t2d[n].append(ds[n].isel(sel).values)
        for n in names_2d:
            block = ds[n].isel(sel).values
            _check_fill(fill, n, block[~valid], ds[n].attrs, block.dtype)
            parts_2d[n].append(block[valid])
        rows = {}
        for kind_code, kind in enumerate(FRAGMENT_KINDS):
            if not names_3d or f"{kind}_{FRAGMENT_PRESENCE_FIELD}" not in names_3d:
                continue
            present = ds[f"{kind}_{FRAGMENT_PRESENCE_FIELD}"].isel(sel).values > 0
            it, ik, islot = np.nonzero(present)
            rows[kind_code] = {"track": (it + a).astype(np.int32), "k": ik.astype(np.int32),
                               "kind": np.full(len(it), kind_code, dtype=np.int8), "slot": islot.astype(np.int16)}
            for field in fields:
                block = ds[f"{kind}_{field}"].isel(sel).values
                _check_fill(fill, f"{kind}_{field}", block[~present], ds[f"{kind}_{field}"].attrs, block.dtype)
                rows[kind_code][field] = block[present]
        parts_frag.append(rows)

    # Concatenate the blocks
    tracks = {n: np.concatenate(parts_1d[n]) for n in names_1d}
    for n in names_tracks2d:
        arr = np.concatenate(parts_t2d[n])
        for j in range(arr.shape[1]):
            tracks[f"{n}__{j}"] = arr[:, j]
    entries = {"track": np.repeat(np.arange(ntracks, dtype=np.int32), duration),
               "k": (np.arange(duration.sum()) - np.repeat(np.cumsum(duration) - duration, duration)).astype(np.int32)}
    for n in names_2d:
        entries[n] = np.concatenate(parts_2d[n]) if parts_2d[n] else np.array([])
    fragments = {}
    if names_3d:
        cols = ["track", "k", "kind", "slot"] + fields
        merged = {c: np.concatenate([rows[kc][c] for rows in parts_frag for kc in sorted(rows)]) for c in cols}
        order = np.lexsort((merged["slot"], merged["kind"], merged["k"], merged["track"]))
        fragments = {c: v[order] for c, v in merged.items()}

    meta = {
        "version": 1, "tracks_dim": tracks_dim, "times_dim": times_dim, "slots_dim": slots_dim, "duration_var": duration_var,
        "dims": {k: int(v) for k, v in ds.sizes.items()},
        "vars_1d": names_1d, "vars_tracks2d": {n: int(ds.sizes[ds[n].dims[1]]) for n in names_tracks2d},
        "vars_2d": names_2d, "vars_3d": names_3d, "fragment_fields": fields,
        "dims_of": {n: list(ds[n].dims) for n in names_1d + names_2d + names_3d + names_tracks2d},
        "dtypes": {n: str(ds[n].dtype) for n in names_1d + names_2d + names_3d + names_tracks2d},
        "fill": {n: ("nan" if isinstance(v, float) and np.isnan(v) else v) for n, v in fill.items()},
        "attrs": {n: {k: _json_value(v) for k, v in ds[n].attrs.items()} for n in list(ds.variables)},
        "global_attrs": {k: _json_value(v) for k, v in ds.attrs.items()},
        "coords": coords_small,
    }
    if isinstance(source, str):
        ds.close()
    return RowTables(tracks, entries, fragments, meta)


def _check_fill(fill, name, outside, attrs, dtype):
    """
    Check (and remember) that a variable holds one constant fill value where it has no data.

    Args:
        fill: dictionary
            {variable name: fill value}, filled in the first block and compared in the later ones.
        name: string
            Variable name.
        outside: numpy.ndarray
            Values of the variable outside the valid entries of this block.
        attrs: dictionary
            Attributes of the variable.
        dtype: numpy.dtype
            dtype of the variable.

    Raises:
        ValueError: if the values outside are not all the fill value.
    """
    if name not in fill:
        fill[name] = _fill_of(outside, attrs, dtype)
    if outside.size and not _is_fill(outside, fill[name]).all():
        raise ValueError(f"variable {name} does not hold one constant fill value outside its valid entries; "
                         f"the dense layout could not be rebuilt exactly")


def rows_to_dense(tables, tracks=None, width=None):
    """
    Rebuild the dense xarray Dataset (today's dimensions, names, dtypes, fill values, attributes) from row tables.

    Args:
        tables: RowTables
            The tables.
        tracks: array-like of int (default: None = all tracks)
            Track indices (0-based, in the order of the tables) to rebuild; the result has them in the given order.
        width: int (default: None)
            Length of the times dimension. None: the original length for all tracks, the longest selected track for a subset.

    Returns:
        ds: xarray.Dataset
            Dense dataset of the selected tracks.
    """
    meta = tables.meta
    tdim, kdim, sdim = meta["tracks_dim"], meta["times_dim"], meta["slots_dim"]
    ntracks = len(tables.duration)
    sel = np.arange(ntracks) if tracks is None else np.asarray(tracks, dtype=np.int64)
    dur = tables.duration[sel]
    offsets = tables.offsets[:-1][sel]
    if width is None:
        width = meta["dims"][kdim] if tracks is None else int(dur.max()) if len(dur) else 0

    # Position of every selected entry in the flat columns and in the dense array
    total = int(dur.sum())
    row = np.repeat(np.arange(len(sel)), dur)
    k = np.arange(total) - np.repeat(np.cumsum(dur) - dur, dur)
    flat_index = np.repeat(offsets, dur) + k
    if total and k.max() >= width:
        raise ValueError(f"width {width} is smaller than the longest selected track")

    data_vars = {}
    for n in meta["vars_1d"]:
        data_vars[n] = ([tdim], tables.tracks[n][sel], meta["attrs"].get(n, {}))
    for n, ncol in meta["vars_tracks2d"].items():
        arr = np.stack([tables.tracks[f"{n}__{j}"][sel] for j in range(ncol)], axis=1)
        data_vars[n] = (meta["dims_of"][n], arr, meta["attrs"].get(n, {}))
    for n in meta["vars_2d"]:
        out = np.full((len(sel), width), _fill_value(meta, n), dtype=meta["dtypes"][n])
        out[row, k] = tables.entries[n][flat_index]
        data_vars[n] = ([tdim, kdim], out, meta["attrs"].get(n, {}))
    if meta["vars_3d"]:
        new_index = np.full(ntracks, -1, dtype=np.int64)
        new_index[sel] = np.arange(len(sel))
        frag = tables.fragments
        keep = new_index[frag["track"]] >= 0
        for n in meta["vars_3d"]:
            kind, field = n.split("_", 1)
            use = keep & (frag["kind"] == FRAGMENT_KINDS.index(kind))
            out = np.full((len(sel), width, meta["dims"][sdim]), _fill_value(meta, n), dtype=meta["dtypes"][n])
            out[new_index[frag["track"][use]], frag["k"][use], frag["slot"][use]] = frag[field][use]
            data_vars[n] = ([tdim, kdim, sdim], out, meta["attrs"].get(n, {}))

    # Coordinates: the stored small ones, the times index for the chosen width
    coords = {}
    for n, c in meta["coords"].items():
        values = np.asarray(c["values"])
        coords[n] = (c["dims"], np.arange(width) if n == kdim and width != len(values) else values, meta["attrs"].get(n, {}))
    ds = xr.Dataset(data_vars, coords=coords, attrs=dict(meta["global_attrs"]))
    # The tracks coordinate is stored as a column of the tracks table when the dense file has one
    if tdim in tables.tracks and tdim not in ds.coords:
        ds = ds.assign_coords({tdim: ([tdim], tables.tracks[tdim][sel])})
    return ds


def _fill_value(meta, name):
    """
    Fill value of a variable from the meta data (the text 'nan' stands for NaN).

    Args:
        meta: dictionary
            RowTables.meta.
        name: string
            Variable name.

    Returns:
        fill: float or int
    """
    v = meta["fill"][name]
    return float("nan") if v == "nan" else v


def write_rows(tables, folder, compression="zstd"):
    """
    Write row tables to a folder: tracks.parquet, entries.parquet, fragments.parquet and meta.json.

    Args:
        tables: RowTables
            The tables.
        folder: string
            Output folder (created).
        compression: string (default: 'zstd')
            Parquet compression.
    """
    os.makedirs(folder, exist_ok=True)
    for name in ("tracks", "entries", "fragments"):
        columns = getattr(tables, name)
        if columns:
            pq.write_table(pa.table(columns), os.path.join(folder, f"{name}.parquet"), compression=compression)
    with open(os.path.join(folder, META_FILE), "w") as f:
        json.dump(tables.meta, f)


class RaggedArray:
    """
    One variable of the entries table: a flat array plus offsets, with the access of a dense [tracks, times] array.

    a[i] is the series of track i (a view of the flat array), a[i, j] the element j of track i, a[[i, j, ...]] or a[mask] a
    RaggedArray of those tracks; max / min / sum / mean give one value per track; dense() the padded array.
    """

    def __init__(self, flat, offsets, fill=np.nan):
        """
        Args:
            flat: numpy.ndarray [nentries]
                Values of all tracks, one track after the other.
            offsets: numpy.ndarray [ntracks + 1]
                Start of every track in `flat` plus the end of the last.
            fill: float or int (default: NaN)
                Value of the padding in dense().
        """
        self.flat = flat
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.fill = fill

    def __len__(self):
        return len(self.offsets) - 1

    @property
    def lengths(self):
        """
        Number of entries of every track (int array [ntracks]).
        """
        return np.diff(self.offsets)

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            i = int(key) % len(self) if key < 0 else int(key)
            return self.flat[self.offsets[i]:self.offsets[i + 1]]
        if isinstance(key, tuple) and len(key) == 2 and isinstance(key[0], (int, np.integer)):
            series = self[key[0]]
            return series[key[1]]
        idx = np.nonzero(key)[0] if getattr(key, "dtype", None) == bool else np.arange(len(self))[key]
        lengths = self.lengths[idx]
        starts = self.offsets[:-1][idx]
        total = int(lengths.sum())
        within = np.arange(total) - np.repeat(np.cumsum(lengths) - lengths, lengths)
        flat = self.flat[np.repeat(starts, lengths) + within]
        return RaggedArray(flat, np.concatenate(([0], np.cumsum(lengths))), self.fill)

    def _reduce(self, ufunc):
        """
        Per-track reduction with a ufunc (np.maximum, np.minimum, np.add); NaN for tracks without entries.
        """
        # reduceat over the starts of the non-empty tracks only: the segments between them are exactly the tracks
        out = np.full(len(self), np.nan)
        nonempty = self.lengths > 0
        if nonempty.any():
            out[nonempty] = ufunc.reduceat(self.flat, self.offsets[:-1][nonempty])
        return out

    def max(self):
        """
        Maximum of every track (NaN-propagating, as np.maximum): float array [ntracks].
        """
        return self._reduce(np.maximum)

    def min(self):
        """
        Minimum of every track: float array [ntracks].
        """
        return self._reduce(np.minimum)

    def sum(self):
        """
        Sum of every track: float array [ntracks].
        """
        return self._reduce(np.add)

    def mean(self):
        """
        Mean of every track: float array [ntracks].
        """
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.sum() / self.lengths

    def dense(self, width=None):
        """
        The padded [tracks, width] array of today's dense layout.

        Args:
            width: int (default: None = the longest track)
                Number of columns.

        Returns:
            arr: numpy.ndarray [ntracks, width]
                Values of the tracks, `fill` after the end of each.
        """
        lengths = self.lengths
        width = int(lengths.max()) if width is None and len(lengths) else (width or 0)
        out = np.full((len(self), width), self.fill, dtype=self.flat.dtype)
        row = np.repeat(np.arange(len(self)), lengths)
        out[row, np.arange(len(self.flat)) - np.repeat(self.offsets[:-1], lengths)] = self.flat
        return out


class RowStats:
    """
    A row-table folder opened lazily: the tracks table is loaded, the columns of the other tables when they are asked for.

        rs = RowStats.open(folder)
        rs["track_duration"]            # numpy array [ntracks] (a 1-D variable)
        rs["area"]                      # RaggedArray (a [tracks, times] variable)
        rs.to_xarray(tracks=idx)        # dense Dataset of a subset of the tracks
    """

    def __init__(self, folder, meta, tracks):
        self.folder = folder
        self.meta = meta
        self.tracks = tracks
        self._entries = {}
        self._fragments = None

    @classmethod
    def open(cls, folder):
        """
        Open a folder written by write_rows.

        Args:
            folder: string
                Folder with meta.json and the Parquet files.

        Returns:
            rs: RowStats
        """
        with open(os.path.join(folder, META_FILE)) as f:
            meta = json.load(f)
        table = pq.read_table(os.path.join(folder, "tracks.parquet"))
        tracks = {n: table.column(n).to_numpy() for n in table.column_names}
        return cls(folder, meta, tracks)

    @property
    def duration(self):
        """
        Number of entries of every track (int array [ntracks]).
        """
        return np.asarray(self.tracks[self.meta["duration_var"]]).astype(np.int64)

    @property
    def offsets(self):
        """
        Offsets of the tracks in the entries columns (int array [ntracks + 1]).
        """
        return np.concatenate(([0], np.cumsum(self.duration)))

    def _column(self, table, name):
        """
        One column of a Parquet table as a numpy array (read from the file).
        """
        return pq.read_table(os.path.join(self.folder, f"{table}.parquet"), columns=[name]).column(name).to_numpy()

    def __getitem__(self, name):
        if name in self.meta["vars_1d"]:
            return self.tracks[name]
        if name in self.meta["vars_2d"]:
            if name not in self._entries:
                self._entries[name] = self._column("entries", name)
            return RaggedArray(self._entries[name], self.offsets, _fill_value(self.meta, name))
        raise KeyError(name)

    def tables(self, variables=None):
        """
        Load the tables (all columns, or only those of the given variables) as RowTables.

        Args:
            variables: list of strings (default: None = all)
                Variables of the entries table to load; 1-D variables and fragments are always loaded.

        Returns:
            tables: RowTables
        """
        names = self.meta["vars_2d"] if variables is None else [v for v in variables if v in self.meta["vars_2d"]]
        entries = {"track": self._column("entries", "track"), "k": self._column("entries", "k")}
        for n in names:
            entries[n] = self._column("entries", n)
        fragments = {}
        if self.meta["vars_3d"]:
            table = pq.read_table(os.path.join(self.folder, "fragments.parquet"))
            fragments = {c: table.column(c).to_numpy() for c in table.column_names}
        meta = dict(self.meta, vars_2d=names)
        return RowTables(self.tracks, entries, fragments, meta)

    def to_pandas(self, table="entries", tracks=None, variables=None):
        """
        One of the tables as a pandas DataFrame, for all or a subset of the tracks.

        Args:
            table: string (default: 'entries')
                'tracks' (one row per track, indexed by the track number), 'entries' (one row per (track, time index): columns
                track, k and the [tracks, times] variables) or 'fragments' (one row per attached merge / split fragment: track,
                k, kind (0 merge, 1 split), slot, cloudnumber, status, area).
            tracks: array-like of int (default: None = all tracks)
                Track indices (0-based) to include. The track column keeps the number of the track in the whole file.
            variables: list of strings (default: None = all)
                [tracks, times] variables to load for the entries table.

        Returns:
            df: pandas.DataFrame
        """
        import pandas as pd
        tables = self.tables(variables)
        select = None if tracks is None else np.asarray(tracks, dtype=np.int64)
        if table == "tracks":
            df = pd.DataFrame({n: v for n, v in tables.tracks.items()})
            return df if select is None else df.iloc[select]
        if table == "entries":
            columns = tables.entries
            if select is None:
                return pd.DataFrame(columns)
            index, _ = gather_index(tables.offsets, select)
            return pd.DataFrame({n: v[index] for n, v in columns.items()})
        if table == "fragments":
            columns = tables.fragments
            keep = np.ones(len(columns["track"]), dtype=bool) if select is None else np.isin(columns["track"], select)
            return pd.DataFrame({n: v[keep] for n, v in columns.items()})
        raise ValueError(f"table must be 'tracks', 'entries' or 'fragments', not '{table}'")

    def to_xarray(self, tracks=None, variables=None, width=None):
        """
        Dense xarray Dataset of all or a subset of the tracks, with today's dimensions, names, dtypes and fill values.

        Args:
            tracks: array-like of int (default: None = all tracks)
                Track indices (0-based) to include.
            variables: list of strings (default: None = all)
                [tracks, times] variables to include (the 1-D variables and fragments are always included).
            width: int (default: None)
                Length of the times dimension (see rows_to_dense).

        Returns:
            ds: xarray.Dataset
        """
        return rows_to_dense(self.tables(variables), tracks=tracks, width=width)


# ---------------------------------------------------------------------------
# Row tables made by the pipeline steps (no dense array is ever built)
# ---------------------------------------------------------------------------

def gather_index(offsets, keep):
    """
    Positions in the flat entries columns of all the entries of some tracks, and the offsets of the selection.

    Args:
        offsets: int array [ntracks + 1]
            Start of every track in the flat columns plus the end of the last.
        keep: int array [nkeep]
            Tracks to select (0-based), in the order they should have in the result.

    Returns:
        index: int array [sum of the durations of the kept tracks]
            flat[index] is the flat column of the selected tracks (track after track, in the order of keep).
        new_offsets: int array [nkeep + 1]
            Offsets of the selected tracks in flat[index].
    """
    keep = np.asarray(keep, dtype=np.int64)
    starts = offsets[keep]
    lengths = offsets[keep + 1] - starts
    new_offsets = np.concatenate(([0], np.cumsum(lengths)))
    index = np.arange(new_offsets[-1]) - np.repeat(new_offsets[:-1], lengths) + np.repeat(starts, lengths)
    return index, new_offsets


def load_sparse_entries(statistics_file, tracks_dimname="tracks", times_dimname="times", duration_var="track_duration",
                        sparse_dimname="sparse_index"):
    """
    Read a sparse trackstats file as flat entries columns instead of sparse matrices.

    The sparse file already is an entries table: for every (track, time index) pair one value per variable on the sparse
    dimension. Here the values are put at offsets[track] + time index of flat arrays with one position for every valid entry of
    every track, so that flat[offsets[i]:offsets[i + 1]] is what the sparse matrix gives for the row i of its first
    track_duration[i] columns (positions that the file has no entry for hold 0, as the sparse matrices do).

    Args:
        statistics_file: string
            Sparse trackstats file.
        tracks_dimname, times_dimname: string (default: 'tracks', 'times')
            Names of the tracks and times dimensions; the index variables of the file are <name>_indices.
        duration_var: string (default: 'track_duration')
            1-D variable with the number of valid entries of each track.
        sparse_dimname: string (default: 'sparse_index')
            Name of the sparse dimension.

    Returns:
        ds_1d: xarray.Dataset
            The variables without the sparse dimension, loaded (mask_and_scale=False, decode_times=False).
        flat: dictionary
            {variable name: numpy array [nentries]} of the sparse variables (not the two index variables), dtype of the file.
        attrs: dictionary
            {variable name: attributes} of the sparse variables.
        offsets: int array [ntracks + 1]
            Start of every track in the flat arrays plus the end of the last.
        longest: int
            Largest time index in the file plus one (0 for an empty file), as resolve_sparse_max_trackduration sees it.
    """
    ds_all = xr.open_dataset(statistics_file, mask_and_scale=False, decode_times=False)
    tracks_idx = ds_all[f"{tracks_dimname}_indices"].values.astype(np.int64)
    times_idx = ds_all[f"{times_dimname}_indices"].values.astype(np.int64)
    ds_1d = ds_all.drop_dims(sparse_dimname).load()
    duration = ds_1d[duration_var].values.astype(np.int64)
    offsets = np.concatenate(([0], np.cumsum(duration)))

    # Every sparse entry has to be one of the valid entries of its track, each of them once
    if len(tracks_idx) and (times_idx.min() < 0 or (times_idx >= duration[tracks_idx]).any()):
        raise ValueError(f"{statistics_file}: entries outside the first {duration_var} time indices of their track "
                         f"cannot be stored as rows")
    position = offsets[tracks_idx] + times_idx
    if len(position) and np.bincount(position, minlength=int(offsets[-1])).max() > 1:
        raise ValueError(f"{statistics_file}: a (track, time) pair is in the sparse file more than once")

    flat, attrs = {}, {}
    for name, var in ds_all.data_vars.items():
        if var.dims[0] != sparse_dimname or name in (f"{tracks_dimname}_indices", f"{times_dimname}_indices"):
            continue
        values = var.values
        column = np.zeros(int(offsets[-1]), dtype=values.dtype)
        column[position] = values
        flat[name], attrs[name] = column, dict(var.attrs)
    ds_all.close()
    return ds_1d, flat, attrs, offsets, (int(times_idx.max()) + 1 if len(times_idx) else 0)


class TableBuilder:
    """
    Assemble RowTables from flat arrays, the way dense_to_rows would have made them from the dense file with the same variables.

    A pipeline step in rows mode adds its variables here (1-D variables, [tracks, times] variables as flat columns, the
    merge / split fragments as rows) and builds the tables; the tables describe the dense layout (dims, dtypes, fill values,
    attributes) so that rows_to_dense gives back what the dense code would have written.

        b = TableBuilder(duration, width=800, nslots=50, global_attrs={"Title": "..."})
        b.add_track_var("is_mcs", is_mcs, {"long_name": "..."})
        b.add_entry_var("area", flat_area, {"units": "km^2", "_FillValue": np.nan})
        b.add_fragment_vars(columns, {"merge_cloudnumber": ({...}, -9999), ...})
        tables = b.build()
    """

    def __init__(self, duration, width, nslots=None, tracks_dim="tracks", times_dim="times", slots_dim="mergers",
                 duration_var="track_duration", global_attrs=None, tracks_coord=True):
        """
        Args:
            duration: int array [ntracks]
                Number of valid entries of every track (the duration_var column is added by the caller as a 1-D variable).
            width: int
                Length of the times dimension of the dense layout (at least the longest duration).
            nslots: int (default: None)
                Length of the fragment slots dimension of the dense layout; None: no fragment variables.
            tracks_dim, times_dim, slots_dim, duration_var: string (default: 'tracks', 'times', 'mergers', 'track_duration')
                Names of the dimensions and of the duration variable.
            global_attrs: dictionary (default: None)
                Global attributes of the dense layout.
            tracks_coord: bool (default: True)
                Store the tracks coordinate (0..ntracks-1) as a column of the tracks table, as the dense files have it.
        """
        self.duration = np.asarray(duration).astype(np.int64)
        if len(self.duration) and (self.duration.min() < 0 or self.duration.max() > width):
            raise ValueError(f"the durations are outside 0..{width}")
        self.width, self.nslots = int(width), nslots
        self.tdim, self.kdim, self.sdim, self.duration_var = tracks_dim, times_dim, slots_dim, duration_var
        self.global_attrs = dict(global_attrs or {})
        self.tracks, self.entries, self.fragment_columns = {}, {}, {}
        self.vars_1d, self.vars_tracks2d, self.vars_2d, self.vars_3d = [], {}, [], []
        self.dtypes, self.fill, self.attrs, self.dims_of, self.coords, self.extra_dims = {}, {}, {}, {}, {}, {}
        if tracks_coord:
            self.add_track_var(tracks_dim, np.arange(len(self.duration)))
        self.add_coord(times_dim, np.arange(self.width))
        if nslots is not None:
            self.add_coord(slots_dim, np.arange(nslots))

    @classmethod
    def from_tables(cls, tables):
        """
        A builder that holds the variables of existing tables, to replace or add variables (the step after identifymcs).

        Args:
            tables: RowTables
                Tables read with read_tables (all [tracks, times] variables loaded).

        Returns:
            builder: TableBuilder
                Its build() gives the same tables until a variable is added or replaced.
        """
        meta = tables.meta
        tdim, kdim, sdim = meta["tracks_dim"], meta["times_dim"], meta["slots_dim"]
        builder = cls(tables.duration, meta["dims"][kdim], nslots=meta["dims"].get(sdim) if meta["vars_3d"] else None,
                      tracks_dim=tdim, times_dim=kdim, slots_dim=sdim, duration_var=meta["duration_var"],
                      global_attrs=meta["global_attrs"], tracks_coord=False)
        builder.tracks = dict(tables.tracks)
        builder.entries = {n: tables.entries[n] for n in meta["vars_2d"]}
        builder.fragment_columns = dict(tables.fragments)
        builder.vars_1d, builder.vars_tracks2d = list(meta["vars_1d"]), dict(meta["vars_tracks2d"])
        builder.vars_2d, builder.vars_3d = list(meta["vars_2d"]), list(meta["vars_3d"])
        builder.dtypes, builder.fill = dict(meta["dtypes"]), dict(meta["fill"])
        builder.attrs = {k: dict(v) for k, v in meta["attrs"].items()}
        builder.dims_of = {k: list(v) for k, v in meta["dims_of"].items()}
        builder.coords = {k: dict(v) for k, v in meta["coords"].items()}
        builder.extra_dims = {k: v for k, v in meta["dims"].items() if k not in (tdim, kdim, sdim)}
        return builder

    def _remember(self, name, attrs, dtype=None, dims=None):
        """Keep the attributes (JSON-ready), dtype and dimensions of a variable."""
        self.attrs[name] = {k: _netcdf_attr(v) for k, v in (attrs or {}).items()}
        if dtype is not None:
            self.dtypes[name] = str(np.dtype(dtype))
        if dims is not None:
            self.dims_of[name] = list(dims)

    def add_track_var(self, name, values, attrs=None):
        """
        Add a 1-D variable [ntracks] (a column of the tracks table); an existing variable of that name is replaced.

        Args:
            name: string
                Variable name.
            values: array [ntracks]
                Values.
            attrs: dictionary (default: None)
                Attributes.
        """
        values = np.asarray(values)
        if values.shape != (len(self.duration),):
            raise ValueError(f"{name}: expected {len(self.duration)} values, got shape {values.shape}")
        self.tracks[name] = values
        if name not in self.vars_1d:
            self.vars_1d.append(name)
        self._remember(name, attrs, values.dtype, [self.tdim])

    def add_track2d_var(self, name, values, dim2, attrs=None):
        """
        Add a small 2-D variable [ntracks, ncol] such as lifecycle_index (columns name__0, name__1, ... of the tracks table).

        Args:
            name: string
                Variable name.
            values: array [ntracks, ncol]
                Values.
            dim2: string
                Name of the second dimension.
            attrs: dictionary (default: None)
                Attributes.
        """
        values = np.asarray(values)
        if values.ndim != 2 or values.shape[0] != len(self.duration):
            raise ValueError(f"{name}: expected shape ({len(self.duration)}, ncol), got {values.shape}")
        for j in range(values.shape[1]):
            self.tracks[f"{name}__{j}"] = values[:, j]
        self.vars_tracks2d[name] = int(values.shape[1])
        self.extra_dims[dim2] = int(values.shape[1])
        self._remember(name, attrs, values.dtype, [self.tdim, dim2])

    def add_entry_var(self, name, flat, attrs=None, fill=None):
        """
        Add a [tracks, times] variable as a flat column (a column of the entries table); an existing one is replaced.

        Args:
            name: string
                Variable name.
            flat: array [nentries]
                Values of the valid entries, track after track (nentries = sum of the durations).
            attrs: dictionary (default: None)
                Attributes.
            fill: int, float or NaN (default: None)
                Value of the dense variable where the track has no entry; None: the _FillValue attribute.
        """
        flat = np.asarray(flat)
        if flat.shape != (int(self.duration.sum()),):
            raise ValueError(f"{name}: expected {int(self.duration.sum())} entries, got shape {flat.shape}")
        if fill is None:
            if "_FillValue" not in (attrs or {}):
                raise ValueError(f"{name}: no fill value given and no _FillValue attribute")
            fill = attrs["_FillValue"]
        self.entries[name] = flat
        if name not in self.vars_2d:
            self.vars_2d.append(name)
        self.fill[name] = "nan" if isinstance(_json_value(fill), float) and np.isnan(_json_value(fill)) else _json_value(fill)
        self._remember(name, attrs, flat.dtype, [self.tdim, self.kdim])

    def add_fragment_vars(self, columns, variables):
        """
        Add the merge / split fragments: the rows, and the attributes and fill value of the <kind>_<field> variables.

        Args:
            columns: dictionary
                {'track': int, 'k': int, 'kind': int8 (index in FRAGMENT_KINDS), 'slot': int, <field>: values} arrays, one
                element per attached fragment (any order; build() sorts them by track, k, kind, slot).
            variables: dictionary
                {variable name (<kind>_<field>, in the order of the dense layout): (attributes, fill value)}.
        """
        self.fragment_columns = dict(columns)
        for name, (attrs, fill) in variables.items():
            kind, field = name.split("_", 1)
            if kind not in FRAGMENT_KINDS or field not in columns:
                raise ValueError(f"{name}: not a <kind>_<field> fragment variable of the columns {sorted(columns)}")
            self.vars_3d.append(name)
            self.fill[name] = "nan" if isinstance(_json_value(fill), float) and np.isnan(_json_value(fill)) else _json_value(fill)
            self._remember(name, attrs, np.asarray(columns[field]).dtype, [self.tdim, self.kdim, self.sdim])

    def add_coord(self, name, values, attrs=None, dims=None):
        """
        Add a small coordinate that has no tracks dimension (times, the fragment slots, ...).

        Args:
            name: string
                Coordinate name.
            values: array
                Values.
            attrs: dictionary (default: None)
                Attributes.
            dims: list of strings (default: None = [name])
                Dimensions.
        """
        values = np.asarray(values)
        dims = [name] if dims is None else list(dims)
        self.coords[name] = {"dims": dims, "values": values.tolist()}
        self.extra_dims[dims[0]] = len(values)
        self._remember(name, attrs)

    def build(self):
        """
        Make the tables.

        Returns:
            tables: RowTables
                The same tables and meta data that dense_to_rows makes from the dense file of these variables.
        """
        ntracks, total = len(self.duration), int(self.duration.sum())
        entries = {"track": np.repeat(np.arange(ntracks, dtype=np.int32), self.duration),
                   "k": (np.arange(total) - np.repeat(np.cumsum(self.duration) - self.duration, self.duration)).astype(np.int32)}
        entries.update(self.entries)
        fragments = {}
        if self.vars_3d:
            order = np.lexsort(tuple(self.fragment_columns[c] for c in ("slot", "kind", "k", "track")))
            fragments = {c: v[order] for c, v in self.fragment_columns.items()}
        fields = sorted({n.split("_", 1)[1] for n in self.vars_3d})
        dims = {self.tdim: ntracks, self.kdim: self.width}
        if self.nslots is not None:
            dims[self.sdim] = int(self.nslots)
        dims.update({k: v for k, v in self.extra_dims.items() if k not in dims})
        meta = {
            "version": 1, "tracks_dim": self.tdim, "times_dim": self.kdim, "slots_dim": self.sdim,
            "duration_var": self.duration_var, "dims": dims,
            "vars_1d": list(self.vars_1d), "vars_tracks2d": dict(self.vars_tracks2d),
            "vars_2d": list(self.vars_2d), "vars_3d": list(self.vars_3d), "fragment_fields": fields,
            "dims_of": {n: self.dims_of[n] for n in self.vars_1d + self.vars_2d + self.vars_3d + list(self.vars_tracks2d)},
            "dtypes": {n: self.dtypes[n] for n in self.vars_1d + self.vars_2d + self.vars_3d + list(self.vars_tracks2d)},
            "fill": dict(self.fill), "attrs": dict(self.attrs),
            "global_attrs": {k: _netcdf_attr(v) for k, v in self.global_attrs.items()}, "coords": dict(self.coords),
        }
        return RowTables(dict(self.tracks), entries, fragments, meta)


def read_tables(path, variables=None, **dense_kwargs):
    """
    Row tables of a stats file in either format.

    Args:
        path: string
            A row folder written by write_rows, or a dense netCDF file (converted with dense_to_rows).
        variables: list of strings (default: None = all)
            [tracks, times] variables to load from a row folder.
        **dense_kwargs:
            Arguments of dense_to_rows for a dense file (block_tracks, ...).

    Returns:
        tables: RowTables
    """
    if os.path.isdir(path):
        return RowStats.open(path).tables(variables)
    return dense_to_rows(path, **dense_kwargs)


def table_differences(a, b, ignore_global_attrs=("Created_on",)):
    """
    Compare two RowTables (the check that a rows-mode step gives what the dense step gives).

    Args:
        a, b: RowTables
            The tables.
        ignore_global_attrs: tuple of strings (default: ('Created_on',))
            Global attributes that may differ (time stamps).

    Returns:
        differences: list of strings
            What differs: table columns that are missing or have another dtype or values (NaN equals NaN), and the entries
            of the meta data. Empty if the tables are identical.
    """
    out = []
    for table in ("tracks", "entries", "fragments"):
        ca, cb = getattr(a, table), getattr(b, table)
        for name in sorted(set(ca) ^ set(cb)):
            out.append(f"{table}.{name}: only in {'first' if name in ca else 'second'}")
        for name in sorted(set(ca) & set(cb)):
            x, y = np.asarray(ca[name]), np.asarray(cb[name])
            if x.dtype != y.dtype:
                out.append(f"{table}.{name}: dtype {x.dtype} / {y.dtype}")
            elif x.shape != y.shape or not np.array_equal(x, y, equal_nan=x.dtype.kind == "f"):
                out.append(f"{table}.{name}: values differ")
    ma, mb = dict(a.meta), dict(b.meta)
    for m in (ma, mb):
        m["global_attrs"] = {k: v for k, v in m["global_attrs"].items() if k not in ignore_global_attrs}
        for key in ("vars_1d", "vars_2d", "vars_3d", "fragment_fields"):        # the order of the names has no meaning
            m[key] = sorted(m[key])
    canon = lambda value: json.dumps(value, sort_keys=True)                      # NaN equals NaN, key order has no meaning
    for key in sorted(set(ma) | set(mb)):
        if canon(ma.get(key)) != canon(mb.get(key)):
            if isinstance(ma.get(key), dict) and isinstance(mb.get(key), dict):
                sub = sorted(k for k in set(ma[key]) | set(mb[key]) if canon(ma[key].get(k)) != canon(mb[key].get(k)))
                out.append(f"meta.{key}: differs in {sub[:8]}")
            else:
                out.append(f"meta.{key}: differs")
    return out


def load_trackstats(path, tracks=None, variables=None, width=None, mask_and_scale=True, decode_times=True, tracks_dim="tracks",
                    times_dim="times", slots_dim="mergers", fragments=True):
    """
    Dense xarray Dataset of a track statistics file in either format: the helper for code that was written for the dense files.

    A drop-in for xarray.open_dataset(path, mask_and_scale=..., decode_times=...) on the statistics files: the same call reads a dense
    netCDF file (mcs_tracks_robust_X.nc) and a row folder (mcs_tracks_robust_X.rows), and a file name ending in .nc whose file does not
    exist but whose row folder does (so scripts that build the old file names keep working). The result has the dimensions, names,
    dtypes, fill values and attributes of the dense file.

    Args:
        path: string
            Dense netCDF file, row folder, or the .nc name of a statistics file that only exists as a row folder.
        tracks: array-like of int (default: None = all tracks)
            Track indices (0-based) to load, in the given order. For a row folder only these tracks are built, and the times
            dimension is as long as their longest track (pass width to choose another length).
        variables: list of strings (default: None = all)
            [tracks, times] variables to load; the 1-D variables and the merge / split variables are always loaded.
        width: int (default: None)
            Length of the times dimension of a subset of tracks from a row folder (rows_to_dense).
        mask_and_scale: bool (default: True)
            As in xarray.open_dataset: fill values to NaN and scale factors applied. False gives the raw values and the
            _FillValue attributes, as the pipeline steps read the files.
        decode_times: bool (default: True)
            As in xarray.open_dataset: time units decoded (False: the numbers).
        tracks_dim, times_dim, slots_dim: string (default: 'tracks', 'times', 'mergers')
            Names of the dimensions of a dense file.
        fragments: bool (default: True)
            False: leave out the [tracks, times, slots] merge / split fragment variables (other three-dimensional variables, e.g.
            [tracks, times, nmaxpf], stay). With variables=[] the result then has only the per-track variables, the cheap first
            look at a file (a row folder never builds a dense array for them).

    Returns:
        ds: xarray.Dataset
    """
    folder = path if os.path.isdir(path) else None
    if folder is None and not os.path.exists(path) and path.endswith(".nc") and os.path.isdir(path[:-3] + ".rows"):
        folder = path[:-3] + ".rows"
    if folder is not None:
        tables = RowStats.open(folder).tables(variables)
        if not fragments:
            tables = RowTables(tables.tracks, tables.entries, {}, dict(tables.meta, vars_3d=[]))
        ds = rows_to_dense(tables, tracks=tracks, width=width)
        return xr.decode_cf(ds, mask_and_scale=mask_and_scale, decode_times=decode_times)
    ds = xr.open_dataset(path, mask_and_scale=mask_and_scale, decode_times=decode_times)
    if variables is not None:
        ds = ds.drop_vars([v for v in ds.data_vars if ds[v].dims == (tracks_dim, times_dim) and v not in variables])
    if not fragments:
        ds = ds.drop_vars([v for v in ds.data_vars if ds[v].dims == (tracks_dim, times_dim, slots_dim)])
    if tracks is not None:
        ds = ds.isel({tracks_dim: np.asarray(tracks, dtype=np.int64)})
    return ds


def stats_exists(path):
    """
    Does a track statistics file exist, in either format? The check for code that asked os.path.isfile for the dense file.

    Args:
        path: string
            Dense netCDF file, row folder, or the .nc name of a statistics file that may only exist as a row folder
            (the names load_trackstats accepts).

    Returns:
        exists: bool
    """
    if os.path.exists(path):
        return True
    return path.endswith(".nc") and os.path.isdir(path[:-3] + ".rows")


def rows_folder(path):
    """
    The row folder of a statistics file name: the folder itself, or the folder next to a .nc name that only exists as rows.

    Args:
        path: string
            Row folder, or the .nc name of a statistics file whose row folder exists.

    Returns:
        folder: string

    Raises:
        FileNotFoundError: if neither exists.
    """
    if os.path.isdir(path):
        return path
    if path.endswith(".nc") and os.path.isdir(path[:-3] + ".rows"):
        return path[:-3] + ".rows"
    raise FileNotFoundError(f"no row folder: {path}")


def dense_netcdf_name(path):
    """
    Name of the dense netCDF file that belongs to a row folder: the folder name with .rows replaced by .nc, next to it. The row
    folders are named like the dense files of the old steps (mcs_tracks_<start>_<end>.rows, mcs_tracks_robust_..., mcs_tracks_final_...),
    so the dense file gets the name the old step gave it.

    Args:
        path: string
            Row folder (.rows), or the .nc name of a file that only exists as a row folder.

    Returns:
        name: string
    """
    return path[:-5] + ".nc" if path.endswith(".rows") else path if path.endswith(".nc") else path.rstrip("/") + ".nc"


def write_dense_netcdf(path, outfile=None, tracks=None, variables=None, block_tracks=None, overwrite=False):
    """
    Write a row-table statistics file as a dense netCDF file, as the dense steps of the pipeline wrote it (the layout of every
    reader of the old files): same dimensions, variable names, dtypes, fill values and attributes, an unlimited tracks dimension and
    zlib compression. dense_to_rows followed by this function gives back the original file variable by variable.

    A whole month as dense arrays is 180 GB: pass tracks / variables to write a part (a year does not fit), or block_tracks to write
    the file block by block so that only one block of tracks is in memory at a time. The result does not depend on the block size.

    Args:
        path: string
            Row folder, or the .nc name of a statistics file that exists only as a row folder.
        outfile: string (default: None)
            Output file. None: the name of the dense file of the old step, the row folder name with .rows replaced by .nc in the same
            folder (dense_netcdf_name).
        tracks: array-like of int (default: None = all tracks)
            Track indices (0-based) to write, in the given order; the times dimension is then as long as their longest track.
        variables: list of strings (default: None = all)
            [tracks, times] variables to write (the 1-D variables and the merge / split variables are always written).
        block_tracks: int (default: None = all tracks in one block)
            Number of tracks built and written at a time.
        overwrite: bool (default: False)
            Replace an existing output file; False: an existing file is an error.

    Returns:
        outfile: string
            The file that was written.

    Raises:
        FileNotFoundError: no row folder.
        FileExistsError: the output file exists and overwrite is False.
    """
    import netCDF4
    folder = rows_folder(path)
    outfile = dense_netcdf_name(folder) if outfile is None else outfile
    if os.path.exists(outfile) and not overwrite:
        raise FileExistsError(f"{outfile} exists (overwrite=True replaces it)")
    rs = RowStats.open(folder)
    tables = rs.tables(variables)
    tdim = rs.meta["tracks_dim"]
    ntracks = len(tables.duration)
    sel = np.arange(ntracks) if tracks is None else np.asarray(tracks, dtype=np.int64)
    # One width for all blocks: the original times dimension, or the longest selected track
    width = rs.meta["dims"][rs.meta["times_dim"]] if tracks is None else int(tables.duration[sel].max()) if len(sel) else 0
    step = len(sel) if not block_tracks or block_tracks >= len(sel) else int(block_tracks)
    step = max(step, 1)
    encoding = lambda ds: {v: dict(zlib=True) for v in ds.data_vars}
    if os.path.exists(outfile):
        os.remove(outfile)

    # First block: written with xarray (creates the dimensions, variables, attributes, fill values and the unlimited tracks dimension)
    first = rows_to_dense(tables, tracks=sel[:step], width=width)
    first.to_netcdf(outfile, mode="w", format="NETCDF4", unlimited_dims=tdim, encoding=encoding(first))
    if step >= len(sel):
        return outfile

    # The other blocks: appended along the unlimited tracks dimension, raw values (no mask or scale applied)
    with netCDF4.Dataset(outfile, "a") as nc:
        nc.set_auto_maskandscale(False)
        for start in range(step, len(sel), step):
            block = rows_to_dense(tables, tracks=sel[start:start + step], width=width)
            stop = start + block.sizes[tdim]
            for name in block.variables:
                if block[name].dims and block[name].dims[0] == tdim:
                    nc[name][start:stop] = block[name].values
    return outfile
