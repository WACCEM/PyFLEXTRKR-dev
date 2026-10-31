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
