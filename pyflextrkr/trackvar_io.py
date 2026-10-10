"""
Input of the track post-processing steps (trackpcp_driver): the data a tracked feature is measured on, which need not
be part of the tracking input: its own file name convention, time axis and file layout.

Two sources with one interface:

    NetCDFFileSource  one or more records per NetCDF file; the valid time of a record comes from the file name
                      (time_format as in ft_utilities.get_basetime_from_filename) or from a time variable of the file
    ZarrSource        a Zarr store with a time coordinate

    source.times            np.array [Epoch seconds], sorted
    source.read(i)          2-D np.array float32 (NaN for missing) of record i
    source.grid_shape       (ny, nx) of the 2-D field

The files are only opened where they are read, so a source can be built in the driver and re-created cheaply in each worker.
"""
import os
import re
import numpy as np
import xarray as xr

from pyflextrkr.ft_utilities import subset_files_timerange


def _epoch_seconds(values):
    """Time values (numpy datetime64, cftime, or numbers that are already Epoch seconds) as float Epoch seconds."""
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.datetime64):
        return values.astype("datetime64[ns]").astype(np.int64) / 1e9
    if values.dtype == object:                              # cftime
        import cftime
        return np.array([(cftime.date2num(v, "seconds since 1970-01-01 00:00:00", calendar="standard")) for v in values],
                        dtype=float)
    return values.astype(float)


class NetCDFFileSource:
    def __init__(self, path, filebase, time_format="yyyymodd_hhmmss", varname="precipitation", time_varname=None,
                 start_basetime=None, end_basetime=None, scale=1.0, offset=0.0, time_dimname="time"):
        """
        Args:
            path, filebase: directory and the leading characters of the file names (files are filebase*).
            time_format: how the date and time are written in the file name, used when time_varname is None.
            varname: variable to read (2-D, or 3-D with a size-1 or indexable time dimension).
            time_varname: variable with the valid time of the records inside a file (None: from the file name,
                one record per file).
            start_basetime, end_basetime: only files inside this range [Epoch seconds] are listed (with
                time_varname the files are filtered by the time of the file name when it can be parsed, so give margins).
            scale, offset: value = scale * data + offset.
        """
        self.varname = varname
        self.time_varname = time_varname
        self.time_dimname = time_dimname
        self.scale = scale
        self.offset = offset
        path = path if path.endswith("/") else path + "/"
        if time_varname is None:
            files, basetimes, _, _ = subset_files_timerange(
                path, filebase, -np.inf if start_basetime is None else start_basetime,
                np.inf if end_basetime is None else end_basetime, time_format=time_format)
            order = np.argsort(basetimes, kind="stable")
            self._files = [files[i] for i in order]
            self.times = np.asarray(basetimes, dtype=float)[order]
            self._index = np.zeros(len(self._files), dtype=int)       # record inside the file
            self._has_time_dim = None
        else:
            names = sorted(f for f in os.listdir(path) if f.startswith(filebase))
            files, times, index = [], [], []
            for name in names:
                with xr.open_dataset(path + name, decode_times=True) as ds:
                    t = _epoch_seconds(np.atleast_1d(ds[time_varname].values))
                for j, tj in enumerate(t):
                    if (start_basetime is None or tj >= start_basetime) and (end_basetime is None or tj <= end_basetime):
                        files.append(path + name)
                        times.append(tj)
                        index.append(j)
            order = np.argsort(times, kind="stable")
            self._files = [files[i] for i in order]
            self.times = np.asarray(times, dtype=float)[order]
            self._index = np.asarray(index, dtype=int)[order]
            self._has_time_dim = True
        self.nrecords = len(self._files)
        self._grid_shape = None

    @property
    def grid_shape(self):
        if self._grid_shape is None:
            self._grid_shape = self.read(0).shape
        return self._grid_shape

    def filename(self, i):
        return self._files[i]

    def read(self, i):
        with xr.open_dataset(self._files[i], decode_times=False) as ds:
            da = ds[self.varname]
            if self.time_dimname in da.dims:
                da = da.isel({self.time_dimname: int(self._index[i]) if self.time_varname is not None else 0})
            values = np.asarray(da.values, dtype=np.float32).squeeze()
        if self.scale != 1.0 or self.offset != 0.0:
            values = values * np.float32(self.scale) + np.float32(self.offset)
        return values


class ZarrSource:
    def __init__(self, path, varname="precipitation", time_varname="time", start_basetime=None, end_basetime=None,
                 scale=1.0, offset=0.0):
        self.path = path
        self.varname = varname
        self.time_varname = time_varname
        self.scale = scale
        self.offset = offset
        ds = xr.open_zarr(path, chunks=None)
        t = _epoch_seconds(ds[time_varname].values)
        keep = np.ones(t.size, dtype=bool)
        if start_basetime is not None:
            keep &= t >= start_basetime
        if end_basetime is not None:
            keep &= t <= end_basetime
        order = np.flatnonzero(keep)
        order = order[np.argsort(t[order], kind="stable")]
        self._index = order
        self.times = t[order]
        self.nrecords = order.size
        self._grid_shape = ds[varname].shape[-2:]
        ds.close()

    @property
    def grid_shape(self):
        return tuple(self._grid_shape)

    def filename(self, i):
        return f"{self.path}[{int(self._index[i])}]"

    def read(self, i):
        ds = _open_zarr_cached(self.path)
        values = np.asarray(ds[self.varname].isel({self.time_varname: int(self._index[i])}).values, dtype=np.float32)
        if self.scale != 1.0 or self.offset != 0.0:
            values = values * np.float32(self.scale) + np.float32(self.offset)
        return values.squeeze()


_ZARR_CACHE = {}


def _open_zarr_cached(path):
    if path not in _ZARR_CACHE:
        _ZARR_CACHE[path] = xr.open_zarr(path, chunks=None)
    return _ZARR_CACHE[path]


def open_data_source(config, start_basetime=None, end_basetime=None):
    """
    Build the data source of the trackpcp steps from the config keys
    trackpcp_data_path, _data_filebase, _data_time_format, _data_varname, _data_format (null = auto: a path that ends in
    .zarr is a Zarr store), _data_time_varname, _data_scale, _data_offset.
    """
    path = config["trackpcp_data_path"]
    fmt = config.get("trackpcp_data_format", None)
    if fmt is None:
        fmt = "zarr" if path.rstrip("/").endswith(".zarr") else "netcdf"
    varname = config["trackpcp_data_varname"]
    scale = float(config.get("trackpcp_data_scale", 1.0))
    offset = float(config.get("trackpcp_data_offset", 0.0))
    if fmt == "zarr":
        return ZarrSource(path, varname=varname, time_varname=config.get("trackpcp_data_time_varname", None) or "time",
                          start_basetime=start_basetime, end_basetime=end_basetime, scale=scale, offset=offset)
    if fmt == "netcdf":
        return NetCDFFileSource(
            path, config.get("trackpcp_data_filebase", ""), time_format=config.get("trackpcp_data_time_format", "yyyymodd_hhmmss"),
            varname=varname, time_varname=config.get("trackpcp_data_time_varname", None),
            start_basetime=start_basetime, end_basetime=end_basetime, scale=scale, offset=offset)
    raise ValueError(f"trackpcp_data_format must be netcdf, zarr or null, not '{fmt}'")


def parse_class_names(long_name):
    """
    Class values and names from the long_name of a class variable, e.g.
    "Most-organized feature class observed on this track: 1=MCS, 2=multi-cell, 3=isolated-cell, 4=non-convective"
    -> {1: "MCS", 2: "multi-cell", 3: "isolated-cell", 4: "non-convective"}. {} if there is no "value=name" list.
    """
    if not long_name:
        return {}
    text = str(long_name).split(":")[-1]
    return {int(v): n.strip() for v, n in re.findall(r"(-?\d+)\s*=\s*([^,;]+)", text)}


def class_info(values, long_name=None, requested=None):
    """
    Class values to process and their names.

    Args:
        values: array-like, the values of the class variable (fill values and NaN are ignored by the caller).
        long_name: long_name of the class variable.
        requested: list of class values or None (= all values found).

    Returns:
        class_values: list of int, class_names: list of str (from the long_name; "class_<value>" if it has none)
    """
    found = sorted({int(v) for v in np.unique(values)})
    names = parse_class_names(long_name)
    class_values = [int(v) for v in requested] if requested else found
    return class_values, [names.get(v, f"class_{v}") for v in class_values]
