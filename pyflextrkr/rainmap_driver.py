"""
Monthly maps of the precipitation of Tb+PF MCS tracking (the optional pipeline step rainmap, run_rainmap: True).

For every calendar month of the tracking period, from the pixel-level files of the MCS (mcstrack_<yyyymodd>_<hhmmss>.nc): the total
precipitation, the precipitation inside the cold cloud shield of the MCS, and how many hours the MCS cloud shield and the MCS precipitation
feature (PF) are present at every pixel. It is the calculation of Analysis/calc_tbpf_mcs_monthly_rainmap.py (the same expressions, variables and
file names, so the scripts that read those files keep working), as a step of the run: the stitched stats job of a campaign runs it after speed,
and with a Dask client the computation of a month is done by the workers.

The files are read INSIDE the Dask tasks (one task per file opens the file, reads what the maps need and closes it), and the maps are written
by the main process from the computed arrays. The script of Analysis/ opens the files lazily in the main process (xr.open_mfdataset) and lets the
workers write the result (to_netcdf of Dask arrays): there the workers get the lazily opened datasets, with the netCDF locks of the main process.
xarray's CachingFileManager.__del__ takes that lock with acquire(blocking=False), and CombinedLock.acquire(blocking=False) is not
all-or-nothing: when the event loop of a worker runs __del__ while a write is between its two releases, it keeps one of the two locks for ever, and the
next read of that worker waits for it for ever (seen on 2019 IMERG: about every second month of the step hung, measured 2026-11-14). Nothing of
xarray's file handling crosses to a worker here, and the files are read by all the workers (the lazily opened datasets kept the reading on a few).

    <stats_outpath>monthly/mcs_rainmap_<YYYYMM>.nc      precipitation, mcs_precipitation [mm], mcs_precipitation_count,
                                                        mcs_cloud_count [hours], ntimes, latitude, longitude

Author: Zhe Feng | zhe.feng@pnnl.gov
"""
import calendar
import datetime
import glob
import logging
import os
import time

import dask
import dask.array as da
import numpy as np
import xarray as xr

from pyflextrkr.trackpcp_driver import months_of_period

PCP_VARNAME = "precipitation"


def rainmap_dir(config):
    """
    Folder of the monthly files: <stats_outpath>monthly/ (rainmap_outdir of the config replaces it).

    Args:
        config: dictionary
            Loaded config (stats_outpath, optional rainmap_outdir).

    Returns:
        folder: string
            With the trailing slash.
    """
    folder = config.get("rainmap_outdir") or f"{config['stats_outpath']}monthly/"
    return folder if folder.endswith("/") else folder + "/"


def rainmap_file(config, year, month):
    """
    The file of one month: <rainmap_dir>mcs_rainmap_<YYYYMM>.nc.

    Args:
        config: dictionary
            Loaded config.
        year, month: int
            Calendar month.

    Returns:
        path: string
    """
    return f"{rainmap_dir(config)}mcs_rainmap_{int(year)}{int(month):02d}.nc"


def pixel_files_of_month(config, year, month):
    """
    The pixel-level files of one calendar month (the month of their names), in time order.

    Args:
        config: dictionary
            Loaded config (pixeltracking_outpath, pixeltracking_filebase).
        year, month: int
            Calendar month.

    Returns:
        files: list of strings
    """
    folder = config["pixeltracking_outpath"]
    base = config.get("pixeltracking_filebase", "mcstrack_")
    return sorted(glob.glob(f"{folder}/{base}{int(year)}{int(month):02d}*_*.nc"))


def expected_outputs(config):
    """
    The files that the step makes: one for every calendar month of the period that has pixel-level files (what the output check asks for).

    Args:
        config: dictionary
            Loaded config.

    Returns:
        files: list of strings
            In time order; empty if the period has no pixel-level file.
    """
    return [rainmap_file(config, y, m) for y, m in months_of_period(config) if pixel_files_of_month(config, y, m)]


def read_pixel_file(path, pcp_varname=PCP_VARNAME, lat=None, lon=None):
    """
    The fields of one pixel-level file that the maps need, read and decoded where the function runs (a Dask task): the file is opened and
    closed here, so no xarray file handle or lock is shared with another process.

    Args:
        path: string
            Pixel-level file (one time step).
        pcp_varname: string (default: PCP_VARNAME)
            Name of the precipitation variable.
        lat, lon: 1D arrays or None (default: None)
            The grid of the first file of the month; if given, this file must have the same grid.

    Returns:
        cloud: bool array [1, lat, lon]
            Pixels inside a tracked cold cloud shield (cloudtracknumber > 0).
        pcp: bool array [1, lat, lon]
            Pixels inside a tracked precipitation feature (pcptracknumber > 0).
        precip: array [1, lat, lon]
            Precipitation as decoded by xarray (NaN where there is no data).
    """
    with xr.open_dataset(path) as ds:
        # The maps are sums over files: every file must hold one time and the grid of the others
        if ds.sizes["time"] != 1:
            raise ValueError(f"{path} holds {ds.sizes['time']} times, the rain maps need pixel-level files of one time step")
        if lat is not None and (not np.array_equal(ds["lat"].values, lat) or not np.array_equal(ds["lon"].values, lon)):
            raise ValueError(f"{path} has another grid than the first file of the month")
        # The decoded fields (track numbers are masked where 0: NaN > 0 is False)
        cloud = ds["cloudtracknumber"].values > 0
        pcp = ds["pcptracknumber"].values > 0
        precip = ds[pcp_varname].values
    return cloud, pcp, precip


def lazy_fields_of_month(mcsfiles, pcp_varname=PCP_VARNAME):
    """
    The fields of the files of a month as Dask arrays [time, lat, lon] with one chunk per file; every chunk is read by its own task
    (read_pixel_file), nothing is opened here except the first file for the grid.

    Args:
        mcsfiles: list of strings
            The pixel-level files in time order (at least one).
        pcp_varname: string (default: PCP_VARNAME)
            Name of the precipitation variable.

    Returns:
        fields: dictionary
            cloud, pcp, precip: DataArrays [time, lat, lon] (lazy); lat, lon: 1D arrays; longitude, latitude: DataArrays [lat, lon] of
            the first file (with their attributes).
    """
    # The grid, the 2D coordinates and the dtype of the precipitation from the first file
    with xr.open_dataset(mcsfiles[0]) as d0:
        lat, lon = d0["lat"].values, d0["lon"].values
        ny, nx = lat.size, lon.size
        pcp_dtype = d0[pcp_varname].dtype
        longitude, latitude = (d0[name].isel(time=0) if "time" in d0[name].dims else d0[name] for name in ("longitude", "latitude"))
        longitude, latitude = longitude.load(), latitude.load()
    # One task per file (three fields each), one chunk per file and field
    read = dask.delayed(read_pixel_file, nout=3, pure=False)
    parts = [read(f, pcp_varname, lat, lon) for f in mcsfiles]
    dims = ("time", "lat", "lon")
    fields = {}
    for i, (name, dtype) in enumerate((("cloud", bool), ("pcp", bool), ("precip", pcp_dtype))):
        chunks = [da.from_delayed(p[i], shape=(1, ny, nx), dtype=dtype) for p in parts]
        fields[name] = xr.DataArray(da.concatenate(chunks, axis=0), dims=dims)
    fields.update(lat=lat, lon=lon, longitude=longitude, latitude=latitude)
    return fields


def compute_monthly_rainmap(config, year, month):
    """
    The maps of one month, written to rainmap_file(config, year, month).

    The maps are a lazy computation (xarray and dask) over the files of the month, run by the Dask client of the run if there is one, otherwise
    by the threaded scheduler; every file is read by a task of its own (read_pixel_file). The computed maps are written by this process.

    Args:
        config: dictionary
            Loaded config.
        year, month: int
            Calendar month.

    Returns:
        outfile: string or None
            The file that was written; None (with a warning) if the month has no pixel-level file.
    """
    logger = logging.getLogger(__name__)
    outfile = rainmap_file(config, year, month)
    mcsfiles = pixel_files_of_month(config, year, month)
    if len(mcsfiles) == 0:
        logger.warning(f"No pixel-level files of {year}-{int(month):02d} in {config['pixeltracking_outpath']}: no rain map")
        return None
    os.makedirs(rainmap_dir(config), exist_ok=True)
    t0 = time.time()

    #########################################################################
    # The files of the month, one chunk per file (read by the tasks)
    #########################################################################
    fields = lazy_fields_of_month(mcsfiles, PCP_VARNAME)
    cloud, pcp, precip = fields["cloud"], fields["pcp"], fields["precip"]
    ntimes = cloud.sizes["time"]
    longitude, latitude = fields["longitude"], fields["latitude"]

    #########################################################################
    # The maps (the graph; nothing is calculated before the file is written)
    #########################################################################
    mcscloudct = cloud.sum(dim="time")
    mcsprecip = precip.where(cloud).sum(dim="time")
    totprecip = precip.sum(dim="time")
    mcspcpct = pcp.sum(dim="time")
    # The time of the month: the first second of it, in Epoch time
    months = np.zeros(1, dtype=int)
    months[0] = calendar.timegm(datetime.datetime(int(year), int(month), 1, 0, 0, 0).timetuple())

    #########################################################################
    # The output dataset
    #########################################################################
    var_dict = {
        "longitude": (["lat", "lon"], longitude.data, longitude.attrs),
        "latitude": (["lat", "lon"], latitude.data, latitude.attrs),
        "precipitation": (["time", "lat", "lon"], totprecip.expand_dims("time", axis=0).data),
        "mcs_precipitation": (["time", "lat", "lon"], mcsprecip.expand_dims("time", axis=0).data),
        "mcs_precipitation_count": (["time", "lat", "lon"], mcspcpct.expand_dims("time", axis=0).data),
        "mcs_cloud_count": (["time", "lat", "lon"], mcscloudct.expand_dims("time", axis=0).data),
        "ntimes": (["time"], xr.DataArray(ntimes).expand_dims("time", axis=0).data),
    }
    coord_dict = {"time": (["time"], months), "lat": (["lat"], fields["lat"]), "lon": (["lon"], fields["lon"])}
    gattr_dict = {"title": "MCS precipitation accumulation", "contact": "Zhe Feng, zhe.feng@pnnl.gov", "created_on": time.ctime(time.time())}
    dsout = xr.Dataset(var_dict, coords=coord_dict, attrs=gattr_dict)
    attrs = {
        "time": {"long_name": "Epoch Time (since 1970-01-01T00:00:00)", "units": "Seconds since 1970-1-1 0:00:00 0:00"},
        "lon": {"long_name": "Longitude", "units": "degree"},
        "lat": {"long_name": "Latitude", "units": "degree"},
        "ntimes": {"long_name": "Number of times in the month", "units": "count"},
        "precipitation": {"long_name": "Total precipitation", "units": "mm"},
        "mcs_precipitation": {"long_name": "MCS precipitation", "units": "mm"},
        "mcs_precipitation_count": {"long_name": "Number of hours MCS precipitation is recorded", "units": "hour"},
        "mcs_cloud_count": {"long_name": "Number of hours MCS cloud is recorded", "units": "hour"},
    }
    for name, values in attrs.items():
        dsout[name].attrs.update(values)

    #########################################################################
    # Calculate (the files are read here, by the tasks), then write from this process
    #########################################################################
    dsout = dsout.compute()
    comp = dict(zlib=True, _FillValue=np.nan, dtype="float32")
    encoding = {var: comp for var in dsout.data_vars}
    if os.path.isfile(outfile):
        os.remove(outfile)
    dsout.to_netcdf(path=outfile, mode="w", format="NETCDF4", unlimited_dims="time", encoding=encoding)
    logger.info(f"{outfile}: {ntimes} files, {time.time() - t0:.0f} s")
    return outfile


def rainmap_driver(config):
    """
    The pipeline step rainmap (run_rainmap: True): the maps of every calendar month of the period that has pixel-level files.

    Args:
        config: dictionary
            Loaded config.

    Returns:
        files: list of strings
            The files that were written (expected_outputs).
    """
    logger = logging.getLogger(__name__)
    months = months_of_period(config)
    logger.info(f"Monthly maps of the MCS precipitation for {len(months)} month(s)")
    # Loop over the months of the period: one file each, calculated one after the other (the workers of the run share the work of a month)
    for year, month in months:
        compute_monthly_rainmap(config, year, month)
    return expected_outputs(config)
