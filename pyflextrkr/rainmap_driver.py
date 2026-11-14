"""
Monthly maps of the precipitation of Tb+PF MCS tracking (the optional pipeline step rainmap, run_rainmap: True).

For every calendar month of the tracking period, from the pixel-level files of the MCS (mcstrack_<yyyymodd>_<hhmmss>.nc): the total
precipitation, the precipitation inside the cold cloud shield of the MCS, and how many hours the MCS cloud shield and the MCS precipitation
feature (PF) are present at every pixel. It is the calculation of Analysis/calc_tbpf_mcs_monthly_rainmap.py (the same expressions, variables and
file names, so the scripts that read those files keep working), as a step of the run: the stitched stats job of a campaign runs it after speed,
and with a Dask client the computation of a month is done by the workers.

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


def compute_monthly_rainmap(config, year, month):
    """
    The maps of one month, written to rainmap_file(config, year, month).

    The computation is lazy (xarray and dask): the graph is built from the files of the month and run while the file is written, by the
    Dask client of the run if there is one, otherwise by the threaded scheduler.

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
    # The files of the month, opened lazily (one chunk per file)
    #########################################################################
    ds = xr.open_mfdataset(mcsfiles, concat_dim="time", combine="nested")
    ntimes = ds.sizes["time"]
    # The grid is the same in every file: the coordinates of the first one
    longitude = ds["longitude"].isel(time=0).compute()
    latitude = ds["latitude"].isel(time=0).compute()

    #########################################################################
    # The maps (the graph; nothing is calculated before the file is written)
    #########################################################################
    mcscloudct = (ds["cloudtracknumber"] > 0).sum(dim="time")
    mcsprecip = ds[PCP_VARNAME].where(ds["cloudtracknumber"] > 0).sum(dim="time")
    totprecip = ds[PCP_VARNAME].sum(dim="time")
    mcspcpmask = xr.where(ds["pcptracknumber"] > 0, 1, 0)
    mcspcpct = mcspcpmask.sum(dim="time")
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
    coord_dict = {"time": (["time"], months), "lat": (["lat"], ds["lat"].data), "lon": (["lon"], ds["lon"].data)}
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
    # Write (the calculation happens here)
    #########################################################################
    comp = dict(zlib=True, _FillValue=np.nan, dtype="float32")
    encoding = {var: comp for var in dsout.data_vars}
    if os.path.isfile(outfile):
        os.remove(outfile)
    dsout.to_netcdf(path=outfile, mode="w", format="NETCDF4", unlimited_dims="time", encoding=encoding)
    ds.close()
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
