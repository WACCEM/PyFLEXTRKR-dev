"""
The dense netCDF file of the final MCS statistics (the optional pipeline step dense_netcdf, run_dense_netcdf).

With stats_format: rows the steps of the Tb+PF pipeline write the statistics as row tables (a Parquet folder, pyflextrkr/rowstats.py) and no
netCDF file. Most users, and the tools written for the old pipeline, expect the dense netCDF file of the final statistics,
mcs_tracks_final_<startdate>_<enddate>.nc ([tracks, times] arrays). This step writes it from the row folder of the final statistics, after the
speed step (the last one that changes the file), under the name that the old dense step gave it, next to the folder:

    <stats_outpath>mcs_tracks_final_<startdate>_<enddate>.rows/     the row tables (the source)
    <stats_outpath>mcs_tracks_final_<startdate>_<enddate>.nc        the dense file (the output of this step)

The conversion is pyflextrkr.rowstats.write_dense_netcdf (the layout, names, dtypes, fill values, attributes, unlimited tracks dimension and zlib
compression of the old file; the result does not depend on the block size). It writes block_tracks tracks at a time, so the memory is that of one
block and not of the dense arrays. The file is written under a temporary name and renamed when it is complete, so that a run that is killed
never leaves a file that looks finished.

Config keys:
    run_dense_netcdf: bool
        Run the step. Not in the config: on for stats_format: rows when the run also does the speed step (the run that writes the final
        statistics), off otherwise (track jobs, post-processing jobs). With stats_format: dense the file of the old step is the dense file
        already and the step does nothing.
    dense_netcdf_block_tracks: int (default: 4000)
        Tracks built and written at a time.
    dense_netcdf_variables: list of strings (default: null = all)
        The [tracks, times] variables to write (the 1-D variables and the merge / split variables are always written).

Author: Zhe Feng | zhe.feng@pnnl.gov
"""
import logging
import os
import time

from pyflextrkr.rowstats import dense_netcdf_name, write_dense_netcdf

# Tracks written at a time when the config has no dense_netcdf_block_tracks (about 2 GB for the Tb+PF final statistics)
DEFAULT_BLOCK_TRACKS = 4000


def default_run_flag(config):
    """
    The value of run_dense_netcdf for a config that does not have the key: the dense file is made by the run that writes the final
    statistics (row tables and the speed step), not by a track job or a post-processing job.

    Args:
        config: dictionary
            Loaded config (stats_format, run_speed).

    Returns:
        flag: bool
            True for stats_format: rows with run_speed on.
    """
    return config.get("stats_format", "dense") == "rows" and bool(config.get("run_speed", False))


def source_folder(config):
    """
    The row folder of the final statistics: <stats_outpath><mcsfinal_filebase><startdate>_<enddate>.rows.

    Args:
        config: dictionary
            Loaded config (stats_outpath, mcsfinal_filebase, startdate, enddate).

    Returns:
        folder: string
    """
    return f"{config['stats_outpath']}{config['mcsfinal_filebase']}{config['startdate']}_{config['enddate']}.rows"


def expected_output(config):
    """
    The dense file that the step makes: the name of the row folder with .rows replaced by .nc, next to it (the name of the old dense step).

    Args:
        config: dictionary
            Loaded config.

    Returns:
        path: string
    """
    return dense_netcdf_name(source_folder(config))


def dense_netcdf_driver(config):
    """
    Write the dense netCDF file of the final statistics from its row folder (write_dense_netcdf), under a temporary name that is renamed when
    the file is complete. An existing file is replaced.

    Args:
        config: dictionary
            Loaded config (stats_outpath, mcsfinal_filebase, startdate, enddate, optional dense_netcdf_block_tracks and dense_netcdf_variables).

    Returns:
        outfile: string
            The dense file.

    Raises:
        FileNotFoundError: the row folder of the final statistics does not exist.
    """
    logger = logging.getLogger(__name__)
    # The source (the row folder after the speed step), the target and the temporary name next to it
    folder = source_folder(config)
    outfile = expected_output(config)
    tmpfile = outfile + ".tmp"
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"{folder} does not exist: the dense file is made from the row tables of the final statistics")
    block_tracks = config.get("dense_netcdf_block_tracks") or DEFAULT_BLOCK_TRACKS
    variables = config.get("dense_netcdf_variables")
    logger.info(f"Dense netCDF: {os.path.basename(folder)} -> {os.path.basename(outfile)} "
                f"(blocks of {block_tracks} tracks, variables: {'all' if variables is None else len(variables)})")

    # The conversion, block by block, into the temporary file (a stale temporary file of a killed run is replaced)
    t_start = time.time()
    write_dense_netcdf(folder, tmpfile, variables=variables, block_tracks=block_tracks, overwrite=True)

    # The file is complete: it gets its name (atomic on the same file system), replacing the file of an earlier run
    os.replace(tmpfile, outfile)
    logger.info(f"Dense netCDF written: {outfile} ({os.path.getsize(outfile) / 1.0e9:.2f} GB, {time.time() - t_start:.1f} s)")
    return outfile
