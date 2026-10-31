"""
Rows-mode movement speed (config stats_format: rows).

The same movement of the tracked features from pairs of pixel files as movement_speed.py (the FFT of every file pair is the
function of the dense step, movement_of_feature_fft), with the track statistics as row tables: the movement of a track at a
time is a position in a flat array (offsets[track] + time index), the filter and interpolation of the speed run on the slice of
a track, and the result is the robust tables plus the five movement variables, written as row tables. No [tracks, times] array
is built (5 x 3 million tracks x 800 times x 4 bytes would be 48 GB for a year).

tests/test_movement_speed_rows.py checks that the tables equal what dense_to_rows makes of the dense file.

Author: Zhe Feng | zhe.feng@pnnl.gov
"""
import logging
import os
import shutil
import sys
import time

import dask
import numpy as np

from pyflextrkr.ft_utilities import subset_files_timerange
from pyflextrkr.movement_speed import filter_track_movement, movement_attrs, movement_entries, movement_of_feature_fft
from pyflextrkr.rowstats import TableBuilder, read_tables, write_rows

# The movement variables, in the order of the dense step
MOVEMENT_NAMES = tuple(movement_attrs(0))


def movement_speed_rows(
        config,
        trackstats_filebase=None,
        trackstats_outfilebase=None,
        pixelpath_basename=None,
        pixeltracking_filebase=None,
):
    """
    Calculate movement speed using pixel level tracked feature: movement_speed.movement_speed with row tables as input and
    output.

    Args:
        config: dictionary
            Config parameters of movement_speed.
        trackstats_filebase: string (default: None)
            Input track statistics file basename (mcsrobust_filebase); the input is the folder f"{stats_outpath}{base}{dates}.rows".
        trackstats_outfilebase: string (default: None)
            Output track statistics file basename (mcsfinal_filebase).
        pixelpath_basename: string (default: None)
            Pixel-level files base path name (see movement_speed).
        pixeltracking_filebase: string (default: None)
            Pixel-level file basename (see movement_speed).

    Returns:
        statistics_outfolder: string
            Folder of the row tables, f"{stats_outpath}{mcsfinal_filebase}{startdate}_{enddate}.rows": the robust tables and the
            movement variables.
    """
    stats_outpath = config["stats_outpath"]
    startdate, enddate = config["startdate"], config["enddate"]
    start_basetime, end_basetime = config["start_basetime"], config["end_basetime"]
    run_parallel = config["run_parallel"]
    lag = config["lag_for_speed"]
    max_speed_thresh = config["max_speed_thresh"]
    logger = logging.getLogger(__name__)
    logger.info('Calculating movement speed using pixel-level tracked feature (row tables)')

    if trackstats_filebase is None:
        trackstats_filebase = config["mcsrobust_filebase"]
    if trackstats_outfilebase is None:
        trackstats_outfilebase = config["mcsfinal_filebase"]
    if pixelpath_basename is None:
        pixeltracking_outpath = config["pixeltracking_outpath"]
    else:
        pixeltracking_outpath = f'{config["root_path"]}/{pixelpath_basename}/{startdate}_{enddate}/'
    if pixeltracking_filebase is None:
        pixeltracking_filebase = config["pixeltracking_filebase"]
    statistics_folder = f"{stats_outpath}{trackstats_filebase}{startdate}_{enddate}.rows"
    statistics_outfolder = f"{stats_outpath}{trackstats_outfilebase}{startdate}_{enddate}.rows"
    if statistics_folder == statistics_outfolder:
        logger.critical(f"The input and output statistics folders are the same ({statistics_folder}). Tracking will now exit.")
        sys.exit()

    # Pixel files to process
    filelist, _, _, _ = subset_files_timerange(pixeltracking_outpath, pixeltracking_filebase, start_basetime, end_basetime)
    nfiles = len(filelist)
    logger.info(f"Total number of files to process: {nfiles}")

    # Track statistics (the first time of each track aligns the movement values to the track times)
    tables = read_tables(statistics_folder)
    offsets, duration = tables.offsets, tables.duration
    ntracks, nentries = len(duration), int(offsets[-1])
    ntimes = tables.meta["dims"][tables.meta["times_dim"]]
    stats_start_basetime = np.full(ntracks, np.nan)
    has_entries = duration > 0
    stats_start_basetime[has_entries] = tables.entries["base_time"][offsets[:-1][has_entries]]

    # Movement of every file pair (the function of the dense step): the files `lag` steps apart, nfiles - lag pairs
    filepairs = list(zip(filelist[0:-lag], filelist[lag::]))
    results = []
    if run_parallel == 0:
        for ifile in range(len(filepairs)):
            results.append(movement_of_feature_fft(filepairs[ifile], ntracks, config, sparse=True))
        final_result = results
    elif run_parallel >= 1:
        from dask.distributed import wait       # only the parallel mode needs the `distributed` package (the CI environment has none)
        for ifile in range(len(filepairs)):
            results.append(dask.delayed(movement_of_feature_fft)(filepairs[ifile], ntracks, config, sparse=True))
        final_result = dask.compute(*results)
        wait(final_result)
    else:
        sys.exit('Valid parallelization flag not provided')

    # Movement of the measured tracks of every file pair -> flat arrays aligned to the track times
    movement = {name: np.full(nentries, np.nan, dtype=np.float32) for name in MOVEMENT_NAMES}
    entries = movement_entries(final_result, stats_start_basetime, ntimes, lag)
    if entries is not None:
        track, k = entries["track"], entries["k"]
        inside = k < duration[track]            # a value behind the last time of its track has no place in the tables
        if not inside.all():
            logger.warning(f"{int((~inside).sum())} movement value(s) are behind the last time of their track and were dropped.")
        position = offsets[track[inside]] + k[inside]
        for name in MOVEMENT_NAMES:
            movement[name][position] = entries[name][inside]

    # Filter and interpolate high movement speeds, track by track (only the tracks that have any movement value)
    times = np.arange(int(duration.max()) if ntracks else 0)
    measured = np.unique(tables.entries["track"][np.isfinite(movement["movement_speed"])])
    for itrack in measured:
        sl = slice(offsets[itrack], offsets[itrack + 1])
        if not filter_track_movement(*(movement[name][sl] for name in ("movement_distance", "movement_speed", "movement_theta",
                                                                         "movement_distance_x", "movement_distance_y")),
                                     times[:sl.stop - sl.start], max_speed_thresh):
            logger.debug('Not enough values in Track %d' % itrack)

    # The robust tables plus the movement variables
    builder = TableBuilder.from_tables(tables)
    attrs = movement_attrs(lag)
    for name in MOVEMENT_NAMES:
        builder.add_entry_var(name, movement[name], attrs[name], fill=np.nan)
    builder.global_attrs["Created_on"] = time.ctime(time.time())
    builder.global_attrs["max_speed_thresh"] = max_speed_thresh
    logger.info(f"Writing {statistics_outfolder}")
    if os.path.isdir(statistics_outfolder):
        shutil.rmtree(statistics_outfolder)
    write_rows(builder.build(), statistics_outfolder)
    logger.info(f"{statistics_outfolder}")
    return statistics_outfolder
