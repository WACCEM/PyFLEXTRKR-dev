"""
Rows-mode mapfeature (config stats_format: rows).

The same mapping of the tracked features to pixel-level files as mapfeature_driver.py, with the track statistics as row tables:
the (track, time) entries of a cloudid file are positions in flat arrays, and the merge / split cloud numbers of the fragments
come from the fragments table. The worker that paints one pixel file (map_feature) is the one of the dense step, called with
the same arguments, so the pixel files are the same.

tests/test_mapfeature_rows.py checks that the pixel files equal those of the dense step.

Author: Zhe Feng | zhe.feng@pnnl.gov
"""
import logging
import os
import sys

import dask
import numpy as np

from pyflextrkr.ft_utilities import match_pairs_to_files, subset_files_timerange
from pyflextrkr.mapfeature_func import map_feature
from pyflextrkr.regionprops_rows import fragment_slots
from pyflextrkr.rowstats import FRAGMENT_KINDS, read_tables


def mapfeature_driver_rows(
        config,
        trackstats_filebase="trackstats_",
        outpath_basename=None,
        outfile_basename=None,
):
    """
    Map tracked features to pixel-level files: mapfeature_driver.mapfeature_driver with row tables as the track statistics.

    Args:
        config: dictionary
            Config parameters of mapfeature_driver.
        trackstats_filebase: string (default: 'trackstats_')
            Track statistics file basename; the input is the folder f"{stats_outpath}{base}{dates}.rows".
        outpath_basename: string (default: None)
            Output path basename for pixel-level files (see mapfeature_driver).
        outfile_basename: string (default: None)
            Output pixel-level file basename (see mapfeature_driver).

    Returns:
        None.
    """
    logger = logging.getLogger(__name__)
    logger.info('Mapping tracked features to pixel-level files (row tables)')

    stats_path = config["stats_outpath"]
    if outpath_basename is None:
        pixeltracking_outpath = config["pixeltracking_outpath"]
    else:
        pixeltracking_outpath = f'{config["root_path"]}/{outpath_basename}/{config["startdate"]}_{config["enddate"]}/'
    pixeltracking_filebase = config["pixeltracking_filebase"] if outfile_basename is None else outfile_basename
    tracking_outpath = config["tracking_outpath"]
    cloudid_filebase = config["cloudid_filebase"]
    startdate, enddate = config["startdate"], config["enddate"]
    start_basetime, end_basetime = config["start_basetime"], config["end_basetime"]
    match_pixel_dt_thresh = config["match_pixel_dt_thresh"]
    run_parallel = config["run_parallel"]
    fillval = config.get("fillval", -9999)

    #########################################################################################
    # Read track stats
    tables = read_tables(f"{stats_path}{trackstats_filebase}{startdate}_{enddate}.rows")
    meta, entries, fragments = tables.meta, tables.entries, tables.fragments
    offsets = tables.offsets
    stats_basetime = entries["base_time"]
    stats_cloudnumber = entries["cloudnumber"]
    stats_trackstatus = entries["track_status"]
    trackstats_comments = meta["attrs"]["track_status"]["comments"]
    track_of = entries["track"]

    # Optional per-cloud feature classification, painted into the pixel files as its own map when present
    mapfeature_class_varname = config.get("mapfeature_class_varname", None)
    stats_featureclass = None
    if mapfeature_class_varname is not None and mapfeature_class_varname in meta["vars_2d"]:
        stats_featureclass = entries[mapfeature_class_varname]

    # Merge / split track numbers (general ones are stored as [tracks, times] variables), else the fill value
    if {"merge_tracknumbers", "split_tracknumbers"}.issubset(meta["vars_2d"]):
        stats_mergetracknumber, stats_splittracknumber = entries["merge_tracknumbers"], entries["split_tracknumbers"]
    else:
        stats_mergetracknumber = np.full(len(track_of), fillval, dtype=int)
        stats_splittracknumber = np.full(len(track_of), fillval, dtype=int)

    # Merge / split cloud numbers of the entries, from the fragments table
    nslots = meta["dims"][meta["slots_dim"]] if meta["vars_3d"] else int(config["nmaxlinks"])
    by_kind = {}
    if {"merge_cloudnumber", "split_cloudnumber"}.issubset(meta["vars_3d"]):
        frag_entry = offsets[fragments["track"]] + fragments["k"]
        slot_fill = meta["fill"]["merge_cloudnumber"]
        for code, kind in enumerate(FRAGMENT_KINDS):
            use = fragments["kind"] == code
            by_kind[kind] = (frag_entry[use], fragments["slot"][use], fragments["cloudnumber"][use])
    else:
        slot_fill = fillval
        for kind in FRAGMENT_KINDS:
            by_kind[kind] = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=int))

    #########################################################################################
    # Identify files to process
    os.makedirs(pixeltracking_outpath, exist_ok=True)
    cloudidfiles, cloudidfiles_basetime, _, _ = subset_files_timerange(
        tracking_outpath, cloudid_filebase, start_basetime, end_basetime)
    nfiles = len(cloudidfiles)
    logger.info(f"Total number of files to process: {nfiles}")

    # The flat entries of each cloudid file (all entries are in one row of a [1, nentries] array: the index in it is the entry)
    file_pairs = match_pairs_to_files(stats_basetime[None, :], cloudidfiles_basetime, match_pixel_dt_thresh)

    results = []
    for ifile in range(0, nfiles):
        entry = file_pairs[ifile][1]
        file_trackindex = track_of[entry].astype(np.int64)        # as the indices of np.nonzero in the dense step
        file_mergecloudnumber = fragment_slots(*by_kind["merge"], entry, nslots, slot_fill)
        file_splitcloudnumber = fragment_slots(*by_kind["split"], entry, nslots, slot_fill)
        if (file_mergecloudnumber.size > 0) & (file_splitcloudnumber.size > 0):
            # Number of the largest merge / split at this time (file): keep the slots that can hold data
            max_merge = np.sum(file_mergecloudnumber > 0, axis=1).max()
            max_split = np.sum(file_splitcloudnumber > 0, axis=1).max()
            file_mergecloudnumber = file_mergecloudnumber[:, :max_merge]
            file_splitcloudnumber = file_splitcloudnumber[:, :max_split]
        args = (
            cloudidfiles[ifile],
            cloudidfiles_basetime[ifile],
            file_trackindex,
            stats_cloudnumber[entry],
            stats_trackstatus[entry],
            stats_mergetracknumber[entry],
            stats_splittracknumber[entry],
            file_mergecloudnumber,
            file_splitcloudnumber,
            trackstats_comments,
            config,
            pixeltracking_outpath,
            pixeltracking_filebase,
        )
        kwargs = {"file_featureclass": stats_featureclass[entry] if stats_featureclass is not None else None}
        if run_parallel == 0:
            map_feature(*args, **kwargs)
        elif run_parallel >= 1:
            results.append(dask.delayed(map_feature)(*args, **kwargs))
        else:
            sys.exit('Valid parallelization flag not provided.')

    if run_parallel >= 1:
        from dask.distributed import wait       # only the parallel mode needs the `distributed` package (the CI environment has none)
        final_result = dask.compute(*results)
        wait(final_result)

    logger.info('Done with mapping features to pixel-level files')
    return
