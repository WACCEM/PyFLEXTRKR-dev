import os
import sys
import logging
import numpy as np
import xarray as xr
import dask
from pyflextrkr.ft_utilities import subset_files_timerange, match_pairs_to_files
from pyflextrkr.mapfeature_func import map_feature

def mapfeature_driver(
        config,
        trackstats_filebase="trackstats_",
        outpath_basename=None,
        outfile_basename=None,
):
    """
    Map tracked features to pixel-level files.

    Args:
        config: dictionary
            Dictionary containing config parameters.
        trackstats_filebase: string, default="trackstats_"
            Track statistics file basename.
        outpath_basename: string, default=None
            Output path basename for pixel-level files.
            If None, pixeltracking_outpath defaults to config["pixeltracking_outpath"].
            Otherwise, pixeltracking_outpath is constructed using the config file:
                f'{config["root_path"]}/{outpath_basename}/{config["startdate"]}_{config["enddate"]}/'
        outfile_basename: string, default=None
            Output pixel-level file basename.
            If None, defaults to use config["pixeltracking_filebase"].

    Returns:
        None.
    """
    logger = logging.getLogger(__name__)
    logger.info('Mapping tracked features to pixel-level files')

    stats_path = config["stats_outpath"]
    if outpath_basename is None:
        pixeltracking_outpath = config["pixeltracking_outpath"]
    else:
        pixeltracking_outpath = f'{config["root_path"]}/{outpath_basename}/{config["startdate"]}_{config["enddate"]}/'
    if outfile_basename is None:
        pixeltracking_filebase = config["pixeltracking_filebase"]
    else:
        pixeltracking_filebase = outfile_basename
    tracking_outpath = config["tracking_outpath"]
    cloudid_filebase = config["cloudid_filebase"]
    startdate = config["startdate"]
    enddate = config["enddate"]
    start_basetime = config["start_basetime"]
    end_basetime = config["end_basetime"]
    # Minimum time difference threshold [second] to match track stats and cloudid pixel files
    match_pixel_dt_thresh = config["match_pixel_dt_thresh"]
    run_parallel = config["run_parallel"]
    # feature_type = config["feature_type"]
    nmaxlinks = config["nmaxlinks"]
    tracks_dimname = config.get("tracks_dimname", "tracks")
    times_dimname = config.get("times_dimname", "times")
    fillval = config.get("fillval", -9999)

    #########################################################################################
    # Read track stats
    trackstats_file = f"{stats_path}{trackstats_filebase}{startdate}_{enddate}.nc"
    # Opened lazily: only the variables used below are read (the statistics file of a month is 180 GB uncompressed)
    ds = xr.open_dataset(
        trackstats_file,
        mask_and_scale=False,
        decode_times=False,
    )
    # Get track stats variable names
    stats_varnames = list(ds.data_vars)
    # Get track stats dimensions
    ntracks = ds.sizes[tracks_dimname]
    ntimes = ds.sizes[times_dimname]
    # Get track variables
    stats_basetime = ds["base_time"].values
    stats_cloudnumber = ds["cloudnumber"].values
    stats_trackstatus = ds["track_status"].values
    trackstats_comments = ds["track_status"].comments

    # Optional per-cloud feature classification (e.g. identifymcs_radar.py's
    # feature_class: 1=MCS, 2=multi-cell, 3=isolated-cell, 4=non-convective),
    # painted into pixel files as its own map when present.
    mapfeature_class_varname = config.get("mapfeature_class_varname", None)
    if mapfeature_class_varname is not None and mapfeature_class_varname in stats_varnames:
        stats_featureclass = ds[mapfeature_class_varname].values
    else:
        stats_featureclass = None

    # Put merge/split tracknumbers & cloudnumbers in a list
    ms_tracknumber = ["merge_tracknumbers", "split_tracknumbers"]
    ms_cloudnumber = ["merge_cloudnumber", "split_cloudnumber"]

    # Check if tracknumber are in the stats dataset
    if (set(ms_tracknumber).issubset(set(stats_varnames))):
        stats_mergetracknumber = ds["merge_tracknumbers"].values
        stats_splittracknumber = ds["split_tracknumbers"].values
    else:
        stats_mergetracknumber = np.full((ntracks, ntimes), fillval, dtype=int)
        stats_splittracknumber = np.full((ntracks, ntimes), fillval, dtype=int)

    # Check if cloudnumber are in the stats dataset
    if (set(ms_cloudnumber).issubset(set(stats_varnames))):
        stats_mergecloudnumber = ds["merge_cloudnumber"].values
        stats_splitcloudnumber = ds["split_cloudnumber"].values
    else:
        stats_mergecloudnumber = np.full((ntracks, ntimes, nmaxlinks), fillval, dtype=int)
        stats_splitcloudnumber = np.full((ntracks, ntimes, nmaxlinks), fillval, dtype=int)
    ds.close()

    #########################################################################################
    # Identify files to process
    # Create pixel tracking file output directory
    os.makedirs(pixeltracking_outpath, exist_ok=True)
    cloudidfiles, \
    cloudidfiles_basetime, \
    cloudidfiles_datestring, \
    cloudidfiles_timestring = subset_files_timerange(tracking_outpath,
                                                     cloudid_filebase,
                                                     start_basetime,
                                                     end_basetime)
    nfiles = len(cloudidfiles)
    logger.info(f"Total number of files to process: {nfiles}")

    # Matching time indices from the stats file to every cloudid file, in one pass over the valid entries
    # (same result as np.where(np.abs(stats_basetime - file_basetime) < match_pixel_dt_thresh) for each file,
    # which took 1.8 s per file for a month of 240,000 tracks x 550 times)
    file_pairs = match_pairs_to_files(stats_basetime, cloudidfiles_basetime, match_pixel_dt_thresh)

    results = []
    # Loop over each pixel file
    for ifile in range(0, nfiles):
        itrack, itime = file_pairs[ifile]

        # Get cloudnumbers for this time (file)
        file_trackindex = itrack
        file_cloudnumber = stats_cloudnumber[itrack, itime]
        file_trackstatus = stats_trackstatus[itrack, itime]

        # Cloudnumbers for merge/split
        file_mergecloudnumber = stats_mergecloudnumber[itrack, itime, :]
        file_splitcloudnumber = stats_splitcloudnumber[itrack, itime, :]
        if (file_mergecloudnumber.size > 0) & (file_splitcloudnumber.size > 0):
            # Get number of max merge/split for all clouds at this time (file)
            max_merge = np.sum(file_mergecloudnumber > 0, axis=1).max()
            max_split = np.sum(file_splitcloudnumber > 0, axis=1).max()
            # Subset arrays containing useful data to reduce array size
            file_mergecloudnumber = file_mergecloudnumber[:, :max_merge]
            file_splitcloudnumber = file_splitcloudnumber[:, :max_split]

        # General merge/split tracknumber
        file_mergetracknumber = stats_mergetracknumber[itrack, itime]
        file_splittracknumber = stats_splittracknumber[itrack, itime]

        # Optional per-cloud feature class for this time
        file_featureclass = stats_featureclass[itrack, itime] if stats_featureclass is not None else None

        # Serial
        if run_parallel == 0:
            result = map_feature(
                cloudidfiles[ifile],
                cloudidfiles_basetime[ifile],
                file_trackindex,
                file_cloudnumber,
                file_trackstatus,
                file_mergetracknumber,
                file_splittracknumber,
                file_mergecloudnumber,
                file_splitcloudnumber,
                trackstats_comments,
                config,
                pixeltracking_outpath,
                pixeltracking_filebase,
                file_featureclass=file_featureclass,
            )
        # Parallel
        elif run_parallel >= 1:
            result = dask.delayed(map_feature)(
                cloudidfiles[ifile],
                cloudidfiles_basetime[ifile],
                file_trackindex,
                file_cloudnumber,
                file_trackstatus,
                file_mergetracknumber,
                file_splittracknumber,
                file_mergecloudnumber,
                file_splitcloudnumber,
                trackstats_comments,
                config,
                pixeltracking_outpath,
                pixeltracking_filebase,
                file_featureclass=file_featureclass,
            )
            results.append(result)
        else:
            sys.exit('Valid parallelization flag not provided.')

    if run_parallel >= 1:
        from dask.distributed import wait       # only the parallel mode needs the `distributed` package (the CI environment has none)
        # Trigger dask computation
        final_result = dask.compute(*results)
        wait(final_result)

    logger.info('Done with mapping features to pixel-level files')
    return