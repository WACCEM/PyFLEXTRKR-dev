import os
import sys
import time
import logging
import numpy as np
import xarray as xr
import dask

from pyflextrkr.ft_utilities import subset_files_timerange, match_pairs_to_files
from pyflextrkr.smooth_trajectory import smooth_trajectory
from pyflextrkr.mcs_lifecycle import classify_lifecycle
from pyflextrkr.identifymcs_radar import identify_mcs_periods_by_method, mcs_size_varname, mcs_params_from_config
from pyflextrkr.regionprops_func import (
    regionprops_singlefile, recomputed_variable_names, conv_dilate_overwrite_km, track_classes_from_config,
    cluster_groups_text, cluster_global_attrs, cluster_radii_from_cloudid,
)

# Variables that a cloudid file must contain for this step (cell_cluster_number: idfeature_radar3d with radii_expand)
REQUIRED_CLOUDID_VARS = [
    "convsf", "cell_cluster_number", "latitude", "longitude", "x", "y",
    "echotop10", "echotop20", "echotop30", "echotop40", "echotop50",
]

# What to do when the cloudid files have no cell_cluster_number
MISSING_CLUSTER_HINT = (" cell_cluster_number is written by idfeature_radar3d with radii_expand (default [1, 2, 3, 4, 5, 6, 7] km, "
                        "null = off): run idfeature with it.")

# Largest fraction of the (track, time) pairs that may stay uncomputed or inconsistent before the step aborts
MAX_BAD_FRACTION = 0.05


def compute_regionprops(config):
    """
    Second pass of the radar_feature MCS statistics: recompute the shape and extreme statistics of every track on its
    merged mask, then stage the lifecycle of the long tracks.

    identifymcs_radar folds the merge/split fragments into the track they belong to, so the mask of a track at one
    time is its own feature plus the fragment features (what mapfeature paints as cloudtracknumber). This step reads
    the file of identifymcs_radar (mcsradarstats_filebase) and, for every cloudid file in parallel, recomputes on the
    merged mask: centroid and reflectivity-weighted centroid, major/minor axis, aspect ratio, orientation, eccentricity,
    convective major axis (conv_majoraxis, of the longest group of nearby convective cells, see cell_cluster_number in
    idfeature_radar3d.py), max_dbz, maxETH_*dbz and the area_*dbz / conv_area_*dbz variables. The smoothed trajectory is
    recomputed from the new centroid, and the lifecycle stages (lifecycle_* variables, see mcs_lifecycle.py) are
    classified from total_conv_area, total_strat_area and the new conv_majoraxis.

    The groups of nearby convective cells are the cell clusters of the cloudid files (cell_cluster_number: the connected
    groups of the inflated cells, grown by radii_expand at idfeature time). If conv_dilate_overwrite_radius_km is set, the
    clusters are dilated further by that radius and labeled again here (the cloudid files are not modified), to connect
    adjacent clusters without re-running idfeature: the tracks do not depend on it, only conv_majoraxis and the lifecycle do.

    With regionprops_track_classes (a list of track_class values, e.g. [1, 2] for MCS and multi-cell tracks; default
    null = all tracks) only those tracks are measured, staged and smoothed; the other tracks keep the first-pass
    values (which describe their own feature, not the merged mask), get no lifecycle staging, and are not
    included in the guardrails. The merged masks of the measured tracks are the same as with all tracks.

    The first-pass file is dense (tracks x times x mergers: 180 GB uncompressed for a month), so it is opened
    lazily: only the variables used here are read, the others are streamed into the output file.

    The tracks and their order are those of the first pass (pixel tracknumber = index + 1 stays valid), and which
    tracks are MCS (is_mcs, track_class, feature_class, mcs_status) stays as decided by identifymcs_radar; this step
    only logs how many tracks the recomputed major axis and 45-dBZ area would change.

    Args:
        config: dictionary
            Dictionary containing config parameters.

    Returns:
        statistics_outfile: string
            Track statistics file name (f"{stats_path}{mcsrobust_filebase}{startdate}_{enddate}.nc"), the input of
            mapfeature_driver and movement_speed.
    """
    logger = logging.getLogger(__name__)
    logger.info("Recomputing track statistics on the merged mask and staging the lifecycle")

    mcsradarstats_filebase = config.get("mcsradarstats_filebase", "mcs_tracks_")
    mcsrobust_filebase = config["mcsrobust_filebase"]
    stats_path = config["stats_outpath"]
    tracking_outpath = config["tracking_outpath"]
    cloudid_filebase = config["cloudid_filebase"]
    startdate = config["startdate"]
    enddate = config["enddate"]
    tracks_dimname = config.get("tracks_dimname", "tracks")
    times_dimname = config.get("times_dimname", "times")
    run_parallel = config.get("run_parallel", 0)
    match_pixel_dt_thresh = config["match_pixel_dt_thresh"]
    time_resolution = config["datatimeresolution"]
    max_speed_kmh = config.get("max_speed_for_trajectory_smoothing", 50) * 3.6
    fillval = config.get("fillval", -9999)
    fillval_f = np.nan
    dbz_area_threshs = config.get("trackstats_dbz_area_threshs", [45, 50])
    mcs_lifecycle_thresh = config.get("mcs_lifecycle_thresh", 8)
    params = mcs_params_from_config(config)
    try:
        overwrite_km = conv_dilate_overwrite_km(config)
        track_classes = track_classes_from_config(config)
    except ValueError as err:
        logger.critical(f"{err} Tracking will now exit.")
        sys.exit()

    firstpass_file = f"{stats_path}{mcsradarstats_filebase}{startdate}_{enddate}.nc"
    statistics_outfile = f"{stats_path}{mcsrobust_filebase}{startdate}_{enddate}.nc"

    #########################################################################
    # First-pass statistics
    if firstpass_file == statistics_outfile:
        logger.critical(f"mcsradarstats_filebase and mcsrobust_filebase give the same file ({firstpass_file}); the first-pass "
                        f"file is read while the output is written, so they must differ. Tracking will now exit.")
        sys.exit()
    ds = xr.open_dataset(firstpass_file, mask_and_scale=False, decode_times=False)    # lazy: closed after the output is written
    ntracks = ds.sizes[tracks_dimname]
    ntimes = ds.sizes[times_dimname]
    basetime = ds["base_time"].values
    cloudnumber = ds["cloudnumber"].values
    merge_cloudnumber = ds["merge_cloudnumber"].values
    split_cloudnumber = ds["split_cloudnumber"].values
    total_area = ds["total_area"].values
    track_duration = ds["track_duration"].values
    # tracks to measure (all, or the selected track classes)
    if track_classes is None:
        select_track = np.ones(ntracks, dtype=bool)
    else:
        if "track_class" not in ds:
            logger.critical(f"regionprops_track_classes is set but {firstpass_file} has no track_class. Tracking will now exit.")
            sys.exit()
        select_track = np.isin(ds["track_class"].values, track_classes)
    n_pairs_all = int(np.isfinite(basetime).sum())
    # the pairs this step is responsible for (guardrails, smoothing, lifecycle)
    valid_pairs = np.isfinite(basetime) & select_track[:, None]
    n_valid = int(valid_pairs.sum())
    selection_text = "" if track_classes is None else (
        f"; regionprops_track_classes {track_classes}: {int(select_track.sum())} tracks, {n_valid} pairs measured")
    logger.info(f"First-pass file {firstpass_file}: {ntracks} tracks, {n_pairs_all} (track, time) pairs{selection_text}")

    #########################################################################
    # Cloudid files, and a check of the first one for the variables this step needs
    cloudidfiles, cloudidfiles_basetime, _, _ = subset_files_timerange(
        tracking_outpath, cloudid_filebase, config["start_basetime"], config["end_basetime"],
    )
    nfiles = len(cloudidfiles)
    if nfiles == 0:
        logger.critical(f"No cloudid files found in {tracking_outpath}. Tracking will now exit.")
        sys.exit()
    required = list(REQUIRED_CLOUDID_VARS)
    # conv_diameter_sum (sum of the circular-equivalent diameters of the convective cells) is recomputed on the merged mask when
    # the first-pass file has it (first-pass files from before that variable existed do not, and then need no cell_number)
    with_conv_diameter_sum = "conv_diameter_sum" in ds
    if with_conv_diameter_sum:
        required = required + ["cell_number"]
    with xr.open_dataset(cloudidfiles[0], decode_times=False, mask_and_scale=False) as dsp:
        missing = [v for v in required + [config.get("feature_varname", "feature_number")] if v not in dsp]
        radii_expand_km = cluster_radii_from_cloudid(dsp)
    if missing:
        hint = MISSING_CLUSTER_HINT if "cell_cluster_number" in missing else ""
        logger.critical(f"Cloudid file {cloudidfiles[0]} has no {missing}.{hint} Tracking will now exit.")
        sys.exit()
    if overwrite_km is not None:
        logger.info(f"cell_cluster_number dilated further by {overwrite_km:g} km and labeled again (conv_dilate_overwrite_radius_km)")
        if config.get("rangemask_varname", None) is not None:
            logger.warning("rangemask_varname is set: the further dilation of the clusters does not use the valid-data mask "
                           "(it is not stored in the cloudid files).")

    #########################################################################
    # Statistics of the merged mask, one cloudid file at a time
    names = recomputed_variable_names(dbz_area_threshs, with_conv_diameter_sum)
    # (track, time) entries of each cloudid file, in one pass over the valid entries
    pair_index = match_pairs_to_files(basetime, cloudidfiles_basetime, match_pixel_dt_thresh)
    results = []
    for ifile in range(nfiles):
        itrack, itime = pair_index[ifile]
        args = (
            cloudidfiles[ifile],
            cloudnumber[itrack, itime],
            merge_cloudnumber[itrack, itime, :],
            split_cloudnumber[itrack, itime, :],
            config,
        )
        kwargs = {} if track_classes is None else {"compute_track": select_track[itrack]}
        if run_parallel == 0:
            results.append(regionprops_singlefile(*args, **kwargs))
        elif run_parallel >= 1:
            results.append(dask.delayed(regionprops_singlefile)(*args, **kwargs))
        else:
            sys.exit('Valid parallelization flag not provided.')
    if run_parallel >= 1:
        from dask.distributed import wait       # only the parallel mode needs the `distributed` package (the CI environment has none)
        results = dask.compute(*results)
        wait(results)

    if track_classes is None:
        values = {name: np.full((ntracks, ntimes), fillval_f, dtype=np.float32) for name in names}
    else:
        # tracks that are not measured keep their first-pass values
        values = {name: ds[name].values.astype(np.float32) for name in names}
    merged_area = np.full((ntracks, ntimes), fillval_f, dtype=np.float32)
    computed = np.zeros((ntracks, ntimes), dtype=bool)
    n_conflict = n_out_of_range = 0
    for (itrack, itime), result in zip(pair_index, results):
        if result is None:
            continue
        out, diag = result
        sel = select_track[itrack]
        itrack, itime = itrack[sel], itime[sel]
        for name in names:
            values[name][itrack, itime] = out[name][sel]
        merged_area[itrack, itime] = out["merged_area"][sel]
        computed[itrack, itime] = True
        n_conflict += diag["n_conflict"]
        n_out_of_range += diag["n_out_of_range"]
    # a track whose own feature is not in its cloudid file has no mask (NaN area) either
    computed &= np.isfinite(merged_area)

    #########################################################################
    # Guardrails: pairs that could not be computed, and merged masks whose area is not total_area
    n_missing = n_valid - int(np.count_nonzero(computed & valid_pairs))
    if n_missing > 0:
        logger.warning(f"{n_missing} of {n_valid} (track, time) pairs have no readable cloudid file or no feature in it; "
                       f"their recomputed statistics are NaN.")
    if n_conflict > 0 or n_out_of_range > 0:
        logger.warning(f"{n_conflict} feature number(s) claimed by more than one track and {n_out_of_range} "
                       f"feature number(s) not in their cloudid file.")
    check = computed & valid_pairs
    area_tol = 0.5 + 2e-6 * np.abs(np.nan_to_num(total_area))
    mismatch = check & (np.abs(merged_area - total_area) > area_tol)
    n_mismatch = int(mismatch.sum())
    logger.info(f"Merged mask area equals total_area at {int(check.sum()) - n_mismatch} of {int(check.sum())} pairs.")
    if n_mismatch > 0:
        logger.warning(f"{n_mismatch} pairs have a merged mask area different from total_area "
                       f"(largest difference {np.nanmax(np.abs(merged_area - total_area)[mismatch]):.1f} km^2).")
    for what, count in (("have no readable cloudid file or no feature in it", n_missing), ("have a merged mask area different from total_area", n_mismatch)):
        if n_valid > 0 and count / n_valid > MAX_BAD_FRACTION:
            logger.critical(f"More than {MAX_BAD_FRACTION:.0%} of the (track, time) pairs {what} ({count} of {n_valid}). "
                            f"Tracking will now exit.")
            sys.exit()

    #########################################################################
    # Assign the recomputed variables (over the first-pass ones)
    dims2d = [tracks_dimname, times_dimname]
    groups_text = cluster_groups_text(radii_expand_km, overwrite_km)
    for name in names:
        attrs = dict(ds[name].attrs)
        attrs["comments"] = "Computed on the merged mask: the track's own feature plus the merge/split fragments folded into it"
        if track_classes is not None:
            subset_note = (f" Only for tracks with track_class in {track_classes}; the other tracks keep the first-pass value "
                           f"(their own feature, not the merged mask).")
            attrs["comments"] += subset_note
        if name == "conv_majoraxis":
            attrs["long_name"] = "Major axis length of the longest convective feature in the merged mask"
            attrs["comments"] = (
                "Convective pixels (Steiner class 4) are grouped by " + groups_text +
                "; the major axis of a group is measured on the region inside the merged mask. "
                "0 if the mask has no convective pixel."
            )
            if track_classes is not None:
                attrs["comments"] += subset_note
        ds[name] = (dims2d, values[name], attrs)

    # Smoothed trajectory from the recomputed centroid
    old_lon_smooth = ds["meanlon_smooth"].values
    old_nan = int(np.isnan(old_lon_smooth)[valid_pairs].sum())
    meanlon, meanlat = values["meanlon"], values["meanlat"]
    if track_classes is None:
        lon_smooth = np.full((ntracks, ntimes), fillval_f, dtype=np.float32)
        lat_smooth = np.full((ntracks, ntimes), fillval_f, dtype=np.float32)
    else:
        lon_smooth = old_lon_smooth.astype(np.float32)
        lat_smooth = ds["meanlat_smooth"].values.astype(np.float32)
    for itrack in np.where(select_track)[0]:
        # A track fills only the first track_duration of the ntimes columns, the rest is NaN and stays NaN: smooth these entries
        # (not all ntimes, which is slow in pure Python for 100,000s of tracks). At least 3, because the smoothing takes other
        # branches for trajectories of 1 and 2 points (tests/test_smooth_prefix.py: same result as the full row)
        n = min(ntimes, max(int(track_duration[itrack]), 3))
        lon_s, lat_s = smooth_trajectory(
            meanlon[itrack, :n], meanlat[itrack, :n], max_speed_kmh=max_speed_kmh, time_step_h=time_resolution,
        )
        lon_smooth[itrack, :n] = lon_s
        lat_smooth[itrack, :n] = lat_s
        lon_smooth[itrack, n:] = meanlon[itrack, n:]
        lat_smooth[itrack, n:] = meanlat[itrack, n:]
    ds["meanlon_smooth"] = (dims2d, lon_smooth, dict(ds["meanlon_smooth"].attrs))
    ds["meanlat_smooth"] = (dims2d, lat_smooth, dict(ds["meanlat_smooth"].attrs))
    new_nan = int(np.isnan(lon_smooth)[valid_pairs].sum())
    logger.info(f"Smoothed trajectory: {old_nan} NaN positions in the first pass, {new_nan} with the merged-mask centroid.")

    #########################################################################
    # MCS criteria with the recomputed size variable (major axis or conv_diameter_sum, see mcs_size_method) and 45-dBZ area
    # (the MCS flags themselves stay as decided by the first pass)
    size_varname = mcs_size_varname(params)
    area45_varname = config.get("mcs_dbz45_area_varname", "area_45dbz")
    is_mcs = ds["is_mcs"].values.astype(bool)
    track_lifetime_hr = track_duration * time_resolution
    lost = gained = 0
    for nt in np.where((track_lifetime_hr >= params["mcs_duration_thresh"]) & select_track)[0]:
        n = int(track_duration[nt])
        _, now_mcs = identify_mcs_periods_by_method(
            ds[size_varname].values[nt, :n], ds[area45_varname].values[nt, :n], basetime[nt, :n],
            time_resolution, params,
        )
        lost += int(is_mcs[nt] and not now_mcs)
        gained += int((not is_mcs[nt]) and now_mcs)
    logger.info(f"With the merged-mask {size_varname} and {area45_varname}: {lost} of the {int(is_mcs.sum())} MCS would "
                f"no longer meet the MCS criteria and {gained} other tracks would (flags not changed).")

    #########################################################################
    # Lifecycle staging
    lifecycle_stage = np.full((ntracks, ntimes), fillval, dtype=np.int16)
    lifecycle_cycle = np.full((ntracks, ntimes), fillval, dtype=np.int16)
    lifecycle_index = np.full((ntracks, 5), fillval, dtype=np.int64)
    lifecycle_ncycles = np.full(ntracks, fillval, dtype=np.int32)
    lifecycle_complete_flag = np.full(ntracks, fillval, dtype=np.int16)
    lifecycle_evaluated = np.zeros(ntracks, dtype=np.int16)
    total_conv_area = ds["total_conv_area"].values
    total_strat_area = ds["total_strat_area"].values
    conv_majoraxis = ds["conv_majoraxis"].values
    for nt in np.where((track_lifetime_hr >= mcs_lifecycle_thresh) & select_track)[0]:
        n = int(track_duration[nt])
        time_hours = (basetime[nt, :n] - basetime[nt, 0]) / 3600.0
        stage, cyc, lc_idx, ncyc, complete = classify_lifecycle(
            total_conv_area[nt, :n], conv_majoraxis[nt, :n], total_strat_area[nt, :n],
            time_resolution, params, time_hours=time_hours,
        )
        lifecycle_stage[nt, :n] = stage
        lifecycle_cycle[nt, :n] = cyc
        lifecycle_index[nt, :] = lc_idx
        lifecycle_ncycles[nt] = ncyc
        lifecycle_complete_flag[nt] = int(complete)
        lifecycle_evaluated[nt] = 1
    logger.info(f"Lifecycle evaluated for {int(lifecycle_evaluated.sum())} tracks (lifetime >= {mcs_lifecycle_thresh} h)")

    lifecycle_attrs = {
        "lifecycle_stage": {"long_name": "MCS lifecycle stage: 1=cloud only, 2=unorganized convection, "
                            "3=genesis, 4=mature, 5=decay", "units": "unitless", "_FillValue": fillval},
        "lifecycle_cycle": {"long_name": "Which organized-convective-line episode (1-based) a time "
                            "belongs to; 0 = not part of any episode", "units": "unitless", "_FillValue": fillval},
        "lifecycle_index": {"long_name": "Time index of the first occurrence of each lifecycle stage "
                            "(1-5); -1 = that stage was not observed", "units": "unitless", "_FillValue": fillval},
        "lifecycle_ncycles": {"long_name": "Number of organized-convective-line episodes found",
                              "units": "unitless", "_FillValue": fillval},
        "lifecycle_complete_flag": {"long_name": "Flag indicating genesis, mature, and decay stages "
                                    "were all observed", "units": "unitless", "_FillValue": fillval},
        "lifecycle_evaluated": {"long_name": "Flag indicating lifecycle staging was evaluated for this "
                                "track (track lifetime >= mcs_lifecycle_thresh)", "units": "unitless", "_FillValue": fillval},
    }
    for name, (dims, arr) in {
        "lifecycle_stage": (dims2d, lifecycle_stage),
        "lifecycle_cycle": (dims2d, lifecycle_cycle),
        "lifecycle_index": ([tracks_dimname, "lifestages"], lifecycle_index),
        "lifecycle_ncycles": ([tracks_dimname], lifecycle_ncycles),
        "lifecycle_complete_flag": ([tracks_dimname], lifecycle_complete_flag),
        "lifecycle_evaluated": ([tracks_dimname], lifecycle_evaluated),
    }.items():
        attrs = dict(lifecycle_attrs[name])
        attrs["comments"] = ("Classified from total_conv_area, total_strat_area and the merged-mask conv_majoraxis "
                             "(mcs_lifecycle.classify_lifecycle)")
        ds[name] = (dims, arr, attrs)

    ds = ds.assign_coords(lifestages=np.arange(5))
    ds.attrs["Title"] = ("Track statistics of radar-based MCS with shape and extremes recomputed on the merged mask, "
                         "and lifecycle staging")
    ds.attrs["mcs_lifecycle_thresh_hour"] = mcs_lifecycle_thresh
    ds.attrs["mcs_lifecycle_smooth_hour"] = params["mcs_lifecycle_smooth_hour"]
    ds.attrs.update(cluster_global_attrs(radii_expand_km, overwrite_km))
    if track_classes is not None:
        ds.attrs["regionprops_track_classes"] = np.array(track_classes, dtype=np.int32)
    ds.attrs["n_pairs_not_computed"] = n_missing
    ds.attrs["n_pairs_area_mismatch"] = n_mismatch
    ds.attrs["Created_on"] = time.ctime(time.time())

    #########################################################################
    # Write output
    # (the INFO time stamps of the lifecycle line, "Writing" and the file name below give the write time)
    logger.info(f"Writing {statistics_outfile}")
    if os.path.isfile(statistics_outfile):
        os.remove(statistics_outfile)
    comp = dict(zlib=True)
    encoding = {var: comp for var in ds.data_vars}
    open(statistics_outfile, 'ab').close()
    ds.to_netcdf(path=statistics_outfile, mode="w", format="NETCDF4", unlimited_dims=tracks_dimname, encoding=encoding)
    ds.close()
    logger.info(f"{statistics_outfile}")
    return statistics_outfile
