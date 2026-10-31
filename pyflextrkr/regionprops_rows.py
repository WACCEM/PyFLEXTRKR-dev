"""
Rows-mode regionprops (config stats_format: rows).

The same second pass as regionprops_driver.compute_regionprops (shape and extreme statistics on the merged mask, smoothed
trajectory, lifecycle staging) on the row tables of identifymcs_rows instead of the dense [tracks, times, slots] file. The (track,
time) entries are positions in flat arrays (offsets[track] + time index), the fragments of an entry are looked up in the
fragments table, and the result is written as row tables. The worker that measures one cloudid file (regionprops_singlefile) is
the one of the dense step, called with the same arguments, so the numbers are the same.

tests/test_regionprops_rows.py checks that the tables equal what dense_to_rows makes of the dense robust file.

Author: Zhe Feng | zhe.feng@pnnl.gov
"""
import logging
import os
import shutil
import sys
import time

import dask
import numpy as np

from pyflextrkr.ft_utilities import match_pairs_to_files, subset_files_timerange
from pyflextrkr.identifymcs_radar import identify_mcs_periods, mcs_params_from_config
from pyflextrkr.mcs_lifecycle import classify_lifecycle
from pyflextrkr.regionprops_driver import MAX_BAD_FRACTION, REQUIRED_CLOUDID_VARS
from pyflextrkr.regionprops_func import (
    conv_dilate_overwrite_km, recomputed_variable_names, regionprops_singlefile, track_classes_from_config,
)
from pyflextrkr.rowstats import FRAGMENT_KINDS, TableBuilder, read_tables, write_rows
from pyflextrkr.smooth_trajectory import smooth_trajectory


def fragment_slots(entry_of_fragment, slot_of_fragment, cloudnumber_of_fragment, entries, nslots, fill):
    """
    The merge (or split) cloud numbers attached at some entries, in the layout of the dense [tracks, times, slots] variable.

    Args:
        entry_of_fragment: int array [nfragments]
            Flat entry (offsets[track] + time index) each fragment of one kind is attached at, ascending.
        slot_of_fragment: int array [nfragments]
            Slot of each fragment.
        cloudnumber_of_fragment: int array [nfragments]
            Cloud number of each fragment.
        entries: int array [npairs]
            Flat entries to look up.
        nslots: int
            Number of slots.
        fill: int
            Value of an empty slot.

    Returns:
        slots: int array [npairs, nslots]
            What the dense variable has at [track, time, :] of every entry.
    """
    out = np.full((len(entries), nslots), fill, dtype=cloudnumber_of_fragment.dtype)
    first = np.searchsorted(entry_of_fragment, entries, side="left")
    count = np.searchsorted(entry_of_fragment, entries, side="right") - first
    pair = np.repeat(np.arange(len(entries)), count)
    index = np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count) + np.repeat(first, count)
    out[pair, slot_of_fragment[index]] = cloudnumber_of_fragment[index]
    return out


def compute_regionprops_rows(config):
    """
    Recompute the shape and extreme statistics of every track on its merged mask and stage the lifecycle of the long tracks:
    regionprops_driver.compute_regionprops with row tables as input and output.

    Args:
        config: dictionary
            Config parameters of compute_regionprops (regionprops_track_classes, conv_dilate_overwrite_radius_km, ...).

    Returns:
        statistics_outfolder: string
            Folder of the row tables f"{stats_path}{mcsrobust_filebase}{startdate}_{enddate}.rows", the input of the rows-mode
            mapfeature and movement speed. The input is the folder of identifymcs_radar_rows.
    """
    logger = logging.getLogger(__name__)
    logger.info("Recomputing track statistics on the merged mask and staging the lifecycle (row tables)")

    mcsradarstats_filebase = config.get("mcsradarstats_filebase", "mcs_tracks_")
    mcsrobust_filebase = config["mcsrobust_filebase"]
    stats_path = config["stats_outpath"]
    tracking_outpath = config["tracking_outpath"]
    cloudid_filebase = config["cloudid_filebase"]
    startdate, enddate = config["startdate"], config["enddate"]
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
    firstpass_folder = f"{stats_path}{mcsradarstats_filebase}{startdate}_{enddate}.rows"
    statistics_outfolder = f"{stats_path}{mcsrobust_filebase}{startdate}_{enddate}.rows"

    #########################################################################
    # First-pass statistics
    if firstpass_folder == statistics_outfolder:
        logger.critical(f"mcsradarstats_filebase and mcsrobust_filebase give the same folder ({firstpass_folder}). "
                        f"Tracking will now exit.")
        sys.exit()
    tables = read_tables(firstpass_folder)
    meta, tracks, entries, fragments = tables.meta, tables.tracks, tables.entries, tables.fragments
    tdim, kdim, sdim = meta["tracks_dim"], meta["times_dim"], meta["slots_dim"]
    offsets, track_duration = tables.offsets, tables.duration
    ntracks, nentries = len(track_duration), int(offsets[-1])
    track_of = entries["track"]
    basetime, cloudnumber, total_area = entries["base_time"], entries["cloudnumber"], entries["total_area"]
    if track_classes is None:
        select_track = np.ones(ntracks, dtype=bool)
    else:
        if "track_class" not in tracks:
            logger.critical(f"regionprops_track_classes is set but {firstpass_folder} has no track_class. "
                            f"Tracking will now exit.")
            sys.exit()
        select_track = np.isin(tracks["track_class"], track_classes)
    n_pairs_all = int(np.isfinite(basetime).sum())
    valid_pairs = np.isfinite(basetime) & select_track[track_of]
    n_valid = int(valid_pairs.sum())
    selection_text = "" if track_classes is None else (
        f"; regionprops_track_classes {track_classes}: {int(select_track.sum())} tracks, {n_valid} pairs measured")
    logger.info(f"First-pass file {firstpass_folder}: {ntracks} tracks, {n_pairs_all} (track, time) pairs{selection_text}")

    #########################################################################
    # Cloudid files, and a check of the first one for the variables this step needs
    import xarray as xr
    cloudidfiles, cloudidfiles_basetime, _, _ = subset_files_timerange(
        tracking_outpath, cloudid_filebase, config["start_basetime"], config["end_basetime"])
    nfiles = len(cloudidfiles)
    if nfiles == 0:
        logger.critical(f"No cloudid files found in {tracking_outpath}. Tracking will now exit.")
        sys.exit()
    required = [v for v in REQUIRED_CLOUDID_VARS if overwrite_km is None or v != "conv_feature_number"]
    with xr.open_dataset(cloudidfiles[0], decode_times=False, mask_and_scale=False) as dsp:
        missing = [v for v in required + [config.get("feature_varname", "feature_number")] if v not in dsp]
        idfeature_radius_km = dsp.attrs.get("conv_dilate_radius_km", config.get("conv_dilate_radius_km", None))
    if missing:
        hint = (" conv_feature_number is written by the version of idfeature_radar3d that has the conv_dilate_radius_km "
                "option: run run_idfeature with it, or set conv_dilate_overwrite_radius_km to recompute it here."
                if "conv_feature_number" in missing else "")
        logger.critical(f"Cloudid file {cloudidfiles[0]} has no {missing}.{hint} Tracking will now exit.")
        sys.exit()
    if overwrite_km is None:
        conv_dilate_radius_km = idfeature_radius_km
    else:
        conv_dilate_radius_km = overwrite_km
        idfeature_text = "no recorded radius" if idfeature_radius_km is None else f"{float(idfeature_radius_km):g} km"
        logger.info(f"conv_feature_number recomputed from convsf at {overwrite_km:g} km (conv_dilate_overwrite_radius_km); "
                    f"the cloudid files have {idfeature_text}")
        if config.get("rangemask_varname", None) is not None:
            logger.warning("rangemask_varname is set: idfeature limits the dilation to the valid-data mask, the "
                           "recomputed conv_feature_number does not (the mask is not stored in the cloudid files).")

    #########################################################################
    # Statistics of the merged mask, one cloudid file at a time
    names = recomputed_variable_names(dbz_area_threshs)
    # The flat entries of each cloudid file (all entries are in one row of a [1, nentries] array: the index in it is the entry)
    pair_index = match_pairs_to_files(basetime[None, :], cloudidfiles_basetime, match_pixel_dt_thresh)
    nslots = meta["dims"][sdim]
    slot_fill = meta["fill"]["merge_cloudnumber"]
    frag_entry = offsets[fragments["track"]] + fragments["k"]
    by_kind = {}
    for code, kind in enumerate(FRAGMENT_KINDS):
        use = fragments["kind"] == code
        by_kind[kind] = (frag_entry[use], fragments["slot"][use], fragments["cloudnumber"][use])
    results = []
    for ifile in range(nfiles):
        entry = pair_index[ifile][1]
        args = (
            cloudidfiles[ifile],
            cloudnumber[entry],
            fragment_slots(*by_kind["merge"], entry, nslots, slot_fill),
            fragment_slots(*by_kind["split"], entry, nslots, slot_fill),
            config,
        )
        kwargs = {} if track_classes is None else {"compute_track": select_track[track_of[entry]]}
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
        values = {name: np.full(nentries, fillval_f, dtype=np.float32) for name in names}
    else:
        values = {name: entries[name].astype(np.float32) for name in names}        # tracks that are not measured keep their values
    merged_area = np.full(nentries, fillval_f, dtype=np.float32)
    computed = np.zeros(nentries, dtype=bool)
    n_conflict = n_out_of_range = 0
    for (_, entry), result in zip(pair_index, results):
        if result is None:
            continue
        out, diag = result
        sel = select_track[track_of[entry]]
        entry = entry[sel]
        for name in names:
            values[name][entry] = out[name][sel]
        merged_area[entry] = out["merged_area"][sel]
        computed[entry] = True
        n_conflict += diag["n_conflict"]
        n_out_of_range += diag["n_out_of_range"]
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
    for what, count in (("have no readable cloudid file or no feature in it", n_missing),
                        ("have a merged mask area different from total_area", n_mismatch)):
        if n_valid > 0 and count / n_valid > MAX_BAD_FRACTION:
            logger.critical(f"More than {MAX_BAD_FRACTION:.0%} of the (track, time) pairs {what} ({count} of {n_valid}). "
                            f"Tracking will now exit.")
            sys.exit()

    #########################################################################
    # Assign the recomputed variables (over the first-pass ones)
    builder = TableBuilder.from_tables(tables)
    radius_text = ("" if conv_dilate_radius_km is None else f" by a disc of radius {float(conv_dilate_radius_km):g} km")
    for name in names:
        attrs = dict(meta["attrs"][name])
        attrs["comments"] = "Computed on the merged mask: the track's own feature plus the merge/split fragments folded into it"
        if track_classes is not None:
            subset_note = (f" Only for tracks with track_class in {track_classes}; the other tracks keep the first-pass value "
                           f"(their own feature, not the merged mask).")
            attrs["comments"] += subset_note
        if name == "conv_majoraxis":
            attrs["long_name"] = "Major axis length of the longest convective feature in the merged mask"
            attrs["comments"] = (
                "Convective pixels (Steiner class 4) are grouped by conv_feature_number (dilated" + radius_text +
                " and labeled); the major axis of a group is measured on the region inside the merged mask. "
                "0 if the mask has no convective pixel."
            )
            if overwrite_km is not None:
                attrs["comments"] += (" The groups were recomputed by run_regionprops (conv_dilate_overwrite_radius_km), "
                                      "not read from the cloudid files.")
            if track_classes is not None:
                attrs["comments"] += subset_note
        # a track with no data holds NaN; with a subset of tracks the others keep their first-pass fill value
        fill = fillval_f if track_classes is None else float(meta["fill"][name])
        builder.add_entry_var(name, values[name], attrs, fill=fill)
    current = lambda name: values[name] if name in values else entries[name]      # the variable as the next lines see it

    # Smoothed trajectory from the recomputed centroid
    old_lon_smooth = entries["meanlon_smooth"]
    old_nan = int(np.isnan(old_lon_smooth)[valid_pairs].sum())
    meanlon, meanlat = values["meanlon"], values["meanlat"]
    if track_classes is None:
        lon_smooth = np.full(nentries, fillval_f, dtype=np.float32)
        lat_smooth = np.full(nentries, fillval_f, dtype=np.float32)
    else:
        lon_smooth = old_lon_smooth.astype(np.float32)
        lat_smooth = entries["meanlat_smooth"].astype(np.float32)
    for itrack in np.where(select_track)[0]:
        # The valid entries, at least 3: the smoothing takes other branches for trajectories of 1 and 2 points
        # (tests/test_smooth_prefix.py: same result as the full row of the dense layout)
        a, b = offsets[itrack], offsets[itrack + 1]
        n = max(int(b - a), 3)
        lon_in, lat_in = np.full(n, np.nan), np.full(n, np.nan)
        lon_in[:b - a], lat_in[:b - a] = meanlon[a:b], meanlat[a:b]
        lon_s, lat_s = smooth_trajectory(lon_in, lat_in, max_speed_kmh=max_speed_kmh, time_step_h=time_resolution)
        lon_smooth[a:b], lat_smooth[a:b] = lon_s[:b - a], lat_s[:b - a]
    builder.add_entry_var("meanlon_smooth", lon_smooth, dict(meta["attrs"]["meanlon_smooth"]), fill=fillval_f)
    builder.add_entry_var("meanlat_smooth", lat_smooth, dict(meta["attrs"]["meanlat_smooth"]), fill=fillval_f)
    new_nan = int(np.isnan(lon_smooth)[valid_pairs].sum())
    logger.info(f"Smoothed trajectory: {old_nan} NaN positions in the first pass, {new_nan} with the merged-mask centroid.")

    #########################################################################
    # MCS criteria with the recomputed major axis and 45-dBZ area (the MCS flags themselves stay as decided
    # by the first pass)
    area45_varname = config.get("mcs_dbz45_area_varname", "area_45dbz")
    majoraxis, area45 = current("majoraxis"), current(area45_varname)
    is_mcs = tracks["is_mcs"].astype(bool)
    track_lifetime_hr = track_duration * time_resolution
    lost = gained = 0
    for nt in np.where((track_lifetime_hr >= params["mcs_duration_thresh"]) & select_track)[0]:
        sl = slice(offsets[nt], offsets[nt + 1])
        _, now_mcs = identify_mcs_periods(majoraxis[sl], area45[sl], basetime[sl], time_resolution, params)
        lost += int(is_mcs[nt] and not now_mcs)
        gained += int((not is_mcs[nt]) and now_mcs)
    logger.info(f"With the merged-mask major axis and {area45_varname}: {lost} of the {int(is_mcs.sum())} MCS would "
                f"no longer meet the MCS criteria and {gained} other tracks would (flags not changed).")

    #########################################################################
    # Lifecycle staging
    lifecycle_stage = np.full(nentries, fillval, dtype=np.int16)
    lifecycle_cycle = np.full(nentries, fillval, dtype=np.int16)
    lifecycle_index = np.full((ntracks, 5), fillval, dtype=np.int64)
    lifecycle_ncycles = np.full(ntracks, fillval, dtype=np.int32)
    lifecycle_complete_flag = np.full(ntracks, fillval, dtype=np.int16)
    lifecycle_evaluated = np.zeros(ntracks, dtype=np.int16)
    total_conv_area, total_strat_area, conv_majoraxis = entries["total_conv_area"], entries["total_strat_area"], current("conv_majoraxis")
    for nt in np.where((track_lifetime_hr >= mcs_lifecycle_thresh) & select_track)[0]:
        sl = slice(offsets[nt], offsets[nt + 1])
        time_hours = (basetime[sl] - basetime[offsets[nt]]) / 3600.0
        stage, cyc, lc_idx, ncyc, complete = classify_lifecycle(
            total_conv_area[sl], conv_majoraxis[sl], total_strat_area[sl], time_resolution, params, time_hours=time_hours)
        lifecycle_stage[sl] = stage
        lifecycle_cycle[sl] = cyc
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
    comment = ("Classified from total_conv_area, total_strat_area and the merged-mask conv_majoraxis "
               "(mcs_lifecycle.classify_lifecycle)")
    attrs_of = lambda name: dict(lifecycle_attrs[name], comments=comment)
    builder.add_entry_var("lifecycle_stage", lifecycle_stage, attrs_of("lifecycle_stage"), fill=fillval)
    builder.add_entry_var("lifecycle_cycle", lifecycle_cycle, attrs_of("lifecycle_cycle"), fill=fillval)
    builder.add_track2d_var("lifecycle_index", lifecycle_index, "lifestages", attrs_of("lifecycle_index"))
    builder.add_track_var("lifecycle_ncycles", lifecycle_ncycles, attrs_of("lifecycle_ncycles"))
    builder.add_track_var("lifecycle_complete_flag", lifecycle_complete_flag, attrs_of("lifecycle_complete_flag"))
    builder.add_track_var("lifecycle_evaluated", lifecycle_evaluated, attrs_of("lifecycle_evaluated"))
    builder.add_coord("lifestages", np.arange(5))

    attrs = builder.global_attrs
    attrs["Title"] = ("Track statistics of radar-based MCS with shape and extremes recomputed on the merged mask, "
                      "and lifecycle staging")
    attrs["mcs_lifecycle_thresh_hour"] = mcs_lifecycle_thresh
    attrs["mcs_lifecycle_smooth_hour"] = params["mcs_lifecycle_smooth_hour"]
    if conv_dilate_radius_km is not None:
        attrs["conv_dilate_radius_km"] = float(conv_dilate_radius_km)
    if overwrite_km is not None and idfeature_radius_km is not None:
        attrs["conv_dilate_radius_km_idfeature"] = float(idfeature_radius_km)
    if track_classes is not None:
        attrs["regionprops_track_classes"] = np.array(track_classes, dtype=np.int32)
    attrs["n_pairs_not_computed"] = n_missing
    attrs["n_pairs_area_mismatch"] = n_mismatch
    attrs["Created_on"] = time.ctime(time.time())
    robust = builder.build()

    #########################################################################
    # Write output
    logger.info(f"Writing {statistics_outfolder}")
    if os.path.isdir(statistics_outfolder):
        shutil.rmtree(statistics_outfolder)
    write_rows(robust, statistics_outfolder)
    logger.info(f"{statistics_outfolder}")
    return statistics_outfolder
