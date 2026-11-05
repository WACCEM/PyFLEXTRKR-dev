"""
Rows-mode identifymcs_radar (config stats_format: rows).

The same MCS identification, merge/split fragment folding and feature classification as identifymcs_radar.py, on flat arrays
(one position for every valid (track, time) entry of every track, track after track) instead of [tracks, times] and
[tracks, times, slots] arrays, writing the row tables of pyflextrkr.rowstats (a Parquet folder) instead of the dense netCDF file.
Memory and time follow the number of entries (about 20 million for a year) and not tracks x longest track x 50 slots (about
3 TB for a year).

The per-track tests (identify_mcs_periods_by_method, find_mergesplit_parents, resolve_fragment_roots, classify_feature_class,
classify_track_class) are those of identifymcs_radar.py, called with slices of the flat arrays. Only the two functions that
work on the [tracks, times] arrays of the fragments have a flat version here (release_noncoexisting_fragments_flat,
attach_fragments_flat), written to give the same result as the dense ones; tests/test_identifymcs_rows.py checks that the
tables equal what dense_to_rows makes of the dense output.

Author: Zhe Feng | zhe.feng@pnnl.gov
"""
import logging
import os
import shutil
import sys
import time

import numpy as np

from pyflextrkr.identifymcs_radar import (
    MCS_NUMBER_LONG_NAME, classify_feature_class, classify_track_class, number_mcs_tracks, find_mergesplit_parents, identify_mcs_periods_by_method, mcs_size_test_text, mcs_size_varname,
    mcs_params_from_config, resolve_fragment_roots,
)
from pyflextrkr.rowstats import FRAGMENT_KINDS, TableBuilder, gather_index, load_sparse_entries, write_rows
from pyflextrkr.smooth_trajectory import smooth_trajectory

# Attributes of the variables that this step makes (the sparse trackstats variables keep the attributes of their file)
ATTRS_1D = {
    "is_mcs": "Flag indicating if this track meets the radar MCS criteria at any time",
    "track_class": "Most-organized feature class observed on this track: 1=MCS, 2=multi-cell, 3=single-cell, 4=stratiform",
}


def release_noncoexisting_fragments_flat(parent_idx, link_type, is_fragment, basetime, offsets):
    """
    Flat version of identifymcs_radar.release_noncoexisting_fragments: keep a candidate fragment only if every one of its time
    points can be painted into its resolved root track, otherwise release it so that it stays a standalone track.

    Args:
        parent_idx, link_type, is_fragment: 1-D arrays [ntracks]
            From find_mergesplit_parents.
        basetime: array [nentries]
            Base time of every entry (0 or NaN = no data), track after track.
        offsets: int array [ntracks + 1]
            Start of every track in the flat arrays plus the end of the last.

    Returns:
        parent_idx, link_type, root_idx, is_fragment_resolved, n_released:
            As the dense function: parent and link type of the resolved fragments (-1 and '' for the other tracks), the final
            root of every track, the final fragment flags and the number of candidates that were released.
    """
    parent_idx, link_type, is_fragment = parent_idx.copy(), link_type.copy(), is_fragment.copy()
    valid_cache = {}

    def valid_times(i):
        """Base times of the valid entries of track i (cached)."""
        if i not in valid_cache:
            row = basetime[offsets[i]:offsets[i + 1]]
            valid_cache[i] = row[(row != 0) & (~np.isnan(row))]
        return valid_cache[i]

    n_released = 0
    while True:
        root_idx, is_fragment_resolved = resolve_fragment_roots(parent_idx, is_fragment)
        released = [i for i in np.where(is_fragment_resolved)[0]
                    if not np.all(np.isin(valid_times(i), valid_times(int(root_idx[i]))))]
        if len(released) == 0:
            break
        is_fragment[released] = False
        parent_idx[released] = -1
        link_type[released] = ''
        n_released += len(released)

    parent_idx[~is_fragment_resolved] = -1
    link_type[~is_fragment_resolved] = ''
    return parent_idx, link_type, root_idx, is_fragment_resolved, n_released


def attach_fragments_flat(root_idx, link_type, basetime, cloudnumber, track_status, area, offsets, keep_track_indices,
                          nmaxmerge, sum_vars=None):
    """
    Flat version of identifymcs_radar.attach_fragments: attach every fragment's cloud number, track status and area at its
    matching base time to its resolved root track, as rows instead of [tracks, times, slots] arrays.

    Args:
        root_idx, link_type: 1-D arrays [ntracks]
            From resolve_fragment_roots / find_mergesplit_parents.
        basetime, cloudnumber, track_status, area: arrays [nentries]
            Flat variables of all tracks (track after track).
        offsets: int array [ntracks + 1]
            Start of every track in the flat arrays plus the end of the last.
        keep_track_indices: int array [nkept]
            Tracks to keep (0-based), in their new order.
        nmaxmerge: int
            Maximum number of fragments per track and time and kind.
        sum_vars: dictionary (default: None)
            {name: flat array [nentries]} of variables to add up over the fragments attached to each kept track.

    Returns:
        columns: dictionary
            Fragment rows {'track' (index among the kept tracks), 'k' (time index in the root), 'kind' (index in
            FRAGMENT_KINDS), 'slot', 'cloudnumber', 'status', 'area'} as arrays, in the order they were attached.
        fragment_sums: dictionary
            {name: float64 array [entries of the kept tracks]}: for every sum_vars entry the sum over the fragments attached to
            the track at that time (0 where none; NaN values are skipped).
        n_unmatched: int
            Fragment time points that could not be matched to a time of their root, or with a cloud number that is not
            positive (a slot is empty where its cloud number is not positive, so such a point cannot be stored).
        n_truncated: int
            Attachments dropped because more than nmaxmerge fragments attach to one track at one time.
    """
    keep = np.asarray(keep_track_indices, dtype=np.int64)
    sum_vars = {} if sum_vars is None else sum_vars
    new_index_of = np.full(len(root_idx), -1, dtype=np.int64)
    new_index_of[keep] = np.arange(len(keep))
    kept_offsets = np.concatenate(([0], np.cumsum(offsets[keep + 1] - offsets[keep])))
    fragment_sums = {name: np.zeros(int(kept_offsets[-1]), dtype=np.float64) for name in sum_vars}
    slots_used = {}
    rows = {c: [] for c in ("track", "k", "kind", "slot", "cloudnumber", "status", "area")}
    n_unmatched = n_truncated = 0

    for orig in range(len(root_idx)):
        lt = link_type[orig]
        if lt not in FRAGMENT_KINDS:
            continue
        root = int(root_idx[orig])
        new_idx = int(new_index_of[root])
        if new_idx < 0:
            n_unmatched += 1
            continue
        a, b = offsets[orig], offsets[orig + 1]
        frag_bt, frag_cn, frag_status, frag_area = basetime[a:b], cloudnumber[a:b], track_status[a:b], area[a:b]
        root_bt = basetime[offsets[root]:offsets[root + 1]]
        kind = FRAGMENT_KINDS.index(lt)
        for t in np.where((frag_bt != 0) & (~np.isnan(frag_bt)))[0]:
            match = np.where(root_bt == frag_bt[t])[0]
            if len(match) == 0:
                n_unmatched += 1
                continue
            tt = int(match[0])
            slot = slots_used.get((kind, new_idx, tt), 0)
            if slot >= nmaxmerge:
                n_truncated += 1
                continue
            # The sums count every attached point; a slot is stored only with a positive cloud number
            for name, src in sum_vars.items():
                value = src[a + t]
                if np.isfinite(value):
                    fragment_sums[name][kept_offsets[new_idx] + tt] += value
            if not frag_cn[t] > 0:
                n_unmatched += 1
                continue
            slots_used[(kind, new_idx, tt)] = slot + 1
            rows["track"].append(new_idx)
            rows["k"].append(tt)
            rows["kind"].append(kind)
            rows["slot"].append(slot)
            rows["cloudnumber"].append(frag_cn[t])
            rows["status"].append(frag_status[t])
            rows["area"].append(frag_area[t])

    dtypes = {"track": np.int32, "k": np.int32, "kind": np.int8, "slot": np.int16, "cloudnumber": np.int32, "status": np.int32,
              "area": np.float32}
    columns = {c: np.array(rows[c], dtype=dtypes[c]) for c in rows}
    return columns, fragment_sums, n_unmatched, n_truncated


def dense_width(longest, max_trackduration, round_base):
    """
    Length of the times dimension that the dense file has: the config value, or the longest track rounded up to a multiple of
    round_base if that is longer (duration_range_auto_update of the dense step).

    Args:
        longest: int
            Longest track in the file.
        max_trackduration: int
            Largest value of duration_range.
        round_base: int
            Round the expanded width up to a multiple of this.

    Returns:
        width: int
    """
    if longest <= max_trackduration:
        return int(max_trackduration)
    round_base = max(int(round_base), 1)
    return int(np.ceil(longest / round_base) * round_base)


def identifymcs_radar_rows(config):
    """
    Identify MCS from radar-based track statistics and classify every surviving track, folding the merge/split fragments into
    the track they belong to: identifymcs_radar.identifymcs_radar with row tables as the output.

    Args:
        config: dictionary
            Config parameters (those of identifymcs_radar; duration_range only sets the width of the dense layout that
            rows_to_dense rebuilds, a longer track never stops the step).

    Returns:
        statistics_outfolder: string
            Folder of the row tables, f"{stats_path}{mcsradarstats_filebase}{startdate}_{enddate}.rows" (tracks.parquet,
            entries.parquet, fragments.parquet, meta.json), the input of the rows-mode regionprops.
    """
    logger = logging.getLogger(__name__)
    logger.info("Identifying MCS from radar-based track statistics (row tables)")

    mcsradarstats_filebase = config.get("mcsradarstats_filebase", "mcs_tracks_")
    stats_path = config["stats_outpath"]
    startdate, enddate = config["startdate"], config["enddate"]
    time_resolution = config["datatimeresolution"]
    tracks_dimname, times_dimname = config["tracks_dimname"], config["times_dimname"]
    nmaxmerge = config["nmaxlinks"]
    fillval, fillval_f = config["fillval"], np.nan
    max_trackduration = int(max(config["duration_range"]))
    round_base = config.get("duration_range_round_base", 10)
    max_speed_kmh = config.get("max_speed_for_trajectory_smoothing", 50) * 3.6
    finalstats_min_area = config.get("finalstats_min_area", 0)
    finalstats_min_duration = config.get("finalstats_min_duration", None)
    params = mcs_params_from_config(config)
    logger.info(mcs_size_test_text(config, params))
    mcs_dbz45_area_varname = params["mcs_dbz45_area_varname"]
    mcs_merge_duration = config.get("mcs_merge_duration", 1.0)
    mcs_split_duration = config.get("mcs_split_duration", 1.0)
    statistics_file = f"{stats_path}{config['trackstats_sparse_filebase']}{startdate}_{enddate}.nc"
    statistics_outfolder = f"{stats_path}{mcsradarstats_filebase}{startdate}_{enddate}.rows"

    #########################################################################
    # Read the sparse trackstats file as flat entries
    #########################################################################
    ds_1d, flat, sparse_attrs, offsets, longest = load_sparse_entries(statistics_file, tracks_dimname, times_dimname)
    width = dense_width(longest, max_trackduration, round_base)
    ntracks_all = ds_1d.sizes[tracks_dimname]
    logger.info(f"Number of tracks to process: {ntracks_all}")
    if ntracks_all == 0:
        logger.warning("No tracks found in trackstats file.")
    if mcs_dbz45_area_varname not in flat:
        logger.critical(f"ERROR: mcs_dbz45_area_varname '{mcs_dbz45_area_varname}' not found in trackstats file. "
                        f"Available: {list(flat.keys())}")
        logger.critical("Tracking will now exit.")
        sys.exit()

    track_duration = ds_1d["track_duration"].values
    end_merge_tracknumber = ds_1d["end_merge_tracknumber"].values.astype(float)
    start_split_tracknumber = ds_1d["start_split_tracknumber"].values.astype(float)
    track_lifetime_hr = track_duration * time_resolution
    size_varname = mcs_size_varname(params)
    if size_varname not in flat:
        logger.critical(f"ERROR: mcs_size_method '{params['mcs_size_method']}' needs the variable '{size_varname}', which is "
                        f"not in the trackstats file. Rerun run_trackstats, or set mcs_size_method: majoraxis. "
                        f"Available: {list(flat.keys())}")
        logger.critical("Tracking will now exit.")
        sys.exit()
    size, dbz45, basetime, ncells = flat[size_varname], flat[mcs_dbz45_area_varname], flat["base_time"], flat["ncells"]
    conv_area, strat_area, area = flat["conv_area"], flat["strat_area"], flat["area"]
    cloudnumber, track_status = flat["cloudnumber"], flat["track_status"]

    #########################################################################
    # MCS test and feature classification, per track
    #########################################################################
    nentries = int(offsets[-1])
    is_mcs = np.zeros(ntracks_all, dtype=bool)
    mcs_status_flat = np.zeros(nentries, dtype=np.int16)
    feature_class_flat = np.full(nentries, fillval, dtype=np.int16)
    track_class = np.full(ntracks_all, fillval, dtype=np.int16)
    for nt in range(ntracks_all):
        ilength = int(track_duration[nt])
        if ilength <= 0:
            continue
        sl = slice(offsets[nt], offsets[nt + 1])
        mcs_status, track_is_mcs = identify_mcs_periods_by_method(size[sl], dbz45[sl], basetime[sl], time_resolution, params)
        mcs_status_flat[sl] = mcs_status
        is_mcs[nt] = track_is_mcs
        feature_class = classify_feature_class(mcs_status, ncells[sl].astype(int), fillval)
        feature_class_flat[sl] = feature_class
        track_class[nt] = classify_track_class(feature_class, fillval)
    logger.info(f"Number of radar-defined MCS: {int(np.sum(is_mcs))}")
    if not is_mcs.any():
        logger.warning("No MCS identified (per mcs_size_method/mcs_size_thresh/mcs_dbz45_area_thresh/mcs_duration_thresh). "
                       "Continuing - all non-fragment tracks are still kept and classified.")

    #########################################################################
    # Merge/split fragment identification and folding
    #########################################################################
    parent_idx, link_type, is_fragment = find_mergesplit_parents(
        track_lifetime_hr, end_merge_tracknumber, start_split_tracknumber, is_protected=is_mcs,
        merge_duration_thresh=mcs_merge_duration, split_duration_thresh=mcs_split_duration)
    parent_idx, link_type, root_idx, is_fragment_resolved, n_released = release_noncoexisting_fragments_flat(
        parent_idx, link_type, is_fragment, basetime, offsets)
    if n_released > 0:
        logger.info(f"{n_released} merge/split candidate(s) kept as standalone tracks: "
                    f"their time points do not all coexist with their resolved root track.")
    keep_track_indices = np.where(~is_fragment_resolved)[0]
    logger.info(f"Number of tracks kept (dropping {ntracks_all - len(keep_track_indices)} merge/split fragments): "
                f"{len(keep_track_indices)}")
    columns, fragment_sums, n_unmatched, n_truncated = attach_fragments_flat(
        root_idx, link_type, basetime, cloudnumber, track_status, area, offsets, keep_track_indices, nmaxmerge,
        sum_vars={"area": area, "conv_area": conv_area, "strat_area": strat_area})
    if n_unmatched > 0:
        logger.warning(f"{n_unmatched} fragment time point(s) could not be matched to their resolved root track's basetime "
                       f"or have no positive cloud number (dropped, not fatal).")
    if n_truncated > 0:
        logger.warning(f"{n_truncated} fragment attachment(s) dropped: more than nmaxmerge ({nmaxmerge}) fragments attached "
                       f"to one track at one time. Increase nmaxlinks to avoid this.")
    logger.info("Merge/split fragments attached to their parent tracks")

    #########################################################################
    # Optional safety-valve filtering (default: keep everything)
    #########################################################################
    final_keep = np.ones(len(keep_track_indices), dtype=bool)
    if finalstats_min_area:
        # Largest area of each track over its entries, NaN skipped (the dense step takes the maximum of the whole row, whose
        # zeros behind the last entry only matter for a track whose areas are all NaN or negative: it fails the test either way)
        area_index, area_offsets = gather_index(offsets, keep_track_indices)
        has_entries = np.diff(area_offsets) > 0
        max_area_kept = np.zeros(len(keep_track_indices))
        if has_entries.any():
            max_area_kept[has_entries] = np.fmax.reduceat(area[area_index], area_offsets[:-1][has_entries])
        final_keep &= (max_area_kept >= finalstats_min_area)
    if finalstats_min_duration:
        final_keep &= (track_lifetime_hr[keep_track_indices] >= finalstats_min_duration)
    if not np.all(final_keep):
        logger.info(f"finalstats_min_area/min_duration filter: dropping {len(final_keep) - int(np.sum(final_keep))} of "
                    f"{len(final_keep)} kept tracks.")
        new_of_kept = np.cumsum(final_keep) - 1                # index among the tracks that stay
        kept_offsets_all = np.concatenate(([0], np.cumsum(offsets[keep_track_indices + 1] - offsets[keep_track_indices])))
        entry_keep = np.repeat(final_keep, np.diff(kept_offsets_all))
        in_rows = final_keep[columns["track"]]
        columns = {c: v[in_rows] for c, v in columns.items()}
        columns["track"] = new_of_kept[columns["track"]].astype(np.int32)
        fragment_sums = {name: value[entry_keep] for name, value in fragment_sums.items()}
        keep_track_indices = keep_track_indices[final_keep]

    #########################################################################
    # Assemble the row tables of the kept tracks
    #########################################################################
    ntracks = len(keep_track_indices)
    index, kept_offsets = gather_index(offsets, keep_track_indices)
    kept_duration = track_duration[keep_track_indices]
    global_attrs = dict(ds_1d.attrs)
    global_attrs.update({
        "Title": "Final track statistics with radar-based MCS identification and classification",
        "mcs_size_method": params['mcs_size_method'], "mcs_size_thresh_km": params['mcs_size_thresh'],
        "mcs_dbz45_area_thresh_km2": params['mcs_dbz45_area_thresh'], "mcs_dbz45_area_varname": params['mcs_dbz45_area_varname'],
        "mcs_duration_thresh_hour": params['mcs_duration_thresh'], "Created_on": time.ctime(time.time()),
    })
    if params['mcs_size_method'] == "majoraxis":
        global_attrs["mcs_majoraxis_thresh_km"] = params['mcs_majoraxis_thresh']
    builder = TableBuilder(kept_duration, width, nslots=nmaxmerge, tracks_dim=tracks_dimname, times_dim=times_dimname,
                           global_attrs=global_attrs)

    # 1-D variables of the trackstats file (the merge / split links are used up and dropped)
    drop = {"start_split_tracknumber", "start_split_timeindex", "end_merge_tracknumber", "end_merge_timeindex", tracks_dimname}
    for name, var in ds_1d.variables.items():
        if name in drop:
            continue
        if var.dims == (tracks_dimname,):
            builder.add_track_var(name, var.values[keep_track_indices], var.attrs)
        elif len(var.dims) == 2 and var.dims[0] == tracks_dimname:
            builder.add_track2d_var(name, var.values[keep_track_indices], var.dims[1], var.attrs)
        elif tracks_dimname not in var.dims:
            builder.add_coord(name, var.values, var.attrs, var.dims)
        else:
            raise ValueError(f"variable {name} with dimensions {var.dims} cannot be stored as rows")

    # [tracks, times] variables of the trackstats file, no data (base_time == 0) is the fill value
    no_data = basetime[index] == 0
    kept = {}
    for name, values in flat.items():
        column = values[index]
        is_float = np.issubdtype(column.dtype, np.floating)
        fill = fillval_f if is_float else fillval
        column[no_data] = fill
        kept[name] = column
        builder.add_entry_var(name, column, sparse_attrs[name], fill=fill)

    # Smoothed trajectory of every kept track, on its valid entries (at least three: a shorter trajectory takes other branches)
    lon_smooth = np.full(len(index), fillval_f, dtype=np.float32)
    lat_smooth = np.full(len(index), fillval_f, dtype=np.float32)
    for i in range(ntracks):
        a, b = kept_offsets[i], kept_offsets[i + 1]
        n = max(int(b - a), 3)
        lon_in, lat_in = np.full(n, np.nan), np.full(n, np.nan)
        lon_in[:b - a], lat_in[:b - a] = kept["meanlon"][a:b], kept["meanlat"][a:b]
        lon_s, lat_s = smooth_trajectory(lon_in, lat_in, max_speed_kmh=max_speed_kmh, time_step_h=time_resolution)
        lon_smooth[a:b], lat_smooth[a:b] = lon_s[:b - a], lat_s[:b - a]
    builder.add_entry_var("meanlon_smooth", lon_smooth, {
        "long_name": "Smoothed longitude of a feature", "units": "degrees_east", "_FillValue": fillval_f}, fill=fillval_f)
    builder.add_entry_var("meanlat_smooth", lat_smooth, {
        "long_name": "Smoothed latitude of a feature", "units": "degrees_north", "_FillValue": fillval_f}, fill=fillval_f)

    # Area, convective area and stratiform area of each track including the folded fragments
    totals = {}
    for name in ("area", "conv_area", "strat_area"):
        own = np.nan_to_num(kept[name].astype(np.float64), nan=0.0)
        total = (own + fragment_sums[name]).astype(np.float32)
        total[no_data] = fillval_f
        totals[f"total_{name}"] = total

    # Classification of the kept tracks
    builder.add_track_var("is_mcs", is_mcs[keep_track_indices].astype(np.int16),
                          {"long_name": ATTRS_1D["is_mcs"], "units": "unitless", "_FillValue": fillval})
    builder.add_track_var("track_class", track_class[keep_track_indices],
                          {"long_name": ATTRS_1D["track_class"], "units": "unitless", "_FillValue": fillval})
    builder.add_track_var("mcs_number", number_mcs_tracks(is_mcs[keep_track_indices]),
                          {"long_name": MCS_NUMBER_LONG_NAME, "units": "unitless", "_FillValue": 0})
    builder.add_entry_var("feature_class", feature_class_flat[index], {
        "long_name": "Per-time feature class: 1=MCS, 2=multi-cell, 3=single-cell, 4=stratiform", "units": "unitless",
        "_FillValue": fillval}, fill=fillval)
    # mcs_status is 0 (not the _FillValue) where a track has no entry, as in the dense file
    builder.add_entry_var("mcs_status", mcs_status_flat[index], {
        "long_name": "Flag indicating the status of MCS based on radar criteria. 1=Yes, 0=No", "units": "unitless",
        "_FillValue": fillval}, fill=0)

    # Merge / split fragments
    attrs_cn = {"merge": "Cloud numbers that merge into this track", "split": "Cloud numbers that split from this track"}
    attrs_st = {"merge": "Track status of the merging cloud", "split": "Track status of the splitting cloud"}
    attrs_ar = {"merge": "Area of the merging cloud", "split": "Area of the splitting cloud"}
    variables = {}
    for kind in FRAGMENT_KINDS:
        variables[f"{kind}_cloudnumber"] = ({"long_name": attrs_cn[kind], "units": "unitless", "_FillValue": fillval}, fillval)
        variables[f"{kind}_status"] = ({"long_name": attrs_st[kind], "units": "unitless", "_FillValue": fillval}, fillval)
        variables[f"{kind}_area"] = ({"long_name": attrs_ar[kind], "units": "km^2", "_FillValue": fillval_f}, fillval_f)
    builder.add_fragment_vars(columns, variables)

    # Totals
    totals_attrs = {
        "total_area": {"long_name": "Total area of the track including the merge/split fragments folded into it",
                       "units": "km^2", "_FillValue": fillval_f,
                       "comments": "area + sum(merge_area over mergers) + sum(split_area over mergers)"},
        "total_conv_area": {"long_name": "Total convective area of the track including the merge/split fragments folded into it",
                            "units": "km^2", "_FillValue": fillval_f,
                            "comments": "conv_area of the track + conv_area of the fragments stored in the merge_*/split_* slots"},
        "total_strat_area": {"long_name": "Total stratiform area of the track including the merge/split fragments folded into it",
                             "units": "km^2", "_FillValue": fillval_f,
                             "comments": "strat_area of the track + strat_area of the fragments stored in the merge_*/split_* slots"},
    }
    for name, column in totals.items():
        builder.add_entry_var(name, column, totals_attrs[name], fill=fillval_f)
    tables = builder.build()

    #########################################################################
    # Write output
    #########################################################################
    logger.info(f"Writing {statistics_outfolder}")
    if os.path.isdir(statistics_outfolder):
        shutil.rmtree(statistics_outfolder)
    write_rows(tables, statistics_outfolder)
    logger.info(f"{statistics_outfolder}")
    return statistics_outfolder
