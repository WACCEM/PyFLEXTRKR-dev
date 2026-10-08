"""
Identify MCS from radar-based track statistics (feature_type='radar_feature').

Unlike identifymcs.py (Tb-based: keeps ONLY MCS-qualifying tracks, discards
everything else) this module follows link_mergesplit_tracks.py's philosophy:
keep EVERY tracked feature except short merge/split fragments, and attach a
4-way classification (MCS / multi-cell / isolated-cell / non-convective) to
every surviving track. The MCS definition itself is major axis + 45-dBZ area +
duration. This is the first pass: the shape and extreme statistics are those of
each track's own feature. regionprops_driver.py is the second pass that
recomputes them on the merged mask (own feature + folded merge/split fragments)
and stages the lifecycle (mcs_lifecycle.py).

This module has no dask import and no I/O side effects in its helper
functions, so the core logic is directly unit-testable with synthetic data
(see tests/test_identifymcs_radar_synthetic.py).
"""
import os
import sys
import time
import logging
import numpy as np
import xarray as xr

from pyflextrkr.ft_utilities import load_sparse_trackstats
from pyflextrkr.smooth_trajectory import smooth_trajectory
from pyflextrkr.mcs_lifecycle import _steps_from_hours

# Feature class codes
CLASS_MCS = 1
CLASS_MULTICELL = 2
CLASS_ISOLATED_CELL = 3
CLASS_NON_CONVECTIVE = 4


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------

def group_contiguous_indices(idx, max_gap_steps):
    """
    Group sorted integer indices into contiguous runs, bridging up to
    max_gap_steps missing intermediate indices.

    Args:
        idx: 1D np.array(int), sorted ascending.
        max_gap_steps: int
            Maximum number of missing intermediate indices still bridged
            into the same group (0 = indices must be perfectly consecutive).

    Returns:
        list of 1D np.array(int): the groups, in order, each non-empty.
    """
    if len(idx) == 0:
        return []
    breaks = np.where(np.diff(idx) > max_gap_steps + 1)[0] + 1
    return [g for g in np.split(idx, breaks) if len(g) > 0]


# ---------------------------------------------------------------------------
# MCS criteria test
# ---------------------------------------------------------------------------

def identify_mcs_periods(majoraxis, dbz45_area, basetime, time_resolution, params):
    """
    Per-track MCS test: majoraxis >= mcs_majoraxis_thresh and
    dbz45_area > mcs_dbz45_area_thresh, sustained for at least
    mcs_duration_thresh, bridging gaps up to mcs_gap_hour.

    Args:
        majoraxis, dbz45_area: 1D np.array, this track's own time series
            (already trimmed to its own track_duration; NaN treated as not
            meeting the criteria).
        basetime: 1D np.array, Epoch seconds, same length.
        time_resolution: float [hour]
        params: dict with mcs_majoraxis_thresh, mcs_dbz45_area_thresh,
            mcs_duration_thresh, mcs_gap_hour.

    Returns:
        mcs_status: 1D np.array(int16), 1 where this time is part of a
            qualifying MCS period, 0 otherwise.
        is_mcs: bool, True if any qualifying period's duration meets
            mcs_duration_thresh.
    """
    n = len(majoraxis)
    mcs_status = np.zeros(n, dtype=np.int16)
    if n == 0:
        return mcs_status, False

    meets_criteria = (
        np.nan_to_num(majoraxis, nan=-1.0) >= params['mcs_majoraxis_thresh']
    ) & (
        np.nan_to_num(dbz45_area, nan=-1.0) > params['mcs_dbz45_area_thresh']
    )
    idx = np.where(meets_criteria)[0]
    gap_steps = _steps_from_hours(params['mcs_gap_hour'], time_resolution)

    is_mcs = False
    for group in group_contiguous_indices(idx, gap_steps):
        duration = (basetime[group[-1]] - basetime[group[0]]) / 3600.0 + time_resolution
        if duration >= params['mcs_duration_thresh']:
            mcs_status[group] = 1
            is_mcs = True
    return mcs_status, is_mcs


# ---------------------------------------------------------------------------
# Merge/split fragment identification and folding
# ---------------------------------------------------------------------------

def find_mergesplit_parents(
        track_lifetime_hr, end_merge_tracknumber, start_split_tracknumber,
        is_protected, merge_duration_thresh, split_duration_thresh,
):
    """
    Determine, for each track, whether it is a droppable "fragment" (a short
    track that ends by merging into, or starts by splitting from, another
    track, and is not protected), and if so its immediate parent.

    A merge link takes precedence over a split link when both are present
    on the same short track, so a fragment only ever attaches - and paints
    pixels into - one of merge_cloudnumber/split_cloudnumber, never both.

    Args:
        track_lifetime_hr: 1D np.array, lifetime [hour] of each track.
        end_merge_tracknumber: 1D np.array, 1-based track number this track
            ends by merging into (0/fillval = none).
        start_split_tracknumber: 1D np.array, 1-based track number this
            track starts by splitting from (0/fillval = none).
        is_protected: 1D bool np.array, tracks that are never fragments
            regardless of duration/merge/split (e.g. is_mcs tracks).
        merge_duration_thresh, split_duration_thresh: float [hour]
            A track is a candidate fragment only if its whole lifetime is
            strictly less than the threshold (a longer track is never
            partially attached). Whether a candidate is finally folded into its
            root is decided by release_noncoexisting_fragments().

    Returns:
        parent_idx: 1D np.array(int), -1 if not a fragment, else the 0-based
            index of its immediate (direct) parent track.
        link_type: 1D np.array(str), '' / 'merge' / 'split' per track.
        is_fragment: 1D bool np.array.
    """
    n = len(track_lifetime_hr)
    parent_idx = np.full(n, -1, dtype=np.int64)
    link_type = np.full(n, '', dtype='<U6')

    end_merge_tracknumber = np.where(np.isnan(end_merge_tracknumber), 0, end_merge_tracknumber)
    start_split_tracknumber = np.where(np.isnan(start_split_tracknumber), 0, start_split_tracknumber)

    is_merge_fragment = (
        (~is_protected)
        & (track_lifetime_hr < merge_duration_thresh)
        & (end_merge_tracknumber > 0)
    )
    is_split_fragment = (
        (~is_protected)
        & (track_lifetime_hr < split_duration_thresh)
        & (start_split_tracknumber > 0)
        & (~is_merge_fragment)
    )
    parent_idx[is_merge_fragment] = end_merge_tracknumber[is_merge_fragment].astype(np.int64) - 1
    link_type[is_merge_fragment] = 'merge'
    parent_idx[is_split_fragment] = start_split_tracknumber[is_split_fragment].astype(np.int64) - 1
    link_type[is_split_fragment] = 'split'

    is_fragment = is_merge_fragment | is_split_fragment
    return parent_idx, link_type, is_fragment


def resolve_fragment_roots(parent_idx, is_fragment):
    """
    Resolve fragment parent chains (A -> B -> C) to their ultimate anchor
    (non-fragment) track.

    Cycles (a fragment chain that loops back on itself without ever
    reaching a non-fragment anchor - theoretically possible but expected to
    be extremely rare in practice) are broken deterministically: the first
    track revisited while walking a chain is forced to become its own
    anchor (un-fragmented), guaranteeing every fragment chain always
    resolves to a genuinely kept track, never a dangling/undefined root.

    Args:
        parent_idx: 1D np.array(int), from find_mergesplit_parents.
        is_fragment: 1D bool np.array, from find_mergesplit_parents.

    Returns:
        root_idx: 1D np.array(int); root_idx[i] == i for a non-fragment (or
            cycle-broken anchor) track, else the resolved anchor index.
        is_fragment_resolved: 1D bool np.array; a copy of is_fragment, with
            any cycle-breaking anchor tracks forced to False.
    """
    n = len(parent_idx)
    root_idx = np.arange(n)
    is_fragment_resolved = is_fragment.copy()

    for i in range(n):
        if not is_fragment[i]:
            continue
        visited = []
        cur = i
        while is_fragment_resolved[cur] and parent_idx[cur] != -1:
            if cur in visited:
                # Cycle detected: force this track to be the anchor.
                is_fragment_resolved[cur] = False
                break
            visited.append(cur)
            cur = parent_idx[cur]
        root_idx[i] = cur

    return root_idx, is_fragment_resolved


def release_noncoexisting_fragments(
        parent_idx, link_type, is_fragment, basetime_full,
):
    """
    Keep a candidate fragment only if all of its time points can be painted
    into its resolved root track; otherwise release it so it stays a normal
    (standalone) track.

    attach_fragments() paints a fragment's pixels into its root only at times
    where the root has the same basetime. A fragment whose chain resolves to a
    root that does not yet exist (or no longer exists) at some of the fragment's
    times would lose those time points - and, because the fragment track itself
    is dropped from the output, those pixels would end up in no track at all.
    Releasing such a fragment keeps every feature in some track.

    Releasing a track changes the root of any chain that passed through it, so
    the check is repeated until no more fragments are released. Each pass can
    only shrink the fragment set, so the loop terminates.

    Args:
        parent_idx, link_type, is_fragment: 1D arrays, from
            find_mergesplit_parents.
        basetime_full: 2D np.array [n_tracks, max_trackduration]; 0 or NaN =
            no data (same convention as attach_fragments).

    Returns:
        parent_idx: 1D np.array(int), -1 for every track that is not a fragment.
        link_type: 1D np.array(str), '' for every track that is not a fragment.
        root_idx: 1D np.array(int), from resolve_fragment_roots (final).
        is_fragment_resolved: 1D bool np.array, final fragment flags.
        n_released: int, number of candidate fragments kept as standalone tracks.
    """
    parent_idx = parent_idx.copy()
    link_type = link_type.copy()
    is_fragment = is_fragment.copy()

    valid_cache = {}

    def _valid_times(i):
        if i not in valid_cache:
            row = basetime_full[i, :]
            valid_cache[i] = row[(row != 0) & (~np.isnan(row))]
        return valid_cache[i]

    n_released = 0
    while True:
        root_idx, is_fragment_resolved = resolve_fragment_roots(parent_idx, is_fragment)
        released = [
            i for i in np.where(is_fragment_resolved)[0]
            if not np.all(np.isin(_valid_times(i), _valid_times(int(root_idx[i]))))
        ]
        if len(released) == 0:
            break
        is_fragment[released] = False
        parent_idx[released] = -1
        link_type[released] = ''
        n_released += len(released)

    # Only resolved fragments keep a link (also covers a cycle-breaking anchor,
    # which resolve_fragment_roots forces to be a kept track)
    parent_idx[~is_fragment_resolved] = -1
    link_type[~is_fragment_resolved] = ''
    return parent_idx, link_type, root_idx, is_fragment_resolved, n_released


def attach_fragments(
        root_idx, link_type, basetime_full, cloudnumber_full,
        track_status_full, area_full, keep_track_indices,
        nmaxmerge, fillval, fillval_f, sum_vars=None,
):
    """
    Build merge_cloudnumber/split_cloudnumber (+status/area) dense arrays
    for every kept track, attaching each fragment's cloudnumber at its
    matching basetime to its resolved root track.

    Args:
        root_idx, link_type: 1D arrays (full track-index space), from
            resolve_fragment_roots / find_mergesplit_parents.
        basetime_full, cloudnumber_full, track_status_full, area_full:
            2D dense arrays [n_tracks_full, max_trackduration].
        keep_track_indices: 1D np.array(int), the 0-based indices (in the
            full track-index space) of tracks to keep, in their new order.
        nmaxmerge: int, maximum number of fragments per track per time.
        fillval, fillval_f: fill values.
        sum_vars: dict {name: 2D np.array [n_tracks_full, max_trackduration]}, optional.
            Variables to add up over the fragments attached to each kept track
            (e.g. area, conv_area, strat_area).

    Returns:
        merge_cloudnumber, merge_status, merge_area,
        split_cloudnumber, split_status, split_area:
            3D dense arrays [n_kept, max_trackduration, nmaxmerge].
        n_unmatched: int
            Count of fragment time points that could not be matched to a
            time slot in their resolved root track (logged by the caller,
            not fatal - see feedback_guardrails_nan_not_crash).
        n_truncated: int
            Count of fragment attachments dropped because more than
            nmaxmerge fragments attached to one track at one time.
        fragment_sums: dict {name: 2D np.array(float64) [n_kept, max_trackduration]}
            For every sum_vars entry, the sum over the merge and split fragments attached to
            the track at that time (0 where none; NaN values are skipped). Only fragments that
            were stored in a merge/split slot count, so these sums stay consistent with
            merge_area/split_area and with the pixels mapfeature paints for the track.
            Empty if sum_vars is None.
    """
    n_kept = len(keep_track_indices)
    max_trackduration = basetime_full.shape[1]
    sum_vars = {} if sum_vars is None else sum_vars
    fragment_sums = {name: np.zeros((n_kept, max_trackduration), dtype=np.float64) for name in sum_vars}
    merge_cloudnumber = np.full((n_kept, max_trackduration, nmaxmerge), fillval, dtype=np.int32)
    merge_status = np.full((n_kept, max_trackduration, nmaxmerge), fillval, dtype=np.int32)
    merge_area = np.full((n_kept, max_trackduration, nmaxmerge), fillval_f, dtype=np.float32)
    split_cloudnumber = np.full((n_kept, max_trackduration, nmaxmerge), fillval, dtype=np.int32)
    split_status = np.full((n_kept, max_trackduration, nmaxmerge), fillval, dtype=np.int32)
    split_area = np.full((n_kept, max_trackduration, nmaxmerge), fillval_f, dtype=np.float32)

    new_index_of = {int(orig): new for new, orig in enumerate(keep_track_indices)}

    n_unmatched = 0
    n_truncated = 0
    for orig_idx in range(len(root_idx)):
        lt = link_type[orig_idx]
        if lt not in ('merge', 'split'):
            continue
        root = int(root_idx[orig_idx])
        if root not in new_index_of:
            # Should not happen (resolve_fragment_roots guarantees every
            # fragment resolves to a kept track) - guard defensively.
            n_unmatched += 1
            continue
        new_idx = new_index_of[root]

        frag_bt = basetime_full[orig_idx, :]
        frag_cn = cloudnumber_full[orig_idx, :]
        frag_status = track_status_full[orig_idx, :]
        frag_area = area_full[orig_idx, :]
        root_bt = basetime_full[root, :]

        valid_frag = (frag_bt != 0) & (~np.isnan(frag_bt))
        if lt == 'merge':
            dest, dstat, darea = merge_cloudnumber, merge_status, merge_area
        else:
            dest, dstat, darea = split_cloudnumber, split_status, split_area

        for t in np.where(valid_frag)[0]:
            match = np.where(root_bt == frag_bt[t])[0]
            if len(match) == 0:
                n_unmatched += 1
                continue
            tt = int(match[0])
            slot = int(np.count_nonzero(dest[new_idx, tt, :] > 0))
            if slot >= nmaxmerge:
                n_truncated += 1
                continue
            dest[new_idx, tt, slot] = frag_cn[t]
            dstat[new_idx, tt, slot] = frag_status[t]
            darea[new_idx, tt, slot] = frag_area[t]
            for name, src in sum_vars.items():
                value = src[orig_idx, t]
                if np.isfinite(value):
                    fragment_sums[name][new_idx, tt] += value

    return (merge_cloudnumber, merge_status, merge_area,
            split_cloudnumber, split_status, split_area,
            n_unmatched, n_truncated, fragment_sums)


# ---------------------------------------------------------------------------
# Feature classification (MCS / multi-cell / isolated-cell / non-convective)
# ---------------------------------------------------------------------------

def classify_feature_class(mcs_status, ncells, fillval=-9999):
    """
    Per-time categorical class: 1=MCS, 2=multi-cell (ncells>=2),
    3=isolated-cell (ncells==1), 4=non-convective (ncells==0). MCS status
    takes precedence over cell count (a feature meeting MCS criteria is
    labeled MCS even if it also has multiple cells).

    Args:
        mcs_status: 1D np.array(int), 1 where this time meets MCS criteria.
        ncells: 1D np.array(int), number of unique convective cells
            (fillval where not evaluated/no data).
        fillval: int

    Returns:
        feature_class: 1D np.array(int16), fillval where ncells is fillval.
    """
    n = len(ncells)
    feature_class = np.full(n, fillval, dtype=np.int16)
    valid = ncells != fillval
    feature_class[valid & (ncells == 0)] = CLASS_NON_CONVECTIVE
    feature_class[valid & (ncells == 1)] = CLASS_ISOLATED_CELL
    feature_class[valid & (ncells >= 2)] = CLASS_MULTICELL
    feature_class[valid & (mcs_status == 1)] = CLASS_MCS
    return feature_class


def classify_track_class(feature_class, fillval=-9999):
    """
    Per-track summary class: the most-organized class observed at any time
    (MCS > multi-cell > isolated-cell > non-convective; the class codes are
    numbered in this "more organized = smaller code" order, so the minimum
    valid code is the most-organized class observed).
    """
    valid = feature_class[feature_class != fillval]
    if len(valid) == 0:
        return fillval
    return int(np.min(valid))


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

def mcs_params_from_config(config):
    """
    MCS criteria and lifecycle staging parameters from the config, with their defaults (shared by
    identifymcs_radar and regionprops_driver).
    """
    return {
        'mcs_majoraxis_thresh': config.get("mcs_majoraxis_thresh", 100),
        'mcs_dbz45_area_thresh': config.get("mcs_dbz45_area_thresh", 0),
        'mcs_duration_thresh': config.get("mcs_duration_thresh", 4),
        'mcs_gap_hour': config.get("mcs_gap_hour", 0.25),
        'mcs_lifecycle_line_thresh': config.get("mcs_lifecycle_line_thresh", 100),
        'mcs_lifecycle_gap_hour': config.get("mcs_lifecycle_gap_hour", 1.0),
        'mcs_lifecycle_genesis_lag_hour': config.get("mcs_lifecycle_genesis_lag_hour", 1.0),
        'mcs_lifecycle_mature_lag_hour': config.get("mcs_lifecycle_mature_lag_hour", 2.0),
        'mcs_lifecycle_sf_area_min': config.get("mcs_lifecycle_sf_area_min", 0),
        'mcs_lifecycle_smooth_hour': config.get("mcs_lifecycle_smooth_hour", 0.5),
    }


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def identifymcs_radar(config):
    """
    Identify MCS from radar-based track statistics and classify every
    surviving (non-fragment) track as MCS / multi-cell / isolated-cell /
    non-convective, and fold the merge/split fragments into the track they belong to.
    First pass of the radar_feature MCS statistics; regionprops_driver.py is the second pass.

    Args:
        config: dictionary
            Dictionary containing config parameters.

    Returns:
        statistics_outfile: string
            Track statistics file name, same role and naming as the MCS Tb statistics of the
            Tb+PF pipelines (f"{stats_path}{mcsradarstats_filebase}{startdate}_{enddate}.nc",
            default mcsradarstats_filebase "mcs_tracks_"). It is the input of regionprops_driver,
            which writes the mcsrobust_filebase file used by mapfeature_driver and
            movement_speed. Besides the own-feature statistics it holds
            the folded merge/split fragments (merge_*/split_* variables) and total_area,
            total_conv_area, total_strat_area: the track's area, convective area and
            stratiform area including those fragments.
    """
    logger = logging.getLogger(__name__)
    logger.info("Identifying MCS from radar-based track statistics")

    mcsradarstats_filebase = config.get("mcsradarstats_filebase", "mcs_tracks_")
    trackstats_sparse_filebase = config["trackstats_sparse_filebase"]
    stats_path = config["stats_outpath"]
    startdate = config["startdate"]
    enddate = config["enddate"]
    time_resolution = config["datatimeresolution"]
    duration_range = config["duration_range"]
    tracks_dimname = config["tracks_dimname"]
    times_dimname = config["times_dimname"]
    nmaxmerge = config["nmaxlinks"]
    fillval = config["fillval"]
    fillval_f = np.nan
    max_trackduration = int(max(duration_range))
    duration_range_auto_update = config.get("duration_range_auto_update", False)
    duration_range_round_base = config.get("duration_range_round_base", 10)
    max_speed_ms = config.get("max_speed_for_trajectory_smoothing", 50)
    max_speed_kmh = max_speed_ms * 3.6

    mcs_dbz45_area_varname = config.get("mcs_dbz45_area_varname", "area_45dbz")
    finalstats_min_area = config.get("finalstats_min_area", 0)
    finalstats_min_duration = config.get("finalstats_min_duration", None)

    params = mcs_params_from_config(config)
    mcs_merge_duration = config.get("mcs_merge_duration", 1.0)
    mcs_split_duration = config.get("mcs_split_duration", 1.0)

    tracks_idx_varname = f"{tracks_dimname}_indices"
    times_idx_varname = f"{times_dimname}_indices"
    statistics_outfile = f"{stats_path}{mcsradarstats_filebase}{startdate}_{enddate}.nc"

    statistics_file = f"{stats_path}{trackstats_sparse_filebase}{startdate}_{enddate}.nc"
    logger.debug(statistics_file)

    # max_trackduration is returned from the file: when this step runs without run_trackstats in the
    # same run, the config's duration_range has not been expanded by trackstats_driver
    # (duration_range_auto_update), so it can be smaller than the longest track in the file.
    ds_1d, sparse_attrs_dict, sparse_dict, max_trackduration = load_sparse_trackstats(
        max_trackduration, statistics_file, times_idx_varname, tracks_dimname, tracks_idx_varname,
        duration_range_auto_update=duration_range_auto_update,
        duration_range_round_base=duration_range_round_base,
    )

    ntracks_all = ds_1d.sizes[tracks_dimname]
    logger.info(f"Number of tracks to process: {ntracks_all}")
    if ntracks_all == 0:
        logger.warning("No tracks found in trackstats file.")

    if mcs_dbz45_area_varname not in sparse_dict:
        logger.critical(
            f"ERROR: mcs_dbz45_area_varname '{mcs_dbz45_area_varname}' not found in "
            f"trackstats file. Available: {list(sparse_dict.keys())}"
        )
        logger.critical("Tracking will now exit.")
        sys.exit()

    track_duration = ds_1d["track_duration"].values
    end_merge_tracknumber = ds_1d["end_merge_tracknumber"].values.astype(float)
    start_split_tracknumber = ds_1d["start_split_tracknumber"].values.astype(float)
    track_lifetime_hr = track_duration * time_resolution

    majoraxis_2d = sparse_dict["majoraxis"].toarray()
    dbz45_2d = sparse_dict[mcs_dbz45_area_varname].toarray()
    basetime_2d = sparse_dict["base_time"].toarray()
    ncells_2d = sparse_dict["ncells"].toarray()
    conv_area_2d = sparse_dict["conv_area"].toarray()
    strat_area_2d = sparse_dict["strat_area"].toarray()
    cloudnumber_2d = sparse_dict["cloudnumber"].toarray()
    track_status_2d = sparse_dict["track_status"].toarray()
    area_2d = sparse_dict["area"].toarray()

    #########################################################################
    # MCS test and feature classification, per track
    #########################################################################
    is_mcs = np.zeros(ntracks_all, dtype=bool)
    mcs_status_2d = np.zeros((ntracks_all, max_trackduration), dtype=np.int16)
    feature_class_2d = np.full((ntracks_all, max_trackduration), fillval, dtype=np.int16)
    track_class = np.full(ntracks_all, fillval, dtype=np.int16)

    for nt in range(ntracks_all):
        ilength = int(track_duration[nt])
        if ilength <= 0:
            continue

        t_majoraxis = majoraxis_2d[nt, 0:ilength]
        t_dbz45 = dbz45_2d[nt, 0:ilength]
        t_basetime = basetime_2d[nt, 0:ilength]
        t_ncells = ncells_2d[nt, 0:ilength]

        mcs_status, track_is_mcs = identify_mcs_periods(
            t_majoraxis, t_dbz45, t_basetime, time_resolution, params,
        )
        mcs_status_2d[nt, 0:ilength] = mcs_status
        is_mcs[nt] = track_is_mcs

        feature_class = classify_feature_class(mcs_status, t_ncells.astype(int), fillval)
        feature_class_2d[nt, 0:ilength] = feature_class
        track_class[nt] = classify_track_class(feature_class, fillval)

    n_mcs = int(np.sum(is_mcs))
    logger.info(f"Number of radar-defined MCS: {n_mcs}")
    if n_mcs == 0:
        logger.warning(
            "No MCS identified (per mcs_majoraxis_thresh/mcs_dbz45_area_thresh/"
            "mcs_duration_thresh). Continuing - all non-fragment tracks are "
            "still kept and classified."
        )

    #########################################################################
    # Merge/split fragment identification and folding
    #########################################################################
    parent_idx, link_type, is_fragment = find_mergesplit_parents(
        track_lifetime_hr, end_merge_tracknumber, start_split_tracknumber,
        is_protected=is_mcs,
        merge_duration_thresh=mcs_merge_duration,
        split_duration_thresh=mcs_split_duration,
    )
    # Fragments that do not fully coexist with their root stay as standalone tracks
    parent_idx, link_type, root_idx, is_fragment_resolved, n_released = release_noncoexisting_fragments(
        parent_idx, link_type, is_fragment, basetime_2d,
    )
    if n_released > 0:
        logger.info(f"{n_released} merge/split candidate(s) kept as standalone tracks: "
                    f"their time points do not all coexist with their resolved root track.")

    keep_mask = ~is_fragment_resolved
    keep_track_indices = np.where(keep_mask)[0]
    ntracks = len(keep_track_indices)
    n_dropped = ntracks_all - ntracks
    logger.info(f"Number of tracks kept (dropping {n_dropped} merge/split fragments): {ntracks}")

    (merge_cloudnumber, merge_status, merge_area,
     split_cloudnumber, split_status, split_area,
     n_unmatched, n_truncated, fragment_sums) = attach_fragments(
        root_idx, link_type, basetime_2d, cloudnumber_2d, track_status_2d, area_2d,
        keep_track_indices, nmaxmerge, fillval, fillval_f,
        sum_vars={"area": area_2d, "conv_area": conv_area_2d, "strat_area": strat_area_2d},
    )
    if n_unmatched > 0:
        # Not expected: release_noncoexisting_fragments() only keeps fragments whose time points all match
        logger.warning(f"{n_unmatched} fragment time point(s) could not be matched to their "
                        f"resolved root track's basetime (dropped, not fatal).")
    if n_truncated > 0:
        logger.warning(f"{n_truncated} fragment attachment(s) dropped: more than nmaxmerge "
                        f"({nmaxmerge}) fragments attached to one track at one time. "
                        f"Increase nmaxlinks to avoid this.")

    #########################################################################
    # Optional safety-valve filtering (default: keep everything)
    #########################################################################
    final_keep = np.ones(ntracks, dtype=bool)
    if finalstats_min_area:
        max_area_kept = np.nanmax(area_2d[keep_track_indices, :], axis=1)
        final_keep &= (max_area_kept >= finalstats_min_area)
    if finalstats_min_duration:
        final_keep &= (track_lifetime_hr[keep_track_indices] >= finalstats_min_duration)
    if not np.all(final_keep):
        logger.info(
            f"finalstats_min_area/min_duration filter: dropping "
            f"{ntracks - int(np.sum(final_keep))} of {ntracks} kept tracks."
        )
        keep_track_indices = keep_track_indices[final_keep]
        merge_cloudnumber = merge_cloudnumber[final_keep]
        merge_status = merge_status[final_keep]
        merge_area = merge_area[final_keep]
        split_cloudnumber = split_cloudnumber[final_keep]
        split_status = split_status[final_keep]
        split_area = split_area[final_keep]
        fragment_sums = {name: value[final_keep] for name, value in fragment_sums.items()}
        ntracks = len(keep_track_indices)

    #########################################################################
    # Assemble output dataset
    #########################################################################
    for ivar in sparse_dict.keys():
        sparse_dict[ivar] = sparse_dict[ivar][keep_track_indices, :]
    sparse_dict.pop(tracks_idx_varname, None)
    sparse_dict.pop(times_idx_varname, None)

    mask = sparse_dict['base_time'].toarray() == 0
    varlist = {}
    for key, value in sparse_dict.items():
        dense_array = value.toarray()
        if isinstance(dense_array[0, 0], np.floating):
            dense_array[mask] = fillval_f
        else:
            dense_array[mask] = fillval
        varlist[key] = ([tracks_dimname, times_dimname], dense_array, sparse_attrs_dict[key])

    tracks_coord = np.arange(0, ntracks)
    times_coord = np.arange(0, max_trackduration)
    mergers_coord = np.arange(0, nmaxmerge)
    coordlist = {
        tracks_dimname: ([tracks_dimname], tracks_coord),
        times_dimname: ([times_dimname], times_coord),
    }
    ds_2d = xr.Dataset(varlist, coords=coordlist)

    ds_1d_kept = ds_1d.sel({tracks_dimname: keep_track_indices})
    ds_1d_kept[tracks_dimname] = tracks_coord
    dsout = xr.merge([ds_1d_kept, ds_2d], compat="override", combine_attrs="no_conflicts")
    drop_vars_list = [
        "start_split_tracknumber", "start_split_timeindex",
        "end_merge_tracknumber", "end_merge_timeindex",
    ]
    dsout = dsout.drop_vars(drop_vars_list, errors='ignore')

    # Smooth trajectory (reused from identifymcs.py's approach)
    lon_smooth = np.full((ntracks, max_trackduration), fillval_f, dtype=np.float32)
    lat_smooth = np.full((ntracks, max_trackduration), fillval_f, dtype=np.float32)
    for itrack in tracks_coord:
        lon_0 = dsout['meanlon'][itrack].values
        lat_0 = dsout['meanlat'][itrack].values
        lon_s, lat_s = smooth_trajectory(lon_0, lat_0, max_speed_kmh=max_speed_kmh, time_step_h=time_resolution)
        lon_smooth[itrack, :] = lon_s
        lat_smooth[itrack, :] = lat_s
    dsout['meanlon_smooth'] = xr.DataArray(
        lon_smooth, dims=[tracks_dimname, times_dimname],
        coords={tracks_dimname: tracks_coord, times_dimname: times_coord},
        attrs={"long_name": "Smoothed longitude of a feature", "units": "degrees_east",
               "_FillValue": fillval_f},
    )
    dsout['meanlat_smooth'] = xr.DataArray(
        lat_smooth, dims=[tracks_dimname, times_dimname],
        coords={tracks_dimname: tracks_coord, times_dimname: times_coord},
        attrs={"long_name": "Smoothed latitude of a feature", "units": "degrees_north",
               "_FillValue": fillval_f},
    )

    # Area, convective area and stratiform area of each track including the merge/split fragments
    # folded into it (fragments are folded into any non-fragment track, MCS or not); NaN where the
    # track has no data
    totals = {}
    for name in ("area", "conv_area", "strat_area"):
        own = np.nan_to_num(dsout[name].values.astype(np.float64), nan=0.0)
        total = (own + fragment_sums[name]).astype(np.float32)
        total[mask] = fillval_f
        totals[f"total_{name}"] = total

    # Merge/split cloudnumber variables
    mergers_dimname = "mergers"
    var_dict = {
        "is_mcs": is_mcs[keep_track_indices].astype(np.int16),
        "track_class": track_class[keep_track_indices],
        "feature_class": feature_class_2d[keep_track_indices, :],
        "mcs_status": mcs_status_2d[keep_track_indices, :],
        "merge_cloudnumber": merge_cloudnumber,
        "merge_status": merge_status,
        "merge_area": merge_area,
        "split_cloudnumber": split_cloudnumber,
        "split_status": split_status,
        "split_area": split_area,
        **totals,
    }
    var_attrs = {
        "is_mcs": {"long_name": "Flag indicating if this track meets the radar MCS criteria at any time",
                   "units": "unitless", "_FillValue": fillval},
        "track_class": {"long_name": "Most-organized feature class observed on this track: "
                         "1=MCS, 2=multi-cell, 3=isolated-cell, 4=non-convective",
                         "units": "unitless", "_FillValue": fillval},
        "feature_class": {"long_name": "Per-time feature class: 1=MCS, 2=multi-cell, "
                           "3=isolated-cell, 4=non-convective", "units": "unitless", "_FillValue": fillval},
        "mcs_status": {"long_name": "Flag indicating the status of MCS based on radar criteria. 1=Yes, 0=No",
                        "units": "unitless", "_FillValue": fillval},
        "merge_cloudnumber": {"long_name": "Cloud numbers that merge into this track",
                               "units": "unitless", "_FillValue": fillval},
        "merge_status": {"long_name": "Track status of the merging cloud", "units": "unitless", "_FillValue": fillval},
        "merge_area": {"long_name": "Area of the merging cloud", "units": "km^2", "_FillValue": fillval_f},
        "split_cloudnumber": {"long_name": "Cloud numbers that split from this track",
                               "units": "unitless", "_FillValue": fillval},
        "split_status": {"long_name": "Track status of the splitting cloud", "units": "unitless", "_FillValue": fillval},
        "split_area": {"long_name": "Area of the splitting cloud", "units": "km^2", "_FillValue": fillval_f},
        "total_area": {"long_name": "Total area of the track including the merge/split fragments folded into it",
                       "units": "km^2", "_FillValue": fillval_f,
                       "comments": "area + sum(merge_area over mergers) + sum(split_area over mergers)"},
        "total_conv_area": {"long_name": "Total convective area of the track including the merge/split fragments "
                                         "folded into it",
                            "units": "km^2", "_FillValue": fillval_f,
                            "comments": "conv_area of the track + conv_area of the fragments stored in the "
                                        "merge_*/split_* slots"},
        "total_strat_area": {"long_name": "Total stratiform area of the track including the merge/split fragments "
                                          "folded into it",
                             "units": "km^2", "_FillValue": fillval_f,
                             "comments": "strat_area of the track + strat_area of the fragments stored in the "
                                         "merge_*/split_* slots"},
    }
    varlist2 = {}
    for key, value in var_dict.items():
        if value.ndim == 1:
            varlist2[key] = ([tracks_dimname], value, var_attrs[key])
        elif value.ndim == 2:
            varlist2[key] = ([tracks_dimname, times_dimname], value, var_attrs[key])
        elif value.ndim == 3:
            varlist2[key] = ([tracks_dimname, times_dimname, mergers_dimname], value, var_attrs[key])
    coordlist2 = {
        tracks_dimname: ([tracks_dimname], tracks_coord),
        times_dimname: ([times_dimname], times_coord),
        mergers_dimname: ([mergers_dimname], mergers_coord),
    }
    ds_vars = xr.Dataset(varlist2, coords=coordlist2)
    dsout = xr.merge([dsout, ds_vars], compat="override", combine_attrs="no_conflicts")

    dsout.attrs["Title"] = "Final track statistics with radar-based MCS identification and classification"
    dsout.attrs["mcs_majoraxis_thresh_km"] = params['mcs_majoraxis_thresh']
    dsout.attrs["mcs_dbz45_area_thresh_km2"] = params['mcs_dbz45_area_thresh']
    dsout.attrs["mcs_duration_thresh_hour"] = params['mcs_duration_thresh']
    dsout.attrs["Created_on"] = time.ctime(time.time())

    #########################################################################
    # Write output
    #########################################################################
    if os.path.isfile(statistics_outfile):
        os.remove(statistics_outfile)
    comp = dict(zlib=True)
    encoding = {var: comp for var in dsout.data_vars}
    open(statistics_outfile, 'ab').close()
    dsout.to_netcdf(path=statistics_outfile, mode="w",
                     format="NETCDF4", unlimited_dims=tracks_dimname, encoding=encoding)
    logger.info(f"{statistics_outfile}")

    return statistics_outfile
