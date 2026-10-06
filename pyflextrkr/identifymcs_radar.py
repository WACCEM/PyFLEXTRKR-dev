"""
Identify MCS from radar-based track statistics (feature_type='radar_feature').

Unlike identifymcs.py (Tb-based: keeps ONLY MCS-qualifying tracks, discards
everything else) this module follows link_mergesplit_tracks.py's philosophy:
keep EVERY tracked feature except short merge/split fragments, and attach a
4-way classification (MCS / multi-cell / isolated-cell / non-convective) to
every surviving track. The MCS definition itself (major axis + 45-dBZ area +
duration) and the lifecycle-stage logic are a hardened port of
robustmcs_radar.py's Coniglio-et-al.-(2010)-based approach, generalized to
irregular (non-hourly) time spacing and multi-cycle (re-intensification)
tracks - see classify_lifecycle's docstring for the specific hardening.

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

# Stage codes
STAGE_CLOUD_ONLY = 1
STAGE_UNORGANIZED = 2
STAGE_GENESIS = 3
STAGE_MATURE = 4
STAGE_DECAY = 5

# Feature class codes
CLASS_MCS = 1
CLASS_MULTICELL = 2
CLASS_ISOLATED_CELL = 3
CLASS_NON_CONVECTIVE = 4


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------

def _steps_from_hours(hours, time_resolution):
    """Convert an hour duration to a (minimum 1) number of time steps."""
    return max(1, int(round(hours / time_resolution)))


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


def attach_fragments(
        root_idx, link_type, basetime_full, cloudnumber_full,
        track_status_full, area_full, keep_track_indices,
        nmaxmerge, fillval, fillval_f,
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
    """
    n_kept = len(keep_track_indices)
    max_trackduration = basetime_full.shape[1]
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

    return (merge_cloudnumber, merge_status, merge_area,
            split_cloudnumber, split_status, split_area,
            n_unmatched, n_truncated)


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
# Lifecycle staging (hardened port of robustmcs_radar.py)
# ---------------------------------------------------------------------------

def classify_lifecycle(conv_area, conv_majoraxis, strat_area, time_resolution, params):
    """
    Hardened, multi-cycle-aware port of robustmcs_radar.py's Coniglio-et-al.
    -2010 lifecycle staging, for ONE track's own time series (already
    trimmed to its own track_duration).

    Stages: 1=cloud only, 2=unorganized convection, 3=genesis (line
    organizing, not yet mature), 4=mature (organized line + developed
    stratiform area), 5=decay.

    Hardening vs. robustmcs_radar.py:
    - Every contiguous "organized convective line" episode
      (conv_majoraxis >= mcs_lifecycle_line_thresh, gaps up to
      mcs_lifecycle_gap_hour bridged) is staged independently, in time
      order, without overwriting earlier stage assignments - this replaces
      robustmcs_radar.py's approach of keeping only the single largest
      contiguous group, which silently drops re-intensification episodes
      (the "odd multi-cycle tracks" the lifecycle logic needed to be made
      robust against).
    - All constants are expressed in hours and converted to time steps via
      time_resolution, instead of robustmcs_radar.py's hard-coded hourly-
      step constants (e.g. "+2"), which would silently misbehave at
      5-minute (or any non-hourly) time resolution.
    - "Stage not observed" is -1 in lifecycle_index, distinct from a track
      not being evaluated at all (handled by the caller).
    - This function is always called and always returns a result (never
      raises/exits for a short or poorly-organized track) - the caller
      decides whether a track is long enough to bother evaluating.

    Args:
        conv_area, conv_majoraxis, strat_area: 1D np.array, this track's own
            time series (length = track_duration). NaN treated as 0/absent.
        time_resolution: float [hour]
        params: dict with mcs_lifecycle_line_thresh, mcs_lifecycle_gap_hour,
            mcs_lifecycle_genesis_lag_hour, mcs_lifecycle_mature_lag_hour,
            mcs_lifecycle_sf_area_min.

    Returns:
        stage: 1D np.array(int16), length = track_duration.
        cycle: 1D np.array(int16), which organized-line episode (1-based) a
            time belongs to; 0 for cloud-only/unorganized-baseline times not
            part of any episode's genesis-through-decay span.
        lifecycle_index: 1D np.array(int64), length 5, the first time INDEX
            of stages 1..5 within this track's own trimmed array; -1 if that
            stage was never observed.
        ncycles: int, number of organized-line episodes found.
        complete: bool, True if stages 3, 4 and 5 are all observed.
    """
    n = len(conv_area)
    conv_area = np.nan_to_num(np.asarray(conv_area, dtype=float), nan=0.0)
    conv_majoraxis = np.nan_to_num(np.asarray(conv_majoraxis, dtype=float), nan=0.0)
    strat_area = np.nan_to_num(np.asarray(strat_area, dtype=float), nan=0.0)

    stage = np.full(n, STAGE_CLOUD_ONLY, dtype=np.int16)
    cycle = np.zeros(n, dtype=np.int16)
    if n == 0:
        return stage, cycle, np.full(5, -1, dtype=np.int64), 0, False

    C = conv_area > 0
    stage[C] = STAGE_UNORGANIZED

    Lf = conv_majoraxis >= params['mcs_lifecycle_line_thresh']
    sf_present = strat_area[strat_area > 0]
    sf_floor = max(
        float(np.mean(sf_present)) if sf_present.size > 0 else 0.0,
        params.get('mcs_lifecycle_sf_area_min', 0),
    )
    S = Lf & (strat_area > 0) & (strat_area >= sf_floor)

    gap_steps = _steps_from_hours(params['mcs_lifecycle_gap_hour'], time_resolution)
    genesis_lag_steps = _steps_from_hours(params['mcs_lifecycle_genesis_lag_hour'], time_resolution)
    mature_lag_steps = _steps_from_hours(params['mcs_lifecycle_mature_lag_hour'], time_resolution)

    line_idx = np.where(Lf)[0]
    episodes = group_contiguous_indices(line_idx, gap_steps)
    ncycles = len(episodes)

    for ic, episode in enumerate(episodes, start=1):
        a, b = int(episode[0]), int(episode[-1])
        g = a + genesis_lag_steps
        if g > b:
            # Episode too short for even genesis to complete (less than
            # mcs_lifecycle_genesis_lag_hour of organized line) - leave the
            # R0 baseline (cloud-only/unorganized) stage for this episode.
            continue

        mature_search_start = g + mature_lag_steps
        mature_candidates = np.where(S[mature_search_start:b + 1])[0]
        if mature_candidates.size > 0:
            s0 = mature_search_start + int(mature_candidates[0])
            s1 = g + int(np.where(S[g:b + 1])[0][-1])
            stage[g:s0] = STAGE_GENESIS
            stage[s0:s1 + 1] = STAGE_MATURE
            decay_start = s1 + 1
        else:
            stage[g:b + 1] = STAGE_GENESIS
            decay_start = b + 1

        decay_end = int(episodes[ic][0]) if ic < len(episodes) else n
        if decay_start < decay_end:
            stage[decay_start:decay_end] = STAGE_DECAY
        cycle[g:max(decay_end, g + 1)] = ic

    lifecycle_index = np.full(5, -1, dtype=np.int64)
    for s in range(1, 6):
        where_s = np.where(stage == s)[0]
        if where_s.size > 0:
            lifecycle_index[s - 1] = where_s[0]
    complete = bool(
        (lifecycle_index[2] != -1) and (lifecycle_index[3] != -1) and (lifecycle_index[4] != -1)
    )
    return stage, cycle, lifecycle_index, ncycles, complete


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def identifymcs_radar(config):
    """
    Identify MCS from radar-based track statistics and classify every
    surviving (non-fragment) track as MCS / multi-cell / isolated-cell /
    non-convective, with hardened lifecycle staging for MCS tracks.

    Args:
        config: dictionary
            Dictionary containing config parameters.

    Returns:
        statistics_outfile: string
            Final track statistics file name
            (f"{stats_path}{finalstats_filebase}{startdate}_{enddate}.nc").
    """
    logger = logging.getLogger(__name__)
    logger.info("Identifying MCS from radar-based track statistics")

    finalstats_filebase = config["finalstats_filebase"]
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
    max_speed_ms = config.get("max_speed_for_trajectory_smoothing", 50)
    max_speed_kmh = max_speed_ms * 3.6

    mcs_dbz45_area_varname = config.get("mcs_dbz45_area_varname", "area_45dbz")
    finalstats_min_area = config.get("finalstats_min_area", 0)
    finalstats_min_duration = config.get("finalstats_min_duration", None)

    params = {
        'mcs_majoraxis_thresh': config.get("mcs_majoraxis_thresh", 100),
        'mcs_dbz45_area_thresh': config.get("mcs_dbz45_area_thresh", 0),
        'mcs_duration_thresh': config.get("mcs_duration_thresh", 4),
        'mcs_gap_hour': config.get("mcs_gap_hour", 0.25),
        'mcs_lifecycle_line_thresh': config.get("mcs_lifecycle_line_thresh", 100),
        'mcs_lifecycle_gap_hour': config.get("mcs_lifecycle_gap_hour", 1.0),
        'mcs_lifecycle_genesis_lag_hour': config.get("mcs_lifecycle_genesis_lag_hour", 1.0),
        'mcs_lifecycle_mature_lag_hour': config.get("mcs_lifecycle_mature_lag_hour", 2.0),
        'mcs_lifecycle_sf_area_min': config.get("mcs_lifecycle_sf_area_min", 0),
    }
    mcs_lifecycle_thresh = config.get("mcs_lifecycle_thresh", 8)
    mcs_merge_duration = config.get("mcs_merge_duration", 1.0)
    mcs_split_duration = config.get("mcs_split_duration", 1.0)

    tracks_idx_varname = f"{tracks_dimname}_indices"
    times_idx_varname = f"{times_dimname}_indices"
    statistics_outfile = f"{stats_path}{finalstats_filebase}{startdate}_{enddate}.nc"

    statistics_file = f"{stats_path}{trackstats_sparse_filebase}{startdate}_{enddate}.nc"
    logger.debug(statistics_file)

    ds_1d, sparse_attrs_dict, sparse_dict, max_trackduration = load_sparse_trackstats(
        max_trackduration, statistics_file, times_idx_varname, tracks_dimname, tracks_idx_varname,
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
    conv_majoraxis_2d = sparse_dict["conv_majoraxis"].toarray()
    strat_area_2d = sparse_dict["strat_area"].toarray()
    cloudnumber_2d = sparse_dict["cloudnumber"].toarray()
    track_status_2d = sparse_dict["track_status"].toarray()
    area_2d = sparse_dict["area"].toarray()

    #########################################################################
    # MCS test and lifecycle staging, per track
    #########################################################################
    is_mcs = np.zeros(ntracks_all, dtype=bool)
    mcs_status_2d = np.zeros((ntracks_all, max_trackduration), dtype=np.int16)
    feature_class_2d = np.full((ntracks_all, max_trackduration), fillval, dtype=np.int16)
    track_class = np.full(ntracks_all, fillval, dtype=np.int16)
    lifecycle_stage_2d = np.full((ntracks_all, max_trackduration), fillval, dtype=np.int16)
    lifecycle_cycle_2d = np.full((ntracks_all, max_trackduration), fillval, dtype=np.int16)
    lifecycle_index_2d = np.full((ntracks_all, 5), fillval, dtype=np.int64)
    lifecycle_ncycles = np.full(ntracks_all, fillval, dtype=np.int32)
    lifecycle_complete_flag = np.full(ntracks_all, fillval, dtype=np.int16)
    lifecycle_evaluated = np.zeros(ntracks_all, dtype=np.int16)

    for nt in range(ntracks_all):
        ilength = int(track_duration[nt])
        if ilength <= 0:
            continue

        t_majoraxis = majoraxis_2d[nt, 0:ilength]
        t_dbz45 = dbz45_2d[nt, 0:ilength]
        t_basetime = basetime_2d[nt, 0:ilength]
        t_ncells = ncells_2d[nt, 0:ilength]
        t_conv_area = conv_area_2d[nt, 0:ilength]
        t_conv_majoraxis = conv_majoraxis_2d[nt, 0:ilength]
        t_strat_area = strat_area_2d[nt, 0:ilength]

        mcs_status, track_is_mcs = identify_mcs_periods(
            t_majoraxis, t_dbz45, t_basetime, time_resolution, params,
        )
        mcs_status_2d[nt, 0:ilength] = mcs_status
        is_mcs[nt] = track_is_mcs

        feature_class = classify_feature_class(mcs_status, t_ncells.astype(int), fillval)
        feature_class_2d[nt, 0:ilength] = feature_class
        track_class[nt] = classify_track_class(feature_class, fillval)

        if track_lifetime_hr[nt] >= mcs_lifecycle_thresh:
            stage, cyc, lc_idx, ncyc, complete = classify_lifecycle(
                t_conv_area, t_conv_majoraxis, t_strat_area, time_resolution, params,
            )
            lifecycle_stage_2d[nt, 0:ilength] = stage
            lifecycle_cycle_2d[nt, 0:ilength] = cyc
            lifecycle_index_2d[nt, :] = lc_idx
            lifecycle_ncycles[nt] = ncyc
            lifecycle_complete_flag[nt] = int(complete)
            lifecycle_evaluated[nt] = 1

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
    root_idx, is_fragment_resolved = resolve_fragment_roots(parent_idx, is_fragment)

    keep_mask = ~is_fragment_resolved
    keep_track_indices = np.where(keep_mask)[0]
    ntracks = len(keep_track_indices)
    n_dropped = ntracks_all - ntracks
    logger.info(f"Number of tracks kept (dropping {n_dropped} merge/split fragments): {ntracks}")

    (merge_cloudnumber, merge_status, merge_area,
     split_cloudnumber, split_status, split_area,
     n_unmatched, n_truncated) = attach_fragments(
        root_idx, link_type, basetime_2d, cloudnumber_2d, track_status_2d, area_2d,
        keep_track_indices, nmaxmerge, fillval, fillval_f,
    )
    if n_unmatched > 0:
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
    lifestage_coord = np.arange(0, 5)
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

    # Merge/split cloudnumber variables
    mergers_dimname = "mergers"
    var_dict = {
        "is_mcs": is_mcs[keep_track_indices].astype(np.int16),
        "track_class": track_class[keep_track_indices],
        "feature_class": feature_class_2d[keep_track_indices, :],
        "mcs_status": mcs_status_2d[keep_track_indices, :],
        "lifecycle_stage": lifecycle_stage_2d[keep_track_indices, :],
        "lifecycle_cycle": lifecycle_cycle_2d[keep_track_indices, :],
        "lifecycle_index": lifecycle_index_2d[keep_track_indices, :],
        "lifecycle_ncycles": lifecycle_ncycles[keep_track_indices],
        "lifecycle_complete_flag": lifecycle_complete_flag[keep_track_indices],
        "lifecycle_evaluated": lifecycle_evaluated[keep_track_indices],
        "merge_cloudnumber": merge_cloudnumber,
        "merge_status": merge_status,
        "merge_area": merge_area,
        "split_cloudnumber": split_cloudnumber,
        "split_status": split_status,
        "split_area": split_area,
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
                                 "track (track lifetime >= mcs_lifecycle_thresh)",
                                 "units": "unitless", "_FillValue": fillval},
        "merge_cloudnumber": {"long_name": "Cloud numbers that merge into this track",
                               "units": "unitless", "_FillValue": fillval},
        "merge_status": {"long_name": "Track status of the merging cloud", "units": "unitless", "_FillValue": fillval},
        "merge_area": {"long_name": "Area of the merging cloud", "units": "km^2", "_FillValue": fillval_f},
        "split_cloudnumber": {"long_name": "Cloud numbers that split from this track",
                               "units": "unitless", "_FillValue": fillval},
        "split_status": {"long_name": "Track status of the splitting cloud", "units": "unitless", "_FillValue": fillval},
        "split_area": {"long_name": "Area of the splitting cloud", "units": "km^2", "_FillValue": fillval_f},
    }
    varlist2 = {}
    for key, value in var_dict.items():
        if value.ndim == 1:
            varlist2[key] = ([tracks_dimname], value, var_attrs[key])
        elif value.ndim == 2 and key == "lifecycle_index":
            varlist2[key] = ([tracks_dimname, "lifestages"], value, var_attrs[key])
        elif value.ndim == 2:
            varlist2[key] = ([tracks_dimname, times_dimname], value, var_attrs[key])
        elif value.ndim == 3:
            varlist2[key] = ([tracks_dimname, times_dimname, mergers_dimname], value, var_attrs[key])
    coordlist2 = {
        tracks_dimname: ([tracks_dimname], tracks_coord),
        times_dimname: ([times_dimname], times_coord),
        mergers_dimname: ([mergers_dimname], mergers_coord),
        "lifestages": (["lifestages"], lifestage_coord),
    }
    ds_vars = xr.Dataset(varlist2, coords=coordlist2)
    dsout = xr.merge([dsout, ds_vars], compat="override", combine_attrs="no_conflicts")

    dsout.attrs["Title"] = "Final track statistics with radar-based MCS identification and classification"
    dsout.attrs["mcs_majoraxis_thresh_km"] = params['mcs_majoraxis_thresh']
    dsout.attrs["mcs_dbz45_area_thresh_km2"] = params['mcs_dbz45_area_thresh']
    dsout.attrs["mcs_duration_thresh_hour"] = params['mcs_duration_thresh']
    dsout.attrs["mcs_lifecycle_thresh_hour"] = mcs_lifecycle_thresh
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
