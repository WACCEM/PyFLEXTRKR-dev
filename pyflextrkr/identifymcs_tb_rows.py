"""
Rows-mode identifymcs for Tb tracking (config stats_format: rows).

The same MCS identification from the cold cloud shield (Tb) as identifymcs.identifymcs_tb, on flat arrays (one position for every valid
(track, time) entry of every track, track after track) instead of the [tracks, times] and [tracks, times, slots] arrays, writing the
row tables of pyflextrkr.rowstats (a Parquet folder) instead of the dense netCDF file. The dense step makes [MCS, longest track, nmaxlinks] arrays
for the cloud numbers, statuses and areas of the merging and splitting clouds (80 kB per MCS and array with 400 times and the 50 slots of the reference
config: 4 GB for 50,000 MCS, six such arrays in memory, four written to the file, and read again by every later step), and it looks for the clouds that
merge into or split from an MCS by a pass over all the tracks for every MCS. Here the memory follows the number of entries, and the merging and
splitting clouds of all the MCS are found in one pass (tests/benchmark_tbpf_rows_scale.py compares the two on a synthetic file of a million tracks).

What is kept as it is in the dense step: a track is an MCS if some period of consecutive times (gaps of at most mcs_tb_gap time
steps) has a cold cloud shield area (core + cold area) above mcs_tb_area_thresh for at least mcs_tb_duration_thresh hours and the track
has a cold core; the clouds that merge into / split from an MCS are the short non-MCS tracks whose merge / split link points to it
(directly, no chains), stored in at most nmaxlinks slots per time in the order of their track numbers. tests/test_identifymcs_tb_rows.py
checks that the tables equal what rowstats.dense_to_rows makes of the dense file of identifymcs_tb.

The dense step takes the position of a cold cloud shield area in the compacted list of the entries the sparse file has as the time index
of the track. That is the time index only for a track that has an entry at every time index (every real track: trackstats writes one),
so the tests use such tracks, and here the time index is used directly (a time without an entry is not part of the shield).

Author: Zhe Feng | zhe.feng@pnnl.gov
"""
import logging
import os
import shutil
import time

import numpy as np

from pyflextrkr.identifymcs_rows import dense_width
from pyflextrkr.rowstats import FRAGMENT_KINDS, TableBuilder, gather_index, load_sparse_entries, write_rows
from pyflextrkr.smooth_trajectory import smooth_trajectory


def identify_tb_mcs_track(core_area, cold_area, time_resolution, area_thresh, duration_thresh, gap):
    """
    The Tb test of one track: at which times is it an MCS?

    A track needs a cold core (a non-zero core area at some time). Its cold cloud shield area is core + cold area. The times with a
    shield area above area_thresh are split into periods at gaps of more than `gap` time steps; every period that lasts at least
    duration_thresh hours (last index - first index + 1 time steps) is an MCS period.

    Args:
        core_area, cold_area: float arrays [duration]
            Core and cold area [km^2] of the track at every time index (0 where the sparse file has no entry).
        time_resolution: float
            Hours per time step.
        area_thresh: float
            Cold cloud shield area threshold [km^2] (mcs_tb_area_thresh).
        duration_thresh: float
            Minimum duration of a period [hours] (mcs_tb_duration_thresh).
        gap: int
            Largest gap in time steps that still joins two times into one period (mcs_tb_gap).

    Returns:
        status: bool array [duration]
            True at the times of the periods that satisfy the duration.
        is_mcs: bool
            True if there is at least one such period.
    """
    status = np.zeros(len(core_area), dtype=bool)
    # Must have a cold core: some non-zero, non-NaN core area, and a positive one
    core = core_area[(~np.isnan(core_area)) & (core_area != 0)]
    if core.shape[0] == 0 or not np.nanmax(core > 0):
        return status, False
    # Cold cloud shield area above the threshold (NaN is not above it); no such time: not an MCS
    ccs_area = core_area + cold_area
    iccs = np.flatnonzero(ccs_area > area_thresh)
    if len(iccs) == 0:
        return status, False
    # Periods of consecutive times, and the duration test of every period
    groups = np.split(iccs, np.where(np.diff(iccs) > gap)[0] + 1)
    is_mcs = False
    for group in groups:
        # Duration of the period: last index - first index + 1 time steps, in hours
        duration_group = np.multiply(group[-1] - group[0] + 1, time_resolution)
        if duration_group >= duration_thresh:
            status[group] = True
            is_mcs = True
    return status, is_mcs


def tb_fragment_links(link_tracknumber, lifetime_hr, duration_thresh, is_mcs):
    """
    The tracks that merge into (or split from) an MCS: short non-MCS tracks whose link points to an MCS track.

    Args:
        link_tracknumber: array [ntracks]
            end_merge_tracknumber (merging) or start_split_tracknumber (splitting): the number (1-based) of the track a track merges
            into / splits from, NaN or a fill value for none.
        lifetime_hr: float array [ntracks]
            Lifetime of every track [hours].
        duration_thresh: float
            A fragment lives shorter than this [hours] (mcs_tb_merge_duration or mcs_tb_split_duration).
        is_mcs: bool array [ntracks]
            Which tracks are MCS.

    Returns:
        fragments: int array [nfragments]
            Track indices (0-based) of the fragments, ascending.
        roots: int array [nfragments]
            Track index of the MCS every fragment belongs to.
    """
    number = np.asarray(link_tracknumber, dtype=float)
    # A link is a whole track number inside the file (NaN and fill values are not)
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(number) & (number >= 1) & (number <= len(is_mcs)) & (number == np.round(number))
    root = np.zeros(len(number), dtype=np.int64)
    root[valid] = number[valid].astype(np.int64) - 1
    # Short, not an MCS itself, linked to an MCS
    candidate = valid & is_mcs[root] & (~is_mcs) & (lifetime_hr < duration_thresh)
    fragments = np.flatnonzero(candidate)
    return fragments, root[fragments]


def attach_tb_fragments_flat(fragments, roots, kind, basetime, cloudnumber, ccs_area, offsets, new_index_of, nmaxmerge):
    """
    The cloud number and cold cloud shield area of every fragment at the times of its MCS, as fragment rows.

    A fragment time point goes to the slot (0, 1, ...) after the ones already used by earlier fragments (lower track numbers) at the
    same time of the same MCS, as in identifymcs_tb; a point whose base time is not a time of the MCS, or that would be the slot
    nmaxmerge or higher, is not stored.

    Args:
        fragments, roots: int arrays [nfragments]
            Fragment tracks (ascending) and the MCS track each belongs to (tb_fragment_links).
        kind: int
            Index in rowstats.FRAGMENT_KINDS (0 merge, 1 split).
        basetime, cloudnumber, ccs_area: arrays [nentries]
            Flat variables of all tracks (track after track); base_time is 0 or NaN where the sparse file has no entry.
        offsets: int array [ntracks + 1]
            Start of every track in the flat arrays plus the end of the last.
        new_index_of: int array [ntracks]
            Index of every MCS track among the kept (MCS) tracks.
        nmaxmerge: int
            Number of slots (nmaxlinks).

    Returns:
        columns: dictionary
            Fragment rows {'track' (index among the MCS tracks), 'k' (time index in the MCS), 'kind', 'slot', 'cloudnumber',
            'ccs_area'} as arrays, in the order they were attached.
        n_unmatched: int
            Time points of fragments that are not a time of their MCS.
        n_truncated: int
            Time points that did not fit into nmaxmerge slots.
    """
    rows = {c: [] for c in ("track", "k", "slot", "cloudnumber", "ccs_area")}
    slots_used = {}
    n_unmatched = n_truncated = 0
    # Loop over the fragment tracks in ascending track order: the order in which the slots of a time are filled
    for f, r in zip(fragments, roots):
        a, b = offsets[f], offsets[f + 1]
        ra, rb = offsets[r], offsets[r + 1]
        frag_bt, root_bt = basetime[a:b], basetime[ra:rb]
        # The time points the fragment has (base time not 0 / NaN) and the time index of the MCS with the same base time
        has_time = np.flatnonzero((frag_bt != 0) & (~np.isnan(frag_bt)))
        if has_time.size == 0:
            continue
        same_time = frag_bt[has_time][:, None] == root_bt[None, :]
        found = same_time.any(axis=1)
        root_time = same_time.argmax(axis=1)
        n_unmatched += int((~found).sum())
        # Loop over the time points of this fragment that are also times of the MCS
        for j in np.flatnonzero(found):
            tt = int(root_time[j])
            new_idx = int(new_index_of[r])
            slot = slots_used.get((kind, new_idx, tt), 0)
            # More fragments at this time than slots: the later ones are dropped
            if slot >= nmaxmerge:
                n_truncated += 1
                continue
            # Store the fragment's cloud number and cold cloud shield area in the next free slot
            slots_used[(kind, new_idx, tt)] = slot + 1
            rows["track"].append(new_idx)
            rows["k"].append(tt)
            rows["slot"].append(slot)
            rows["cloudnumber"].append(cloudnumber[a + has_time[j]])
            rows["ccs_area"].append(ccs_area[a + has_time[j]])
    dtypes = {"track": np.int32, "k": np.int32, "slot": np.int16, "cloudnumber": cloudnumber.dtype, "ccs_area": ccs_area.dtype}
    columns = {c: np.array(rows[c], dtype=dtypes[c]) for c in rows}
    columns["kind"] = np.full(len(columns["track"]), kind, dtype=np.int8)
    return columns, n_unmatched, n_truncated


def identifymcs_tb_rows(config):
    """
    Identify MCS from the Tb statistics of the tracks: identifymcs.identifymcs_tb with row tables as the output.

    Args:
        config: dictionary
            Config parameters (those of identifymcs_tb; duration_range only sets the width of the dense layout that rows_to_dense
            rebuilds, a longer track never stops the step).

    Returns:
        statistics_outfolder: string
            Folder of the row tables, f"{stats_path}{mcstbstats_filebase}{startdate}_{enddate}.rows" (tracks.parquet,
            entries.parquet, fragments.parquet, meta.json), the input of the rows-mode matchpf. With no MCS the tables are empty
            (a warning is logged) and the output check of the run script stops the run.
    """
    logger = logging.getLogger(__name__)
    logger.info("Identifying MCS based on Tb statistics (row tables)")

    #########################################################################
    # Settings and the sparse trackstats file as flat entries
    #########################################################################
    mcstbstats_filebase = config["mcstbstats_filebase"]
    stats_path = config["stats_outpath"]
    startdate, enddate = config["startdate"], config["enddate"]
    time_resolution = config["datatimeresolution"]
    mcs_tb_area_thresh = config["mcs_tb_area_thresh"]
    duration_thresh = config["mcs_tb_duration_thresh"]
    split_duration = config["mcs_tb_split_duration"]
    merge_duration = config["mcs_tb_merge_duration"]
    nmaxmerge = config["nmaxlinks"]
    gap = config["mcs_tb_gap"]
    tracks_dimname, times_dimname = config["tracks_dimname"], config["times_dimname"]
    fillval, fillval_f = config["fillval"], np.nan
    max_trackduration = int(max(config["duration_range"]))
    round_base = config.get("duration_range_round_base", 10)
    max_speed_kmh = config.get("max_speed_for_trajectory_smoothing", 50) * 3.6
    statistics_file = f"{stats_path}{config['trackstats_sparse_filebase']}{startdate}_{enddate}.nc"
    statistics_outfolder = f"{stats_path}{mcstbstats_filebase}{startdate}_{enddate}.rows"
    ds_1d, flat, sparse_attrs, offsets, longest = load_sparse_entries(statistics_file, tracks_dimname, times_dimname)
    width = dense_width(longest, max_trackduration, round_base)
    ntracks_all = ds_1d.sizes[tracks_dimname]
    logger.info(f"Number of tracks to process: {ntracks_all}")
    track_duration = ds_1d["track_duration"].values
    track_lifetime_hr = np.multiply(track_duration, time_resolution)
    basetime, cloudnumber = flat["base_time"], flat["cloudnumber"]
    core_area, cold_area = flat["core_area"], flat["cold_area"]
    ccs_area_all = core_area + cold_area
    nentries = int(offsets[-1])

    #########################################################################
    # MCS test, per track
    #########################################################################
    is_mcs = np.zeros(ntracks_all, dtype=bool)
    mcs_period_flat = np.zeros(nentries, dtype=bool)
    # Only a track with a cold cloud shield area above the threshold at some time can be an MCS: the others are not looked at
    n_big = np.zeros(ntracks_all, dtype=np.int64)
    nonempty = track_duration > 0
    if nonempty.any():
        n_big[nonempty] = np.add.reduceat((ccs_area_all > mcs_tb_area_thresh).astype(np.int64), offsets[:-1][nonempty])
    # Loop over the tracks that have such a time
    for nt in np.flatnonzero(n_big > 0):
        sl = slice(offsets[nt], offsets[nt + 1])
        # The Tb test of this track: the times of its MCS periods
        status, track_is_mcs = identify_tb_mcs_track(core_area[sl], cold_area[sl], time_resolution, mcs_tb_area_thresh, duration_thresh, gap)
        mcs_period_flat[sl] = status
        is_mcs[nt] = track_is_mcs
    trackidx_mcs = np.flatnonzero(is_mcs)
    nmcs = len(trackidx_mcs)
    logger.info(f"Number of Tb defined MCS: {nmcs}")
    if nmcs == 0:
        logger.warning("No MCS identified (per mcs_tb_area_thresh / mcs_tb_duration_thresh). The tables are empty.")

    #########################################################################
    # Clouds that merge into and split from the MCS
    #########################################################################
    new_index_of = np.full(ntracks_all, -1, dtype=np.int64)
    new_index_of[trackidx_mcs] = np.arange(nmcs)
    merge_tracks, merge_roots = tb_fragment_links(ds_1d["end_merge_tracknumber"].values, track_lifetime_hr, merge_duration, is_mcs)
    split_tracks, split_roots = tb_fragment_links(ds_1d["start_split_tracknumber"].values, track_lifetime_hr, split_duration, is_mcs)
    parts, n_unmatched, n_truncated = [], 0, 0
    # Loop over the two kinds of fragments (merge, split): their slots are counted separately
    for kind, (tracks, roots) in enumerate(((merge_tracks, merge_roots), (split_tracks, split_roots))):
        part, unmatched, truncated = attach_tb_fragments_flat(
            tracks, roots, kind, basetime, cloudnumber, ccs_area_all, offsets, new_index_of, nmaxmerge)
        parts.append(part)
        n_unmatched, n_truncated = n_unmatched + unmatched, n_truncated + truncated
    columns = {c: np.concatenate([p[c] for p in parts]) for c in parts[0]}
    logger.info(f"Merging clouds: {len(merge_tracks)}, splitting clouds: {len(split_tracks)}")
    if n_truncated > 0:
        logger.warning(f"{n_truncated} fragment time point(s) dropped: more than nmaxmerge ({nmaxmerge}) fragments at one time of one "
                       f"MCS. Increase nmaxlinks to avoid this.")

    #########################################################################
    # Assemble the row tables of the MCS tracks
    #########################################################################
    index, kept_offsets = gather_index(offsets, trackidx_mcs)
    kept_duration = track_duration[trackidx_mcs]
    global_attrs = dict(ds_1d.attrs)
    global_attrs.update({"Title": "Statistics of each MCS track", "MCS_duration_hr": duration_thresh,
                         "MCS_area_km^2": mcs_tb_area_thresh, "Created_on": time.ctime(time.time())})
    builder = TableBuilder(kept_duration, width, nslots=nmaxmerge, tracks_dim=tracks_dimname, times_dim=times_dimname,
                           global_attrs=global_attrs)

    # 1-D variables of the trackstats file (the merge / split links are used up and dropped, as the 2-D merge / split track numbers)
    drop = {"merge_tracknumbers", "split_tracknumbers", "start_split_tracknumber", "start_split_timeindex",
            "end_merge_tracknumber", "end_merge_timeindex", tracks_dimname}
    for name, var in ds_1d.variables.items():
        if name in drop:
            continue
        if var.dims == (tracks_dimname,):
            builder.add_track_var(name, var.values[trackidx_mcs], var.attrs)
        elif len(var.dims) == 2 and var.dims[0] == tracks_dimname:
            builder.add_track2d_var(name, var.values[trackidx_mcs], var.dims[1], var.attrs)
        elif tracks_dimname not in var.dims:
            builder.add_coord(name, var.values, var.attrs, var.dims)
        else:
            raise ValueError(f"variable {name} with dimensions {var.dims} cannot be stored as rows")

    # [tracks, times] variables of the trackstats file; no data (base_time == 0) is the fill value
    no_data = basetime[index] == 0
    kept = {}
    for name, values in flat.items():
        if name in drop:
            continue
        column = values[index]
        is_float = np.issubdtype(column.dtype, np.floating)
        fill = fillval_f if is_float else fillval
        column[no_data] = fill
        kept[name] = column
        builder.add_entry_var(name, column, sparse_attrs[name], fill=fill)

    # Smoothed trajectory of every MCS, on its valid entries (at least three: a shorter trajectory takes other branches)
    lon_smooth = np.full(len(index), fillval_f, dtype=np.float32)
    lat_smooth = np.full(len(index), fillval_f, dtype=np.float32)
    # Loop over the MCS tracks
    for i in range(nmcs):
        a, b = kept_offsets[i], kept_offsets[i + 1]
        n = max(int(b - a), 3)
        lon_in, lat_in = np.full(n, np.nan), np.full(n, np.nan)
        lon_in[:b - a], lat_in[:b - a] = kept["meanlon"][a:b], kept["meanlat"][a:b]
        # Smooth the longitude and latitude of this track and keep the valid part
        lon_s, lat_s = smooth_trajectory(lon_in, lat_in, max_speed_kmh=max_speed_kmh, time_step_h=time_resolution)
        lon_smooth[a:b], lat_smooth[a:b] = lon_s[:b - a], lat_s[:b - a]
    builder.add_entry_var("meanlon_smooth", lon_smooth, {
        "long_name": "Smoothed longitude of a feature", "units": "degrees_east", "_FillValue": fillval_f,
        "comments": f"Smoothed using a maximum speed threshold of {max_speed_kmh} km/h"}, fill=fillval_f)
    builder.add_entry_var("meanlat_smooth", lat_smooth, {
        "long_name": "Smoothed latitude of a feature", "units": "degrees_north", "_FillValue": fillval_f,
        "comments": f"Smoothed using a maximum speed threshold of {max_speed_kmh} km/h"}, fill=fillval_f)

    # Cold cloud shield area and the Tb flag of every entry: 1 above the threshold, 0 below it (area > 0), fill where there is no entry
    ccs_area = kept["core_area"] + kept["cold_area"]
    mcs_status = np.full(len(index), fillval, dtype=np.int16)
    mcs_status[ccs_area > mcs_tb_area_thresh] = 1
    mcs_status[(ccs_area <= mcs_tb_area_thresh) & (ccs_area > 0)] = 0
    builder.add_entry_var("ccs_area", ccs_area, {"long_name": "Area of cold cloud shield", "units": "km^2", "_FillValue": fillval_f},
                          fill=fillval_f)
    builder.add_entry_var("mcs_status", mcs_status, {
        "long_name": "Flag indicating the status of MCS based on Tb. 1 = Yes, 0 = No", "units": "unitless", "_FillValue": fillval},
        fill=fillval)
    # Duration of the MCS periods of every MCS track [time steps]
    mcs_duration = np.zeros(nmcs, dtype=np.int64)
    if nmcs > 0:
        mcs_duration = np.add.reduceat(mcs_period_flat[index].astype(np.int64), kept_offsets[:-1])
    builder.add_track_var("mcs_duration", mcs_duration, {
        "long_name": "Duration of MCS stage", "units": "unitless",
        "comments": "Multiply by time_resolution_hour to convert to physical units"})

    # Merge / split fragments
    attrs_cn = {"merge": "Cloud numbers that merge into MCS", "split": "Cloud numbers that split from MCS"}
    attrs_ar = {"merge": "Cold cloud shield area that merge into MCS", "split": "Cold cloud shield area that split from MCS"}
    variables = {}
    for kind in FRAGMENT_KINDS:
        variables[f"{kind}_cloudnumber"] = ({"long_name": attrs_cn[kind], "units": "unitless", "_FillValue": fillval}, fillval)
        variables[f"{kind}_ccs_area"] = ({"long_name": attrs_ar[kind], "units": "km^2", "_FillValue": fillval_f}, fillval_f)
    builder.add_fragment_vars(columns, variables)
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
