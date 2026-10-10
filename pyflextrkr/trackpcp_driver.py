"""
Post-processing drivers: measure a gridded variable (typically precipitation) inside tracked features.

    compute_track_precip(config)        step 1: statistics of the variable (and, for precipitation, rain totals, convective /
                                        stratiform split and precipitation features) in the swath of every track, written into
                                        a new copy of the track statistics file and as a Parquet table
    compute_track_pixel_accum(config)   step 2: accumulated depth at every pixel of every track (Parquet long table) and
                                        per-track metrics for finding high-impact tracks
    compute_class_maps(config)          step 3: accumulated precipitation, feature counts per class on the grid

The data need not be part of the tracking: own files, own time axis (NetCDF files or a Zarr store), same grid as the
tracking assumed (checked). Each data record represents a window of track times (trackpcp_time_match, trackpcp_window_sec):
see trackpcp_func.py for the swath, convective swath and one-owner-per-pixel definitions.

Config keys (all prefixed trackpcp_; the tracking config supplies paths and dates)

  data          data_path, data_filebase, data_time_format, data_varname, data_format (null = auto), data_time_varname,
                data_scale, data_offset
  matching      time_match (prior | after | nearest), window_sec (period one record represents), tolerance_sec
  tracks        stats_filebase (input track statistics), out_filebase, mask_varname (pixel files), pixel_filebase,
                pixel_time_format, class_varname (null = no classes), classes (class values or null = all),
                class_priority (null = ascending class value), convective_varname (null = no convective split),
                convective_value
  statistics    stats (count, area, mean, median, min, max, sum, std, pNN), stats_prefix, precip (true: precipitation
                module), rain_thresh, heavy_thresh, nmaxpf, pf_min_area
  step 2        box_km (box sizes for the local metrics), accum_thresholds_mm, cluster_thresh_mm
  step 3        pcp_thresh (mm/h), coverage_varname (pixel variable whose missing values mark where the tracking input has
                no data, e.g. dbz_comp; null = off)
  output        outdir (default: stats_outpath), max_memory_gb
"""
import os
import sys
import time
import shutil
import logging
import numpy as np
import xarray as xr
import pandas as pd
import dask
import dask.array as da

from pyflextrkr.ft_utilities import (
    subset_files_timerange, data_time_windows, match_pairs_to_windows, get_pixel_area,
)
from pyflextrkr.trackvar_io import open_data_source, class_info
from pyflextrkr.parallel_utils import run_bounded
from pyflextrkr import trackpcp_func as F

MAX_BAD_FRACTION = 0.05


########################################################################################################################
# Settings, tracks, windows
########################################################################################################################
def settings(config):
    """The trackpcp_* keys with their defaults."""
    g = config.get
    S = {
        "time_match": g("trackpcp_time_match", "prior"),
        "window_sec": float(g("trackpcp_window_sec", 3600.0)),
        "tolerance_sec": g("trackpcp_tolerance_sec", None),
        "stats_filebase": g("trackpcp_stats_filebase", g("mcsrobust_filebase", "mcs_tracks_final_")),
        "out_filebase": g("trackpcp_out_filebase", "trackpcp_tracks_"),
        "mask_varname": g("trackpcp_mask_varname", "cloudtracknumber"),
        "pixel_filebase": g("trackpcp_pixel_filebase", g("pixeltracking_filebase", "mcstrack_")),
        "pixel_time_format": g("trackpcp_pixel_time_format", "yyyymodd_hhmmss"),
        "class_varname": g("trackpcp_class_varname", "track_class"),
        "classes": g("trackpcp_classes", None),
        "class_priority": g("trackpcp_class_priority", None),
        "conv_varname": g("trackpcp_convective_varname", "convsf"),
        "conv_value": g("trackpcp_convective_value", 4),
        "stats": list(g("trackpcp_stats", ["mean", "median", "max", "p90", "p95", "p99"])),
        "precip": bool(g("trackpcp_precip", True)),
        "rain_thresh": float(g("trackpcp_rain_thresh", 2.0)),
        "heavy_thresh": float(g("trackpcp_heavy_thresh", 10.0)),
        "nmaxpf": int(g("trackpcp_nmaxpf", 3)),
        "pf_min_area": float(g("trackpcp_pf_min_area", 0.0)),
        "box_km": list(g("trackpcp_box_km", [15, 45, 99])),
        "accum_thresholds": list(g("trackpcp_accum_thresholds_mm", [25, 50, 100])),
        "cluster_thresh": float(g("trackpcp_cluster_thresh_mm", 50.0)),
        "pcp_thresh": float(g("trackpcp_pcp_thresh", 1.0)),
        "outdir": g("trackpcp_outdir", None) or config["stats_outpath"],
        "max_memory_gb": float(g("trackpcp_max_memory_gb", 300.0)),
        "cov_varname": g("trackpcp_coverage_varname", None),
        "lon_varname": g("trackpcp_lon_varname", "longitude"),
        "lat_varname": g("trackpcp_lat_varname", "latitude"),
        "files_per_task": int(g("trackpcp_files_per_task", 12)),
    }
    S["stats_prefix"] = g("trackpcp_stats_prefix", None) or str(g("trackpcp_data_varname", "var"))
    # optional sub-period of the tracking period (data records whose time is in it), 'YYYYMMDD.HHMM'
    S["period"] = None
    if g("trackpcp_start", None) is not None or g("trackpcp_end", None) is not None:
        from pyflextrkr.ft_utilities import get_basetime_from_string
        def date_string(v):                    # YAML reads an unquoted 20250513.0000 as a float
            return f"{v:.4f}" if isinstance(v, float) else str(v)
        t0 = get_basetime_from_string(date_string(g("trackpcp_start", config["startdate"])))
        t1 = get_basetime_from_string(date_string(g("trackpcp_end", config["enddate"])))
        S["period"] = (float(t0), float(t1))
    if S["classes"] is not None:
        S["classes"] = [int(c) for c in S["classes"]]
    S["outdir"] = S["outdir"] if S["outdir"].endswith("/") else S["outdir"] + "/"
    return S


def _pixel_files(config, S):
    files, basetimes, _, _ = subset_files_timerange(
        config["pixeltracking_outpath"], S["pixel_filebase"], config["start_basetime"], config["end_basetime"],
        time_format=S["pixel_time_format"])
    order = np.argsort(basetimes, kind="stable")
    return [files[i] for i in order], np.asarray(basetimes, dtype=float)[order]


def open_tracks(config, S):
    """Track statistics file (lazy), the base times and the class information."""
    logger = logging.getLogger(__name__)
    stats_file = f"{config['stats_outpath']}{S['stats_filebase']}{config['startdate']}_{config['enddate']}.nc"
    if not os.path.isfile(stats_file):
        logger.critical(f"{stats_file} not found. Tracking will now exit.")
        sys.exit(1)
    ds = xr.open_dataset(stats_file, mask_and_scale=False, decode_times=False)
    basetime = ds["base_time"].values.astype(float)
    ntracks, ntimes = basetime.shape
    T = {"ds": ds, "stats_file": stats_file, "basetime": basetime, "ntracks": ntracks, "ntimes": ntimes,
         "class_lut": None, "class2d": None, "class_values": None, "class_names": None}
    cv = S["class_varname"]
    if cv and cv in ds:
        var = ds[cv]
        vals = var.values
        fill = var.attrs.get("_FillValue", -9999)
        valid = np.isfinite(vals) & (vals != fill)
        class_values, class_names = class_info(vals[valid], var.attrs.get("long_name"), S["classes"])
        T["class_values"], T["class_names"] = class_values, class_names
        if vals.ndim == 1:
            lut = np.zeros(ntracks + 1, dtype=np.int16)
            lut[1:] = np.where(valid, vals, 0)
            T["class_lut"] = lut
        else:
            T["class2d"] = np.where(valid, vals, 0).astype(np.int16)
    else:
        if cv:
            logger.warning(f"class variable '{cv}' is not in {stats_file}: all tracked features are one group")
        if S["classes"]:
            logger.warning("trackpcp_classes is set but there is no class variable: ignored")
    return T


def select_basetime(T, S):
    """base_time with NaN where a (track, time) is not of a selected class."""
    bt = T["basetime"]
    if not S["classes"]:
        return bt
    if T["class_lut"] is not None:
        keep = np.isin(T["class_lut"][1:], S["classes"])
        return np.where(keep[:, None], bt, np.nan)
    if T["class2d"] is not None:
        return np.where(np.isin(T["class2d"], S["classes"]), bt, np.nan)
    return bt


def plan_windows(config, S, T, source, pixel_files, pixel_bt, include_empty=False, period=None):
    """
    One task per data record: the pixel files of its window and the tracks that are present.

    Returns tasks (list of dict) and diagnostics. A task has window (index in the source), data_index, data_time,
    files, tracks (track numbers, int32), place_itime (time index at which each track's record is stored),
    pair_nsteps, class_at (2-D class variable only).
    """
    logger = logging.getLogger(__name__)
    wstart, wend, closed = data_time_windows(source.times, S["time_match"], S["window_sec"], S["tolerance_sec"])
    bt_sel = select_basetime(T, S)
    pairs = match_pairs_to_windows(bt_sel, wstart, wend, closed)
    side = "right" if closed == "right" else "left"
    fa = np.searchsorted(pixel_bt, wstart, side=side)
    fb = np.searchsorted(pixel_bt, wend, side=side)
    tasks, n_nofile, n_pairs = [], 0, 0
    for w in range(len(source.times)):
        if period is not None:
            t = source.times[w]
            inside = (period[0] < t <= period[1]) if closed == "right" else (period[0] <= t < period[1])
            if not inside:
                continue
        it, jt = pairs[w]
        files = [pixel_files[i] for i in range(fa[w], fb[w])]
        if it.size == 0 and not include_empty:
            continue
        if it.size and not files:
            n_nofile += it.size
            if not include_empty:
                continue
        n_pairs += it.size
        task = {"window": w, "data_index": w, "data_time": float(source.times[w]), "files": files,
                "tracks": np.zeros(0, np.int32), "place_itime": np.zeros(0, np.int32), "pair_nsteps": np.zeros(0, np.int32)}
        if it.size:
            tracks, first, counts = np.unique(it, return_index=True, return_counts=True)   # it is sorted by track
            t_pair = T["basetime"][it, jt]
            if S["time_match"] == "prior":
                pick = first + counts - 1
            elif S["time_match"] == "after":
                pick = first
            else:
                diff = np.abs(t_pair - source.times[w])
                order = np.lexsort((diff, it))
                pick = order[np.r_[True, it[order][1:] != it[order][:-1]]]
            task["tracks"] = (tracks + 1).astype(np.int32)
            task["place_itime"] = jt[pick].astype(np.int32)
            task["pair_nsteps"] = counts.astype(np.int32)
            if T["class2d"] is not None:
                task["class_at"] = T["class2d"][tracks, jt[pick]]
        tasks.append(task)
    if n_nofile:
        logger.warning(f"{n_nofile} (track, time) pairs have no pixel file in their window and are left out")
    return tasks, {"n_pairs": n_pairs, "n_pairs_no_file": n_nofile}


def build_static(config, S, T, source, pixel_files, extra=None):
    """Read-only data shared by all tasks."""
    logger = logging.getLogger(__name__)
    with xr.open_dataset(pixel_files[0], decode_times=False, mask_and_scale=False) as dsp:
        lon = np.asarray(dsp[S["lon_varname"]].values, dtype=np.float64).squeeze()
        lat = np.asarray(dsp[S["lat_varname"]].values, dtype=np.float64).squeeze()
    if tuple(lon.shape) != tuple(source.grid_shape):
        logger.critical(f"data grid {tuple(source.grid_shape)} differs from the tracking grid {tuple(lon.shape)}: "
                        f"remap the data to the tracking grid first. Tracking will now exit.")
        sys.exit(1)
    area = get_pixel_area(config, lat, lon)
    pixel_length = float(np.sqrt(np.nanmean(area))) if np.ndim(area) else float(config["pixel_radius"])
    mean_area = float(np.nanmean(area)) if np.ndim(area) else float(area)
    dx = pixel_length
    static = {
        "source": source, "mask_varname": S["mask_varname"], "conv_varname": S["conv_varname"] or None,
        "cov_varname": S["cov_varname"] or None,
        "conv_value": S["conv_value"], "nlut": T["ntracks"] + 1, "class_lut": T["class_lut"],
        "class_values": S["classes"], "class_priority": S["class_priority"] or (
            T["class_values"] if T["class_values"] is not None else None),
        "pixel_area": area, "pixel_length": pixel_length, "window_sec": S["window_sec"], "stats": S["stats"],
        "stats_prefix": S["stats_prefix"], "precip": S["precip"], "rain_thresh": S["rain_thresh"],
        "heavy_thresh": S["heavy_thresh"], "nmaxpf": S["nmaxpf"],
        "pf_min_npix": int(np.ceil(S["pf_min_area"] / mean_area)) if S["pf_min_area"] > 0 else 1,
        "lon": lon, "lat": lat,
        # (label, odd number of pixels) of the boxes of the local metrics, e.g. ("45km", 15)
        "boxes": [(f"{k:g}km", max(1, int(round(k / dx))) | 1) for k in S["box_km"]],
        "accum_thresholds": S["accum_thresholds"], "cluster_thresh": S["cluster_thresh"], "nx": lon.shape[1],
        "pcp_thresh": S["pcp_thresh"],
    }
    if extra:
        static.update(extra)
    return static


def _check_failures(failed, ntasks, what):
    logger = logging.getLogger(__name__)
    if failed:
        logger.warning(f"{len(failed)} of {ntasks} {what} failed; their values are NaN / missing")
    if ntasks and len(failed) / ntasks > MAX_BAD_FRACTION:
        logger.critical(f"More than {MAX_BAD_FRACTION:.0%} of the {what} failed. Tracking will now exit.")
        sys.exit(1)


def _source_for(config, S):
    margin = S["window_sec"] + (S["tolerance_sec"] or 0.0) + 1.0
    return open_data_source(config, config["start_basetime"] - margin, config["end_basetime"] + margin)


########################################################################################################################
# Step 1
########################################################################################################################
VAR_UNITS = {
    "total_rain": ("Sum of the rain rates of the pixels in the swath", "mm/h"),
    "conv_rain": ("Sum of the rain rates of the convective swath", "mm/h"),
    "strat_rain": ("Sum of the rain rates of the stratiform part of the swath", "mm/h"),
    "total_volrain": ("Volumetric rain rate of the swath: sum of rain rate x pixel area", "mm/h km^2"),
    "conv_volrain": ("Volumetric rain rate of the convective swath", "mm/h km^2"),
    "strat_volrain": ("Volumetric rain rate of the stratiform part of the swath", "mm/h km^2"),
    "total_rainvolume": ("Water volume that fell on the swath during the window", "10^6 m^3"),
    "conv_rainvolume": ("Water volume that fell on the convective swath during the window", "10^6 m^3"),
    "strat_rainvolume": ("Water volume that fell on the stratiform part of the swath during the window", "10^6 m^3"),
    "pcp_area": ("Area of the swath", "km^2"),
    "pcp_conv_area": ("Area of the convective swath", "km^2"),
    "pcp_strat_area": ("Area of the stratiform part of the swath", "km^2"),
    "total_heavyrain": ("Sum of the rain rates above the heavy-rain threshold", "mm/h"),
    "total_heavyvolrain": ("Volumetric rain rate above the heavy-rain threshold", "mm/h km^2"),
    "rainrate_heavyrain": ("Mean rain rate above the heavy-rain threshold", "mm/h"),
    "pcp_valid_frac": ("Fraction of the swath pixels with a valid data value", "unitless"),
    "pcp_nsteps": ("Number of tracking steps of the track in the window", "unitless"),
    "pcp_time": ("Valid time of the data record this row is for", "seconds since 1970-01-01"),
    "pf_npf": ("Number of precipitation features (PF) in the swath", "unitless"),
}


def _attrs(name, S, window_h):
    if name in VAR_UNITS:
        long_name, units = VAR_UNITS[name]
    elif name.startswith("pf_"):
        units = "km^2" if "area" in name else ("degrees" if ("lon" in name or "lat" in name or "orientation" in name) else (
            "unitless" if name in ("pf_skewness", "pf_aspectratio", "pf_eccentricity") else (
                "km" if ("axis" in name or "perimeter" in name) else "mm/h")))
        long_name = f"Precipitation feature statistic {name}"
    else:
        units = "same as the data variable (area: km^2, count: unitless)"
        long_name = f"Statistic of {S['stats_prefix']} in the swath: {name}"
    return {"long_name": long_name, "units": units,
            "comments": (f"Per tracking time; set only at the track's step nearest the data time in the window "
                         f"({S['time_match']}, {S['window_sec']:g} s), NaN at other steps. Swath, owner and convective "
                         f"swath definitions: pyflextrkr/trackpcp_func.py.")}


def _fill_block(itrack, itime, values, nrows, ntimes, dtype):
    extra = values.shape[1:] if values.ndim == 2 else ()
    block = np.full((nrows, ntimes) + tuple(extra), np.nan, dtype=dtype)
    block[itrack, itime] = values
    return block


def dense_from_records(itrack, itime, values, ntracks, ntimes, rows_per_block=1024, dtype=np.float32):
    """Lazy dense (ntracks, ntimes[, ncol]) array of NaN with the record values set, built block by block when it is
    written (the sparse records stay small). itrack must be sorted ascending."""
    extra = (values.shape[1],) if values.ndim == 2 else ()
    blocks = []
    for a in range(0, ntracks, rows_per_block):
        b = min(a + rows_per_block, ntracks)
        lo, hi = np.searchsorted(itrack, [a, b])
        blk = dask.delayed(_fill_block)(itrack[lo:hi] - a, itime[lo:hi], values[lo:hi], b - a, ntimes, dtype)
        blocks.append(da.from_delayed(blk, shape=(b - a, ntimes) + extra, dtype=dtype))
    return da.concatenate(blocks, axis=0)


def run_tasks(func, tasks, static, config, label, consume=None, retries=1):
    run_parallel = config.get("run_parallel", 0)
    return run_bounded(func, tasks, static=static, run_parallel=1 if run_parallel >= 1 else 0, retries=retries,
                       label=label, consume=consume)


def compute_track_precip(config):
    """
    Step 1: statistics of the data inside every track at every data time, added to the track statistics file.

    Each data record is stored at one tracking step of each track present in its window (the step nearest the data time
    inside the window), NaN at the other steps; pcp_nsteps and pcp_time say how many steps the record covers and when
    it is valid. The same rows are written as a Parquet table (output file name with .parquet).

    Returns the output file name.
    """
    logger = logging.getLogger(__name__)
    S = settings(config)
    t0 = time.time()
    logger.info("Track-matched statistics: reading the tracks and planning the windows")
    T = open_tracks(config, S)
    source = _source_for(config, S)
    pixel_files, pixel_bt = _pixel_files(config, S)
    if source.nrecords == 0 or len(pixel_files) == 0:
        logger.critical(f"No data records ({source.nrecords}) or pixel files ({len(pixel_files)}) in the period. Exiting.")
        sys.exit(1)
    tasks, diag = plan_windows(config, S, T, source, pixel_files, pixel_bt, period=S["period"])
    static = build_static(config, S, T, source, pixel_files)
    ntracks, ntimes = T["ntracks"], T["ntimes"]
    n_valid = int(np.isfinite(select_basetime(T, S)).sum())
    logger.info(f"{ntracks} tracks, {n_valid} (track, time) pairs selected; {len(tasks)} windows with tracks "
                f"({diag['n_pairs']} pairs matched to windows, {n_valid - diag['n_pairs']} outside the data period); "
                f"{len(pixel_files)} pixel files, {source.nrecords} data records")

    results, failed = run_tasks(F.trackpcp_stats_window, tasks, static, config, "windows")
    _check_failures(failed, len(tasks), "windows")
    done = [i for i in range(len(tasks)) if results[i] is not None]
    if not done:
        logger.critical("No window finished. Tracking will now exit.")
        sys.exit(1)

    # sparse records, sorted by track
    itrack = np.concatenate([tasks[i]["tracks"].astype(np.int64) - 1 for i in done])
    itime = np.concatenate([tasks[i]["place_itime"].astype(np.int64) for i in done])
    data_time = np.concatenate([np.full(tasks[i]["tracks"].size, tasks[i]["data_time"]) for i in done])
    names = list(results[done[0]]["vars"].keys())
    values = {n: np.concatenate([results[i]["vars"][n] for i in done], axis=0) for n in names}
    o = np.argsort(itrack, kind="stable")
    itrack, itime, data_time = itrack[o], itime[o], data_time[o]
    values = {n: v[o] for n, v in values.items()}
    contested = sum(results[i]["diag"]["n_pixels_contested"] for i in done)
    npixels = sum(results[i]["diag"]["n_pixels"] for i in done)
    logger.info(f"{itrack.size} records; {contested} of {npixels} owner pixels of the windows were claimed by more than one track")

    # per-track totals
    totals = {}
    for name in ("total_rainvolume", "conv_rainvolume", "strat_rainvolume"):
        if name in values:
            fin = np.isfinite(values[name])
            tot = np.bincount(itrack[fin], weights=values[name][fin], minlength=ntracks)
            seen = np.bincount(itrack[fin], minlength=ntracks) > 0
            totals[f"track_{name}"] = np.where(seen, tot, np.nan).astype(np.float32)
    nwin = np.bincount(itrack, minlength=ntracks).astype(np.float32)
    totals["track_pcp_nwindows"] = np.where(nwin > 0, nwin, np.nan).astype(np.float32)
    if "total_rainvolume" in values:
        fin = np.isfinite(values["total_rainvolume"])
        mx = np.full(ntracks, -np.inf)
        np.maximum.at(mx, itrack[fin], values["total_rainvolume"][fin])
        totals["track_max_window_rainvolume"] = np.where(np.isfinite(mx), mx, np.nan).astype(np.float32)

    # the new variables only, built block by block when written; the output file starts as a copy of the input file
    # (a byte copy: the existing variables are not read and compressed again) and the new variables are appended to it
    tdim, timedim = config.get("tracks_dimname", "tracks"), config.get("times_dimname", "times")
    comp = dict(zlib=True, complevel=4, _FillValue=np.float32(np.nan), dtype="float32")
    encoding = {}
    window_h = S["window_sec"] / 3600.0
    dnew = xr.Dataset()
    for name in names:
        arr = values[name]
        if arr.ndim == 2 and name != "pf_npf":
            dnew[name] = ([tdim, timedim, "pf"], dense_from_records(itrack, itime, arr.astype(np.float32), ntracks, ntimes),
                          _attrs(name, S, window_h))
            encoding[name] = dict(comp, chunksizes=(min(512, ntracks), ntimes, arr.shape[1]))
        else:
            dnew[name] = ([tdim, timedim], dense_from_records(itrack, itime, arr.astype(np.float32), ntracks, ntimes),
                          _attrs(name, S, window_h))
            encoding[name] = dict(comp, chunksizes=(min(512, ntracks), ntimes))
    dnew["pcp_time"] = ([tdim, timedim], dense_from_records(itrack, itime, data_time, ntracks, ntimes, dtype=np.float64),
                        {"long_name": VAR_UNITS["pcp_time"][0], "units": VAR_UNITS["pcp_time"][1]})
    encoding["pcp_time"] = dict(comp, dtype="float64", _FillValue=np.nan, chunksizes=(min(512, ntracks), ntimes))
    for name, arr in totals.items():
        dnew[name] = ([tdim], arr, {"long_name": f"Track total of {name[6:]} over the data windows",
                                    "units": "10^6 m^3" if "volume" in name else "unitless"})
    dnew.attrs["trackpcp_time_match"] = S["time_match"]
    dnew.attrs["trackpcp_window_sec"] = S["window_sec"]
    dnew.attrs["trackpcp_data_path"] = str(config.get("trackpcp_data_path", ""))
    dnew.attrs["trackpcp_data_varname"] = str(config.get("trackpcp_data_varname", ""))
    dnew.attrs["trackpcp_classes"] = np.array(S["classes"] if S["classes"] else [], dtype=np.int32)
    dnew.attrs["trackpcp_n_windows_failed"] = len(failed)
    dnew.attrs["trackpcp_created_on"] = time.ctime(time.time())

    os.makedirs(S["outdir"], exist_ok=True)
    outfile = f"{S['outdir']}{S['out_filebase']}{config['startdate']}_{config['enddate']}.nc"
    if os.path.isfile(outfile):
        os.remove(outfile)
    logger.info(f"Writing {outfile}")
    T["ds"].close()
    shutil.copyfile(T["stats_file"], outfile)
    with dask.config.set(scheduler="threads"):
        dnew.to_netcdf(path=outfile, mode="a", format="NETCDF4", encoding=encoding)

    # the same records as a Parquet table
    cols = {"track_number": (itrack + 1).astype(np.int32), "time_index": itime.astype(np.int32),
            "base_time": T["basetime"][itrack, itime], "data_time": data_time}
    for name in names:
        arr = values[name]
        if arr.ndim == 2 and name != "pf_npf":
            for k in range(arr.shape[1]):
                cols[f"{name}_{k + 1}"] = arr[:, k]
        else:
            cols[name] = arr
    df = pd.DataFrame(cols)
    pq_file = outfile[:-3] + ".parquet"
    df.to_parquet(pq_file, engine="pyarrow", compression="zstd", index=False)
    logger.info(f"{pq_file}: {len(df)} rows, {len(df.columns)} columns; {(time.time() - t0) / 60:.1f} min in total")
    return outfile


########################################################################################################################
# Step 2
########################################################################################################################
def compute_track_pixel_accum(config):
    """
    Step 2: accumulated depth [mm] at every pixel of every selected track over its lifetime, and per-track metrics.

    Writes (in outdir)
      trackpcp_pixels_START_END.parquet   long table sorted by track: track_number, pix_index (flat iy * nx + ix of the
                                          tracking grid), accum_mm, rate_mm_day (accum / lifetime in days), max_1h_mm
                                          (largest single-window depth), n_windows
      trackpcp_metrics_START_END.parquet  one row per track: times, duration, class, volume, maxima, areas above
                                          thresholds, local_depth_mm_<L>km (largest mean depth in a box of about L x L km),
                                          local_rate_mm_h_<L>km (largest box-mean hourly rain), cluster area / volume
      trackpcp_grid.nc                    latitude / longitude of the grid
    """
    logger = logging.getLogger(__name__)
    S = settings(config)
    t0 = time.time()
    T = open_tracks(config, S)
    source = _source_for(config, S)
    pixel_files, pixel_bt = _pixel_files(config, S)
    tasks, diag = plan_windows(config, S, T, source, pixel_files, pixel_bt, period=S["period"])
    static = build_static(config, S, T, source, pixel_files)
    ntracks = T["ntracks"]
    nx = static["nx"]
    logger.info(f"Pixel accumulation: {len(tasks)} windows, classes {S['classes'] or 'all'}, boxes {static['boxes']}")

    merged = None
    local = {label: np.full(ntracks + 1, -np.inf) for label, _ in static["boxes"]}
    buffer = []
    stats = {"contested": 0}

    def reduce_buffer():
        nonlocal merged, buffer
        merged = F.merge_accumulations(([merged] if merged is not None else []) + buffer)
        buffer = []

    def consume(i, res):
        buffer.append({"trk": res["trk"], "pix": res["pix"], "depth": res["depth"]})
        for label, _ in static["boxes"]:
            np.maximum.at(local[label], res["local_tracks"], res["local_rate"][label])
        stats["contested"] += res["n_pixels_contested"]
        if len(buffer) >= 48:
            reduce_buffer()

    _, failed = run_tasks(F.trackpcp_accum_window, tasks, static, config, "windows", consume=consume)
    _check_failures(failed, len(tasks), "windows")
    reduce_buffer()
    if merged is None or merged["trk"].size == 0:
        logger.critical("No pixel accumulations. Exiting.")
        sys.exit(1)
    logger.info(f"{merged['trk'].size} (track, pixel) rows for {np.unique(merged['trk']).size} tracks; "
                f"{stats['contested']} owner pixels were claimed by more than one track")

    # metrics, in blocks of tracks
    tr_u, first = np.unique(merged["trk"], return_index=True)
    offsets = np.r_[first, merged["trk"].size]
    block = 2000
    mtasks = []
    for a in range(0, tr_u.size, block):
        b = min(a + block, tr_u.size)
        mtasks.append({"tracks": tr_u[a:b], "offsets": offsets[a:b + 1] - offsets[a],
                       "pix": merged["pix"][offsets[a]:offsets[b]], "accum": merged["accum"][offsets[a]:offsets[b]]})
    mres, mfailed = run_tasks(F.trackpcp_metrics_block, mtasks, static, config, "track blocks")
    _check_failures(mfailed, len(mtasks), "track blocks")
    metrics = {"track_number": np.concatenate([r["tracks"] for r in mres if r is not None])}
    for name in mres[[i for i, r in enumerate(mres) if r is not None][0]]["metrics"]:
        metrics[name] = np.concatenate([r["metrics"][name] for r in mres if r is not None])
    mdf = pd.DataFrame(metrics)
    for label, _ in static["boxes"]:
        rate = local[label][mdf["track_number"].values]
        mdf[f"local_rate_mm_h_{label}"] = np.where(np.isfinite(rate), rate, np.nan).astype(np.float32)
    # track information
    ds = T["ds"]
    idx = mdf["track_number"].values - 1
    bt = T["basetime"]
    with np.errstate(all="ignore"):
        first_t = np.nanmin(bt[idx], axis=1)
        last_t = np.nanmax(bt[idx], axis=1)
    mdf["start_time"] = first_t
    mdf["end_time"] = last_t
    duration_h = (last_t - first_t) / 3600.0 + float(config.get("datatimeresolution", 0.0))
    mdf["duration_h"] = duration_h.astype(np.float32)
    if T["class_lut"] is not None:
        mdf["class_value"] = T["class_lut"][mdf["track_number"].values]
    for name, col in (("meanlon", "start_lon"), ("meanlat", "start_lat")):
        if name in ds:
            mdf[col] = ds[name].values[idx, 0].astype(np.float32)
    # rate per pixel over the track lifetime (days)
    days = pd.Series(duration_h / 24.0, index=mdf["track_number"].values)
    rate_mm_day = merged["accum"] / days.reindex(merged["trk"]).values.astype(np.float32)
    os.makedirs(S["outdir"], exist_ok=True)
    tag = f"{config['startdate']}_{config['enddate']}"
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.table({
        "track_number": merged["trk"].astype(np.int32), "pix_index": merged["pix"].astype(np.int32),
        "accum_mm": merged["accum"].astype(np.float32), "rate_mm_day": rate_mm_day.astype(np.float32),
        "max_1h_mm": merged["max_1h"].astype(np.float32), "n_windows": merged["nwin"].astype(np.int16)})
    ny = static["lon"].shape[0]
    meta = {b"grid_ny": str(ny).encode(), b"grid_nx": str(nx).encode(),
            b"pix_index": b"flat index iy * nx + ix on the tracking grid (row-major)",
            b"accum_mm": b"depth accumulated over the track lifetime, one owner per pixel and window",
            b"rate_mm_day": b"accum_mm / track lifetime in days",
            b"window": f"{S['time_match']} {S['window_sec']:g} s".encode()}
    table = table.replace_schema_metadata(meta)
    pix_file = f"{S['outdir']}trackpcp_pixels_{tag}.parquet"
    pq.write_table(table, pix_file, compression="zstd", row_group_size=1_000_000)
    met_file = f"{S['outdir']}trackpcp_metrics_{tag}.parquet"
    mdf.to_parquet(met_file, engine="pyarrow", compression="zstd", index=False)
    xr.Dataset({"latitude": (["y", "x"], static["lat"]), "longitude": (["y", "x"], static["lon"])}).to_netcdf(
        f"{S['outdir']}trackpcp_grid.nc")
    logger.info(f"{pix_file}, {met_file}: {len(mdf)} tracks; {(time.time() - t0) / 60:.1f} min in total")
    return pix_file, met_file


def track_pixels_to_map(df, track_number, ny, nx, column="accum_mm"):
    """2-D map (ny, nx) of one track's pixel values from the long table (rows of that track)."""
    rows = df[df["track_number"] == track_number] if "track_number" in df else df
    out = np.full(ny * nx, np.nan, dtype=np.float32)
    out[rows["pix_index"].values] = rows[column].values
    return out.reshape(ny, nx)


########################################################################################################################
# Step 3
########################################################################################################################
def compute_class_maps(config, year=None, month=None, outfile=None):
    """
    Step 3: accumulated precipitation (mm), feature counts and precipitation counts per class on the grid, for the
    tracking period of the config (or one calendar month of it).

    Output NetCDF with dimension class (coordinates class_value, class_name) and y, x.
    """
    logger = logging.getLogger(__name__)
    S = settings(config)
    t0 = time.time()
    T = open_tracks(config, S)
    source = _source_for(config, S)
    pixel_files, pixel_bt = _pixel_files(config, S)
    period = S["period"] or (float(config["start_basetime"]), float(config["end_basetime"]))
    if year is not None and month is not None:
        import calendar
        import datetime
        m0 = calendar.timegm(datetime.datetime(int(year), int(month), 1).timetuple())
        ny_, nm_ = (int(year) + (int(month) == 12), 1 if int(month) == 12 else int(month) + 1)
        m1 = calendar.timegm(datetime.datetime(ny_, nm_, 1).timetuple())
        period = (m0, m1)
    tasks, diag = plan_windows(config, S, T, source, pixel_files, pixel_bt, include_empty=True, period=period)
    if T["class_values"] is None:
        class_values, class_names = [1], ["all"]
        extra = {"class_lut": np.where(np.arange(T["ntracks"] + 1) > 0, 1, 0).astype(np.int16)}
    else:
        class_values, class_names = T["class_values"], T["class_names"]
        extra = {}
    static = build_static(config, S, T, source, pixel_files, extra={"class_values": class_values, **extra})
    ny, nx = static["lon"].shape
    ncls = len(class_values)
    nwindows = len(tasks)
    logger.info(f"Class maps: {nwindows} windows, classes {dict(zip(class_values, class_names))}")
    nblocks = int(config.get("trackpcp_nblocks", config.get("nprocesses", 8)))
    nblocks = max(1, min(nblocks, nwindows))
    edges = np.linspace(0, nwindows, nblocks + 1).astype(int)
    btasks = [{"windows": tasks[edges[k]:edges[k + 1]]} for k in range(nblocks) if edges[k + 1] > edges[k]]

    acc = {"nwindows": 0, "contested": 0}

    def consume(i, res):
        if res is None:
            return
        for k, v in res.items():
            if k == "nwindows":
                acc["nwindows"] += v
            elif k == "n_pixels_contested":
                acc["contested"] += v
            else:
                if k not in acc:
                    acc[k] = np.zeros_like(v)
                acc[k] += v

    _, failed = run_tasks(F.trackpcp_classmap_block, btasks, static, config, "blocks", consume=consume, retries=1)
    _check_failures(failed, len(btasks), "blocks")
    n = acc["nwindows"]
    shape3 = (ncls, ny, nx)
    ds = xr.Dataset(coords={"class": np.arange(ncls), "class_value": ("class", np.array(class_values, dtype=np.int32)),
                            "class_name": ("class", np.array(class_names)), "y": np.arange(ny), "x": np.arange(nx)})
    ds["latitude"] = (["y", "x"], static["lat"])
    ds["longitude"] = (["y", "x"], static["lon"])
    ds["precip"] = (["class", "y", "x"], acc["precip"].reshape(shape3).astype(np.float32),
                    {"long_name": "Precipitation accumulated while the pixel is in a feature of the class", "units": "mm"})
    ds["cloud_count"] = (["class", "y", "x"], acc["cloud_count"].reshape(shape3).astype(np.int16),
                         {"long_name": "Number of data windows in which the pixel is in a feature of the class",
                          "units": "windows"})
    ds["precip_count"] = (["class", "y", "x"], acc["precip_count"].reshape(shape3).astype(np.int16),
                          {"long_name": f"Number of windows with precipitation >= {S['pcp_thresh']:g} mm/h in the class",
                           "units": "windows"})
    if "precip_conv" in acc:
        ds["precip_conv"] = (["class", "y", "x"], acc["precip_conv"].reshape(shape3).astype(np.float32),
                             {"long_name": "Precipitation in the convective swath of the class", "units": "mm"})
        ds["precip_strat"] = (["class", "y", "x"], acc["precip_strat"].reshape(shape3).astype(np.float32),
                              {"long_name": "Precipitation in the stratiform part of the swath of the class", "units": "mm"})
    ds["total_precip"] = (["y", "x"], acc["total_precip"].reshape(ny, nx).astype(np.float32),
                          {"long_name": "Total precipitation", "units": "mm"})
    ds["total_precip_count"] = (["y", "x"], acc["total_precip_count"].reshape(ny, nx).astype(np.int16),
                                {"long_name": f"Number of windows with precipitation >= {S['pcp_thresh']:g} mm/h",
                                 "units": "windows"})
    if static.get("cov_varname"):
        ds["precip_uncovered"] = (["y", "x"], acc["precip_uncovered"].reshape(ny, nx).astype(np.float32),
                                  {"long_name": f"Precipitation where {static['cov_varname']} has no value in any step of the window "
                                                "(outside the coverage of the tracking input)", "units": "mm"})
    ds["ntimes"] = ((), np.int32(n), {"long_name": "Number of data windows in the period", "units": "windows"})
    ds.attrs.update({"title": "Precipitation by tracked feature class", "time_match": S["time_match"],
                     "window_sec": S["window_sec"], "pcp_thresh_mm_h": S["pcp_thresh"],
                     "class_variable": S["class_varname"] or "none",
                     "owner_rule": "one owner per pixel and window: best class, most steps, lowest track number",
                     "n_pixels_contested": int(acc.get("contested", 0)), "n_blocks_failed": len(failed),
                     "created_on": time.ctime(time.time())})
    os.makedirs(S["outdir"], exist_ok=True)
    if outfile is None:
        tag = f"{year}{int(month):02d}" if year is not None else f"{config['startdate']}_{config['enddate']}"
        outfile = f"{S['outdir']}trackpcp_classmaps_{tag}.nc"
    enc = {v: dict(zlib=True, complevel=4) for v in ds.data_vars if ds[v].ndim >= 2}
    ds.to_netcdf(outfile, mode="w", format="NETCDF4", encoding=enc)
    logger.info(f"{outfile}: {n} windows, {(time.time() - t0) / 60:.1f} min")
    return outfile
