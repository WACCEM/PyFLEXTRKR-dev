#!/usr/bin/env python
"""
Plot the life cycle of tracks from the radar_feature MCS tracking (regionprops output), to judge how well
the lifecycle staging (classify_lifecycle: 1 cloud only, 2 unorganized, 3 genesis, 4 mature, 5 decay) describes
real tracks. One 2x2 figure per track:

  (a) map: composite-reflectivity snapshots at equal time cadence over a fixed extent (the swath of the whole track
      plus a small --margin). The alpha of a snapshot is a Gaussian in time around the largest --peak-var (default
      area_45dbz, from its running median over mcs_lifecycle_smooth_hour): --alpha-max (0.8) at the peak, falling
      with sigma = --alpha-sigma x lifetime towards --alpha-min x --alpha-max. On every snapshot the longest dilated
      convective group inside the track's mask (the group that sets conv_majoraxis, from conv_feature_number in
      the pixel files) is lightly filled (--fill-color, --fill-alpha-scale x the alpha of that snapshot) and outlined
      (thin line, the alpha of the snapshot). Date/time next to each snapshot, a straight arrow from the first to
      the last position, a dot in the stage color per snapshot.
  (b) stratiform area (left axis) and convective area (right axis, scaled so that the smoothed convective peak sits
      10 % below the smoothed stratiform peak); total areas including the merge/split fragments folded into the
      track; dashed horizontal line: the stratiform area a time needs for the mature stage
  (c) maximum 20 and 40 dBZ echo-top height
  (d) convective major axis and circular-equivalent diameter, with the line threshold of the staging
  In (b)-(d) the thick solid lines are the running median (the series the staging thresholds, window from the stats
  file), the thin dashed lines of the same color the raw values. The series panels are shaded by lifecycle stage
  (stage names written in each panel), with a strip for the times the MCS criteria are met.

usage:
  plot_mcs_radar_lifecycle.py --config MCS_CONFIG.yml --outdir DIR --tracks 4525 12202
  plot_mcs_radar_lifecycle.py --config MCS_CONFIG.yml --outdir DIR --select evaluated --workers 8

--tracks are track numbers of the stats file (= pixel tracknumber, index + 1). --select evaluated plots every MCS
whose lifecycle was evaluated. Figures go to <outdir>/<group>/<prefix>mcs<N>_lifecycle.png, where the group describes the
outcome of the staging (complete_ideal, complete_multicycle, mature_no_decay [the track ends while mature],
genesis_decay_no_mature, no_organized_line, episode_too_short_for_genesis, not_evaluated); selection_summary.csv lists
the flags of every plotted track.
"""
import os
import re
import csv
import sys
import glob
import time
import calendar
import argparse
import datetime
import numpy as np
import xarray as xr
import yaml
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgba
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from skimage.measure import regionprops
from pyflextrkr.mcs_lifecycle import running_median, smooth_inputs, stratiform_floor

STAGE_NAMES = {1: "cloud only", 2: "unorganized", 3: "genesis", 4: "mature", 5: "decay"}
STAGE_COLORS = {1: "#e6e6e6", 2: "#cdeccb", 3: "#fff7a8", 4: "#ffd2c4", 5: "#cfe6f5"}   # light shading
STAGE_DOT = {1: "#9e9e9e", 2: "#4caf50", 3: "#e6c200", 4: "#e8573c", 5: "#3b8fd1"}      # snapshot dots
GROUPS = ["complete_ideal", "complete_multicycle", "mature_no_decay", "genesis_decay_no_mature", "no_organized_line",
          "episode_too_short_for_genesis", "not_evaluated"]
FILL = -9999
# Reflectivity color scale of the snapshots (the colorbar is made before the map, so its size is known)
DBZ_LEVELS = np.arange(5, 70.1, 5)
DBZ_CMAP = plt.get_cmap("gist_ncar")
DBZ_NORM = BoundaryNorm(DBZ_LEVELS, ncolors=DBZ_CMAP.N, clip=True)
# Canvas and text. The canvas is FIG_SCALE x the former 15.5 x 11.2 in so that text of twice the former size fits:
# tick and axis labels, panel titles and legends are 2x the former point sizes. The annotations inside the plots
# (snapshot times, stage names, line-threshold text) and all line widths and marker sizes grow with the canvas only.
FIG_SCALE = 1.5
FIG_SIZE = (15.5 * FIG_SCALE, 11.2 * FIG_SCALE)
DPI = 100
FS = dict(tick=20, label=20, title=23, legend=17, figlegend=19, figlegend_title=18, cbar=18, suptitle=24, grid=18)
FS_ANNOT = dict(snapshot=8.5 * FIG_SCALE, stage=7.5 * FIG_SCALE, thresh=8.5 * FIG_SCALE)
LW = 1.8 * FIG_SCALE         # smoothed lines
LW_RAW = 0.8 * FIG_SCALE     # raw lines (thin, dashed)
MS = 3 * FIG_SCALE           # markers of the smoothed lines


###############################################################################
# Paths, track data, summary
###############################################################################
def paths_from_config(config_file):
    cfg = yaml.full_load(open(config_file))
    dates = f"{cfg['startdate']}_{cfg['enddate']}"
    root = cfg["root_path"].rstrip("/")
    return {
        "stats": f"{root}/{cfg.get('stats_path_name', 'stats')}/{cfg['mcsrobust_filebase']}{dates}.nc",
        "pixdir": f"{root}/{cfg['pixel_path_name']}/{dates}/",
        "pixbase": cfg["pixeltracking_filebase"],
        "line_thresh": float(cfg.get("mcs_lifecycle_line_thresh", 100.0)),
        "smooth_hour": float(cfg.get("mcs_lifecycle_smooth_hour", 0.5)),
        "sf_area_min": float(cfg.get("mcs_lifecycle_sf_area_min", 0.0)),
        "pixel_km": float(cfg.get("pixel_radius", 3.0)),
    }


def stamp(path):
    return re.search(r"(\d{8}_\d{6})", os.path.basename(path)).group(1)


def pixel_file_index(pixdir, pixbase):
    files = sorted(glob.glob(f"{pixdir}{pixbase}*.nc"))
    if not files:
        sys.exit(f"no pixel files {pixdir}{pixbase}*.nc")
    epoch = np.array([calendar.timegm(time.strptime(stamp(p), "%Y%m%d_%H%M%S")) for p in files], dtype=float)
    return files, epoch


def nearest_file(t, files, epoch):
    i = int(np.argmin(np.abs(epoch - t)))
    return files[i] if abs(epoch[i] - t) <= 3 else None


def smooth(x, size=7):
    """7 steps = 35 min at 5-min resolution."""
    return uniform_filter1d(np.nan_to_num(np.asarray(x, dtype=float)), size=size, mode="nearest")


def n_peaks(x):
    s = smooth(x)
    if s.max() <= 0:
        return 0
    return len(find_peaks(s, prominence=0.25 * s.max())[0])


def ival(x):
    """int of a stored flag; the fill value of non-evaluated tracks decodes to NaN."""
    return int(np.nan_to_num(float(x), nan=FILL))


def load_track(ds, tn, peak_var, smooth_hour=0.5, sf_area_min=0.0):
    """Arrays of one track trimmed to its duration (NaN = missing)."""
    row = ds.isel(tracks=tn - 1)       # one row: reads only this track from the file (a month has ~200,000 tracks)
    dur = int(row["track_duration"].values)
    g = lambda v: row[v].values[:dur].astype(float)
    lon, lat = g("meanlon"), g("meanlat")
    if "meanlon_smooth" in ds:
        # smoothed trajectory where it exists (it has NaN steps in some tracks), else the raw centroid
        lon_s, lat_s = g("meanlon_smooth"), g("meanlat_smooth")
        ok = np.isfinite(lon_s) & np.isfinite(lat_s)
        lon, lat = np.where(ok, lon_s, lon), np.where(ok, lat_s, lat)
    stage = np.nan_to_num(row["lifecycle_stage"].values[:dur].astype(float), nan=FILL).astype(int)
    bt = g("base_time")
    hours = (bt - bt[0]) / 3600.0
    # areas of the whole track (with the merge/split fragments folded into it) when the stats file has them
    totals = "total_conv_area" in ds and "total_strat_area" in ds
    smooth_hour = float(ds.attrs.get("mcs_lifecycle_smooth_hour", smooth_hour))
    peak = g(peak_var)
    conv, major, strat = g("total_conv_area" if totals else "conv_area"), g("conv_majoraxis"), g("total_strat_area" if totals else "strat_area")
    # the series as the staging thresholds them (NaN -> 0, running median) and the stratiform area that the
    # mature stage needs; the echo-top heights are smoothed for display only (NaN-aware: they have real gaps)
    params = {"mcs_lifecycle_smooth_hour": smooth_hour, "mcs_lifecycle_sf_area_min": sf_area_min}
    conv_s, major_s, strat_s = smooth_inputs(conv, major, strat, hours, params)
    eth20, eth40 = g("maxETH_20dbz"), g("maxETH_40dbz")
    return {
        "tn": tn, "dur": dur, "bt": bt, "hours": hours, "totals": totals,
        "dt_h": float(ds.attrs.get("time_resolution_hour", 1.0 / 12.0)),
        "lon": ((lon + 180.0) % 360.0) - 180.0, "lat": lat,
        "area": g("area"),
        "conv_area": conv, "conv_area_s": conv_s, "strat_area": strat, "strat_area_s": strat_s,
        "sf_floor": stratiform_floor(strat_s, params),
        "peak": peak, "peak_smooth": running_median(np.nan_to_num(peak, nan=0.0), hours, smooth_hour),
        "smooth_hour": smooth_hour, "radius_km": ds.attrs.get("conv_dilate_radius_km", None),
        "eth20": eth20, "eth40": eth40,
        "eth20_s": running_median(eth20, hours, smooth_hour, skipna=True),
        "eth40_s": running_median(eth40, hours, smooth_hour, skipna=True),
        "conv_major": major, "conv_major_s": major_s, "stage": stage,
        "mcs_status": np.nan_to_num(row["mcs_status"].values[:dur].astype(float), nan=0) == 1,
        "evaluated": ival(row["lifecycle_evaluated"].values) == 1,
        "ncycles": ival(row["lifecycle_ncycles"].values), "complete": ival(row["lifecycle_complete_flag"].values) == 1,
        "life_index": np.nan_to_num(row["lifecycle_index"].values, nan=FILL).astype(int),
    }


def group_of(t):
    """Outcome of the lifecycle staging for one track."""
    if not t["evaluated"]:
        return "not_evaluated"
    peaks_strat = n_peaks(t["strat_area"])
    if t["complete"]:
        return "complete_ideal" if (t["ncycles"] == 1 and peaks_strat <= 1) else "complete_multicycle"
    if t["ncycles"] <= 0:
        return "no_organized_line"
    genesis, mature, decay = (t["life_index"][i] >= 0 for i in (2, 3, 4))
    if mature and not decay:
        return "mature_no_decay"          # the track ends while mature
    if genesis or decay:
        return "genesis_decay_no_mature"
    return "episode_too_short_for_genesis"


def summary_row(t, group, line_thresh, png):
    st = t["stage"]
    return {
        "track": t["tn"], "group": group,
        "start_utc": utc(t["bt"][0]).strftime("%Y-%m-%d %H:%M"), "end_utc": utc(t["bt"][-1]).strftime("%Y-%m-%d %H:%M"),
        "hours": round(t["dur"] * t["dt_h"], 2), "evaluated": int(t["evaluated"]), "ncycles": t["ncycles"],
        "complete": int(t["complete"]),
        "steps_cloud_only": int((st == 1).sum()), "steps_unorganized": int((st == 2).sum()),
        "steps_genesis": int((st == 3).sum()), "steps_mature": int((st == 4).sum()), "steps_decay": int((st == 5).sum()),
        "n_peaks_strat_area": n_peaks(t["strat_area"]), "n_peaks_peak_var": n_peaks(t["peak"]),
        "max_conv_majoraxis_km": round(float(np.nanmax(t["conv_major"])), 1),
        "frac_time_conv_majoraxis_ge_threshold": round(float(np.nanmean(np.nan_to_num(t["conv_major"]) >= line_thresh)), 3),
        "frac_time_smoothed_conv_majoraxis_ge_threshold": round(float(np.mean(t["conv_major_s"] >= line_thresh)), 3),
        "mature_strat_floor_km2": round(t["sf_floor"]),
        "mcs_criteria_steps": int(t["mcs_status"].sum()), "uses_total_areas": int(t["totals"]),
        "conv_dilate_radius_km": "" if t["radius_km"] is None else float(t["radius_km"]), "figure": png,
    }


def utc(t):
    return datetime.datetime.fromtimestamp(float(t), datetime.timezone.utc)


###############################################################################
# Map panel
###############################################################################
def track_swath(t, files, epoch):
    """Union bounding box of the track's mask over ALL time steps, and the pixel file of every time step."""
    tn = t["tn"]
    lon = lat = None
    box = [np.inf, -np.inf, np.inf, -np.inf]
    step_files = []
    for bt in t["bt"]:
        f = nearest_file(bt, files, epoch)
        step_files.append(f)
        if f is None:
            continue
        with xr.open_dataset(f, decode_times=False, mask_and_scale=False) as ds:
            if lon is None:
                lon = ((ds["longitude"].values + 180.0) % 360.0) - 180.0
                lat = ds["latitude"].values
            m = np.nan_to_num(ds["cloudtracknumber"].values.squeeze(), nan=0) == tn
        if m.any():
            box = [min(box[0], lon[m].min()), max(box[1], lon[m].max()), min(box[2], lat[m].min()), max(box[3], lat[m].max())]
    return box, step_files, lon, lat


def alpha_gauss(hours, peak_hour, amin, amax, sigma_hours):
    """Alpha of the snapshots: amax at the peak time, falling like a Gaussian in time (width sigma_hours) towards the
    floor amin * amax."""
    g = np.exp(-0.5 * ((np.asarray(hours, dtype=float) - peak_hour) / sigma_hours) ** 2)
    return amax * (amin + (1.0 - amin) * g)


def longest_group(track_mask, group_number, convsf, pixel_km):
    """
    The dilated convective group with the largest major axis inside a track's mask: among the conv_feature_number groups
    that contain a convective pixel (convsf == 4) of the track, the one whose region (the group inside the mask) is
    longest. Same recipe as regionprops_func (second pass), so its length is the stored conv_majoraxis.

    Returns:
        region: 2D bool array of that group inside the mask (None if the mask has no such group)
        length_km: float, its major axis length
    """
    in_mask = track_mask & (group_number > 0)
    keep = np.unique(group_number[track_mask & (convsf == 4)])
    keep = keep[keep > 0]
    if keep.size == 0:
        return None, 0.0
    labels = np.where(in_mask, group_number, 0).astype(np.int32)
    best, best_len = None, -1.0
    for rp in regionprops(labels):
        if rp.label in keep and rp.axis_major_length * pixel_km > best_len:
            best, best_len = rp.label, rp.axis_major_length * pixel_km
    return labels == best, best_len


def label_snapshots(ax, t, idx, fontsize=FS_ANNOT["snapshot"]):
    """Date/time text next to each snapshot dot. Greedy placement in axes-fraction space (call after the axes size is
    final): the first of several offsets around the dot whose label box overlaps no earlier label and no dot."""
    fig = ax.figure
    bb = ax.get_window_extent()
    w_in, h_in = bb.width / fig.dpi, bb.height / fig.dpi
    inv = ax.transAxes.inverted()
    pos = np.array([inv.transform(ax.transData.transform((t["lon"][i], t["lat"][i]))) for i in idx])
    texts = [utc(t["bt"][i]).strftime("%m/%d %H:%MZ") for i in idx]
    bw, bh = (len(texts[0]) * 0.6 * fontsize / 72.0 + 0.05) / w_in, (1.5 * fontsize / 72.0) / h_in
    dot = 0.012
    placed = []
    halo = [pe.withStroke(linewidth=2.6 * FIG_SCALE, foreground="white")]

    def box(p, ang, r):
        c, s_ = np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))
        ax_, ay_ = p[0] + c * r / w_in, p[1] + s_ * r / h_in
        ha = "left" if c > 0.3 else ("right" if c < -0.3 else "center")
        va = "bottom" if s_ > 0.3 else ("top" if s_ < -0.3 else "center")
        x0 = ax_ if ha == "left" else (ax_ - bw if ha == "right" else ax_ - bw / 2)
        y0 = ay_ if va == "bottom" else (ay_ - bh if va == "top" else ay_ - bh / 2)
        return (x0, y0, x0 + bw, y0 + bh), (ax_, ay_), ha, va

    def hit(b, o):
        return not (b[2] < o[0] or o[2] < b[0] or b[3] < o[1] or o[3] < b[1])

    for k, p in enumerate(pos):
        best = None
        for r in (0.10, 0.28, 0.50, 0.75):
            for ang in (0, 45, -45, 90, -90, 135, -135, 180):
                b, anchor, ha, va = box(p, ang, r)
                bad = sum(hit(b, o) for o in placed)
                bad += sum(hit(b, (q[0] - dot, q[1] - dot, q[0] + dot, q[1] + dot)) for m, q in enumerate(pos) if m != k)
                bad += 5 * (b[0] < 0.0 or b[1] < 0.0 or b[2] > 1.0 or b[3] > 1.0)
                if best is None or bad < best[0]:
                    best = (bad, b, anchor, ha, va, r)
                if bad == 0:
                    break
            if best[0] == 0:
                break
        _, b, anchor, ha, va, r = best
        placed.append(b)
        ax.annotate(texts[k], xy=(t["lon"][idx[k]], t["lat"][idx[k]]), xycoords="data", xytext=anchor, textcoords="axes fraction",
                    ha=ha, va=va, fontsize=fontsize, fontweight="bold", path_effects=halo, zorder=21,
                    arrowprops=dict(arrowstyle="-", lw=0.6 * FIG_SCALE, color="0.25", shrinkA=0, shrinkB=3) if r > 0.1 else None)


def draw_map(ax, t, files, epoch, args, panel_aspect):
    nframes, amin, amax, margin = args.nframes, args.alpha_min, args.alpha_max, args.margin
    box, step_files, lon, lat = track_swath(t, files, epoch)
    lat_c, lon_c = 0.5 * (box[2] + box[3]), 0.5 * (box[0] + box[1])
    half_h = 0.5 * (box[3] - box[2]) + margin
    half_w = 0.5 * (box[1] - box[0]) + margin
    coslat = np.cos(np.deg2rad(lat_c))
    # widen the shorter side so the map keeps roughly equal km per inch in x and y
    if half_w * coslat / half_h < panel_aspect:
        half_w = panel_aspect * half_h / coslat
    else:
        half_h = half_w * coslat / panel_aspect
    ext = (lon_c - half_w, lon_c + half_w, lat_c - half_h, lat_c + half_h)
    inwin = (lon >= ext[0]) & (lon <= ext[1]) & (lat >= ext[2]) & (lat <= ext[3])
    rows, cols = np.where(inwin)
    sl = (slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1))
    ax.set_extent(ext, crs=ccrs.PlateCarree())
    ax.set_aspect("auto")

    levels, cmap, norm = DBZ_LEVELS, DBZ_CMAP, DBZ_NORM
    idx = np.unique(np.round(np.linspace(0, t["dur"] - 1, max(nframes, 2))).astype(int))
    peak_hour = t["hours"][int(np.nanargmax(t["peak_smooth"]))]
    sigma_hours = max(args.alpha_sigma * (t["hours"][-1] - t["hours"][0]), 0.25)
    alphas = alpha_gauss(t["hours"][idx], peak_hour, amin, amax, sigma_hours)
    cf = None
    fill_diff, fill_missing = [], False
    for j, i in enumerate(idx):
        if step_files[i] is None:
            continue
        with xr.open_dataset(step_files[i], decode_times=False, mask_and_scale=False) as ds:
            dbz = ds["dbz_comp"].values.squeeze()[sl].astype(float)
            ctn = np.nan_to_num(ds["cloudtracknumber"].values.squeeze(), nan=0)[sl]
            has_groups = args.mask_fill == "longest" and "conv_feature_number" in ds
            if has_groups:
                cfn = np.nan_to_num(ds["conv_feature_number"].values.squeeze(), nan=0)[sl].astype(np.int64)
                convsf = ds["convsf"].values.squeeze()[sl]
            elif args.mask_fill == "longest":
                fill_missing = True
        show = np.isfinite(dbz) & (dbz >= levels[0])
        if args.mask == "track":
            show &= ctn == t["tn"]
        cf = ax.pcolormesh(lon[sl], lat[sl], np.ma.masked_where(~show, dbz), cmap=cmap, norm=norm, alpha=float(alphas[j]),
                           transform=ccrs.PlateCarree(), zorder=2 + j * 0.01, shading="auto")
        if has_groups:
            region, length_km = longest_group(ctn == t["tn"], cfn, convsf, args.paths["pixel_km"])
            if region is not None:
                # the length must be the stored conv_majoraxis of this time step
                fill_diff.append(abs(length_km - t["conv_major"][i]))
                ax.pcolormesh(lon[sl], lat[sl], np.ma.masked_where(~region, np.ones(region.shape)),
                              cmap=ListedColormap([args.fill_color]), vmin=0, vmax=1,
                              alpha=float(args.fill_alpha_scale * alphas[j]), transform=ccrs.PlateCarree(),
                              zorder=2 + j * 0.01 + 0.005, shading="auto")
                if args.fill_outline > 0:
                    # thin outline of the group, same color, alpha of the snapshot (same curve as the reflectivity)
                    ax.contour(lon[sl], lat[sl], region.astype(float), levels=[0.5], colors=[args.fill_color],
                               linewidths=args.fill_outline * FIG_SCALE, alpha=float(alphas[j]), transform=ccrs.PlateCarree(),
                               zorder=2 + j * 0.01 + 0.006)
    if fill_missing:
        print(f"MCS {t['tn']}: no conv_feature_number in the pixel files, no group fill", flush=True)
    elif fill_diff and max(fill_diff) > 0.01 * max(float(np.nanmax(t["conv_major"])), 1.0):
        print(f"MCS {t['tn']}: WARNING longest group differs from the stored conv_majoraxis by up to "
              f"{max(fill_diff):.1f} km", flush=True)
    states = cfeature.NaturalEarthFeature("cultural", "admin_1_states_provinces_lakes", "50m")
    ax.add_feature(states, edgecolor="gray", facecolor="none", linewidth=0.5 * FIG_SCALE, zorder=1)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.3 * FIG_SCALE, linestyle="--", color="gray")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": FS["grid"]}
    gl.ylabel_style = {"size": FS["grid"]}
    gl.xlocator = mticker.MaxNLocator(5)
    gl.ylocator = mticker.MaxNLocator(5)

    for j, i in enumerate(idx):
        sc = STAGE_DOT.get(int(t["stage"][i]), "#9e9e9e") if t["evaluated"] else "#9e9e9e"
        ax.scatter(t["lon"][i], t["lat"][i], s=36 * FIG_SCALE ** 2, c=sc, edgecolor="k", linewidth=0.8 * FIG_SCALE,
                   zorder=20, transform=ccrs.PlateCarree())
    ax.annotate("", xy=(t["lon"][-1], t["lat"][-1]), xytext=(t["lon"][0], t["lat"][0]), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
                textcoords=ccrs.PlateCarree()._as_mpl_transform(ax), zorder=19,
                arrowprops=dict(arrowstyle="-|>", color="k", lw=2.6 * FIG_SCALE, shrinkA=0, shrinkB=0,
                                mutation_scale=22 * FIG_SCALE))
    ax.scatter([t["lon"][0]], [t["lat"][0]], s=70 * FIG_SCALE ** 2, c="k", zorder=22, transform=ccrs.PlateCarree())
    return cf, idx, peak_hour, sigma_hours


###############################################################################
# Series panels
###############################################################################
def time_edges(h):
    """Cell edges between the (irregular: ~4/6 min) time steps, so shading tiles without gaps."""
    mid = 0.5 * (h[1:] + h[:-1])
    return np.r_[h[0] - (mid[0] - h[0]), mid, h[-1] + (h[-1] - mid[-1])]


def shade_stages(ax, t, label=True, label_y=0.07):
    if not t["evaluated"]:
        return
    h, st = t["hours"], t["stage"]
    e = time_edges(h)
    i = 0
    while i < len(st):
        j = i
        while j + 1 < len(st) and st[j + 1] == st[i]:
            j += 1
        if st[i] in STAGE_COLORS:
            x0, x1 = e[i], e[j + 1]
            ax.axvspan(x0, x1, color=STAGE_COLORS[int(st[i])], zorder=0, linewidth=0)
            if label and (x1 - x0) > 0.12 * (e[-1] - e[0]):
                ax.text(0.5 * (x0 + x1), label_y, STAGE_NAMES[int(st[i])], transform=ax.get_xaxis_transform(), ha="center",
                        va="bottom", fontsize=FS_ANNOT["stage"], color="0.2", zorder=3,
                        path_effects=[pe.withStroke(linewidth=2.0 * FIG_SCALE, foreground="white")])
        i = j + 1


def mcs_strip(ax, t):
    e, m = time_edges(t["hours"]), t["mcs_status"]
    i = 0
    while i < len(m):
        if m[i]:
            j = i
            while j + 1 < len(m) and m[j + 1]:
                j += 1
            ax.axvspan(e[i], e[j + 1], ymin=0, ymax=0.045, color="#d94fb4", alpha=0.55, zorder=1, linewidth=0)
            i = j + 1
        else:
            i += 1


def style_series(ax, t, title, ylabel, ylim_top, peak_hour, legend_handles=None, label_y=0.07, legend_loc=("upper right", None)):
    shade_stages(ax, t, label=True, label_y=label_y)
    mcs_strip(ax, t)
    ax.axvline(peak_hour, color="k", ls="--", lw=1.0 * FIG_SCALE, zorder=1)
    e = time_edges(t["hours"])
    ax.set_xlim(e[0], e[-1])
    ax.set_ylim(0, ylim_top)
    ax.set_title(title, loc="left", fontsize=FS["title"])
    ax.set_ylabel(ylabel, fontsize=FS["label"])
    ax.tick_params(labelsize=FS["tick"])
    ax.grid(alpha=0.35, ls="--")
    loc, anchor = legend_loc
    if legend_handles is None:
        ax.legend(loc=loc, bbox_to_anchor=anchor, fontsize=FS["legend"], framealpha=0.9)
    else:
        ax.legend(handles=legend_handles, loc=loc, bbox_to_anchor=anchor, fontsize=FS["legend"], framealpha=0.9)


def legend_corner(ax, hours, fractions, labels):
    """
    Corner of the axes ('upper left', 'upper right', 'lower right', 'lower left') where a legend with these labels covers
    the fewest plotted samples; ties go to the upper corners.
    Returns (loc, anchor): the legend location and, for the lower corners, an anchor above the row of stage names.

    Args:
        ax: the axes (its size sets how large the legend is relative to it)
        hours: 1D np.array, sample times
        fractions: list of 1D np.array, the plotted series as fractions of the axis height
        labels: list of str, the legend entries
    """
    bb = ax.get_window_extent()
    w_in, h_in = bb.width / ax.figure.dpi, bb.height / ax.figure.dpi
    wf = min(0.9, (max(len(x) for x in labels) * FS["legend"] * 0.55 + 80) / 72.0 / w_in)
    hf = min(0.9, (len(labels) * FS["legend"] * 1.5 + 14) / 72.0 / h_in)
    x = (hours - hours[0]) / (hours[-1] - hours[0])
    boxes = {"upper left": (0.0, wf, 1 - hf, 1.0), "upper right": (1 - wf, 1.0, 1 - hf, 1.0),
             "lower right": (1 - wf, 1.0, 0.10, 0.10 + hf), "lower left": (0.0, wf, 0.10, 0.10 + hf)}
    best, best_n = "upper right", None
    for loc, (x0, x1, y0, y1) in boxes.items():
        inside = (x >= x0) & (x <= x1)
        n = sum(int(np.sum(inside & (f >= y0) & (f <= y1))) for f in fractions) + (1 if loc.startswith("lower") else 0)
        if best_n is None or n < best_n:
            best, best_n = loc, n
    anchor = {"lower right": (0.995, 0.11), "lower left": (0.005, 0.11)}.get(best)
    return best, anchor


def plot_pair(ax, hours, raw, smoothed, color, label):
    """Thick solid line of the smoothed series (circle markers) and a thin dashed line of the raw values, same color."""
    ln, = ax.plot(hours, smoothed, "o-", color=color, ms=MS, lw=LW, label=label, zorder=4)
    ax.plot(hours, raw, ls="--", color=color, lw=LW_RAW, zorder=3)
    return ln


def make_figure(t, group, files, epoch, args, line_thresh, outfile):
    fig = plt.figure(figsize=FIG_SIZE, dpi=DPI, facecolor="w")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1.0], wspace=0.17, hspace=0.32, left=0.05, right=0.945, top=0.865,
                          bottom=0.15)
    ax_map = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax_a = fig.add_subplot(gs[0, 1])
    ax_e = fig.add_subplot(gs[1, 0])
    ax_s = fig.add_subplot(gs[1, 1])
    # the colorbar first (opaque colors: the snapshots have different alpha), so that the map panel has its final
    # size when the extent is chosen
    sm = plt.cm.ScalarMappable(norm=DBZ_NORM, cmap=DBZ_CMAP)
    cb = fig.colorbar(sm, ax=ax_map, orientation="horizontal", fraction=0.05, pad=0.12, extend="max")
    cb.set_label("Composite reflectivity (dBZ)", fontsize=FS["label"])
    cb.ax.tick_params(labelsize=FS["tick"])
    bb = ax_map.get_window_extent()
    cf, idx, peak_hour, sigma_hours = draw_map(ax_map, t, files, epoch, args, panel_aspect=bb.width / bb.height)
    ax_map.set_title(f"(a) Composite reflectivity, {len(idx)} snapshots", loc="left", fontsize=FS["title"])
    label_snapshots(ax_map, t, idx)

    h = t["hours"]
    # (b) stratiform area (left axis) and convective area (right axis); thick = running median, thin dashed = raw
    strat, strat_s = t["strat_area"] / 1e3, t["strat_area_s"] / 1e3
    conv, conv_s = t["conv_area"] / 1e3, t["conv_area_s"] / 1e3
    floor = t["sf_floor"] / 1e3
    mx = lambda x: float(np.nanmax(x)) if np.isfinite(x).any() else 0.0
    ylim_strat = 1.08 * max(mx(strat), mx(strat_s), 1e-9)
    ln_s = plot_pair(ax_a, h, strat, strat_s, "green", "Strat (left axis)")
    ax_c = ax_a.twinx()
    ln_c = plot_pair(ax_c, h, conv, conv_s, "red", "Conv (right axis)")
    # secondary axis: the smoothed convective peak sits at 90 % of the height of the smoothed stratiform peak
    ylim_conv = mx(conv_s) * ylim_strat / (0.9 * mx(strat_s)) if (mx(strat_s) > 0 and mx(conv_s) > 0) else 1.08 * max(mx(conv), 1e-9)
    ax_c.set_ylim(0, ylim_conv)
    handles_b = [ln_s, ln_c]
    if t["evaluated"] and floor > 0:
        # the mature stage needs the line threshold of (d) AND at least this stratiform area (mean of its positive values)
        handles_b.append(ax_a.axhline(floor, color="darkgreen", ls=(0, (6, 3)), lw=1.5 * FIG_SCALE, zorder=2.5,
                                      label=f"mature: strat >= {floor:.0f}"))
    corner = legend_corner(ax_a, h, [strat / ylim_strat, strat_s / ylim_strat, conv / ylim_conv, conv_s / ylim_conv],
                           [x.get_label() for x in handles_b])
    style_series(ax_a, t, "(b) Stratiform and convective area", "Stratiform area (x10$^3$ km$^2$)", ylim_strat, peak_hour,
                 legend_handles=handles_b, legend_loc=corner)
    ax_a.yaxis.label.set_color("green")
    ax_a.tick_params(axis="y", colors="green")
    ax_c.set_ylabel("Convective area (x10$^3$ km$^2$)", color="red", fontsize=FS["label"])
    ax_c.tick_params(axis="y", colors="red", labelsize=FS["tick"])
    # (c) echo-top height
    plot_pair(ax_e, h, t["eth20"], t["eth20_s"], "darkorange", "Max 20 dBZ")
    plot_pair(ax_e, h, t["eth40"], t["eth40_s"], "magenta", "Max 40 dBZ")
    ylim_eth = 1.08 * np.nanmax(np.r_[t["eth20"], t["eth40"]])
    style_series(ax_e, t, "(c) Convective echo-top height", "ETH (km)", ylim_eth, peak_hour,
                 legend_loc=legend_corner(ax_e, h, [t["eth20"] / ylim_eth, t["eth40"] / ylim_eth, t["eth20_s"] / ylim_eth,
                                                    t["eth40_s"] / ylim_eth], ["Max 20 dBZ", "Max 40 dBZ"]))
    ax_e.set_xlabel("Hours since detection", fontsize=FS["label"])
    # (d) convective length
    diam = 2.0 * np.sqrt(np.clip(t["conv_area"], 0, None) / np.pi)
    diam_s = 2.0 * np.sqrt(np.clip(t["conv_area_s"], 0, None) / np.pi)
    plot_pair(ax_s, h, t["conv_major"], t["conv_major_s"], "navy", "Conv major axis")
    plot_pair(ax_s, h, diam, diam_s, "firebrick", "Conv equivalent diameter")
    ax_s.axhline(line_thresh, color="navy", ls=":", lw=1.3 * FIG_SCALE)
    near = np.abs(t["conv_major_s"] - line_thresh) < 0.15 * line_thresh
    left_busy = near[: len(near) // 2].sum() > near[len(near) // 2:].sum()
    ax_s.text(0.995 if left_busy else 0.005, line_thresh, f"line threshold {line_thresh:g} km", transform=ax_s.get_yaxis_transform(),
              va="bottom", ha="right" if left_busy else "left", fontsize=FS_ANNOT["thresh"], color="navy",
              bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.8), zorder=6)
    ylim_len = 1.08 * max(np.nanmax(t["conv_major"]), np.nanmax(diam), line_thresh)
    thr_frac = line_thresh / ylim_len
    # the stage names normally sit at 7 % of the panel height; lift them above the threshold label if that is near
    label_y = 0.07 if abs(thr_frac - 0.07) > 0.07 else thr_frac + 0.065
    style_series(ax_s, t, "(d) Convective feature size", "Length (km)", ylim_len, peak_hour, label_y=label_y,
                 legend_loc=legend_corner(ax_s, h, [t["conv_major"] / ylim_len, t["conv_major_s"] / ylim_len, diam / ylim_len],
                                          ["Conv major axis", "Conv equivalent diameter"]))
    ax_s.set_xlabel("Hours since detection", fontsize=FS["label"])

    if t["evaluated"]:
        life = (f"lifecycle: {t['ncycles']} line episode(s), {'complete' if t['complete'] else 'incomplete'}  |  "
                f"steps: unorganized {int((t['stage'] == 2).sum())}, genesis {int((t['stage'] == 3).sum())}, "
                f"mature {int((t['stage'] == 4).sum())}, decay {int((t['stage'] == 5).sum())}")
    else:
        life = "lifecycle not evaluated (lifetime below mcs_lifecycle_thresh): no stage shading"
    group_text = (f"conv. grouping radius {float(t['radius_km']):g} km, " if t["radius_km"] is not None else "") + \
                 f"lifecycle smoothing {t['smooth_hour']:g} h"
    areas = "total" if t["totals"] else "own-feature"
    fig.suptitle(f"MCS {t['tn']}   {utc(t['bt'][0]).strftime('%Y-%m-%d %H:%M')} - {utc(t['bt'][-1]).strftime('%m-%d %H:%M')} UTC   "
                 f"(lifetime {t['dur'] * t['dt_h']:.1f} h)   group: {group}\n{life}\n"
                 f"{areas} areas  |  smoothed peaks: strat area {n_peaks(t['strat_area'])}, {args.peak_var} {n_peaks(t['peak'])}  |  {group_text}",
                 fontsize=FS["suptitle"], y=0.99)
    handles = [Patch(facecolor=STAGE_COLORS[s], edgecolor="0.6", label=STAGE_NAMES[s]) for s in (1, 2, 3, 4, 5)]
    handles += [Patch(facecolor="#d94fb4", alpha=0.55, label="MCS criteria met"),
                Line2D([0], [0], color="k", ls="--", lw=1.0 * FIG_SCALE, label=f"time of max {args.peak_var}"),
                Line2D([0], [0], color="0.3", ls="-", marker="o", ms=MS, lw=LW, label=f"running median {t['smooth_hour']:g} h"),
                Line2D([0], [0], color="0.3", ls="--", lw=LW_RAW, label="raw")]
    if args.mask_fill == "longest":
        handles.append(Patch(facecolor=to_rgba(args.fill_color, args.fill_alpha_scale * args.alpha_max),
                             edgecolor=to_rgba(args.fill_color, args.alpha_max), label="longest dilated convective group"))
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=FS["figlegend"], frameon=False, bbox_to_anchor=(0.5, 0.0),
               title=f"stage shading and snapshot dots; snapshot alpha: Gaussian in time around the max {args.peak_var} "
                     f"(sigma {sigma_hours:.1f} h)", title_fontsize=FS["figlegend_title"])
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    fig.savefig(outfile, dpi=DPI)
    plt.close(fig)


###############################################################################
def plot_one(tn, args_dict):
    """Worker: figure for one track; returns its summary row."""
    args = argparse.Namespace(**args_dict)
    ds = xr.open_dataset(args.paths["stats"], mask_and_scale=True, decode_times=False)
    t = load_track(ds, tn, args.peak_var, args.paths["smooth_hour"], args.paths["sf_area_min"])
    ds.close()
    group = group_of(t)
    files, epoch = pixel_file_index(args.paths["pixdir"], args.paths["pixbase"])
    out = os.path.join(args.outdir, group, f"{args.prefix}mcs{tn}_lifecycle.png")
    make_figure(t, group, files, epoch, args, args.paths["line_thresh"], out)
    print(f"MCS {tn}: {group} -> {out}", flush=True)
    return summary_row(t, group, args.paths["line_thresh"], out)


def main():
    ap = argparse.ArgumentParser(description="Life cycle figure (map snapshots + stage-shaded series) for MCS tracks")
    ap.add_argument("--config", required=True, help="MCS tracking yaml (radar_feature)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tracks", type=int, nargs="+", help="track numbers (stats index + 1)")
    ap.add_argument("--select", choices=["evaluated"], help="plot every MCS whose lifecycle was evaluated")
    ap.add_argument("--nframes", type=int, default=8, help="number of reflectivity snapshots (default 8)")
    ap.add_argument("--alpha-max", type=float, default=0.8, help="alpha of the snapshot at the peak (default 0.8)")
    ap.add_argument("--alpha-min", type=float, default=0.2,
                    help="alpha far from the peak, as a fraction of --alpha-max (default 0.2)")
    ap.add_argument("--alpha-sigma", type=float, default=0.12,
                    help="width of the Gaussian fall-off of the snapshot alpha, as a fraction of the lifetime (default 0.12)")
    ap.add_argument("--mask-fill", choices=["longest", "none"], default="longest",
                    help="fill the longest dilated convective group of each snapshot (default) or nothing")
    ap.add_argument("--fill-alpha-scale", type=float, default=0.4,
                    help="alpha of the group fill as a fraction of the alpha of that snapshot (default 0.5)")
    ap.add_argument("--fill-color", default="k", help="color of the group fill and its outline (default k)")
    ap.add_argument("--fill-outline", type=float, default=1.0,
                    help="line width [pt] of a thin outline of the filled group, alpha of the snapshot (0 = none; default 1.0)")
    ap.add_argument("--peak-var", default="area_45dbz",
                    help="variable whose running-median maximum sets the time of alpha = --alpha-max (default area_45dbz)")
    ap.add_argument("--margin", type=float, default=0.1, help="map margin around the swath [deg] (default 0.1)")
    ap.add_argument("--mask", choices=["track", "none"], default="track",
                    help="show reflectivity only inside the track's mask (default) or everywhere in the window")
    ap.add_argument("--workers", type=int, default=1, help="tracks plotted in parallel (use a compute node for > 1)")
    ap.add_argument("--prefix", default="",
                    help="put in front of the figure and summary file names, e.g. 202505_ (track numbers repeat across "
                         "monthly stats files); default none")
    args = ap.parse_args()
    if not args.tracks and not args.select:
        ap.error("give --tracks or --select evaluated")
    args.paths = paths_from_config(args.config)

    ds = xr.open_dataset(args.paths["stats"], mask_and_scale=True, decode_times=False)
    tracks = list(args.tracks or [])
    if args.select == "evaluated":
        sel = np.where((ds["is_mcs"].values == 1) & (ds["lifecycle_evaluated"].values == 1))[0] + 1
        tracks = sorted(set(tracks) | set(int(s) for s in sel))
    ds.close()
    print(f"{len(tracks)} track(s) -> {args.outdir}", flush=True)

    args_dict = vars(args)
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as ex:
            rows = list(ex.map(plot_one, tracks, [args_dict] * len(tracks)))
    else:
        rows = [plot_one(tn, args_dict) for tn in tracks]
    os.makedirs(args.outdir, exist_ok=True)
    csv_file = os.path.join(args.outdir, f"{args.prefix}selection_summary.csv")
    with open(csv_file, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (GROUPS.index(r["group"]), r["track"])))
    counts = {g: sum(r["group"] == g for r in rows) for g in GROUPS if any(r["group"] == g for r in rows)}
    print("groups:", counts, "| summary:", csv_file, flush=True)


if __name__ == "__main__":
    main()
