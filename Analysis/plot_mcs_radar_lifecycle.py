#!/usr/bin/env python
"""
Plot the life cycle of tracks from the radar_feature MCS tracking (identifymcs_radar output), to judge how well
the lifecycle staging (classify_lifecycle: 1 cloud only, 2 unorganized, 3 genesis, 4 mature, 5 decay) describes
real tracks. One 2x2 figure per track:

  (a) map: composite-reflectivity snapshots at equal time cadence over a fixed extent (the swath of the whole track),
      alpha rising from --alpha-min at the start to 1 at the time of the largest --peak-var (default area_45dbz)
      and falling back to --alpha-min at the end; date/time next to each snapshot; a straight arrow from the first
      to the last position; a dot in the stage color at each snapshot
  (b) convective and stratiform area
  (c) maximum 20 and 40 dBZ echo-top height
  (d) convective major axis and circular-equivalent diameter, with the line threshold of the staging
  The series panels are shaded by lifecycle stage, with a strip for the times the MCS criteria are met.

usage:
  plot_mcs_radar_lifecycle.py --config MCS_CONFIG.yml --outdir DIR --tracks 4525 12202
  plot_mcs_radar_lifecycle.py --config MCS_CONFIG.yml --outdir DIR --select evaluated --workers 8

--tracks are track numbers of the stats file (= pixel tracknumber, index + 1). --select evaluated plots every MCS
whose lifecycle was evaluated. Figures go to <outdir>/<group>/mcs<N>_lifecycle.png, where the group describes the
outcome of the staging (complete_ideal, complete_multicycle, genesis_decay_no_mature, no_organized_line,
episode_too_short_for_genesis, not_evaluated); selection_summary.csv lists the flags of every plotted track.
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
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
import cartopy.crs as ccrs
import cartopy.feature as cfeature

STAGE_NAMES = {1: "cloud only", 2: "unorganized", 3: "genesis", 4: "mature", 5: "decay"}
STAGE_COLORS = {1: "#e6e6e6", 2: "#cdeccb", 3: "#fff7a8", 4: "#ffd2c4", 5: "#cfe6f5"}   # light shading
STAGE_DOT = {1: "#9e9e9e", 2: "#4caf50", 3: "#e6c200", 4: "#e8573c", 5: "#3b8fd1"}      # snapshot dots
GROUPS = ["complete_ideal", "complete_multicycle", "genesis_decay_no_mature", "no_organized_line",
          "episode_too_short_for_genesis", "not_evaluated"]
FILL = -9999


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


def load_track(ds, tn, peak_var):
    """Arrays of one track trimmed to its duration (NaN = missing)."""
    k = tn - 1
    dur = int(ds["track_duration"].values[k])
    g = lambda v: ds[v].values[k, :dur].astype(float)
    lon, lat = g("meanlon"), g("meanlat")
    if "meanlon_smooth" in ds:
        # smoothed trajectory where it exists (it has NaN steps in some tracks), else the raw centroid
        lon_s, lat_s = g("meanlon_smooth"), g("meanlat_smooth")
        ok = np.isfinite(lon_s) & np.isfinite(lat_s)
        lon, lat = np.where(ok, lon_s, lon), np.where(ok, lat_s, lat)
    stage = np.nan_to_num(ds["lifecycle_stage"].values[k, :dur].astype(float), nan=FILL).astype(int)
    bt = g("base_time")
    return {
        "tn": tn, "dur": dur, "bt": bt, "hours": (bt - bt[0]) / 3600.0,
        "dt_h": float(ds.attrs.get("time_resolution_hour", 1.0 / 12.0)),
        "lon": ((lon + 180.0) % 360.0) - 180.0, "lat": lat,
        "area": g("area"), "conv_area": g("conv_area"), "strat_area": g("strat_area"),
        "peak": g(peak_var), "eth20": g("maxETH_20dbz"), "eth40": g("maxETH_40dbz"),
        "conv_major": g("conv_majoraxis"), "stage": stage,
        "mcs_status": np.nan_to_num(ds["mcs_status"].values[k, :dur].astype(float), nan=0) == 1,
        "evaluated": ival(ds["lifecycle_evaluated"].values[k]) == 1,
        "ncycles": ival(ds["lifecycle_ncycles"].values[k]), "complete": ival(ds["lifecycle_complete_flag"].values[k]) == 1,
        "life_index": np.nan_to_num(ds["lifecycle_index"].values[k], nan=FILL).astype(int),
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
    if t["life_index"][2] >= 0 or t["life_index"][4] >= 0:
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
        "mcs_criteria_steps": int(t["mcs_status"].sum()), "figure": png,
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


def alpha_ramp(hours, peak_hour, amin):
    """amin at the start, 1 at the peak, amin at the end (linear in time)."""
    h0, h1 = hours[0], hours[-1]
    a = np.empty_like(hours, dtype=float)
    for i, h in enumerate(hours):
        if h <= peak_hour:
            a[i] = 1.0 if peak_hour <= h0 else amin + (1 - amin) * (h - h0) / (peak_hour - h0)
        else:
            a[i] = 1.0 if h1 <= peak_hour else amin + (1 - amin) * (h1 - h) / (h1 - peak_hour)
    return np.clip(a, amin, 1.0)


def label_snapshots(ax, t, idx, fontsize=8.5):
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
    halo = [pe.withStroke(linewidth=2.6, foreground="white")]

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
                    arrowprops=dict(arrowstyle="-", lw=0.6, color="0.25", shrinkA=0, shrinkB=3) if r > 0.1 else None)


def draw_map(ax, t, files, epoch, nframes, amin, margin, mask_mode, panel_aspect):
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

    levels = np.arange(5, 70.1, 5)
    cmap = plt.get_cmap("gist_ncar")
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    idx = np.unique(np.round(np.linspace(0, t["dur"] - 1, max(nframes, 2))).astype(int))
    peak_hour = t["hours"][int(np.nanargmax(np.nan_to_num(t["peak"], nan=-1)))]
    alphas = alpha_ramp(t["hours"][idx], peak_hour, amin)
    cf = None
    for j, i in enumerate(idx):
        if step_files[i] is None:
            continue
        with xr.open_dataset(step_files[i], decode_times=False, mask_and_scale=False) as ds:
            dbz = ds["dbz_comp"].values.squeeze()[sl].astype(float)
            ctn = np.nan_to_num(ds["cloudtracknumber"].values.squeeze(), nan=0)[sl]
        show = np.isfinite(dbz) & (dbz >= levels[0])
        if mask_mode == "track":
            show &= ctn == t["tn"]
        cf = ax.pcolormesh(lon[sl], lat[sl], np.ma.masked_where(~show, dbz), cmap=cmap, norm=norm, alpha=float(alphas[j]),
                           transform=ccrs.PlateCarree(), zorder=2 + j * 0.01, shading="auto")
    states = cfeature.NaturalEarthFeature("cultural", "admin_1_states_provinces_lakes", "50m")
    ax.add_feature(states, edgecolor="gray", facecolor="none", linewidth=0.5, zorder=1)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.3, linestyle="--", color="gray")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 9}
    gl.ylabel_style = {"size": 9}

    for j, i in enumerate(idx):
        sc = STAGE_DOT.get(int(t["stage"][i]), "#9e9e9e") if t["evaluated"] else "#9e9e9e"
        ax.scatter(t["lon"][i], t["lat"][i], s=36, c=sc, edgecolor="k", linewidth=0.8, zorder=20, transform=ccrs.PlateCarree())
    ax.annotate("", xy=(t["lon"][-1], t["lat"][-1]), xytext=(t["lon"][0], t["lat"][0]), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
                textcoords=ccrs.PlateCarree()._as_mpl_transform(ax), zorder=19,
                arrowprops=dict(arrowstyle="-|>", color="k", lw=2.6, shrinkA=0, shrinkB=0, mutation_scale=22))
    ax.scatter([t["lon"][0], t["lon"][-1]], [t["lat"][0], t["lat"][-1]], s=70, c="k", zorder=22, transform=ccrs.PlateCarree())
    return cf, idx, peak_hour


###############################################################################
# Series panels
###############################################################################
def time_edges(h):
    """Cell edges between the (irregular: ~4/6 min) time steps, so shading tiles without gaps."""
    mid = 0.5 * (h[1:] + h[:-1])
    return np.r_[h[0] - (mid[0] - h[0]), mid, h[-1] + (h[-1] - mid[-1])]


def shade_stages(ax, t, label=False):
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
            if label and (x1 - x0) > 0.085 * (e[-1] - e[0]):
                ax.text(0.5 * (x0 + x1), 0.085, STAGE_NAMES[int(st[i])], transform=ax.get_xaxis_transform(), ha="center",
                        va="bottom", fontsize=8.5, color="0.25")
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


def style_series(ax, t, title, ylabel, ylim_top, peak_hour, label_stages=False):
    shade_stages(ax, t, label=label_stages)
    mcs_strip(ax, t)
    ax.axvline(peak_hour, color="k", ls="--", lw=1.0, zorder=1)
    e = time_edges(t["hours"])
    ax.set_xlim(e[0], e[-1])
    ax.set_ylim(0, ylim_top)
    ax.set_title(title, loc="left", fontsize=11.5)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.35, ls="--")
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)


def make_figure(t, group, files, epoch, args, line_thresh, outfile):
    fig = plt.figure(figsize=(15.5, 11.2), dpi=130, facecolor="w")
    gs = fig.add_gridspec(2, 2, wspace=0.17, hspace=0.30, left=0.05, right=0.985, top=0.88, bottom=0.115)
    ax_map = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax_a = fig.add_subplot(gs[0, 1])
    ax_e = fig.add_subplot(gs[1, 0])
    ax_s = fig.add_subplot(gs[1, 1])
    cf, idx, peak_hour = draw_map(ax_map, t, files, epoch, args.nframes, args.alpha_min, args.margin, args.mask, panel_aspect=1.45)
    ax_map.set_title(f"(a) Composite reflectivity, {len(idx)} snapshots", loc="left", fontsize=11.5)
    if cf is not None:
        sm = plt.cm.ScalarMappable(norm=cf.norm, cmap=cf.cmap)   # opaque colors (the snapshots have different alpha)
        cb = fig.colorbar(sm, ax=ax_map, orientation="horizontal", fraction=0.045, pad=0.1, extend="max")
        cb.set_label(f"Composite reflectivity (dBZ); alpha {args.alpha_min:g} -> 1 (max {args.peak_var}) -> {args.alpha_min:g}", fontsize=9)
    label_snapshots(ax_map, t, idx)

    h = t["hours"]
    ax_a.plot(h, t["conv_area"] / 1e3, "o-", color="red", ms=3, lw=1.8, label="Conv")
    ax_a.plot(h, t["strat_area"] / 1e3, "o-", color="green", ms=3, lw=1.8, label="Strat")
    top = 1.08 * np.nanmax(np.r_[t["conv_area"], t["strat_area"]]) / 1e3
    style_series(ax_a, t, "(b) Convective and stratiform area", "Area (x10$^3$ km$^2$)", top, peak_hour)
    ax_e.plot(h, t["eth20"], "o-", color="darkorange", ms=3, lw=1.8, label="Max 20 dBZ")
    ax_e.plot(h, t["eth40"], "o-", color="magenta", ms=3, lw=1.8, label="Max 40 dBZ")
    style_series(ax_e, t, "(c) Convective echo-top height", "ETH (km)", 1.08 * np.nanmax(np.r_[t["eth20"], t["eth40"]]), peak_hour,
                 label_stages=True)
    ax_e.set_xlabel("Hours since detection")
    diam = 2.0 * np.sqrt(np.clip(t["conv_area"], 0, None) / np.pi)
    ax_s.plot(h, t["conv_major"], "o-", color="navy", ms=3, lw=1.8, label="Conv major axis")
    ax_s.plot(h, diam, "o-", color="firebrick", ms=3, lw=1.8, label="Conv equivalent diameter")
    ax_s.axhline(line_thresh, color="navy", ls=":", lw=1.3)
    near = np.abs(np.nan_to_num(t["conv_major"]) - line_thresh) < 0.15 * line_thresh
    left_busy = near[: len(near) // 2].sum() > near[len(near) // 2:].sum()
    ax_s.text(0.995 if left_busy else 0.005, line_thresh, f"line threshold {line_thresh:g} km", transform=ax_s.get_yaxis_transform(),
              va="bottom", ha="right" if left_busy else "left", fontsize=8.5, color="navy",
              bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.8), zorder=6)
    style_series(ax_s, t, "(d) Convective feature size", "Length (km)",
                 1.08 * max(np.nanmax(t["conv_major"]), np.nanmax(diam), line_thresh), peak_hour)
    ax_s.set_xlabel("Hours since detection")

    if t["evaluated"]:
        life = (f"lifecycle: {t['ncycles']} organized-line episode(s), {'complete' if t['complete'] else 'incomplete'}  |  "
                f"steps: unorganized {int((t['stage'] == 2).sum())}, genesis {int((t['stage'] == 3).sum())}, "
                f"mature {int((t['stage'] == 4).sum())}, decay {int((t['stage'] == 5).sum())}")
    else:
        life = "lifecycle not evaluated (lifetime below mcs_lifecycle_thresh): no stage shading"
    fig.suptitle(f"MCS {t['tn']}   {utc(t['bt'][0]).strftime('%Y-%m-%d %H:%M')} - {utc(t['bt'][-1]).strftime('%m-%d %H:%M')} UTC   "
                 f"(lifetime {t['dur'] * t['dt_h']:.1f} h)   group: {group}\n{life}  |  smoothed peaks: "
                 f"strat area {n_peaks(t['strat_area'])}, {args.peak_var} {n_peaks(t['peak'])}", fontsize=12.5, y=0.985)
    handles = [Patch(facecolor=STAGE_COLORS[s], edgecolor="0.6", label=STAGE_NAMES[s]) for s in (1, 2, 3, 4, 5)]
    handles += [Patch(facecolor="#d94fb4", alpha=0.55, label="MCS criteria met"),
                Line2D([0], [0], color="k", ls="--", lw=1.0, label=f"time of max {args.peak_var} (alpha = 1)")]
    fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, 0.005),
               title="stage shading (series panels)  /  snapshot dots use the same stage colors", title_fontsize=9)
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    fig.savefig(outfile, dpi=130)
    plt.close(fig)


###############################################################################
def plot_one(tn, args_dict):
    """Worker: figure for one track; returns its summary row."""
    args = argparse.Namespace(**args_dict)
    ds = xr.open_dataset(args.paths["stats"], mask_and_scale=True, decode_times=False)
    t = load_track(ds, tn, args.peak_var)
    ds.close()
    group = group_of(t)
    files, epoch = pixel_file_index(args.paths["pixdir"], args.paths["pixbase"])
    out = os.path.join(args.outdir, group, f"mcs{tn}_lifecycle.png")
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
    ap.add_argument("--alpha-min", type=float, default=0.2, help="alpha at the start and the end (default 0.2)")
    ap.add_argument("--peak-var", default="area_45dbz", help="variable whose maximum sets alpha=1 (default area_45dbz)")
    ap.add_argument("--margin", type=float, default=0.5, help="map margin around the swath [deg]")
    ap.add_argument("--mask", choices=["track", "none"], default="track",
                    help="show reflectivity only inside the track's mask (default) or everywhere in the window")
    ap.add_argument("--workers", type=int, default=1, help="tracks plotted in parallel (use a compute node for > 1)")
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
    csv_file = os.path.join(args.outdir, "selection_summary.csv")
    with open(csv_file, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (GROUPS.index(r["group"]), r["track"])))
    counts = {g: sum(r["group"] == g for r in rows) for g in GROUPS if any(r["group"] == g for r in rows)}
    print("groups:", counts, "| summary:", csv_file, flush=True)


if __name__ == "__main__":
    main()
