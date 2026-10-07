#!/usr/bin/env python
"""
Compare the MCS pixel masks of one MCS track between the previous and the current merge/split
code (identifymcs_radar before / after "keep merge/split fragments that do not coexist with their root").

Inputs: the work directory made by setup_comparison.py and filled by run_tree.py, with one root per
code version (<workdir>/old and <workdir>/new, each with stats/ and the mapped pixel files).

Outputs (<outdir>/track<TN>/): one 3-panel PNG per time frame of the track (previous | current |
difference), frame_metrics.csv, a time series PNG, a key-frame montage, an mp4 (if ffmpeg is found)
and summary.json (also all-MCS totals and what the Tb-style attach logic would do on this data).

usage: analyze_mask_difference.py --workdir DIR --outdir DIR --tn TRACKNUMBER [--workers 64]
       TRACKNUMBER is the MCS track number in the CURRENT code's stats (pixel tracknumber = index + 1).
"""
import os
import sys
import json
import glob
import time
import calendar
import argparse
import re
import datetime
import numpy as np
import xarray as xr

# Set by configure() from the work directory (module globals so the Dask frame tasks can use them)
M = None            # work directory: old/, new/, code_new/ (made by setup_comparison.py)
DATES = None        # "<startdate>_<enddate>" of the tracking
STATS_NAME = None   # stats file of the MCS tracks, relative to a version root
PIX_DIR = None      # pixel-level directory, relative to a version root
PIX_BASE = None     # pixel-level file base name
PIXEL_KM2 = 9.0     # area of one grid cell [km^2] (3 km x 3 km)


def configure(workdir, pixel_km2=9.0):
    """Read the dates and file names from the config that setup_comparison.py wrote for the new version."""
    import yaml
    global M, DATES, STATS_NAME, PIX_DIR, PIX_BASE, PIXEL_KM2
    M = os.path.abspath(workdir)
    cfg = yaml.full_load(open(f"{M}/new/config.yml"))
    DATES = f"{cfg['startdate']}_{cfg['enddate']}"
    STATS_NAME = f"stats/{cfg['mcsrobust_filebase']}{DATES}.nc"
    PIX_BASE = cfg["pixeltracking_filebase"]
    PIX_DIR = f"{cfg['pixel_path_name']}/{DATES}/"
    PIXEL_KM2 = float(pixel_km2)


def stamp(path):
    """YYYYMMDD_HHMMSS of a pixel-level file or figure name."""
    return re.search(r"(\d{8}_\d{6})", os.path.basename(path)).group(1)


def hhmm(t):
    return datetime.datetime.utcfromtimestamp(float(t)).strftime("%m-%d %H:%M")


###############################################################################
# Track lookup
###############################################################################
def open_stats(version):
    return xr.open_dataset(f"{M}/{version}/{STATS_NAME}", decode_times=False, mask_and_scale=False)


def find_old_index(new, old, idx_new):
    """Row of the same MCS in the old stats: MCS tracks are identical in both versions, so match on
    their per-time basetime and cloudnumber (track numbers differ because released fragments are now kept)."""
    n_bt = new["base_time"].values[idx_new]
    n_cn = new["cloudnumber"].values[idx_new]
    o_bt = old["base_time"].values
    same_bt = ((o_bt == n_bt) | (np.isnan(o_bt) & np.isnan(n_bt))).all(axis=1)   # NaN = fill value
    match = np.where(same_bt & (old["cloudnumber"].values == n_cn).all(axis=1))[0]
    if len(match) != 1:
        raise RuntimeError(f"expected exactly one matching old track, got {match}")
    return int(match[0])


def attached_series(ds, idx, dur):
    """Per-time own area, attached fragment count and attached fragment area for one track [km^2]."""
    own = ds["area"].values[idx, :dur].astype(float)
    out = {"own_area": own}
    for kind in ("merge", "split"):
        cn = ds[f"{kind}_cloudnumber"].values[idx, :dur, :]
        ar = ds[f"{kind}_area"].values[idx, :dur, :].astype(float)
        valid = cn > 0
        out[f"{kind}_n"] = valid.sum(axis=1)
        out[f"{kind}_area"] = np.where(valid, np.nan_to_num(ar, nan=0.0), 0.0).sum(axis=1)
    out["attached_n"] = out["merge_n"] + out["split_n"]
    out["attached_area"] = out["merge_area"] + out["split_area"]
    return out


def pixel_file_for(epoch, files_epoch, files):
    i = int(np.argmin(np.abs(files_epoch - epoch)))
    if abs(files_epoch[i] - epoch) > 3:
        return None
    return files[i]


###############################################################################
# One frame
###############################################################################
def frame_task(fp_old, fp_new, old_tn, new_tn, figname, title_extra):
    """Make the 3-panel figure for one frame and return per-frame metrics."""
    import matplotlib
    matplotlib.use("agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from scipy.ndimage import binary_erosion, binary_dilation

    do = xr.open_dataset(fp_old)
    dn = xr.open_dataset(fp_new)
    lon = ((dn["longitude"].values + 180.0) % 360.0) - 180.0
    lat = dn["latitude"].values
    dbz = dn["dbz_comp"].values.squeeze()
    ctn_o = np.nan_to_num(do["cloudtracknumber"].values.squeeze(), nan=0).astype(np.int64)
    ctn_n = np.nan_to_num(dn["cloudtracknumber"].values.squeeze(), nan=0).astype(np.int64)
    feat = np.nan_to_num(dn["feature_number"].values.squeeze(), nan=0)
    tn_o = np.nan_to_num(do["tracknumber"].values.squeeze(), nan=0).astype(np.int64)   # track's own features only
    tn_n = np.nan_to_num(dn["tracknumber"].values.squeeze(), nan=0).astype(np.int64)
    timestr = str(dn["time"].values.squeeze())[:19].replace("T", " ")

    prev = ctn_o == old_tn
    curr = ctn_n == new_tn
    only_prev = prev & ~curr
    only_curr = curr & ~prev
    orph_prev = (feat > 0) & (ctn_o == 0)
    orph_curr = (feat > 0) & (ctn_n == 0)
    recovered = orph_prev & (ctn_n > 0)
    only_prev_stand = only_prev & (ctn_n > 0)    # left the MCS mask, now a standalone track
    only_prev_none = only_prev & (ctn_n == 0)    # left the MCS mask, now in no track

    metrics = {
        "time": timestr,
        "prev_px": int(prev.sum()), "curr_px": int(curr.sum()),
        "both_px": int((prev & curr).sum()),
        "only_prev_px": int(only_prev.sum()), "only_curr_px": int(only_curr.sum()),
        "feature_px": int((feat > 0).sum()),
        "orphan_prev_px": int(orph_prev.sum()), "orphan_curr_px": int(orph_curr.sum()),
        "recovered_px": int(recovered.sum()),
        "recovered_in_window_px": 0,
        "only_prev_standalone_px": int(only_prev_stand.sum()), "only_prev_none_px": int(only_prev_none.sum()),
        "prev_own_px": int((tn_o == old_tn).sum()), "curr_own_px": int((tn_n == new_tn).sum()),
    }

    # ---- plotting window: union of both masks, with a margin, near-square in km ----
    union = prev | curr
    if not union.any():
        union = prev | curr | recovered
    if not union.any():
        return metrics
    lon_u, lat_u = lon[union], lat[union]
    lat_c = 0.5 * (lat_u.min() + lat_u.max())
    lon_c = 0.5 * (lon_u.min() + lon_u.max())
    coslat = np.cos(np.deg2rad(lat_c))
    half_h = max(1.8, 0.5 * (lat_u.max() - lat_u.min()) + 0.5)
    half_w = max(half_h / coslat, 0.5 * (lon_u.max() - lon_u.min()) + 0.5)
    half_h = max(half_h, half_w * coslat)
    x0, x1, y0, y1 = lon_c - half_w, lon_c + half_w, lat_c - half_h, lat_c + half_h
    inwin = (lon >= x0) & (lon <= x1) & (lat >= y0) & (lat <= y1)
    rows, cols = np.where(inwin)
    sl = (slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1))
    lon_s, lat_s, dbz_s = lon[sl], lat[sl], dbz[sl]
    prev_s, curr_s = prev[sl], curr[sl]
    only_prev_s, only_curr_s, recovered_s = only_prev[sl], only_curr[sl], recovered[sl]
    only_prev_stand_s, only_prev_none_s = only_prev_stand[sl], only_prev_none[sl]
    # tracks that now hold the pixels that left the MCS mask (standalone tracks)
    stand_ids = np.unique(ctn_n[only_prev_stand])
    stand_mask_s = np.isin(ctn_n, stand_ids)[sl] if len(stand_ids) else np.zeros_like(prev_s)
    metrics["recovered_in_window_px"] = int(recovered_s.sum())

    def perimeter(mask, width=1):
        return mask & ~binary_erosion(mask, iterations=width)

    def draw_mask_outline(ax, mask, color, width=1, zorder=5, halo=True):
        if not mask.any():
            return
        per = perimeter(mask, width)
        if halo:
            halo_m = binary_dilation(per, iterations=1) & ~per
            ax.pcolormesh(lon_s, lat_s, np.ma.masked_where(~halo_m, np.ones(halo_m.shape)),
                          cmap=ListedColormap(["k"]), vmin=0, vmax=1, alpha=0.85,
                          transform=ccrs.PlateCarree(), zorder=zorder, shading="auto")
        ax.pcolormesh(lon_s, lat_s, np.ma.masked_where(~per, np.ones(per.shape)),
                      cmap=ListedColormap([color]), vmin=0, vmax=1,
                      transform=ccrs.PlateCarree(), zorder=zorder + 0.1, shading="auto")

    def draw_fill(ax, mask, color, alpha, zorder):
        if not mask.any():
            return
        ax.pcolormesh(lon_s, lat_s, np.ma.masked_where(~mask, np.ones(mask.shape)),
                      cmap=ListedColormap([color]), vmin=0, vmax=1, alpha=alpha,
                      transform=ccrs.PlateCarree(), zorder=zorder, shading="auto")

    levels = np.arange(-10, 70.1, 5)
    cmap = plt.get_cmap("gist_ncar")
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    dbz_m = np.ma.masked_where(~np.isfinite(dbz_s) | (dbz_s < levels[0]), dbz_s)
    states = cfeature.NaturalEarthFeature("cultural", "admin_1_states_provinces_lakes", "50m")

    fig = plt.figure(figsize=(21, 7.6), dpi=120, facecolor="w")
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.025], wspace=0.07,
                          left=0.035, right=0.955, top=0.83, bottom=0.145)
    axes = [fig.add_subplot(gs[0, i], projection=ccrs.PlateCarree()) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])
    cf = None
    for ax in axes:
        ax.set_extent([x0, x1, y0, y1], crs=ccrs.PlateCarree())
        ax.set_aspect("auto")
        cf = ax.pcolormesh(lon_s, lat_s, dbz_m, cmap=cmap, norm=norm, alpha=0.7,
                           transform=ccrs.PlateCarree(), zorder=2, shading="auto")
        ax.add_feature(states, edgecolor="gray", facecolor="none", linewidth=0.5, zorder=3)
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.0)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 9}
        gl.ylabel_style = {"size": 9}

    fire = "firebrick"
    # (a) previous, (b) current: MCS mask (light fill + outline)
    draw_fill(axes[0], prev_s, fire, 0.22, 4)
    draw_mask_outline(axes[0], prev_s, fire, zorder=5)
    draw_fill(axes[1], curr_s, fire, 0.22, 4)
    draw_mask_outline(axes[1], stand_mask_s, "royalblue", zorder=4.4, halo=True)
    draw_mask_outline(axes[1], curr_s, fire, zorder=5)
    # (c) difference
    draw_fill(axes[2], recovered_s, "gold", 0.85, 4)
    draw_fill(axes[2], only_prev_stand_s, "royalblue", 0.9, 4.2)
    draw_fill(axes[2], only_prev_none_s, "k", 0.9, 4.2)
    draw_fill(axes[2], only_curr_s, "magenta", 0.9, 4.2)
    draw_mask_outline(axes[2], prev_s, "white", zorder=5, halo=True)
    draw_mask_outline(axes[2], curr_s, fire, zorder=6, halo=False)

    a_prev, a_curr = metrics["prev_px"] * PIXEL_KM2, metrics["curr_px"] * PIXEL_KM2
    axes[0].set_title(f"(a) Previous code: MCS mask {metrics['prev_px']:,} px ({a_prev:,.0f} km$^2$)", fontsize=12)
    axes[1].set_title(f"(b) Current code: MCS mask {metrics['curr_px']:,} px ({a_curr:,.0f} km$^2$)\n"
                      f"blue outline = fragments now standalone tracks", fontsize=12)
    axes[2].set_title("(c) Difference: current $-$ previous", fontsize=12)
    handles = [
        Patch(facecolor="royalblue", label=f"only in previous mask, now a standalone track: {metrics['only_prev_standalone_px']:,} px"),
        Patch(facecolor="k", label=f"only in previous mask, now in no track: {metrics['only_prev_none_px']:,} px"),
        Patch(facecolor="magenta", label=f"only in current mask: {metrics['only_curr_px']:,} px"),
        Patch(facecolor="gold", label=f"no track before, standalone track now: {metrics['recovered_in_window_px']:,} px"),
        Patch(facecolor="none", edgecolor="white", label="previous mask outline"),
        Patch(facecolor="none", edgecolor=fire, label="current mask outline"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10, framealpha=0.95, facecolor="w",
               bbox_to_anchor=(0.46, 0.0))
    cb = fig.colorbar(cf, cax=cax, ticks=levels[::2], extend="both")
    cb.set_label("Composite reflectivity (dBZ), alpha=0.7", fontsize=10)
    fig.suptitle(f"{title_extra}   {timestr} UTC     "
                 f"(feature pixels with no track, whole domain: previous {metrics['orphan_prev_px']:,} -> current "
                 f"{metrics['orphan_curr_px']:,})", fontsize=14, y=0.965)
    fig.savefig(figname, dpi=120)
    plt.close(fig)
    return metrics


###############################################################################
# Summary figures
###############################################################################
def make_timeseries(rows, ser_old, ser_new, tn_new, tn_old, outfile):
    import matplotlib
    matplotlib.use("agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    t = np.array([datetime.datetime.strptime(r["time"], "%Y-%m-%d %H:%M:%S") for r in rows])
    fig, ax = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    kp, kc = "#1f77b4", "#d62728"
    ax[0].plot(t, [r["prev_px"] * PIXEL_KM2 / 1e3 for r in rows], color=kp, lw=2, label="previous code")
    ax[0].plot(t, [r["curr_px"] * PIXEL_KM2 / 1e3 for r in rows], color=kc, lw=2, label="current code")
    ax[0].plot(t, [r["prev_own_px"] * PIXEL_KM2 / 1e3 for r in rows], color=kp, lw=1.5, ls=":",
               label="previous code, own features only (var tracknumber, as in the earlier plot)")
    ax[0].set_ylabel("MCS mask area\n(10$^3$ km$^2$)")
    ax[0].set_ylim(0, 1.3 * max(r["curr_px"] for r in rows) * PIXEL_KM2 / 1e3)
    ax[0].set_title(f"MCS {tn_new} (previous-code track number {tn_old}): pixel-level mask and attached fragments")
    ax[0].legend(loc="upper left")
    ax[1].bar(t, [-r["only_prev_standalone_px"] * PIXEL_KM2 / 1e3 for r in rows], width=0.0035, color="royalblue",
              label="only in previous (now a standalone track)")
    ax[1].bar(t, [-r["only_prev_none_px"] * PIXEL_KM2 / 1e3 for r in rows], width=0.0035, color="k",
              bottom=[-r["only_prev_standalone_px"] * PIXEL_KM2 / 1e3 for r in rows], label="only in previous (now in no track)")
    ax[1].bar(t, [r["only_curr_px"] * PIXEL_KM2 / 1e3 for r in rows], width=0.0035, color="magenta", label="only in current")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].margins(y=0.12)
    ax[1].set_ylabel("mask difference\n(10$^3$ km$^2$)")
    ax[1].legend(loc="lower left")
    tt = np.array([datetime.datetime.utcfromtimestamp(float(x)) for x in ser_new["bt"]])
    ax[2].step(tt, ser_old["attached_n"], where="post", color=kp, lw=2, label="previous code")
    ax[2].step(tt, ser_new["attached_n"], where="post", color=kc, lw=2, label="current code")
    ax[2].set_ylabel("merge/split features\nattached to the track")
    ax[2].legend(loc="upper left")
    ax[3].plot(t, [r["orphan_prev_px"] * PIXEL_KM2 / 1e3 for r in rows], color=kp, lw=2, label="previous code")
    ax[3].plot(t, [r["orphan_curr_px"] * PIXEL_KM2 / 1e3 for r in rows], color=kc, lw=2, label="current code")
    ax[3].set_ylabel("feature area in NO track\n(whole domain, 10$^3$ km$^2$)")
    ax[3].legend(loc="upper left")
    ax[3].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax[3].set_xlabel("UTC")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outfile, dpi=140)
    plt.close(fig)


def make_montage(frame_pngs, outfile, scale=0.55):
    from PIL import Image
    ims = [Image.open(p).convert("RGB") for p in frame_pngs]
    ims = [im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS) for im in ims]
    w = max(im.width for im in ims)
    h = sum(im.height for im in ims)
    canvas = Image.new("RGB", (w, h), "white")
    y = 0
    for im in ims:
        canvas.paste(im, (0, y))
        y += im.height
    canvas.save(outfile)


###############################################################################
# All MCS tracks: how much do the attached fragments change?
###############################################################################
def all_mcs_summary(old, new):
    out = []
    mcs_new = np.where(new["is_mcs"].values == 1)[0]
    for k in mcs_new:
        dur = int(new["track_duration"].values[k])
        try:
            j = find_old_index(new, old, k)
        except RuntimeError:
            continue
        so, sn = attached_series(old, j, dur), attached_series(new, k, dur)
        out.append({
            "track_new": int(k) + 1, "track_old": j + 1, "steps": dur,
            "attached_pts_old": int(so["attached_n"].sum()), "attached_pts_new": int(sn["attached_n"].sum()),
            "attached_area_old_km2": float(so["attached_area"].sum()), "attached_area_new_km2": float(sn["attached_area"].sum()),
        })
    return out


###############################################################################
# What would the Tb-style attach logic (identifymcs.py / link_mergesplit_tracks.py) do on this data?
###############################################################################
def tb_style_quantification():
    sys.path.insert(0, f"{M}/code_new")
    from pyflextrkr.ft_utilities import load_sparse_trackstats
    from pyflextrkr import identifymcs_radar as im
    f = f"{M}/new/stats/trackstats_sparse_{DATES}.nc"
    ds1d, attrs, sp, maxdur = load_sparse_trackstats(300, f, "times_indices", "tracks", "tracks_indices")
    n = ds1d.sizes["tracks"]
    tr = 0.083333
    dur = ds1d["track_duration"].values
    life = dur * tr
    em = np.nan_to_num(ds1d["end_merge_tracknumber"].values.astype(float))
    ss = np.nan_to_num(ds1d["start_split_tracknumber"].values.astype(float))
    maj, d45, bt = sp["majoraxis"].toarray(), sp["area_45dbz"].toarray(), sp["base_time"].toarray()
    params = dict(mcs_majoraxis_thresh=100, mcs_dbz45_area_thresh=0, mcs_duration_thresh=4, mcs_gap_hour=0.25)
    is_mcs = np.array([im.identify_mcs_periods(maj[i, :int(dur[i])], d45[i, :int(dur[i])], bt[i, :int(dur[i])], tr, params)[1]
                       if dur[i] > 0 else False for i in range(n)])
    thr = 4.0
    par, lt, isf = im.find_mergesplit_parents(life, em, ss, is_mcs, thr, thr)
    root, isfr = im.resolve_fragment_roots(par, isf)
    res = {"n_tracks": int(n), "n_mcs": int(is_mcs.sum()), "threshold_hours": thr}
    # Tb-style: only direct children of an MCS (no chain following), only coexisting time points attached
    direct = {"fragments": 0, "fully": 0, "partial": 0, "never": 0, "pts_attached": 0, "pts_not_coexisting": 0}
    for i in np.where(lt != "")[0]:
        p = int(par[i])
        if not is_mcs[p]:
            continue
        direct["fragments"] += 1
        pb = bt[p, :int(dur[p])]
        ib = bt[i, :int(dur[i])]
        m = np.isin(ib, pb)
        direct["pts_attached"] += int(m.sum())
        direct["pts_not_coexisting"] += int((~m).sum())
        direct["fully" if m.all() else ("partial" if m.any() else "never")] += 1
    res["direct_children_of_mcs"] = direct
    # chain members: root is an MCS but the direct parent is another fragment (never attached by the Tb-style logic)
    chain = {"fragments": 0, "points": 0}
    for i in np.where(lt != "")[0]:
        r = int(root[i])
        if is_mcs[r] and int(par[i]) != r:
            chain["fragments"] += 1
            chain["points"] += int(dur[i])
    res["chain_members_rooted_at_mcs"] = chain
    return res


###############################################################################
def main():
    ap = argparse.ArgumentParser(description="Previous vs current merge/split code: pixel masks of one MCS track")
    ap.add_argument("--workdir", required=True, help="work directory made by setup_comparison.py")
    ap.add_argument("--outdir", required=True, help="output directory (a track<TN> folder is made inside)")
    ap.add_argument("--tn", type=int, required=True, help="MCS track number in the CURRENT code's stats")
    ap.add_argument("--workers", type=int, default=64, help="Dask workers for the frames")
    ap.add_argument("--pixel-km2", type=float, default=9.0, help="area of a grid cell [km^2]")
    ap.add_argument("--ref-time", default=None,
                    help="optional YYYY-MM-DDTHH:MM:SS: also put the frame nearest to this time in the key-frame montage")
    ap.add_argument("--max_frames", type=int, default=0, help="debug: limit number of frames")
    args = ap.parse_args()
    configure(args.workdir, args.pixel_km2)

    outdir = os.path.join(args.outdir, f"track{args.tn}")
    framedir = os.path.join(outdir, "frames")
    os.makedirs(framedir, exist_ok=True)

    old, new = open_stats("old"), open_stats("new")
    k_new = args.tn - 1
    dur = int(new["track_duration"].values[k_new])
    assert int(new["is_mcs"].values[k_new]) == 1, "requested track is not an MCS in the current stats"
    k_old = find_old_index(new, old, k_new)
    tn_old = k_old + 1
    bt = new["base_time"].values[k_new, :dur]
    print(f"track {args.tn} (current) = track {tn_old} (previous); {dur} steps, {hhmm(bt[0])} -> {hhmm(bt[-1])} UTC", flush=True)

    ser_old, ser_new = attached_series(old, k_old, dur), attached_series(new, k_new, dur)
    ser_old["bt"] = ser_new["bt"] = bt
    print("attached fragment features over the life: previous", int(ser_old["attached_n"].sum()),
          "| current", int(ser_new["attached_n"].sum()), flush=True)

    # pixel files of each version for the track's times
    def file_index(version):
        fl = sorted(glob.glob(f"{M}/{version}/{PIX_DIR}{PIX_BASE}*.nc"))
        ep = np.array([calendar.timegm(time.strptime(stamp(p), "%Y%m%d_%H%M%S")) for p in fl], dtype=float)
        return fl, ep
    fo, eo = file_index("old")
    fn, en = file_index("new")
    jobs = []
    for i, t in enumerate(bt):
        po, pn = pixel_file_for(t, eo, fo), pixel_file_for(t, en, fn)
        if po is None or pn is None:
            print("no pixel file for", hhmm(t), flush=True)
            continue
        png = os.path.join(framedir, f"mcs{args.tn}_{stamp(pn)}.png")
        jobs.append((po, pn, png))
    if args.max_frames:
        jobs = jobs[:args.max_frames]
    print(f"{len(jobs)} frames -> {framedir}", flush=True)

    from dask.distributed import Client, LocalCluster, wait
    cluster = LocalCluster(n_workers=args.workers, threads_per_worker=1)
    client = Client(cluster)
    title = f"MCS {args.tn}: previous vs current merge/split code"
    t0 = time.time()
    futs = [client.submit(frame_task, po, pn, tn_old, args.tn, png, title, pure=False) for po, pn, png in jobs]
    wait(futs, timeout=300)
    rows = []
    for f, (po, pn, png) in zip(futs, jobs):
        if f.status == "finished":
            rows.append(f.result())
        else:
            # a worker task hung or failed (seen once for a frame that renders fine alone): redo it here
            print(f"task not finished ({f.status}), redoing serially: {os.path.basename(png)}", flush=True)
            client.cancel(f)
            rows.append(frame_task(po, pn, tn_old, args.tn, png, title))
    print(f"frames done in {time.time() - t0:.0f} s", flush=True)
    client.close()
    cluster.close()
    rows = [r for r in rows if "prev_px" in r]

    # ---- summaries ----
    import csv
    with open(os.path.join(outdir, "frame_metrics.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    make_timeseries(rows, ser_old, ser_new, args.tn, tn_old, os.path.join(outdir, f"timeseries_track{args.tn}.png"))

    # key frames: biggest loss from the mask, biggest recovery of 'no track' pixels near the MCS, and (optional) a reference time
    pngs_by_time = {stamp(p): p for _, _, p in jobs}
    key = {}
    key["largest_mask_loss"] = max(rows, key=lambda r: r["only_prev_px"])
    key["largest_recovery"] = max(rows, key=lambda r: r["recovered_in_window_px"])
    if args.ref_time:
        t_ref = datetime.datetime.fromisoformat(args.ref_time)
        key["reference_time"] = min(rows, key=lambda r: abs(datetime.datetime.strptime(r["time"], "%Y-%m-%d %H:%M:%S") - t_ref))
    chosen = []
    for name, r in key.items():
        tag = r["time"].replace("-", "").replace(":", "").replace(" ", "_")
        if tag in pngs_by_time and pngs_by_time[tag] not in chosen:
            chosen.append(pngs_by_time[tag])
    make_montage(chosen, os.path.join(outdir, f"montage_keyframes_track{args.tn}.png"))

    # mp4 (ffmpeg from the running environment, else from PATH; skipped if there is none)
    import shutil
    ffmpeg = os.path.join(os.path.dirname(sys.executable), "ffmpeg")
    if not os.path.isfile(ffmpeg):
        ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        lst = os.path.join(outdir, "frames_list.txt")
        with open(lst, "w") as fh:
            for _, _, png in sorted(jobs, key=lambda x: x[2]):
                fh.write(f"file '{png}'\n")
        mp4 = os.path.join(outdir, f"mcs{args.tn}_previous_vs_current.mp4")
        os.system(f"{ffmpeg} -loglevel error -y -f concat -safe 0 -r 6 -i {lst} -vf scale=2100:-2 "
                  f"-c:v libx264 -r 12 -crf 21 -pix_fmt yuv420p {mp4}")
    else:
        print("ffmpeg not found: skipping the mp4", flush=True)

    summary = {
        "track_current": args.tn, "track_previous": tn_old, "steps": dur,
        "start": hhmm(bt[0]), "end": hhmm(bt[-1]), "n_frames": len(rows),
        "mask_px_previous_mean": float(np.mean([r["prev_px"] for r in rows])),
        "mask_px_current_mean": float(np.mean([r["curr_px"] for r in rows])),
        "frames_with_only_prev": int(sum(r["only_prev_px"] > 0 for r in rows)),
        "frames_with_only_curr": int(sum(r["only_curr_px"] > 0 for r in rows)),
        "max_only_prev_px": int(max(r["only_prev_px"] for r in rows)),
        "only_prev_px_total_frames": int(sum(r["only_prev_px"] for r in rows)),
        "only_prev_standalone_px_total_frames": int(sum(r["only_prev_standalone_px"] for r in rows)),
        "only_prev_none_px_total_frames": int(sum(r["only_prev_none_px"] for r in rows)),
        "prev_px_total_frames": int(sum(r["prev_px"] for r in rows)),
        "curr_px_total_frames": int(sum(r["curr_px"] for r in rows)),
        "prev_own_px_total_frames": int(sum(r["prev_own_px"] for r in rows)),
        "curr_own_px_total_frames": int(sum(r["curr_own_px"] for r in rows)),
        "max_only_curr_px": int(max(r["only_curr_px"] for r in rows)),
        "max_recovered_in_window_px": int(max(r["recovered_in_window_px"] for r in rows)),
        "domain_orphan_px_mean_previous": float(np.mean([r["orphan_prev_px"] for r in rows])),
        "domain_orphan_px_mean_current": float(np.mean([r["orphan_curr_px"] for r in rows])),
        "attached_features_total_previous": int(ser_old["attached_n"].sum()),
        "attached_features_total_current": int(ser_new["attached_n"].sum()),
        "key_frames": {k: v["time"] for k, v in key.items()},
        "all_mcs": all_mcs_summary(old, new),
        "tb_style_logic_on_this_data": tb_style_quantification(),
    }
    with open(os.path.join(outdir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "all_mcs"}, indent=2), flush=True)
    print("DONE", outdir, flush=True)


if __name__ == "__main__":
    main()
