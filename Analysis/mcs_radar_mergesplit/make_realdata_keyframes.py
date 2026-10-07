#!/usr/bin/env python
"""
Real-data illustration of what the current identifymcs_radar folds into an MCS and what it does not.
One row per frame, two panels per row (reflectivity underneath, alpha 0.7):

  left   "pieces"  root MCS feature (dark red); every fragment folded into the MCS, one outline each (blue);
                   every fragment NOT folded, one outline each (orange). A fragment is not folded when some of its
                   time points lie outside the lifetime of the MCS it merges into / splits from.
  right  "result"  the final MCS mask = root + folded fragments (dark red); the fragments that were not folded,
                   outlined as the standalone tracks they stay (orange).

Folded fragments are the cloud numbers attached to the MCS in the CURRENT stats; not-folded ones are the cloud numbers
attached in the PREVIOUS stats but not in the current ones (both are matched to feature_number in the pixel files).

usage: make_realdata_keyframes.py --workdir DIR --outdir DIR [--frame TN TIME ...]
       --workdir  the directory made by setup_comparison.py and filled by run_tree.py (old/ and new/ roots)
       --frame    MCS track number in the CURRENT stats and a time (YYYY-MM-DDTHH:MM:SS, nearest time step of the
                  track) or "auto" = the time with the largest attached-fragment area that the current code does not fold.
                  Repeat for several rows. Without --frame the three frames used for the PR are made.
"""
import os
import sys
import json
import glob
import time
import calendar
import argparse
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import binary_erosion, binary_dilation

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze_mask_difference as amd   # shared helpers: stats lookup, old/new track matching, file naming

ROOT_COLOR, FOLD_COLOR, UNFOLD_COLOR = "#8b0000", "royalblue", "darkorange"
DEFAULT_FRAMES = [("4525", "2025-05-02T19:00:43"), ("4525", "2025-05-02T15:20:39"), ("12202", "auto")]


def label_edges(lab):
    """Outline pixels of every labelled region separately: labelled pixels with a 4-neighbour of another label."""
    p = np.pad(lab, 1, mode="constant", constant_values=0)
    c = p[1:-1, 1:-1]
    edge = np.zeros(c.shape, dtype=bool)
    for sl in ((slice(0, -2), slice(1, -1)), (slice(2, None), slice(1, -1)),
               (slice(1, -1), slice(0, -2)), (slice(1, -1), slice(2, None))):
        edge |= p[sl] != c
    return edge & (c > 0)


def file_index(version):
    files = sorted(glob.glob(f"{amd.M}/{version}/{amd.PIX_DIR}{amd.PIX_BASE}*.nc"))
    epoch = np.array([calendar.timegm(time.strptime(amd.stamp(p), "%Y%m%d_%H%M%S")) for p in files], dtype=float)
    return files, epoch


def attached_cloudnumbers(ds, k, t):
    out = []
    for kind in ("merge", "split"):
        out += [int(v) for v in ds[f"{kind}_cloudnumber"].values[k, t, :] if v > 0]
    return out


def frame_data(old, new, tn, when, index):
    """Everything one row needs: windowed arrays and the counts."""
    k_new = tn - 1
    dur = int(new["track_duration"].values[k_new])
    if int(new["is_mcs"].values[k_new]) != 1:
        sys.exit(f"track {tn} is not an MCS in the current stats")
    k_old = amd.find_old_index(new, old, k_new)
    bt = new["base_time"].values[k_new, :dur]
    s_old, s_new = amd.attached_series(old, k_old, dur), amd.attached_series(new, k_new, dur)
    if when == "auto":
        t = int(np.argmax(s_old["attached_area"] - s_new["attached_area"]))
    else:
        target = calendar.timegm(time.strptime(when, "%Y-%m-%dT%H:%M:%S"))
        t = int(np.argmin(np.abs(bt - target)))
        if abs(bt[t] - target) > 600:
            print(f"warning: nearest time step of track {tn} is {amd.hhmm(bt[t])}, more than 10 min from {when}")
    folded = attached_cloudnumbers(new, k_new, t)
    unfolded = [c for c in attached_cloudnumbers(old, k_old, t) if c not in set(folded)]
    own_cn = int(new["cloudnumber"].values[k_new, t])

    po = amd.pixel_file_for(bt[t], index["old"][1], index["old"][0])
    pn = amd.pixel_file_for(bt[t], index["new"][1], index["new"][0])
    if po is None or pn is None:
        sys.exit(f"no pixel file for track {tn} at {amd.hhmm(bt[t])}")
    dn = xr.open_dataset(pn)
    lon = ((dn["longitude"].values + 180.0) % 360.0) - 180.0
    lat = dn["latitude"].values
    dbz = dn["dbz_comp"].values.squeeze()
    fn = np.nan_to_num(dn["feature_number"].values.squeeze(), nan=0).astype(np.int64)
    ctn = np.nan_to_num(dn["cloudtracknumber"].values.squeeze(), nan=0).astype(np.int64)
    timestr = str(dn["time"].values.squeeze())[:19].replace("T", " ")

    root = fn == own_cn
    lab_fold = np.where(np.isin(fn, folded), fn, 0)
    lab_unfold = np.where(np.isin(fn, unfolded), fn, 0)
    final = ctn == tn
    stand_ids = np.unique(ctn[lab_unfold > 0])
    stand_ids = stand_ids[(stand_ids > 0) & (stand_ids != tn)]
    lab_stand = np.where(np.isin(ctn, stand_ids), ctn, 0)
    px = amd.PIXEL_KM2
    counts = {
        "track": tn, "track_previous": k_old + 1, "time": timestr,
        "root_px": int(root.sum()), "n_folded": len(folded), "folded_px": int((lab_fold > 0).sum()),
        "n_unfolded": len(unfolded), "unfolded_px": int((lab_unfold > 0).sum()),
        "final_px": int(final.sum()), "n_standalone_tracks": int(len(stand_ids)),
        # the final mask must be exactly root + folded fragments
        "final_minus_pieces_px": int((final & ~(root | (lab_fold > 0))).sum() + ((root | (lab_fold > 0)) & ~final).sum()),
    }
    union = root | (lab_fold > 0) | (lab_unfold > 0) | final
    lon_u, lat_u = lon[union], lat[union]
    lat_c, lon_c = 0.5 * (lat_u.min() + lat_u.max()), 0.5 * (lon_u.min() + lon_u.max())
    coslat = np.cos(np.deg2rad(lat_c))
    half_h = max(1.6, 0.5 * (lat_u.max() - lat_u.min()) + 0.5)
    half_w = max(half_h / coslat, 0.5 * (lon_u.max() - lon_u.min()) + 0.5)
    half_h = max(half_h, half_w * coslat)
    ext = (lon_c - half_w, lon_c + half_w, lat_c - half_h, lat_c + half_h)
    inwin = (lon >= ext[0]) & (lon <= ext[1]) & (lat >= ext[2]) & (lat <= ext[3])
    rows, cols = np.where(inwin)
    sl = (slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1))
    return {
        "lon": lon[sl], "lat": lat[sl], "dbz": dbz[sl], "ext": ext, "counts": counts, "px_km2": px,
        "root": root[sl], "lab_fold": lab_fold[sl], "lab_unfold": lab_unfold[sl], "final": final[sl], "lab_stand": lab_stand[sl],
    }


def draw_edges(ax, d, edge, color, zorder, halo=True, alpha=1.0):
    if not edge.any():
        return
    if halo:
        h = binary_dilation(edge, iterations=1) & ~edge
        ax.pcolormesh(d["lon"], d["lat"], np.ma.masked_where(~h, np.ones(h.shape)), cmap=ListedColormap(["k"]),
                      vmin=0, vmax=1, alpha=0.85, transform=ccrs.PlateCarree(), zorder=zorder, shading="auto")
    ax.pcolormesh(d["lon"], d["lat"], np.ma.masked_where(~edge, np.ones(edge.shape)), cmap=ListedColormap([color]),
                  vmin=0, vmax=1, alpha=alpha, transform=ccrs.PlateCarree(), zorder=zorder + 0.1, shading="auto")


def draw_fill(ax, d, mask, color, alpha, zorder):
    if mask.any():
        ax.pcolormesh(d["lon"], d["lat"], np.ma.masked_where(~mask, np.ones(mask.shape)), cmap=ListedColormap([color]),
                      vmin=0, vmax=1, alpha=alpha, transform=ccrs.PlateCarree(), zorder=zorder, shading="auto")


def render(frames, outfile):
    n = len(frames)
    levels = np.arange(-10, 70.1, 5)
    cmap = plt.get_cmap("gist_ncar")
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    states = cfeature.NaturalEarthFeature("cultural", "admin_1_states_provinces_lakes", "50m")
    fig = plt.figure(figsize=(15.5, 5.9 * n + 1.5), dpi=130, facecolor="w")
    gs = fig.add_gridspec(n, 2, wspace=0.07, hspace=0.26, left=0.045, right=0.925,
                          top=1 - 1.25 / (5.9 * n + 1.5), bottom=0.9 / (5.9 * n + 1.5) + 0.02)
    cf = None
    for i, d in enumerate(frames):
        c = d["counts"]
        km2 = d["px_km2"]
        axl = fig.add_subplot(gs[i, 0], projection=ccrs.PlateCarree())
        axr = fig.add_subplot(gs[i, 1], projection=ccrs.PlateCarree())
        for ax in (axl, axr):
            ax.set_extent(d["ext"], crs=ccrs.PlateCarree())
            ax.set_aspect("auto")
            dbz = np.ma.masked_where(~np.isfinite(d["dbz"]) | (d["dbz"] < levels[0]), d["dbz"])
            cf = ax.pcolormesh(d["lon"], d["lat"], dbz, cmap=cmap, norm=norm, alpha=0.7,
                               transform=ccrs.PlateCarree(), zorder=2, shading="auto")
            ax.add_feature(states, edgecolor="gray", facecolor="none", linewidth=0.5, zorder=3)
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.0)
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {"size": 9}
            gl.ylabel_style = {"size": 9}
        # left: the pieces, every fragment outlined separately
        draw_edges(axl, d, label_edges(d["lab_unfold"]), UNFOLD_COLOR, 5)
        draw_edges(axl, d, label_edges(d["lab_fold"]), FOLD_COLOR, 6)
        draw_edges(axl, d, d["root"] & ~binary_erosion(d["root"]), ROOT_COLOR, 7)
        # right: the result
        draw_fill(axr, d, d["final"], ROOT_COLOR, 0.18, 4)
        draw_edges(axr, d, label_edges(d["lab_stand"]), UNFOLD_COLOR, 5)
        draw_edges(axr, d, d["final"] & ~binary_erosion(d["final"]), ROOT_COLOR, 7)
        axl.set_title(f"MCS {c['track']}   {c['time']} UTC", loc="left", fontsize=13, fontweight="bold", pad=12)
        axr.set_title(f"final mask {c['final_px'] * km2:,.0f} km$^2$ = root {c['root_px'] * km2:,.0f} + "
                      f"{c['n_folded']} folded fragments {c['folded_px'] * km2:,.0f}\n"
                      f"{c['n_unfolded']} not-folded fragments ({c['unfolded_px'] * km2:,.0f} km$^2$) stay standalone",
                      loc="left", fontsize=10, pad=8)
        if i == 0:
            axl.annotate("Pieces: root MCS + fragments (folded / not folded)",
                         xy=(0.5, 1), xycoords="axes fraction", xytext=(0, 42), textcoords="offset points",
                         ha="center", fontsize=13, fontweight="bold")
            axr.annotate("Result: final MCS mask + standalone fragments",
                         xy=(0.5, 1), xycoords="axes fraction", xytext=(0, 42), textcoords="offset points",
                         ha="center", fontsize=13, fontweight="bold")
    cax = fig.add_axes([0.94, 0.18, 0.012, 0.62])
    cb = fig.colorbar(cf, cax=cax, ticks=levels[::2], extend="both")
    cb.set_label("Composite reflectivity (dBZ), alpha=0.7", fontsize=10)
    handles = [
        Patch(facecolor="none", edgecolor=ROOT_COLOR, linewidth=2.5, label="root MCS feature (left) / final MCS mask = root + folded fragments (right)"),
        Patch(facecolor="none", edgecolor=FOLD_COLOR, linewidth=2.5, label="fragment folded into the MCS (its times all lie inside the MCS lifetime)"),
        Patch(facecolor="none", edgecolor=UNFOLD_COLOR, linewidth=2.5, label="fragment NOT folded (some of its times lie outside the MCS lifetime): stays a standalone track"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=1, fontsize=10.5, frameon=False,
               bbox_to_anchor=(0.5, 0.004))
    fig.savefig(outfile, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Real-data figure: fragments folded into an MCS vs fragments that stay standalone")
    ap.add_argument("--workdir", required=True, help="directory made by setup_comparison.py and filled by run_tree.py")
    ap.add_argument("--outdir", required=True, help="output directory")
    ap.add_argument("--frame", nargs=2, action="append", metavar=("TN", "TIME"),
                    help="MCS track number (current stats) and time (YYYY-MM-DDTHH:MM:SS or auto); repeat per row")
    ap.add_argument("--pixel-km2", type=float, default=9.0, help="area of a grid cell [km^2]")
    args = ap.parse_args()
    amd.configure(args.workdir, args.pixel_km2)
    os.makedirs(args.outdir, exist_ok=True)

    old, new = amd.open_stats("old"), amd.open_stats("new")
    index = {v: file_index(v) for v in ("old", "new")}
    frames, summary = [], []
    for tn, when in (args.frame or DEFAULT_FRAMES):
        d = frame_data(old, new, int(tn), when, index)
        frames.append(d)
        summary.append(d["counts"])
        print(json.dumps(d["counts"]), flush=True)
        if d["counts"]["final_minus_pieces_px"] != 0:
            print("  warning: the final mask is not exactly root + folded fragments", flush=True)
    render(frames, os.path.join(args.outdir, "realdata_folded_vs_unfolded.png"))
    for d in frames:
        c = d["counts"]
        stamp = c["time"].replace("-", "").replace(":", "").replace(" ", "_")
        render([d], os.path.join(args.outdir, f"realdata_mcs{c['track']}_{stamp}.png"))
    with open(os.path.join(args.outdir, "realdata_frames.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("DONE", args.outdir)


if __name__ == "__main__":
    main()
