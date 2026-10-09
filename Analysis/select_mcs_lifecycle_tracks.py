#!/usr/bin/env python
"""
Select MCS tracks of a region for plot_mcs_radar_lifecycle.py: from several monthly robust stats files (radar_feature
tracking), take every MCS whose lifecycle was evaluated (is_mcs == 1 and lifecycle_evaluated == 1) with its position at
the time of its largest area inside a lon/lat box, sort them into the lifecycle groups of the plot script (group_of), and
pick --per-group of each group at random (fixed --seed; all of them if a group has fewer).

The CSV lists every MCS of the box with the column "selected" (1 = to be plotted); the counts per group and month are
printed. Track numbers repeat across months, so a track is identified by (month, track).

usage:
  select_mcs_lifecycle_tracks.py --configs cfg_202505.yaml cfg_202506.yaml ... --out selection.csv \
      [--box LON_MIN LON_MAX LAT_MIN LAT_MAX] [--per-group 50] [--seed 0]
The default box is the "Southeast US region" of the CONUS animations (BNF site 34.723 N, 273.428 E, +-6 deg lon, +-4 deg lat).
Then, for each month (see plot_mcs_radar_lifecycle.py):
  plot_mcs_radar_lifecycle.py --config cfg_YYYYMM.yaml --outdir DIR --prefix YYYYMM_ --tracks <selected tracks of YYYYMM>
"""
import csv
import argparse
import yaml
import numpy as np
import xarray as xr
from plot_mcs_radar_lifecycle import GROUPS, paths_from_config, load_track, group_of, utc

BNF_LON, BNF_LAT = 273.428 - 360.0, 34.723


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--configs", nargs="+", required=True, help="one MCS tracking yaml per month")
    ap.add_argument("--out", required=True, help="CSV of the MCS in the box")
    ap.add_argument("--box", type=float, nargs=4, metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
                    default=[BNF_LON - 6.0, BNF_LON + 6.0, BNF_LAT - 4.0, BNF_LAT + 4.0],
                    help="region [deg, lon -180..180] (default: BNF +-6 lon, +-4 lat)")
    ap.add_argument("--per-group", type=int, default=50, help="tracks to select per group (default 50)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--peak-var", default="area_45dbz", help="as in plot_mcs_radar_lifecycle.py (default area_45dbz)")
    args = ap.parse_args()
    lon0, lon1, lat0, lat1 = args.box

    rows = []
    for cfg in args.configs:
        paths = paths_from_config(cfg)
        month = str(yaml.full_load(open(cfg))["startdate"])[:6]
        with xr.open_dataset(paths["stats"], mask_and_scale=True, decode_times=False) as ds:
            cand = np.where((ds["is_mcs"].values == 1) & (ds["lifecycle_evaluated"].values == 1))[0] + 1
            n_box = 0
            for tn in cand:
                t = load_track(ds, int(tn), args.peak_var, paths["smooth_hour"], paths["sf_area_min"])
                if not np.isfinite(t["area"]).any():
                    continue
                ip = int(np.nanargmax(t["area"]))
                lon, lat = float(t["lon"][ip]), float(t["lat"][ip])
                if not (lon0 <= lon <= lon1 and lat0 <= lat <= lat1):
                    continue
                n_box += 1
                rows.append({"month": month, "track": int(tn), "group": group_of(t), "lon_at_max_area": round(lon, 3),
                             "lat_at_max_area": round(lat, 3), "time_max_area_utc": utc(t["bt"][ip]).strftime("%Y-%m-%d %H:%M"),
                             "start_utc": utc(t["bt"][0]).strftime("%Y-%m-%d %H:%M"), "hours": round(t["dur"] * t["dt_h"], 2),
                             "selected": 0, "seed": args.seed})
        print(f"{month}: {len(cand)} evaluated MCS, {n_box} in the box", flush=True)

    rows.sort(key=lambda r: (r["month"], r["track"]))
    for g in GROUPS:
        idx = [i for i, r in enumerate(rows) if r["group"] == g]
        pick = np.random.default_rng(args.seed).choice(len(idx), size=min(args.per_group, len(idx)), replace=False)
        for j in pick:
            rows[idx[j]]["selected"] = 1
    rows.sort(key=lambda r: (GROUPS.index(r["group"]), r["month"], r["track"]))
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    months = sorted({r["month"] for r in rows})
    print(f"\n{'group':32s} {'in box':>7s} {'selected':>9s}   selected per month " + " ".join(months))
    for g in GROUPS:
        gr = [r for r in rows if r["group"] == g]
        if gr:
            sel = [r for r in gr if r["selected"]]
            print(f"{g:32s} {len(gr):7d} {len(sel):9d}   " + " " * 19 + " ".join(f"{sum(r['month'] == m for r in sel):6d}" for m in months))
    print("selection:", args.out)


if __name__ == "__main__":
    main()
