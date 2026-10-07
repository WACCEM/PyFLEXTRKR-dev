"""
Build the synthetic track statistics and run ONE module of ONE code tree on them.

usage: synthetic_run.py <code_tree> <radar|tb|linkms> <outdir>

The same input is used for every module/version, so differences in the output are caused
by the code only. Time step = 5 min; cloudnumber = 1000 * track number + time step, so every
feature can be traced back to the track it came from.

Tracks (track number = index + 1), times in steps:
  1  M   MCS, steps  30-125 (8 h)
  2  F1  ends by merging into M,  steps  60-80   (inside M's lifetime)
  3  F2  ends by merging into M,  steps  12-40   (starts 18 steps before M exists)
  4  F3  starts by splitting from M, steps 110-140 (outlives M by 15 steps)
  5  A   ends by merging into B,  steps   0-14   (entirely before M exists)
  6  B   ends by merging into M,  steps  10-34   (starts 20 steps before M exists)
  7  C   ends by merging into D,  steps  90-96   (chain C -> D -> M, entirely inside M's lifetime)
  8  D   ends by merging into M,  steps  85-100  (inside M's lifetime)
"""
import sys
import os
import json
import numpy as np
import xarray as xr

DT_HR = 1.0 / 12.0
T0 = 1746200000.0
DT_SEC = DT_HR * 3600.0
START, END = "20250101.0000", "20250101.1200"

# id: (name, first step, last step, end_merge_tracknumber, start_split_tracknumber)
TRACKS = {
    1: ("M", 30, 125, 0, 0),
    2: ("F1", 60, 80, 1, 0),
    3: ("F2", 12, 40, 1, 0),
    4: ("F3", 110, 140, 0, 1),
    5: ("A", 0, 14, 6, 0),
    6: ("B", 10, 34, 1, 0),
    7: ("C", 90, 96, 8, 0),
    8: ("D", 85, 100, 1, 0),
}
MAXDUR = 150


def build_sparse_file(stats_path):
    ntracks = len(TRACKS)
    rows, cols = [], []
    dur = np.zeros(ntracks, dtype=np.int32)
    for tid, (_, s0, s1, _, _) in TRACKS.items():
        n = s1 - s0 + 1
        dur[tid - 1] = n
        rows += [tid - 1] * n
        cols += list(range(n))
    rows = np.array(rows, dtype=np.int64)
    cols = np.array(cols, dtype=np.int64)

    def per_point(fn):
        vals = np.zeros(len(rows))
        for i, (r, c) in enumerate(zip(rows, cols)):
            tid = r + 1
            vals[i] = fn(tid, TRACKS[tid][1] + c)
        return vals

    is_m = lambda tid: tid == 1
    sparse = {
        "base_time": per_point(lambda tid, step: T0 + step * DT_SEC),
        "cloudnumber": per_point(lambda tid, step: 1000 * tid + step).astype(np.int32),
        "track_status": per_point(lambda tid, step: 1).astype(np.int32),
        "meanlat": per_point(lambda tid, step: 35.0 + 0.001 * step),
        "meanlon": per_point(lambda tid, step: -100.0 + 0.001 * step),
        # radar_feature pipeline
        "majoraxis": per_point(lambda tid, step: 150.0 if is_m(tid) else 30.0),
        "area_45dbz": per_point(lambda tid, step: 50.0 if is_m(tid) else 0.0),
        "ncells": per_point(lambda tid, step: 3 if is_m(tid) else 1).astype(np.int32),
        "conv_area": per_point(lambda tid, step: 50.0 if is_m(tid) else 10.0),
        "conv_majoraxis": per_point(lambda tid, step: 150.0 if is_m(tid) else 20.0),
        "strat_area": per_point(lambda tid, step: 0.0),
        # radar pipeline / link_mergesplit_tracks
        "area": per_point(lambda tid, step: 60100.0 if is_m(tid) else 3050.0),
        # Tb pipeline (identifymcs.py)
        "core_area": per_point(lambda tid, step: 100.0 if is_m(tid) else 50.0),
        "cold_area": per_point(lambda tid, step: 60000.0 if is_m(tid) else 3000.0),
        "tracks_indices": rows,
        "times_indices": cols,
    }
    varlist = {k: (["sparse_index"], v) for k, v in sparse.items()}
    varlist["track_duration"] = (["tracks"], dur)
    varlist["end_merge_tracknumber"] = (["tracks"], np.array([TRACKS[i + 1][3] for i in range(ntracks)], dtype=float))
    varlist["start_split_tracknumber"] = (["tracks"], np.array([TRACKS[i + 1][4] for i in range(ntracks)], dtype=float))
    # variables the Tb identifymcs.py drops explicitly
    for v in ("merge_tracknumbers", "split_tracknumbers", "start_split_timeindex", "end_merge_timeindex"):
        varlist[v] = (["tracks"], np.zeros(ntracks))
    ds = xr.Dataset(varlist, coords={"tracks": np.arange(ntracks), "sparse_index": np.arange(len(rows))})
    os.makedirs(stats_path, exist_ok=True)
    ds.to_netcdf(os.path.join(stats_path, f"trackstats_sparse_{START}_{END}.nc"))


def common_config(stats_path):
    return {
        "trackstats_sparse_filebase": "trackstats_sparse_",
        "stats_outpath": stats_path,
        "startdate": START, "enddate": END,
        "datatimeresolution": DT_HR,
        "duration_range": [2, MAXDUR],
        "tracks_dimname": "tracks", "times_dimname": "times",
        "nmaxlinks": 10, "fillval": -9999,
        "max_speed_for_trajectory_smoothing": 40,
    }


def main():
    tree, which, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    sys.path.insert(0, tree)
    import pyflextrkr
    assert os.path.realpath(pyflextrkr.__file__).startswith(os.path.realpath(tree)), pyflextrkr.__file__
    stats_path = os.path.join(outdir, "stats") + "/"
    build_sparse_file(stats_path)
    cfg = common_config(stats_path)
    if which == "radar":
        from pyflextrkr.identifymcs_radar import identifymcs_radar
        cfg.update({
            # old code reads finalstats_filebase, current code mcsrobust_filebase
            "finalstats_filebase": "mcs_tracks_robust_", "mcsrobust_filebase": "mcs_tracks_robust_",
            "mcs_majoraxis_thresh": 100, "mcs_dbz45_area_thresh": 0, "mcs_duration_thresh": 4,
            "mcs_gap_hour": 0.25, "mcs_lifecycle_line_thresh": 100, "mcs_lifecycle_gap_hour": 1.0,
            "mcs_lifecycle_genesis_lag_hour": 1.0, "mcs_lifecycle_mature_lag_hour": 2.0,
            "mcs_lifecycle_sf_area_min": 0, "mcs_lifecycle_thresh": 100,
            "mcs_merge_duration": 3.0, "mcs_split_duration": 3.0,
            "mcs_dbz45_area_varname": "area_45dbz", "finalstats_min_area": 0, "finalstats_min_duration": None,
        })
        out = identifymcs_radar(cfg)
    elif which == "tb":
        from pyflextrkr.identifymcs import identifymcs_tb
        cfg.update({
            "mcstbstats_filebase": "mcs_tracks_",
            "mcs_tb_area_thresh": 40000, "mcs_tb_duration_thresh": 4, "mcs_tb_gap": 3,
            "mcs_tb_merge_duration": 3.0, "mcs_tb_split_duration": 3.0,
        })
        out = identifymcs_tb(cfg)
    elif which == "linkms":
        from pyflextrkr.link_mergesplit_tracks import link_mergesplit_tracks
        cfg.update({
            "finalstats_filebase": "trackstats_final_",
            "maintrack_area_thresh": 20000, "maintrack_lifetime_thresh": 4,
            "merge_duration": 3.0, "split_duration": 3.0,
        })
        out = link_mergesplit_tracks(cfg)
    else:
        raise SystemExit(f"unknown module {which}")
    print("OUTFILE=" + out)


if __name__ == "__main__":
    main()
