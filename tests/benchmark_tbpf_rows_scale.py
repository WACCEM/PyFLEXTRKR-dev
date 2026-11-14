"""
Time and memory of the row-table Tb+PF steps on a synthetic sparse trackstats file of the size of a global year (about a million tracks), to measure
what the row tables save: the dense Tb step makes [MCS, longest track, nmaxlinks] arrays for the merging and splitting clouds and scans all the tracks for
every MCS. Not a test (no test_ prefix): run it by hand on a compute node.

    python tests/benchmark_tbpf_rows_scale.py WORKDIR [NTRACKS] [dense]     # default 1,000,000 tracks; "dense": the dense step instead of the rows steps

The peak memory is that of the process (ru_maxrss), so the dense step runs in a process of its own.

The tracks are random but have the shape of real ones: durations of a few hours (a long tail up to 400), a few percent MCS-sized, a quarter of the
short tracks merge into or split from another track. identifymcs_tb_rows runs on the sparse file; robustmcs_rows runs on the PF statistics that are made
for the MCS it finds (random PF variables of the shape of matchpf_rows' output). matchpf is calculated per file by matchtbpf_func and is not part of this
benchmark. Prints the wall time and the peak memory of the process (ru_maxrss) of each step.
"""
import os
import resource
import sys
import time

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synthetic_tbpf import ENDDATE, NMAXPF, STARTDATE, T0, tb_config  # noqa: E402

DT = 3600.0


def peak_gb():
    """
    Peak resident memory of this process so far [GB].

    Returns:
        gb: float
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2


def make_sparse_file(workdir, ntracks, seed=1):
    """
    Write the sparse trackstats file of random tracks (vectorized).

    Args:
        workdir: string
            Folder (the file is trackstats_sparse_<STARTDATE>_<ENDDATE>.nc).
        ntracks: int
            Number of tracks.
        seed: int (default: 1)
            Random seed.

    Returns:
        nentries: int
            Number of (track, time) entries.
    """
    rng = np.random.default_rng(seed)
    duration = np.minimum(1 + rng.geometric(1 / 6.0, ntracks), 400).astype(np.int32)
    is_big = rng.random(ntracks) < 0.03
    duration[is_big] = np.maximum(duration[is_big], 6)
    nentries = int(duration.sum())
    track_of = np.repeat(np.arange(ntracks, dtype=np.int32), duration)
    k = (np.arange(nentries) - np.repeat(np.cumsum(duration) - duration, duration)).astype(np.int32)
    start = rng.integers(0, 8700, ntracks)
    # Short tracks that merge into / split from another track (a quarter of them each); most of the links point to an MCS-sized track, as
    # in real tracks where the small clouds merge into the large systems, and a merging track exists while its target exists
    short = duration <= 3
    big_numbers = np.flatnonzero(is_big) + 1
    target = lambda: np.where(rng.random(ntracks) < 0.7, big_numbers[rng.integers(0, len(big_numbers), ntracks)], rng.integers(1, ntracks + 1, ntracks))
    end_merge = np.where(short & (rng.random(ntracks) < 0.25), target(), -9999).astype(np.int32)
    start_split = np.where(short & (rng.random(ntracks) < 0.25), target(), -9999).astype(np.int32)
    linked = np.flatnonzero(end_merge > 0)
    goal = end_merge[linked] - 1
    start[linked] = np.clip(start[goal] + rng.integers(0, np.maximum(duration[goal] - duration[linked] + 1, 1)), 0, 8700)
    base_time = T0 + (start[track_of] + k) * DT
    big_entry = is_big[track_of]
    core = np.where(big_entry, rng.uniform(2000, 15000, nentries), rng.uniform(0, 3000, nentries)).astype(np.float32)
    cold = np.where(big_entry, rng.uniform(40000, 100000, nentries), rng.uniform(500, 20000, nentries)).astype(np.float32)
    lon = rng.uniform(-179, 179, nentries).astype(np.float32)
    lat = rng.uniform(-59, 59, nentries).astype(np.float32)
    fill_f, fill_i = {"_FillValue": np.float32(np.nan)}, {"_FillValue": np.int32(-9999)}
    variables = {
        "base_time": (["sparse_index"], base_time, {"_FillValue": -9999.0}), "meanlat": (["sparse_index"], lat, fill_f),
        "meanlon": (["sparse_index"], lon, fill_f), "area": (["sparse_index"], core + cold, fill_f),
        "cloudnumber": (["sparse_index"], rng.integers(1, 3000, nentries).astype(np.int32), fill_i),
        "track_status": (["sparse_index"], np.zeros(nentries, dtype=np.int32), fill_i),
        "core_area": (["sparse_index"], core, fill_f), "cold_area": (["sparse_index"], cold, fill_f),
        "merge_tracknumbers": (["sparse_index"], np.full(nentries, -9999, dtype=np.int32), fill_i),
        "split_tracknumbers": (["sparse_index"], np.full(nentries, -9999, dtype=np.int32), fill_i),
        "tracks_indices": (["sparse_index"], track_of), "times_indices": (["sparse_index"], k),
        "track_duration": (["tracks"], duration), "end_merge_tracknumber": (["tracks"], end_merge),
        "start_split_tracknumber": (["tracks"], start_split),
        "end_merge_timeindex": (["tracks"], np.full(ntracks, -9999, dtype=np.int32)),
        "start_split_timeindex": (["tracks"], np.full(ntracks, -9999, dtype=np.int32)),
    }
    ds = xr.Dataset(variables, coords={"tracks": np.arange(ntracks), "sparse_index": np.arange(nentries)},
                    attrs={"Title": "benchmark", "time_resolution_hour": 1.0})
    ds.to_netcdf(os.path.join(workdir, f"trackstats_sparse_{STARTDATE}_{ENDDATE}.nc"))
    return nentries


def make_pf_tables(workdir, config):
    """
    The PF statistics of the MCS that identifymcs_tb_rows found: random PF variables of the shape of the output of matchtbpf_rows.

    Args:
        workdir: string
            Folder with the Tb row tables (with the trailing slash).
        config: dictionary
            Config of the steps.
    """
    from pyflextrkr.rowstats import TableBuilder, read_tables, write_rows
    rng = np.random.default_rng(2)
    tables = read_tables(f"{workdir}mcs_tracks_{STARTDATE}_{ENDDATE}.rows")
    builder = TableBuilder.from_tables(tables)
    n = int(tables.duration.sum())
    for name, scale in (("pf_area", 4000), ("pf_majoraxis", 80), ("pf_rainrate", 8), ("pf_skewness", 2), ("pf_accumrain", 40000),
                        ("pf_accumrainheavy", 15000), ("pf_maxrainrate", 30)):
        values = (rng.random((n, NMAXPF)) * scale).astype(np.float32)
        values[:, 1:][rng.random((n, NMAXPF - 1)) < 0.3] = np.nan
        builder.add_entry3d_var(name, values, "nmaxpf", {"_FillValue": np.float32(np.nan)}, fill=np.nan)
    builder.add_entry_var("total_volrain", (rng.random(n) * 60000).astype(np.float32), {"_FillValue": np.float32(np.nan)}, fill=np.nan)
    write_rows(builder.build(), f"{workdir}mcs_tracks_pf_{STARTDATE}_{ENDDATE}.rows")


def main():
    """
    Make the file, run the steps and print the times.
    """
    workdir = os.path.abspath(sys.argv[1]) + "/"
    ntracks = int(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000
    dense = len(sys.argv) > 3 and sys.argv[3] == "dense"
    os.makedirs(workdir, exist_ok=True)
    config = tb_config(workdir, robustmcs_definition="mcsmip", mcs_pf_majoraxis_thresh=30, max_pf_majoraxis_thresh=100000, nmaxlinks=50,
                       duration_range=[2, 400], mcs_tb_split_duration=12, mcs_tb_merge_duration=12)
    t0 = time.time()
    sparse = os.path.join(workdir, f"trackstats_sparse_{STARTDATE}_{ENDDATE}.nc")
    if dense and os.path.isfile(sparse):
        nentries = int(xr.open_dataset(sparse)["track_duration"].values.sum())
        print(f"sparse file of the rows run: {ntracks:,} tracks, {nentries:,} entries")
    else:
        nentries = make_sparse_file(workdir, ntracks)
        print(f"sparse file: {ntracks:,} tracks, {nentries:,} entries, made in {time.time() - t0:.0f} s")
    if dense:
        # The dense step alone, in this process (the peak memory is the one of the step plus the process that was just started)
        from pyflextrkr.identifymcs import identifymcs_tb
        before = peak_gb()
        t0 = time.time()
        identifymcs_tb(config)
        print(f"identifymcs_tb (dense): {time.time() - t0:.0f} s, peak memory {peak_gb():.1f} GB (before: {before:.1f} GB)")
        print(f"  dense file: {os.path.getsize(os.path.join(workdir, f'mcs_tracks_{STARTDATE}_{ENDDATE}.nc')) / 1e9:.2f} GB")
        return
    from pyflextrkr.identifymcs_tb_rows import identifymcs_tb_rows
    from pyflextrkr.robustmcs_rows import define_robust_mcs_pf_rows
    from pyflextrkr.rowstats import read_tables
    before = peak_gb()
    t0 = time.time()
    folder = identifymcs_tb_rows(config)
    print(f"identifymcs_tb_rows: {time.time() - t0:.0f} s, peak memory {peak_gb():.1f} GB (before: {before:.1f} GB)")
    tables = read_tables(folder)
    print(f"  {len(tables.duration):,} MCS, {len(tables.entries['track']):,} entries, {len(tables.fragments['track']):,} merge / split fragments")
    make_pf_tables(workdir, config)
    t0 = time.time()
    define_robust_mcs_pf_rows(config)
    print(f"robustmcs_rows (mcsmip): {time.time() - t0:.0f} s, peak memory {peak_gb():.1f} GB")
    config["robustmcs_definition"] = "feng21"
    config.update({"coefs_pf_area": [4.2, 0.2], "coefs_pf_rr": [0.5, 0.01], "coefs_pf_skew": [0.2, 0.01], "coefs_pf_heavyratio": [10, 0.5]})
    t0 = time.time()
    define_robust_mcs_pf_rows(config)
    print(f"robustmcs_rows (feng21): {time.time() - t0:.0f} s, peak memory {peak_gb():.1f} GB")
    folder_size = sum(os.path.getsize(os.path.join(root, f)) for root, _, files in os.walk(f"{workdir}mcs_tracks_{STARTDATE}_{ENDDATE}.rows") for f in files)
    print(f"  Tb row tables: {folder_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
