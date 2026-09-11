#!/usr/bin/env python
"""
Measure and visualize performance of BFS vs EDT growth methods in label_and_grow_features.

For each demo dataset this script:
  1. Loads input Tb files and preprocesses them (same as idclouds_tbpf).
  2. Times `label_and_grow_features` with growth_method='bfs' and 'edt'.
  3. Produces a two-panel bar-chart PNG:
       Top panel    : absolute timing (BFS / EDT bars)
       Bottom panel : speedup (BFS / EDT) on log scale
  4. Saves cloudid arrays from both methods for comparison.

Demos supported
---------------
  demo_mcs_tbpf_idealized : Small idealized domain, fast (verification)
  demo_mcs_imerg          : Global 0.1° IMERG domain (performance benchmark)

Usage
-----
  # 1. Download demo data (only needed once)
  python tests/run_demo_tests.py --demos demo_mcs_tbpf_idealized demo_mcs_imerg -n 4

  # 2. Run benchmark on idealized (fast, multiple frames)
  python tests/plot_label_grow_speedup.py \\
      --demo demo_mcs_tbpf_idealized \\
      --data_root ~/data/demo \\
      --outdir ~/data/demo/benchmark_results/

  # 3. Run benchmark on global IMERG (large domain, performance test)
  python tests/plot_label_grow_speedup.py \\
      --demo demo_mcs_imerg \\
      --data_root ~/data/demo \\
      --nfiles 5 \\
      --outdir ~/data/demo/benchmark_results/

  # 4. Both methods' cloudid outputs saved to:
  #    <outdir>/cloudid_bfs/
  #    <outdir>/cloudid_edt/

Requirements
------------
  matplotlib, numpy, scipy, xarray (available in pyflextrkr environment)
  Demo data under $PYFLEXTRKR_TEST_DATA or --data_root.
"""

import argparse
import glob
import os
import time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyflextrkr.label_and_grow_features import label_and_grow_features

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument(
    "--demo",
    default="demo_mcs_tbpf_idealized",
    choices=["demo_mcs_tbpf_idealized", "demo_mcs_imerg"],
    help="Which demo to benchmark (default: demo_mcs_tbpf_idealized)",
)
parser.add_argument(
    "--data_root",
    default=os.environ.get("PYFLEXTRKR_TEST_DATA", os.path.expanduser("~/data/demo")),
    help="Path to demo data root (default: $PYFLEXTRKR_TEST_DATA or ~/data/demo)",
)
parser.add_argument(
    "--outdir",
    default=".",
    help="Directory to save output figures and cloudid arrays (default: .)",
)
parser.add_argument(
    "--nfiles",
    type=int,
    default=0,
    help="Number of files to time (0 = all available)",
)
args = parser.parse_args()

# ── Demo specifications ───────────────────────────────────────────────────────
DEMOS = {
    "demo_mcs_tbpf_idealized": {
        "name": "MCS Idealized",
        "data_subdir": "mcs_tbpf/idealized/test4/input",
        "pattern": "*.nc",
        "tb_varname": "Tb",
        "pixel_radius": 10.0,
        "thresholds": [225.0, 241.0, 261.0, 261.0],
        "area_thresh": 800.0,
        "min_core_npix": 4,
        "smooth_size": 5,
        "expand_to_tertiary": 0,
        "config": {"pbc_direction": "none"},
    },
    "demo_mcs_imerg": {
        "name": "IMERG Global (60S-60N, 0.1°)",
        "data_subdir": "mcs_tbpf/imerg/input",
        "pattern": "merg_*.nc",
        "tb_varname": "Tb",
        "pixel_radius": 10.0,
        "thresholds": [225.0, 241.0, 261.0, 261.0],
        "area_thresh": 800.0,
        "min_core_npix": 4,
        # Matches smoothwindowdimensions in config/config_imerg_mcs_tbpf_example.yml
        # (the actual config this demo runs with) - not 5, which is only
        # correct for the idealized demo's config_mcs_idealized.yml.
        "smooth_size": 10,
        "expand_to_tertiary": 0,
        "config": {"pbc_direction": "none"},
    },
}


def load_tb_files(demo_spec, data_root, nfiles):
    """Load and preprocess Tb data from demo input files."""
    import xarray as xr
    from scipy.signal import medfilt2d

    input_dir = os.path.join(data_root, demo_spec["data_subdir"])
    files = sorted(glob.glob(os.path.join(input_dir, demo_spec["pattern"])))
    if not files:
        files = sorted(
            glob.glob(os.path.join(input_dir, "**", demo_spec["pattern"]), recursive=True)
        )
    if nfiles > 0:
        files = files[:nfiles]

    if not files:
        raise FileNotFoundError(
            f"No input files found in {input_dir} with pattern {demo_spec['pattern']}.\n"
            f"Download demo data first:\n"
            f"  python tests/run_demo_tests.py --demos {args.demo} -n 4"
        )

    print(f"  Found {len(files)} input file(s) in {input_dir}")

    tb_arrays = []
    for filepath in files:
        ds = xr.open_dataset(filepath, decode_timedelta=False)
        varname = demo_spec["tb_varname"]
        if varname not in ds:
            # Try common alternatives
            for alt in ["Tb", "tb", "brightness_temperature"]:
                if alt in ds:
                    varname = alt
                    break
        if varname not in ds:
            ds.close()
            continue

        tb = ds[varname].values
        ds.close()

        # Handle 3D (take first time)
        if tb.ndim == 3:
            tb = tb[0, :, :]

        # Preprocess (same as idclouds_tbpf)
        tb_filt = medfilt2d(tb.astype(np.float64), kernel_size=5)
        out_tb = np.copy(tb)
        missmask = np.isnan(tb)
        out_tb[missmask] = tb_filt[missmask]
        out_tb[out_tb < 160] = np.nan
        out_tb[out_tb > 330] = np.nan

        # Skip frames with too much missing data
        ny, nx = out_tb.shape
        if np.count_nonzero(np.isnan(out_tb)) / (ny * nx) >= 0.4:
            continue

        tb_arrays.append((out_tb, os.path.basename(filepath)))

    return tb_arrays


def time_methods(tb_arrays, demo_spec, outdir):
    """Time BFS and EDT methods on all input frames."""
    timings = defaultdict(list)
    results_bfs = []
    results_edt = []

    params = {
        "pixel_radius": demo_spec["pixel_radius"],
        "thresholds": demo_spec["thresholds"],
        "area_thresh": demo_spec["area_thresh"],
        "min_core_npix": demo_spec["min_core_npix"],
        "smooth_size": demo_spec["smooth_size"],
        "expand_to_tertiary": demo_spec["expand_to_tertiary"],
        "config": demo_spec["config"],
    }

    for tb, fname in tb_arrays:
        ny, nx = tb.shape

        # Time BFS
        t0 = time.perf_counter()
        result_bfs = label_and_grow_features(
            tb, params["pixel_radius"], params["thresholds"],
            params["area_thresh"], params["min_core_npix"],
            params["smooth_size"], params["expand_to_tertiary"],
            params["config"], core_operator="lt", growth_method="bfs",
        )
        timings["bfs"].append(time.perf_counter() - t0)

        # Time EDT
        t0 = time.perf_counter()
        result_edt = label_and_grow_features(
            tb, params["pixel_radius"], params["thresholds"],
            params["area_thresh"], params["min_core_npix"],
            params["smooth_size"], params["expand_to_tertiary"],
            params["config"], core_operator="lt", growth_method="edt",
        )
        timings["edt"].append(time.perf_counter() - t0)

        results_bfs.append((fname, result_bfs))
        results_edt.append((fname, result_edt))

        # Print per-file summary
        bfs_t = timings["bfs"][-1]
        edt_t = timings["edt"][-1]
        speedup = bfs_t / max(edt_t, 1e-9)
        n_features = result_bfs["final_nFeature"]
        print(
            f"    {fname}: {ny}x{nx}, {n_features} features, "
            f"BFS={bfs_t:.3f}s, EDT={edt_t:.3f}s, speedup={speedup:.1f}x"
        )

    # Save cloudid arrays
    bfs_dir = os.path.join(outdir, "cloudid_bfs")
    edt_dir = os.path.join(outdir, "cloudid_edt")
    os.makedirs(bfs_dir, exist_ok=True)
    os.makedirs(edt_dir, exist_ok=True)

    for fname, result in results_bfs:
        np.savez_compressed(
            os.path.join(bfs_dir, fname.replace(".nc", ".npz")),
            cloudnumber=result["final_Feature_Number"],
            convcold_cloudnumber=result["final_CoreSecondary_Number"],
            cloudtype=result["final_Feature_Type"],
        )
    for fname, result in results_edt:
        np.savez_compressed(
            os.path.join(edt_dir, fname.replace(".nc", ".npz")),
            cloudnumber=result["final_Feature_Number"],
            convcold_cloudnumber=result["final_CoreSecondary_Number"],
            cloudtype=result["final_Feature_Type"],
        )

    print(f"  Saved cloudid arrays to {bfs_dir} and {edt_dir}")
    return timings


def make_figure(demo_name, timings, outdir, nfiles):
    """Create two-panel bar chart: timing + speedup."""
    bfs_times = np.array(timings["bfs"])
    edt_times = np.array(timings["edt"])
    n = len(bfs_times)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    fig.suptitle(
        f"label_and_grow_features: BFS vs EDT — {demo_name} (n={n} files)",
        fontsize=12,
    )

    x = np.arange(n)
    width = 0.35

    # Top panel: absolute timing
    ax1.bar(x - width / 2, bfs_times, width, label="BFS", color="steelblue")
    ax1.bar(x + width / 2, edt_times, width, label="EDT", color="forestgreen")
    ax1.set_ylabel("Time per file (s)")
    ax1.set_xlabel("File index")
    ax1.legend()
    ax1.set_title("Absolute timing")
    ax1.grid(axis="y", alpha=0.3)

    # Add mean annotations
    mean_bfs = np.mean(bfs_times)
    mean_edt = np.mean(edt_times)
    ax1.axhline(mean_bfs, color="steelblue", linestyle="--", alpha=0.5)
    ax1.axhline(mean_edt, color="forestgreen", linestyle="--", alpha=0.5)
    ax1.text(
        n - 0.5, mean_bfs, f"mean={mean_bfs:.3f}s",
        color="steelblue", va="bottom", ha="right", fontsize=8,
    )
    ax1.text(
        n - 0.5, mean_edt, f"mean={mean_edt:.3f}s",
        color="forestgreen", va="bottom", ha="right", fontsize=8,
    )

    # Bottom panel: speedup ratio
    speedups = bfs_times / np.maximum(edt_times, 1e-9)
    bars = ax2.bar(x, speedups, color="darkorange", alpha=0.85)
    for xi, sp in zip(x, speedups):
        lbl = f"{sp:.0f}x" if sp >= 10 else f"{sp:.1f}x"
        ax2.text(xi, sp * 1.05, lbl, ha="center", va="bottom", fontsize=7)
    ax2.axhline(1, color="k", linewidth=0.8, linestyle="--")
    ax2.set_ylabel("Speedup (BFS / EDT)")
    ax2.set_xlabel("File index")
    ax2.set_yscale("log")
    ax2.set_title(f"Speedup ratio (mean = {np.mean(speedups):.1f}x)")
    ax2.grid(axis="y", alpha=0.3, which="both")

    safe_name = demo_name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
    outfile = os.path.join(outdir, f"label_grow_speedup_{safe_name}.png")
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {outfile}")
    return outfile


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo_spec = DEMOS[args.demo]
    os.makedirs(args.outdir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Benchmark: {demo_spec['name']}")
    print(f"  Data root: {args.data_root}")
    print(f"{'='*60}")

    # Load data
    tb_arrays = load_tb_files(demo_spec, args.data_root, args.nfiles)
    print(f"  Loaded {len(tb_arrays)} preprocessed Tb frame(s)")

    if not tb_arrays:
        raise SystemExit("ERROR: No valid Tb frames found.")

    # Time methods
    print("\n  Timing BFS vs EDT:")
    timings = time_methods(tb_arrays, demo_spec, args.outdir)

    # Summary
    bfs_mean = np.mean(timings["bfs"])
    edt_mean = np.mean(timings["edt"])
    speedup_mean = bfs_mean / max(edt_mean, 1e-9)
    print(f"\n  Summary ({len(tb_arrays)} files):")
    print(f"    BFS mean: {bfs_mean:.3f}s")
    print(f"    EDT mean: {edt_mean:.3f}s")
    print(f"    Mean speedup: {speedup_mean:.1f}x")

    # Make figure
    outfile = make_figure(demo_spec["name"], timings, args.outdir, args.nfiles)
    print(f"\n  Done. Output in: {args.outdir}")
