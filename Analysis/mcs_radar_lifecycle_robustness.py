#!/usr/bin/env python
"""
Monte-Carlo robustness of the MCS lifecycle staging (pyflextrkr.mcs_lifecycle.classify_lifecycle) on synthetic
MCS with a known lifecycle, compared with the staging as it was before it became time-based (classify_lifecycle_v1,
index steps and no smoothing). Used to choose the default of mcs_lifecycle_smooth_hour.

For every case (textbook single cycle, two cycles, no stratiform area, line too short for genesis, track truncated
before decay, never organized) x time resolution (5, 15, 30, 60 min, irregular 4/6 min) x disturbance (clean,
moderate, severe: log-normal noise, isolated spikes, dropped samples) it draws --nreal noisy realizations, classifies
each with the old function and with the new one at several smoothing windows, and scores them against the truth (the
old function on the dense, noise-free 1-minute series; see tests/lifecycle_synthetic.py).

A realization passes if the first genesis, mature and decay samples are within max(1 h, 2 samples) of the truth (or
absent in both), no stage is invented or missed, and the number of organized-line episodes is right.

usage:
  mcs_radar_lifecycle_robustness.py --outdir DIR [--nreal 200] [--windows 0 0.25 0.5 1.0] [--workers 64]

writes in DIR:
  robustness_summary.csv   one row per case x resolution x noise x variant
  robustness_summary.md    pass rate averaged over the cases, per noise x resolution x variant
  robustness_pass_rate.png pass rate per variant and resolution, one panel per disturbance
  robustness_onset_error.png  onset-error distributions (textbook case) old vs new
"""
import os
import sys
import csv
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tests"))
sys.path.insert(0, os.path.join(HERE, ".."))
import lifecycle_synthetic as L                                  # noqa: E402
from pyflextrkr.mcs_lifecycle import classify_lifecycle           # noqa: E402

STAGES = ["genesis", "mature", "decay"]


def variant_names(windows):
    return ["before"] + [f"smooth {w:g} h" if w > 0 else "time-based, no smoothing" for w in windows]


def make_classifier(name, windows):
    """classify(conv, length, strat, dt, params, t) for a variant name."""
    if name == "before":
        return lambda conv, length, strat, dt, p, t: L.classify_lifecycle_v1(conv, length, strat, dt, p)
    window = 0.0 if name.startswith("time-based") else float(name.split()[1])
    return lambda conv, length, strat, dt, p, t: classify_lifecycle(
        conv, length, strat, dt, dict(p, mcs_lifecycle_smooth_hour=window), time_hours=t)


def evaluate_cell(args):
    """All variants on one case x resolution x noise cell (same noisy realizations for every variant)."""
    case, resolution, noise, nreal, seed, windows = args
    rows, errors = [], {}
    for name in variant_names(windows):
        r = L.run_case(make_classifier(name, windows), case, resolution, noise, nreal, seed)
        finite = np.abs(r["err"][np.isfinite(r["err"])])
        both = finite[finite >= 0]
        rows.append(dict(
            case=case, resolution=resolution, noise=noise, variant=name, nreal=nreal,
            pass_rate=L.pass_rate(r), agreement=float(r["agreement"].mean()),
            invented_rate=float(r["invented"].any(axis=1).mean()), missed_rate=float(r["missed"].any(axis=1).mean()),
            ncycles_ok=float(r["ncycles_ok"].mean()),
            abs_err_median_h=float(np.median(both)) if both.size else np.nan,
            abs_err_p90_h=float(np.percentile(both, 90)) if both.size else np.nan,
        ))
        errors[name] = r["err"]
    return rows, (case, resolution, noise), errors


def write_markdown(rows, windows, path):
    names = variant_names(windows)
    noises = ["clean", "moderate", "severe"]
    lines = ["Pass rate (mean over the six cases; fraction of realizations with all onsets within max(1 h, 2 samples), "
             "no stage invented or missed, right number of episodes)\n"]
    for noise in noises:
        lines.append(f"\n**{noise}**\n")
        lines.append("| resolution | " + " | ".join(names) + " |")
        lines.append("|---|" + "---|" * len(names))
        for res in L.RESOLUTIONS:
            vals = []
            for name in names:
                v = [r["pass_rate"] for r in rows if r["noise"] == noise and r["resolution"] == res and r["variant"] == name]
                vals.append(f"{np.mean(v):.2f}")
            lines.append(f"| {res if res == 'irregular' else str(res) + ' min'} | " + " | ".join(vals) + " |")
    open(path, "w").write("\n".join(lines) + "\n")


def plot_pass_rate(rows, windows, path):
    names = variant_names(windows)
    noises = ["clean", "moderate", "severe"]
    colors = ["#9e9e9e"] + list(plt.cm.Blues(np.linspace(0.3, 0.95, len(names) - 1)))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    x = np.arange(len(L.RESOLUTIONS))
    width = 0.8 / len(names)
    for ax, noise in zip(axes, noises):
        for iv, name in enumerate(names):
            mean, lo, hi = [], [], []
            for res in L.RESOLUTIONS:
                v = [r["pass_rate"] for r in rows if r["noise"] == noise and r["resolution"] == res and r["variant"] == name]
                mean.append(np.mean(v)); lo.append(np.mean(v) - np.min(v)); hi.append(np.max(v) - np.mean(v))
            ax.bar(x + (iv - (len(names) - 1) / 2) * width, mean, width, yerr=[lo, hi], color=colors[iv],
                   label=name, error_kw=dict(lw=0.8, capsize=1.5))
        ax.set_xticks(x)
        ax.set_xticklabels([r if r == "irregular" else f"{r} min" for r in L.RESOLUTIONS])
        ax.set_title(noise)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("pass rate (mean over cases; bars: min-max)")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_onset_error(cell_errors, windows, path, case="textbook"):
    names = variant_names(windows)
    pick = ["before", "smooth 0.5 h"] if "smooth 0.5 h" in names else [names[0], names[-1]]
    noises = ["moderate", "severe"]
    resolutions = [5, 15, "irregular"]
    fig, axes = plt.subplots(len(noises), len(resolutions), figsize=(13, 6.5), sharey=True)
    for i, noise in enumerate(noises):
        for j, res in enumerate(resolutions):
            ax = axes[i, j]
            errs = cell_errors[(case, res, noise)]
            data, pos, cols = [], [], []
            for istage in range(3):
                for k, name in enumerate(pick):
                    e = errs[name][:, istage]
                    data.append(e[np.isfinite(e)])
                    pos.append(istage * 3 + k)
                    cols.append(["#9e9e9e", "#2b7bba"][k])
            bp = ax.boxplot(data, positions=pos, widths=0.8, patch_artist=True, showfliers=False)
            for patch, c in zip(bp["boxes"], cols):
                patch.set_facecolor(c)
            for istage in range(3):
                for k, name in enumerate(pick):
                    e = errs[name][:, istage]
                    ax.text(istage * 3 + k, 2.95, f"{100 * np.mean(~np.isfinite(e)):.0f}%", ha="center", fontsize=7)
            ax.axhline(0, color="k", lw=0.6)
            ax.axhspan(-1, 1, color="#cfe6f5", alpha=0.3, zorder=0)
            ax.set_xticks([0.5, 3.5, 6.5])
            ax.set_xticklabels(STAGES)
            ax.set_ylim(-3, 3.2)
            ax.set_title(f"{noise}, {res if res == 'irregular' else str(res) + ' min'}", fontsize=10)
            if j == 0:
                ax.set_ylabel("onset error [h]")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ["#9e9e9e", "#2b7bba"]]
    fig.legend(handles, pick, loc="lower center", ncol=2, frameon=False)
    fig.suptitle(f"{case}: onset error of the first genesis / mature / decay sample "
                 "(numbers: % of realizations with the stage invented or missed)", x=0.02, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Monte-Carlo robustness of the MCS lifecycle staging")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--nreal", type=int, default=200, help="noisy realizations per case x resolution x noise")
    ap.add_argument("--windows", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0],
                    help="mcs_lifecycle_smooth_hour values of the new classifier [hour]")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=1, help="cells evaluated in parallel (use a compute node for > 4)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    cells = [(case, res, noise, args.nreal, args.seed + 1000 * ic + 100 * ir + 10 * inz, args.windows)
             for ic, case in enumerate(L.CASES) for ir, res in enumerate(L.RESOLUTIONS)
             for inz, noise in enumerate(["clean", "moderate", "severe"])]
    t0 = time.time()
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as ex:
            results = list(ex.map(evaluate_cell, cells, chunksize=1))
    else:
        results = [evaluate_cell(c) for c in cells]
    print(f"{len(cells)} cells x {len(variant_names(args.windows))} variants x {args.nreal} realizations "
          f"in {time.time() - t0:.0f} s")

    rows = [r for rs, _, _ in results for r in rs]
    cell_errors = {key: errs for _, key, errs in results}
    with open(os.path.join(args.outdir, "robustness_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    write_markdown(rows, args.windows, os.path.join(args.outdir, "robustness_summary.md"))
    plot_pass_rate(rows, args.windows, os.path.join(args.outdir, "robustness_pass_rate.png"))
    plot_onset_error(cell_errors, args.windows, os.path.join(args.outdir, "robustness_onset_error.png"))
    print(open(os.path.join(args.outdir, "robustness_summary.md")).read())


if __name__ == "__main__":
    main()
