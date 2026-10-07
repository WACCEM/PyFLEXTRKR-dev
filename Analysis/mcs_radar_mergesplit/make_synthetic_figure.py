#!/usr/bin/env python
"""
Illustrate how the previous and the current identifymcs_radar treat merge/split tracks, with a small
synthetic data set that is run through the REAL code of each version (nothing here is hand drawn
except the layout): synthetic_run.py builds the input and runs one module of one code tree.

Figure 1: the 8 input tracks, then the output of the previous and the current code.
Figure 1 (PR version): the input tracks, then ONE full-width panel with the output of the current code.
Figure 2: the same input through identifymcs.py (Tb) and link_mergesplit_tracks.py, for comparison.

usage: make_synthetic_figure.py --workdir DIR --outdir DIR
       --workdir is the directory made by setup_comparison.py (it holds the code_old and code_new trees)
"""
import os
import sys
import subprocess
import argparse
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from synthetic_run import TRACKS, T0, DT_SEC   # same definitions as the input that is run

DT_H = DT_SEC / 3600.0
COLORS = {1: "#b22222", 2: "#1f77b4", 3: "#ff7f0e", 4: "#2ca02c", 5: "#9467bd", 6: "#8c564b", 7: "#e377c2", 8: "#17becf"}
DESC = {
    1: "M   MCS",
    2: "F1  merges into M, inside M's lifetime",
    3: "F2  merges into M, starts before M exists",
    4: "F3  splits from M, outlives M",
    5: "A   merges into B, before M exists",
    6: "B   merges into M, starts before M exists",
    7: "C   merges into D (chain C -> D -> M), inside M",
    8: "D   merges into M, inside M's lifetime",
}
NAME = {k: v[0] for k, v in TRACKS.items()}


def run_module(workdir, tree, which, tag):
    """Run one module of one code tree (<workdir>/<tree>) on the synthetic input; returns the output file."""
    outdir = os.path.join(workdir, "synthetic", f"out_{tag}")
    os.makedirs(outdir, exist_ok=True)
    r = subprocess.run([sys.executable, os.path.join(HERE, "synthetic_run.py"), os.path.join(workdir, tree), which, outdir],
                       capture_output=True, text=True)
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("OUTFILE=")]
    if r.returncode != 0 or not lines:
        raise RuntimeError(f"{tag} failed:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    return lines[-1].split("=", 1)[1]


def load_output(path):
    """Per output track: its own time points and the fragment time points attached to it (origin = input track)."""
    d = xr.open_dataset(path, decode_times=False, mask_and_scale=False)
    bt, cn = d["base_time"].values, d["cloudnumber"].values
    mc, sc = d["merge_cloudnumber"].values, d["split_cloudnumber"].values
    tracks = []
    for k in range(d.sizes["tracks"]):
        valid = np.where(cn[k] > 0)[0]
        origin = int(cn[k][valid[0]] // 1000)
        step = lambda t: int(round((bt[k, t] - T0) / DT_SEC))
        own = {step(t): origin for t in valid}
        att = {}
        for arr in (mc, sc):
            for t in valid:
                for v in arr[k, t, :]:
                    if v > 0:
                        att.setdefault(step(t), []).append(int(v // 1000))
        tracks.append({"origin": origin, "own": own, "att": att})
    return tracks


def input_points():
    pts = set()
    for tid, (_, s0, s1, _, _) in TRACKS.items():
        pts |= {(tid, s) for s in range(s0, s1 + 1)}
    return pts


def summarize(tracks):
    covered, attached_to_m, standalone = set(), {}, []
    for t in tracks:
        covered |= {(o, s) for s, o in t["own"].items()}
        for s, lst in t["att"].items():
            covered |= {(o, s) for o in lst}
            if t["origin"] == 1:
                for o in lst:
                    attached_to_m[o] = attached_to_m.get(o, 0) + 1
        if t["origin"] != 1:
            standalone.append(t["origin"])
    lost = sorted(input_points() - covered)
    return {"attached_to_m": attached_to_m, "standalone": sorted(standalone), "lost": lost, "n_tracks": len(tracks)}


def draw_lifelines(ax):
    ids = list(TRACKS.keys())
    ypos = {tid: i for i, tid in enumerate(ids)}
    for tid, (nm, s0, s1, em, ss) in TRACKS.items():
        ax.barh(ypos[tid], (s1 + 1 - s0) * DT_H, left=s0 * DT_H, height=0.5, color=COLORS[tid],
                alpha=0.95 if tid == 1 else 0.85, zorder=3)
    for tid, (nm, s0, s1, em, ss) in TRACKS.items():
        if em:
            x = (s1 + 1) * DT_H - 0.03
            ax.annotate("", xy=(x, ypos[em]), xytext=(x, ypos[tid]), zorder=4,
                        arrowprops=dict(arrowstyle="-|>", color=COLORS[tid], lw=1.6, shrinkA=3, shrinkB=3))
        if ss:
            x = s0 * DT_H + 0.03
            ax.annotate("", xy=(x, ypos[tid]), xytext=(x, ypos[ss]), zorder=4,
                        arrowprops=dict(arrowstyle="-|>", color=COLORS[tid], lw=1.6, shrinkA=3, shrinkB=3))
    ms, me = TRACKS[1][1] * DT_H, (TRACKS[1][2] + 1) * DT_H
    for x in (ms, me):
        ax.axvline(x, color="gray", ls=":", lw=1.2, zorder=1)
    ax.text(ms, -0.75, "MCS M starts", ha="center", va="bottom", fontsize=9.5, color="0.35")
    ax.text(me, -0.75, "MCS M ends", ha="center", va="bottom", fontsize=9.5, color="0.35")
    ax.set_ylim(len(ids) - 0.4, -1.4)
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels([DESC[t] for t in ids], fontsize=10)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("time since start of data (h)")
    ax.set_title("Input: 8 tracks (5-min steps) and their links: arrow into the parent = merge, arrow out of the parent = split",
                 fontsize=11.5, loc="left", pad=12)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)


ROWS = ["M: own feature", "M: merged-in / split-off fragments", "standalone track F2", "standalone track F3",
        "standalone track A", "standalone track B", "time points in NO track"]
ROW_OF_STANDALONE = {3: 2, 4: 3, 5: 4, 6: 5}


def draw_output(ax, tracks, title, subtitle, show_labels=True, lost_color="red", removed_text=None, lost_text=None,
                title_pad=40):
    h = 0.72
    nrow = len(ROWS)

    def rect(step, row, color, frac=(0.0, 1.0), **kw):
        y0 = row + (1 - h) / 2 + frac[0] * h
        ax.add_patch(Rectangle((step * DT_H, y0), DT_H, (frac[1] - frac[0]) * h, linewidth=0, facecolor=color, **kw))

    for t in tracks:
        o = t["origin"]
        if o == 1:
            for s in t["own"]:
                rect(s, 0, COLORS[1])
            for s, lst in t["att"].items():
                k = len(lst)
                for i, origin in enumerate(sorted(lst)):
                    rect(s, 1, COLORS[origin], frac=(i / k, (i + 1) / k))
        else:
            row = ROW_OF_STANDALONE[o]
            for s in t["own"]:
                rect(s, row, COLORS[o])
    summ = summarize(tracks)
    for (o, s) in summ["lost"]:
        rect(s, nrow - 1, COLORS[o], alpha=0.35)
        ax.add_patch(Rectangle((s * DT_H, nrow - 1 + (1 - h) / 2), DT_H, h, fill=False, hatch="////", edgecolor=lost_color,
                               linewidth=0))
    kept_rows = {0, 1} | {ROW_OF_STANDALONE[o] for o in summ["standalone"]}
    xmid = 0.5 * (TRACKS[1][1] + TRACKS[1][2]) * DT_H
    if removed_text:
        for row in range(2, nrow - 1):
            if row not in kept_rows:
                ax.text(xmid, row + 0.5, removed_text, ha="center", va="center", fontsize=9, color="0.45", style="italic")
    if lost_text:
        ax.text(xmid, nrow - 0.5, lost_text, ha="center", va="center", fontsize=9, color="0.25", style="italic",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5))
    for x in (TRACKS[1][1] * DT_H, (TRACKS[1][2] + 1) * DT_H):
        ax.axvline(x, color="gray", ls=":", lw=1.0, zorder=0)
    ax.set_ylim(nrow, -0.1)
    ax.set_yticks(np.arange(nrow) + 0.5)
    ax.set_yticklabels(ROWS if show_labels else [], fontsize=10)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title, fontsize=13, fontweight="bold", loc="left", pad=title_pad)
    ax.text(0.0, 1.015, subtitle, transform=ax.transAxes, va="bottom", ha="left", fontsize=9.3, color="0.2")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    return summ


def sentence(summ):
    att = summ["attached_to_m"]
    return (f"folded into M: {len(att)} fragment tracks ({sum(att.values())} points)  |  output tracks: {summ['n_tracks']}\n"
            f"standalone tracks: {len(summ['standalone'])}  |  time points in NO track: {len(summ['lost'])}")


def figure1(prev, curr, outbase):
    fig = plt.figure(figsize=(17, 13.2), dpi=150, facecolor="w")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 2.1], hspace=0.42, wspace=0.05,
                          left=0.215, right=0.985, top=0.925, bottom=0.19)
    ax0 = fig.add_subplot(gs[0, :])
    draw_lifelines(ax0)
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax2 = fig.add_subplot(gs[1, 1], sharex=ax0)
    draw_output(ax1, prev, "Previous code", sentence(summarize(prev)), show_labels=True,
                removed_text="not an output track (removed as a fragment)")
    draw_output(ax2, curr, "Current code", sentence(summarize(curr)), show_labels=False)
    for a in (ax1, ax2):
        a.set_xlabel("time since start of data (h)")
    ax0.set_xlim(0, 12.2)
    fig.suptitle("How identifymcs_radar treats short merge/split tracks: previous vs current code\n"
                 "(synthetic input, output of the real code of each version)", fontsize=15, fontweight="bold",
                 x=0.012, ha="left", y=0.985)
    legend = [Patch(facecolor=COLORS[i], label=f"{NAME[i]}") for i in TRACKS]
    legend.append(Patch(facecolor="white", edgecolor="red", hatch="////", label="time points of a removed fragment: in NO track"))
    fig.legend(handles=legend, loc="lower center", ncol=5, fontsize=10, frameon=False, bbox_to_anchor=(0.6, 0.075))
    fig.text(0.012, 0.012,
             "Previous: every short merge/split track is folded into its root; only the time points where the root exists are attached, the rest is\n"
             "discarded together with the track, so those pixels belong to no track.\n"
             "Current: a short merge/split track is folded into its root only if ALL of its time points exist on the root's time axis;\n"
             "otherwise it stays a normal track and its pixels keep a track number.",
             fontsize=10.3, ha="left", va="bottom", linespacing=1.5)
    for ext in ("png", "svg"):
        fig.savefig(f"{outbase}.{ext}")
    plt.close(fig)


def figure1_pr(prev, curr, outbase):
    """PR version of figure 1: the input tracks on top and ONE full-width panel with the output of the current code."""
    n_lost_before = len(summarize(prev)["lost"])
    fig = plt.figure(figsize=(17, 11.8), dpi=150, facecolor="w")
    gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 2.0], hspace=0.40, left=0.215, right=0.985, top=0.915, bottom=0.19)
    ax0 = fig.add_subplot(gs[0, 0])
    draw_lifelines(ax0)
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    draw_output(ax1, curr, "Output of the current code", sentence(summarize(curr)).replace("\n", "   |   "),
                show_labels=True, title_pad=28)
    ax1.set_xlabel("time since start of data (h)")
    ax0.set_xlim(0, 12.2)
    fig.suptitle("How identifymcs_radar treats short merge/split tracks (current code)\n"
                 "(synthetic input, output of the real code)", fontsize=15, fontweight="bold", x=0.012, ha="left", y=0.985)
    legend = [Patch(facecolor=COLORS[i], label=f"{NAME[i]}") for i in TRACKS]
    fig.legend(handles=legend, loc="lower center", ncol=8, fontsize=10.5, frameon=False, bbox_to_anchor=(0.6, 0.075))
    fig.text(0.012, 0.012,
             "Folded into M: F1, D and C. All of their time points lie inside M's lifetime, so they become part of M's mask.\n"
             "Kept as standalone tracks: F2, F3, A and B. Some of their time points lie outside M's lifetime, so their pixels keep a track number.\n"
             f"Before this change such tracks were always folded; the time points M did not cover ({n_lost_before} here) were discarded "
             "together with the track, leaving those pixels in no track.",
             fontsize=10.3, ha="left", va="bottom", linespacing=1.5)
    for ext in ("png", "svg"):
        fig.savefig(f"{outbase}.{ext}")
    plt.close(fig)


def figure2(prev, tb, link, outbase):
    fig = plt.figure(figsize=(21, 8.2), dpi=150, facecolor="w")
    gs = fig.add_gridspec(1, 3, wspace=0.06, left=0.14, right=0.99, top=0.80, bottom=0.2)
    axes = [fig.add_subplot(gs[0, i], sharex=None) for i in range(3)]
    gray = "0.35"
    chain = "chain member C (C -> D -> M) is never attached: only direct children of M are searched"
    s_prev = summarize(prev)
    s_tb, s_link = summarize(tb), summarize(link)
    draw_output(axes[0], prev, "Previous identifymcs_radar", sentence(s_prev), show_labels=True,
                removed_text="not an output track (removed as a fragment)")
    draw_output(axes[1], tb, "identifymcs.py (Tb)",
                f"output tracks: {s_tb['n_tracks']}  |  fragment points attached to M: {sum(s_tb['attached_to_m'].values())}\n{chain}",
                show_labels=False, lost_color=gray, removed_text="not output (only MCS tracks are kept)",
                lost_text="not in M's mask; non-MCS tracks are not a product here")
    draw_output(axes[2], link, "link_mergesplit_tracks.py",
                f"output tracks: {s_link['n_tracks']}  |  fragment points attached to M: {sum(s_link['attached_to_m'].values())}\n{chain}",
                show_labels=False, lost_color=gray, removed_text="not output (only main tracks are kept)",
                lost_text="not in M's mask; non-main tracks are not a product here")
    for ax in axes:
        ax.set_xlim(0, 12.2)
        ax.set_xlabel("time since start of data (h)")
    fig.suptitle("Same synthetic input through the two other modules: like the previous radar code they attach the time points a fragment\n"
                 "shares with M (partial attach), but they keep only M and follow direct children only", fontsize=14,
                 fontweight="bold", x=0.012, ha="left", y=0.985)
    legend = [Patch(facecolor=COLORS[i], label=f"{NAME[i]}") for i in TRACKS]
    fig.legend(handles=legend, loc="lower center", ncol=8, fontsize=10.5, frameon=False, bbox_to_anchor=(0.55, 0.02))
    for ext in ("png", "svg"):
        fig.savefig(f"{outbase}.{ext}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Synthetic illustration of the previous vs current merge/split handling")
    ap.add_argument("--workdir", required=True, help="directory made by setup_comparison.py (holds code_old and code_new)")
    ap.add_argument("--outdir", required=True, help="where the figures are written")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    w = os.path.abspath(args.workdir)
    paths = {
        "prev": run_module(w, "code_old", "radar", "old_radar"),
        "curr": run_module(w, "code_new", "radar", "new_radar"),
        "tb": run_module(w, "code_new", "tb", "new_tb"),
        "link": run_module(w, "code_new", "linkms", "new_linkms"),
    }
    out = {k: load_output(v) for k, v in paths.items()}
    for k, v in out.items():
        s = summarize(v)
        print(k, "| output tracks:", s["n_tracks"], "| attached to M:", {NAME[o]: n for o, n in s["attached_to_m"].items()},
              "| standalone:", [NAME[o] for o in s["standalone"]], "| points in no track / not in M:", len(s["lost"]))
    figure1(out["prev"], out["curr"], os.path.join(args.outdir, "synthetic_previous_vs_current"))
    figure1_pr(out["prev"], out["curr"], os.path.join(args.outdir, "synthetic_current_pr"))
    figure2(out["prev"], out["tb"], out["link"], os.path.join(args.outdir, "synthetic_other_modules"))
    print("DONE", args.outdir)


if __name__ == "__main__":
    main()
