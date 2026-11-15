#!/usr/bin/env python
"""
Make the three campaign figures of docs/performance_tbpf.md from the JSON of Analysis/collect_campaign_performance.py:

  perf_tbpf_<name>_campaign   wall-clock time and node-hours by step: the 12 track jobs, the stats job, and below a dashed line the post-processing
                              steps (dense netCDF file of the final statistics, rain maps, accumulation along the tracks) that run in the same allocation. The bars are node-hours (axis at the
                              top); the axes at the bottom are the wall-clock time of each row
  perf_tbpf_<name>_timeline   the campaign on a wall-clock axis: every track job and the stats job split into their steps
  perf_tbpf_<name>_steps      node-hours of each step, ranked, and the spread of the track-job steps over the months

PNG files, 24 bit, written to --outdir (not to the repository: the images of the documentation are assets of the docs-assets release, see
docs/docs_assets.md). Everything on the figures comes from the JSON, including the size of the axes, the position of the braces and of the call-outs,
which are measured from the text. Colors: the eight categorical slots of the CONUS figures (checked with the data-viz validator: lightness band,
chroma, CVD and normal-vision separation of neighbours), a darker tint of the identifymcs slot for the robust MCS step, a neutral gray for the
setup, a teal pair for the rain maps and the accumulation along the tracks and a violet (checked between the speed red and the teal) for the dense
netCDF file; text is always in the ink colors.

    python Analysis/make_performance_tbpf_figures.py --data campaign.json [--outdir DIR] [--name SLUG] [--only campaign timeline steps]

Author: Zhe Feng | zhe.feng@pnnl.gov
"""
import argparse
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.axis  # noqa: E402
import matplotlib.text  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, PathPatch  # noqa: E402
from matplotlib.path import Path  # noqa: E402
from PIL import Image  # noqa: E402

SCHEMA = "tbpf_campaign_performance/1"
# Colors: surface, ink (text), recessive grid
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# The steps of the jobs: the track jobs, the stats job up to speed, the post-processing inside the stats job
TRACK_STEPS = ("idfeature", "tracksingle", "gettracks")
STATS_STEPS = ("trackstats", "identifymcs", "matchpf", "robustmcs", "mapfeature", "speed")
POST_STEPS = ("dense_netcdf", "rainmap", "trackpcp")
# The color of a step is the same in every figure: the categorical slots in their validated order (matchpf takes the slot of the regionprops step of
# the CONUS figures), the robust MCS step a darker tint of the identifymcs slot (the serial MCS steps are one family), the setup neutral gray, the two
# post-processing steps a teal ramp, the dense netCDF step (between speed and the rain maps) a violet that the validator passes against both neighbours
STEP_COLOR = {"idfeature": "#2a78d6", "tracksingle": "#eb6834", "gettracks": "#1baf7a", "trackstats": "#eda100", "identifymcs": "#e87ba4",
              "matchpf": "#008300", "robustmcs": "#b8457a", "mapfeature": "#6250d6", "speed": "#e34948", "setup": "#9a9993",
              "dense_netcdf": "#8e6bbf", "rainmap": "#25a6ad", "trackpcp": "#0e6f78"}
STEP_LABEL = {"setup": "setup + stitch", "dense_netcdf": "dense netCDF"}
# What each post-processing step makes, for the label of the post-processing row
STEP_WHAT = {"dense_netcdf": "dense netCDF file", "rainmap": "rain maps", "trackpcp": "precipitation accumulated along the tracks"}
# Texts that stick out of the figure are an error when this is True (the tests), a warning otherwise
STRICT = False


########################################################################################################################
# Style helpers (the look of the CONUS performance figures)
########################################################################################################################
def setup():
    """
    Set the matplotlib defaults of all figures: surface color, ink colors, hairline grid, font sizes, hatch line width.
    """
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": GRID, "grid.color": GRID, "grid.linewidth": 0.8, "axes.linewidth": 0.8,
        "font.size": 10.5, "axes.titlesize": 11.5, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "legend.frameon": False, "legend.fontsize": 9.5, "hatch.linewidth": 1.1, "axes.axisbelow": True,
    })


def style(ax, grid="y"):
    """
    Recessive axes: no top / right spine, hairline grid on one axis only.

    Args:
        ax: matplotlib Axes
        grid: string (default: 'y')
            'x', 'y' or '' for the axis that gets grid lines.
    """
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(False)
    if grid:
        ax.grid(True, axis=grid)


def title(fig, text, sub=None):
    """
    Left-aligned title and an optional grey subtitle at the top of a figure.

    Args:
        fig: matplotlib Figure
        text: string
            Title.
        sub: string (default: None)
            Subtitle (what was measured, where); may have several lines.
    """
    fig.text(0.012, 1 - 0.1 / fig.get_figheight(), text, ha="left", va="top", fontsize=13.5, fontweight="semibold", color=INK)
    if sub:
        fig.text(0.012, 1 - 0.46 / fig.get_figheight(), sub, ha="left", va="top", fontsize=9.5, color=INK2, linespacing=1.35)


def text_color(hex_color):
    """
    Ink for a label inside a filled mark: white on dark fills, near-black on light ones (relative luminance of the fill).

    Args:
        hex_color: string
            Fill color '#rrggbb'.

    Returns:
        color: string
    """
    rgb = np.array([int(hex_color[i:i + 2], 16) / 255.0 for i in (1, 3, 5)])
    lin = np.where(rgb <= 0.03928, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    luminance = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    return "#0b0b0b" if luminance > 0.30 else "#ffffff"


def bar(ax, y, left, length, height, color, hatch=None):
    """
    A horizontal bar segment with a surface-colored gap around it (a hatched one, with a gray outline, is time that was not work).

    Args:
        ax: matplotlib Axes
        y: float
            Position of the bar.
        left: float
            Where it starts.
        length: float
            Its length.
        height: float
            Its thickness in data units.
        color: string
            Fill color.
        hatch: string (default: None)
            Hatch pattern ('///'): the bar is drawn in the surface color with this pattern in the ink, not filled.
    """
    if hatch:
        ax.barh(y, length, height=height, left=left, facecolor=SURFACE, edgecolor=INK2, hatch=hatch, linewidth=0.9, zorder=3)
        return
    ax.barh(y, length, height=height, left=left, facecolor=color, edgecolor=SURFACE, linewidth=1.2, zorder=3)


def text_size_in(fig, text, **kwargs):
    """
    Size of a text in inches (it is drawn once to measure it, and removed).

    Args:
        fig: matplotlib Figure
        text: string
            The text (lines separated by newlines).
        **kwargs:
            Text properties (fontsize, fontweight, linespacing).

    Returns:
        size: tuple of float
            (width, height) [in].
    """
    t = fig.text(0, 0, text, **kwargs)
    box = t.get_window_extent(fig.canvas.get_renderer())
    t.remove()
    return box.width / fig.dpi, box.height / fig.dpi


def curly_brace(fig, x, y_bottom, y_top, width, color, linewidth=1.4):
    """
    Draw a right-pointing curly brace '}' in figure inches: the straight sides at x, the tip at x + width, from y_bottom to y_top.

    Args:
        fig: matplotlib Figure
        x, y_bottom, y_top, width: float
            Position and size in inches (from the lower left corner of the figure).
        color: string
        linewidth: float (default: 1.4)

    Returns:
        tip: float
            x of the tip [in].
    """
    r = min(width / 2.0, (y_top - y_bottom) / 4.0)
    ym = 0.5 * (y_top + y_bottom)
    verts = [(x, y_top), (x + r, y_top), (x + r, y_top - r), (x + r, ym + r), (x + r, ym), (x + 2 * r, ym), (x + r, ym), (x + r, ym - r),
             (x + r, y_bottom + r), (x + r, y_bottom), (x, y_bottom)]
    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3,
             Path.CURVE3]
    fig.add_artist(PathPatch(Path(verts, codes), transform=fig.dpi_scale_trans, facecolor="none", edgecolor=color, linewidth=linewidth, capstyle="round"))
    return x + 2 * r


def step_legend(fig, x, y, steps, ncol):
    """
    Legend of the step colors (one patch per step) at figure position (x, y), upper left.

    Args:
        fig: matplotlib Figure
        x, y: float
            Figure coordinates of the upper left corner.
        steps: sequence of string
            The steps.
        ncol: int
            Columns.
    """
    handles = [Patch(facecolor=STEP_COLOR[s], label=STEP_LABEL.get(s, s)) for s in steps]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(x, y), ncol=ncol, columnspacing=1.3, handlelength=1.2, handletextpad=0.5, fontsize=9.2)


def texts_outside(fig):
    """
    The visible texts of a figure that are not inside it.

    Args:
        fig: matplotlib Figure
            A drawn figure.

    Returns:
        texts: list of string
            The strings that stick out (empty if all fit).
    """
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    renderer = fig.canvas.get_renderer()
    # The tick labels that are drawn: the labels of ticks outside the view limits exist (the locator makes them) but are not drawn
    all_ticks, drawn_ticks = set(), set()
    for axis in fig.findobj(matplotlib.axis.Axis):
        low, high = sorted(axis.get_view_interval())
        margin = 1e-6 * max(abs(high - low), 1e-12)
        for tick in axis.majorTicks + axis.minorTicks:
            all_ticks.update([tick.label1, tick.label2])
            if low - margin <= tick.get_loc() <= high + margin:
                drawn_ticks.update([tick.label1, tick.label2])
    out = []
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text().strip() or (t in all_ticks and t not in drawn_ticks):
            continue
        box = t.get_window_extent(renderer)
        if box.x0 < -1 or box.y0 < -1 or box.x1 > width + 1 or box.y1 > height + 1:
            out.append(t.get_text().replace("\n", " / ")[:60])
    return out


def save(fig, outdir, name):
    """
    Write a figure as a 24-bit RGB PNG (no alpha channel); texts outside the figure are reported (an error when STRICT).

    Args:
        fig: matplotlib Figure
        outdir: string
            Output folder.
        name: string
            File name without extension.

    Returns:
        path: string

    Raises:
        ValueError: STRICT and a text sticks out of the figure.
    """
    outside = texts_outside(fig)
    if outside:
        message = f"{name}: text outside the figure: {outside}"
        if STRICT:
            raise ValueError(message)
        print(f"WARNING: {message}", file=sys.stderr)
    rgba = np.asarray(fig.canvas.buffer_rgba())
    path = os.path.join(outdir, f"{name}.png")
    Image.fromarray(rgba[..., :3]).save(path, optimize=True)
    plt.close(fig)
    return path


def nice_ticks(high, nbins=5):
    """
    Round tick values from 0 up to the last one that is not beyond high.

    Args:
        high: float
            The end of the axis.
        nbins: int (default: 5)
            About this many intervals.

    Returns:
        ticks: list of float
    """
    ticks = matplotlib.ticker.MaxNLocator(nbins=nbins, steps=[1, 2, 2.5, 5, 10]).tick_values(0, high)
    return [float(t) for t in ticks if 0 <= t <= high * 1.0000001] or [0.0]


########################################################################################################################
# The numbers of the figures from the JSON
########################################################################################################################
def check_schema(data):
    """
    The JSON of the collector has what the figures need.

    Args:
        data: dictionary
            Loaded JSON.

    Raises:
        ValueError: a key is missing (the message lists them).
    """
    missing = []
    if data.get("schema") != SCHEMA:
        missing.append(f"schema == {SCHEMA}")
    derived = data.get("derived") or {}
    for key in ("t0", "track_jobs", "stats_job", "node_hours", "wall_h", "track_start_spread_min", "stats_wait_min"):
        if key not in derived:
            missing.append(f"derived.{key}")
    for job in derived.get("track_jobs") or []:
        for key in ("name", "month", "start_h", "end_h", "elapsed_min", "nodes", "steps_min"):
            if key not in job:
                missing.append(f"derived.track_jobs[{job.get('name')}].{key}")
    stats = derived.get("stats_job") or {}
    for key in ("name", "start_h", "end_h", "elapsed_min", "nodes", "setup_min", "steps_min"):
        if key not in stats:
            missing.append(f"derived.stats_job.{key}")
    if "campaign" not in data or "tag" not in (data.get("campaign") or {}):
        missing.append("campaign.tag")
    if missing:
        raise ValueError("the campaign JSON lacks: " + ", ".join(missing))


def campaign_rows(derived):
    """
    The numbers of the rows of the campaign figure and of the other figures.

    Args:
        derived: dictionary
            data['derived'].

    Returns:
        c: dictionary
            jobs (track jobs), stats, nodes (stats job), per_month {step: array of minutes of the months}, track {step: (node-hours, wall minutes of
            the mean job)}, stats_rows [(step, node-hours, wall minutes)] (setup, trackstats ... speed), post_rows [(step, node-hours, wall minutes)]
            (dense_netcdf, rainmap, trackpcp: the LAST run of each step supplies the numbers, whether it is the stats job's own or a job of its own with the nodes of
            that job; the earlier runs are superseded), post_nodes (the node counts of the drawn runs), post_source {step: where it ran},
            post_job_of {step: name of the post job that ran it, None for the stats job}, drawn_post_jobs (names of the post jobs that supply a drawn
            step), post_scale (the nodes the post row stands for), post_jobs, totals (track_nh, stats_nh, post_nh, charged_nh = the node-hours of the
            work, superseded runs included), superseded_nh (node-hours of the runs that a later run of the same step replaces: charged, not drawn),
            other_nh (charged minus drawn minus superseded), job_min_mean/sd (the three track steps together), track_longest_min (the longest track job),
            stats_compute_min (the stats job without the hung time and without the steps that a post job re-ran), post_compute_min (the post jobs
            that supply drawn steps), compute_min (the critical path without queue waits: the three together), speed_end_h (hours since the first start).
    """
    jobs, stats = derived["track_jobs"], derived["stats_job"]
    post_jobs = derived.get("post_jobs") or []
    steps = [s for s in TRACK_STEPS if any(s in j["steps_min"] for j in jobs)]
    per_month = {s: np.array([j["steps_min"].get(s, 0.0) for j in jobs], dtype=float) for s in steps}
    together = sum(per_month.values()) if per_month else np.zeros(len(jobs))
    # Node-hours of a step: its minutes x the nodes of the job / 60, summed over the jobs
    track = {s: (sum(j["steps_min"].get(s, 0.0) * j["nodes"] for j in jobs) / 60.0, float(per_month[s].mean())) for s in steps}
    nodes = stats["nodes"]
    stats_rows = [("setup", stats["setup_min"] * nodes / 60.0, stats["setup_min"])]
    stats_rows += [(s, stats["steps_min"][s] * nodes / 60.0, stats["steps_min"][s]) for s in STATS_STEPS if s in stats["steps_min"]]
    # The post-processing steps: every run of a step (the stats job's own and those of jobs of their own, each with the nodes of the job that ran it) in
    # time order; the LAST run supplies the numbers that are drawn, the earlier runs are superseded (their node-hours stay charged)
    runs = {}
    for s in POST_STEPS:
        if s in stats["steps_min"]:
            runs.setdefault(s, []).append((stats["start_h"], stats["steps_min"][s] * nodes / 60.0, stats["steps_min"][s], nodes, f"stats job, {nodes} nodes", None))
    for job in post_jobs:
        for s in (*STATS_STEPS, *POST_STEPS):
            if s in job["steps_min"]:
                where = f"post job, {nodes_text([job['nodes']])}{', interactive' if job.get('interactive') else ''}"
                runs.setdefault(s, []).append((job["start_h"], job["steps_min"][s] * job["nodes"] / 60.0, job["steps_min"][s], job["nodes"], where, job["name"]))
    post_rows, post_source, post_job_of, post_nodes, superseded_nh = [], {}, {}, set(), 0.0
    for s, step_runs in runs.items():
        step_runs.sort(key=lambda r: r[0])
        last = step_runs[-1]
        post_rows.append((s, last[1], last[2]))
        post_source[s], post_job_of[s] = last[4], last[5]
        post_nodes.add(last[3])
        superseded_nh += sum(r[1] for r in step_runs[:-1])
    post_rows.sort(key=lambda r: POST_STEPS.index(r[0]) if r[0] in POST_STEPS else len(POST_STEPS))
    drawn_post_jobs = sorted({name for name in post_job_of.values() if name})
    # The nodes the post row stands for on the wall-clock axis: node-hours over hours (the count itself if every step ran on the same number)
    post_wall_h = sum(r[2] for r in post_rows) / 60.0
    post_scale = sum(r[1] for r in post_rows) / post_wall_h if post_wall_h > 0 else float(nodes)
    totals = {"track_nh": sum(v[0] for v in track.values()), "stats_nh": sum(r[1] for r in stats_rows), "post_nh": sum(r[1] for r in post_rows)}
    nh = derived["node_hours"]
    charged = nh["track"] + nh["stats"] + nh.get("post", 0.0)
    drawn = totals["track_nh"] + totals["stats_nh"] + totals["post_nh"]
    speed_end_h = stats["start_h"] + sum(r[2] for r in stats_rows) / 60.0
    # The compute time: the longest track job, the stats job (not its hung time, not the steps that a post job ran again), the post jobs that supply
    # drawn steps; queue waits are not part of it
    track_longest = max(j["elapsed_min"] for j in jobs)
    rerun_in_stats = [s for s in POST_STEPS if s in stats["steps_min"] and not post_source.get(s, "").startswith("stats job")]
    stats_compute = stats["setup_min"] + sum(v for s, v in stats["steps_min"].items() if s not in rerun_in_stats)
    if not stats.get("incomplete"):
        stats_compute += stats.get("other_min", 0.0)
    post_compute = sum(j["elapsed_min"] for j in post_jobs if j["name"] in drawn_post_jobs)
    return {"jobs": jobs, "stats": stats, "nodes": nodes, "per_month": per_month, "track": track, "stats_rows": stats_rows, "post_rows": post_rows,
            "totals": dict(totals, charged_nh=charged), "job_min_mean": float(together.mean()), "job_min_sd": float(together.std(ddof=1)) if len(jobs) > 1 else 0.0,
            "speed_end_h": speed_end_h, "other_nh": charged - drawn - superseded_nh, "superseded_nh": superseded_nh, "post_nodes": sorted(post_nodes),
            "post_jobs": post_jobs, "post_source": post_source, "post_job_of": post_job_of, "drawn_post_jobs": drawn_post_jobs, "post_scale": post_scale,
            "track_longest_min": track_longest, "stats_compute_min": stats_compute, "post_compute_min": post_compute,
            "compute_min": track_longest + stats_compute + post_compute}


def nodes_text(nodes):
    """
    The node counts of a row as text.

    Args:
        nodes: list of int
            Node counts (sorted).

    Returns:
        text: string
            '4 nodes', '2 nodes', '2 and 4 nodes'.
    """
    return " and ".join(str(n) for n in nodes) + (" node" if nodes == [1] else " nodes")


def counts_text(data):
    """
    The counts of the output checks as a line for the subtitle.

    Args:
        data: dictionary
            Loaded JSON.

    Returns:
        text: string
            e.g. '8,759 frames, 1.2 million tracks, 40 robust MCS' (only what the JSON has).
    """
    c, parts = data.get("counts") or {}, []
    if c.get("cloudid_files"):
        parts.append(f"{c['cloudid_files']:,} frames")
    for key, name in (("trackstats_tracks", "tracks"), ("identifymcs_mcs_tb", "Tb MCS"), ("robustmcs_mcs_robust", "robust MCS")):
        if c.get(key):
            parts.append(f"{c[key]:,} {name}" if c[key] < 1e6 else f"{c[key] / 1e6:.2f} million {name}")
    return ", ".join(parts)


def slug(name):
    """
    A file name part from a campaign name.

    Args:
        name: string
            e.g. 'E3SM ne120 2009'.

    Returns:
        text: string
            e.g. 'e3sm_ne120_2009'.
    """
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


########################################################################################################################
# Figure 2: the campaign by step, in node-hours and wall-clock time
########################################################################################################################
def fig_campaign(data, outdir, name):
    """
    The campaign as stacked rows split by step: the track jobs (the mean month x the number of jobs, whisker = the SD of the monthly total), the stats
    job up to speed and, below a dashed line, the post-processing of the same allocation. The bars are node-hours (axis at the top); the axes at the
    bottom are the wall-clock time of each row. The numbers in the segments are the wall-clock minutes of the step and its node-hours; a brace gives the
    totals of the tracking (rows 1 and 2), a second one the cost of the post-processing. Steps under a minute are listed in the note.

    Args:
        data: dictionary
            Loaded campaign JSON.
        outdir: string
            Output folder.
        name: string
            File name part.

    Returns:
        path: string
    """
    d = data["derived"]
    c = campaign_rows(d)
    njobs, nodes = len(c["jobs"]), c["nodes"]
    track_nodes = c["jobs"][0]["nodes"]
    # The compute time of the tracking (rows 1 and 2): the longest track job and the stats job up to speed (no queue waits)
    tracking_compute_h = (c["track_longest_min"] + sum(seg[2] for seg in c["stats_rows"])) / 60.0
    own_jobs = any(not src.startswith("stats job") for src in c["post_source"].values())
    rows = [("track", [(s, c["track"][s][0], c["track"][s][1]) for s in c["track"]]), ("stats", c["stats_rows"])]
    if c["post_rows"]:
        rows.append(("post", c["post_rows"]))
    nrow = len(rows)
    figw, row_in, gap_in = 14.0, 1.24, 1.12
    left_in, right_in = 1.75, 0.3
    width_in = figw - left_in - right_in
    # The notes at the bottom: 5 lines, one more for a superseded run, for a cancelled stats job and for the steps under a minute
    n_notes = 5 + (c["superseded_nh"] > 0) + bool(c["stats"].get("incomplete")) + 1
    top_in, bottom_in = 2.05, 0.45 + 0.17 * n_notes
    # The gaps below the rows (the wall-clock axis of the row); the one above the post-processing row also holds the dashed line and its label
    gaps = [gap_in] * (nrow - 1) + [gap_in - 0.5]
    if nrow == 3:
        gaps[1] += 0.62
    figh = top_in + nrow * row_in + sum(gaps[:nrow - 1]) + 0.62 + bottom_in
    fig = plt.figure(figsize=(figw, figh))
    row_totals = [sum(seg[1] for seg in segs) for _, segs in rows]

    # Right-hand labels of the rows: measured, so that the axis is as long as the bars and these texts need
    wall_stats_h = sum(seg[2] for seg in c["stats_rows"]) / 60.0
    wall_post_h = sum(seg[2] for seg in c["post_rows"]) / 60.0
    labels = {"track": (f"{c['job_min_mean']:.1f} min of steps per job", f"({c['totals']['track_nh']:.1f} node-hours: {njobs} jobs × {track_nodes} node)"),
              "stats": (f"{wall_stats_h:.2f} h", f"({c['totals']['stats_nh']:.1f} node-hours: {nodes} nodes)"),
              "post": (f"{wall_post_h:.2f} h", f"({c['totals']['post_nh']:.1f} node-hours: {nodes_text(c['post_nodes'])})")}
    sd_nh = njobs * c["job_min_sd"] / 60.0 * track_nodes
    label_in = {}
    for key, (big, small) in labels.items():
        label_in[key] = max(text_size_in(fig, big, fontsize=10.5, fontweight="bold")[0], text_size_in(fig, small, fontsize=9)[0])
    # The texts right of the braces (their widths set the room that the bars may take): the tracking compute time and node-hours, the post-processing cost
    tracking_nh = c["totals"]["track_nh"] + c["totals"]["stats_nh"]
    with_them = f"with them: {c['compute_min'] / 60.0:.1f} h of compute, {tracking_nh + c['totals']['post_nh']:.1f} node-hours"
    brace_text_in = max(text_size_in(fig, f"{tracking_compute_h:.1f} h of compute", fontsize=13.5, fontweight="bold")[0],
                        text_size_in(fig, "longest track job + stats job up to speed", fontsize=9)[0],
                        text_size_in(fig, "tracking: all jobs, hours × nodes", fontsize=9)[0],
                        text_size_in(fig, f"{tracking_nh:.1f} node-hours", fontsize=13.5)[0],
                        text_size_in(fig, f"+ {wall_post_h:.2f} h, + {c['totals']['post_nh']:.1f} node-hours", fontsize=11.5, fontweight="bold")[0] if c["post_rows"] else 0.0,
                        (text_size_in(fig, with_them, fontsize=9)[0] - 0.18) if c["post_rows"] else 0.0)
    # The axis: every row (its bars + whisker + label) and the brace with its text fit; node-hours per inch follow from that
    need_in = max(row_totals[i] * 0 + label_in[k] + 0.3 for i, (k, _) in enumerate(rows)) + 0.55 + brace_text_in + 0.35
    xmax = max(row_totals) * width_in / max(width_in - need_in, 4.0)
    inch_per_unit = width_in / xmax
    # The rows by their lower edge (from the top)
    axes, y_tops = [], []
    for i in range(nrow):
        y_top = figh - top_in - i * row_in - sum(gaps[:i])
        y_tops.append(y_top)
        axes.append(fig.add_axes([left_in / figw, (y_top - row_in) / figh, width_in / figw, row_in / figh]))
    ylim, bar_h = (-1.12, 0.5), 0.62
    narrow = {}

    def draw_row(ax, segs, key):
        """Stacked segments of one row with the minutes and node-hours in them (a call-out below the bar where the segment is too narrow)."""
        left, short = 0.0, []
        for step, nh, wall_min in segs:
            bar(ax, 0, left, nh, bar_h, STEP_COLOR[step])
            minutes_text = f"{wall_min:.0f} min"
            hours_text = f"({nh:.1f} node-h)"
            if wall_min < 1.0:
                short.append(f"{STEP_LABEL.get(step, step)} {wall_min * 60:.0f} s")
            elif nh * inch_per_unit >= 0.66:
                ax.text(left + nh / 2, 0, f"{minutes_text}\n{hours_text}", ha="center", va="center", fontsize=7.7, color=text_color(STEP_COLOR[step]),
                        linespacing=1.25, zorder=5)
            else:
                narrow.setdefault(key, []).append((left + nh / 2, minutes_text, hours_text))
            left += nh
        return left, short

    ends, shorts = [], []
    for ax, (key, segs) in zip(axes, rows):
        end, short = draw_row(ax, segs, key)
        ends.append(end)
        shorts += short
    # Call-outs below the bars of the narrow segments: left to right, shifted to the right where two would overlap, with a leader line
    for ax, (key, _) in zip(axes, rows):
        right_edge = -1e9
        for xc, t1, t2 in sorted(narrow.get(key, [])):
            w_units = text_size_in(fig, f"{t1}\n{t2}", fontsize=8.0, linespacing=1.25)[0] / inch_per_unit
            xl = max(xc - w_units / 2, right_edge + 0.1 / inch_per_unit)
            ax.plot([xc, xl + w_units / 2 if xl > xc - w_units / 2 else xc], [-bar_h / 2, -0.66], color=INK2, linewidth=0.8, zorder=4)
            ax.text(xl + w_units / 2, -0.7, f"{t1}\n{t2}", ha="center", va="top", fontsize=8.0, color=INK, linespacing=1.25)
            right_edge = xl + w_units
    # The whisker of the track row: the SD of the monthly total x the jobs
    if njobs > 1:
        axes[0].errorbar([ends[0]], [0.42], xerr=[sd_nh], fmt="none", ecolor=INK, elinewidth=1.3, capsize=4, capthick=1.3, zorder=6)
        axes[0].text(ends[0] - sd_nh - 0.1 / inch_per_unit, 0.42, f"whisker: {njobs} × the SD of the monthly total", ha="right", va="center", fontsize=8, color=INK2)
    for ax, (key, _), end in zip(axes, rows, ends):
        x0 = end + (sd_nh if key == "track" else 0.0) + 0.25 / inch_per_unit
        ax.text(x0, 0.1, labels[key][0], ha="left", va="center", fontsize=10.5, color=INK, fontweight="bold")
        ax.text(x0, -0.16, labels[key][1], ha="left", va="center", fontsize=9, color=INK)

    for ax in axes:
        ax.set_xlim(0, xmax)
        ax.set_ylim(*ylim)
        ax.set_yticks([0])
        ax.set_yticklabels([])
        style(ax, grid="x")
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
    row_names = {"track": f"{njobs} track jobs\n{track_nodes} node each", "stats": f"stats job\n{nodes} nodes",
                 "post": f"post-\nprocessing\n{nodes_text(c['post_nodes'])}"}
    for ax, (key, _) in zip(axes, rows):
        ax.set_yticklabels([row_names[key]], fontsize=10)
    # The node-hour axis at the top of the first row
    axes[0].xaxis.tick_top()
    axes[0].xaxis.set_label_position("top")
    axes[0].tick_params(axis="x", top=True, labeltop=True)
    axes[0].set_xlabel("node-hours (hours × nodes: what the jobs are charged)", color=INK2)
    axes[0].set_xticks([t for t in nice_ticks(xmax, 8) if t <= xmax])
    # The wall-clock axis of every row at its bottom: the hours (minutes below 1.5 h) of one job / of the allocation
    scales = {"track": njobs * track_nodes, "stats": nodes, "post": c["post_scale"]}
    clock_names = {"track": f"of one track job (the {njobs} jobs run side by side)", "stats": f"of the stats job ({nodes} nodes)",
                   "post": f"of the post-processing ({nodes_text(c['post_nodes'])}, after the steps above)"}
    for ax, (key, _), end in zip(axes, rows, ends):
        total_h = end / scales[key]
        unit, factor = ("minutes", 60.0) if total_h < 1.5 else ("hours", 1.0)
        sec = ax.secondary_xaxis("bottom", functions=(lambda x, s=scales[key], f=factor: x / s * f, lambda h, s=scales[key], f=factor: h * s / f))
        # At most one tick per 0.6 inch of the row's own length (a short row, e.g. a few minutes of post-processing, has two or three ticks)
        ticks = nice_ticks(total_h * factor, int(min(5, max(1, end * inch_per_unit / 0.6))))
        sec.set_xticks(ticks)
        sec.set_xlabel(f"wall-clock {unit} {clock_names[key]}", color=INK2, loc="left")
        sec.spines["bottom"].set_color("#c9c8c3")
        sec.spines["bottom"].set_bounds(0, max(ticks) / factor * scales[key])
        sec.tick_params(length=3, colors=INK2)

    # A dashed line above the post-processing row
    fig.canvas.draw()
    if c["post_rows"]:
        what = ", ".join(STEP_WHAT[r[0]] for r in c["post_rows"])
        y_line = y_tops[2] + 0.30
        fig.add_artist(Line2D([0.012, 0.99], [y_line / figh] * 2, transform=fig.transFigure, color=INK2, linewidth=0.9, linestyle=(0, (5, 4))))
        if own_jobs:
            why = (", after the stats job was cancelled in the rain maps" if c["stats"].get("incomplete") else
                   ", a re-run with the fixed rain-map step" if c["superseded_nh"] > 0 else "")
            post_label = f"post-processing (optional steps) in a job of its own{why}: {what}"
        else:
            post_label = f"post-processing in the same allocation, after the steps above (optional steps): {what}"
        fig.text(0.012, (y_line + 0.06) / figh, post_label, ha="left", va="bottom", fontsize=9.3, color=INK2, style="italic")

    # Braces right of the rows: the tracking campaign (rows 1 and 2), then the cost of the post-processing
    inv = fig.dpi_scale_trans.inverted()
    x_b = max(ends[i] + (sd_nh if rows[i][0] == "track" else 0.0) + (label_in[rows[i][0]] + 0.3) / inch_per_unit for i in range(nrow)) + 0.1 / inch_per_unit
    top_pt = inv.transform(axes[0].transData.transform((x_b, bar_h / 2)))
    bot_pt = inv.transform(axes[1].transData.transform((x_b, -bar_h / 2)))
    tip_x = curly_brace(fig, top_pt[0], bot_pt[1], top_pt[1], 0.34, INK2)
    mid_y = 0.5 * (top_pt[1] + bot_pt[1])
    t = fig.dpi_scale_trans
    for dy, text, kwargs in ((0.22, f"{tracking_compute_h:.1f} h of compute", dict(fontsize=13.5, fontweight="bold", color=INK)),
                             (-0.06, "longest track job + stats job up to speed", dict(fontsize=9, color=INK2)),
                             (-0.40, f"{tracking_nh:.1f} node-hours", dict(fontsize=13.5, color=INK)),
                             (-0.68, "tracking: all jobs, hours × nodes", dict(fontsize=9, color=INK2))):
        fig.add_artist(matplotlib.text.Text(tip_x + 0.18, mid_y + dy, text, transform=t, ha="left", va="center", **kwargs))
    if c["post_rows"]:
        top3 = inv.transform(axes[2].transData.transform((x_b, bar_h / 2)))
        bot3 = inv.transform(axes[2].transData.transform((x_b, -bar_h / 2)))
        tip3 = curly_brace(fig, top3[0], bot3[1], top3[1], 0.2, INK2)
        mid3 = 0.5 * (top3[1] + bot3[1])
        fig.add_artist(matplotlib.text.Text(tip3 + 0.18, mid3 + 0.10, f"+ {wall_post_h:.2f} h, + {c['totals']['post_nh']:.1f} node-hours", transform=t, ha="left",
                                            va="center", fontsize=11.5, fontweight="bold", color=INK))
        fig.add_artist(matplotlib.text.Text(tip3 + 0.18, mid3 - 0.17, with_them, transform=t, ha="left", va="center", fontsize=9, color=INK2))

    all_steps = [s for s in (*TRACK_STEPS, "setup", *STATS_STEPS) if any(s == r[0] for _, segs in rows[:2] for r in segs)]
    step_legend(fig, left_in / figw, 1 - 1.1 / figh, all_steps, ncol=len(all_steps))
    if c["post_rows"]:
        step_legend(fig, left_in / figw, 1 - 1.4 / figh, [r[0] for r in c["post_rows"]], ncol=len(c["post_rows"]))
    pm = c["per_month"]
    note = [f"Track jobs: the mean of the {njobs} months for one job (± SD): " + ", ".join(f"{s} {pm[s].mean():.1f} ± {pm[s].std(ddof=1) if njobs > 1 else 0:.1f}" for s in pm) +
            f" min, together {c['job_min_mean']:.1f} ± {c['job_min_sd']:.1f}; the bar is {njobs} × these means, the whisker {njobs} × the SD of the monthly total.",
            "In the segments: the wall-clock minutes of the step and (its node-hours). Wall-clock times are the run times of the jobs; queue waits are not counted.",
            "Node-hours count every node of a job during every step, also the serial ones of the stats job.",
            f"The step times drawn add to {c['totals']['track_nh'] + c['totals']['stats_nh'] + c['totals']['post_nh']:.1f} of the {c['totals']['charged_nh']:.1f} node-hours charged; "
            f"the rest ({c['other_nh']:.1f}) is the start and shutdown of the jobs and the checks between the steps. node-h = node-hours."]
    started = (f"the {njobs} track jobs started over {d['track_start_spread_min']:.0f} min as nodes became free" if d["track_start_spread_min"] >= 1.0
               else f"the {njobs} track jobs all started at the same time")
    note.append(f"Compute time = the longest track job + the stats job (+ the post job): {c['compute_min'] / 60.0:.2f} h. Queue waits are not in it: the stats job waited "
                f"{d['stats_wait_min']:.0f} min for its {nodes} nodes; {started}.")
    if c["superseded_nh"] > 0:
        note.append(f"Superseded: an earlier run of the post-processing steps ({c['superseded_nh']:.1f} node-hours) is charged but not drawn; the figure shows the "
                    f"{nodes_text(c['post_nodes'])} re-run.")
    if c["stats"].get("incomplete"):
        s_ = c["stats"]
        note.append(f"The stats job was cancelled after {s_['elapsed_min'] / 60:.2f} h in a hung step: the {s_['other_min'] / 60:.2f} h after its last finished step are not drawn "
                    f"({s_['other_min'] * nodes / 60:.1f} node-hours, charged as failed); the steps it did not finish ran afterwards in jobs of their own.")
    if shorts:
        note.append("Steps under a minute are not labeled: " + ", ".join(shorts) + ".")
    fig.text(0.012, 0.15 / figh, "\n".join(note), ha="left", va="bottom", fontsize=8.3, color=INK2, linespacing=1.45)
    sub = counts_text(data)
    nh = d["node_hours"]
    cancelled = f" (cancelled after {c['stats']['elapsed_min'] / 60:.2f} h)" if c["stats"].get("incomplete") else ""
    post_job = f" and a post-processing job on {nodes_text(c['post_nodes'])}" if own_jobs else ""
    failed = f" ({nh['failed']:.1f} of them in failed attempts)" if nh["failed"] > 0 else ""
    title(fig, f"MCS tracking, {data['campaign']['tag']}: compute time and node-hours by step",
          (f"{sub}.\n" if sub else "") + f"{njobs} track jobs and one stats job on {nodes} nodes{cancelled}{post_job}; {nh['total']:.1f} node-hours charged in all{failed}.")
    return save(fig, outdir, f"perf_tbpf_{name}_campaign")


########################################################################################################################
# Figure 10: the campaign in compute time
########################################################################################################################
def timeline_layout(c):
    """
    The geometry of the compute-time timeline, in hours: every track job starts at 0 (they run side by side), the stats job starts when the longest track
    job ends, and the steps of the post-processing jobs follow it. Queue waits and the hung time of a cancelled stats job are not part of it.

    Args:
        c: dictionary
            campaign_rows result.

    Returns:
        layout: dictionary
            track_rows [{name, month, elapsed_min, segments [(step, start_h, length_h)]}] (setup, the steps, what is left of the job: the gray 'setup'),
            stats_segments [(step, start_h, length_h)] (the stats job, then the post jobs that supply drawn steps), stats_start_h, end_h (end of the
            stats row) and post_span (start_h, end_h) of the post jobs' segments (None without them).
    """
    track_rows = []
    for job in c["jobs"]:
        left, segs = 0.0, []
        for step, minutes in [("setup", job.get("setup_min", 0.0))] + [(s, job["steps_min"].get(s, 0.0)) for s in TRACK_STEPS]:
            if minutes > 0:
                segs.append((step, left, minutes / 60.0))
                left += minutes / 60.0
        rest = job["elapsed_min"] / 60.0 - left
        if rest > 1e-9:
            segs.append(("setup", left, rest))
        track_rows.append({"name": job["name"], "month": job.get("month"), "elapsed_min": job["elapsed_min"], "segments": segs})
    stats = c["stats"]
    start = c["track_longest_min"] / 60.0
    state = {"left": start, "segs": []}

    def add(step, minutes):
        if minutes and minutes > 0:
            state["segs"].append((step, state["left"], minutes / 60.0))
            state["left"] += minutes / 60.0

    # The stats job: setup, the steps up to speed, the post steps it ran itself (and kept), its shutdown (a cancelled job has none: its hung time is not drawn)
    add("setup", stats["setup_min"])
    for s in STATS_STEPS:
        if s in stats["steps_min"]:
            add(s, stats["steps_min"][s])
    for s in POST_STEPS:
        if s in stats["steps_min"] and c["post_source"].get(s, "").startswith("stats job"):
            add(s, stats["steps_min"][s])
    if not stats.get("incomplete"):
        add("setup", stats.get("other_min", 0.0))
    post_start = state["left"]
    # The post jobs that supply drawn steps, in time order: setup, their steps, shutdown
    for job in sorted((j for j in c["post_jobs"] if j["name"] in c["drawn_post_jobs"]), key=lambda j: j["start_h"]):
        add("setup", job.get("setup_min", 0.0))
        for s in (*STATS_STEPS, *POST_STEPS):
            if c["post_job_of"].get(s) == job["name"] and s in job["steps_min"]:
                add(s, job["steps_min"][s])
        add("setup", job.get("other_min", 0.0))
    post_span = (post_start, state["left"]) if state["left"] > post_start + 1e-9 else None
    return {"track_rows": track_rows, "stats_segments": state["segs"], "stats_start_h": start, "end_h": state["left"], "post_span": post_span}


def fig_timeline(data, outdir, name):
    """
    The campaign in compute time: the 12 track jobs side by side from 0, every job split into its steps (the gray is setup and shutdown), the stats job
    from the end of the longest track job, followed by the post steps. The axis is the run time of the jobs only: queue waits (they differ from day to day)
    and the hung time of a cancelled job are not drawn.

    Args:
        data: dictionary
            Loaded campaign JSON.
        outdir: string
            Output folder.
        name: string
            File name part.

    Returns:
        path: string
    """
    d = data["derived"]
    c = campaign_rows(d)
    lay = timeline_layout(c)
    stats, nrow = c["stats"], len(lay["track_rows"])
    figw, figh = 11.5, 2.75 + 0.3 * (nrow + 1)
    fig = plt.figure(figsize=(figw, figh))
    ax = fig.add_axes([1.45 / figw, 0.95 / figh, (figw - 1.45 - 0.25) / figw, (figh - 0.95 - 1.9) / figh])
    end_h = lay["end_h"]
    for i, row in enumerate(lay["track_rows"]):
        y = nrow + 1 - i
        for step, start, length in row["segments"]:
            bar(ax, y, start, length, 0.74, STEP_COLOR[step])
        ax.text(row["elapsed_min"] / 60.0 + 0.01 * end_h, y, f"{row['elapsed_min']:.0f} min", ha="left", va="center", fontsize=8.2, color=INK2)
    # The stats row, from the end of the longest track job (a short dotted line joins the stats row to the track rows above it)
    ax.plot([lay["stats_start_h"]] * 2, [0.5, 1.6], color=INK2, linewidth=0.8, linestyle=(0, (2, 3)), zorder=1)
    for step, start, length in lay["stats_segments"]:
        bar(ax, 0, start, length, 0.9, STEP_COLOR[step])
        wide = length / max(end_h, 1e-9) * (figw - 1.7) >= 0.62
        if step != "setup" and wide:
            ax.text(start + length / 2, 0, f"{STEP_LABEL.get(step, step)}\n{length * 60:.0f} min", ha="center", va="center", fontsize=7.4, color=text_color(STEP_COLOR[step]),
                    linespacing=1.15)
    row_nodes = sorted({c["nodes"]} | set(c["post_nodes"]))
    ax.text(end_h + 0.01 * end_h, 0, f"{end_h - lay['stats_start_h']:.2f} h\n{nodes_text(row_nodes)}", ha="left", va="center", fontsize=9.5, color=INK, fontweight="semibold")
    # A bracket under the post steps: they are a job of their own (a re-run)
    if lay["post_span"]:
        x0, x1 = lay["post_span"]
        ax.plot([x0, x1], [-0.62, -0.62], color=INK2, linewidth=0.9, zorder=4)
        ax.text((x0 + x1) / 2, -0.7, f"job of its own, {nodes_text(c['post_nodes'])}", ha="center", va="top", fontsize=8, color=INK2)
    ax.set_yticks(list(range(nrow + 1, 1, -1)) + [0])
    ax.set_yticklabels([r["month"] or r["name"] for r in lay["track_rows"]] + ["stats job" + ("\n+ post-processing" if lay["post_span"] or c["post_rows"] else "")], fontsize=9)
    ax.set_ylim(-1.25, nrow + 1.7)
    ax.set_xlim(0, end_h * 1.16)
    ax.set_xlabel("compute time (h): run times of the jobs only, queue waits removed")
    style(ax, grid="x")
    ax.spines["left"].set_visible(False)
    used = {seg[0] for row in lay["track_rows"] for seg in row["segments"]} | {seg[0] for seg in lay["stats_segments"]}
    legend_steps = [s for s in (*TRACK_STEPS, "setup", *STATS_STEPS, *POST_STEPS) if s in used and s in STEP_COLOR]
    step_legend(fig, 1.45 / figw, 1 - 1.3 / figh, legend_steps, ncol=min(len(legend_steps), 7))
    nh = d["node_hours"]
    spread = (f"The track jobs started over {d['track_start_spread_min']:.0f} min in the queue" if d["track_start_spread_min"] >= 1.0 else "The track jobs started together")
    lines = [f"The {nrow} track jobs run side by side from 0; the stats job starts when the longest ends ({c['track_longest_min']:.0f} min). Queue waits are not drawn:",
             f"the stats job waited {d['stats_wait_min']:.0f} min for its {c['nodes']} nodes. {spread}."]
    if stats.get("incomplete"):
        lines.append(f"The stats job was cancelled in the rain maps (hung for {stats['other_min'] / 60:.1f} h, not drawn); the post steps are those of jobs of their own. "
                     f"{c['compute_min'] / 60:.2f} h of compute, {nh['total']:.1f} node-hours charged.")
    elif lay["post_span"]:
        lines.append(f"The post steps are those of jobs of their own (a step that the stats job ran too is drawn once, as its last run). {c['compute_min'] / 60:.2f} h of compute, "
                     f"{nh['total']:.1f} node-hours charged.")
    else:
        lines.append(f"{c['compute_min'] / 60:.2f} h of compute, {nh['total']:.1f} node-hours charged.")
    title(fig, f"MCS tracking, {data['campaign']['tag']}, in compute time: {nrow} track jobs, then one stats job", "\n".join(lines))
    return save(fig, outdir, f"perf_tbpf_{name}_timeline")


########################################################################################################################
# Figure 11: cost by step and the spread over the months
########################################################################################################################
def fig_steps(data, outdir, name):
    """
    Node-hours of every step, ranked (left), and, for the three steps of the track jobs, the spread over the months (right).

    Args:
        data: dictionary
            Loaded campaign JSON.
        outdir: string
            Output folder.
        name: string
            File name part.

    Returns:
        path: string
    """
    d = data["derived"]
    c = campaign_rows(d)
    cost = {s: v[0] for s, v in c["track"].items()}
    where = {s: f"{len(c['jobs'])} track jobs" for s in c["track"]}
    for step, nh, minutes in c["stats_rows"] + c["post_rows"]:
        cost[step] = nh
        place = c["post_source"].get(step, "stats job")
        where[step] = f"{place}, {minutes:.0f} min wall" if minutes >= 1 else f"{place}, {minutes * 60:.0f} s wall"
    total = sum(cost.values())
    order = sorted(cost, key=lambda s: -cost[s])
    figw, figh = 11.8, 3.2 + 0.22 * len(order)
    fig = plt.figure(figsize=(figw, figh))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], left=0.115, right=0.985, bottom=0.14, top=1 - 1.35 / figh, wspace=0.42)
    ax, axm = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ypos = np.arange(len(order))[::-1]
    for y, step in zip(ypos, order):
        bar(ax, y, 0, cost[step], 0.62, STEP_COLOR[step])
        ax.text(cost[step] + 0.012 * max(cost.values()), y, f"{cost[step]:.1f} h   {cost[step] / total * 100:.0f} %", ha="left", va="center", fontsize=9.5, color=INK,
                fontweight="semibold")
        ax.text(cost[step] + 0.012 * max(cost.values()), y - 0.3, where[step], ha="left", va="center", fontsize=8, color=INK2)
    ax.set_yticks(ypos)
    ax.set_yticklabels([STEP_LABEL.get(s, s) for s in order])
    ax.set_xlim(0, max(cost.values()) * 1.55)
    ax.set_xlabel("node-hours")
    ax.set_title(f"cost of each step ({total:.1f} node-hours in all)")
    style(ax, grid="x")
    ax.spines["left"].set_visible(False)
    # The spread of the three track-job steps over the months
    rng = np.random.default_rng(3)
    steps = list(c["per_month"])
    xhigh = max(max(v.max(), v.mean() + (v.std(ddof=1) if len(v) > 1 else 0.0)) for v in c["per_month"].values())
    for k, step in enumerate(steps[::-1]):
        vals = c["per_month"][step]
        jitter = np.linspace(-0.2, 0.2, len(vals))[rng.permutation(len(vals))]
        axm.plot(vals, k + jitter - 0.08, marker="o", markersize=7, linestyle="none", color=STEP_COLOR[step], markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=4)
        mean, sd = vals.mean(), vals.std(ddof=1) if len(vals) > 1 else 0.0
        axm.errorbar([mean], [k + 0.36], xerr=[sd], fmt="D", markersize=6, color=INK, ecolor=INK, elinewidth=1.4, capsize=3.5, zorder=5)
        axm.text(max(vals.max(), mean + sd) + 0.03 * xhigh, k - 0.08, f"{mean:.1f} ± {sd:.1f} min", ha="left", va="center", fontsize=9, color=INK)
    axm.set_yticks(range(len(steps)))
    axm.set_yticklabels(steps[::-1])
    axm.set_ylim(-0.5, len(steps) - 0.35)
    axm.set_xlim(0, xhigh * 1.5)
    axm.set_xlabel("minutes of one track job")
    axm.set_title(f"the {len(c['jobs'])} months (dot = a month)")
    style(axm, grid="x")
    axm.spines["left"].set_visible(False)
    title(fig, f"MCS tracking, {data['campaign']['tag']}: cost by step and its spread over the months",
          f"Node-hours: the track jobs {c['jobs'][0]['nodes']} node each, the stats job its wall time × {c['nodes']} nodes"
          f"{', the steps of jobs of their own the nodes of their job' if any(not v.startswith('stats job') for v in c['post_source'].values()) else ''}."
          "\nRight: the dots are the months, the diamond is the mean ± SD.")
    return save(fig, outdir, f"perf_tbpf_{name}_steps")


FIGURES = {"campaign": fig_campaign, "timeline": fig_timeline, "steps": fig_steps}


def make_figures(data, outdir, name=None, only=None):
    """
    The figures of one campaign.

    Args:
        data: dictionary
            Loaded campaign JSON (collect_campaign_performance.py).
        outdir: string
            Output folder (made if needed).
        name: string (default: None)
            File name part (default: the slug of the campaign tag).
        only: list of string (default: None)
            Make only these figures (keys of FIGURES).

    Returns:
        paths: list of string
            The PNG files.

    Raises:
        ValueError: the JSON lacks keys.
    """
    check_schema(data)
    os.makedirs(outdir, exist_ok=True)
    name = name or slug(data["campaign"]["tag"])
    # The look of these figures (surface and ink colors, fonts, grid) is set for the figures only: a program that calls this function (a test session,
    # a notebook) keeps its own matplotlib settings
    with plt.rc_context():
        setup()
        return [FIGURES[key](data, outdir, name) for key in (only or FIGURES)]


def main(argv=None):
    """
    Command line: make the figures of one or more campaigns.

    Args:
        argv: list of strings (default: None)
            Arguments (None: sys.argv[1:]).

    Returns:
        code: int
            0.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", nargs="+", required=True, help="JSON file(s) of Analysis/collect_campaign_performance.py")
    parser.add_argument("--outdir", default="perf_figures", help="output folder (default: %(default)s; not the repository: the images are docs-assets)")
    parser.add_argument("--name", nargs="+", default=None, help="file name part of each campaign (default: the slug of its tag)")
    parser.add_argument("--only", nargs="+", choices=sorted(FIGURES), help="make only these figures")
    args = parser.parse_args(argv)
    if args.name and len(args.name) != len(args.data):
        parser.error("--name needs one name per --data file")
    for i, path in enumerate(args.data):
        with open(path) as f:
            data = json.load(f)
        for png in make_figures(data, args.outdir, args.name[i] if args.name else None, args.only):
            print(png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
