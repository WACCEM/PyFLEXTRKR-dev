"""
A synthetic campaign in the format of Analysis/collect_campaign_performance.py (the 'derived' part that Analysis/make_performance_tbpf_figures.py
reads), with numbers of the size of a global year on Perlmutter CPU nodes: 12 track jobs of about 10 minutes and a stats job on 4 nodes.
For the tests of the figure script and to look at the figures without a campaign.

    python tests/synthetic_campaign.py OUT.json [--no-post] [--nodes N]
"""
import json
import sys

import numpy as np

SCHEMA = "tbpf_campaign_performance/1"


def campaign_json(n_jobs=12, stats_nodes=4, post=True, failed_node_hours=0.0, seed=7, tag="synthetic 2009"):
    """
    A campaign: n_jobs track jobs (one node each, started over 25 minutes as the queue frees nodes) and a stats job that starts 6 minutes after the last
    track job ends.

    Args:
        n_jobs: int (default: 12)
            Number of track jobs (months).
        stats_nodes: int (default: 4)
            Nodes of the stats job.
        post: bool (default: True)
            The stats job also ran the post-processing steps (rainmap, trackpcp).
        failed_node_hours: float (default: 0.0)
            Node-hours of failed attempts that are charged too.
        seed: int (default: 7)
            Random seed of the month-to-month variation.
        tag: string (default: 'synthetic 2009')
            Name of the campaign.

    Returns:
        data: dictionary
            schema, campaign, derived, counts.
    """
    rng = np.random.default_rng(seed)
    base = {"idfeature": 8.4, "tracksingle": 1.6, "gettracks": 0.7}
    starts = np.sort(rng.uniform(0, 25, n_jobs)) / 60.0
    jobs = []
    for i in range(n_jobs):
        steps = {s: round(float(m * rng.normal(1.0, 0.05)), 3) for s, m in base.items()}
        setup = 0.4
        elapsed = sum(steps.values()) + setup + 0.2
        jobs.append({"name": f"track2009{i + 1:02d}", "month": f"2009-{i + 1:02d}", "start_h": round(float(starts[i]), 4),
                     "end_h": round(float(starts[i] + elapsed / 60.0), 4), "elapsed_min": round(elapsed, 2), "nodes": 1, "steps_min": steps,
                     "setup_min": setup, "other_min": 0.2})
    last_end = max(j["end_h"] for j in jobs)
    stats_start = last_end + 6.0 / 60.0
    steps = {"trackstats": 14.2, "identifymcs": 1.6, "matchpf": 9.3, "robustmcs": 0.2, "mapfeature": 11.0, "speed": 3.4}
    if post:
        steps.update({"rainmap": 5.2, "trackpcp": 14.8})
    setup = 2.1
    elapsed = setup + sum(steps.values()) + 0.9
    stats = {"name": "stats200901_200912", "start_h": round(stats_start, 4), "end_h": round(stats_start + elapsed / 60.0, 4), "elapsed_min": round(elapsed, 2),
             "nodes": stats_nodes, "setup_min": setup, "steps_min": steps, "other_min": 0.9}
    track_nh = sum(j["elapsed_min"] * j["nodes"] for j in jobs) / 60.0
    stats_nh = stats["elapsed_min"] * stats_nodes / 60.0
    derived = {"t0": "2026-11-20T10:00:00-08:00", "track_jobs": jobs, "stats_job": stats,
               "node_hours": {"track": round(track_nh, 3), "stats": round(stats_nh, 3), "failed": failed_node_hours,
                              "total": round(track_nh + stats_nh + failed_node_hours, 3)},
               "wall_h": stats["end_h"], "track_start_spread_min": round(float(starts[-1]) * 60, 1), "stats_wait_min": 6.0, "check_s_total": 12.0}
    counts = {"input_files": 8761, "cloudid_files": 8760, "frames_skipped": 1, "trackstats_tracks": 1180000, "identifymcs_mcs_tb": 41000,
              "robustmcs_mcs_robust": 7600}
    return {"schema": SCHEMA, "campaign": {"tag": tag}, "derived": derived, "counts": counts}


if __name__ == "__main__":
    out = sys.argv[1]
    data = campaign_json(post="--no-post" not in sys.argv, stats_nodes=int(sys.argv[sys.argv.index("--nodes") + 1]) if "--nodes" in sys.argv else 4)
    with open(out, "w") as f:
        json.dump(data, f, indent=1)
    print(out)
