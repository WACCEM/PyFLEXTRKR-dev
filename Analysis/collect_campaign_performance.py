#!/usr/bin/env python
"""
Collect the performance of a Tb+PF MCS tracking campaign (config/generate_tbpf_mcstracking_scripts.py: one track job per month, then one stats
job) into one JSON file, from what the jobs left behind:

  - sacct: job state, start and end, elapsed time, nodes, and the largest memory (MaxRSS) of the batch step (the main process and the local
    workers of a track job; the main process and the scheduler of the stats job) and of the worker step of a multi-node stats job
  - the step ledgers (step_profile_dir, one folder per job: <root>/profile/<job name>/step_ledger.jsonl): wall time, CPU time (effective cores),
    I/O volume and memory of every step; the lines carry the Slurm job id, so the lines of each attempt of a job are told apart
  - the logs of the jobs (<campaign dir>/logs/log_<job name>_<job id>.log): the time between 'Step X finished in' and 'Step X outputs verified'
    (the output check of the step), the end of the stitching, the counts of the checks, the Dask warnings of the stats job
  - the size of the output folders

    python Analysis/collect_campaign_performance.py --campaign-dir CAMPAIGN_DIR --root ROOT_PATH -o campaign.json [--tag "E3SM ne120 2009"]

The job ids come from submissions.jsonl of the campaign folder (written by generate_tbpf_mcstracking_scripts.py --submit); a job that was
submitted again has several attempts, and the figures use the last one that completed (the steps it skipped come from the attempt that ran
them). The output is the input of Analysis/make_performance_tbpf_figures.py, whose three figures need only the 'derived' part. The sacct text
of a finished campaign can be kept (--save-sacct) and read again (--sacct-file), so the numbers can be made again once the job records have aged
out of sacct.

Author: Zhe Feng | zhe.feng@pnnl.gov
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

SCHEMA = "tbpf_campaign_performance/1"
# The steps of a track job and of a stats job, in the order they run (the optional ones last)
TRACK_STEPS = ("idfeature", "tracksingle", "gettracks")
STATS_STEPS = ("trackstats", "identifymcs", "matchpf", "robustmcs", "mapfeature", "speed", "rainmap", "trackpcp")
# The fields of sacct (one text line per job, step of a job: <id>, <id>.batch, <id>.extern, <id>.0)
SACCT_FIELDS = ("JobIDRaw", "JobName", "State", "ExitCode", "Submit", "Eligible", "Start", "End", "ElapsedRaw", "TimelimitRaw", "NNodes",
                "NodeList", "WorkDir", "MaxRSS", "MaxRSSNode", "MaxRSSTask", "AveRSS", "TotalCPU")
# Patterns of the Dask messages (the logs of the Dask scheduler and workers of a stats job, and the log of every job: the LocalCluster of a track
# job logs there) that say the cluster was in trouble (counted, not interpreted; the lease message is a CRITICAL line that Dask prints when a task
# that holds the lease of a file for netCDF writing was busy for longer than the lease timeout: a harmless overbooking of a semaphore of one file)
# ('Event loop was unresponsive in Worker for N s' is an INFO line that Dask prints whenever a task holds the GIL for a few seconds: it is counted
# apart, a worker or scheduler that is unresponsive is a sign of trouble)
DASK_WARNINGS = {"event_loop_unresponsive": r"Event loop was unresponsive", "unresponsive": r"^(?!.*Event loop was unresponsive).*unresponsive",
                 "gather_failed": r"Couldn't gather", "worker_removed": r"Removing worker",
                 "killed_worker": r"KilledWorker", "memory_95": r"95\s*%|memory use is high|exceeded .* memory budget",
                 "comm_closed": r"CommClosedError", "timeout": r"TimeoutError|timed out", "lease_timeout": r"Refreshing an unknown lease ID"}
# What the 'Step X outputs verified: ...' text of every step holds: the counts that the doc reports
VERIFIED_COUNTS = {
    "idfeature": r"(?P<cloudid_files>\d+) cloudid files for (?P<input_files>\d+) input files(?: \((?P<frames_skipped>\d+) frames skipped)?",
    "tracksingle": r"for all (?P<pairs>\d+) pairs of the (?P<cloudid_files>\d+) cloudid files",
    "gettracks": r"(?P<track_numbers>\d+) track numbers",
    "trackstats": r"(?P<tracks>\d+) tracks",
    "identifymcs": r"(?P<mcs_tb>\d+) MCS tracks",
    "matchpf": r"(?P<matched_entries>\d+) of (?P<entries>\d+) entries",
    "robustmcs": r"(?P<mcs_robust>\d+) robust MCS tracks",
    "mapfeature": r"(?P<pixel_files>\d+) pixel files for (?P<cloudid_files>\d+) cloudid files",
    "speed": r"(?P<finite_speeds>\d+) finite speeds",
    "rainmap": r"(?P<monthly_files>\d+) monthly files",
    "trackpcp": r"(?P<files>\d+) files",
}
LOG_LINE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3}) - ([\w.]+) - (\w+) - (.*)$")


########################################################################################################################
# Parsing: sacct, the step ledger, the logs
########################################################################################################################
def parse_time(text):
    """
    A time of sacct (ISO with an offset, e.g. 2026-11-14T09:23:03-0800) or of the ledger (ISO with +00:00) as an aware datetime.

    Args:
        text: string
            The time ('Unknown' and '' give None).

    Returns:
        time: datetime.datetime or None
    """
    if not text or text in ("Unknown", "None"):
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    return datetime.fromisoformat(text)


def parse_rss_gib(text):
    """
    MaxRSS of sacct (--units=M, e.g. '1234M' or '1234.5M') in GiB.

    Args:
        text: string
            The field ('' when the row has no value).

    Returns:
        gib: float or None
    """
    text = (text or "").strip()
    if not text:
        return None
    scale = {"K": 1.0 / 1024 ** 2, "M": 1.0 / 1024, "G": 1.0, "T": 1024.0}
    unit = text[-1].upper()
    try:
        return float(text[:-1]) * scale[unit] if unit in scale else float(text) / 1024 ** 3
    except ValueError:
        return None


def parse_sacct(text):
    """
    The jobs of the text of sacct --parsable2 --noheader with SACCT_FIELDS (without -X, so the steps are there).

    Args:
        text: string
            The output of sacct.

    Returns:
        jobs: dictionary
            {job id: {'job': {field: value} of the allocation, 'steps': {'batch': {...}, '0': {...}}}}.
    """
    jobs = {}
    for line in text.splitlines():
        words = line.rstrip("\n").split("|")
        if len(words) < len(SACCT_FIELDS):
            continue
        row = dict(zip(SACCT_FIELDS, words))
        job_id, _, step = row["JobIDRaw"].partition(".")
        entry = jobs.setdefault(job_id, {"job": None, "steps": {}})
        if step:
            entry["steps"][step] = row
        else:
            entry["job"] = row
    return jobs


def read_ledger(path):
    """
    The lines of a step ledger.

    Args:
        path: string
            step_ledger.jsonl ('' lines are skipped).

    Returns:
        records: list of dictionaries
            Empty if the file does not exist.
    """
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def read_log(path, tz="America/Los_Angeles"):
    """
    The step lines of the log of a job: when each step finished and when its outputs were verified, the end of the stitching, the counts.

    Args:
        path: string
            Log of the job (the python output, with the time of every line in the time zone of the node).
        tz: string (default: 'America/Los_Angeles')
            Time zone of the time stamps of the log lines.

    Returns:
        info: dictionary
            finished: {step: (time, minutes printed)}, verified: {step: (time, text)}, stitched: time or None, the times as UTC datetimes.
    """
    zone = ZoneInfo(tz)
    info = {"finished": {}, "verified": {}, "stitched": None}
    if not os.path.isfile(path):
        return info
    with open(path, errors="replace") as f:
        for line in f:
            m = LOG_LINE.match(line.rstrip("\n"))
            if not m:
                continue
            when = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=zone).astimezone(timezone.utc)
            when = when.replace(microsecond=int(m.group(2)) * 1000)
            message = m.group(5)
            if (done := re.match(r"Step (\w+) finished in ([\d.]+) min", message)):
                info["finished"][done.group(1)] = (when, float(done.group(2)))
            elif (ok := re.match(r"Step (\w+) outputs verified: (.*)", message)):
                info["verified"][ok.group(1)] = (when, ok.group(2))
            elif re.match(r"Stitched \d+ periods into", message):
                info["stitched"] = when
    return info


def verified_counts(step, text):
    """
    The numbers of the 'outputs verified' text of a step.

    Args:
        step: string
            Step name.
        text: string
            The text after 'outputs verified: '.

    Returns:
        counts: dictionary
            {name: int} (empty if the step is not known or the text does not match).
    """
    pattern = VERIFIED_COUNTS.get(step)
    m = re.search(pattern, text) if pattern else None
    return {k: int(v) for k, v in m.groupdict().items() if v is not None} if m else {}


def count_dask_warnings(paths):
    """
    How often the logs of the Dask scheduler and workers show the signs of a cluster in trouble.

    Args:
        paths: list of strings
            Log files (missing ones are skipped).

    Returns:
        counts: dictionary
            {DASK_WARNINGS key: number of lines}.
    """
    counts = {key: 0 for key in DASK_WARNINGS}
    for path in paths:
        if not os.path.isfile(path):
            continue
        with open(path, errors="replace") as f:
            for line in f:
                for key, pattern in DASK_WARNINGS.items():
                    if re.search(pattern, line):
                        counts[key] += 1
    return counts


def folder_size_gb(path):
    """
    Size and number of files of a folder tree.

    Args:
        path: string
            Folder.

    Returns:
        size: dictionary
            {'gb': size [GB], 'files': number of files}; zero for a missing folder.
    """
    total, files = 0, 0
    for root, _, names in os.walk(path):
        for name in names:
            try:
                total += os.path.getsize(os.path.join(root, name))
                files += 1
            except OSError:
                pass
    return {"gb": round(total / 1e9, 3), "files": files}


########################################################################################################################
# One attempt of a job
########################################################################################################################
def step_record(rec, log_minutes=None, check_s=None):
    """
    What the figures and tables use of one step: a ledger line (or only the log's minutes) and the time of its output check.

    Args:
        rec: dictionary or None
            Ledger line of the step.
        log_minutes: float (default: None)
            Minutes of 'Step X finished in N min' (the wall time if the ledger has no line).
        check_s: float (default: None)
            Seconds between 'finished' and 'outputs verified'.

    Returns:
        out: dictionary
            wall_s, status, check_s, and from the ledger: start, end, report_s (the ledger's time minus the step's wall time: the Dask report),
            main_process_only (run_parallel 2: cores, cpu_s, memory and the workers are those of the main process only), cores, cpu_s, read_gb,
            write_gb, tree_rss_gb, main_rss_gb, node_used_gb, worker_max_gb, worker_median_gb, n_workers.
    """
    if rec is None:
        return {"wall_s": None if log_minutes is None else round(log_minutes * 60.0, 2), "status": "ok" if log_minutes is not None else None,
                "check_s": check_s}
    start, end = parse_time(rec.get("start")), parse_time(rec.get("end"))
    worker = rec.get("worker_rss_peak_gb") or {}
    return {
        "wall_s": rec["wall_s"], "status": rec.get("status"), "check_s": check_s, "start": rec.get("start"), "end": rec.get("end"),
        # run_parallel 2: the workers run on other nodes and the ledger only sees the main process (cores, CPU time and memory are its own)
        "main_process_only": rec.get("run_parallel") == 2,
        "report_s": round((end - start).total_seconds() - rec["wall_s"], 1) if start and end else None,
        "cores": rec.get("effective_cores"), "cpu_s": rec.get("tree_cpu_s"), "read_gb": rec.get("read_gb"), "write_gb": rec.get("write_gb"),
        "tree_rss_gb": rec.get("tree_rss_peak_gb"), "main_rss_gb": rec.get("main_rss_peak_gb"), "node_used_gb": rec.get("node_used_peak_gb"),
        "worker_max_gb": worker.get("max"), "worker_median_gb": worker.get("median"), "n_workers": rec.get("n_workers"),
    }


def attempt_of(name, job_id, sacct_job, ledger, log, expected_steps):
    """
    One attempt (one Slurm job) of a job of the campaign.

    Args:
        name: string
            Job name (track200901, stats200901_200912).
        job_id: string
            Slurm job id.
        sacct_job: dictionary or None
            parse_sacct entry of the job.
        ledger: list of dictionaries
            Ledger lines of the folder of the job (all attempts).
        log: dictionary
            read_log of the log of this attempt.
        expected_steps: tuple of strings
            The steps this kind of job runs.

    Returns:
        attempt: dictionary
            job_id, state, exit_code, submit, start, end (ISO), elapsed_s, nodes, timelimit_min, nodelist, max_rss_gib {batch, worker_step},
            steps {step: step_record}, phases {setup_s, stitch_s, dask_start_s, check_s_total}, counts, sacct (the raw rows).
    """
    job = (sacct_job or {}).get("job") or {}
    steps_sacct = (sacct_job or {}).get("steps", {})
    start, end = parse_time(job.get("Start")), parse_time(job.get("End"))
    # The ledger lines of this attempt: by job id (the lines carry it), else those that fall in the time of the job
    mine = [r for r in ledger if r.get("slurm_job_id") == str(job_id)]
    if not mine and start and end:
        mine = [r for r in ledger if (t := parse_time(r.get("start"))) and start <= t <= end]
    steps = {}
    for step in expected_steps:
        recs = [r for r in mine if r["step"] == step]
        rec = recs[-1] if recs else None
        finished = log["finished"].get(step)
        verified = log["verified"].get(step)
        check_s = round((verified[0] - finished[0]).total_seconds(), 3) if finished and verified else None
        if rec is not None or finished is not None:
            steps[step] = step_record(rec, None if finished is None else finished[1], check_s)
            if verified:
                steps[step]["verified"] = verified[1]
    # The phases before the first step: module load, links and stitching of a stats job, the start of the Dask workers
    first = min((parse_time(s["start"]) for s in steps.values() if s.get("start")), default=None)
    phases = {"setup_s": round((first - start).total_seconds(), 1) if first and start else None, "stitch_s": None, "dask_start_s": None,
              "check_s_total": round(sum(s["check_s"] for s in steps.values() if s.get("check_s") is not None), 2)}
    if log["stitched"] and start:
        phases["stitch_s"] = round((log["stitched"] - start).total_seconds(), 1)
        if first:
            phases["dask_start_s"] = round((first - log["stitched"]).total_seconds(), 1)
    return {
        "job_id": str(job_id), "state": job.get("State"), "exit_code": job.get("ExitCode"), "submit": job.get("Submit"), "start": job.get("Start"),
        "end": job.get("End"), "elapsed_s": int(job["ElapsedRaw"]) if job.get("ElapsedRaw") else None,
        "nodes": int(job["NNodes"]) if job.get("NNodes") else None, "timelimit_min": int(job["TimelimitRaw"]) if job.get("TimelimitRaw") else None,
        "nodelist": job.get("NodeList"),
        "max_rss_gib": {"batch": parse_rss_gib((steps_sacct.get("batch") or {}).get("MaxRSS")),
                        "worker_step": parse_rss_gib((steps_sacct.get("0") or {}).get("MaxRSS"))},
        "steps": steps, "phases": phases, "counts": {s: verified_counts(s, v["verified"]) for s, v in steps.items() if v.get("verified")},
        "sacct": {"job": job, "steps": steps_sacct},
    }


def final_attempt(attempts):
    """
    The attempt the figures use: the last one that completed, else the last one.

    Args:
        attempts: list of dictionaries
            attempt_of results in submission order.

    Returns:
        index: int
    """
    done = [i for i, a in enumerate(attempts) if a["state"] == "COMPLETED"]
    return done[-1] if done else len(attempts) - 1


def merged_steps(attempts, final):
    """
    The steps of the final attempt, with the steps that it skipped (skip_completed_steps) taken from the earlier attempts that ran them.

    Args:
        attempts: list of dictionaries
            attempt_of results.
        final: int
            Index of the final attempt.

    Returns:
        steps: dictionary
            {step: step_record} with the key 'attempt' (the job id that ran it).
    """
    steps = {}
    for a in attempts[:final + 1]:
        for step, rec in a["steps"].items():
            if rec.get("status") == "ok":
                steps[step] = dict(rec, attempt=a["job_id"])
    return steps


########################################################################################################################
# The campaign
########################################################################################################################
def hours_since(t0, text):
    """
    Hours from t0 to a time of sacct.

    Args:
        t0: datetime.datetime
            The first start of the campaign.
        text: string
            Time (ISO).

    Returns:
        hours: float or None
    """
    t = parse_time(text)
    return None if t is None else round((t - t0).total_seconds() / 3600.0, 4)


def derive(jobs):
    """
    The numbers the three figures draw (campaign wall-clock, node-hours, the minutes of every step of every job).

    Args:
        jobs: list of dictionaries
            The 'jobs' of the collector output (each with attempts and final).

    Returns:
        derived: dictionary
            t0 (ISO), track_jobs [{name, month, start_h, end_h, elapsed_min, nodes, steps_min, setup_min, other_min}], stats_job {name, start_h,
            end_h, elapsed_min, nodes, steps_min (setup + stitch, the steps), other_min}, node_hours {track, stats, failed, total}, wall_h,
            track_start_spread_min, stats_wait_min (from the last track job's end to the stats job's start), check_s_total.
    """
    finals = []
    for job in jobs:
        attempt = job["attempts"][job["final"]]
        finals.append((job, attempt))
    starts = [parse_time(a["start"]) for _, a in finals if a["start"]]
    t0 = min(starts)
    track_jobs, stats_job = [], None
    for job, a in finals:
        steps = merged_steps(job["attempts"], job["final"])
        minutes = {s: round(r["wall_s"] / 60.0, 3) for s, r in steps.items() if r.get("wall_s") is not None}
        elapsed = a["elapsed_s"] / 60.0
        if job["kind"] == "stats":
            setup = (a["phases"]["setup_s"] or 0.0) / 60.0
            stats_job = {"name": job["name"], "start_h": hours_since(t0, a["start"]), "end_h": hours_since(t0, a["end"]), "elapsed_min": round(elapsed, 2),
                         "nodes": a["nodes"], "setup_min": round(setup, 3), "steps_min": minutes, "other_min": round(elapsed - setup - sum(minutes.values()), 3)}
        else:
            track_jobs.append({"name": job["name"], "month": job.get("month"), "start_h": hours_since(t0, a["start"]), "end_h": hours_since(t0, a["end"]),
                               "elapsed_min": round(elapsed, 2), "nodes": a["nodes"], "steps_min": minutes,
                               "setup_min": round((a["phases"]["setup_s"] or 0.0) / 60.0, 3), "other_min": round(elapsed - sum(minutes.values()), 3)})
    # Node-hours: every attempt of every job counts (a failed attempt is paid for too), the failed ones are listed apart
    track_nh = stats_nh = failed_nh = 0.0
    for job in jobs:
        for i, a in enumerate(job["attempts"]):
            nh = (a["elapsed_s"] or 0) * (a["nodes"] or 0) / 3600.0
            if i != job["final"]:
                failed_nh += nh
            elif job["kind"] == "stats":
                stats_nh += nh
            else:
                track_nh += nh
    last_track_end = max((j["end_h"] for j in track_jobs), default=0.0)
    return {
        "t0": t0.isoformat(), "track_jobs": track_jobs, "stats_job": stats_job,
        "node_hours": {"track": round(track_nh, 3), "stats": round(stats_nh, 3), "failed": round(failed_nh, 3),
                       "total": round(track_nh + stats_nh + failed_nh, 3)},
        "wall_h": round(max((j["end_h"] for j in track_jobs + ([stats_job] if stats_job else [])), default=0.0), 4),
        "track_start_spread_min": round(max((j["start_h"] for j in track_jobs), default=0.0) * 60.0, 1),
        "stats_wait_min": round(((stats_job["start_h"] if stats_job else 0.0) - last_track_end) * 60.0, 1),
        "check_s_total": round(sum(a["phases"]["check_s_total"] for j in jobs for a in [j["attempts"][j["final"]]]), 1),
    }


def merged_counts(attempts, final):
    """
    The counts of the output checks of the steps of a job, with the steps that the final attempt skipped taken from the attempts that ran them.

    Args:
        attempts: list of dictionaries
            attempt_of results.
        final: int
            Index of the final attempt.

    Returns:
        counts: dictionary
            {step: {count name: number}}.
    """
    counts = {}
    for a in attempts[:final + 1]:
        for step, values in a["counts"].items():
            if a["steps"].get(step, {}).get("status") == "ok":
                counts[step] = values
    return counts


def campaign_counts(jobs):
    """
    The counts of the output checks of the campaign: frames, tracks and MCSs.

    Args:
        jobs: list of dictionaries
            The 'jobs' of the collector output.

    Returns:
        counts: dictionary
            input_files and cloudid_files (summed over the track jobs, minus the files that neighbouring months share), frames_skipped, and the
            counts of the stats job's steps.
    """
    track = [j for j in jobs if j["kind"] != "stats"]
    per_month = [merged_counts(j["attempts"], j["final"]).get("idfeature", {}) for j in track]
    n_seam = max(len(track) - 1, 0)
    out = {"input_files": sum(c.get("input_files", 0) for c in per_month) - n_seam, "cloudid_files": sum(c.get("cloudid_files", 0) for c in per_month) - n_seam,
           "frames_skipped": sum(c.get("frames_skipped", 0) for c in per_month)}
    for job in jobs:
        if job["kind"] == "stats":
            for step, counts in merged_counts(job["attempts"], job["final"]).items():
                out.update({f"{step}_{k}": v for k, v in counts.items()})
    return out


def find_job_ids(campaign_dir, names, sacct_text=None):
    """
    The job ids of the campaign by job name, in submission order, from submissions.jsonl.

    Args:
        campaign_dir: string
            Folder of the generated scripts.
        names: list of strings
            Job names of jobs.json.
        sacct_text: string (default: None)
            Not used by this function: the ids must come from submissions.jsonl (the generator writes it with --submit).

    Returns:
        ids: dictionary
            {job name: [job ids in submission order]}.

    Raises:
        SystemExit: no submissions.jsonl, or a job of jobs.json was never submitted.
    """
    path = os.path.join(campaign_dir, "submissions.jsonl")
    if not os.path.isfile(path):
        raise SystemExit(f"ERROR: {path} not found: the job ids come from it (generate_tbpf_mcstracking_scripts.py --submit writes it)")
    ids = {name: [] for name in names}
    with open(path) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if rec["name"] in ids:
                    ids[rec["name"]].append(str(rec["job_id"]))
    missing = [n for n, v in ids.items() if not v]
    if missing:
        raise SystemExit(f"ERROR: no submission of {', '.join(missing)} in {path}")
    return ids


def run_sacct(job_ids):
    """
    sacct for the jobs and their steps.

    Args:
        job_ids: list of strings
            Slurm job ids.

    Returns:
        text: string
            The parsable output with SACCT_FIELDS (memory in MB), times as ISO with the offset.
    """
    env = dict(os.environ, SLURM_TIME_FORMAT="%Y-%m-%dT%H:%M:%S%z")
    cmd = ["sacct", "-j", ",".join(job_ids), "-D", "--parsable2", "--noheader", "--units=M", "--format=" + ",".join(SACCT_FIELDS)]
    return subprocess.run(cmd, capture_output=True, text=True, check=True, env=env).stdout


def collect(campaign_dir, root, tag, sacct_text=None, profile_dir=None, tz="America/Los_Angeles", with_sizes=True):
    """
    The performance of a campaign.

    Args:
        campaign_dir: string
            Folder of the generated scripts (jobs.json, submissions.jsonl, logs/).
        root: string
            root_path of the outputs (tracking/, stats/, mcstracking/, profile/).
        tag: string
            Name of the campaign for the figures.
        sacct_text: string (default: None)
            The text of sacct; None runs sacct for the job ids of submissions.jsonl.
        profile_dir: string (default: None)
            Folder with one ledger folder per job (default: <root>/profile).
        tz: string (default: 'America/Los_Angeles')
            Time zone of the log lines.
        with_sizes: bool (default: True)
            Walk the output folders for their size.

    Returns:
        result: dictionary
            schema, campaign, jobs, derived, counts, outputs, dask ({job name: Dask message counts}; see the module description).
    """
    with open(os.path.join(campaign_dir, "jobs.json")) as f:
        job_defs = json.load(f)
    ids = find_job_ids(campaign_dir, [j["name"] for j in job_defs])
    if sacct_text is None:
        sacct_text = run_sacct([i for v in ids.values() for i in v])
    sacct = parse_sacct(sacct_text)
    profile_dir = profile_dir or os.path.join(root, "profile")
    jobs = []
    for job_def in job_defs:
        name, kind = job_def["name"], job_def["kind"]
        expected = TRACK_STEPS if kind == "track" else (TRACK_STEPS + STATS_STEPS if kind == "month" else STATS_STEPS)
        ledger = read_ledger(os.path.join(profile_dir, name, "step_ledger.jsonl"))
        attempts = []
        for job_id in ids[name]:
            log_path = os.path.join(campaign_dir, "logs", f"log_{name}_{job_id}.log")
            if not os.path.isfile(log_path):
                log_path = os.path.join(campaign_dir, "logs", f"log_{name}.log")
            attempts.append(attempt_of(name, job_id, sacct.get(job_id), ledger, read_log(log_path, tz), expected))
        final = final_attempt(attempts)
        month = re.search(r"(\d{4})(\d{2})$", name)
        jobs.append({"name": name, "kind": kind, "month": f"{month.group(1)}-{month.group(2)}" if month and kind == "track" else None,
                     "attempts": attempts, "final": final})
    # The Dask messages of every job: its log, and the logs of the Dask scheduler and workers of a stats job, of every attempt
    dask = {}
    for job in jobs:
        logs = []
        for a in job["attempts"]:
            logs += glob.glob(os.path.join(campaign_dir, "logs", f"log_{job['name']}_{a['job_id']}.log"))
            logs += glob.glob(os.path.join(campaign_dir, "logs", f"dask_*_{a['job_id']}.log"))
        dask[job["name"]] = count_dask_warnings(logs)
    commits = sorted({r.get("git") for j in job_defs for r in read_ledger(os.path.join(profile_dir, j["name"], "step_ledger.jsonl")) if r.get("git")})
    return {
        "schema": SCHEMA, "campaign": {"tag": tag, "campaign_dir": os.path.abspath(campaign_dir), "root": root, "git": commits,
                                       "collected": datetime.now(timezone.utc).isoformat(timespec="seconds"), "argv": sys.argv[1:]},
        "jobs": jobs, "derived": derive(jobs), "counts": campaign_counts(jobs), "dask": dask,
        "outputs": {name: folder_size_gb(os.path.join(root, name)) for name in ("tracking", "mcstracking", "stats")} if with_sizes else {},
    }


def main(argv=None):
    """
    Command line: collect a campaign and write the JSON.

    Args:
        argv: list of strings (default: None)
            Arguments (None: sys.argv[1:]).

    Returns:
        code: int
            0.
    """
    parser = argparse.ArgumentParser(description="Collect the performance of a Tb+PF MCS tracking campaign from sacct, the step ledgers and the logs.")
    parser.add_argument("--campaign-dir", required=True, help="folder of the generated scripts (jobs.json, submissions.jsonl, logs/)")
    parser.add_argument("--root", required=True, help="root_path of the campaign outputs (tracking/, stats/, mcstracking/, profile/)")
    parser.add_argument("-o", "--output", required=True, help="the JSON file to write")
    parser.add_argument("--tag", default=None, help="name of the campaign in the figures (default: the name of the campaign folder)")
    parser.add_argument("--profile-dir", default=None, help="folder with a ledger folder per job (default: <root>/profile)")
    parser.add_argument("--sacct-file", default=None, help="read the text of sacct from this file instead of running sacct")
    parser.add_argument("--save-sacct", default=None, help="keep the text of sacct in this file (so that the collection can be repeated later)")
    parser.add_argument("--tz", default="America/Los_Angeles", help="time zone of the log lines (default: %(default)s)")
    parser.add_argument("--no-sizes", action="store_true", help="do not walk the output folders for their size")
    args = parser.parse_args(argv)
    sacct_text = open(args.sacct_file).read() if args.sacct_file else None
    if args.save_sacct and sacct_text is None:
        with open(os.path.join(args.campaign_dir, "jobs.json")) as f:
            names = [j["name"] for j in json.load(f)]
        sacct_text = run_sacct([i for v in find_job_ids(args.campaign_dir, names).values() for i in v])
        with open(args.save_sacct, "w") as f:
            f.write(sacct_text)
    result = collect(args.campaign_dir, args.root, args.tag or os.path.basename(os.path.abspath(args.campaign_dir)), sacct_text, args.profile_dir,
                     args.tz, not args.no_sizes)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=1)
    d = result["derived"]
    print(f"{args.output}: {len(result['jobs'])} jobs, {d['node_hours']['total']} node-hours, {d['wall_h']} h from the first start to the last end")
    return 0


if __name__ == "__main__":
    sys.exit(main())
