"""
Tests of Analysis/collect_campaign_performance.py on a synthetic campaign: a folder of generated scripts (jobs.json, submissions.jsonl, logs/),
the step ledgers, the text of sacct, with a job that was submitted again and a stats job that skipped the steps it had finished. The numbers are
chosen so that every result can be checked by hand. Nothing runs sacct or Slurm.
"""
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "Analysis"))
import collect_campaign_performance as C  # noqa: E402

# The jobs: (name, kind, [(job id, state, start, elapsed s, nodes)]) with the time of the node (PST, -0800) of the first start 2026-11-14 10:00
JOBS = [("track202001", "track", [("101", "COMPLETED", "10:00:00", 1200, 1)]),
        ("track202002", "track", [("102", "FAILED", "10:00:10", 300, 1), ("104", "COMPLETED", "10:25:00", 1500, 1)]),
        ("stats202001_202002", "stats", [("103", "FAILED", "10:30:00", 1800, 4), ("105", "COMPLETED", "11:00:00", 2400, 4)])]


def _sacct_text():
    """The text of sacct for the jobs above: the allocation rows, the batch steps (memory) and the worker step of the 4-node jobs."""
    rows = []
    for name, kind, attempts in JOBS:
        for job_id, state, start, elapsed, nodes in attempts:
            hh, mm, ss = (int(x) for x in start.split(":"))
            end_s = hh * 3600 + mm * 60 + ss + elapsed
            end = f"{end_s // 3600:02d}:{end_s % 3600 // 60:02d}:{end_s % 60:02d}"
            t = lambda clock: f"2026-11-14T{clock}-0800"
            rows.append(f"{job_id}|{name}|{state}|{'0:0' if state == 'COMPLETED' else '1:0'}|{t('09:50:00')}|{t('09:50:00')}|{t(start)}|{t(end)}|{elapsed}|60|{nodes}|nid[1-{nodes}]|/wd||||||1-00:00:00")
            rows.append(f"{job_id}.batch|batch|{state}|0:0|{t(start)}|{t(start)}|{t(start)}|{t(end)}|{elapsed}||1|nid1||2048M|nid1|0|1024M|00:10:00")
            rows.append(f"{job_id}.extern|extern|COMPLETED|0:0|{t(start)}|{t(start)}|{t(start)}|{t(end)}|{elapsed}||{nodes}|nid[1-{nodes}]||0|nid1|0|0|00:00:00")
            if nodes > 1:
                rows.append(f"{job_id}.0|dask-worker|{state}|0:0|{t(start)}|{t(start)}|{t(start)}|{t(end)}|{elapsed}||{nodes}|nid[1-{nodes}]||4096M|nid2|5|2048M|10:00:00")
    return "\n".join(rows) + "\n"


def _ledger_line(step, job_id, start_utc, wall_s, nodes=1):
    """A line of a step ledger (the fields the collector reads)."""
    hh, mm, ss = (int(x) for x in start_utc.split(":"))
    end_s = hh * 3600 + mm * 60 + ss + int(wall_s) + 2
    end = f"{end_s // 3600:02d}:{end_s % 3600 // 60:02d}:{end_s % 60:02d}"
    return json.dumps({"step": step, "status": "ok", "start": f"2026-11-14T{start_utc}+00:00", "end": f"2026-11-14T{end}+00:00", "wall_s": wall_s,
                       "slurm_job_id": job_id, "slurm_nnodes": nodes, "effective_cores": 40.0, "tree_cpu_s": 40.0 * wall_s, "read_gb": 1.5, "write_gb": 0.5,
                       "tree_rss_peak_gb": 90.0, "main_rss_peak_gb": 2.0, "node_used_peak_gb": 120.0, "git": "abc1234", "run_parallel": 2 if nodes > 1 else 1,
                       "worker_rss_peak_gb": {"n": 128, "median": 1.0, "p95": 1.2, "max": 1.5}, "n_workers": 128})


def _log(path, lines):
    """A log: the lines (local time, message) in the format of the python logging of the run script."""
    with open(path, "w") as f:
        for clock, message in lines:
            f.write(f"2026-11-14 {clock} - pyflextrkr.step_timing - INFO - {message}\n")
        f.write("Sat Nov 14 12:00:00 PM PST 2026\n")


@pytest.fixture
def campaign(tmp_path):
    """The folder of a synthetic campaign: scripts folder, root with the ledgers, the logs and the text of sacct."""
    folder, root = tmp_path / "campaign", tmp_path / "root"
    (folder / "logs").mkdir(parents=True)
    (root / "profile").mkdir(parents=True)
    (folder / "jobs.json").write_text(json.dumps([{"name": n, "kind": k, "script": f"slurm_{n}.sh", "needs": [], "config": "c.yml"} for n, k, _ in JOBS]))
    submissions = [("track202001", "101"), ("track202002", "102"), ("stats202001_202002", "103"), ("track202002", "104"), ("stats202001_202002", "105")]
    (folder / "submissions.jsonl").write_text("".join(json.dumps({"name": n, "job_id": i}) + "\n" for n, i in submissions))
    (tmp_path / "sacct.txt").write_text(_sacct_text())
    # track202001 (job 101, 10:00 PST = 18:00 UTC): 30 s of setup, then the three steps
    steps = [("idfeature", "18:00:30", 600.0), ("tracksingle", "18:10:32", 120.0), ("gettracks", "18:12:34", 60.0)]
    (root / "profile" / "track202001").mkdir()
    (root / "profile" / "track202001" / "step_ledger.jsonl").write_text("\n".join(_ledger_line(s, "101", t, w) for s, t, w in steps) + "\n")
    _log(folder / "logs" / "log_track202001_101.log", [
        ("10:05:00,000", "distributed.semaphore - CRITICAL - Refreshing an unknown lease ID abc for /x/cloudid_1.nc. This might be due to leases timing out"),
        ("10:10:30,500", "Step idfeature finished in 10.00 min"), ("10:10:30,700", "Step idfeature outputs verified: 744 cloudid files for 745 input files (1 frames skipped by miss_thresh)"),
        ("10:12:32,000", "Step tracksingle finished in 2.00 min"), ("10:12:32,100", "Step tracksingle outputs verified: track files for all 743 pairs of the 744 cloudid files"),
        ("10:13:34,000", "Step gettracks finished in 1.00 min"), ("10:13:34,050", "Step gettracks outputs verified: 5000 track numbers, last file cloudid_20200201_000000.nc")])
    # track202002: the first attempt (102) failed in idfeature, the second (104, 10:25 PST = 18:25 UTC) completed
    (root / "profile" / "track202002").mkdir()
    attempt2 = [("idfeature", "18:25:20", 660.0), ("tracksingle", "18:36:22", 100.0), ("gettracks", "18:38:04", 50.0)]
    (root / "profile" / "track202002" / "step_ledger.jsonl").write_text(
        json.dumps({"step": "idfeature", "status": "failed: MemoryError", "start": "2026-11-14T18:00:20+00:00", "end": "2026-11-14T18:05:00+00:00", "wall_s": 280.0,
                    "slurm_job_id": "102", "git": "abc1234"}) + "\n" + "\n".join(_ledger_line(s, "104", t, w) for s, t, w in attempt2) + "\n")
    _log(folder / "logs" / "log_track202002_104.log", [
        ("10:36:20,000", "Step idfeature finished in 11.00 min"), ("10:36:20,300", "Step idfeature outputs verified: 672 cloudid files for 673 input files"),
        ("10:38:02,000", "Step tracksingle finished in 1.67 min"), ("10:38:02,100", "Step tracksingle outputs verified: track files for all 671 pairs of the 672 cloudid files"),
        ("10:38:54,000", "Step gettracks finished in 0.83 min"), ("10:38:54,100", "Step gettracks outputs verified: 4000 track numbers, last file x")])
    # the stats job: the first attempt (103, 10:30 PST) ran trackstats and identifymcs and failed in matchpf; the second (105, 11:00 PST) skipped
    # the finished steps (skip_completed_steps) and ran from matchpf on
    (root / "profile" / "stats202001_202002").mkdir()
    lines = [_ledger_line("trackstats", "103", "18:35:00", 300.0, 4), _ledger_line("identifymcs", "103", "18:40:02", 40.0, 4),
             _ledger_line("matchpf", "105", "19:03:00", 200.0, 4), _ledger_line("robustmcs", "105", "19:06:22", 10.0, 4), _ledger_line("mapfeature", "105", "19:06:34", 500.0, 4),
             _ledger_line("speed", "105", "19:14:56", 100.0, 4), _ledger_line("rainmap", "105", "19:16:38", 90.0, 4), _ledger_line("trackpcp", "105", "19:18:10", 400.0, 4)]
    (root / "profile" / "stats202001_202002" / "step_ledger.jsonl").write_text("\n".join(lines) + "\n")
    _log(folder / "logs" / "log_stats202001_202002_103.log", [
        ("10:33:00,000", "Stitched 2 periods into /x/tracknumbers.nc"),
        ("10:40:00,000", "Step trackstats finished in 5.00 min"), ("10:40:00,400", "Step trackstats outputs verified: 9000 tracks"),
        ("10:40:42,000", "Step identifymcs finished in 0.67 min"), ("10:40:42,100", "Step identifymcs outputs verified: 300 MCS tracks")])
    _log(folder / "logs" / "log_stats202001_202002_105.log", [
        ("11:02:00,000", "Stitched 2 periods into /x/tracknumbers.nc"),
        ("11:06:20,000", "Step matchpf finished in 3.33 min"), ("11:06:20,100", "Step matchpf outputs verified: 600 of 600 entries (100.0 %) have PF statistics"),
        ("11:06:32,000", "Step robustmcs finished in 0.17 min"), ("11:06:32,100", "Step robustmcs outputs verified: 40 robust MCS tracks"),
        ("11:14:54,000", "Step mapfeature finished in 8.33 min"), ("11:14:54,500", "Step mapfeature outputs verified: 1417 pixel files for 1417 cloudid files"),
        ("11:16:36,000", "Step speed finished in 1.67 min"), ("11:16:36,100", "Step speed outputs verified: 700 finite speeds"),
        ("11:18:08,000", "Step rainmap finished in 1.50 min"), ("11:18:08,100", "Step rainmap outputs verified: 2 monthly files"),
        ("11:24:50,000", "Step trackpcp finished in 6.67 min"), ("11:24:50,100", "Step trackpcp outputs verified: 3 files")])
    (folder / "logs" / "dask_scheduler_105.log").write_text(
        "distributed.scheduler - INFO - ok\ndistributed.core - WARNING - Removing worker tcp://x\nWorker unresponsive\n"
        "distributed.core - INFO - Event loop was unresponsive in Worker for 3.20s.  This is often caused by GIL-holding functions\n")
    (folder / "logs" / "dask_workers_105.log").write_text("Couldn't gather 3 keys\nCommClosedError: x\n")
    return folder, root, tmp_path / "sacct.txt"


def test_parsers_of_times_memory_and_sacct_rows():
    assert C.parse_time("2026-11-14T10:00:00-0800").isoformat() == "2026-11-14T10:00:00-08:00"
    assert C.parse_time("2026-11-14T18:00:30+00:00").isoformat() == "2026-11-14T18:00:30+00:00" and C.parse_time("Unknown") is None
    assert C.parse_rss_gib("2048M") == 2.0 and C.parse_rss_gib("") is None and C.parse_rss_gib("1.5G") == 1.5
    jobs = C.parse_sacct(_sacct_text())
    assert set(jobs) == {"101", "102", "103", "104", "105"} and set(jobs["105"]["steps"]) == {"batch", "extern", "0"}
    assert jobs["105"]["job"]["State"] == "COMPLETED" and jobs["101"]["steps"]["batch"]["MaxRSS"] == "2048M"


def test_log_lines_give_the_time_of_the_check_and_of_the_stitching(campaign):
    folder, _, _ = campaign
    log = C.read_log(str(folder / "logs" / "log_stats202001_202002_103.log"))
    # the log is in PST: 10:33:00 is 18:33:00 UTC; the time of the output check is the difference of two times of the same log
    assert log["stitched"].isoformat() == "2026-11-14T18:33:00+00:00"
    assert log["finished"]["trackstats"][1] == 5.0 and (log["verified"]["trackstats"][0] - log["finished"]["trackstats"][0]).total_seconds() == pytest.approx(0.4)
    assert C.verified_counts("matchpf", "600 of 600 entries (100.0 %) have PF statistics") == {"matched_entries": 600, "entries": 600}
    assert C.verified_counts("idfeature", "744 cloudid files for 745 input files (1 frames skipped by miss_thresh)") == {
        "cloudid_files": 744, "input_files": 745, "frames_skipped": 1}
    assert C.verified_counts("nostep", "x") == {} and C.read_log(str(folder / "no_such_log"))["finished"] == {}


def test_a_track_job_has_its_steps_phases_checks_and_memory(campaign):
    folder, root, sacct = campaign
    result = C.collect(str(folder), str(root), "synthetic", sacct.read_text(), with_sizes=False)
    job = result["jobs"][0]
    assert job["name"] == "track202001" and job["month"] == "2020-01" and job["final"] == 0
    a = job["attempts"][0]
    assert (a["state"], a["nodes"], a["elapsed_s"], a["timelimit_min"]) == ("COMPLETED", 1, 1200, 60)
    assert a["max_rss_gib"] == {"batch": 2.0, "worker_step": None}
    # the steps: the wall time of the ledger, the check time of the log, the cores and memory of the ledger
    idf = a["steps"]["idfeature"]
    assert idf["wall_s"] == 600.0 and idf["check_s"] == pytest.approx(0.2) and idf["cores"] == 40.0 and idf["worker_max_gb"] == 1.5
    assert idf["report_s"] == pytest.approx(2.0)          # the ledger line ends 2 s after the step plus its wall time
    assert a["counts"]["idfeature"] == {"cloudid_files": 744, "input_files": 745, "frames_skipped": 1}
    # the setup before the first step: 30 s; the checks of the job: 0.2 + 0.1 + 0.05 s
    assert a["phases"]["setup_s"] == 30.0 and a["phases"]["check_s_total"] == pytest.approx(0.35) and a["phases"]["stitch_s"] is None


def test_a_job_that_was_submitted_again_keeps_its_attempts_and_the_figures_use_the_last_completed_one(campaign):
    folder, root, sacct = campaign
    result = C.collect(str(folder), str(root), "synthetic", sacct.read_text(), with_sizes=False)
    job = result["jobs"][1]
    assert [a["job_id"] for a in job["attempts"]] == ["102", "104"] and [a["state"] for a in job["attempts"]] == ["FAILED", "COMPLETED"] and job["final"] == 1
    # the failed attempt's ledger line is its own (the job id), the second attempt has the three steps
    assert job["attempts"][0]["steps"] == {} or job["attempts"][0]["steps"]["idfeature"]["status"].startswith("failed")
    assert {s: r["wall_s"] for s, r in job["attempts"][1]["steps"].items()} == {"idfeature": 660.0, "tracksingle": 100.0, "gettracks": 50.0}


def test_a_stats_job_that_skipped_its_finished_steps_is_merged_from_its_attempts(campaign):
    folder, root, sacct = campaign
    result = C.collect(str(folder), str(root), "synthetic", sacct.read_text(), with_sizes=False)
    job = result["jobs"][2]
    first, second = job["attempts"]
    assert job["final"] == 1 and first["state"] == "FAILED" and second["nodes"] == 4
    assert list(first["steps"]) == ["trackstats", "identifymcs"] and list(second["steps"]) == ["matchpf", "robustmcs", "mapfeature", "speed", "rainmap", "trackpcp"]
    merged = C.merged_steps(job["attempts"], job["final"])
    assert list(merged) == ["trackstats", "identifymcs", "matchpf", "robustmcs", "mapfeature", "speed", "rainmap", "trackpcp"]
    assert merged["trackstats"]["attempt"] == "103" and merged["matchpf"]["attempt"] == "105"
    # the memory of the worker step (a multi-node job) and of the batch step; the ledger of a job with a remote scheduler sees the main process only
    assert second["max_rss_gib"] == {"batch": 2.0, "worker_step": 4.0}
    assert second["steps"]["matchpf"]["main_process_only"] is True and job["attempts"][0]["steps"]["trackstats"]["main_process_only"] is True
    assert result["jobs"][0]["attempts"][0]["steps"]["idfeature"]["main_process_only"] is False
    # the stitching ended 2 min after the start of the second attempt (11:00 -> 11:02 PST), the first step started 1 min later
    assert second["phases"]["stitch_s"] == 120.0 and second["phases"]["setup_s"] == 180.0 and second["phases"]["dask_start_s"] == 60.0


def test_the_derived_numbers_of_the_figures(campaign):
    folder, root, sacct = campaign
    d = C.collect(str(folder), str(root), "synthetic", sacct.read_text(), with_sizes=False)["derived"]
    # the first start is 10:00:00 PST (job 101): the second track job's final attempt starts at 10:25:00, the stats job's at 11:00:00
    assert [(j["name"], j["start_h"], j["end_h"], j["elapsed_min"]) for j in d["track_jobs"]] == [
        ("track202001", 0.0, 0.3333, 20.0), ("track202002", 0.4167, 0.8333, 25.0)]
    assert d["track_jobs"][1]["steps_min"] == {"idfeature": 11.0, "tracksingle": 1.667, "gettracks": 0.833}
    # the stats job: the steps of both attempts (the minutes are those of the attempt that ran them), setup of the final attempt 3 min
    s = d["stats_job"]
    assert (s["start_h"], s["end_h"], s["elapsed_min"], s["nodes"]) == (1.0, 1.6667, 40.0, 4)
    assert s["steps_min"]["trackstats"] == 5.0 and s["steps_min"]["trackpcp"] == pytest.approx(6.667, abs=1e-3) and s["setup_min"] == 3.0
    # node-hours: 1 x 20 min + 1 x 25 min (track), 4 x 40 min (stats), the failed attempts 1 x 5 min and 4 x 30 min are listed apart
    nh = d["node_hours"]
    assert nh["track"] == pytest.approx(45 / 60, abs=1e-3) and nh["stats"] == pytest.approx(160 / 60, abs=1e-3) and nh["failed"] == pytest.approx((5 + 120) / 60, abs=1e-3)
    assert nh["total"] == pytest.approx(nh["track"] + nh["stats"] + nh["failed"], abs=1e-3)
    assert d["wall_h"] == pytest.approx(1.6667, abs=1e-3) and d["track_start_spread_min"] == 25.0
    assert d["stats_wait_min"] == pytest.approx(10.0, abs=0.1)          # the last track job ends 10:50, the stats job starts 11:00


def test_counts_of_the_campaign_and_the_dask_warnings(campaign):
    folder, root, sacct = campaign
    (root / "tracking").mkdir()
    (root / "tracking" / "a.nc").write_bytes(b"x" * 1000)
    result = C.collect(str(folder), str(root), "synthetic", sacct.read_text())
    c = result["counts"]
    # two months share one input file and one cloudid file: 745 + 673 - 1 and 744 + 672 - 1
    assert (c["input_files"], c["cloudid_files"], c["frames_skipped"]) == (1417, 1415, 1)
    # the steps that the second attempt of the stats job skipped come from the first attempt
    assert c["trackstats_tracks"] == 9000 and c["identifymcs_mcs_tb"] == 300
    assert c["robustmcs_mcs_robust"] == 40 and c["matchpf_matched_entries"] == 600 and c["mapfeature_pixel_files"] == 1417
    assert result["dask"]["stats202001_202002"] == {"event_loop_unresponsive": 1, "unresponsive": 1, "gather_failed": 1, "worker_removed": 1,
                                                    "killed_worker": 0, "memory_95": 0, "comm_closed": 1, "timeout": 0, "lease_timeout": 0}
    # a track job's own log counts too: the LocalCluster logs there
    assert result["dask"]["track202001"]["lease_timeout"] == 1 and result["dask"]["track202002"]["lease_timeout"] == 0
    assert result["outputs"]["tracking"] == {"gb": 0.0, "files": 1} and result["outputs"]["stats"] == {"gb": 0.0, "files": 0}


def test_the_job_ids_come_from_submissions_jsonl(campaign, tmp_path):
    folder, _, _ = campaign
    assert C.find_job_ids(str(folder), ["track202002"]) == {"track202002": ["102", "104"]}
    with pytest.raises(SystemExit, match="no submission of nothing"):
        C.find_job_ids(str(folder), ["nothing"])
    with pytest.raises(SystemExit, match="submissions.jsonl"):
        C.find_job_ids(str(tmp_path), ["track202002"])


def test_the_command_line_writes_the_json_from_a_saved_sacct_text(campaign, tmp_path, capsys):
    folder, root, sacct = campaign
    out = tmp_path / "campaign.json"
    assert C.main(["--campaign-dir", str(folder), "--root", str(root), "-o", str(out), "--sacct-file", str(sacct), "--tag", "synthetic", "--no-sizes"]) == 0
    result = json.loads(out.read_text())
    assert result["schema"] == C.SCHEMA and result["campaign"]["tag"] == "synthetic" and result["campaign"]["git"] == ["abc1234"]
    assert "3 jobs" in capsys.readouterr().out and len(result["jobs"]) == 3
