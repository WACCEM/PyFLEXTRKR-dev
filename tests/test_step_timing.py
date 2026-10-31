"""
Tests of pyflextrkr.step_timing (per-step wall time, memory and profiling of the pipeline steps).

No tracking data is needed. The process-tree test starts a child Python process that holds ~200 MB and keeps a core
busy, which stands in for a Dask worker.
"""
import logging
import subprocess
import sys
import time

import pytest

from pyflextrkr.step_timing import read_ledger, timed_step

psutil = pytest.importorskip("psutil")


def _config(tmp_path, **kwargs):
    """
    Config dictionary with profiling on, sampling every 0.05 s, writing to <tmp_path>/prof.

    Args:
        tmp_path: pathlib.Path
            Test directory.
        **kwargs:
            Further config keys.

    Returns:
        config: dictionary
    """
    config = {"step_profile_dir": str(tmp_path / "prof"), "step_sample_seconds": 0.05,
              "root_path": "root/", "startdate": "20200101.0000", "enddate": "20200102.0000",
              "run_parallel": 1, "nprocesses": 2}
    config.update(kwargs)
    return config


def _busy(n=1_500_000):
    """
    A bit of CPU work.
    """
    return sum(i * i for i in range(n))


def test_without_profile_dir_only_logs_and_writes_nothing(tmp_path, caplog):
    config = {"root_path": str(tmp_path)}
    with caplog.at_level(logging.INFO, logger="pyflextrkr.step_timing"):
        with timed_step("demo", config):
            time.sleep(0.05)
    assert "Step demo finished in" in caplog.text
    assert list(tmp_path.iterdir()) == []


def test_ledger_and_timeline_fields(tmp_path):
    with timed_step("demo", _config(tmp_path)) as t:
        t.add(n_files=3)
        _busy()
    records = read_ledger(tmp_path / "prof" / "step_ledger.jsonl")
    assert len(records) == 1
    r = records[0]
    assert r["step"] == "demo" and r["status"] == "ok" and r["n_files"] == 3
    assert r["startdate"] == "20200101.0000" and r["nprocesses"] == 2
    assert r["wall_s"] > 0 and r["tree_cpu_s"] > 0 and r["effective_cores"] > 0
    assert r["tree_rss_peak_gb"] > 0 and r["node_total_gb"] > 0 and r["n_samples"] >= 2
    lines = (tmp_path / "prof" / "memory_timeline.csv").read_text().splitlines()
    assert lines[0].startswith("time_utc,step,node_used_gb") and len(lines) >= 3
    assert all(",demo," in line for line in lines[1:])


def test_steps_append_to_one_ledger_and_one_timeline(tmp_path):
    config = _config(tmp_path)
    with timed_step("first", config):
        _busy(200_000)
    with timed_step("second", config):
        _busy(200_000)
    records = read_ledger(tmp_path / "prof" / "step_ledger.jsonl")
    assert [r["step"] for r in records] == ["first", "second"]
    lines = (tmp_path / "prof" / "memory_timeline.csv").read_text().splitlines()
    assert sum(1 for line in lines if line.startswith("time_utc")) == 1  # header written once


def test_child_process_memory_and_cpu_are_counted(tmp_path):
    # A living child process: its peak RSS counts as a worker and its CPU time is added to the tree
    code = "a = b'x' * 200_000_000\nwhile True:\n    pass"
    with timed_step("child", _config(tmp_path, step_sample_seconds=0.1)):
        child = subprocess.Popen([sys.executable, "-c", code])
        time.sleep(2.0)
    child.kill()
    child.wait()
    r = read_ledger(tmp_path / "prof" / "step_ledger.jsonl")[0]
    assert r["worker_rss_peak_gb"] is not None and r["worker_rss_peak_gb"]["n"] >= 1
    assert r["worker_rss_peak_gb"]["max"] > 0.15
    assert r["tree_cpu_s"] > 1.0
    assert r["tree_rss_peak_gb"] >= r["worker_rss_peak_gb"]["max"]


def test_exception_is_recorded_and_reraised(tmp_path):
    with pytest.raises(ValueError):
        with timed_step("boom", _config(tmp_path)):
            raise ValueError("expected")
    assert read_ledger(tmp_path / "prof" / "step_ledger.jsonl")[0]["status"] == "failed: ValueError"


def test_system_exit_is_recorded_and_reraised(tmp_path):
    with pytest.raises(SystemExit):
        with timed_step("quit", _config(tmp_path)):
            raise SystemExit(3)
    assert read_ledger(tmp_path / "prof" / "step_ledger.jsonl")[0]["status"] == "exit 3"


def test_cprofile_only_for_the_listed_steps(tmp_path):
    config = _config(tmp_path, step_cprofile_steps=["profiled"])
    with timed_step("profiled", config):
        _busy()
    with timed_step("plain", config):
        _busy(10_000)
    prof = tmp_path / "prof"
    assert (prof / "cprofile_profiled.prof").is_file()
    text = (prof / "cprofile_profiled.txt").read_text()
    assert "top 40 by cumulative time" in text and "_busy" in text
    assert not (prof / "cprofile_plain.prof").exists()


def test_dask_client_gives_report_and_worker_count(tmp_path):
    distributed = pytest.importorskip("distributed")
    with distributed.Client(processes=False, n_workers=1, threads_per_worker=1, dashboard_address=None) as client:
        with timed_step("dask", _config(tmp_path), client=client):
            assert client.submit(sum, [1, 2, 3]).result() == 6
    assert (tmp_path / "prof" / "dask_report_dask.html").is_file()
    assert read_ledger(tmp_path / "prof" / "step_ledger.jsonl")[0]["n_workers"] == 1
