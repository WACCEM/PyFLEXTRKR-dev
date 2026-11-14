"""
Per-step timing, memory and (optional) deep profiling of the PyFLEXTRKR pipeline steps.

A run script wraps each pipeline step in `timed_step`:

    from pyflextrkr.step_timing import timed_step
    with timed_step("gettracks", config, client=client):
        gettracknumbers(config)

Always (cheap): one INFO line per step with its wall time.

When the config key `step_profile_dir` is a path (null = off, the default) the step also writes, into that directory:
  - step_ledger.jsonl      one JSON line per step: wall time, CPU seconds of the process tree (effective cores),
                           bytes read / written, peak memory of the tree, of the node and of the worker processes
                           and the Slurm job id and number of nodes of the job (null outside Slurm)
  - memory_timeline.csv    node and process-tree memory every `step_sample_seconds` (default 5) seconds
  - dask_report_<step>.html  Dask performance report (task stream, worker profile), if a Dask client is given
  - cprofile_<step>.prof/.txt  cProfile of the main process, for the steps listed in `step_cprofile_steps`
                           (null = none; cProfile slows Python loops, so use it in a separate run from the timing run)

The process tree is this process plus all its children, i.e. the workers of a Dask LocalCluster. Workers of an
external Dask scheduler (run_parallel: 2) run on other nodes and are not seen by the sampler.

Author: Zhe Feng | zhe.feng@pnnl.gov
"""
import contextlib
import cProfile
import datetime
import functools
import io
import json
import logging
import os
import pstats
import socket
import subprocess
import threading
import time

import numpy as np

try:
    import psutil
except ImportError:  # psutil is a dependency of dask.distributed; keep the module importable without it
    psutil = None

logger = logging.getLogger(__name__)

GB = 1.0e9
# Processes whose peak RSS is below this (Dask nannies, multiprocessing helpers) are not counted as workers
MIN_WORKER_RSS_BYTES = 100.0e6


def _env_int(name):
    """
    An integer environment variable.

    Args:
        name: string
            Variable name (e.g. 'SLURM_NNODES').

    Returns:
        value: int or None
            None if the variable is not set or is not an integer.
    """
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return None


def _utc_now():
    """
    Current UTC time as ISO text.

    Returns:
        text: string
            e.g. '2026-10-30T18:03:12+00:00' (seconds resolution).
    """
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


@functools.lru_cache(maxsize=1)
def _git_head():
    """
    Short git commit of the PyFLEXTRKR checkout that is running, to tie a ledger line to the code version.

    Returns:
        commit: string or None
            Output of `git rev-parse --short HEAD` in the package directory; None if it cannot be determined.
    """
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        out = subprocess.run(["git", "-C", here, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def _process_tree():
    """
    This process and all its children (recursive).

    Returns:
        procs: list of psutil.Process
            Main process first. Empty if psutil is not installed.
    """
    if psutil is None:
        return []
    main = psutil.Process()
    try:
        return [main] + main.children(recursive=True)
    except psutil.Error:
        return [main]


def _tree_counters():
    """
    Cumulative CPU time and I/O characters of every process in the tree.

    Returns:
        counters: dictionary
            {pid: (cpu_seconds, read_chars, write_chars)}; cpu_seconds = user + system. Characters are 0 where the
            platform does not provide them.
    """
    out = {}
    for p in _process_tree():
        try:
            ct = p.cpu_times()
            try:
                io_c = p.io_counters()
                rd, wr = io_c.read_chars, io_c.write_chars
            except (AttributeError, psutil.Error):
                rd = wr = 0
            out[p.pid] = (ct.user + ct.system, rd, wr)
        except psutil.Error:
            continue
    return out


class _TreeSampler(threading.Thread):
    """
    Background thread that samples the memory of the node and of the process tree at a fixed interval.

    Attributes after stop():
        pid_peak_rss: dictionary {pid: peak RSS in bytes seen in this step}
        tree_rss_peak: float, peak of the summed RSS of the tree [bytes]
        node_used_start, node_used_peak: float, node memory in use (total - available) at the start / peak [bytes]
        node_total: float, node memory [bytes]
        n_samples: int, number of samples taken
    """

    def __init__(self, step, interval=5.0, timeline_file=None):
        """
        Args:
            step: string
                Step name, written to the timeline file.
            interval: float (default: 5.0)
                Seconds between samples.
            timeline_file: string (default: None)
                CSV file that gets one line per sample (appended; header written if the file is new). None: no file.
        """
        super().__init__(name=f"step-sampler-{step}", daemon=True)
        self.step = step
        self.interval = float(interval)
        self.timeline_file = timeline_file
        self._stop_event = threading.Event()
        self.pid_peak_rss = {}
        self.tree_rss_peak = 0.0
        self.node_used_start = None
        self.node_used_peak = 0.0
        self.node_total = 0.0
        self.n_samples = 0
        self._fh = None
        if timeline_file is not None:
            is_new = not os.path.isfile(timeline_file) or os.path.getsize(timeline_file) == 0
            self._fh = open(timeline_file, "a", buffering=1)
            if is_new:
                self._fh.write("time_utc,step,node_used_gb,tree_rss_gb,n_procs,system_cpu_pct\n")
        psutil.cpu_percent(interval=None)  # first call only primes the counter

    def sample(self):
        """
        Take one sample: node memory, RSS of every process of the tree, system CPU percent since the last sample.
        """
        vm = psutil.virtual_memory()
        node_used = float(vm.total - vm.available)
        self.node_total = float(vm.total)
        tree = 0.0
        procs = _process_tree()
        for p in procs:
            try:
                rss = float(p.memory_info().rss)
            except psutil.Error:
                continue
            tree += rss
            if rss > self.pid_peak_rss.get(p.pid, 0.0):
                self.pid_peak_rss[p.pid] = rss
        self.tree_rss_peak = max(self.tree_rss_peak, tree)
        self.node_used_peak = max(self.node_used_peak, node_used)
        if self.node_used_start is None:
            self.node_used_start = node_used
        self.n_samples += 1
        if self._fh is not None:
            cpu_pct = psutil.cpu_percent(interval=None)
            self._fh.write(f"{_utc_now()},{self.step},{node_used / GB:.3f},{tree / GB:.3f},{len(procs)},{cpu_pct:.1f}\n")

    def run(self):
        """
        Thread body: sample, then wait for `interval` seconds or until stop() is called.
        """
        while not self._stop_event.is_set():
            self.sample()
            self._stop_event.wait(self.interval)

    def stop(self):
        """
        Stop the thread, take a last sample, close the timeline file.
        """
        self._stop_event.set()
        self.join()
        self.sample()
        if self._fh is not None:
            self._fh.close()
            self._fh = None


class StepTimer:
    """
    Context manager that times one pipeline step and, if `step_profile_dir` is set, records its resource use.

    See the module docstring for the config keys and the files. Exceptions (including SystemExit) are recorded in the
    ledger and re-raised; profiling problems never stop the pipeline.
    """

    def __init__(self, step, config, client=None):
        """
        Args:
            step: string
                Step name, e.g. 'gettracks'.
            config: dictionary
                Config dictionary; uses step_profile_dir (default None), step_cprofile_steps (default None),
                step_sample_seconds (default 5) and, for the ledger line, root_path, startdate, enddate, nprocesses,
                run_parallel.
            client: dask.distributed.Client (default: None)
                Client of the run, for the number of workers and the performance report.
        """
        self.step = step
        self.config = config
        self.client = client
        self.profile_dir = config.get("step_profile_dir", None)
        self.cprofile_steps = config.get("step_cprofile_steps", None) or []
        self.interval = float(config.get("step_sample_seconds", 5.0))
        self.extra = {}
        self._sampler = None
        self._cprofile = None
        self._stack = None

    def add(self, **kwargs):
        """
        Attach extra fields (e.g. n_files=1189) to the ledger line of this step.

        Args:
            **kwargs: JSON-serializable values.
        """
        self.extra.update(kwargs)

    def __enter__(self):
        self._t0 = time.perf_counter()
        self._iso0 = _utc_now()
        if psutil is not None:
            self._counters0 = _tree_counters()
        if self.profile_dir is not None and psutil is not None:
            try:
                os.makedirs(self.profile_dir, exist_ok=True)
                self._sampler = _TreeSampler(self.step, self.interval,
                                             os.path.join(self.profile_dir, "memory_timeline.csv"))
                self._sampler.start()
                if self.step in self.cprofile_steps:
                    self._cprofile = cProfile.Profile()
                    self._cprofile.enable()
                if self.client is not None:
                    from distributed import performance_report
                    self._stack = contextlib.ExitStack()
                    self._stack.enter_context(performance_report(
                        filename=os.path.join(self.profile_dir, f"dask_report_{self.step}.html")))
            except Exception as err:  # profiling must never stop the pipeline
                logger.warning(f"Step {self.step}: profiling could not be started ({err!r})")
        return self

    def __exit__(self, exc_type, exc, tb):
        wall = time.perf_counter() - self._t0
        # Status of the step; exceptions are re-raised (return False below)
        if exc_type is None:
            status = "ok"
        elif issubclass(exc_type, SystemExit):
            status = f"exit {getattr(exc, 'code', None)}"
        else:
            status = f"failed: {exc_type.__name__}"
        if self._cprofile is not None:
            self._cprofile.disable()
        counters1 = _tree_counters() if psutil is not None else {}
        if self._sampler is not None:
            self._sampler.stop()
        try:
            if self._stack is not None:
                self._stack.close()  # writes the Dask report
            if self._cprofile is not None:
                self._write_cprofile()
            if self.profile_dir is not None:
                self._write_ledger(wall, status, counters1)
        except Exception as err:
            logger.warning(f"Step {self.step}: profile output could not be written ({err!r})")
        if status == "ok":
            logger.info(f"Step {self.step} finished in {wall / 60.0:.2f} min")
        else:
            logger.warning(f"Step {self.step} ended after {wall / 60.0:.2f} min with status '{status}'")
        return False

    def _write_cprofile(self):
        """
        Write the cProfile of the step: <dir>/cprofile_<step>.prof and a text with the top 40 functions by
        cumulative and by own time.
        """
        base = os.path.join(self.profile_dir, f"cprofile_{self.step}")
        self._cprofile.dump_stats(base + ".prof")
        buf = io.StringIO()
        stats = pstats.Stats(self._cprofile, stream=buf)
        buf.write(f"== {self.step}: top 40 by cumulative time\n")
        stats.sort_stats("cumulative").print_stats(40)
        buf.write(f"\n== {self.step}: top 40 by own time\n")
        stats.sort_stats("tottime").print_stats(40)
        with open(base + ".txt", "w") as f:
            f.write(buf.getvalue())

    def _n_workers(self):
        """
        Number of workers of the Dask client.

        Returns:
            n: int or None
                None if there is no client or the scheduler cannot be asked.
        """
        if self.client is None:
            return None
        try:
            return len(self.client.scheduler_info(n_workers=-1)["workers"])      # the default reports at most 5 workers
        except Exception:
            return None

    def _write_ledger(self, wall, status, counters1):
        """
        Append the ledger line of this step to <step_profile_dir>/step_ledger.jsonl.

        Args:
            wall: float
                Wall time of the step [s].
            status: string
                'ok', 'exit <code>' or 'failed: <exception name>'.
            counters1: dictionary
                _tree_counters() at the end of the step.
        """
        rec = {
            "step": self.step, "status": status, "start": self._iso0, "end": _utc_now(), "wall_s": round(wall, 2),
            "host": socket.gethostname(), "pid": os.getpid(), "git": _git_head(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_nnodes": _env_int("SLURM_NNODES"),
            "root_path": self.config.get("root_path"),
            "startdate": self.config.get("startdate"), "enddate": self.config.get("enddate"),
            "run_parallel": self.config.get("run_parallel"), "nprocesses": self.config.get("nprocesses"),
            "n_workers": self._n_workers(),
        }
        if psutil is not None and self._sampler is not None:
            # CPU seconds and I/O of the tree during the step: processes seen at the start count from their start
            # value, processes that appeared later from zero (processes that ended during the step are lost)
            c0 = self._counters0
            cpu = sum(v[0] - c0.get(pid, (0.0, 0, 0))[0] for pid, v in counters1.items())
            rd = sum(v[1] - c0.get(pid, (0.0, 0, 0))[1] for pid, v in counters1.items())
            wr = sum(v[2] - c0.get(pid, (0.0, 0, 0))[2] for pid, v in counters1.items())
            main_pid = os.getpid()
            main_cpu = counters1[main_pid][0] - c0.get(main_pid, (0.0, 0, 0))[0] if main_pid in counters1 else None
            s = self._sampler
            workers = [v for pid, v in s.pid_peak_rss.items() if pid != main_pid and v >= MIN_WORKER_RSS_BYTES]
            rec.update({
                "main_cpu_s": None if main_cpu is None else round(main_cpu, 2),
                "tree_cpu_s": round(cpu, 2),
                "effective_cores": round(cpu / wall, 2) if wall > 0 else None,
                "read_gb": round(rd / GB, 3), "write_gb": round(wr / GB, 3),
                "tree_rss_peak_gb": round(s.tree_rss_peak / GB, 3),
                "main_rss_peak_gb": round(s.pid_peak_rss.get(main_pid, 0.0) / GB, 3),
                "node_used_start_gb": round((s.node_used_start or 0.0) / GB, 2),
                "node_used_peak_gb": round(s.node_used_peak / GB, 2),
                "node_total_gb": round(s.node_total / GB, 1),
                "worker_rss_peak_gb": ({
                    "n": len(workers),
                    "median": round(float(np.median(workers)) / GB, 3),
                    "p95": round(float(np.percentile(workers, 95)) / GB, 3),
                    "max": round(float(np.max(workers)) / GB, 3),
                } if workers else None),
                "n_samples": s.n_samples,
            })
        rec.update(self.extra)
        with open(os.path.join(self.profile_dir, "step_ledger.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")


def timed_step(step, config, client=None):
    """
    Time (and, with step_profile_dir, profile) one pipeline step: `with timed_step('gettracks', config, client): ...`

    Args:
        step: string
            Step name.
        config: dictionary
            Config dictionary (see StepTimer).
        client: dask.distributed.Client (default: None)
            Client of the run.

    Returns:
        timer: StepTimer
            Context manager; `with ... as t: t.add(n_files=10)` adds fields to the ledger line.
    """
    return StepTimer(step, config, client)


def read_ledger(path):
    """
    Read a step ledger.

    Args:
        path: string
            step_ledger.jsonl.

    Returns:
        records: list of dictionaries
            One per ledger line, in the order written.
    """
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
