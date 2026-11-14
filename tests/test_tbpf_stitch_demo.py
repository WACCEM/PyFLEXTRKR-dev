"""
Stitching on the Tb+PF demos: a period tracked as 2 or 3 chunks, each with its own idfeature, tracksingle and gettracks and with the seam
file shared by neighbouring chunks, whose track numbers are stitched and followed by trackstats, identifymcs, matchpf, robustmcs, mapfeature
and speed over the whole period, gives the results of one continuous run: every cloudid file, pairwise link, statistics file and
pixel-level file is the same (tests/tbpf_compare.py), for dense statistics and for row tables, with and without periodic boundaries.

The campaigns are made and run the way a production campaign is: config/generate_tbpf_mcstracking_scripts.py writes the configs and the
slurm scripts for the chunks (periods of any length), and the scripts are run with bash on the node in the order of their dependencies (the
stats job on one node). The continuous run is the run of the unified script of tests/test_tbpf_unified_demo.py of the same case and format,
made here if it is not there.

Needs the demo input below $PYFLEXTRKR_TEST_DATA, folders below $PYFLEXTRKR_TEST_WORKDIR (the cases of both test files share the continuous runs),
workers $PYFLEXTRKR_TEST_NPROCS (16). A compute node.
"""
import glob
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import yaml

from tbpf_compare import compare_runs
from test_tbpf_unified_demo import CASES, DemoRunner, _data_input

pytestmark = [pytest.mark.local, pytest.mark.demo]

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "config"))
import generate_tbpf_mcstracking_scripts as G  # noqa: E402

# case, number of chunks, format of the statistics (the dense ones are compared with the dense continuous run)
STITCH_CASES = [
    ("sahel_feng21", 2, "rows"), ("sahel_feng21", 2, "dense"), ("sahel_feng21", 3, "rows"),
    ("imerg_feng21", 3, "rows"), ("sahel_mcsmip", 3, "rows"), ("global_mcsmip", 2, "rows"),
]


def _seam_offset_minutes(root, startdate):
    """
    Minutes by which the time of the cloudid files is later than the time in the names of the input files: the first cloudid file of a run
    against the start date (the IMERG combined files are named 00:00 and their data say 00:30; the global demo and E3SM have no offset).

    Args:
        root: string
            Root of a finished run of the case.
        startdate: string
            YYYYMMDD.HHMM, the start of the run (the time in the name of the first input file).

    Returns:
        minutes: int
            0 to 59.
    """
    first = sorted(glob.glob(f"{root}/tracking/cloudid_*.nc"))[0]
    time_of_data = datetime.strptime(os.path.basename(first)[len("cloudid_"):-3], "%Y%m%d_%H%M%S")
    return int(((time_of_data - datetime.strptime(startdate, "%Y%m%d.%H%M")).total_seconds() // 60) % 60)


def _periods(start, end, n, offset_minutes=0):
    """
    n periods that cover [start, end] with seams at whole hours of the start; a period that ends at a seam ends offset_minutes later, so that
    the cloudid file of the seam instant is in the window of both periods (see generate_tbpf_mcstracking_scripts.month_range).

    Args:
        start, end: string
            YYYYMMDD.HHMM.
        n: int
            Number of periods.
        offset_minutes: int (default: 0)
            Minutes by which the time of the cloudid files is later than the time in the names of the input files.

    Returns:
        periods: list of dictionaries
            tag, name, start, end; the start of a period is the seam instant, the end of the previous one is the seam instant plus the offset;
            the first start and the last end are those of the run.
    """
    fmt = "%Y%m%d.%H%M"
    t0, t1 = datetime.strptime(start, fmt), datetime.strptime(end, fmt)
    hours = int((t1 - t0).total_seconds() // 3600)
    edges = [t0 + timedelta(hours=round(hours * i / n)) for i in range(n)] + [t1]
    ends = [edges[i + 1] + timedelta(minutes=offset_minutes) for i in range(n - 1)] + [t1]
    return [{"tag": f"p{i}", "name": f"p{i}", "start": edges[i].strftime(fmt), "end": ends[i].strftime(fmt)} for i in range(n)]


def _template_of(case, tmp_path, runner):
    """
    The config of a demo as a template of the generator: placeholders for the period and the folder of the month, the robust MCS definition,
    the format of the statistics, the workers.

    Returns:
        path, startdate, enddate: string, string, string
    """
    subdir, template, script, definition, landmask = CASES[case]
    text = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", template)).read()
    config = yaml.safe_load(text.replace("INPUT_DIR", "x").replace("TRACK_DIR", "x"))
    startdate, enddate = str(config["startdate"]), str(config["enddate"])
    text = re.sub(r"^(startdate\s*:\s*).*$", r"\g<1>'STARTDATE'", text, flags=re.M)
    text = re.sub(r"^(enddate\s*:\s*).*$", r"\g<1>'ENDDATE'", text, flags=re.M)
    text = re.sub(r"^(tracking_path_name\s*:\s*).*$", r"\g<1>tracking/YEAR-MONTH", text, flags=re.M)
    text = re.sub(r"^(nprocesses\s*:\s*).*$", rf"\g<1>{os.environ.get('PYFLEXTRKR_TEST_NPROCS', '16')}", text, flags=re.M)
    text = re.sub(r"^(run_remap_healpix\s*:\s*).*$", r"\g<1>False", text, flags=re.M)
    if definition == "mcsmip" and "mcs_min_rainvol_thresh" not in text:
        text += "\nmcs_min_rainvol_thresh: 20000\nmcs_volrain_duration_thresh: 1.0\n"
    text += f"\nrobustmcs_definition: {definition}\n"
    path = tmp_path / f"template_{case}.yml"
    path.write_text(text)
    return str(path), startdate, enddate


@pytest.fixture(scope="module")
def runner(tmp_path_factory):
    """The runner of the continuous runs (shared in $PYFLEXTRKR_TEST_WORKDIR with test_tbpf_unified_demo.py)."""
    workdir = os.environ.get("PYFLEXTRKR_TEST_WORKDIR") or str(tmp_path_factory.mktemp("tbpf_stitch"))
    os.makedirs(workdir, exist_ok=True)
    return DemoRunner(workdir)


@pytest.mark.parametrize("case,nchunks,fmt", STITCH_CASES, ids=[f"{c}-{n}chunks-{f}" for c, n, f in STITCH_CASES])
def test_stitched_chunks_equal_the_continuous_run(runner, case, nchunks, fmt):
    subdir = CASES[case][0]
    input_dir = _data_input(subdir)
    continuous = runner.run(case, "rows" if fmt == "rows" else "dense")
    campaign = os.path.join(runner.workdir, f"{case}_stitch_{nchunks}_{fmt}")
    os.makedirs(campaign, exist_ok=True)
    template, startdate, enddate = _template_of(case, Path(campaign), runner)
    # The landmask that the global demo config looks for in the root of the run
    root = os.path.join(campaign, "root")
    os.makedirs(root, exist_ok=True)
    if CASES[case][4] is not None:
        os.system(f"cp {os.path.join(input_dir, CASES[case][4])} {root}/")
    overrides = {"rainmap": False}
    offset = _seam_offset_minutes(continuous, startdate)
    periods = _periods(startdate, enddate, nchunks, offset)
    jobs = G.build_campaign(periods, template, os.path.join(campaign, "jobs"), input_dir, root, account="none", stats_nodes=1, overrides=overrides)
    # The format of the statistics of the stats job: rows (Parquet folders) or dense (the netCDF files of the old steps)
    stats_job = jobs[-1]
    text = open(stats_job["config"]).read()
    open(stats_job["config"], "w").write(G.set_key(text, "stats_format", fmt))
    print(f"{case}: seam offset {offset} min, periods {[(p['start'], p['end']) for p in periods]}")
    assert [j["kind"] for j in jobs] == ["track"] * nchunks + ["stats"]
    # Run the scripts in the order of the jobs: the track jobs, then the stats job (the dependencies of sbatch are the order here)
    for job in jobs:
        result = subprocess.run(["bash", job["script"]], cwd=os.path.join(campaign, "jobs"), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        open(os.path.join(campaign, f"log_{job['name']}.txt"), "w").write(result.stdout)
        assert result.returncode == 0, f"{job['name']}: exit code {result.returncode}, see {campaign}/log_{job['name']}.txt"
    link_folder = f"tracking/campaign_{periods[0]['tag']}_{periods[-1]['tag']}"
    report = compare_runs(continuous, root, rows_b=(fmt == "rows"), rows_a=(fmt == "rows"), tracking_b=link_folder)
    print(f"{case} {nchunks} chunks {fmt}: compared {report['compared']}")
    assert report["compared"]["stats"] >= 7 and report["compared"]["pixel"] > 0 and report["compared"]["tracking"] > 0
    assert report["differences"] == []
