"""
The E3SM 0.1 degree global workflow on real data (a compute node): the first days of 2009 of the E3SM ne120 simulation, tracked as one-day
periods whose track numbers are stitched (the generator's own scripts, run with bash in the order of their dependencies), give the results of
one continuous run of the same days: every cloudid file, pairwise link, statistics file and pixel-level file (tests/tbpf_compare.py), the
monthly rain map and the precipitation accumulated along the tracks. The file of 01 UTC on 1 January 2009 has no OLR (FLUT is missing everywhere;
the file of 00 UTC is complete): that frame is skipped, recorded in the manifest of idfeature, and the output checks pass.

Needs $PYFLEXTRKR_E3SM_INPUT (the folder with the year folders, .../olr_pcp) and, because of the `local` marker, $PYFLEXTRKR_TEST_DATA set to any
folder; run folders below $PYFLEXTRKR_TEST_WORKDIR (default: a pytest temporary folder; about 4 GB for 3 days), the days $PYFLEXTRKR_E3SM_DAYS (3),
the workers $PYFLEXTRKR_TEST_NPROCS (the 128 of the template); $PYFLEXTRKR_TEST_REUSE=1 compares the runs that finished before in that folder
without making them again. The step ledgers
(wall time, CPU, memory of every step) are in <workdir>/<run>/ledger/step_ledger.jsonl. Nothing is submitted: the scripts run on this node.
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta

import pandas as pd
import pytest

from tbpf_compare import compare_runs, netcdf_differences

pytestmark = [pytest.mark.local, pytest.mark.demo]

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "config"))
import generate_tbpf_mcstracking_scripts as G  # noqa: E402

TEMPLATE = os.path.join(REPO, "config", "config_mcs_tbpf_e3sm_0p1deg_template.yml")
START = datetime(2009, 1, 1)
SECONDS = "%Y%m%d.%H%M"


def _periods(ndays):
    """
    One period per day from 2009-01-01 00:00; the end of a day is the start of the next.

    Args:
        ndays: int
            Number of days.

    Returns:
        periods: list of dictionaries
            tag, name, start, end.
    """
    days = [START + timedelta(days=d) for d in range(ndays + 1)]
    return [{"tag": f"d{d}", "name": f"d{d}", "start": days[d].strftime(SECONDS), "end": days[d + 1].strftime(SECONDS)} for d in range(ndays)]


def _profile(jobs, ledger):
    """
    Write the step ledger folder into the config of every job (the wall time, CPU and memory of every step).

    Args:
        jobs: list of dictionaries
            Jobs of the generator.
        ledger: string
            Folder of the ledger.
    """
    for job in jobs:
        with open(job["config"]) as f:
            text = f.read()
        with open(job["config"], "w") as f:
            f.write(G.set_key(text, "step_profile_dir", ledger))


def _run(jobs, folder):
    """
    Run the scripts of the jobs with bash in the order of the list (the order of their dependencies), on this node.

    Args:
        jobs: list of dictionaries
            Jobs of the generator.
        folder: string
            Folder of the jobs (the scripts write logs/ there; the console output goes to log_<job>.txt next to it).

    Returns:
        seconds: dictionary
            {job name: wall time [s]}.
    """
    seconds = {}
    for job in jobs:
        t0 = time.time()
        result = subprocess.run(["bash", job["script"]], cwd=folder, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        seconds[job["name"]] = time.time() - t0
        with open(os.path.join(folder, f"log_{job['name']}.txt"), "w") as f:
            f.write(result.stdout)
        assert result.returncode == 0, f"{job['name']}: exit code {result.returncode}, see {folder}/log_{job['name']}.txt"
    return seconds


def _frames_without_olr(input_dir, nhours):
    """
    The hourly files of the first nhours + 1 hours of 2009 whose OLR is missing everywhere: idfeature skips them (miss_thresh), so they have no cloudid file.

    Args:
        input_dir: string
            Folder with the year folders of the E3SM files (<input_dir>/2009/ne120_olr_pcp_YYYYMMDDHH.nc).
        nhours: int
            Number of hours after the first file (the files of nhours + 1 instants are looked at).

    Returns:
        empty: list of strings
            Names of the files whose FLUT is all NaN.
    """
    import numpy as np
    import xarray as xr
    empty = []
    # One file per hour from the start of the period to its end (the end instant is included, as in the tracking)
    for hour in range(nhours + 1):
        name = f"ne120_olr_pcp_{(START + timedelta(hours=hour)).strftime('%Y%m%d%H')}.nc"
        with xr.open_dataset(os.path.join(input_dir, "2009", name)) as ds:
            # A frame with no valid OLR at all is the case this test needs to cover
            if bool(np.isnan(ds["FLUT"].values).all()):
                empty.append(name)
    return empty


def _ledger_table(path):
    """
    The step ledger as a table: one line per step with the wall time, the effective cores and the memory.

    Args:
        path: string
            step_ledger.jsonl.

    Returns:
        lines: list of strings
    """
    rows = [json.loads(line) for line in open(path)]
    out = [f"{'step':12s} {'period':26s} {'wall min':>9s} {'cores':>6s} {'main GB':>8s} {'node GB':>8s} {'worker max GB':>14s}"]
    for r in rows:
        worker = r.get("worker_rss_peak_gb") or {}
        out.append(f"{r['step']:12s} {r['startdate']}-{r['enddate']:>13s} {r['wall_s'] / 60:9.2f} {r.get('effective_cores', 0) or 0:6.1f} "
                   f"{r.get('main_rss_peak_gb', 0):8.1f} {r.get('node_used_peak_gb', 0):8.1f} {worker.get('max', 0):14.2f}")
    return out


def test_stitched_days_equal_the_continuous_run_on_the_e3sm_data(tmp_path_factory):
    input_dir = os.environ.get("PYFLEXTRKR_E3SM_INPUT")
    if not input_dir or not os.path.isdir(os.path.join(input_dir, "2009")):
        pytest.skip("$PYFLEXTRKR_E3SM_INPUT (the folder with the year folders of the E3SM files) is not set")
    ndays = int(os.environ.get("PYFLEXTRKR_E3SM_DAYS", "3"))
    workdir = os.environ.get("PYFLEXTRKR_TEST_WORKDIR") or str(tmp_path_factory.mktemp("e3sm"))
    nprocs = os.environ.get("PYFLEXTRKR_TEST_NPROCS")
    stitched, continuous = os.path.join(workdir, "stitched"), os.path.join(workdir, "continuous")
    for folder in (stitched, continuous):
        os.makedirs(folder, exist_ok=True)
    periods = _periods(ndays)
    whole = {"tag": "all", "name": "all", "start": periods[0]["start"], "end": periods[-1]["end"]}
    # The template is used as it is (the workers of the node, the optional steps: rain maps on and, for this test, the accumulation along the tracks)
    template = TEMPLATE
    if nprocs:
        text = G.set_key(open(TEMPLATE).read(), "nprocesses", int(nprocs))
        template = os.path.join(workdir, "template.yml")
        open(template, "w").write(text)
    options = dict(input_dir=input_dir, account="none", stats_nodes=1, overrides={"rainmap": True, "trackpcp": True})
    jobs_s = G.build_campaign(periods, template, os.path.join(stitched, "jobs"), root_path=os.path.join(stitched, "root"), **options)
    jobs_c = G.build_campaign([whole], template, os.path.join(continuous, "jobs"), root_path=os.path.join(continuous, "root"), chunk_size=1, **options)
    assert [j["kind"] for j in jobs_s] == ["track"] * ndays + ["stats"] and [j["kind"] for j in jobs_c] == ["month"]
    _profile(jobs_s, os.path.join(stitched, "ledger"))
    _profile(jobs_c, os.path.join(continuous, "ledger"))

    # With $PYFLEXTRKR_TEST_REUSE the runs that finished before (they left .run_ok) are not made again, only compared
    markers = [os.path.join(folder, ".run_ok") for folder in (continuous, stitched)]
    if os.environ.get("PYFLEXTRKR_TEST_REUSE") and all(os.path.isfile(m) for m in markers):
        print("the finished runs of this folder are used (PYFLEXTRKR_TEST_REUSE)")
    else:
        for marker in markers:
            if os.path.isfile(marker):
                os.remove(marker)
        seconds_c = _run(jobs_c, os.path.join(continuous, "jobs"))
        seconds_s = _run(jobs_s, os.path.join(stitched, "jobs"))
        print("continuous job:", {k: round(v / 60, 1) for k, v in seconds_c.items()}, "min")
        print("stitched jobs:", {k: round(v / 60, 1) for k, v in seconds_s.items()}, "min")
        for marker in markers:
            open(marker, "w").close()
    for name, folder in (("continuous", continuous), ("stitched", stitched)):
        print(f"--- steps of the {name} run")
        print("\n".join(_ledger_table(os.path.join(folder, "ledger", "step_ledger.jsonl"))))

    # The frames without OLR are skipped and counted (the number is taken from the input files), the other frames have a cloudid file
    root_c, root_s = os.path.join(continuous, "root"), os.path.join(stitched, "root")
    manifest = json.load(open(os.path.join(root_c, "tracking", "all", "idfeature_manifest.json")))
    empty = _frames_without_olr(input_dir, 24 * ndays)
    print(f"frames without OLR in the input: {empty}; manifest: {manifest}")
    assert len(empty) >= 1, "the first days of 2009 are expected to hold a frame without OLR (the skip path is not covered otherwise)"
    assert manifest["n_inputs"] == 24 * ndays + 1 and manifest["n_calls_skipped"] == len(empty), manifest
    # Every file equal: the continuous run against the stitched one (its cloudid and track files are those of the folder of links)
    link_folder = f"tracking/campaign_{periods[0]['tag']}_{periods[-1]['tag']}"
    report = compare_runs(root_c, root_s, rows_a=True, rows_b=True, tracking_a="tracking/all", tracking_b=link_folder)
    print(f"compared {report['compared']}")
    # The statistics files: tracknumbers, the sparse trackstats and the four MCS files (no dense trackstats file: trackstats_dense_netcdf is 0)
    assert report["compared"]["stats"] == 6 and report["compared"]["pixel"] > 0 and report["compared"]["tracking"] > 0
    assert report["differences"] == []
    # The monthly rain map and the accumulation along the tracks
    maps = [os.path.join(r, "stats", "monthly", "mcs_rainmap_200901.nc") for r in (root_c, root_s)]
    assert netcdf_differences(maps[0], maps[1], ignore_global_attrs=("created_on",)) == []
    tag = f"{periods[0]['start']}_{periods[-1]['end']}"
    for name in ("pixels", "metrics"):
        tables = [pd.read_parquet(os.path.join(r, "stats", f"trackpcp_{name}_{tag}.parquet")) for r in (root_c, root_s)]
        pd.testing.assert_frame_equal(tables[0], tables[1])
        assert len(tables[0]) > 0
