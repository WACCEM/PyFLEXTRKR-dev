"""
The unified Tb+PF workflow (runscripts/run_mcs_tbpf_unified.py) gives the results of the old scripts on the Tb+PF demo data: every
cloudid file, pairwise link, statistics file and pixel-level file is the same as that of the run of the old script on the same input, for the
dense statistics (stats_format: dense) and, as row tables, for stats_format: rows (the tables equal what dense_to_rows makes of the dense file
of the old script).

Each case runs the old script and the unified script (twice: dense and rows) on one demo data set of 48 hourly files, in its own root
folder, with the Dask LocalCluster of the demo config (a compute node). The cases:

  imerg_feng21    IMERG Tb + precipitation, Jan 2019 (config_imerg_mcs_tbpf_example.yml)       run_mcs_tbpf.py          feng21
  sahel_feng21    IMERG V07B over the Sahel (config_imergv7_mcs_tbpf_sahel.yml)                  run_mcs_tbpf.py          feng21
  sahel_mcsmip    the same data and config with the MCSMIP keys added                           run_mcs_tbpf_mcsmip.py   mcsmip
  global_mcsmip   global IMERG with periodic boundaries (config_mcs_tbpf_imergv7_mcsmip_demo.yml) run_mcs_tbpf_mcsmip.py  mcsmip

Needs the demo input below $PYFLEXTRKR_TEST_DATA (mcs_tbpf/<imerg|imergv7_sahel|imerg_global>/input). Folders of the runs: below
$PYFLEXTRKR_TEST_WORKDIR (default: a pytest temporary folder; the global case writes several GB), workers: $PYFLEXTRKR_TEST_NPROCS (16);
$PYFLEXTRKR_TEST_REUSE=1 does not run a case again whose run finished before in that folder.
"""
import os
import re
import subprocess
import sys

import pytest

from tbpf_compare import compare_runs

pytestmark = [pytest.mark.local, pytest.mark.demo]

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(REPO, "config")

# case: (demo data folder, config template, old script, robust MCS definition, landmask file name that the config looks for in TRACK_DIR)
CASES = {
    "imerg_feng21": ("mcs_tbpf/imerg", "config_imerg_mcs_tbpf_example.yml", "run_mcs_tbpf.py", "feng21", None),
    "sahel_feng21": ("mcs_tbpf/imergv7_sahel", "config_imergv7_mcs_tbpf_sahel.yml", "run_mcs_tbpf.py", "feng21", None),
    "sahel_mcsmip": ("mcs_tbpf/imergv7_sahel", "config_imergv7_mcs_tbpf_sahel.yml", "run_mcs_tbpf_mcsmip.py", "mcsmip", None),
    "global_mcsmip": ("mcs_tbpf/imerg_global", "config_mcs_tbpf_imergv7_mcsmip_demo.yml", "run_mcs_tbpf_mcsmip.py", "mcsmip",
                      "IMERG_landmask_180W-180E_60S-60N.nc"),
}
# keys of the MCSMIP definition for the configs that were written for the definition of Feng et al.
MCSMIP_KEYS = "mcs_min_rainvol_thresh: 20000\nmcs_volrain_duration_thresh: 1.0\n"


def _data_input(subdir):
    """The demo input folder, or skip the test."""
    root = os.environ.get("PYFLEXTRKR_TEST_DATA")
    path = os.path.join(root, subdir, "input") if root else None
    if not path or not os.path.isdir(path) or len(os.listdir(path)) < 10:
        pytest.skip(f"demo input not found below $PYFLEXTRKR_TEST_DATA/{subdir}/input")
    return path


class DemoRunner:
    """
    Runs a case once for each variant (old script, unified dense, unified rows) and remembers the result.
    """

    def __init__(self, workdir):
        self.workdir = workdir
        self.done = {}

    def config_text(self, case, root, variant):
        """
        The config of a run: the demo template with the paths, the number of workers, no HEALPix remapping and, for the unified script,
        the robust MCS definition (and stats_format: rows for the rows variant).
        """
        subdir, template, script, definition, landmask = CASES[case]
        text = open(os.path.join(CONFIG_DIR, template)).read()
        text = text.replace("INPUT_DIR", _data_input(subdir)).replace("TRACK_DIR", root + "/")
        text = re.sub(r"^(nprocesses\s*:\s*).*$", rf"\g<1>{os.environ.get('PYFLEXTRKR_TEST_NPROCS', '16')}", text, flags=re.MULTILINE)
        text = re.sub(r"^(run_remap_healpix\s*:\s*).*$", r"\g<1>False", text, flags=re.MULTILINE)
        if definition == "mcsmip" and "mcs_min_rainvol_thresh" not in text:
            text += "\n" + MCSMIP_KEYS
        if variant != "legacy":
            text += f"\nrobustmcs_definition: {definition}\n" + ("stats_format: rows\n" if variant == "rows" else "")
        return text

    def run(self, case, variant):
        """
        Run one variant of a case (once) and return the root folder of the run.

        Args:
            case: string
                A key of CASES.
            variant: string
                'legacy' (the old script), 'dense' or 'rows' (the unified script).

        Returns:
            root: string
        """
        key = (case, variant)
        if key in self.done:
            return self.done[key]
        subdir, template, script, definition, landmask = CASES[case]
        root = os.path.join(self.workdir, case, variant)
        os.makedirs(root, exist_ok=True)
        # With $PYFLEXTRKR_TEST_REUSE a run that finished before (it left .run_ok) is not made again: the stitching tests and a second call
        # of these tests use the runs of the first one
        marker = os.path.join(root, ".run_ok")
        if os.environ.get("PYFLEXTRKR_TEST_REUSE") and os.path.isfile(marker):
            self.done[key] = root
            return root
        if os.path.isfile(marker):
            os.remove(marker)
        # The landmask that the global demo config looks for in the root of the run
        if landmask is not None:
            os.system(f"cp {os.path.join(_data_input(subdir), landmask)} {root}/")
        config_path = os.path.join(self.workdir, case, f"config_{variant}.yml")
        with open(config_path, "w") as f:
            f.write(self.config_text(case, root, variant))
        runscript = os.path.join(REPO, "runscripts", script if variant == "legacy" else "run_mcs_tbpf_unified.py")
        with open(os.path.join(self.workdir, case, f"log_{variant}.txt"), "w") as log:
            result = subprocess.run([sys.executable, runscript, config_path], cwd=CONFIG_DIR, stdout=log, stderr=subprocess.STDOUT)
        assert result.returncode == 0, f"{case} {variant}: exit code {result.returncode}, see {self.workdir}/{case}/log_{variant}.txt"
        open(marker, "w").close()
        self.done[key] = root
        return root


@pytest.fixture(scope="module")
def runner(tmp_path_factory):
    """The runner of the cases; the runs are made in $PYFLEXTRKR_TEST_WORKDIR if it is set."""
    workdir = os.environ.get("PYFLEXTRKR_TEST_WORKDIR") or str(tmp_path_factory.mktemp("tbpf_unified"))
    os.makedirs(workdir, exist_ok=True)
    return DemoRunner(workdir)


@pytest.mark.parametrize("case", list(CASES))
def test_unified_dense_equals_the_old_script(runner, case):
    legacy, unified = runner.run(case, "legacy"), runner.run(case, "dense")
    report = compare_runs(legacy, unified)
    print(f"{case}: compared {report['compared']}")
    assert report["compared"]["stats"] >= 7 and report["compared"]["pixel"] > 0 and report["compared"]["tracking"] > 0
    assert report["differences"] == []


@pytest.mark.parametrize("case", list(CASES))
def test_unified_rows_equals_the_old_script(runner, case):
    legacy, rows = runner.run(case, "legacy"), runner.run(case, "rows")
    report = compare_runs(legacy, rows, rows_b=True)
    print(f"{case}: compared {report['compared']}")
    assert report["compared"]["stats"] >= 7 and report["compared"]["pixel"] > 0 and report["compared"]["tracking"] > 0
    assert report["differences"] == []
