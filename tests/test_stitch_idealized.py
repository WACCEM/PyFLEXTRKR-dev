"""
Stitching of track numbers on the idealized demo data (config/demo_mcs_tbpf_idealized.sh): four small scenarios of hourly Tb +
precipitation fields (19-26 frames each) with merging, splitting and dissipating systems. idfeature and tracksingle run on each
scenario (seconds); then the tracks of the periods between the cuts are stitched and compared with a single gettracks run
over all frames, for every single cut and a sample of pairs of cuts. A variant removes the files around one frame so that two periods
are 2 h apart without a pair linking them (a restart, as gettracknumbers does for a missing track file).

Needs the demo data: export PYFLEXTRKR_TEST_DATA=<root> with mcs_tbpf/idealized/test4/input/MCS-test-N_20200101.nc below it.
"""
import itertools
import os
import random
import shutil

import numpy as np
import pytest

import synthetic_tracks as syn
from pyflextrkr import gettracks, stitch_tracks
from pyflextrkr.ft_utilities import load_config, subset_files_timerange
from pyflextrkr.idfeature_driver import idfeature_driver
from pyflextrkr.tracksingle_driver import tracksingle_driver

pytestmark = pytest.mark.local

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(REPO, "config", "config_mcs_idealized.yml")
INPUT_SUBDIR = os.path.join("mcs_tbpf", "idealized", "test4", "input")


def _input_dir():
    root = os.environ.get("PYFLEXTRKR_TEST_DATA")
    path = os.path.join(root, INPUT_SUBDIR) if root else None
    if not path or not os.path.isdir(path):
        pytest.skip(f"idealized demo input not found below $PYFLEXTRKR_TEST_DATA/{INPUT_SUBDIR}")
    return path


@pytest.fixture(scope="module")
def scenarios(tmp_path_factory):
    """
    Run idfeature and tracksingle on the four idealized scenarios.

    Returns:
        scenarios: dictionary
            {n: (config, frame times)}; config is the loaded demo config of scenario n, frame times the epoch seconds of its
            cloudid files.
    """
    input_dir = _input_dir()
    out = {}
    for n in (1, 2, 3, 4):
        root = tmp_path_factory.mktemp(f"test{n}")
        text = open(TEMPLATE).read().replace("INPUT_DIR", input_dir + "/").replace("TRACK_DIR", str(root) + "/") \
            .replace("BASENAME", f"MCS-test-{n}_")
        cfg_path = str(root / "config.yml")
        open(cfg_path, "w").write(text)
        config = load_config(cfg_path)
        config["run_parallel"] = 0
        idfeature_driver(config)
        tracksingle_driver(config)
        files, times, _, _ = subset_files_timerange(config["tracking_outpath"], config["cloudid_filebase"],
                                                    config["start_basetime"], config["end_basetime"])
        out[n] = (config, [int(t) for t in times])
    return out


def _period_config(config, times, first, last, stats_dir, tag):
    """
    Config for gettracknumbers over the pairs from the one with reference frame `first` to the one that ends in frame `last`
    (frame numbers are positions in the list of cloudid times).
    """
    cfg = dict(config)
    cfg.update(stats_outpath=str(stats_dir) + "/", startdate=f"{tag}a", enddate=f"{tag}b",
               start_basetime=times[first] + 1, end_basetime=times[last])
    os.makedirs(cfg["stats_outpath"], exist_ok=True)
    return cfg


def _stitched_and_continuous(config, times, cuts, tmp_path, tag):
    bounds = [0] + list(cuts) + [len(times) - 1]
    paths = [gettracks.gettracknumbers(_period_config(config, times, a, b, tmp_path / f"{tag}_{i}", f"{tag}{i}"))
             for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:]))]
    out = str(tmp_path / f"{tag}_stitched.nc")
    stitch_tracks.stitch_tracknumbers(paths, out, tracking_dirs=[config["tracking_outpath"]] * len(paths))
    whole = gettracks.gettracknumbers(_period_config(config, times, 0, len(times) - 1, tmp_path / f"{tag}_all", f"{tag}all"))
    return syn.read_tracknumbers(out), syn.read_tracknumbers(whole)


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_every_single_cut_and_a_sample_of_pairs(scenarios, tmp_path, n):
    config, times = scenarios[n]
    cuts = list(range(1, len(times) - 1))
    for cut in cuts:
        stitched, whole = _stitched_and_continuous(config, times, [cut], tmp_path, f"s{cut}")
        assert syn.same_tracknumbers(stitched, whole) == [], f"scenario {n}, cut at frame {cut}"
    pairs = list(itertools.combinations(cuts, 2))
    for cut_a, cut_b in random.Random(n).sample(pairs, min(25, len(pairs))):
        stitched, whole = _stitched_and_continuous(config, times, [cut_a, cut_b], tmp_path, f"p{cut_a}_{cut_b}")
        assert syn.same_tracknumbers(stitched, whole) == [], f"scenario {n}, cuts at {cut_a} and {cut_b}"
    assert int(whole["ntracks"][0]) > 5                                   # the scenario has tracks, merges and splits


def test_two_periods_with_no_pair_between_them_restart(scenarios, tmp_path):
    config, times = scenarios[4]
    # Remove the cloudid file of frame 10 and the two track files around it (copy of the folder): frames 9 and 11 are 2 h apart,
    # closer than the timegap of 3.1 h, but no track file links them
    folder = tmp_path / "gap"
    shutil.copytree(config["tracking_outpath"], folder)
    stamp = lambda t: __import__("pandas").to_datetime(t, unit="s").strftime("%Y%m%d_%H%M%S")
    for name in (f"cloudid_{stamp(times[10])}.nc", f"track_{stamp(times[10])}.nc", f"track_{stamp(times[11])}.nc"):
        os.remove(folder / name)
    cfg = dict(config, tracking_outpath=str(folder) + "/")
    kept = [t for i, t in enumerate(times) if i != 10]
    pa = gettracks.gettracknumbers(_period_config(cfg, kept, 0, 9, tmp_path / "gA", "gA"))
    pb = gettracks.gettracknumbers(_period_config(cfg, kept, 10, len(kept) - 1, tmp_path / "gB", "gB"))
    out = str(tmp_path / "gap_stitched.nc")
    stitch_tracks.stitch_tracknumbers([pa, pb], out)
    whole = gettracks.gettracknumbers(_period_config(cfg, kept, 0, len(kept) - 1, tmp_path / "gAll", "gAll"))
    assert syn.same_tracknumbers(out, whole) == []
    assert int(syn.read_tracknumbers(out)["attrs"]["n_chain_breaks"]) == 1
