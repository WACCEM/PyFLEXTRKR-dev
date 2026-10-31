"""
Tests of the rows mode of regionprops (pyflextrkr/regionprops_rows.py): on the same cloudid files and the same first-pass
statistics, the row tables of compute_regionprops_rows are identical (every column, dtype and value, and the meta data) to what
dense_to_rows makes of the dense file of compute_regionprops. Run on the synthetic case of the dense tests (merge fragment, two
tracks, lifecycle staging, dilation radii, subset of track classes, unreadable files). Synthetic data only.
"""
import os

import numpy as np
import pytest

from pyflextrkr.regionprops_driver import compute_regionprops
from pyflextrkr.regionprops_rows import compute_regionprops_rows, fragment_slots
from pyflextrkr.rowstats import dense_to_rows, read_tables, table_differences, write_rows
from test_regionprops_radar_feature import NSTEPS, TestTrackClassesOption, build_run

DATES = "20250502.1900_20250502.2000"


def convert_first_pass(config):
    """
    Write the row tables of the dense first-pass file next to it (the input of the rows-mode step).

    Args:
        config: dictionary
            Config of build_run.
    """
    stats = config["stats_outpath"]
    write_rows(dense_to_rows(f"{stats}mcs_tracks_{DATES}.nc"), f"{stats}mcs_tracks_{DATES}.rows")


def run_both(config):
    """
    Run the dense and the rows step on the same inputs.

    Args:
        config: dictionary
            Config of build_run.

    Returns:
        dense_tables, rows_tables: RowTables
            dense_to_rows of the dense robust file, and the tables the rows step wrote.
    """
    convert_first_pass(config)
    dense_file = compute_regionprops(config)
    rows_folder = compute_regionprops_rows(config)
    assert os.path.basename(rows_folder) == f"mcs_tracks_robust_{DATES}.rows"
    return dense_to_rows(dense_file), read_tables(rows_folder)


@pytest.mark.parametrize("radius_km", [0.0, 10.0, 20.0])
def test_rows_step_equals_dense_step(tmp_path, radius_km):
    config, _ = build_run(tmp_path, radius_km=radius_km)
    dense, rows = run_both(config)
    assert table_differences(dense, rows) == []
    assert rows.tracks["lifecycle_evaluated"].tolist() == [1, 1] and rows.meta["global_attrs"]["n_pairs_not_computed"] == 0


def test_rows_step_with_the_dilation_radius_recomputed_in_the_worker(tmp_path):
    config, _ = build_run(tmp_path, radius_km=10.0)
    config["conv_dilate_overwrite_radius_km"] = 20.0
    config["x_dimname"] = "x"
    config["dx"] = 3000
    config["dy"] = 3000
    dense, rows = run_both(config)
    assert table_differences(dense, rows) == []
    assert rows.meta["global_attrs"]["conv_dilate_radius_km_idfeature"] == 10.0


def test_rows_step_with_missing_cloudid_files(tmp_path):
    config, _ = build_run(tmp_path, radius_km=10.0)
    for name in sorted(os.listdir(config["tracking_outpath"]))[:1]:               # one of 40 files: 2.5 % of the pairs
        os.remove(config["tracking_outpath"] + name)
    dense, rows = run_both(config)
    assert table_differences(dense, rows) == []
    assert rows.meta["global_attrs"]["n_pairs_not_computed"] == 2
    assert np.isnan(rows.entries["majoraxis"]).sum() == 2


def test_too_many_missing_cloudid_files_abort_like_the_dense_step(tmp_path):
    config, _ = build_run(tmp_path, radius_km=10.0)
    for name in sorted(os.listdir(config["tracking_outpath"]))[:10]:
        os.remove(config["tracking_outpath"] + name)
    convert_first_pass(config)
    with pytest.raises(SystemExit):
        compute_regionprops_rows(config)


@pytest.mark.parametrize("classes", [[1], [3], [1, 3]])
def test_rows_step_with_a_subset_of_track_classes(tmp_path, classes):
    config = TestTrackClassesOption._prepare(tmp_path, "subset", classes)
    dense, rows = run_both(config)
    assert table_differences(dense, rows) == []
    assert np.atleast_1d(rows.meta["global_attrs"]["regionprops_track_classes"]).tolist() == classes   # netCDF: one element = scalar


def test_the_two_file_bases_must_differ(tmp_path):
    config, _ = build_run(tmp_path, radius_km=10.0)
    convert_first_pass(config)
    config["mcsrobust_filebase"] = "mcs_tracks_"
    with pytest.raises(SystemExit):
        compute_regionprops_rows(config)


def test_fragment_slots_look_up_the_slots_of_the_entries():
    entry = np.array([2, 2, 5, 9, 9, 9])                 # fragments of one kind at the entries 2, 5 and 9
    slot = np.array([0, 1, 0, 0, 1, 3])
    cloudnumber = np.array([11, 12, 13, 14, 15, 16], dtype=np.int32)
    got = fragment_slots(entry, slot, cloudnumber, np.array([9, 0, 2, 5, 7]), nslots=4, fill=-9999)
    assert got.dtype == np.int32
    assert got.tolist() == [[14, 15, -9999, 16], [-9999] * 4, [11, 12, -9999, -9999], [13, -9999, -9999, -9999], [-9999] * 4]
    assert fragment_slots(entry[:0], slot[:0], cloudnumber[:0], np.array([1, 2]), nslots=2, fill=-1).tolist() == [[-1, -1]] * 2
