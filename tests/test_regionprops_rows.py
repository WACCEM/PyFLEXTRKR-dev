"""
Tests of the rows mode of regionprops (pyflextrkr/regionprops_rows.py): on the same cloudid files and the same first-pass
statistics, the row tables of compute_regionprops_rows are identical (every column, dtype and value, and the meta data) to what
dense_to_rows makes of the dense file of compute_regionprops. Run on the synthetic case of the dense tests (merge fragment, two
tracks, lifecycle staging, dilation radii, subset of track classes, unreadable files). Synthetic data only.
"""
import os

import numpy as np
import pytest

from pyflextrkr.mcs_lifecycle import upscale_growth
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


def test_rows_step_with_conv_diameter_sum(tmp_path):
    """First-pass tables with conv_diameter_sum: the merged-mask values are the same in both modes (3 and 1 cells of 16.93 km)."""
    config, _ = build_run(tmp_path, radius_km=10.0, with_conv_diameter_sum=True)
    # the first-pass tables of this fixture have area_45dbz and no track_class: the keys of the earlier definitions
    config.update(mcs_size_method="conv_diameter_sum", mcs_size_thresh=40, mcs_dbz45_area_varname="area_45dbz", mcs_lifecycle_classes=None)
    dense, rows = run_both(config)
    assert table_differences(dense, rows) == []
    one_cell = 2.0 * np.sqrt(25 * 9.0 / np.pi)
    values = rows.entries["conv_diameter_sum"].reshape(-1)
    np.testing.assert_allclose(np.unique(values[np.isfinite(values)]), [one_cell, 3 * one_cell], rtol=1e-5)


def test_rows_step_with_the_clusters_dilated_further_in_the_worker(tmp_path):
    config, _ = build_run(tmp_path, radius_km=10.0)
    config["conv_dilate_overwrite_radius_km"] = 20.0
    config["x_dimname"] = "x"
    config["dx"] = 3000
    config["dy"] = 3000
    dense, rows = run_both(config)
    assert table_differences(dense, rows) == []
    assert rows.meta["global_attrs"]["conv_dilate_overwrite_radius_km"] == 20.0
    np.testing.assert_array_equal(rows.meta["global_attrs"]["radii_expand_km"], [10.0])


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


@pytest.mark.parametrize("lifecycle_classes, expected", [([1], [1, 0]), ([3], [0, 1]), ([1, 3], [1, 1]), (None, [1, 1])])
def test_lifecycle_is_staged_for_the_asked_classes_without_a_minimum_lifetime(tmp_path, lifecycle_classes, expected):
    """mcs_lifecycle_classes picks the tracks by track_class (here 1 and 3), the same in the dense and the rows step; thresh 0 = no minimum."""
    config = TestTrackClassesOption._prepare(tmp_path, "life", None)
    config.update(mcs_lifecycle_classes=lifecycle_classes, mcs_lifecycle_thresh=0)
    dense, rows = run_both(config)
    assert table_differences(dense, rows) == []
    assert rows.tracks["lifecycle_evaluated"].tolist() == expected
    attrs = rows.meta["global_attrs"]
    assert attrs["mcs_lifecycle_thresh_hour"] == 0 and attrs["mcs_lifecycle_smooth_samples"] >= 1
    assert ("mcs_lifecycle_classes" in attrs) == (lifecycle_classes is not None)


def test_a_minimum_lifetime_still_applies_when_it_is_set(tmp_path):
    config = TestTrackClassesOption._prepare(tmp_path, "life", None)
    config.update(mcs_lifecycle_classes=None, mcs_lifecycle_thresh=1000)
    dense, rows = run_both(config)
    assert rows.tracks["lifecycle_evaluated"].tolist() == [0, 0]


def test_lifecycle_classes_need_track_class_in_the_first_pass_file(tmp_path):
    config = TestTrackClassesOption._prepare(tmp_path, "life", None, track_class=None)
    config.update(mcs_lifecycle_classes=[1])
    convert_first_pass(config)
    with pytest.raises(SystemExit):
        compute_regionprops(config)
    with pytest.raises(SystemExit):
        compute_regionprops_rows(config)


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


def test_single_pass_lifecycle_with_the_diameter_sum_is_the_same_in_both_modes(tmp_path):
    """Track A has 3 cells of 16.93 km (line 50.8 km >= 40), track B one cell (no line): the new per-track variables and six stages."""
    config, _ = build_run(tmp_path, radius_km=10.0, with_conv_diameter_sum=True)
    config.update(mcs_size_method="conv_diameter_sum", mcs_size_thresh=40, mcs_dbz45_area_varname="area_45dbz", mcs_lifecycle_classes=None,
                  mcs_lifecycle_method="single", mcs_lifecycle_line_method="conv_diameter_sum", mcs_lifecycle_line_thresh=40)
    dense, rows = run_both(config)
    assert table_differences(dense, rows) == []
    assert rows.tracks["lifecycle_nmature"].tolist() == [1, 0] and rows.tracks["lifecycle_erratic_flag"].tolist() == [0, 0]
    assert "lifecycle_index__5" in rows.tracks                                   # six stages: the sixth column is redeveloping
    stages = rows.entries["lifecycle_stage"]
    assert 3 in stages and 4 in stages and 6 not in stages
    # natural upscale growth: the flag and the three times of every staged track are those of upscale_growth on its stages
    # (the default minimum time to mature is 1 h); the dense step wrote the same values (table_differences above)
    offsets = rows.offsets
    basetime = np.asarray(rows.entries["base_time"], dtype=float)
    for nt in range(len(offsets) - 1):
        sl = slice(offsets[nt], offsets[nt + 1])
        if not rows.tracks["lifecycle_evaluated"][nt]:
            assert rows.tracks["lifecycle_upscale_flag"][nt] == -9999 and np.isnan(rows.tracks["lifecycle_time_to_genesis_hr"][nt])
            continue
        flag, t_g, t_m, t_r = upscale_growth(np.asarray(stages[sl]), (basetime[sl] - basetime[sl][0]) / 3600.0, 1.0)
        assert rows.tracks["lifecycle_upscale_flag"][nt] == flag
        got = [rows.tracks[f"lifecycle_time_to_{name}_hr"][nt] for name in ("genesis", "mature", "redeveloping")]
        np.testing.assert_allclose(got, [t_g, t_m, t_r], equal_nan=True, atol=1e-5)
    assert rows.tracks["lifecycle_upscale_flag"][0] & 2 and rows.tracks["lifecycle_upscale_flag"][0] & 4        # track A: genesis and mature
    assert rows.meta["global_attrs"]["mcs_upscale_min_mature_hour"] == 1.0
    attrs = rows.meta["global_attrs"]
    assert attrs["mcs_lifecycle_method"] == "single" and attrs["mcs_lifecycle_line_method"] == "conv_diameter_sum"


def test_the_episodes_method_keeps_five_stages_and_no_new_variables(tmp_path):
    """An earlier config: the major axis, the episodes, the five columns of lifecycle_index and none of the new per-track variables."""
    config, _ = build_run(tmp_path, radius_km=10.0, with_conv_diameter_sum=True)
    config.update(mcs_size_method="conv_diameter_sum", mcs_size_thresh=40, mcs_dbz45_area_varname="area_45dbz", mcs_lifecycle_classes=None,
                  mcs_lifecycle_method="episodes", mcs_lifecycle_line_method="conv_majoraxis")
    dense, rows = run_both(config)
    assert table_differences(dense, rows) == []
    assert "lifecycle_index__4" in rows.tracks and "lifecycle_index__5" not in rows.tracks
    assert "lifecycle_nmature" not in rows.tracks and "lifecycle_erratic_flag" not in rows.tracks
    assert "lifecycle_upscale_flag" not in rows.tracks and "lifecycle_time_to_mature_hr" not in rows.tracks
    assert rows.meta["global_attrs"]["mcs_lifecycle_method"] == "episodes"


def test_the_diameter_sum_line_needs_conv_diameter_sum_in_the_first_pass_file(tmp_path):
    config, _ = build_run(tmp_path, radius_km=10.0)                          # first-pass tables from before conv_diameter_sum existed
    config.update(mcs_size_method="majoraxis", mcs_dbz45_area_varname="area_45dbz", mcs_lifecycle_classes=None,
                  mcs_lifecycle_method="single", mcs_lifecycle_line_method="conv_diameter_sum")
    convert_first_pass(config)
    with pytest.raises(SystemExit):
        compute_regionprops(config)
    with pytest.raises(SystemExit):
        compute_regionprops_rows(config)
