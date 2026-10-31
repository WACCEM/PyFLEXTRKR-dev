"""
Tests of the rows mode of movement speed (pyflextrkr/movement_speed_rows.py): on the same pixel files and the same robust
statistics, the row tables of movement_speed_rows are identical (every column, dtype and value, and the meta data) to what
dense_to_rows makes of the dense file of movement_speed. Uses the synthetic pixel files of test_movement_speed.py.
"""
import os

import numpy as np
import pytest

from pyflextrkr import movement_speed as dense_step
from pyflextrkr.movement_speed_rows import MOVEMENT_NAMES, movement_speed_rows
from pyflextrkr.rowstats import dense_to_rows, read_tables, table_differences, write_rows
from test_movement_speed import make_config, pixel_dir, write_pixel_files  # noqa: F401  (pixel_dir is a fixture)

DATES = "20250502.1900_20250502.2000"


def run_both(pixel_dir, name, **overrides):
    """
    Run the dense and the rows step on the same inputs.

    Args:
        pixel_dir: string
            Folder with the synthetic pixel files (fixture of test_movement_speed).
        name: string
            Name of the run (folder below pixel_dir).
        **overrides:
            Config keys to change.

    Returns:
        dense_tables, rows_tables: RowTables
            dense_to_rows of the dense final file, and the tables the rows step wrote.
    """
    config = dict(make_config(pixel_dir, name), **overrides)
    stats = config["stats_outpath"]
    write_rows(dense_to_rows(f"{stats}robust_{DATES}.nc"), f"{stats}robust_{DATES}.rows")
    dense_file = dense_step.movement_speed(config)
    rows_folder = movement_speed_rows(config)
    assert os.path.basename(rows_folder) == f"final_{DATES}.rows"
    return dense_to_rows(dense_file), read_tables(rows_folder)


def test_rows_step_equals_dense_step(pixel_dir):
    dense, rows = run_both(pixel_dir, "rows_a")
    assert table_differences(dense, rows) == []
    speed = rows.entries["movement_speed"]
    assert np.isfinite(speed).sum() == 18 and np.allclose(speed[np.isfinite(speed)][:7], 20.0, rtol=1e-5)


@pytest.mark.parametrize("threshold", [50, 12, 5])
def test_rows_step_equals_dense_step_for_other_speed_thresholds(pixel_dir, threshold):
    # 12 m/s: the 20 m/s of track 1 is interpolated or set to NaN; 5 m/s: too few values below it, tracks become NaN
    dense, rows = run_both(pixel_dir, f"rows_b{threshold}", max_speed_thresh=threshold)
    assert table_differences(dense, rows) == []


@pytest.mark.parametrize("lag", [2, 3])
def test_rows_step_equals_dense_step_for_longer_lags(pixel_dir, lag):
    dense, rows = run_both(pixel_dir, f"rows_lag{lag}", lag_for_speed=lag)
    assert table_differences(dense, rows) == []
    speed = rows.entries["movement_speed"]
    assert np.isfinite(speed).sum() >= 10 - lag + 1                       # track 2 alone has nfiles - lag values


def test_the_robust_variables_are_kept_and_the_movement_variables_added(pixel_dir):
    _, rows = run_both(pixel_dir, "rows_c")
    robust = read_tables(make_config(pixel_dir, "rows_c")["stats_outpath"] + f"robust_{DATES}.rows")
    assert set(rows.meta["vars_2d"]) == set(robust.meta["vars_2d"]) | set(MOVEMENT_NAMES)
    assert np.array_equal(rows.entries["base_time"], robust.entries["base_time"])
    assert rows.meta["global_attrs"]["max_speed_thresh"] == 50


def test_input_and_output_must_differ(pixel_dir):
    config = make_config(pixel_dir, "rows_d")
    stats = config["stats_outpath"]
    write_rows(dense_to_rows(f"{stats}robust_{DATES}.nc"), f"{stats}robust_{DATES}.rows")
    config["mcsfinal_filebase"] = "robust_"
    with pytest.raises(SystemExit):
        movement_speed_rows(config)
