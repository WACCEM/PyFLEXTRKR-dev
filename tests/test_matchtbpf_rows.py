"""
Tests of the rows mode of matchpf (pyflextrkr/matchtbpf_rows.py): for the same Tb statistics of the MCS and the same per-file function
(a deterministic stand-in of matchtbpf_func.matchtbpf_singlefile, whose calculation is not touched), the row tables of
match_tbpf_tracks_rows are identical (every column, dtype and value, and the meta data) to what dense_to_rows makes of the dense file of
match_tbpf_tracks. With and without the periodic-boundary adjustment of the longitudes and latitudes, with merge / split slots, with
files that give no result. Synthetic data only.
"""
import os

import numpy as np
import pytest
import xarray as xr

from pyflextrkr.identifymcs import identifymcs_tb
from pyflextrkr.identifymcs_tb_rows import identifymcs_tb_rows
from pyflextrkr.matchtbpf_driver import match_tbpf_tracks
from pyflextrkr.matchtbpf_rows import match_tbpf_tracks_rows
from pyflextrkr.rowstats import RowStats, dense_to_rows, read_tables, rows_to_dense, table_differences
from synthetic_tbpf import (ENDDATE, NMAXPF, STARTDATE, T0, assert_same_dense, fake_matchtbpf_singlefile, random_tb_sparse_file, tb_config,
                            write_cloudid_files)


def prepare(tmp_path, seed, **overrides):
    """
    The Tb statistics of the MCS in both formats (dense file and row folder) and the cloudid files of every hour of the tracks, with
    the config of the matchpf steps.

    Args:
        tmp_path: pathlib.Path
            Folder for the run.
        seed: int
            Random seed of the tracks.
        **overrides:
            Config keys to change.

    Returns:
        config: dictionary
    """
    stats, tracking = tmp_path / "stats", tmp_path / "tracking"
    stats.mkdir()
    tracking.mkdir()
    random_tb_sparse_file(str(stats), seed, ntracks=90, crowd=4)
    config = tb_config(str(stats) + "/", feature_type="tb_pf", tracking_outpath=str(tracking) + "/", cloudid_filebase="cloudid_",
                       run_parallel=0, match_pixel_dt_thresh=60, start_basetime=T0 - 3600.0, end_basetime=T0 + 200 * 3600.0,
                       pbc_direction="none", pbc_max_domain_fraction=0.5)
    config.update(overrides)
    identifymcs_tb(config)
    identifymcs_tb_rows(config)
    # A cloudid file for every hour of the period of the tracks (a few of them have no MCS in it)
    basetimes = np.unique(read_tables(f"{config['stats_outpath']}mcs_tracks_{STARTDATE}_{ENDDATE}.rows").entries["base_time"])
    write_cloudid_files(str(tracking) + "/", np.arange(basetimes.min() - 2 * 3600.0, basetimes.max() + 3 * 3600.0, 3600.0))
    return config


def run_both(config, monkeypatch):
    """
    Run the dense and the rows matchpf on the same Tb statistics with the stand-in per-file function.

    Returns:
        dense_file, dense_tables, rows_tables: string, RowTables, RowTables
    """
    monkeypatch.setattr("pyflextrkr.matchtbpf_func.matchtbpf_singlefile", fake_matchtbpf_singlefile)
    dense_file = match_tbpf_tracks(config)
    rows_folder = match_tbpf_tracks_rows(config)
    assert rows_folder.endswith(f"mcs_tracks_pf_{STARTDATE}_{ENDDATE}.rows") and os.path.isdir(rows_folder)
    return dense_file, dense_to_rows(dense_file), read_tables(rows_folder)


@pytest.mark.parametrize("seed", [0, 3, 8])
def test_rows_step_equals_dense_step(tmp_path, monkeypatch, seed):
    config = prepare(tmp_path, seed)
    dense_file, dense, rows = run_both(config, monkeypatch)
    assert table_differences(dense, rows) == []
    # The PF statistics are small 3-D variables of the entries table, the others are columns; the Tb variables are still there
    assert rows.meta["vars_entry3d"]["pf_area"] == NMAXPF and "total_rain" in rows.meta["vars_2d"] and "ccs_area" in rows.meta["vars_2d"]
    assert rows.meta["global_attrs"]["nmaxpf"] == NMAXPF and rows.meta["global_attrs"]["landfrac_thresh"] == [0, 90]
    # Some entries were matched and some were not (a file without a result leaves NaN)
    matched = np.isfinite(rows.entries["total_rain"])
    assert 0 < matched.sum() < len(matched)


def test_rows_back_to_dense_equal_the_dense_file(tmp_path, monkeypatch):
    config = prepare(tmp_path, 2)
    dense_file, _, rows = run_both(config, monkeypatch)
    want = xr.open_dataset(dense_file, mask_and_scale=False, decode_times=False)
    assert_same_dense(rows_to_dense(rows), want)


@pytest.mark.parametrize("seed", [1, 5])
def test_the_periodic_boundary_adjustment_gives_the_dense_values(tmp_path, monkeypatch, seed):
    config = prepare(tmp_path, seed, pbc_direction="x")
    dense_file, dense, rows = run_both(config, monkeypatch)
    assert table_differences(dense, rows) == []
    # The adjustment did something: the stand-in makes longitudes inside +-180 degrees, so values beyond them were adjusted
    lon = np.concatenate([rows.entries[f"pf_lon__{j}"] for j in range(NMAXPF)])
    assert np.nanmax(np.abs(lon)) > 180.0


def test_the_slots_of_the_merge_and_split_clouds_are_passed_in_full_width(tmp_path, monkeypatch):
    # total_heavyrain of the stand-in carries 0.001 x the number of slots that were passed, the number of slots that are filled and
    # the sum of their cloud numbers: the tables are equal only if the rows step passes the arrays of the dense step (all 6 slots)
    config = prepare(tmp_path, 4, nmaxlinks=6)
    dense_file, dense, rows = run_both(config, monkeypatch)
    assert table_differences(dense, rows) == []
    assert rows.fragments["slot"].max() >= 1 and np.isfinite(rows.entries["total_heavyrain"]).any()


def test_no_result_for_any_file_stops_the_rows_step(tmp_path, monkeypatch):
    config = prepare(tmp_path, 0)
    monkeypatch.setattr("pyflextrkr.matchtbpf_func.matchtbpf_singlefile", lambda *args: None)
    with pytest.raises(SystemExit) as stop:
        match_tbpf_tracks_rows(config)
    assert stop.value.code == 1


def test_rowstats_reads_the_pf_variables_of_the_written_folder(tmp_path, monkeypatch):
    config = prepare(tmp_path, 6)
    _, dense, rows = run_both(config, monkeypatch)
    rs = RowStats.open(f"{config['stats_outpath']}mcs_tracks_pf_{STARTDATE}_{ENDDATE}.rows")
    pf_area = rs["pf_area"]
    assert pf_area.flat.shape == (len(rows.entries["track"]), NMAXPF) and len(pf_area) == len(rows.duration)
    assert np.array_equal(pf_area.flat[:, 1], rows.entries["pf_area__1"], equal_nan=True)
