"""
Tests of tests/tbpf_compare.py, the comparison of two tracking runs that the identity tests rely on: it has to find a changed value, a changed
attribute, a missing variable or file, and must not find the time stamp Created_on or the difference between a dense file and its row tables.
Synthetic files only.
"""
import os
import shutil

import numpy as np
import xarray as xr

from pyflextrkr.rowstats import dense_to_rows, write_rows
from tbpf_compare import compare_runs, netcdf_differences, stats_differences
from test_rowstats import _write
from test_rowstats_entry3d import _dense_pf


def _dataset(created="a", fill=1.0):
    ds = xr.Dataset({"x": (["t"], np.array([1.0, np.nan, 3.0], dtype=np.float32), {"units": "m", "_FillValue": np.float32(np.nan)}),
                     "n": (["t"], np.array([1, 2, 3], dtype=np.int32))}, attrs={"Title": "t", "Created_on": created, "levels": [1, 2]})
    ds["x"].values[0] = fill
    return ds


def test_netcdf_differences_finds_changes_and_ignores_the_time_stamp(tmp_path):
    _dataset("Mon").to_netcdf(tmp_path / "a.nc")
    _dataset("Tue").to_netcdf(tmp_path / "b.nc")
    assert netcdf_differences(str(tmp_path / "a.nc"), str(tmp_path / "b.nc")) == []             # NaN equals NaN, Created_on is ignored
    _dataset(fill=2.0).to_netcdf(tmp_path / "c.nc")
    assert netcdf_differences(str(tmp_path / "a.nc"), str(tmp_path / "c.nc")) == ["x: 1 of 3 values differ"]
    changed = _dataset()
    changed["x"].attrs["units"] = "km"
    changed.attrs["levels"] = [1, 3]
    changed["extra"] = (["t"], np.zeros(3))
    changed["n"] = changed["n"].astype(np.int64)
    changed.to_netcdf(tmp_path / "d.nc")
    found = netcdf_differences(str(tmp_path / "a.nc"), str(tmp_path / "d.nc"))
    assert any(d.startswith("x: attributes differ") for d in found) and "extra: only in the second file" in found
    assert any(d.startswith("n: dimensions / dtype differ") for d in found) and "global attributes differ: ['levels']" in found


def test_stats_differences_compares_a_dense_file_with_its_rows_and_finds_a_change(tmp_path):
    path = _write(tmp_path, _dense_pf())
    rows = str(tmp_path / "x.rows")
    write_rows(dense_to_rows(path), rows)
    assert stats_differences(path, rows) == [] and stats_differences(rows, rows) == []
    changed = dense_to_rows(path)
    changed.entries["pf_area__1"] = changed.entries["pf_area__1"] + 1
    other = str(tmp_path / "y.rows")
    write_rows(changed, other)
    assert stats_differences(path, other) == ["entries.pf_area__1: values differ"]


def _make_run(root, scale=1.0, with_rows=False):
    """A tiny run: two cloudid files, one pairwise link, three statistics files, one pixel file."""
    for sub in ("tracking", "stats", "mcstracking/20250101.0000_20250102.0000"):
        os.makedirs(f"{root}/{sub}", exist_ok=True)
    for name in ("cloudid_20250101_000000.nc", "cloudid_20250101_010000.nc", "track_20250101_010000.nc"):
        _dataset().to_netcdf(f"{root}/tracking/{name}")
    for name in ("tracknumbers_", "trackstats_sparse_", "trackstats_"):
        _dataset().to_netcdf(f"{root}/stats/{name}20250101.0000_20250102.0000.nc")
    _dataset(fill=scale).to_netcdf(f"{root}/mcstracking/20250101.0000_20250102.0000/mcstrack_20250101_000000.nc")


def test_compare_runs_counts_the_files_and_finds_a_missing_file_and_a_changed_pixel_file(tmp_path):
    _make_run(tmp_path / "a")
    _make_run(tmp_path / "b")
    report = compare_runs(str(tmp_path / "a"), str(tmp_path / "b"))
    assert report == {"compared": {"tracking": 3, "stats": 3, "pixel": 1}, "differences": []}
    # a changed pixel file, a track file that one run does not have, a statistics file that one run does not have
    _make_run(tmp_path / "c", scale=5.0)
    os.remove(tmp_path / "c" / "tracking" / "track_20250101_010000.nc")
    os.remove(tmp_path / "c" / "stats" / "tracknumbers_20250101.0000_20250102.0000.nc")
    found = compare_runs(str(tmp_path / "a"), str(tmp_path / "c"))["differences"]
    assert "tracking/track_20250101_010000.nc: only in the first run" in found
    assert "stats/tracknumbers_: only in the first run" in found
    assert "mcstracking/mcstrack_20250101_000000.nc: x: 1 of 3 values differ" in found and len(found) == 3


def test_compare_runs_takes_the_file_of_the_whole_period_in_a_stitched_root(tmp_path):
    # The root of a stitched campaign has the tracknumbers file of every period next to the stitched file of the whole run
    _make_run(tmp_path / "a")
    _make_run(tmp_path / "b")
    for period in ("20250101.0000_20250101.1200", "20250101.1200_20250102.0000"):
        _dataset(fill=9.0).to_netcdf(tmp_path / "b" / "stats" / f"tracknumbers_{period}.nc")
    report = compare_runs(str(tmp_path / "a"), str(tmp_path / "b"))
    assert report == {"compared": {"tracking": 3, "stats": 3, "pixel": 1}, "differences": []}
    # a stitched file that differs from the continuous one is found (the per-period files are not the ones compared)
    _dataset(fill=5.0).to_netcdf(tmp_path / "b" / "stats" / "tracknumbers_20250101.0000_20250102.0000.nc")
    assert compare_runs(str(tmp_path / "a"), str(tmp_path / "b"))["differences"] == ["stats/tracknumbers_: x: 1 of 3 values differ"]
    # two files of the longest period, or a file without dates, cannot be told apart: no file is compared and the run reports it
    shutil.copy(tmp_path / "b" / "stats" / "tracknumbers_20250101.0000_20250102.0000.nc", tmp_path / "b" / "stats" / "tracknumbers_20250201.0000_20250202.0000.nc")
    assert compare_runs(str(tmp_path / "a"), str(tmp_path / "b"))["differences"] == ["stats/tracknumbers_: only in the first run"]


def test_compare_runs_reads_the_row_folders_of_the_second_run(tmp_path):
    for name in ("a", "b"):
        _make_run(tmp_path / name)
    dense = _write(tmp_path, _dense_pf())
    shutil.copy(dense, tmp_path / "a" / "stats" / "mcs_tracks_20250101.0000_20250102.0000.nc")
    write_rows(dense_to_rows(dense), str(tmp_path / "b" / "stats" / "mcs_tracks_20250101.0000_20250102.0000.rows"))
    report = compare_runs(str(tmp_path / "a"), str(tmp_path / "b"), rows_b=True)
    assert report["differences"] == [] and report["compared"]["stats"] == 4
    # without rows_b the row folder is not found where the dense file is expected
    assert compare_runs(str(tmp_path / "a"), str(tmp_path / "b"))["differences"] == ["stats/mcs_tracks_: only in the first run"]
