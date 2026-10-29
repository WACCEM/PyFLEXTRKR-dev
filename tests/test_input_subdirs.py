"""
Tests for input files stored in sub-folders (config key `clouddata_subdir_format`, a strftime pattern such as
'%Y%m%d' for one folder per day): `ft_utilities.list_input_subdirs`, `subset_files_timerange(subdir_format=...)`,
`load_config`, and the netcdf branch of `idfeature_driver`.

The file lists are tested with empty files (names only). The driver test writes small MRMS-like files (the recipe of
test_idfeature_driver_batching.py) once in daily folders and once in one flat folder and compares the cloudid output.
Nothing here needs data or a Dask cluster.
"""

import glob
import os
import shutil

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from pyflextrkr import ft_utilities
from pyflextrkr.ft_utilities import list_input_subdirs, subset_files_timerange, load_config

BASENAME = "MRMS_HRRR_"
TIME_FORMAT = "yyyymodd_hhmmss"
DAYS = ["20250430", "20250501", "20250502", "20250503"]
TIMES = ["000000", "060000", "120000", "180000", "235500"]


def _epoch(text):
    """Epoch time [s] of a UTC date string, e.g. '2025-05-01T22:00'."""
    return int(pd.Timestamp(text, tz="UTC").timestamp())


def _make_files(root, folder_of_day=None):
    """
    Write empty MRMS_HRRR_<day>_<time>.nc files for DAYS x TIMES below root.

    Args:
        root: string
            Top directory.
        folder_of_day: function or None
            Sub-folder (relative to root) of a day 'YYYYMMDD'. None: all files directly in root.
    Returns:
        root with the trailing '/'.
    """
    for day in DAYS:
        folder = os.path.join(root, folder_of_day(day)) if folder_of_day else root
        os.makedirs(folder, exist_ok=True)
        for t in TIMES:
            open(os.path.join(folder, f"{BASENAME}{day}_{t}.nc"), "w").close()
    return str(root) + "/"


@pytest.fixture
def flat_dir(tmp_path):
    return _make_files(tmp_path / "flat")


@pytest.fixture
def daily_dir(tmp_path):
    return _make_files(tmp_path / "daily", lambda day: day)


def _names(info):
    """File names (no folder) of the first item of a subset_files_timerange result."""
    return [os.path.basename(f) for f in info[0]]


class TestSubsetFilesTimerange:

    def test_daily_folders_give_the_flat_result(self, flat_dir, daily_dir):
        # A window across midnight: the last file of 05-01 and the first two of 05-02
        start, end = _epoch("2025-05-01T22:00"), _epoch("2025-05-02T07:00")
        flat = subset_files_timerange(flat_dir, BASENAME, start, end, TIME_FORMAT)
        daily = subset_files_timerange(daily_dir, BASENAME, start, end, TIME_FORMAT, subdir_format="%Y%m%d")
        assert _names(daily) == _names(flat) == [
            "MRMS_HRRR_20250501_235500.nc", "MRMS_HRRR_20250502_000000.nc", "MRMS_HRRR_20250502_060000.nc",
        ]
        # the files are in their day folders
        assert daily[0][0] == os.path.join(daily_dir, "20250501", "MRMS_HRRR_20250501_235500.nc")
        np.testing.assert_array_equal(daily[1], flat[1])
        assert daily[2] == flat[2] and daily[3] == flat[3]

    def test_whole_period_is_in_time_order(self, flat_dir, daily_dir):
        start, end = _epoch("2025-04-30T00:00"), _epoch("2025-05-03T23:59")
        flat = subset_files_timerange(flat_dir, BASENAME, start, end, TIME_FORMAT)
        daily = subset_files_timerange(daily_dir, BASENAME, start, end, TIME_FORMAT, subdir_format="%Y%m%d")
        assert len(daily[0]) == len(DAYS) * len(TIMES)
        assert _names(daily) == _names(flat)
        assert np.all(np.diff(daily[1]) > 0)

    def test_only_the_folders_of_the_time_range_are_read(self, daily_dir, monkeypatch):
        listed = []
        real_listdir = os.listdir

        def recording_listdir(path):
            if str(path).startswith(daily_dir):
                listed.append(os.path.basename(os.path.normpath(path)))
            return real_listdir(path)

        monkeypatch.setattr(ft_utilities.os, "listdir", recording_listdir)
        start, end = _epoch("2025-05-02T00:00"), _epoch("2025-05-02T12:00")
        subset_files_timerange(daily_dir, BASENAME, start, end, TIME_FORMAT, subdir_format="%Y%m%d")
        # the day of the range plus one day of padding on each side; 04-30 is not read
        assert listed == ["20250501", "20250502", "20250503"]

    def test_missing_day_folder_is_skipped(self, daily_dir):
        shutil.rmtree(os.path.join(daily_dir, "20250502"))
        start, end = _epoch("2025-05-01T00:00"), _epoch("2025-05-03T23:59")
        out = subset_files_timerange(daily_dir, BASENAME, start, end, TIME_FORMAT, subdir_format="%Y%m%d")
        assert sorted(set(out[2])) == ["20250501", "20250503"]
        assert len(out[0]) == 2 * len(TIMES)

    def test_nested_year_month_day_folders(self, tmp_path, flat_dir):
        nested = _make_files(tmp_path / "nested", lambda day: f"{day[:4]}/{day[4:6]}/{day[6:]}")
        start, end = _epoch("2025-04-30T12:00"), _epoch("2025-05-01T12:00")
        flat = subset_files_timerange(flat_dir, BASENAME, start, end, TIME_FORMAT)
        out = subset_files_timerange(nested, BASENAME, start, end, TIME_FORMAT, subdir_format="%Y/%m/%d")
        assert _names(out) == _names(flat)
        assert len(out[0]) == 3 + 3     # 04-30 12:00, 18:00, 23:55 and 05-01 00:00, 06:00, 12:00

    def test_none_means_files_directly_in_data_path(self, flat_dir, daily_dir):
        start, end = _epoch("2025-05-01T00:00"), _epoch("2025-05-01T23:59")
        assert len(subset_files_timerange(flat_dir, BASENAME, start, end, TIME_FORMAT, subdir_format=None)[0]) == len(TIMES)
        # the default is None: files in sub-folders are not found, nothing is searched below data_path
        assert subset_files_timerange(daily_dir, BASENAME, start, end, TIME_FORMAT)[0] == []


class TestListInputSubdirs:

    def test_padding_and_time_order(self, daily_dir):
        start, end = _epoch("2025-05-02T05:00"), _epoch("2025-05-02T06:00")
        names = lambda folders: [os.path.basename(os.path.normpath(f)) for f in folders]
        assert names(list_input_subdirs(daily_dir, "%Y%m%d", start, end, pad_days=0)) == ["20250502"]
        assert names(list_input_subdirs(daily_dir, "%Y%m%d", start, end)) == ["20250501", "20250502", "20250503"]
        assert all(f.endswith("/") for f in list_input_subdirs(daily_dir, "%Y%m%d", start, end))

    def test_padding_finds_a_file_stored_in_the_previous_day_folder(self, daily_dir):
        # 00:00 of 05-02 stored in the folder of 05-01
        shutil.move(os.path.join(daily_dir, "20250502", "MRMS_HRRR_20250502_000000.nc"),
                    os.path.join(daily_dir, "20250501", "MRMS_HRRR_20250502_000000.nc"))
        start, end = _epoch("2025-05-02T00:00"), _epoch("2025-05-02T01:00")
        out = subset_files_timerange(daily_dir, BASENAME, start, end, TIME_FORMAT, subdir_format="%Y%m%d")
        assert _names(out) == ["MRMS_HRRR_20250502_000000.nc"]

    def test_monthly_folders_are_listed_once(self, tmp_path):
        root = _make_files(tmp_path / "monthly", lambda day: day[:6])
        start, end = _epoch("2025-04-30T00:00"), _epoch("2025-05-03T00:00")
        folders = list_input_subdirs(root, "%Y%m", start, end)
        assert [os.path.basename(os.path.normpath(f)) for f in folders] == ["202504", "202505"]


class TestLoadConfig:

    def _write(self, tmp_path, **extra):
        config = {
            "clouddata_path": str(tmp_path) + "/", "databasename": BASENAME, "time_format": TIME_FORMAT,
            "root_path": str(tmp_path / "out"), "tracking_path_name": "tracking", "stats_path_name": "stats",
            "pixel_path_name": "pixel",
        }
        config.update(extra)
        path = tmp_path / "config.yml"
        path.write_text(yaml.dump(config))
        return str(path)

    def test_dates_are_required_with_subdir_format(self, tmp_path):
        with pytest.raises(ValueError, match="clouddata_subdir_format"):
            load_config(self._write(tmp_path, clouddata_subdir_format="%Y%m%d"), make_dirs=False)

    def test_config_with_dates_keeps_the_key(self, tmp_path):
        config = load_config(
            self._write(tmp_path, clouddata_subdir_format="%Y%m%d", startdate="20250501.0000", enddate="20250502.0000"),
            make_dirs=False,
        )
        assert config["clouddata_subdir_format"] == "%Y%m%d"
        assert config["start_basetime"] == _epoch("2025-05-01T00:00")


class TestIdfeatureDriver:
    """The netcdf branch of idfeature_driver (serial, one task per file) reads daily folders like a flat folder."""

    TIMESTAMPS = ["20250502-235000", "20250502-235500", "20250503-000000", "20250503-000500"]

    def _run(self, batching, input_dir, tracking_outpath, **overrides):
        from pyflextrkr.idfeature_driver import idfeature_driver
        os.makedirs(tracking_outpath, exist_ok=True)     # load_config makes it in a real run
        config = batching._base_config(input_dir, tracking_outpath, combine_files=False, run_parallel=0)
        config.update(start_basetime=_epoch("2025-05-02T23:50"), end_basetime=_epoch("2025-05-03T00:05"))
        config.update(overrides)
        idfeature_driver(config)
        return sorted(glob.glob(os.path.join(tracking_outpath, "cloudid_*.nc")))

    def test_daily_folders_match_a_flat_folder(self, tmp_path):
        pytest.importorskip("distributed")      # idfeature_driver imports dask.distributed at the top of the module
        import test_idfeature_driver_batching as batching
        flat, daily = tmp_path / "flat", tmp_path / "daily"
        flat.mkdir()
        for ts in self.TIMESTAMPS:
            batching._write_mrms_like_file(str(flat), ts)
            (daily / ts[:8]).mkdir(parents=True, exist_ok=True)
            shutil.copy(flat / f"MRMS_Reflectivity_HRRR_{ts}.nc", daily / ts[:8])

        out_flat = self._run(batching, str(flat) + "/", str(tmp_path / "out_flat") + "/")
        out_daily = self._run(batching, str(daily) + "/", str(tmp_path / "out_daily") + "/",
                              clouddata_subdir_format="%Y%m%d")

        assert len(out_flat) == len(out_daily) == len(self.TIMESTAMPS)
        assert [os.path.basename(f) for f in out_daily] == [os.path.basename(f) for f in out_flat]
        for f1, f2 in zip(out_flat, out_daily):
            ds1 = xr.open_dataset(f1, decode_times=False, mask_and_scale=False)
            ds2 = xr.open_dataset(f2, decode_times=False, mask_and_scale=False)
            np.testing.assert_array_equal(ds1["feature_number"].values, ds2["feature_number"].values)
            np.testing.assert_allclose(ds1["dbz_comp"].values, ds2["dbz_comp"].values, equal_nan=True)
            np.testing.assert_array_equal(ds1["base_time"].values, ds2["base_time"].values)

    def test_no_file_in_the_time_range_stops_with_a_message(self, tmp_path):
        pytest.importorskip("distributed")
        import test_idfeature_driver_batching as batching
        (tmp_path / "daily" / "20250502").mkdir(parents=True)
        with pytest.raises(SystemExit):
            self._run(batching, str(tmp_path / "daily") + "/", str(tmp_path / "out") + "/",
                      clouddata_subdir_format="%Y%m%d")
