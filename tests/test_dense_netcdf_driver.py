"""
Tests of the step dense_netcdf (pyflextrkr/dense_netcdf_driver.py, run_dense_netcdf): the dense netCDF file of the final statistics is written
from the row folder under the name of the old dense step, equal to the dense file that the row folder was made from; it is written under a
temporary name; its output check; the default of the flag and its place in the steps of the run script. Synthetic files only.
"""
import importlib.util
import os
import time

import netCDF4
import pytest
import xarray as xr

from pyflextrkr import dense_netcdf_driver as driver, step_outputs as so
from test_rowstats import _assert_same_dataset, _raw, _rows_in_folder

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _config(tmp_path, **extra):
    """
    The config of a rows run whose final statistics are the folder r/mcs_tracks_a_b.rows of test_rowstats._rows_in_folder.
    """
    return dict({"feature_type": "tb_pf", "stats_format": "rows", "stats_outpath": str(tmp_path / "r") + "/", "mcsfinal_filebase": "mcs_tracks_",
                 "startdate": "a", "enddate": "b"}, **extra)


def _unified():
    """The run script as a module (it is a script in runscripts/, not a package)."""
    spec = importlib.util.spec_from_file_location("run_mcs_tbpf_unified", os.path.join(REPO, "runscripts", "run_mcs_tbpf_unified.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_dense_file_is_the_dense_file_the_rows_were_made_from_and_it_has_the_name_of_the_old_step(tmp_path):
    path, rows = _rows_in_folder(tmp_path)
    config = _config(tmp_path)
    assert driver.expected_output(config) == rows[:-5] + ".nc"
    out = driver.dense_netcdf_driver(config)
    assert out == rows[:-5] + ".nc"
    # every variable, dtype, fill value and attribute of the original dense file; the unlimited tracks dimension and zlib of the old step
    _assert_same_dataset(_raw(out), _raw(path))
    with netCDF4.Dataset(out) as nc:
        assert nc.dimensions["tracks"].isunlimited() and nc["area"].filters()["zlib"]
    # the temporary file is gone
    assert not os.path.exists(out + ".tmp")


def test_the_block_size_and_the_variables_come_from_the_config_and_an_existing_file_is_replaced(tmp_path):
    path, rows = _rows_in_folder(tmp_path)
    out = driver.dense_netcdf_driver(_config(tmp_path, dense_netcdf_block_tracks=2))
    _assert_same_dataset(_raw(out), _raw(path))
    # only some [tracks, times] variables (the 1-D and the merge / split variables are always written); the old file is replaced
    driver.dense_netcdf_driver(_config(tmp_path, dense_netcdf_variables=["area"]))
    names = set(_raw(out).variables)
    assert "area" in names and "cloudnumber" not in names and "track_duration" in names and "merge_cloudnumber" in names


def test_a_failed_conversion_leaves_no_file_that_looks_complete_and_keeps_the_old_one(tmp_path, monkeypatch):
    path, rows = _rows_in_folder(tmp_path)
    config = _config(tmp_path)
    # no row folder: an error that names it
    with pytest.raises(FileNotFoundError, match="does not exist"):
        driver.dense_netcdf_driver(_config(tmp_path, startdate="x"))
    # a run that dies while it writes: the stale file of the killed run stays under the temporary name, never under the final one
    real = driver.write_dense_netcdf

    def dies(folder, outfile, **kwargs):
        real(folder, outfile, **kwargs)
        raise RuntimeError("killed")
    monkeypatch.setattr(driver, "write_dense_netcdf", dies)
    with pytest.raises(RuntimeError):
        driver.dense_netcdf_driver(config)
    final = driver.expected_output(config)
    assert not os.path.exists(final) and os.path.exists(final + ".tmp")
    # the next run replaces the stale temporary file, and an earlier complete file stays until the new one is complete
    monkeypatch.setattr(driver, "write_dense_netcdf", real)
    driver.dense_netcdf_driver(config)
    assert os.path.exists(final) and not os.path.exists(final + ".tmp")
    before = os.path.getmtime(final)
    monkeypatch.setattr(driver, "write_dense_netcdf", dies)
    with pytest.raises(RuntimeError):
        driver.dense_netcdf_driver(config)
    assert os.path.getmtime(final) == before
    _assert_same_dataset(_raw(final), _raw(path))


def test_the_default_of_the_flag_is_rows_mode_in_the_run_that_does_the_speed_step():
    assert driver.default_run_flag({"stats_format": "rows", "run_speed": True})
    assert not driver.default_run_flag({"stats_format": "rows", "run_speed": False})          # a track job or a post-processing job
    assert not driver.default_run_flag({"stats_format": "rows"})
    assert not driver.default_run_flag({"stats_format": "dense", "run_speed": True})          # the old step writes the dense file
    assert not driver.default_run_flag({"run_speed": True})


def test_the_run_script_sets_the_flag_and_puts_the_step_after_speed_and_before_the_rain_maps(tmp_path, monkeypatch):
    unified = _unified()
    # the key is not in the config: rows mode with the speed step gets the dense file, a job without the speed step does not
    config = {"stats_format": "rows", "run_speed": True}
    unified.set_dense_netcdf_flag(config)
    assert config["run_dense_netcdf"] is True
    config = {"stats_format": "rows", "run_speed": False}
    unified.set_dense_netcdf_flag(config)
    assert config["run_dense_netcdf"] is False
    # the key is in the config: it is kept (off even in the run with the speed step), and with stats_format dense it has no effect
    config = {"stats_format": "rows", "run_speed": True, "run_dense_netcdf": False}
    unified.set_dense_netcdf_flag(config)
    assert config["run_dense_netcdf"] is False
    config = {"stats_format": "dense", "run_speed": True, "run_dense_netcdf": True}
    unified.set_dense_netcdf_flag(config)
    assert config["run_dense_netcdf"] is False
    # the place of the step: after speed, before the optional rain maps (stand-in module: rainmap needs its own imports)
    import sys
    import types
    fake = types.ModuleType("pyflextrkr.rainmap_driver")
    fake.rainmap_driver = lambda cfg: None
    monkeypatch.setitem(sys.modules, "pyflextrkr.rainmap_driver", fake)
    config = {"stats_format": "rows", "robustmcs_definition": "mcsmip", "mcsrobust_filebase": "r_", "run_dense_netcdf": True, "run_rainmap": True}
    names = [s[0] for s in unified.build_steps(config)]
    assert names[-3:] == ["speed", "dense_netcdf", "rainmap"]
    assert "dense_netcdf" not in [s[0] for s in unified.build_steps(dict(config, run_dense_netcdf=False))]


def test_the_check_of_the_step_asks_for_the_tracks_the_times_the_variables_and_a_file_newer_than_the_rows(tmp_path):
    path, rows = _rows_in_folder(tmp_path)
    config = _config(tmp_path)
    # no file yet, then the file of the step
    ok, message = so.check_step_outputs("dense_netcdf", config)
    assert not ok and "does not exist" in message
    driver.dense_netcdf_driver(config)
    ok, message = so.check_step_outputs("dense_netcdf", config)
    assert ok and message == "7 tracks x 6 times"
    # variables that the config asks for must be in the file
    ok, message = so.check_step_outputs("dense_netcdf", dict(config, dense_netcdf_variables=["area", "not_there"]))
    assert not ok and "['not_there'] are missing" in message
    # strict (can the step be skipped?): not when the row tables are newer than the file (the speed step wrote them again); after the run: yes
    entries = os.path.join(rows, "entries.parquet")
    later = time.time() + 100
    os.utime(entries, (later, later))
    ok, message = so.check_step_outputs("dense_netcdf", config, strict=True)
    assert not ok and "older than the row tables" in message
    assert so.check_step_outputs("dense_netcdf", config, strict=False)[0]
    # a file with other sizes than the row folder (another run) is refused
    other = xr.open_dataset(driver.expected_output(config), mask_and_scale=False, decode_times=False).isel(tracks=slice(0, 3)).load()
    os.remove(driver.expected_output(config))
    other.to_netcdf(driver.expected_output(config), unlimited_dims="tracks")
    ok, message = so.check_step_outputs("dense_netcdf", config, strict=False)
    assert not ok and "3 tracks x 6 times" in message and "7 x 6" in message


def test_run_step_runs_the_step_verifies_it_and_skips_it_when_its_file_is_complete(tmp_path):
    path, rows = _rows_in_folder(tmp_path)
    config = _config(tmp_path, run_dense_netcdf=True, skip_completed_steps=True, verify_step_outputs=True)
    state = {"stale": False}
    # the first run writes the file
    result = so.run_step("dense_netcdf", config, state, driver.dense_netcdf_driver)
    assert result == driver.expected_output(config) and os.path.isfile(result)
    # a later run finds it complete and newer than the row tables: skipped
    later = time.time() + 100
    os.utime(result, (later, later))
    called = []
    assert so.run_step("dense_netcdf", config, {"stale": False}, lambda cfg: called.append(1)) is None and not called
    # a step before it that ran (state stale) makes it run again; a config with the step off never runs it
    so.run_step("dense_netcdf", config, {"stale": True}, lambda cfg: called.append(1))
    assert called == [1]
    assert so.run_step("dense_netcdf", dict(config, run_dense_netcdf=False), {"stale": True}, lambda cfg: called.append(2)) is None
    # after an empty period (no tracked feature) the step is skipped like the other steps after trackstats
    assert so.run_step("dense_netcdf", config, {"stale": True, "empty": True}, lambda cfg: called.append(3)) is None and 3 not in called


def test_with_stats_format_dense_the_check_only_asks_that_the_final_file_opens(tmp_path):
    path, rows = _rows_in_folder(tmp_path)
    config = _config(tmp_path, stats_format="dense")
    ok, message = so.check_step_outputs("dense_netcdf", config)
    assert not ok and "does not exist" in message
    os.replace(path, rows[:-5] + ".nc")
    assert so.check_step_outputs("dense_netcdf", config)[0]
