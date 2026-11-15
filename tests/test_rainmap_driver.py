"""
The monthly rain maps of the pipeline step rainmap (pyflextrkr/rainmap_driver.py): the maps of every calendar month equal plain sums over the
hours of the month, the files equal those of Analysis/calc_tbpf_mcs_monthly_rainmap.py for the same pixel-level files (same variables, values,
dtypes, attributes), and the output check of the step sees them. Synthetic pixel-level files (tests/synthetic_pixels.py) spanning two months.
"""
import importlib.util
import os
import sys

import numpy as np
import pytest
import xarray as xr

from pyflextrkr import rainmap_driver as R
from pyflextrkr.step_outputs import check_step_outputs
from synthetic_pixels import ENDDATE, NHOUR, NX, NY, STARTDATE, T0, write_pixel_files
from tbpf_compare import netcdf_differences

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JAN_HOURS, FEB_HOURS = [0, 1, 2], [3, 4, 5]


@pytest.fixture
def run(tmp_path):
    """The pixel files of two months and the config of a run."""
    pixel, stats = tmp_path / "mcstracking", tmp_path / "stats"
    pixel.mkdir()
    stats.mkdir()
    fields = write_pixel_files(str(pixel))
    config = {"stats_outpath": str(stats) + "/", "startdate": STARTDATE, "enddate": ENDDATE, "start_basetime": T0,
              "end_basetime": T0 + NHOUR * 3600, "pixeltracking_outpath": str(pixel) + "/", "pixeltracking_filebase": "mcstrack_",
              "feature_type": "tb_pf"}
    return config, fields


def _expected(fields, hours):
    """The four maps of some hours, plain sums."""
    mask, pcpmask, rain = (fields[k][hours] for k in ("cloudtracknumber", "pcptracknumber", "precipitation"))
    return {"precipitation": np.nansum(rain, axis=0), "mcs_precipitation": np.nansum(np.where(mask > 0, rain, np.nan), axis=0),
            "mcs_precipitation_count": (pcpmask > 0).sum(axis=0), "mcs_cloud_count": (mask > 0).sum(axis=0)}


def test_the_files_of_the_months_have_the_sums_of_their_hours(run):
    config, fields = run
    assert [R.rainmap_file(config, y, m).rsplit("/", 1)[-1] for y, m in ((2009, 1), (2009, 2))] == ["mcs_rainmap_200901.nc", "mcs_rainmap_200902.nc"]
    files = R.rainmap_driver(config)
    assert [f.rsplit("/", 1)[-1] for f in files] == ["mcs_rainmap_200901.nc", "mcs_rainmap_200902.nc"]
    for month, hours in ((1, JAN_HOURS), (2, FEB_HOURS)):
        with xr.open_dataset(R.rainmap_file(config, 2009, month)) as ds:
            for name, truth in _expected(fields, hours).items():
                np.testing.assert_allclose(ds[name].values[0], truth, rtol=1e-6, atol=1e-6, err_msg=f"{name} month {month}")
            assert int(ds["ntimes"].values[0]) == 3 and ds["precipitation"].dtype == np.float32
            assert ds["time"].values[0] == np.datetime64(f"2009-{month:02d}-01", "ns")                    # the first second of the month
            assert ds["precipitation"].attrs["units"] == "mm" and ds["mcs_cloud_count"].attrs["units"] == "hour"
            assert ds["longitude"].shape == (NY, NX)


def test_the_files_equal_those_of_the_analysis_script(run, tmp_path, monkeypatch):
    pytest.importorskip("psutil")
    pytest.importorskip("pytz")
    config, fields = run
    R.rainmap_driver(config)
    # The script of Analysis/ writes to <stats_outpath>monthly/: give it another stats folder, its argument are config, year, month
    spec = importlib.util.spec_from_file_location("calc_tbpf_mcs_monthly_rainmap", os.path.join(REPO, "Analysis", "calc_tbpf_mcs_monthly_rainmap.py"))
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    other = dict(config, stats_outpath=str(tmp_path / "legacy_stats") + "/")
    monkeypatch.setattr(legacy, "load_config", lambda path: other)
    for year, month in ((2009, 1), (2009, 2)):
        monkeypatch.setattr(sys, "argv", ["calc_tbpf_mcs_monthly_rainmap.py", "config.yml", str(year), str(month)])
        legacy.main()
        # created_on is the time stamp of the script, the same attribute the driver writes
        assert netcdf_differences(R.rainmap_file(config, year, month), R.rainmap_file(other, year, month), ignore_global_attrs=("created_on",)) == []


def test_a_month_without_pixel_files_has_no_file_and_the_check_follows_the_period(run):
    config, fields = run
    # A period of January to March: March has no file and is not asked for, the check needs the two months that have files
    wide = dict(config, end_basetime=T0 + 60 * 24 * 3600)
    assert [f.rsplit("/", 1)[-1] for f in R.expected_outputs(wide)] == ["mcs_rainmap_200901.nc", "mcs_rainmap_200902.nc"]
    assert R.compute_monthly_rainmap(wide, 2009, 3) is None
    ok, message = check_step_outputs("rainmap", wide)
    assert not ok and "2 of 2 monthly files are missing or empty" in message
    R.rainmap_driver(wide)
    assert check_step_outputs("rainmap", wide) == (True, "2 monthly files")
    # an empty file is not a result
    open(R.rainmap_file(wide, 2009, 2), "w").close()
    assert not check_step_outputs("rainmap", wide)[0]
    # no pixel files at all
    empty = dict(config, pixeltracking_outpath=config["stats_outpath"])
    assert check_step_outputs("rainmap", empty) == (False, "no pixel-level files in the period")


def test_a_second_run_replaces_the_files_and_rainmap_outdir_moves_them(run, tmp_path):
    config, fields = run
    R.rainmap_driver(config)
    R.rainmap_driver(config)                                                          # runs again without a conflict with its own files
    moved = dict(config, rainmap_outdir=str(tmp_path / "maps"))
    R.rainmap_driver(moved)
    assert os.path.isfile(str(tmp_path / "maps" / "mcs_rainmap_200901.nc")) and R.expected_outputs(moved)[0].startswith(str(tmp_path / "maps"))


def test_no_xarray_file_handle_or_lock_is_in_the_graph(run):
    """
    The graph of the maps holds paths and arrays only. A lazily opened dataset (open_mfdataset) puts its file handles and the netCDF locks of the
    main process into the graph; on the workers the lock was left locked by xarray's CachingFileManager.__del__ (CombinedLock.acquire(blocking=False)
    is not all-or-nothing) and the step hung (2019 IMERG, 2026-11-14).
    """
    cloudpickle = pytest.importorskip("cloudpickle")
    config, fields = run
    files = R.pixel_files_of_month(config, 2009, 1)
    lazy = R.lazy_fields_of_month(files)
    graph = cloudpickle.dumps([lazy[k].data.__dask_graph__() for k in ("cloud", "pcp", "precip")])
    assert b"xarray.backends" not in graph
    # the control: the way of the script of Analysis/ does put them into the graph
    ds = xr.open_mfdataset(files, concat_dim="time", combine="nested")
    assert b"xarray.backends" in cloudpickle.dumps(ds["precipitation"].data.__dask_graph__())
    ds.close()


def test_a_dask_cluster_gives_the_same_files_as_the_threaded_scheduler(run, tmp_path):
    distributed = pytest.importorskip("distributed")
    config, fields = run
    R.rainmap_driver(config)                                                          # no client: the threaded scheduler
    other = dict(config, rainmap_outdir=str(tmp_path / "cluster"))
    # two worker processes with one thread each, as the workers of a campaign
    with distributed.LocalCluster(n_workers=2, threads_per_worker=1, processes=True, dashboard_address=None) as cluster, \
            distributed.Client(cluster):
        R.rainmap_driver(other)
    for year, month in ((2009, 1), (2009, 2)):
        assert netcdf_differences(R.rainmap_file(config, year, month), R.rainmap_file(other, year, month), ignore_global_attrs=("created_on",)) == []


def test_a_file_with_another_grid_or_with_two_times_is_refused(run):
    config, fields = run
    files = R.pixel_files_of_month(config, 2009, 2)
    # the grid of the second file is shifted
    with xr.open_dataset(files[1]) as ds:
        ds = ds.load()
    ds.assign_coords(lat=ds["lat"] + 1).to_netcdf(files[1])
    with pytest.raises(ValueError, match="another grid"):
        R.compute_monthly_rainmap(config, 2009, 2)
    # the third file holds two times
    with xr.open_dataset(files[2]) as ds:
        ds = ds.load()
    xr.concat([ds, ds], dim="time").to_netcdf(files[2])
    ds.to_netcdf(files[1])                                                            # the second file is the original again
    with pytest.raises(ValueError, match="holds 2 times"):
        R.compute_monthly_rainmap(config, 2009, 2)
