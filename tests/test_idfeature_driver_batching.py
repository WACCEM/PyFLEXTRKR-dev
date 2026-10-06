"""
Tests for idfeature_driver.py's new "combine_files" batched time-frame dispatch
(feature_type='radar_feature' and friends), added in addition to the existing
per-file/per-time dispatch.

This test uses dask's default (threaded) scheduler via dask.compute() - no
dask.distributed Client/LocalCluster is created here, so no extra worker
processes are spawned. It is a lightweight, CI-safe check of the batching
*logic* (file combination, per-time dispatch, batch sizing), not a
multi-worker production run (that is exercised separately on a compute node).

pytest.importorskip below guards against environments lacking the
`distributed` package, since idfeature_driver.py imports
`dask.distributed.wait` at module level (needed as a plain function call,
not to create a cluster).
"""

import os
import glob
import numpy as np
import pytest
import xarray as xr

pytest.importorskip("distributed")

from pyflextrkr.idfeature_driver import idfeature_driver


NY, NX, NZ = 40, 40, 6
DX = 3000.0
HEIGHT_1D = np.arange(NZ, dtype=np.float32) * 1000.0
CORE_ROWS = slice(10, 15)
CORE_COLS = slice(10, 18)
SHIELD_ROWS = slice(5, 20)
SHIELD_COLS = slice(5, 23)

TIMESTAMPS = ["20250502-190000", "20250502-190500", "20250502-191000"]


def _write_mrms_like_file(dirpath, timestamp_str):
    no_echo = -99.0
    refl = np.full((1, NZ, NY, NX), no_echo, dtype=np.float32)
    refl[0, :, SHIELD_ROWS, SHIELD_COLS] = 15.0
    refl[0, :, CORE_ROWS, CORE_COLS] = 45.0

    x_1d = np.arange(NX, dtype=np.float64) * DX
    y_1d = np.arange(NY, dtype=np.float64) * DX
    lon2d, lat2d = np.meshgrid(-100.0 + x_1d / 111000.0, 35.0 + y_1d / 111000.0)

    iso_time = (
        f"{timestamp_str[0:4]}-{timestamp_str[4:6]}-{timestamp_str[6:8]}T"
        f"{timestamp_str[9:11]}:{timestamp_str[11:13]}:{timestamp_str[13:15]}"
    )
    ds = xr.Dataset(
        {
            'Reflectivity': xr.DataArray(refl, dims=['time', 'z', 'y', 'x'], attrs={'units': 'dBZ'}),
            'lon': xr.DataArray(lon2d, dims=['y', 'x']),
            'lat': xr.DataArray(lat2d, dims=['y', 'x']),
        },
        coords={
            'time': xr.DataArray(np.array([iso_time], dtype='datetime64[ns]'), dims=['time']),
            'x': xr.DataArray(x_1d, dims=['x']),
            'y': xr.DataArray(y_1d, dims=['y']),
            'z': xr.DataArray(HEIGHT_1D, dims=['z']),
        },
    )
    fname = f"MRMS_Reflectivity_HRRR_{timestamp_str}.nc"
    ds.to_netcdf(os.path.join(dirpath, fname))
    return fname


def _epoch(timestamp_str):
    import pandas as pd
    iso_time = (
        f"{timestamp_str[0:4]}-{timestamp_str[4:6]}-{timestamp_str[6:8]}T"
        f"{timestamp_str[9:11]}:{timestamp_str[11:13]}:{timestamp_str[13:15]}"
    )
    return int((pd.Timestamp(iso_time) - pd.Timestamp('1970-01-01')).total_seconds())


def _base_config(clouddata_path, tracking_outpath, combine_files, run_parallel):
    return {
        'run_parallel': run_parallel,
        'nprocesses': 2,
        'feature_type': 'radar_feature',
        'input_format': 'netcdf',
        'combine_files': combine_files,
        'clouddata_path': clouddata_path,
        'databasename': 'MRMS_Reflectivity_HRRR_',
        'time_format': 'yyyymodd-hhmmss',
        'start_basetime': _epoch(TIMESTAMPS[0]),
        'end_basetime': _epoch(TIMESTAMPS[-1]),
        'time_dimname': 'time',
        'tracking_outpath': tracking_outpath,
        'cloudid_filebase': 'cloudid_',
        'reflectivity_varname': 'Reflectivity',
        'x_dimname': 'x', 'y_dimname': 'y', 'z_dimname': 'z',
        'time_coordname': 'time',
        'x_coordname': 'x', 'y_coordname': 'y', 'z_coordname': 'z',
        'lon_coordname': 'lon', 'lat_coordname': 'lat',
        'dx': DX, 'dy': DX,
        'radar_sensitivity': 0.0,
        'sfc_dz_min': 0.0, 'sfc_dz_max': 20000.0,
        'z_coord_type': 'height',
        'echotop_gap': 1,
        'fillval': -9999,
        'pixel_radius': DX / 1000.0,
        'area_method': 'fixed',
        'feature_varname': 'feature_number',
        'nfeature_varname': 'nfeatures',
        'featuresize_varname': 'npix_feature',
        'feature_dbz_threshs': [30, 10],
        'smooth_size': 5,
        'min_core_npix': 4,
        'area_thresh': 36,
        'core_operator': 'gt',
        'growth_method': 'edt',
        'feature_expand_tertiary': 0,
        'absConvThres': 45, 'minZdiff': 8, 'truncZconvThres': 42,
        'mindBZuse': 20, 'dBZforMaxConvRadius': 45,
        'conv_rad_start': 1.0, 'maxConvRadius': 2.0, 'conv_rad_increment': 0.5,
        'bkg_refl_increment': 5, 'bkgrndRadius': 6.0, 'weakEchoThres': 15,
        'min_corearea': 0, 'min_cellarea': 0,
        'remove_smallcores': True, 'remove_smallcells': False,
        'return_diag': False, 'convolve_method': 'ndimage', 'dilate_method': 'orig',
        'cell_number_source': 'score_orig', 'cell_min_npix': 0,
    }


@pytest.fixture(scope='module')
def input_dir(tmp_path_factory):
    dirpath = str(tmp_path_factory.mktemp('mrms_batching_input')) + "/"
    for ts in TIMESTAMPS:
        _write_mrms_like_file(dirpath, ts)
    return dirpath


def _run_and_collect(input_dir, combine_files, run_parallel, tmp_path_factory, **overrides):
    tracking_outpath = str(tmp_path_factory.mktemp('mrms_batching_out')) + "/"
    config = _base_config(input_dir, tracking_outpath, combine_files, run_parallel)
    config.update(overrides)
    idfeature_driver(config)
    outfiles = sorted(glob.glob(os.path.join(tracking_outpath, "cloudid_*.nc")))
    return outfiles, config


class TestCombineFilesBatching:

    def test_combine_files_matches_per_file_dispatch_serial(self, input_dir, tmp_path_factory):
        files_per_file, _ = _run_and_collect(input_dir, combine_files=False, run_parallel=0,
                                              tmp_path_factory=tmp_path_factory)
        files_combined, _ = _run_and_collect(input_dir, combine_files=True, run_parallel=0,
                                              tmp_path_factory=tmp_path_factory)

        assert len(files_per_file) == len(TIMESTAMPS)
        assert len(files_combined) == len(TIMESTAMPS)

        for f1, f2 in zip(files_per_file, files_combined):
            ds1 = xr.open_dataset(f1, decode_times=False, mask_and_scale=False)
            ds2 = xr.open_dataset(f2, decode_times=False, mask_and_scale=False)
            np.testing.assert_array_equal(ds1['feature_number'].values, ds2['feature_number'].values)
            np.testing.assert_allclose(ds1['dbz_comp'].values, ds2['dbz_comp'].values, equal_nan=True)
            assert int(ds1['nfeatures'].values.squeeze()) == int(ds2['nfeatures'].values.squeeze())
            np.testing.assert_array_equal(ds1['base_time'].values, ds2['base_time'].values)

    @pytest.mark.local
    def test_combine_files_batched_matches_single_batch(self, input_dir, tmp_path_factory, caplog):
        """
        With run_parallel=1 and dask_batch_size=1, each of the 3 time frames
        should be its own batch (3 batches); results must still match the
        unbatched (large batch size) case.

        This is marked 'local' (skipped by default, see conftest.py) rather
        than run automatically: the driver's run_parallel>=1 path calls
        dask.distributed.wait(), which requires an active distributed
        Client/LocalCluster to be running - i.e. real multi-worker execution,
        not just dask's lightweight default threaded scheduler. Per the
        project's compute-node policy, that must run on a compute node, not
        the NERSC login node - run this explicitly there with
        PYFLEXTRKR_TEST_DATA set, e.g.:
          PYFLEXTRKR_TEST_DATA=1 <python> -m pytest tests/test_idfeature_driver_batching.py -m local -v
        """
        import logging
        from dask.distributed import LocalCluster, Client

        caplog.set_level(logging.INFO, logger="pyflextrkr.idfeature_driver")

        cluster = LocalCluster(n_workers=2, threads_per_worker=1)
        client = Client(cluster)
        try:
            files_small_batch, _ = _run_and_collect(
                input_dir, combine_files=True, run_parallel=1,
                tmp_path_factory=tmp_path_factory, dask_batch_size=1,
            )
            nbatch_msgs = [r.message for r in caplog.records if "batch(es) total" in r.message]
            assert len(nbatch_msgs) >= 1
            assert "3 batch(es) total" in nbatch_msgs[-1], nbatch_msgs

            files_one_batch, _ = _run_and_collect(
                input_dir, combine_files=True, run_parallel=1,
                tmp_path_factory=tmp_path_factory, dask_batch_size=100,
            )
        finally:
            client.close()
            cluster.close()

        assert len(files_small_batch) == len(TIMESTAMPS)
        assert len(files_one_batch) == len(TIMESTAMPS)
        for f1, f2 in zip(files_small_batch, files_one_batch):
            ds1 = xr.open_dataset(f1, decode_times=False, mask_and_scale=False)
            ds2 = xr.open_dataset(f2, decode_times=False, mask_and_scale=False)
            np.testing.assert_array_equal(ds1['feature_number'].values, ds2['feature_number'].values)

    def test_dask_batch_multiplier_default_sizing(self):
        from pyflextrkr.idfeature_driver import _get_dask_batch_size
        config = {'run_parallel': 1, 'nprocesses': 3}
        # default multiplier is 64 -> 3 * 64 = 192
        assert _get_dask_batch_size(config) == 192
        config['dask_batch_multiplier'] = 2
        assert _get_dask_batch_size(config) == 6
        config['dask_batch_size'] = 10
        assert _get_dask_batch_size(config) == 10
