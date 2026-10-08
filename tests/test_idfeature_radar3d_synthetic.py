"""
Synthetic end-to-end test for the radar-3D MCS feature identification module
(pyflextrkr.idfeature_radar3d.idfeature_radar3d), feature_type='radar_feature'.

This test builds a small synthetic 3D (and, separately, 2D) Cartesian radar
reflectivity dataset entirely in memory / a tmp_path, runs the full
idfeature_radar3d() function (reader -> label_and_grow_features ->
Steiner classification -> cell labeling -> echo-top heights -> netCDF write),
and checks the written cloudid file's contents. No real radar data is needed.

Synthetic grid layout (60x60 at 3 km, matching the MRMS/HRRR test grid):
  - A 6x10 pixel >30 dBZ "core" sitting inside a 20x20 pixel >10 dBZ "shield"
    (one contiguous detect-and-grow feature, with a height-dependent 3D
    profile designed to give a monotonically decreasing echo-top staircase:
    ETH10 > ETH20 > ETH30 > ETH40, and ETH50 undefined/NaN).
  - A separate 5x5 pixel >10 dBZ coreless blob (an "isolated secondary"
    feature per label_and_grow_features' own labeling of unlabeled secondary
    regions).
  - A single isolated >10 dBZ pixel, too small to pass area_thresh (should
    NOT become a feature).
  - A NaN band (outside radar coverage).
  - Everywhere else: -99 (no-echo sentinel, matching the real MRMS convention).
"""

import os
import numpy as np
import pytest
import xarray as xr
from scipy import ndimage

from pyflextrkr.idfeature_radar3d import idfeature_radar3d


NY, NX, NZ = 60, 60, 12
DX = 3000.0
HEIGHT_1D = np.arange(NZ, dtype=np.float32) * 1000.0  # 0..11 km, 1 km spacing

# Core block: rows 15-20 (6 rows), cols 15-24 (10 cols)
CORE_ROWS = slice(15, 21)
CORE_COLS = slice(15, 25)
# Shield block (contains the core): rows 10-29, cols 10-29 (20x20)
SHIELD_ROWS = slice(10, 30)
SHIELD_COLS = slice(10, 30)
# Isolated coreless blob: rows 40-44, cols 40-44 (5x5)
BLOB_ROWS = slice(40, 45)
BLOB_COLS = slice(40, 45)
# Too-small isolated speck: single pixel
SPECK_ROW, SPECK_COL = 5, 5
# NaN band: cols 50-54, all rows
NAN_COLS = slice(50, 55)


def _make_synthetic_dataset(is_3d=True):
    """Build a synthetic xr.Dataset matching the MRMS/HRRR variable layout."""
    no_echo = -99.0

    if is_3d:
        refl = np.full((1, NZ, NY, NX), no_echo, dtype=np.float32)
        # Shield: 15 dBZ at all levels
        refl[0, :, SHIELD_ROWS, SHIELD_COLS] = 15.0
        # Core: height-dependent profile -> ETH staircase
        # levels 0-3: 45 dBZ, 4-6: 35 dBZ, 7-9: 25 dBZ, 10-11: 15 dBZ
        refl[0, 0:4, CORE_ROWS, CORE_COLS] = 45.0
        refl[0, 4:7, CORE_ROWS, CORE_COLS] = 35.0
        refl[0, 7:10, CORE_ROWS, CORE_COLS] = 25.0
        refl[0, 10:12, CORE_ROWS, CORE_COLS] = 15.0
        # Isolated coreless blob: 15 dBZ at all levels
        refl[0, :, BLOB_ROWS, BLOB_COLS] = 15.0
        # Too-small speck: 12 dBZ at all levels
        refl[0, :, SPECK_ROW, SPECK_COL] = 12.0
        # NaN band (outside radar coverage)
        refl[0, :, :, NAN_COLS] = np.nan
        dims = ['time', 'z', 'y', 'x']
    else:
        refl2d = np.full((1, NY, NX), no_echo, dtype=np.float32)
        refl2d[0, SHIELD_ROWS, SHIELD_COLS] = 15.0
        refl2d[0, CORE_ROWS, CORE_COLS] = 45.0
        refl2d[0, BLOB_ROWS, BLOB_COLS] = 15.0
        refl2d[0, SPECK_ROW, SPECK_COL] = 12.0
        refl2d[0, :, NAN_COLS] = np.nan
        refl = refl2d
        dims = ['time', 'y', 'x']

    x_1d = np.arange(NX, dtype=np.float64) * DX
    y_1d = np.arange(NY, dtype=np.float64) * DX
    lon2d, lat2d = np.meshgrid(-100.0 + x_1d / 111000.0, 35.0 + y_1d / 111000.0)

    data_vars = {
        'Reflectivity': xr.DataArray(refl, dims=dims, attrs={'units': 'dBZ'}),
        'lon': xr.DataArray(lon2d.astype(np.float64), dims=['y', 'x']),
        'lat': xr.DataArray(lat2d.astype(np.float64), dims=['y', 'x']),
    }
    coords = {
        'time': xr.DataArray(
            np.array(['2025-05-02T19:00:43'], dtype='datetime64[ns]'), dims=['time'],
        ),
        'x': xr.DataArray(x_1d, dims=['x'], attrs={'units': 'm'}),
        'y': xr.DataArray(y_1d, dims=['y'], attrs={'units': 'm'}),
    }
    if is_3d:
        coords['z'] = xr.DataArray(HEIGHT_1D, dims=['z'], attrs={'units': 'm'})

    return xr.Dataset(data_vars, coords=coords)


def _minimal_config(tracking_outpath, dx=DX, return_diag=True):
    return {
        'input_format': 'netcdf',
        'reflectivity_varname': 'Reflectivity',
        'x_dimname': 'x', 'y_dimname': 'y', 'z_dimname': 'z',
        'time_dimname': 'time', 'time_coordname': 'time',
        'x_coordname': 'x', 'y_coordname': 'y', 'z_coordname': 'z',
        'lon_coordname': 'lon', 'lat_coordname': 'lat',
        'dx': dx, 'dy': dx,
        'radar_sensitivity': 0.0,
        'sfc_dz_min': 0.0, 'sfc_dz_max': 20000.0,
        'z_coord_type': 'height',
        'echotop_gap': 1,
        'echotop_method': 'orig',
        'fillval': -9999,
        'pixel_radius': dx / 1000.0,
        'area_method': 'fixed',
        'tracking_outpath': tracking_outpath,
        'cloudid_filebase': 'cloudid_',
        'feature_varname': 'feature_number',
        'nfeature_varname': 'nfeatures',
        'featuresize_varname': 'npix_feature',
        # label_and_grow_features
        'feature_dbz_threshs': [30, 10],
        'smooth_size': 5,
        'min_core_npix': 4,
        'area_thresh': 36,  # km^2 -> 4 pixels at 9 km^2/pixel
        'core_operator': 'gt',
        'growth_method': 'edt',
        'feature_expand_tertiary': 0,
        # Steiner classification (scaled to this synthetic grid; not meant to
        # match the real MRMS cell-tracking config's values, just exercised)
        'absConvThres': 45,
        'minZdiff': 8,
        'truncZconvThres': 42,
        'mindBZuse': 20,
        'dBZforMaxConvRadius': 45,
        'conv_rad_start': 1.0,
        'maxConvRadius': 2.0,
        'conv_rad_increment': 0.5,
        'bkg_refl_increment': 5,
        'bkgrndRadius': 6.0,  # km
        'weakEchoThres': 15,
        'min_corearea': 0,
        'min_cellarea': 0,
        'remove_smallcores': True,
        'remove_smallcells': False,
        'return_diag': return_diag,
        'convolve_method': 'ndimage',
        'dilate_method': 'orig',
        'cell_number_source': 'score_orig',
        'cell_min_npix': 0,
    }


class TestIdfeatureRadar3dSynthetic3D:

    @pytest.fixture(scope='class')
    def outputs(self, tmp_path_factory):
        tmpdir = tmp_path_factory.mktemp('radar3d_feature')
        tracking_outpath = str(tmpdir) + "/"
        ds = _make_synthetic_dataset(is_3d=True)
        fpath = os.path.join(tracking_outpath, 'input.nc')
        ds.to_netcdf(fpath)
        config = _minimal_config(tracking_outpath)

        # (a) filename input
        outfile_from_name = idfeature_radar3d(fpath, config)
        ds_out_name = xr.open_dataset(outfile_from_name, decode_times=False, mask_and_scale=False)

        # (b) Dataset input (mirrors idfeature_driver's combine_files path,
        # which hands in ds.isel({time_dimname:[i]}))
        config_ds = _minimal_config(tracking_outpath)
        config_ds['cloudid_filebase'] = 'cloudid_dsinput_'
        ds_in = xr.open_dataset(fpath)
        outfile_from_ds = idfeature_radar3d(ds_in.isel(time=[0]), config_ds)
        ds_out_ds = xr.open_dataset(outfile_from_ds, decode_times=False, mask_and_scale=False)

        return ds_out_name, ds_out_ds, config

    # ------------------------------------------------------------------
    def test_output_contract_dims_and_vars(self, outputs):
        ds_out, _, config = outputs
        assert set(['time', 'lat', 'lon', 'features']).issubset(set(ds_out.sizes))
        for v in ['feature_number', 'nfeatures', 'npix_feature', 'base_time',
                  'latitude', 'longitude', 'dbz_comp', 'convsf', 'conv_core',
                  'cell_number', 'echotop10', 'echotop20', 'echotop30',
                  'echotop40', 'echotop50']:
            assert v in ds_out, f"Missing variable '{v}' in output"
        assert ds_out['feature_number'].dims == ('time', 'lat', 'lon')
        assert 'units' in ds_out['base_time'].attrs

    def test_features_detected_and_sorted_by_size(self, outputs):
        ds_out, _, _ = outputs
        feature_number = ds_out['feature_number'].values.squeeze()
        nfeatures = int(ds_out['nfeatures'].values.squeeze())
        npix_feature = ds_out['npix_feature'].values

        assert nfeatures >= 2, "Expected at least 2 features (core+shield, and the isolated blob)"
        labels_present = np.unique(feature_number)
        labels_present = labels_present[labels_present > 0]
        assert list(labels_present) == list(range(1, nfeatures + 1)), \
            "Feature labels should be contiguous 1..N"
        assert len(npix_feature) == nfeatures
        # Sizes must be sorted largest to smallest
        assert all(npix_feature[i] >= npix_feature[i + 1] for i in range(len(npix_feature) - 1)), \
            f"npix_feature not sorted descending: {npix_feature}"
        # npix_feature must match actual pixel counts (bincount consistency)
        for lbl in labels_present:
            assert np.count_nonzero(feature_number == lbl) == npix_feature[lbl - 1]

        # The core+shield feature (larger) must be label 1 and cover the core pixel
        assert feature_number[17, 20] == 1
        # The isolated 5x5 blob must also be labeled (some feature > 0)
        assert feature_number[42, 42] > 0
        # The too-small single-pixel speck must NOT be labeled as a feature
        assert feature_number[SPECK_ROW, SPECK_COL] == 0
        # No feature pixel should sit in the NaN band
        assert np.all(feature_number[:, NAN_COLS] == 0)

    def test_no_feature_pixel_below_secondary_threshold(self, outputs):
        ds_out, _, config = outputs
        feature_number = ds_out['feature_number'].values.squeeze()
        dbz_comp = ds_out['dbz_comp'].values.squeeze()
        secondary_thresh = config['feature_dbz_threshs'][1]
        labeled = feature_number > 0
        # Every labeled pixel must have dbz_comp > secondary_thresh (10 dBZ),
        # and must not be NaN.
        assert np.all(dbz_comp[labeled] > secondary_thresh)
        assert not np.any(np.isnan(dbz_comp[labeled]))

    def test_convective_stratiform_and_cell_number(self, outputs):
        ds_out, _, _ = outputs
        convsf = ds_out['convsf'].values.squeeze()
        cell_number = ds_out['cell_number'].values.squeeze()

        # Core pixel should be classified CONVECTIVE (4)
        assert convsf[17, 20] == 4
        # Shield-only pixel away from the core should be STRATIFORM (3) or
        # at least not CONVECTIVE
        assert convsf[11, 11] != 4
        # cell_number should be labeled (>0) at the core, and distinct cells
        # should be positive, contiguous-ish integers
        assert cell_number[17, 20] > 0
        unique_cells = np.unique(cell_number)
        unique_cells = unique_cells[unique_cells > 0]
        assert len(unique_cells) >= 1

    def test_echotop_staircase_at_core(self, outputs):
        ds_out, _, _ = outputs
        et10 = ds_out['echotop10'].values.squeeze()
        et20 = ds_out['echotop20'].values.squeeze()
        et30 = ds_out['echotop30'].values.squeeze()
        et40 = ds_out['echotop40'].values.squeeze()
        et50 = ds_out['echotop50'].values.squeeze()

        r, c = 17, 20  # inside the core
        assert et10[r, c] >= et20[r, c] >= et30[r, c] >= et40[r, c], \
            f"Expected monotonically decreasing ETH: {et10[r,c]}, {et20[r,c]}, {et30[r,c]}, {et40[r,c]}"
        assert np.isnan(et50[r, c]), "ETH50 should be undefined (max reflectivity is 45 dBZ)"
        assert et40[r, c] > 0

    def test_filename_and_dataset_input_agree(self, outputs):
        ds_out_name, ds_out_ds, _ = outputs
        np.testing.assert_array_equal(
            ds_out_name['feature_number'].values, ds_out_ds['feature_number'].values
        )
        np.testing.assert_allclose(
            ds_out_name['dbz_comp'].values, ds_out_ds['dbz_comp'].values, equal_nan=True
        )
        np.testing.assert_array_equal(
            ds_out_name['convsf'].values, ds_out_ds['convsf'].values
        )


class TestIdfeatureRadar3dSynthetic2D:

    @pytest.fixture(scope='class')
    def output_2d(self, tmp_path_factory):
        tmpdir = tmp_path_factory.mktemp('radar2d_feature')
        tracking_outpath = str(tmpdir) + "/"
        ds = _make_synthetic_dataset(is_3d=False)
        fpath = os.path.join(tracking_outpath, 'input2d.nc')
        ds.to_netcdf(fpath)
        config = _minimal_config(tracking_outpath)
        outfile = idfeature_radar3d(fpath, config)
        return xr.open_dataset(outfile, decode_times=False, mask_and_scale=False)

    def test_2d_input_gives_nan_echotops(self, output_2d):
        ds_out = output_2d
        for v in ['echotop10', 'echotop20', 'echotop30', 'echotop40', 'echotop50']:
            assert np.all(np.isnan(ds_out[v].values)), f"{v} should be all-NaN for 2D input"
        # Features should still be identified from the 2D composite directly
        assert int(ds_out['nfeatures'].values.squeeze()) >= 2


class TestIdfeatureRadar3dAllNoEcho:

    def test_all_noecho_frame_still_writes_file_with_zero_features(self, tmp_path):
        tracking_outpath = str(tmp_path) + "/"
        refl = np.full((1, NZ, NY, NX), -99.0, dtype=np.float32)
        x_1d = np.arange(NX, dtype=np.float64) * DX
        y_1d = np.arange(NY, dtype=np.float64) * DX
        lon2d, lat2d = np.meshgrid(-100.0 + x_1d / 111000.0, 35.0 + y_1d / 111000.0)
        ds = xr.Dataset(
            {
                'Reflectivity': xr.DataArray(refl, dims=['time', 'z', 'y', 'x']),
                'lon': xr.DataArray(lon2d, dims=['y', 'x']),
                'lat': xr.DataArray(lat2d, dims=['y', 'x']),
            },
            coords={
                'time': xr.DataArray(
                    np.array(['2025-05-02T19:05:00'], dtype='datetime64[ns]'), dims=['time'],
                ),
                'x': xr.DataArray(x_1d, dims=['x']),
                'y': xr.DataArray(y_1d, dims=['y']),
                'z': xr.DataArray(HEIGHT_1D, dims=['z']),
            },
        )
        fpath = os.path.join(tracking_outpath, 'input_noecho.nc')
        ds.to_netcdf(fpath)
        config = _minimal_config(tracking_outpath)

        outfile = idfeature_radar3d(fpath, config)
        assert outfile is not None
        assert os.path.isfile(outfile)
        ds_out = xr.open_dataset(outfile, decode_times=False, mask_and_scale=False)
        assert int(ds_out['nfeatures'].values.squeeze()) == 0
        assert ds_out.sizes['features'] == 0
        assert np.all(ds_out['feature_number'].values == 0)


class TestIdfeatureRadar3dMissThresh:

    def test_frame_skipped_by_miss_thresh_returns_none(self, tmp_path):
        tracking_outpath = str(tmp_path) + "/"
        # Entirely NaN composite -> missing fraction = 1.0
        refl = np.full((1, NZ, NY, NX), np.nan, dtype=np.float32)
        x_1d = np.arange(NX, dtype=np.float64) * DX
        y_1d = np.arange(NY, dtype=np.float64) * DX
        lon2d, lat2d = np.meshgrid(-100.0 + x_1d / 111000.0, 35.0 + y_1d / 111000.0)
        ds = xr.Dataset(
            {
                'Reflectivity': xr.DataArray(refl, dims=['time', 'z', 'y', 'x']),
                'lon': xr.DataArray(lon2d, dims=['y', 'x']),
                'lat': xr.DataArray(lat2d, dims=['y', 'x']),
            },
            coords={
                'time': xr.DataArray(
                    np.array(['2025-05-02T19:10:00'], dtype='datetime64[ns]'), dims=['time'],
                ),
                'x': xr.DataArray(x_1d, dims=['x']),
                'y': xr.DataArray(y_1d, dims=['y']),
                'z': xr.DataArray(HEIGHT_1D, dims=['z']),
            },
        )
        fpath = os.path.join(tracking_outpath, 'input_allnan.nc')
        ds.to_netcdf(fpath)
        config = _minimal_config(tracking_outpath)
        config['miss_thresh'] = 0.5

        outfile = idfeature_radar3d(fpath, config)
        assert outfile is None


class TestIdfeatureRadar3dConvFeatureNumber:
    """conv_feature_number: the Steiner convective pixels dilated by conv_dilate_radius_km, then labeled."""

    @staticmethod
    def _run(tmp_path, radius_km=None):
        tracking_outpath = str(tmp_path) + "/"
        fpath = os.path.join(tracking_outpath, 'input.nc')
        _make_synthetic_dataset(is_3d=True).to_netcdf(fpath)
        config = _minimal_config(tracking_outpath)
        if radius_km is not None:
            config['conv_dilate_radius_km'] = radius_km
        outfile = idfeature_radar3d(fpath, config)
        return xr.open_dataset(outfile, decode_times=False, mask_and_scale=False)

    def test_default_labels_the_8_connected_convective_regions(self, tmp_path):
        ds = self._run(tmp_path)
        cfn = ds['conv_feature_number'].values.squeeze()
        conv = ds['convsf'].values.squeeze() == 4
        assert ds['conv_feature_number'].dims == ('time', 'lat', 'lon')
        assert conv.any()
        expected, n = ndimage.label(conv, structure=np.ones((3, 3)))
        assert np.array_equal(cfn > 0, conv)          # no radius: nothing is added
        assert cfn.max() == n
        # the same partition as the plain 8-connected labeling (only the numbers differ)
        assert len(set(zip(cfn[conv].tolist(), expected[conv].tolist()))) == n
        assert ds['conv_feature_number'].attrs['dilation_radius_km'] == 0.0
        assert ds.attrs['conv_dilate_radius_km'] == 0

    def test_radius_extends_the_features_by_a_disc_around_the_convective_pixels(self, tmp_path):
        ds = self._run(tmp_path, radius_km=12.0)
        cfn = ds['conv_feature_number'].values.squeeze()
        conv = ds['convsf'].values.squeeze() == 4
        assert np.all(cfn[conv] > 0)                  # every convective pixel is in a feature
        assert (cfn > 0).sum() > conv.sum()           # and the features extend beyond them
        dist = ndimage.distance_transform_edt(~conv, sampling=(DX, DX))
        assert dist[cfn > 0].max() <= 12e3 + 1e-6     # nothing farther than the radius
        assert np.all(cfn[dist <= 12e3] > 0)          # and everything within the radius is in
        assert ds['conv_feature_number'].attrs['dilation_radius_km'] == 12.0
        assert ds.attrs['conv_dilate_radius_km'] == 12.0
