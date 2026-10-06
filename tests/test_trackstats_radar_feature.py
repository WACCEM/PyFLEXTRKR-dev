"""
Synthetic test for trackstats_func.py's new feature_type='radar_feature'
branch (calc_stats_singlefile), added for the radar3D MCS workflow.

Builds a synthetic cloudid file directly via netcdf_io.write_radar_featureid
(the exact writer idfeature_radar3d.py uses), containing:
  - Feature 1: a long, thin, axis-aligned rectangle (so majoraxis/minoraxis/
    orientation have simple, predictable values) with 2 separate convective
    cells (cell_number 1 and 2) and a surrounding stratiform region.
  - Feature 2: a single isolated pixel (no convective cells).
  - Feature 3: an entirely stratiform feature (zero convective pixels).

Then calls calc_stats_singlefile directly (bypassing the dask-based driver,
so this runs on CI with no distributed dependency) and checks the computed
statistics against hand-derived expected values.
"""

import os
import numpy as np
import pytest
import xarray as xr
from netCDF4 import stringtochar

from pyflextrkr.trackstats_func import calc_stats_singlefile
from pyflextrkr.netcdf_io import write_radar_featureid


NY, NX = 60, 80
DX = 3000.0  # m

# Feature 1: long horizontal rectangle, rows 25-29 (5 rows), cols 5-64 (60 cols)
F1_ROWS = slice(25, 30)
F1_COLS = slice(5, 65)
# Convective blob A (cell 1) within feature 1
CELL1_ROWS = slice(26, 29)
CELL1_COLS = slice(15, 20)
# Convective blob B (cell 2) within feature 1
CELL2_ROWS = slice(26, 29)
CELL2_COLS = slice(45, 50)
# Feature 2: single isolated pixel
F2_ROW, F2_COL = 5, 5
# Feature 3: entirely stratiform block
F3_ROWS = slice(45, 51)
F3_COLS = slice(10, 21)


def _build_synthetic_cloudid(tmp_path):
    feature_number = np.zeros((NY, NX), dtype=np.int32)
    convsf = np.zeros((NY, NX), dtype=np.int32)  # 1=NO_SURF_ECHO,2=WEAK_ECHO,3=STRATIFORM,4=CONVECTIVE
    cell_number = np.zeros((NY, NX), dtype=np.int32)
    dbz_comp = np.full((NY, NX), np.nan, dtype=np.float32)
    echotop10 = np.full((NY, NX), np.nan, dtype=np.float32)
    echotop20 = np.full((NY, NX), np.nan, dtype=np.float32)
    echotop30 = np.full((NY, NX), np.nan, dtype=np.float32)
    echotop40 = np.full((NY, NX), np.nan, dtype=np.float32)
    echotop50 = np.full((NY, NX), np.nan, dtype=np.float32)
    feature_pixel_type = np.zeros((NY, NX), dtype=np.int32)

    # Feature 1: stratiform rectangle base
    feature_number[F1_ROWS, F1_COLS] = 1
    convsf[F1_ROWS, F1_COLS] = 3  # STRATIFORM
    dbz_comp[F1_ROWS, F1_COLS] = 20.0
    echotop10[F1_ROWS, F1_COLS] = 2000.0

    # Convective cells within feature 1
    convsf[CELL1_ROWS, CELL1_COLS] = 4  # CONVECTIVE
    cell_number[CELL1_ROWS, CELL1_COLS] = 1
    dbz_comp[CELL1_ROWS, CELL1_COLS] = 50.0
    echotop10[CELL1_ROWS, CELL1_COLS] = 10000.0
    echotop20[CELL1_ROWS, CELL1_COLS] = 8000.0
    echotop30[CELL1_ROWS, CELL1_COLS] = 6000.0
    echotop40[CELL1_ROWS, CELL1_COLS] = 4000.0
    echotop50[CELL1_ROWS, CELL1_COLS] = 2000.0

    convsf[CELL2_ROWS, CELL2_COLS] = 4
    cell_number[CELL2_ROWS, CELL2_COLS] = 2
    dbz_comp[CELL2_ROWS, CELL2_COLS] = 46.0
    echotop10[CELL2_ROWS, CELL2_COLS] = 9000.0
    echotop20[CELL2_ROWS, CELL2_COLS] = 7000.0
    echotop30[CELL2_ROWS, CELL2_COLS] = 5000.0
    echotop40[CELL2_ROWS, CELL2_COLS] = 3000.0
    # No echotop50 for cell 2 (max reflectivity 46 < 50 dBZ) -> stays NaN

    # Feature 2: single isolated pixel, no convective cells
    feature_number[F2_ROW, F2_COL] = 2
    convsf[F2_ROW, F2_COL] = 1  # NO_SURF_ECHO (edge-of-feature classification)
    dbz_comp[F2_ROW, F2_COL] = 12.0

    # Feature 3: entirely stratiform, no convective cells
    feature_number[F3_ROWS, F3_COLS] = 3
    convsf[F3_ROWS, F3_COLS] = 3
    dbz_comp[F3_ROWS, F3_COLS] = 18.0
    echotop10[F3_ROWS, F3_COLS] = 1500.0

    feature_pixel_type[feature_number > 0] = 1
    npix_feature = np.array([
        np.count_nonzero(feature_number == 1),
        np.count_nonzero(feature_number == 2),
        np.count_nonzero(feature_number == 3),
    ], dtype=np.int32)
    nfeatures = 3

    x_coords = (np.arange(NX, dtype=np.float64)) * DX
    y_coords = (np.arange(NY, dtype=np.float64)) * DX
    lon2d, lat2d = np.meshgrid(-100.0 + x_coords / 111000.0, 35.0 + y_coords / 111000.0)
    out_lon = xr.DataArray(lon2d.astype(np.float64), dims=['lat', 'lon'])
    out_lat = xr.DataArray(lat2d.astype(np.float64), dims=['lat', 'lon'])
    dbz_comp_da = xr.DataArray(dbz_comp, dims=['lat', 'lon'])
    dbz_lowlevel_da = xr.DataArray(dbz_comp.copy(), dims=['lat', 'lon'])

    file_basetime = np.array([1746212443.0])  # 2025-05-02 19:00:43 UTC (arbitrary for this test)
    file_datestring = "20250502"
    file_timestring = "190043"

    config = {
        'feature_varname': 'feature_number',
        'nfeature_varname': 'nfeatures',
        'featuresize_varname': 'npix_feature',
    }

    cloudid_outfile = os.path.join(str(tmp_path), f"cloudid_{file_datestring}_{file_timestring}.nc")
    write_radar_featureid(
        cloudid_outfile, file_basetime, file_datestring, file_timestring,
        out_lon, out_lat, x_coords, y_coords,
        dbz_comp_da, dbz_lowlevel_da,
        convsf, conv_core=(convsf == 4).astype(np.int32),
        cell_number=cell_number, feature_pixel_type=feature_pixel_type,
        echotop10=echotop10, echotop20=echotop20, echotop30=echotop30,
        echotop40=echotop40, echotop50=echotop50,
        feature_number=feature_number, npix_feature=npix_feature, nfeatures=nfeatures,
        reflectivity_file="synthetic_test.nc", config=config,
    )
    return cloudid_outfile, os.path.basename(cloudid_outfile)


@pytest.fixture(scope='module')
def stats_result(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp('trackstats_radar_feature')
    cloudid_outfile, cloudid_fname = _build_synthetic_cloudid(tmpdir)

    nfeatures = 3
    # Each feature is its own track: tracknumbers[cloudnumber-1] = track number
    tracknumbers = np.array([1, 2, 3], dtype=np.int32)
    trackstatus = np.array([1, 1, 1], dtype=np.int32)
    trackmerge = np.zeros(nfeatures, dtype=np.int32)
    tracksplit = np.zeros(nfeatures, dtype=np.int32)
    trackreset = np.zeros(nfeatures, dtype=np.int32)
    cloudidfile = stringtochar(np.array([cloudid_fname], dtype='S'))[0]

    config = {
        'tracking_outpath': str(tmpdir) + "/",
        'pixel_radius': DX / 1000.0,
        'area_method': 'fixed',
        'feature_type': 'radar_feature',
        'terrain_file': None,
        'rangemask_varname': None,
        'feature_varname': 'feature_number',
        'pbc_direction': 'none',
        'ref_varname': 'dbz_comp',
        'trackstats_dbz_area_threshs': [45, 50],
    }

    out_dict, out_dict_attrs = calc_stats_singlefile(
        tracknumbers, cloudidfile, trackstatus, trackmerge, tracksplit, trackreset, config,
    )
    return out_dict, out_dict_attrs


class TestTrackstatsRadarFeature:

    def test_basic_vars_present(self, stats_result):
        out_dict, out_dict_attrs = stats_result
        expected_vars = [
            "mean_x", "mean_y", "lat_weightedcentroid", "lon_weightedcentroid",
            "x_weightedcentroid", "y_weightedcentroid", "max_dbz",
            "majoraxis", "minoraxis", "aspectratio", "orientation", "eccentricity",
            "conv_area", "strat_area", "ncells",
            "maxETH_10dbz", "maxETH_20dbz", "maxETH_30dbz", "maxETH_40dbz", "maxETH_50dbz",
            "area_45dbz", "area_50dbz", "conv_area_45dbz", "conv_area_50dbz",
            "area", "meanlat", "meanlon",
        ]
        for v in expected_vars:
            assert v in out_dict, f"Missing variable '{v}'"
            assert v in out_dict_attrs, f"Missing attrs for '{v}'"

    def test_feature1_shape_stats(self, stats_result):
        out_dict, _ = stats_result
        # Feature 1 (track index 0): long rectangle, 5 rows x 60 cols @ 3km
        # Major axis should be along x (columns, ~180km); minor along y (~15km)
        majoraxis = out_dict["majoraxis"][0]
        minoraxis = out_dict["minoraxis"][0]
        aspectratio = out_dict["aspectratio"][0]
        orientation = out_dict["orientation"][0]
        print(f"Feature 1: majoraxis={majoraxis:.1f} km, minoraxis={minoraxis:.1f} km, "
              f"aspectratio={aspectratio:.2f}, orientation={orientation:.1f} deg")
        assert majoraxis > minoraxis
        assert 100 < majoraxis < 250
        assert 5 < minoraxis < 30
        assert aspectratio > 5
        # Long axis is horizontal (along columns) -> orientation near +/-90 deg
        # (skimage orientation is angle between row-axis and major axis)
        assert abs(abs(orientation) - 90) < 15

    def test_feature1_ncells_and_areas(self, stats_result):
        out_dict, _ = stats_result
        ncells = out_dict["ncells"][0]
        conv_area = out_dict["conv_area"][0]
        strat_area = out_dict["strat_area"][0]
        area = out_dict["area"][0]
        print(f"Feature 1: ncells={ncells}, conv_area={conv_area}, strat_area={strat_area}, area={area}")

        assert ncells == 2
        assert conv_area > 0
        assert strat_area > 0
        # Convective + stratiform area must not exceed total feature area
        assert conv_area + strat_area <= area + 1e-6

    def test_feature1_dbz_area_thresholds(self, stats_result):
        out_dict, _ = stats_result
        area_45 = out_dict["area_45dbz"][0]
        area_50 = out_dict["area_50dbz"][0]
        conv_area_45 = out_dict["conv_area_45dbz"][0]
        conv_area_50 = out_dict["conv_area_50dbz"][0]
        print(f"area_45dbz={area_45}, area_50dbz={area_50}, "
              f"conv_area_45dbz={conv_area_45}, conv_area_50dbz={conv_area_50}")

        # Cell 1 (50 dBZ) and cell 2 (46 dBZ) both exceed 45 dBZ -> area_45dbz > 0
        assert area_45 > 0
        # Only cell 1 (50 dBZ) exceeds 50 dBZ strictly (dbz > 50 is False for ==50)
        assert area_50 == 0
        assert conv_area_45 > 0
        assert conv_area_45 <= area_45 + 1e-6

    def test_feature1_max_dbz_and_eth(self, stats_result):
        out_dict, _ = stats_result
        assert out_dict["max_dbz"][0] == pytest.approx(50.0, abs=0.1)
        # ETH is restricted to convective pixels; cell 1 has the highest ETH values
        assert out_dict["maxETH_10dbz"][0] == pytest.approx(10.0, abs=0.01)  # km
        assert out_dict["maxETH_50dbz"][0] == pytest.approx(2.0, abs=0.01)  # km, only cell 1 has ETH50

    def test_feature1_weighted_centroid_shifted_toward_cell1(self, stats_result):
        """
        Cell 1 (50 dBZ, cols 15-19) is more intense than cell 2 (46 dBZ,
        cols 45-49); the reflectivity-weighted centroid should sit closer to
        cell 1 than the simple (unweighted) mean position would.
        """
        out_dict, _ = stats_result
        mean_x = out_dict["mean_x"][0]
        x_weightedcentroid = out_dict["x_weightedcentroid"][0]
        # Unweighted mean_x should be near the rectangle's horizontal center
        # (cols 5-64 -> center around col 34.5); weighted centroid should be
        # pulled toward cell 1 (cols 15-19, left of center) more than toward
        # cell 2 (cols 45-49, right of center) since cell 1 is more intense.
        print(f"mean_x={mean_x}, x_weightedcentroid={x_weightedcentroid}")
        assert x_weightedcentroid < mean_x

    def test_feature2_single_pixel(self, stats_result):
        out_dict, _ = stats_result
        # Feature 2 (track index 1): single isolated pixel, no convective cells
        assert out_dict["ncells"][1] == 0
        assert out_dict["conv_area"][1] == 0
        # A single-pixel region's minor axis is 0 -> aspectratio undefined (NaN)
        assert np.isnan(out_dict["aspectratio"][1]) or out_dict["minoraxis"][1] == 0

    def test_feature3_entirely_stratiform(self, stats_result):
        out_dict, _ = stats_result
        # Feature 3 (track index 2): entirely stratiform, zero convective pixels
        assert out_dict["ncells"][2] == 0
        assert out_dict["conv_area"][2] == 0
        assert out_dict["strat_area"][2] > 0
        # No convective pixels -> echo-top stats should remain fill (NaN)
        assert np.isnan(out_dict["maxETH_10dbz"][2])

    def test_ncells_is_int_dtype_in_output(self, stats_result):
        out_dict, _ = stats_result
        assert np.issubdtype(out_dict["ncells"].dtype, np.integer)
