"""
Tests for mapfeature_func.py's new feature_type=='radar_feature' branch:
the dbztracknumber pixel map, and the optional feature_class pixel map
(painted from identifymcs_radar.py's per-track classification, with
merge/split fragment pixels taking their parent track's class).

Builds a synthetic cloudid file directly via netcdf_io.write_radar_featureid
and calls map_feature directly (no dask), so this runs on CI.
"""

import os
import numpy as np
import pytest
import xarray as xr

from pyflextrkr.mapfeature_func import map_feature
from pyflextrkr.netcdf_io import write_radar_featureid
from pyflextrkr.steiner_func import dilate_label_conv_mask
from pyflextrkr.identifymcs_radar import CLASS_MCS, CLASS_ISOLATED_CELL


NY, NX = 30, 30
DX = 3000.0

# Feature 1 (the main MCS track): rows 5-19, cols 5-24
F1_ROWS = slice(5, 20)
F1_COLS = slice(5, 25)
# Feature 2 (a small fragment that merges into feature 1's track): rows 22-24, cols 22-24
F2_ROWS = slice(22, 25)
F2_COLS = slice(22, 25)


def _build_cloudid(tmp_path, conv_radius_km=None):
    """conv_radius_km: write conv_feature_number at that dilation radius, with a second convective block 5 px (15 km)
    from the first (rows 6-9, cols 14-17); None: no conv_feature_number and one block, as the cloudid files of
    the other tests."""
    feature_number = np.zeros((NY, NX), dtype=np.int32)
    feature_number[F1_ROWS, F1_COLS] = 1
    feature_number[F2_ROWS, F2_COLS] = 2

    dbz_comp = np.full((NY, NX), np.nan, dtype=np.float32)
    dbz_comp[F1_ROWS, F1_COLS] = 20.0
    # Make part of feature 1 exceed the dbz_thresh (10) and part not
    dbz_comp[6:10, 6:10] = 30.0
    dbz_comp[F2_ROWS, F2_COLS] = 15.0

    convsf = np.zeros((NY, NX), dtype=np.int32)
    convsf[6:10, 6:10] = 4
    if conv_radius_km is not None:
        convsf[6:10, 14:18] = 4
    cell_number = np.zeros((NY, NX), dtype=np.int32)
    cell_number[6:10, 6:10] = 1
    feature_pixel_type = np.where(feature_number > 0, 1, 0).astype(np.int32)
    echotop = np.full((NY, NX), np.nan, dtype=np.float32)

    npix_feature = np.array([
        np.count_nonzero(feature_number == 1),
        np.count_nonzero(feature_number == 2),
    ], dtype=np.int32)

    x_coords = np.arange(NX, dtype=np.float64) * DX
    y_coords = np.arange(NY, dtype=np.float64) * DX
    lon2d, lat2d = np.meshgrid(-100.0 + x_coords / 111000.0, 35.0 + y_coords / 111000.0)
    out_lon = xr.DataArray(lon2d, dims=['lat', 'lon'])
    out_lat = xr.DataArray(lat2d, dims=['lat', 'lon'])
    dbz_comp_da = xr.DataArray(dbz_comp, dims=['lat', 'lon'])

    file_basetime = np.array([1746212443.0])
    file_datestring, file_timestring = "20250502", "190043"
    config = {
        'feature_varname': 'feature_number',
        'nfeature_varname': 'nfeatures',
        'featuresize_varname': 'npix_feature',
    }
    conv_kwargs = {}
    if conv_radius_km is not None:
        config['conv_dilate_radius_km'] = conv_radius_km
        conv_kwargs = {
            'conv_feature_number': dilate_label_conv_mask(convsf == 4, conv_radius_km * 1000.0, DX, DX),
            'feature_params': {'conv_dilate_radius_km': conv_radius_km},
        }
    cloudid_outfile = os.path.join(str(tmp_path), f"cloudid_{file_datestring}_{file_timestring}.nc")
    write_radar_featureid(
        cloudid_outfile, file_basetime, file_datestring, file_timestring,
        out_lon, out_lat, x_coords, y_coords,
        dbz_comp_da, dbz_comp_da.copy(),
        convsf, conv_core=(convsf == 4).astype(np.int32),
        cell_number=cell_number, feature_pixel_type=feature_pixel_type,
        echotop10=echotop, echotop20=echotop, echotop30=echotop,
        echotop40=echotop, echotop50=echotop,
        feature_number=feature_number, npix_feature=npix_feature, nfeatures=2,
        reflectivity_file="synthetic_test.nc", config=config, **conv_kwargs,
    )
    # map_feature's own filebasetime param is always an int in the real
    # pipeline (from subset_files_timerange's integer-typed files_basetime,
    # derived from filenames); return that form here rather than the float
    # file_basetime used above for write_radar_featureid.
    return cloudid_outfile, np.int64(file_basetime[0])


class TestMapFeatureRadarFeature:

    def test_dbztracknumber_and_feature_class(self, tmp_path):
        cloudid_outfile, filebasetime = _build_cloudid(tmp_path)
        pixeltracking_outpath = str(tmp_path) + "/"
        os.makedirs(pixeltracking_outpath, exist_ok=True)

        # Track 0 matched to cloudnumber 1 (feature 1) at this time; feature 2
        # (cloudnumber 2) merges into track 0.
        file_trackindex = np.array([0])
        file_cloudnumber = np.array([1])
        file_trackstatus = np.array([1])
        file_mergetracknumber = np.array([0])  # no track-level merge info needed for this check
        file_splittracknumber = np.array([0])
        file_mergecloudnumber = np.array([[2]])  # feature 2 merges into track 0
        file_splitcloudnumber = np.array([[0]])
        file_featureclass = np.array([CLASS_MCS])

        config = {
            'feature_varname': 'feature_number',
            'feature_type': 'radar_feature',
            'dbz_thresh': 10,
            'ref_varname': 'dbz_comp',
            'fillval': -9999,
        }

        outfile = map_feature(
            cloudid_outfile, filebasetime, file_trackindex, file_cloudnumber,
            file_trackstatus, file_mergetracknumber, file_splittracknumber,
            file_mergecloudnumber, file_splitcloudnumber,
            "status explanation", config, pixeltracking_outpath, "mcstrack_",
            file_featureclass=file_featureclass,
        )
        assert os.path.isfile(outfile)
        ds = xr.open_dataset(outfile, decode_times=False, mask_and_scale=False)

        assert "dbztracknumber" in ds
        assert "feature_class" in ds
        dbztracknumber = ds["dbztracknumber"].values.squeeze()
        feature_class = ds["feature_class"].values.squeeze()
        cloudtracknumber = ds["cloudtracknumber"].values.squeeze()

        # dbztracknumber should be nonzero only where dbz_comp > 10 AND the
        # pixel belongs to a tracked (incl. merge/split) cloud - i.e. the
        # small 30-dBZ sub-block within feature 1, plus feature 2's merged-in
        # pixels (15 dBZ > 10).
        assert np.all(dbztracknumber[6:10, 6:10] == 1)
        assert np.all(dbztracknumber[F2_ROWS, F2_COLS] == 1)
        # The rest of feature 1 (20 dBZ < threshold... wait 20>10, check logic)
        # Feature 1's base is 20 dBZ, which IS > 10 dBZ threshold too, so the
        # whole of cloudtracknumber's footprint should get dbztracknumber=1
        # wherever dbz_comp > 10 - confirm this holds for the base of feature 1 too.
        assert dbztracknumber[10, 10] == 1  # within feature 1's base (20 dBZ), not in the 30dBZ sub-block

        # feature_class: track 0's class (MCS) should be painted across all of
        # cloudtracknumber's footprint, INCLUDING the merged-in feature 2
        # fragment pixels (parent's class, not its own).
        assert np.all(feature_class[cloudtracknumber == 1] == CLASS_MCS)
        assert np.all(feature_class[F2_ROWS, F2_COLS] == CLASS_MCS)
        # Pixels outside any tracked cloud should stay at fillval
        assert feature_class[0, 0] == -9999

    def test_without_featureclass_arg_omits_output_var(self, tmp_path):
        """file_featureclass defaults to None - feature_class must not appear."""
        cloudid_outfile, filebasetime = _build_cloudid(tmp_path)
        pixeltracking_outpath = str(tmp_path) + "/"
        os.makedirs(pixeltracking_outpath, exist_ok=True)

        file_trackindex = np.array([0])
        file_cloudnumber = np.array([1])
        file_trackstatus = np.array([1])
        z = np.array([0])
        mz = np.array([[0]])

        config = {
            'feature_varname': 'feature_number',
            'feature_type': 'radar_feature',
            'dbz_thresh': 10,
            'ref_varname': 'dbz_comp',
            'fillval': -9999,
        }
        outfile = map_feature(
            cloudid_outfile, filebasetime, file_trackindex, file_cloudnumber,
            file_trackstatus, z, z, mz, mz,
            "status explanation", config, pixeltracking_outpath, "mcstrack_",
        )
        ds = xr.open_dataset(outfile, decode_times=False, mask_and_scale=False)
        assert "dbztracknumber" in ds
        assert "feature_class" not in ds


class TestConvDilateOverwrite:
    """conv_dilate_overwrite_radius_km: the pixel files get the convective groups that regionprops_driver used for the
    statistics (recomputed from convsf), not the groups of the cloudid file."""

    @staticmethod
    def _map(tmp_path, conv_radius_km, **config_extra):
        cloudid_outfile, filebasetime = _build_cloudid(tmp_path, conv_radius_km)
        config = {
            'feature_varname': 'feature_number',
            'feature_type': 'radar_feature',
            'dbz_thresh': 10,
            'ref_varname': 'dbz_comp',
            'fillval': -9999,
            **config_extra,
        }
        z = np.array([0])
        outfile = map_feature(
            cloudid_outfile, filebasetime, np.array([0]), np.array([1]), np.array([1]), z, z,
            np.array([[2]]), np.array([[0]]), "status explanation", config, str(tmp_path) + "/", "mcstrack_",
        )
        return xr.open_dataset(outfile, decode_times=False, mask_and_scale=False), cloudid_outfile

    def test_groups_are_recomputed_at_the_overwrite_radius(self, tmp_path):
        ds, cloudid = self._map(tmp_path, 5.0, dx=DX, dy=DX, conv_dilate_overwrite_radius_km=9)
        groups = ds["conv_feature_number"].values.squeeze()
        np.testing.assert_array_equal(groups, dilate_label_conv_mask(ds["convsf"].values.squeeze() == 4, 9000.0, DX, DX))
        assert groups[7, 7] == groups[7, 16] > 0              # the two blocks, 15 km apart, are one group at 9 km
        with xr.open_dataset(cloudid, decode_times=False, mask_and_scale=False) as own:
            own_groups = own["conv_feature_number"].values.squeeze()
        assert own_groups[7, 7] != own_groups[7, 16]           # two groups at the 5 km of the cloudid file
        assert ds["conv_feature_number"].dims == ("time", "lat", "lon")
        assert ds["conv_feature_number"].attrs["dilation_radius_km"] == 9.0
        assert ds["conv_feature_number"].attrs["units"] == "unitless"
        assert "conv_dilate_overwrite_radius_km" in ds["conv_feature_number"].attrs["comments"]
        assert ds.attrs["conv_dilate_radius_km"] == 9.0
        assert ds.attrs["conv_dilate_radius_km_idfeature"] == 5.0

    def test_without_the_option_the_groups_of_the_cloudid_file_are_copied(self, tmp_path):
        for extra in ({}, {"conv_dilate_overwrite_radius_km": None, "dx": DX, "dy": DX}):
            ds, cloudid = self._map(tmp_path, 5.0, **extra)
            with xr.open_dataset(cloudid, decode_times=False, mask_and_scale=False) as own:
                np.testing.assert_array_equal(ds["conv_feature_number"].values, own["conv_feature_number"].values)
            assert ds["conv_feature_number"].attrs["dilation_radius_km"] == 5.0
            assert ds.attrs["conv_dilate_radius_km"] == 5.0
            assert "conv_dilate_radius_km_idfeature" not in ds.attrs

    def test_the_variable_is_added_when_the_cloudid_file_has_none(self, tmp_path):
        ds, _ = self._map(tmp_path, None, dx=DX, dy=DX, conv_dilate_overwrite_radius_km=9)
        groups = ds["conv_feature_number"].values.squeeze()
        np.testing.assert_array_equal(groups, dilate_label_conv_mask(ds["convsf"].values.squeeze() == 4, 9000.0, DX, DX))
        assert groups.max() == 1
        assert ds["conv_feature_number"].attrs["dilation_radius_km"] == 9.0
        assert ds.attrs["conv_dilate_radius_km"] == 9.0
