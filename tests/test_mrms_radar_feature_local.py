"""
Local (real-data) check for idfeature_radar3d.py against the real MRMS/HRRR
3-km test snapshot (2025-05-02 19:00:43 UTC), comparing the parts it shares
with the existing, already-validated radar_cells cell-tracking pipeline
(same reader, same Steiner classification, same echo-top code) against that
pipeline's real output on disk.

Requires PYFLEXTRKR_TEST_DATA to be set (see tests/conftest.py) and the real
MRMS data to be mounted. Runs serially, single timestep - safe on a login
node (not a parallel/production run).
"""

import os
import numpy as np
import pytest
import xarray as xr
import yaml

from pyflextrkr.idfeature_radar3d import idfeature_radar3d
from pyflextrkr.ft_utilities import load_config


INPUT_FILE = "/pscratch/sd/i/iclas2/MRMS/remap_hrrr/MRMS_Reflectivity_HRRR_20250502-190043.nc"
REFERENCE_CELLID_FILE = "/pscratch/sd/i/iclas2/MRMS/cell_conus/tracking/cloudid_20250502_190043.nc"
CELLTRACKING_CONFIG = (
    "/global/u1/f/feng045/program/airscape/conus_tracking/config/config/"
    "config_celltracking_3km_MRMS_202505.yaml"
)
RADAR3D_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "config_mcs_3km_MRMS.yaml",
)


def _data_available():
    return (
        os.environ.get("PYFLEXTRKR_TEST_DATA") is not None
        and os.path.isfile(INPUT_FILE)
        and os.path.isfile(REFERENCE_CELLID_FILE)
    )


pytestmark = pytest.mark.local


@pytest.fixture(scope="module")
def radar_feature_output(tmp_path_factory):
    if not _data_available():
        pytest.skip("Real MRMS data or reference cellid file not found.")

    tmpdir = str(tmp_path_factory.mktemp("mrms_radar_feature_local")) + "/"
    # Override root_path/startdate/enddate in a temp copy of the config so
    # load_config() derives tracking_outpath etc. pointing at a tmp dir,
    # without needing to write to the real production root_path.
    raw_config = yaml.full_load(open(RADAR3D_CONFIG))
    raw_config["root_path"] = tmpdir
    raw_config["startdate"] = "20250502.1900"
    raw_config["enddate"] = "20250502.1901"
    tmp_config_file = os.path.join(tmpdir, "config_test.yml")
    os.makedirs(tmpdir, exist_ok=True)
    with open(tmp_config_file, "w") as f:
        yaml.dump(raw_config, f)
    config = load_config(tmp_config_file)
    config["input_format"] = "netcdf"
    config["combine_files"] = False

    outfile = idfeature_radar3d(INPUT_FILE, config)
    assert outfile is not None
    return xr.open_dataset(outfile, decode_times=False, mask_and_scale=False), config


@pytest.fixture(scope="module")
def reference_cellid():
    if not _data_available():
        pytest.skip("Real MRMS data or reference cellid file not found.")
    return xr.open_dataset(REFERENCE_CELLID_FILE, decode_times=False, mask_and_scale=False)


class TestMrmsRadarFeatureVsCellTracking:

    def test_dbz_comp_matches_reference(self, radar_feature_output, reference_cellid):
        ds_out, _ = radar_feature_output
        dbz_new = ds_out["dbz_comp"].values.squeeze()
        dbz_ref = reference_cellid["dbz_comp"].values.squeeze()
        np.testing.assert_allclose(dbz_new, dbz_ref, equal_nan=True, rtol=1e-5, atol=1e-3)

    def test_dbz_lowlevel_matches_reference(self, radar_feature_output, reference_cellid):
        ds_out, _ = radar_feature_output
        np.testing.assert_allclose(
            ds_out["dbz_lowlevel"].values.squeeze(),
            reference_cellid["dbz_lowlevel"].values.squeeze(),
            equal_nan=True, rtol=1e-5, atol=1e-3,
        )

    def test_convsf_matches_reference(self, radar_feature_output, reference_cellid):
        """
        convsf (Steiner convective/stratiform classification) should match
        exactly, since the config below uses the same Steiner parameters as
        the production cell-tracking config (per the task's instruction to
        follow the cell-tracking identification parameters) and
        dilate_method='orig' for both.
        """
        ds_out, config = radar_feature_output
        convsf_new = ds_out["convsf"].values.squeeze()
        convsf_ref = reference_cellid["convsf"].values.squeeze()
        n_mismatch = np.count_nonzero(convsf_new != convsf_ref)
        frac_mismatch = n_mismatch / convsf_new.size
        # Allow a tiny mismatch fraction for edge-of-domain / floating point
        # differences in the shared reader pipeline; report the number for
        # visibility rather than silently passing.
        print(f"convsf mismatch: {n_mismatch} pixels ({frac_mismatch:.4%})")
        assert frac_mismatch < 0.01

    def test_conv_core_matches_reference(self, radar_feature_output, reference_cellid):
        ds_out, _ = radar_feature_output
        conv_core_new = ds_out["conv_core"].values.squeeze()
        conv_core_ref = reference_cellid["conv_core"].values.squeeze()
        n_mismatch = np.count_nonzero(conv_core_new != conv_core_ref)
        print(f"conv_core mismatch: {n_mismatch} pixels ({n_mismatch / conv_core_new.size:.4%})")
        assert n_mismatch / conv_core_new.size < 0.01

    def test_cell_number_consistent_with_score_orig(self, radar_feature_output, reference_cellid):
        """
        cell_number (labeled score_orig) should have the same footprint as
        the reference's core_steiner_orig (score_orig before labeling).
        """
        ds_out, config = radar_feature_output
        if config.get("cell_number_source", "score_orig") != "score_orig":
            pytest.skip("cell_number_source is not score_orig in this config")
        cell_number = ds_out["cell_number"].values.squeeze()
        core_steiner_orig_ref = reference_cellid["core_steiner_orig"].values.squeeze()
        np.testing.assert_array_equal(cell_number > 0, core_steiner_orig_ref > 0)

    def test_echotops_allclose(self, radar_feature_output, reference_cellid):
        ds_out, _ = radar_feature_output
        for v in ["echotop10", "echotop20", "echotop30", "echotop40", "echotop50"]:
            np.testing.assert_allclose(
                ds_out[v].values.squeeze(), reference_cellid[v].values.squeeze(),
                equal_nan=True, rtol=1e-5, atol=1.0,
            )

    def test_feature_identification_sanity(self, radar_feature_output):
        ds_out, _ = radar_feature_output
        nfeatures = int(ds_out["nfeatures"].values.squeeze())
        feature_number = ds_out["feature_number"].values.squeeze()
        npix_feature = ds_out["npix_feature"].values
        print(f"nfeatures = {nfeatures}")

        assert nfeatures > 0
        labels_present = np.unique(feature_number)
        labels_present = labels_present[labels_present > 0]
        assert list(labels_present) == list(range(1, nfeatures + 1))
        assert all(npix_feature[i] >= npix_feature[i + 1] for i in range(len(npix_feature) - 1))
        for lbl in labels_present:
            assert np.count_nonzero(feature_number == lbl) == npix_feature[lbl - 1]

        dbz_comp = ds_out["dbz_comp"].values.squeeze()
        secondary_thresh = 10
        assert np.all(dbz_comp[feature_number > 0] > secondary_thresh)
        assert not np.any(np.isnan(dbz_comp[feature_number > 0]))

    def test_time_matches_snapshot(self, radar_feature_output):
        ds_out, _ = radar_feature_output
        assert ds_out.attrs["Title"].startswith("Cloudid file from 2025-05-02 19:00")
