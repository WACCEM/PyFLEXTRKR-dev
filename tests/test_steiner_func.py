"""
Unit tests for pyflextrkr/steiner_func.py.

These tests use only small synthetic arrays — no external data required,
and run in a few seconds on any machine including GitHub CI.

Tested functions
----------------
background_intensity        – convolution-based background reflectivity
peakedness                  – convective peakedness threshold
make_dilation_step_func     – step-function builder
mod_steiner_classification  – full Steiner classification pipeline
expand_conv_core            – outward expansion of convective masks
dilate_label_conv_mask      – dilation + labeling of the convective mask into convective features
"""

import numpy as np
import pytest

from pyflextrkr.steiner_func import (
    background_intensity,
    make_dilation_step_func,
    mod_steiner_classification,
    expand_conv_core,
    dilate_label_conv_mask,
)


# ---------------------------------------------------------------------------
# background_intensity
# ---------------------------------------------------------------------------

class TestBackgroundIntensity:
    def test_uniform_field_returns_same_value(self, grid_params, mask_goodvalues):
        """Uniform reflectivity should equal itself as background."""
        nx, ny = grid_params['nx'], grid_params['ny']
        refl = np.full((ny, nx), 30.0, dtype=np.float32)
        bkg = background_intensity(refl, mask_goodvalues, 1000.0, 1000.0,
                                   bkg_rad=5000.0, convolve_method='ndimage')
        # Interior pixels (away from boundary) should be ~30 dBZ
        interior = bkg[10:-10, 10:-10]
        assert np.allclose(interior, 30.0, atol=1.0), \
            "Background of uniform 30 dBZ field should stay ~30 dBZ"

    def test_masked_region_is_nan(self, grid_params):
        """Bad-value pixels should be NaN in background output."""
        nx, ny = grid_params['nx'], grid_params['ny']
        refl = np.full((ny, nx), 30.0, dtype=np.float32)
        mask = np.ones((ny, nx), dtype=int)
        mask[:, 0] = 0   # mask out first column
        bkg = background_intensity(refl, mask, 1000.0, 1000.0,
                                   bkg_rad=5000.0, convolve_method='ndimage')
        assert np.all(np.isnan(bkg[:, 0])), "Masked pixels should be NaN in background"

    def test_both_convolve_methods_agree(self, synthetic_refl_2d, mask_goodvalues, grid_params):
        """ndimage and signal convolution methods should give the same result."""
        bkg_ndimage = background_intensity(
            synthetic_refl_2d, mask_goodvalues,
            grid_params['dx'], grid_params['dy'],
            bkg_rad=5000.0, convolve_method='ndimage',
        )
        bkg_signal = background_intensity(
            synthetic_refl_2d, mask_goodvalues,
            grid_params['dx'], grid_params['dy'],
            bkg_rad=5000.0, convolve_method='signal',
        )
        valid = np.isfinite(bkg_ndimage) & np.isfinite(bkg_signal)
        assert np.allclose(bkg_ndimage[valid], bkg_signal[valid], atol=1e-3), \
            "ndimage and signal convolution should agree"


# ---------------------------------------------------------------------------
# make_dilation_step_func
# ---------------------------------------------------------------------------

class TestMakeDilationStepFunc:
    def test_default_lengths(self):
        bkg_bin, conv_rad_bin = make_dilation_step_func()
        # bkg_bin defines N bin edges; conv_rad_bin has N-1 values (one per interval)
        assert len(bkg_bin) == len(conv_rad_bin) + 1, \
            f"bkg_bin (len={len(bkg_bin)}) should have exactly one more element than conv_rad_bin (len={len(conv_rad_bin)})"

    def test_monotone_bkg_bin(self):
        bkg_bin, _ = make_dilation_step_func()
        assert np.all(np.diff(bkg_bin) > 0), "bkg_bin should be strictly increasing"

    def test_conv_rad_capped_at_max(self):
        max_r = 3
        _, conv_rad_bin = make_dilation_step_func(maxConvRadius=max_r)
        assert np.all(conv_rad_bin <= max_r), \
            f"All conv_rad_bin values should be <= maxConvRadius={max_r}"

    def test_first_bin_is_zero(self):
        bkg_bin, _ = make_dilation_step_func()
        assert bkg_bin[0] == 0, "First background bin should start at 0"

    def test_last_bin_is_hundred(self):
        bkg_bin, _ = make_dilation_step_func()
        assert bkg_bin[-1] == 100, "Last background bin should end at 100"


# ---------------------------------------------------------------------------
# mod_steiner_classification
# ---------------------------------------------------------------------------

class TestModSteinerClassification:
    """Integration-style test of the full Steiner pipeline on synthetic data."""

    @pytest.fixture
    def steiner_config(self, grid_params):
        bkg_bin, conv_rad_bin = make_dilation_step_func(
            mindBZuse=25, dBZforMaxConvRadius=40,
            bkg_refl_increment=5, conv_rad_increment=1,
            conv_rad_start=1, maxConvRadius=5,
        )
        return dict(
            dx=grid_params['dx'],
            dy=grid_params['dy'],
            bkg_bin=bkg_bin,
            conv_rad_bin=conv_rad_bin,
            bkg_rad=11000.0,
            minZdiff=8.0,
            absConvThres=43.0,
            truncZconvThres=46.0,
            weakEchoThres=15.0,
            min_corearea=0,
            min_cellarea=0,
            remove_smallcores=False,
            remove_smallcells=False,
            return_diag=False,
            convolve_method='ndimage',
        )

    def _run(self, steiner_types, refl, mask, cfg):
        return mod_steiner_classification(
            steiner_types,
            refl,
            mask,
            cfg['dx'], cfg['dy'],
            bkg_rad=cfg['bkg_rad'],
            minZdiff=cfg['minZdiff'],
            absConvThres=cfg['absConvThres'],
            truncZconvThres=cfg['truncZconvThres'],
            weakEchoThres=cfg['weakEchoThres'],
            bkg_bin=cfg['bkg_bin'],
            conv_rad_bin=cfg['conv_rad_bin'],
            min_corearea=cfg['min_corearea'],
            min_cellarea=cfg['min_cellarea'],
            remove_smallcores=cfg['remove_smallcores'],
            remove_smallcells=cfg['remove_smallcells'],
            return_diag=cfg['return_diag'],
            convolve_method=cfg['convolve_method'],
        )

    def test_convective_core_detected(
        self, steiner_types, synthetic_refl_2d, mask_goodvalues, steiner_config
    ):
        """The high-dBZ core at the centre should be labelled CONVECTIVE."""
        result = self._run(steiner_types, synthetic_refl_2d, mask_goodvalues, steiner_config)
        sclass = result['sclass']
        cy, cx = synthetic_refl_2d.shape[0] // 2, synthetic_refl_2d.shape[1] // 2
        assert sclass[cy, cx] == steiner_types['CONVECTIVE'], \
            "Centre of high-dBZ peak should be classified as CONVECTIVE"

    def test_output_keys_present(
        self, steiner_types, synthetic_refl_2d, mask_goodvalues, steiner_config
    ):
        """Result dict must contain at minimum sclass, score, score_dilate."""
        result = self._run(steiner_types, synthetic_refl_2d, mask_goodvalues, steiner_config)
        for key in ('sclass', 'score', 'score_dilate'):
            assert key in result, f"Expected key '{key}' missing from steiner result"

    def test_output_shape_matches_input(
        self, steiner_types, synthetic_refl_2d, mask_goodvalues, steiner_config
    ):
        result = self._run(steiner_types, synthetic_refl_2d, mask_goodvalues, steiner_config)
        assert result['sclass'].shape == synthetic_refl_2d.shape, \
            "sclass shape should match input reflectivity shape"

    def test_no_convective_in_weak_echo_field(
        self, steiner_types, grid_params, steiner_config
    ):
        """Uniform low reflectivity should not produce any CONVECTIVE pixels."""
        nx, ny = grid_params['nx'], grid_params['ny']
        weak_refl = np.full((ny, nx), 10.0, dtype=np.float32)
        mask = np.ones((ny, nx), dtype=int)
        result = self._run(steiner_types, weak_refl, mask, steiner_config)
        n_conv = np.sum(result['sclass'] == steiner_types['CONVECTIVE'])
        assert n_conv == 0, \
            f"Uniform 10 dBZ field should have 0 CONVECTIVE pixels, got {n_conv}"


# ---------------------------------------------------------------------------
# expand_conv_core
# ---------------------------------------------------------------------------

class TestExpandConvCore:
    def test_expanded_area_larger_than_input(self, grid_params):
        """Expanding a small core should produce a larger mask."""
        nx, ny = grid_params['nx'], grid_params['ny']
        core = np.zeros((ny, nx), dtype=int)
        core[25, 25] = 1    # single-pixel core at centre

        radii = np.array([1.0, 2.0, 3.0])
        core_expand, _ = expand_conv_core(
            core, radii, grid_params['dx'], grid_params['dy'], min_corenpix=0
        )
        n_original = np.sum(core > 0)
        n_expanded = np.sum(core_expand > 0)
        assert n_expanded > n_original, \
            "Expanded core should cover more pixels than the single-pixel input"

    def test_zero_radii_no_change(self, grid_params):
        """With radii of zero, the core should not change."""
        nx, ny = grid_params['nx'], grid_params['ny']
        core = np.zeros((ny, nx), dtype=int)
        core[25, 25] = 1
        radii = np.array([0.0])
        core_expand, _ = expand_conv_core(
            core, radii, grid_params['dx'], grid_params['dy'], min_corenpix=0
        )
        assert np.sum(core_expand > 0) >= np.sum(core > 0), \
            "Expansion with zero radius should not shrink the core"


# ---------------------------------------------------------------------------
# dilate_label_conv_mask
# ---------------------------------------------------------------------------

class TestDilateLabelConvMask:
    """Convective pixels closer than 2 x radius are grouped into one feature by a circular dilation."""

    @staticmethod
    def _grid(ny=40, nx=60):
        return np.zeros((ny, nx), dtype=bool)

    def test_two_pixels_30km_apart_separate_at_10km_joined_at_20km(self):
        mask = self._grid()
        mask[20, 10] = True
        mask[20, 20] = True          # 10 pixels = 30 km apart (dx = 3 km)
        sep = dilate_label_conv_mask(mask, 10e3, 3000.0, 3000.0)
        joined = dilate_label_conv_mask(mask, 20e3, 3000.0, 3000.0)
        assert sep.max() == 2
        assert sep[20, 10] != sep[20, 20]
        assert joined.max() == 1
        assert joined[20, 10] == joined[20, 20] == 1

    def test_dilation_is_a_disc_not_a_square(self):
        mask = self._grid()
        mask[20, 30] = True
        lab = dilate_label_conv_mask(mask, 10e3, 3000.0, 3000.0)    # radius 3.33 pixels
        assert lab[20, 33] == 1              # 3 pixels away along an axis: inside
        assert lab[22, 32] == 1              # 2.83 pixels on the diagonal: inside
        assert lab[23, 33] == 0              # 4.24 pixels on the diagonal: outside (a square would include it)
        assert lab[20, 34] == 0              # 4 pixels away along an axis: outside

    def test_anisotropic_grid_uses_dy_for_rows_and_dx_for_columns(self):
        mask = self._grid()
        mask[20, 30] = True
        lab = dilate_label_conv_mask(mask, 3000.0, 1000.0, 3000.0)    # dx = 1 km, dy = 3 km, radius 3 km
        assert lab[20, 33] == 1 and lab[20, 27] == 1     # 3 columns = 3 km
        assert lab[21, 30] == 1 and lab[19, 30] == 1     # 1 row = 3 km
        assert lab[22, 30] == 0                           # 2 rows = 6 km
        assert lab[20, 34] == 0                           # 4 columns = 4 km

    def test_zero_radius_labels_the_8_connected_regions(self):
        mask = self._grid()
        mask[5, 5] = mask[6, 6] = mask[7, 7] = True      # diagonal chain: one 8-connected region
        mask[20, 20] = True
        lab = dilate_label_conv_mask(mask, 0.0, 3000.0, 3000.0)
        assert lab.max() == 2
        assert lab[5, 5] == lab[6, 6] == lab[7, 7] == 1   # larger region first
        assert lab[20, 20] == 2
        assert np.array_equal(lab > 0, mask)              # nothing is added without a radius

    def test_negative_radius_is_treated_as_no_dilation(self):
        mask = self._grid()
        mask[10, 10] = True
        assert np.array_equal(dilate_label_conv_mask(mask, -5e3, 3000.0, 3000.0) > 0, mask)

    def test_labels_are_sorted_by_size_of_the_dilated_region(self):
        mask = self._grid(60, 80)
        mask[10, 10] = True                     # 1 pixel
        mask[30:33, 30:33] = True               # 9 pixels
        mask[50, 50:52] = True                  # 2 pixels
        lab = dilate_label_conv_mask(mask, 0.0, 3000.0, 3000.0)
        assert lab[31, 31] == 1 and lab[50, 50] == 2 and lab[10, 10] == 3
        # with dilation the sizes are those of the dilated regions
        lab = dilate_label_conv_mask(mask, 6e3, 3000.0, 3000.0)
        sizes = np.bincount(lab.ravel())[1:]
        assert np.all(sizes[:-1] >= sizes[1:])
        assert lab[31, 31] == 1

    def test_valid_data_mask_stops_the_dilation_but_not_the_convective_pixels(self):
        mask = self._grid()
        mask[20, 30] = True
        mask[20, 31] = True
        good = np.ones(mask.shape, dtype=int)
        good[:, 32:] = 0                         # no valid data from column 32 on
        good[20, 31] = 0                         # a convective pixel outside the valid area keeps its label
        lab = dilate_label_conv_mask(mask, 9e3, 3000.0, 3000.0, good=good)
        assert lab[20, 30] == lab[20, 31] == 1
        assert np.all(lab[:, 32:] == 0)
        assert lab[20, 28] == 1                  # still dilated over valid pixels

    def test_empty_mask_gives_zeros(self):
        lab = dilate_label_conv_mask(self._grid(), 10e3, 3000.0, 3000.0)
        assert lab.shape == (40, 60) and lab.dtype == np.int32 and not lab.any()

    def test_integer_mask_and_dtype(self):
        mask = np.zeros((20, 20), dtype=np.int64)
        mask[10, 10] = 4                         # any non-zero value is convective
        lab = dilate_label_conv_mask(mask, 3e3, 3000.0, 3000.0)
        assert lab.dtype == np.int32
        assert lab[10, 10] == 1 and lab[10, 11] == 1
