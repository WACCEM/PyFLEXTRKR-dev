"""
Tests of the rows mode of the robust MCS step (pyflextrkr/robustmcs_rows.py): for the same PF statistics, the row tables of
define_robust_mcs_pf_rows are identical (every column, dtype and value, and the meta data) to what dense_to_rows makes of the dense
file of robustmcspf.define_robust_mcs_pf (robustmcs_definition: feng21) and of robustmcspf_saag.define_robust_mcs_pf (mcsmip).
Random PF statistics with strong, weak and mixed tracks, missing PF, 0 / 0 heavy rain ratios, a PF major axis exactly at the lifetime
threshold, and different thresholds. Synthetic data only.
"""
import numpy as np
import pytest
import xarray as xr

from pyflextrkr import robustmcspf, robustmcspf_saag
from pyflextrkr.robustmcs_rows import (define_robust_mcs_pf_rows, feng21_mcs_periods, mcsmip_mcs_periods, robustmcs_params)
from pyflextrkr.rowstats import dense_to_rows, read_tables, rows_to_dense, table_differences
from synthetic_tbpf import NMAXPF, assert_same_dense, random_pf_stats_file, tb_config

DENSE_STEP = {"feng21": robustmcspf.define_robust_mcs_pf, "mcsmip": robustmcspf_saag.define_robust_mcs_pf}


def run_both(tmp_path, seed, definition, **overrides):
    """
    Run the dense and the rows robust MCS step on the same random PF statistics.

    Args:
        tmp_path: pathlib.Path
            Folder for the run.
        seed: int
            Random seed.
        definition: string
            'feng21' or 'mcsmip'.
        **overrides:
            Config keys to change.

    Returns:
        dense_file, dense_tables, rows_tables: string, RowTables, RowTables
    """
    stats = str(tmp_path) + "/"
    config = tb_config(stats, robustmcs_definition=definition, **overrides)
    random_pf_stats_file(stats, seed, config)
    dense_file = DENSE_STEP[definition](config)
    rows_folder = define_robust_mcs_pf_rows(config)
    return dense_file, dense_to_rows(dense_file), read_tables(rows_folder)


@pytest.mark.parametrize("definition", ["feng21", "mcsmip"])
@pytest.mark.parametrize("seed", range(8))
def test_rows_step_equals_dense_step(tmp_path, definition, seed):
    dense_file, dense, rows = run_both(tmp_path, seed, definition, mcs_pf_majoraxis_thresh=30, max_pf_majoraxis_thresh=70)
    assert table_differences(dense, rows) == []
    # The step selected some of the tracks (not all, not none) and kept the PF variables as small 3-D variables
    assert 0 < len(rows.duration) and rows.meta["vars_entry3d"]["pf_area"] == NMAXPF and "pf_mcsstatus" in rows.meta["vars_2d"]
    assert rows.meta["dims"]["tracks"] == len(rows.duration)


@pytest.mark.parametrize("definition", ["feng21", "mcsmip"])
def test_the_step_selects_a_part_of_the_tracks_and_marks_periods(tmp_path, definition):
    dense_file, dense, rows = run_both(tmp_path, 3, definition, mcs_pf_majoraxis_thresh=30, max_pf_majoraxis_thresh=70)
    n_before = read_tables(str(tmp_path) + f"/mcs_tracks_pf_20250101.0000_20250103.0000.rows").duration.size
    assert 0 < len(rows.duration) < n_before
    status = rows.entries["pf_mcsstatus"]
    assert set(np.unique(status)) <= {1, -9999} and (status == 1).any() and (status == -9999).any()
    assert rows.tracks["pf_lifetime"].dtype == np.float32 and (rows.tracks["pf_lifetime"] >= 0).all()


@pytest.mark.parametrize("definition", ["feng21", "mcsmip"])
@pytest.mark.parametrize("overrides", [
    dict(mcs_pf_majoraxis_thresh=0, max_pf_majoraxis_thresh=100000),                  # no limits on the major axis (the IMERG reference)
    dict(mcs_pf_majoraxis_thresh=30, max_pf_majoraxis_thresh=70, mcs_pf_gap=2, mcs_pf_durationthresh=6),
    dict(mcs_pf_majoraxis_thresh=25, max_pf_majoraxis_thresh=80, mcs_pf_majoraxis_for_lifetime=40, mcs_pf_durationthresh=3,
         coefs_pf_heavyratio=[5, 0.2], heavy_rainrate_thresh=6, mcs_min_rainvol_thresh=10000, mcs_volrain_duration_thresh=2.0),
])
def test_rows_step_equals_dense_step_with_other_thresholds(tmp_path, definition, overrides):
    dense_file, dense, rows = run_both(tmp_path, 5, definition, **overrides)
    assert table_differences(dense, rows) == []


@pytest.mark.parametrize("definition", ["feng21", "mcsmip"])
def test_rows_back_to_dense_equal_the_dense_file(tmp_path, definition):
    dense_file, _, rows = run_both(tmp_path, 1, definition, mcs_pf_majoraxis_thresh=30, max_pf_majoraxis_thresh=70)
    want = xr.open_dataset(dense_file, mask_and_scale=False, decode_times=False)
    assert_same_dense(rows_to_dense(rows), want)


@pytest.mark.parametrize("definition", ["feng21", "mcsmip"])
def test_a_pf_major_axis_exactly_at_the_lifetime_threshold_is_counted_as_that_value_like_the_dense_step(tmp_path, definition):
    # The dense step turns the largest PF major axis into 1 above mcs_pf_majoraxis_for_lifetime and 0 below it and leaves a value that
    # is exactly equal as it is, so such a time adds the threshold (20) to pf_lifetime: a lifetime longer than the track. The random
    # PF statistics have such entries (5 % of them), and the rows step gives the lifetime of the dense step including them.
    dense_file, dense, rows = run_both(tmp_path, 2, definition, mcs_pf_majoraxis_thresh=30, max_pf_majoraxis_thresh=70)
    assert (rows.tracks["pf_lifetime"] > rows.duration).any()
    assert np.array_equal(rows.tracks["pf_lifetime"], dense.tracks["pf_lifetime"])


def test_the_periods_of_one_track_for_both_definitions():
    p = dict(mcs_pf_majoraxis_thresh=30, max_pf_majoraxis_thresh=70, mcs_pf_durationthresh=4, mcs_pf_gap=1,
             coefs_pf_area=[4.2, 0.2], coefs_pf_rr=[0.5, 0.01], coefs_pf_skew=[0.2, 0.01], coefs_pf_heavyratio=[10, 0.5],
             heavy_rainrate_thresh=10, mcs_min_rainvol_thresh=20000, mcs_volrain_duration_thresh=1.0)
    f32 = lambda *v: np.array(v, dtype=np.float32)
    major = f32(50, 50, 50, 50, 50, 10, 50)
    ones = np.ones(7, dtype=np.float32)
    # Feng21: five hours of a big PF with a heavy rain ratio of 50 % (> 10 + 0.5 x 5), then a short one: one period of five hours
    status = feng21_mcs_periods(major, 100 * ones, 5 * ones, 2 * ones, 100 * ones, 50 * ones, 1.0, p)
    assert status.tolist() == [True, True, True, True, True, False, False]
    # No rain at all: the ratio is 0 / 0 = NaN, not an MCS
    assert not feng21_mcs_periods(major, 100 * ones, 5 * ones, 2 * ones, 0 * ones, 0 * ones, 1.0, p).any()
    # A heavy rain ratio below the fitted value (10 % < 12.5 %): not an MCS
    assert not feng21_mcs_periods(major, 100 * ones, 5 * ones, 2 * ones, 100 * ones, 10 * ones, 1.0, p).any()
    # MCSMIP: the maximum rain rate is above 10 for four hours and the volume above 20000 for one hour
    rr = f32(12, 12, 12, 12, 3, 3, 3)
    vol = f32(25000, 100, 100, 100, 100, 100, 100)
    assert mcsmip_mcs_periods(major, rr, vol, 1.0, p).tolist() == [True] * 5 + [False] * 2
    assert not mcsmip_mcs_periods(major, rr, vol * 0 + 100, 1.0, p).any()            # no hour with a large volume
    assert not mcsmip_mcs_periods(major, f32(12, 12, 12, 3, 3, 3, 3), vol, 1.0, p).any()  # only three hours of heavy rain


def test_the_definition_is_required_and_its_keys_are_checked():
    config = tb_config("s/")
    with pytest.raises(ValueError, match="robustmcs_definition must be one of"):
        robustmcs_params(config)
    with pytest.raises(ValueError, match="robustmcs_definition must be one of"):
        robustmcs_params(dict(config, robustmcs_definition="saag"))
    assert robustmcs_params(dict(config, robustmcs_definition="feng21"))[0] == "feng21"
    broken = {k: v for k, v in dict(config, robustmcs_definition="mcsmip").items() if k != "mcs_min_rainvol_thresh"}
    with pytest.raises(ValueError, match="needs the config keys.*mcs_min_rainvol_thresh"):
        robustmcs_params(broken)


@pytest.mark.parametrize("definition", ["feng21", "mcsmip"])
def test_no_robust_mcs_gives_empty_tables_while_the_dense_step_exits(tmp_path, definition):
    stats = str(tmp_path) + "/"
    config = tb_config(stats, robustmcs_definition=definition, mcs_pf_majoraxis_thresh=10**6, max_pf_majoraxis_thresh=10**7)
    random_pf_stats_file(stats, 2, config)
    with pytest.raises(SystemExit):
        DENSE_STEP[definition](config)
    rows = read_tables(define_robust_mcs_pf_rows(config))
    assert len(rows.duration) == 0 and rows.meta["dims"]["tracks"] == 0 and rows.tracks["pf_lifetime"].size == 0
    from pyflextrkr import step_outputs as so
    ok, message = so.check_step_outputs("robustmcs", dict(config, feature_type="tb_pf", stats_format="rows"))
    assert not ok and message == "0 robust MCS tracks"
