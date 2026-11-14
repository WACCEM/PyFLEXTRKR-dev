"""
Tests of the rows mode of the Tb identifymcs (pyflextrkr/identifymcs_tb_rows.py).

The central check: for the same sparse trackstats file, the row tables of identifymcs_tb_rows are identical (every column, dtype and
value, and the meta data: dims, dtypes, fill values, attributes) to what dense_to_rows makes of the dense file of identifymcs_tb.
Run on random tracks with long MCS-sized roots (periods split by dips, too short periods, no cold core), fragments that merge into and
split from them (also one fragment that does both), full slot dimensions, and a track longer than duration_range. Synthetic data only.
"""
import os

import numpy as np
import pytest

from pyflextrkr.identifymcs import identifymcs_tb
from pyflextrkr.identifymcs_tb_rows import (attach_tb_fragments_flat, identify_tb_mcs_track, identifymcs_tb_rows,
                                            tb_fragment_links)
from pyflextrkr.rowstats import dense_to_rows, read_tables, rows_to_dense, table_differences
from synthetic_tbpf import ENDDATE, STARTDATE, assert_same_dense, random_tb_sparse_file, tb_config


def run_both(stats_path, **overrides):
    """
    Run the dense and the rows step on the same sparse file.

    Args:
        stats_path: string
            Folder with the sparse trackstats file.
        **overrides:
            Config keys to change.

    Returns:
        dense_file, dense_tables, rows_tables: string, RowTables, RowTables
            The dense file, dense_to_rows of it, and the tables the rows step wrote.
    """
    config = tb_config(stats_path, **overrides)
    dense_file = identifymcs_tb(config)
    rows_folder = identifymcs_tb_rows(config)
    assert rows_folder.endswith(f"mcs_tracks_{STARTDATE}_{ENDDATE}.rows") and os.path.isdir(rows_folder)
    return dense_file, dense_to_rows(dense_file), read_tables(rows_folder)


def test_the_tb_test_of_one_track_periods_gap_duration_and_cold_core():
    f32 = lambda *v: np.array(v, dtype=np.float32)
    zeros = lambda n: np.zeros(n, dtype=np.float32)
    # Five hours above the threshold: one period of five hours
    status, is_mcs = identify_tb_mcs_track(f32(10, 50000, 50000, 50000, 50000, 50000, 10), zeros(7), 1.0, 40000, 4, 1)
    assert is_mcs and status.tolist() == [False, True, True, True, True, True, False]
    # A one-hour dip inside: with a gap of 1 two periods of two and three hours (both too short for 4 hours), with a gap of 2 one period
    core = f32(50000, 50000, 10, 50000, 50000, 50000)
    status, is_mcs = identify_tb_mcs_track(core, zeros(6), 1.0, 40000, 4, 1)
    assert not is_mcs and not status.any()
    status, is_mcs = identify_tb_mcs_track(core, zeros(6), 1.0, 40000, 4, 2)
    assert is_mcs and status.tolist() == [True, True, False, True, True, True]
    # Half-hour steps and a threshold of 4 hours: 6 steps are 3 hours (no), 8 steps are 4 hours (yes: the test is >=)
    assert not identify_tb_mcs_track(f32(*[50000] * 6), zeros(6), 0.5, 40000, 4, 1)[1]
    assert identify_tb_mcs_track(f32(*[50000] * 8), zeros(8), 0.5, 40000, 4, 1)[1]
    # No cold core (core area 0 everywhere): not an MCS however large the cold area is
    assert not identify_tb_mcs_track(zeros(6), np.full(6, 90000, dtype=np.float32), 1.0, 40000, 4, 1)[1]
    # A NaN is not part of the shield: the periods are [0, 1] (2 h) and [3, 4, 5] (3 h), and only the second one lasts 3 hours
    core = f32(1000, 1000, np.nan, 1000, 1000, 1000)
    status, is_mcs = identify_tb_mcs_track(core, np.full(6, 90000, dtype=np.float32), 1.0, 40000, 3, 1)
    assert is_mcs and status.tolist() == [False, False, False, True, True, True]


def test_fragment_links_need_a_whole_mcs_track_number_a_short_lifetime_and_no_mcs():
    is_mcs = np.array([True, False, True, False, False, False])
    lifetime = np.array([20.0, 2.0, 20.0, 1.0, 5.0, 2.0])
    link = np.array([-9999, 1, np.nan, 3, 3, 2.5])               # track 1 -> MCS 0 (number 1), track 3 -> MCS 2, track 4 too long
    fragments, roots = tb_fragment_links(link, lifetime, 3.0, is_mcs)
    assert fragments.tolist() == [1, 3] and roots.tolist() == [0, 2]
    # A number outside the file, a fill value and a non-integer number are no links; an MCS is not a fragment of another MCS
    assert tb_fragment_links(np.array([9, 0, -1, -9999, 1.5, 7]), lifetime, 3.0, is_mcs)[0].tolist() == []
    assert tb_fragment_links(np.array([3, 1, 1, 1, 1, 1]), np.full(6, 1.0), 3.0, is_mcs)[0].tolist() == [1, 3, 4, 5]


def test_fragments_fill_the_slots_in_track_order_and_drop_what_does_not_fit():
    # MCS track 0 has times 100..104 (hourly), fragments 1, 2, 3 have a time that is a time of the MCS (and fragment 3 a time that is not)
    offsets = np.array([0, 5, 7, 8, 10])
    basetime = np.array([100, 101, 102, 103, 104, 101, 102, 102, 500, 0], dtype=np.float64)
    cloudnumber = np.arange(10, dtype=np.int32) + 1
    ccs_area = np.arange(10, dtype=np.float32) * 10
    new_index_of = np.array([0, -1, -1, -1])
    columns, n_unmatched, n_truncated = attach_tb_fragments_flat(
        np.array([1, 2, 3]), np.array([0, 0, 0]), 0, basetime, cloudnumber, ccs_area, offsets, new_index_of, nmaxmerge=1)
    # Time index 1: fragment 1 takes slot 0; time index 2: fragment 1 takes it, fragment 2 (same time) is truncated
    assert columns["k"].tolist() == [1, 2] and columns["slot"].tolist() == [0, 0] and columns["cloudnumber"].tolist() == [6, 7]
    assert columns["ccs_area"].tolist() == [50.0, 60.0] and set(columns["kind"].tolist()) == {0}
    assert n_truncated == 1 and n_unmatched == 1                       # fragment 3: base time 500 is no time of the MCS, 0 has no entry


@pytest.mark.parametrize("seed", range(12))
def test_rows_step_equals_dense_step_on_random_tracks(tmp_path, seed):
    random_tb_sparse_file(str(tmp_path), seed, ntracks=90, crowd=3)
    dense_file, dense, rows = run_both(str(tmp_path) + "/")
    assert table_differences(dense, rows) == []
    assert len(dense.duration) > 0 and "vars_entry3d" not in rows.meta


def test_the_random_tracks_have_mcs_fragments_of_both_kinds_and_several_slots_in_use(tmp_path):
    kinds, nmcs, slots = set(), 0, set()
    for seed in range(4):
        folder = tmp_path / str(seed)
        folder.mkdir()
        random_tb_sparse_file(str(folder), seed, ntracks=90, crowd=3)
        _, _, rows = run_both(str(folder) + "/")
        kinds |= set(rows.fragments["kind"].tolist())
        slots |= set(rows.fragments["slot"].tolist())
        nmcs += len(rows.duration)
    assert kinds == {0, 1} and nmcs > 30 and {0, 1, 2} <= slots


@pytest.mark.parametrize("seed,nmaxlinks", [(1, 1), (4, 2), (7, 4)])
def test_rows_step_equals_dense_step_when_fragments_do_not_fit_in_the_slots(tmp_path, seed, nmaxlinks):
    random_tb_sparse_file(str(tmp_path), seed, ntracks=120, crowd=6)
    # Fewer slots than fragments at one time (the later fragments are dropped), a gap of two time steps and other thresholds
    dense_file, dense, rows = run_both(str(tmp_path) + "/", nmaxlinks=nmaxlinks, mcs_tb_gap=2, mcs_tb_duration_thresh=6,
                                       mcs_tb_merge_duration=5, mcs_tb_split_duration=2)
    assert table_differences(dense, rows) == []
    # The crowd of six fragments at one time filled all the slots of that time
    assert rows.fragments["slot"].max() == nmaxlinks - 1


def test_rows_back_to_dense_equal_the_dense_file(tmp_path):
    random_tb_sparse_file(str(tmp_path), 3)
    dense_file, _, rows = run_both(str(tmp_path) + "/")
    import xarray as xr
    want = xr.open_dataset(dense_file, mask_and_scale=False, decode_times=False)
    assert_same_dense(rows_to_dense(rows), want)


def test_a_track_longer_than_duration_range_widens_the_dense_layout_like_the_dense_step(tmp_path):
    random_tb_sparse_file(str(tmp_path), 5, nlong=2)
    dense_file, dense, rows = run_both(str(tmp_path) + "/", duration_range=[2, 40], duration_range_round_base=10)
    assert rows.meta["dims"]["times"] == dense.meta["dims"]["times"] == 50       # 45 steps, rounded up to a multiple of 10
    assert table_differences(dense, rows) == []


def test_no_mcs_gives_empty_tables_while_the_dense_step_exits(tmp_path):
    random_tb_sparse_file(str(tmp_path), 2)
    config = tb_config(str(tmp_path) + "/", mcs_tb_area_thresh=10**9)
    with pytest.raises(SystemExit):
        identifymcs_tb(config)
    rows = read_tables(identifymcs_tb_rows(config))
    assert len(rows.duration) == 0 and rows.meta["dims"]["tracks"] == 0 and len(rows.fragments["track"]) == 0
    # the output check of the run script stops a run that has no MCS (exit code 1)
    from pyflextrkr import step_outputs as so
    ok, message = so.check_step_outputs("identifymcs", dict(config, feature_type="tb_pf", stats_format="rows"))
    assert not ok and message == "0 MCS tracks"
