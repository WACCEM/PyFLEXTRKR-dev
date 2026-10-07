"""
Tests for the duration_range guardrail in ft_utilities.load_sparse_trackstats.

trackstats_driver expands the config's max track duration (duration_range_auto_update)
only in memory, so a step run on its own - without run_trackstats in the same run -
sees the unexpanded config value while the sparse file holds longer tracks.
"""
import logging

import numpy as np
import pytest
import xarray as xr

from pyflextrkr.ft_utilities import (
    convert_trackstats_sparse2dense,
    load_sparse_trackstats,
    resolve_sparse_max_trackduration,
)

TRACK_DURATIONS = [3, 12, 5]  # longest track = 12 time steps -> max time index 11


def _write_sparse_file(tmp_path, durations=TRACK_DURATIONS):
    rows, cols = [], []
    for it, dur in enumerate(durations):
        rows.extend([it] * dur)
        cols.extend(range(dur))
    rows = np.array(rows, dtype=np.int64)
    cols = np.array(cols, dtype=np.int64)
    ds = xr.Dataset(
        {
            "base_time": (["sparse_index"], 1.0e9 + 300.0 * cols),
            "tracks_indices": (["sparse_index"], rows),
            "times_indices": (["sparse_index"], cols),
            "track_duration": (["tracks"], np.array(durations, dtype=np.int32)),
        },
        coords={
            "tracks": (["tracks"], np.arange(len(durations))),
            "sparse_index": (["sparse_index"], np.arange(len(rows))),
        },
    )
    outfile = str(tmp_path / "trackstats_sparse_test.nc")
    ds.to_netcdf(outfile)
    return outfile


def _load(outfile, max_trackduration, **kwargs):
    return load_sparse_trackstats(
        max_trackduration, outfile, "times_indices", "tracks", "tracks_indices", **kwargs
    )


class TestLoadSparseTrackstatsGuardrail:

    def test_config_larger_than_data_is_unchanged(self, tmp_path):
        outfile = _write_sparse_file(tmp_path)
        _, _, sparse_dict, max_td = _load(outfile, 20)
        assert max_td == 20
        assert sparse_dict["base_time"].shape == (3, 20)

    def test_config_exactly_data_max_is_unchanged(self, tmp_path):
        outfile = _write_sparse_file(tmp_path)
        _, _, sparse_dict, max_td = _load(outfile, 12)
        assert max_td == 12
        assert sparse_dict["base_time"].shape == (3, 12)

    def test_auto_update_expands_and_rounds_up(self, tmp_path, caplog):
        outfile = _write_sparse_file(tmp_path)
        with caplog.at_level(logging.WARNING):
            _, _, sparse_dict, max_td = _load(
                outfile, 10, duration_range_auto_update=True, duration_range_round_base=10,
            )
        # 12 steps in the file -> rounded up to the nearest multiple of 10
        assert max_td == 20
        assert sparse_dict["base_time"].shape == (3, 20)
        assert "Auto-updating max_trackduration: 10 -> 20" in caplog.text
        # Nothing is lost: all 20 track-time points are present
        assert sparse_dict["base_time"].nnz == sum(TRACK_DURATIONS)

    def test_auto_update_round_base(self, tmp_path):
        outfile = _write_sparse_file(tmp_path)
        _, _, _, max_td = _load(
            outfile, 10, duration_range_auto_update=True, duration_range_round_base=5,
        )
        assert max_td == 15

    def test_auto_update_off_exits_with_critical_message(self, tmp_path, caplog):
        outfile = _write_sparse_file(tmp_path)
        with caplog.at_level(logging.CRITICAL):
            with pytest.raises(SystemExit):
                _load(outfile, 10, duration_range_auto_update=False)
        assert "exceeds 'duration_range' max (10)" in caplog.text
        assert "duration_range_auto_update: True" in caplog.text

    def test_default_does_not_auto_update(self, tmp_path):
        """Callers that do not pass the flag keep the strict behavior."""
        outfile = _write_sparse_file(tmp_path)
        with pytest.raises(SystemExit):
            _load(outfile, 10)

    def test_empty_file_does_not_crash(self, tmp_path):
        outfile = _write_sparse_file(tmp_path, durations=[0, 0])
        _, _, sparse_dict, max_td = _load(outfile, 10, duration_range_auto_update=True)
        assert max_td == 10
        assert sparse_dict["base_time"].shape == (2, 10)


class TestResolveSparseMaxTrackduration:

    def test_fits_unchanged(self):
        assert resolve_sparse_max_trackduration(np.arange(12), 12, "f.nc") == 12
        assert resolve_sparse_max_trackduration(np.arange(12), 30, "f.nc") == 30

    def test_expands_to_round_base_multiple(self):
        assert resolve_sparse_max_trackduration(
            np.arange(283), 10, "f.nc", duration_range_auto_update=True, duration_range_round_base=10,
        ) == 290

    def test_exact_multiple_not_bumped(self):
        assert resolve_sparse_max_trackduration(
            np.arange(70), 10, "f.nc", duration_range_auto_update=True, duration_range_round_base=10,
        ) == 70

    def test_invalid_round_base_does_not_crash(self):
        assert resolve_sparse_max_trackduration(
            np.arange(12), 10, "f.nc", duration_range_auto_update=True, duration_range_round_base=0,
        ) == 12

    def test_no_auto_update_exits(self):
        with pytest.raises(SystemExit):
            resolve_sparse_max_trackduration(np.arange(12), 10, "f.nc")


class TestConvertSparse2DenseGuardrail:

    @staticmethod
    def _convert(tmp_path, max_trackduration, **kwargs):
        sparse = _write_sparse_file(tmp_path)
        dense = str(tmp_path / "trackstats_dense_test.nc")
        status = convert_trackstats_sparse2dense(
            sparse, dense, max_trackduration, "tracks_indices", "times_indices",
            "tracks", "times", -9999, np.nan, **kwargs,
        )
        return status, dense

    def test_sufficient_max_unchanged(self, tmp_path):
        status, dense = self._convert(tmp_path, 20)
        assert status is True
        assert xr.open_dataset(dense, decode_times=False).sizes["times"] == 20

    def test_stale_max_with_auto_update_expands_and_keeps_all_data(self, tmp_path):
        status, dense = self._convert(
            tmp_path, 10, duration_range_auto_update=True, duration_range_round_base=10,
        )
        assert status is True
        ds = xr.open_dataset(dense, decode_times=False, mask_and_scale=False)
        assert ds.sizes["times"] == 20
        base_time = ds["base_time"].values
        # every track-time point of the sparse file is present (last point of the 12-step track)
        assert np.count_nonzero(~np.isnan(base_time)) == sum(TRACK_DURATIONS)
        assert not np.isnan(base_time[1, 11])

    def test_stale_max_without_auto_update_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            self._convert(tmp_path, 10)
