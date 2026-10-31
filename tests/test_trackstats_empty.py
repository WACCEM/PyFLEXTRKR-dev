"""
Tests of trackstats_driver for frames and periods without features.

  - A period in which no file has a feature: gettracks still gives the extra (unlinked) cloud of every file a one-file track, the
    duration filter removes them all, and the driver used to fail in the merger/split correction on zero tracks. It now writes an
    empty sparse file (zero tracks) and ends cleanly.
  - A first frame without tracks returns (None, None) from calc_stats_singlefile; the search for the first result used to test the
    tuple (never None) and failed with 'NoneType' has no attribute 'keys' although the later frames had tracks.

Synthetic files only (tests/synthetic_tracks.py, and the radar_feature cloudid file of test_trackstats_radar_feature.py).
"""
import shutil

import netCDF4
import numpy as np
import pytest
import xarray as xr

import synthetic_tracks as syn
from pyflextrkr import gettracks
from pyflextrkr.step_outputs import period_has_no_features, trackstats_ntracks
from pyflextrkr.trackstats_driver import trackstats_driver, write_empty_trackstats_sparse
from test_trackstats_radar_feature import DX, _build_synthetic_cloudid


def driver_config(config, **overrides):
    """
    Add the keys that trackstats_driver reads to a config of the synthetic tracking.

    Args:
        config: dictionary
            From synthetic_tracks.make_config.
        **overrides:
            Keys to change.

    Returns:
        config: dictionary
    """
    config = dict(config, trackstats_filebase="trackstats_", trackstats_sparse_filebase="trackstats_sparse_",
                  duration_range=[2, 10], run_parallel=0, tracks_dimname="tracks", times_dimname="times", remove_shorttracks=1,
                  trackstats_dense_netcdf=0, pixel_radius=3.0, datatimeresolution=1 / 12, feature_type="generic",
                  feature_varname="feature_number", area_method="fixed", cloudid_filebase="cloudid_")
    config.update(overrides)
    return config


def add_grid(info):
    """
    Add what the first cloudid file is read for (2-D latitude and longitude) and an empty feature map to every cloudid file.

    Args:
        info: dictionary
            From synthetic_tracks.write_synthetic_tracking.
    """
    ny, nx = 6, 8
    lat = np.tile(np.linspace(30, 31, ny)[:, None], (1, nx))
    lon = np.tile(np.linspace(-100, -99, nx)[None, :], (ny, 1))
    for frame in info["frames"]:
        path = f"{info['folder']}/{info['names'][frame]}"
        with xr.open_dataset(path) as src:
            ds = src.load()
        ds["latitude"] = (("lat", "lon"), lat)
        ds["longitude"] = (("lat", "lon"), lon)
        ds["feature_number"] = (("time", "lat", "lon"), np.zeros((1, ny, nx), dtype=np.int32))
        ds["base_time"] = (("time",), np.array([info["epoch"][frame]], dtype="int64"),
                           {"units": "seconds since 1970-01-01", "long_name": "base time"})
        ds.to_netcdf(path)


def test_a_period_without_features_gives_an_empty_trackstats_file(tmp_path):
    info = syn.write_synthetic_tracking(str(tmp_path / "data"), nframes=12, seed=1, zero_frames=tuple(range(12)),
                                        symmetric_links=True)
    add_grid(info)
    config = driver_config(syn.make_config(info, str(tmp_path / "stats"), tag="x"))
    gettracks.gettracknumbers(config)
    assert period_has_no_features(config)
    trackstats_driver(config)
    assert trackstats_ntracks(config) == 0
    with xr.open_dataset(f"{config['stats_outpath']}trackstats_sparse_x-0_x-11.nc") as ds:
        assert ds.sizes["tracks"] == 0 and ds["track_duration"].size == 0 and "tracks_indices" in ds


def test_a_period_with_features_is_not_empty(tmp_path):
    info = syn.write_synthetic_tracking(str(tmp_path / "data"), nframes=6, seed=1, zero_frames=(0, 1, 2, 3, 4), symmetric_links=True)
    config = dict(syn.make_config(info, str(tmp_path / "stats"), tag="x"), cloudid_filebase="cloudid_")
    assert not period_has_no_features(config)                          # frame 5 has clouds
    assert not period_has_no_features(dict(config, tracking_outpath=str(tmp_path / "nothing") + "/"))    # no cloudid file: cannot tell


def test_the_empty_sparse_file_has_the_variables_of_the_later_steps(tmp_path):
    path = str(tmp_path / "trackstats_sparse_x.nc")
    write_empty_trackstats_sparse({"tracks_dimname": "tracks", "times_dimname": "times"}, path)
    write_empty_trackstats_sparse({"tracks_dimname": "tracks", "times_dimname": "times"}, path)     # an existing file is replaced
    with netCDF4.Dataset(path) as ds:
        assert ds.dimensions["tracks"].size == 0
        assert {"track_duration", "end_merge_tracknumber", "start_split_tracknumber", "tracks_indices", "times_indices"} <= set(ds.variables)


def test_a_first_frame_without_tracks_does_not_stop_the_statistics(tmp_path):
    # file 0 has no track (all track numbers are the fill value), file 1 has the three features of the radar_feature test as tracks
    first, name = _build_synthetic_cloudid(tmp_path)
    second = "cloudid_20250502_190543.nc"
    shutil.copy(first, tmp_path / second)
    names = np.zeros((2, len(name)), dtype="S1")
    for i, text in enumerate((name, second)):
        names[i, :] = list(text)
    fill = -9999
    numbers = np.full((1, 2, 30), fill, dtype=int)
    numbers[0, 1, :3] = [1, 2, 3]
    status = np.full((1, 2, 30), fill, dtype=int)
    status[0, 1, :3] = 1
    reset = np.full((1, 2, 30), fill, dtype=int)
    reset[0, 1, :3] = 0
    nomerge = np.full((1, 2, 30), fill, dtype=int)                    # no merger or split: the fill value (0 would be a track number)
    stats = tmp_path / "stats"
    stats.mkdir()
    gettracks.write_tracknumbers(str(stats / "tracknumbers_a_b.nc"), 4, np.array(["2025-05-02T19:00:43", "2025-05-02T19:05:43"],
                                                                                dtype="datetime64[ns]"),
                                 names, numbers, status, nomerge, nomerge, reset, "track_", "a", "b", 0.25)
    config = {"tracking_outpath": str(tmp_path) + "/", "stats_outpath": str(stats) + "/", "tracknumbers_filebase": "tracknumbers_",
              "startdate": "a", "enddate": "b", "fillval": fill, "timegap": 0.25, "feature_type": "radar_feature",
              "ref_varname": "dbz_comp", "trackstats_dbz_area_threshs": [45, 50], "terrain_file": None, "rangemask_varname": None,
              "pixel_radius": DX / 1000.0}
    config = driver_config(config, duration_range=[1, 4])              # remove_shorttracks 1: the row of the unused track number 0 goes
    trackstats_driver(config)
    assert trackstats_ntracks(config) == 3
    with xr.open_dataset(f"{config['stats_outpath']}trackstats_sparse_a_b.nc") as ds:
        assert (ds["track_duration"].values == 1).all()
