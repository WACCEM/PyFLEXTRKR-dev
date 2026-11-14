"""
The accumulation of precipitation along the Tb+PF tracks (trackpcp step 2) with the precipitation of the pixel-level files of the run itself
(trackpcp_data_path: null): for every track the rain at every pixel of its mask (cloudtracknumber) summed over the hours of its life. The
expected values are computed with plain loops on synthetic files (tests/synthetic_pixels.py), independent of the vectorized code.
"""
import numpy as np
import pandas as pd
import pytest

from pyflextrkr import trackpcp_driver as D
from pyflextrkr.step_outputs import check_step_outputs
from pyflextrkr.trackvar_io import open_data_source
from synthetic_pixels import ENDDATE, NHOUR, NX, NY, STARTDATE, T0, write_pixel_files, write_track_statistics


@pytest.fixture
def run(tmp_path):
    """The pixel files, the statistics and the config of a run."""
    pixel, stats = tmp_path / "mcstracking", tmp_path / "stats"
    pixel.mkdir()
    stats.mkdir()
    fields = write_pixel_files(str(pixel))
    write_track_statistics(str(stats) + "/")
    config = {
        "stats_outpath": str(stats) + "/", "startdate": STARTDATE, "enddate": ENDDATE, "start_basetime": T0, "end_basetime": T0 + NHOUR * 3600,
        "pixeltracking_outpath": str(pixel) + "/", "pixeltracking_filebase": "mcstrack_", "pixel_radius": 5.0, "area_method": "fixed",
        "run_parallel": 0, "datatimeresolution": 1.0, "mcsrobust_filebase": "mcs_tracks_robust_",
        # the data are the pixel files: no path; hourly data and hourly tracks at the same times
        "trackpcp_data_path": None, "trackpcp_time_match": "nearest", "trackpcp_window_sec": 3600,
        "trackpcp_class_varname": None, "trackpcp_convective_varname": None, "trackpcp_steps": [2], "trackpcp_outdir": str(tmp_path / "out"),
        "trackpcp_box_km": [10, 20], "trackpcp_accum_thresholds_mm": [20], "trackpcp_cluster_thresh_mm": 10,
    }
    return config, fields


def _truth(fields, track):
    """The accumulated rain of a track at every pixel of its mask, summed over the hours: plain loops."""
    truth = np.zeros((NY, NX))
    for h in range(NHOUR):
        for y in range(NY):
            for x in range(NX):
                if fields["cloudtracknumber"][h, y, x] == track and np.isfinite(fields["precipitation"][h, y, x]):
                    truth[y, x] += fields["precipitation"][h, y, x]              # a window of one hour: the depth is the rate
    return truth


def test_the_accumulation_of_every_track_equals_the_sum_over_its_hours_of_the_rain_in_its_mask(run):
    config, fields = run
    pix_file, met_file = D.compute_track_pixel_accum(config)
    df = pd.read_parquet(pix_file)
    assert set(df["track_number"].unique()) == {1, 2, 3}
    for track in (1, 2, 3):
        got = np.nan_to_num(D.track_pixels_to_map(df, track, NY, NX))
        np.testing.assert_allclose(got, _truth(fields, track), rtol=1e-5, atol=1e-5)
    # the metrics: the largest accumulation and the number of hours of every pixel
    met = pd.read_parquet(met_file).set_index("track_number")
    for track in (1, 2, 3):
        rows = df[df["track_number"] == track]
        assert met.loc[track, "accum_max_mm"] == pytest.approx(rows["accum_mm"].max(), rel=1e-5)
    # the pixel that tracks 1 (hour 2) and 2 (hour 4) cover at different hours has the rain of both hours, split between the two tracks
    y, x = 10, 6
    by_track = {t: np.nan_to_num(D.track_pixels_to_map(df, t, NY, NX))[y, x] for t in (1, 2)}
    assert by_track[1] > 0 and by_track[2] > 0
    # the second blob of track 3 (merged fragment, the same number) is part of its mask: its pixels are in the table
    fragment = np.nan_to_num(D.track_pixels_to_map(df, 3, NY, NX))[25:27, 30:34]
    assert (fragment > 0).any()


def test_a_data_path_gives_the_same_result_as_no_path(run, tmp_path):
    config, fields = run
    D.compute_track_pixel_accum(config)
    first = pd.read_parquet(D.pixel_accum_files(config, D.settings(config))[0])
    # the old way: the same files named as the data of the steps
    other = dict(config, trackpcp_data_path=config["pixeltracking_outpath"], trackpcp_data_filebase="mcstrack_",
                 trackpcp_data_varname="precipitation", trackpcp_outdir=str(tmp_path / "out2"))
    D.compute_track_pixel_accum(other)
    second = pd.read_parquet(D.pixel_accum_files(other, D.settings(other))[0])
    pd.testing.assert_frame_equal(first, second)


def test_the_data_source_without_a_path_is_the_pixel_files_and_a_path_is_used_as_before(run):
    config, fields = run
    source = open_data_source(config, T0 - 10, T0 + NHOUR * 3600)
    assert source.nrecords == NHOUR and source.varname == "precipitation" and list(source.times) == [T0 + 3600 * h for h in range(NHOUR)]
    np.testing.assert_array_equal(np.nan_to_num(source.read(2)), np.nan_to_num(fields["precipitation"][2]))
    # another variable of the pixel files can be named; an explicit path with its own keys is the old behaviour
    named = open_data_source(dict(config, trackpcp_data_varname="pcptracknumber"), T0, T0 + 6 * 3600)
    assert named.varname == "pcptracknumber" and named.nrecords == NHOUR
    # only the records inside the time range are listed
    assert open_data_source(config, T0 + 3600 - 10, T0 + 3 * 3600 + 10).nrecords == 3
    explicit = open_data_source(dict(config, trackpcp_data_path=config["pixeltracking_outpath"], trackpcp_data_filebase="mcstrack_",
                                     trackpcp_data_varname="precipitation"), T0, T0 + 6 * 3600)
    assert explicit.nrecords == NHOUR
    with pytest.raises(KeyError):
        open_data_source(dict(config, trackpcp_data_path="x/"), T0, T0 + 3600)          # an explicit path still needs its variable name


def test_the_pipeline_step_makes_the_files_and_the_check_sees_them(run):
    config, fields = run
    config = dict(config, feature_type="tb_pf")
    assert not check_step_outputs("trackpcp", config)[0]
    D.trackpcp_driver(config)
    assert check_step_outputs("trackpcp", config)[0]
    assert [f.rsplit("/", 1)[-1] for f in D.expected_outputs(config)] == [f"trackpcp_pixels_{STARTDATE}_{ENDDATE}.parquet",
                                                                          f"trackpcp_metrics_{STARTDATE}_{ENDDATE}.parquet", "trackpcp_grid.nc"]
