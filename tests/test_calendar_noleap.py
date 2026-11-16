"""
Tests of the config key `calendar` (ft_utilities.get_calendar, elapsed_seconds): a model with no 29 February (E3SM) has files from
28 February 23:00 to 1 March 00:00 in a leap year, 1 hour apart in the model and 25 hours apart when the file names are read as dates
of the Gregorian calendar. With `calendar: noleap` the tests that decide whether two files are consecutive (tracksingle, gettracks,
the step checks, the stitching) measure the time in the model calendar; without the key everything is as before.
Synthetic files only (tests/synthetic_tracks.py), no data needed.
"""
import numpy as np
import pandas as pd
import pytest

import synthetic_tracks as syn
from pyflextrkr import gettracks, stitch_tracks, step_outputs as so
from pyflextrkr.ft_utilities import elapsed_seconds, get_calendar


def _epoch(text):
    """
    Epoch time [s] of a UTC time text.
    """
    return int(pd.Timestamp(text, tz="UTC").timestamp())


########################################################################################################################
# The calendar key and the time difference
########################################################################################################################
@pytest.mark.parametrize("value, expected", [
    (None, "standard"), ("standard", "standard"), ("gregorian", "standard"), ("proleptic_gregorian", "standard"),
    ("noleap", "noleap"), ("NoLeap", "noleap"), ("365_day", "noleap"), ("no_leap", "noleap")])
def test_calendar_key_values(value, expected):
    assert get_calendar({"calendar": value}) == expected


def test_missing_calendar_key_is_the_standard_calendar():
    assert get_calendar({}) == "standard"


def test_a_wrong_calendar_is_refused():
    with pytest.raises(ValueError, match="calendar must be"):
        get_calendar({"calendar": "360_day"})
    with pytest.raises(ValueError, match="calendar must be"):
        elapsed_seconds(10, 0, "360_day")


def test_over_29_february_of_a_leap_year():
    new, ref = _epoch("2012-03-01 00:00"), _epoch("2012-02-28 23:00")
    assert elapsed_seconds(new, ref, "standard") == 25 * 3600
    assert elapsed_seconds(new, ref, "noleap") == 3600
    # the other order is the negative of it, and the same instants give 0
    assert elapsed_seconds(ref, new, "noleap") == -3600
    assert elapsed_seconds(new, new, "noleap") == 0


@pytest.mark.parametrize("ref, new", [
    ("2012-03-01 00:00", "2012-03-02 00:00"),      # after 29 February of a leap year
    ("2012-01-01 00:00", "2012-02-28 23:00"),      # before it
    ("2013-02-28 23:00", "2013-03-01 00:00"),      # a year without 29 February: 1 hour in both calendars
    ("2100-02-28 23:00", "2100-03-01 00:00"),      # 2100 is not a leap year either
])
def test_no_29_february_between_the_times_nothing_changes(ref, new):
    assert elapsed_seconds(_epoch(new), _epoch(ref), "noleap") == elapsed_seconds(_epoch(new), _epoch(ref), "standard")


def test_several_leap_years_and_the_century_rule():
    # 2012 and 2016 are in the range, 2020 starts after its 29 February is not yet reached on 1 January
    gregorian_days = 8 * 365 + 2
    assert elapsed_seconds(_epoch("2020-01-01 00:00"), _epoch("2012-01-01 00:00"), "noleap") == (gregorian_days - 2) * 86400
    # 2000 is a leap year (divisible by 400)
    assert elapsed_seconds(_epoch("2000-03-01 00:00"), _epoch("2000-02-28 00:00"), "noleap") == 86400


def test_arrays_broadcast_and_a_time_inside_29_february_counts_the_part_before_it():
    ref = _epoch("2012-02-28 23:00")
    new = np.array([_epoch("2012-02-29 00:00"), _epoch("2012-02-29 12:00"), _epoch("2012-03-01 00:00"), _epoch("2012-03-01 01:00")])
    out = elapsed_seconds(new, ref, "noleap")
    assert out.shape == new.shape
    assert list(out) == [3600, 3600, 3600, 7200]


########################################################################################################################
# The pairs that tracksingle makes (the check of the step), the track numbers and the stitching
########################################################################################################################
def _leap_data(tmp_path, calendar="noleap", **kwargs):
    """
    Hourly synthetic tracking from 28 February 2012 20:00 for 10 frames in a calendar without 29 February: the frames 0-3 are
    28 February 20:00 - 23:00, the frames 4-9 are 1 March 00:00 - 05:00 (file names of the Gregorian calendar), with track files for
    every pair (the pair 3-4 too).
    """
    return syn.write_synthetic_tracking(str(tmp_path / "data"), nframes=10, seed=7, dt=3600, t0="2012-02-28 20:00:00",
                                        timegap_hours=3.1, symmetric_links=True, calendar=calendar, **kwargs)


def test_the_synthetic_files_jump_a_day_over_29_february(tmp_path):
    info = _leap_data(tmp_path)
    assert info["names"][3] == "cloudid_20120228_230000.nc" and info["names"][4] == "cloudid_20120301_000000.nc"
    assert (3, 4) in info["pairs"] and len(info["pairs"]) == 9


def test_find_missing_pairs_in_both_calendars(tmp_path):
    info = _leap_data(tmp_path)
    files = [info["names"][f] for f in info["frames"]]
    times = [info["epoch"][f] for f in info["frames"]]
    existing = {so.track_file_name(times[i + 1]) for i in range(len(times) - 1)}
    without = existing - {so.track_file_name(times[4])}                 # the track file of the pair 3-4 is not there
    # the calendar of the data: the pair 3-4 is 1 h apart, so its missing track file is reported
    assert [m[1] for m in so.find_missing_pairs(files, times, without, 3.1, calendar="noleap")] == [files[4]]
    # the standard calendar: the pair is 25 h apart, no track file is expected for it
    assert so.find_missing_pairs(files, times, without, 3.1) == []


def test_gettracks_links_over_29_february_with_the_noleap_calendar(tmp_path):
    info = _leap_data(tmp_path)
    config = syn.make_config(info, str(tmp_path / "noleap"), tag="nl")
    config["calendar"] = "noleap"
    new = syn.read_tracknumbers(gettracks.gettracknumbers(config))
    # no gap anywhere: no track ends before the last file, and the attribute tells the calendar to the stitching
    assert not (new["track_reset"][0] == 2).any(axis=1)[:-1].any()
    assert new["attrs"]["calendar"] == "noleap"
    # the same files in the standard calendar: the pair 3-4 is a gap of 25 h, the tracks end in the column of frame 3 and restart
    config = syn.make_config(info, str(tmp_path / "standard"), tag="st")
    standard = syn.read_tracknumbers(gettracks.gettracknumbers(config))
    assert "calendar" not in standard["attrs"]
    # (the last column carries the final end flag of the data and is not counted)
    assert (standard["track_reset"][0] == 2).any(axis=1)[:-1].nonzero()[0].tolist() == [3]


def test_a_wrong_calendar_stops_gettracks(tmp_path):
    info = _leap_data(tmp_path)
    config = syn.make_config(info, str(tmp_path / "bad"), tag="bad")
    config["calendar"] = "360_day"
    with pytest.raises(ValueError, match="calendar must be"):
        gettracks.gettracknumbers(config)


def test_stitching_periods_in_the_calendar_of_the_data(tmp_path):
    # The periods [0, 3] and [4, 9] share no cloudid file and the files 3 and 4 are 1 h apart in the model: no pair links them
    # (the track file of the pair 3-4 is not there), so the boundary is a restart, and require-seams refuses it. In the standard
    # calendar they are 25 h apart: a data gap, which is no error.
    info = _leap_data(tmp_path, skip_track_files=(4,))
    paths = {}
    for calendar in ("noleap", "standard"):
        paths[calendar] = []
        for first, last, tag in ((0, 3, "A"), (4, 9, "B")):
            config = syn.make_config(info, str(tmp_path / f"{calendar}_{tag}"), first=first, last=last, tag=f"{calendar}{tag}")
            config["calendar"] = calendar
            paths[calendar].append(gettracks.gettracknumbers(config))
    with pytest.raises(ValueError, match="share no cloudid file"):
        stitch_tracks.stitch_tracknumbers(paths["noleap"], str(tmp_path / "nl.nc"), restart="error")
    out = stitch_tracks.stitch_tracknumbers(paths["noleap"], str(tmp_path / "nl.nc"))
    assert int(syn.read_tracknumbers(out)["attrs"]["n_chain_breaks"]) == 1
    assert syn.read_tracknumbers(out)["attrs"]["calendar"] == "noleap"
    out = stitch_tracks.stitch_tracknumbers(paths["standard"], str(tmp_path / "st.nc"), restart="error")
    assert int(syn.read_tracknumbers(out)["attrs"]["n_chain_breaks"]) == 0
    assert "calendar" not in syn.read_tracknumbers(out)["attrs"]
