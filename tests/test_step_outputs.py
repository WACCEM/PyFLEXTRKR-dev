"""
Tests of pyflextrkr.step_outputs: the checks that the outputs of every pipeline step are complete, the missing pairs of
cloudid files, and run_step (skip a completed step, run all later steps after one has run, stop when the outputs of a step are
not complete). Synthetic files only (tests/synthetic_tracks.py and a few minimal netCDF files).
"""
import os

import netCDF4
import numpy as np
import pandas as pd
import pytest

import synthetic_tracks as syn
from pyflextrkr import gettracks, step_outputs as so
from pyflextrkr.trackstats_driver import write_empty_trackstats_sparse

T0 = int(pd.Timestamp("2020-01-01 00:00:00", tz="UTC").timestamp())


def _epoch(text):
    return int(pd.Timestamp(text, tz="UTC").timestamp())


def test_track_file_name_and_missing_pairs():
    assert so.track_file_name(_epoch("2025-06-01 00:00:40")) == "track_20250601_000040.nc"
    stamps = ["2025-05-31 23:50:40", "2025-05-31 23:54:40", "2025-06-01 00:00:40", "2025-06-01 00:04:41", "2025-06-01 02:00:00"]
    times = [_epoch(s) for s in stamps]
    files = [f"cloudid_{i}.nc" for i in range(len(stamps))]
    existing = {so.track_file_name(times[1]), so.track_file_name(times[3])}
    # pair 1-2 is the bridge over the month boundary; pair 3-4 is further apart than timegap and gets no track file by design
    assert so.find_missing_pairs(files, times, existing, 0.25) == [("cloudid_1.nc", "cloudid_2.nc", times[1], times[2])]
    assert so.find_missing_pairs(files, times, existing | {so.track_file_name(times[2])}, 0.25) == []


def make_run(tmp_path, **tracking_kwargs):
    """
    A small finished tracking run: input files, cloudid and track files, and the config that describes it.

    Args:
        tmp_path: pathlib.Path
            Folder for the run.
        **tracking_kwargs:
            Options of synthetic_tracks.write_synthetic_tracking (e.g. zero_frames).

    Returns:
        info, config, tmp_path
    """
    info = syn.write_synthetic_tracking(str(tmp_path / "tracking"), nframes=12, seed=3, symmetric_links=True, **tracking_kwargs)
    inputs = tmp_path / "input"
    inputs.mkdir()
    for frame in info["frames"]:
        stamp = info["names"][frame][len("cloudid_"):-3]
        (inputs / f"in_{stamp}.nc").write_bytes(b"x")
    config = syn.make_config(info, str(tmp_path / "stats"), tag="t")
    config.update(
        start_basetime=info["epoch"][info["frames"][0]], cloudid_filebase="cloudid_", clouddata_path=str(inputs) + "/",
        databasename="in_", time_format="yyyymodd_hhmmss", trackstats_sparse_filebase="trackstats_sparse_",
        trackstats_filebase="trackstats_", mcsradarstats_filebase="mcs_tracks_", mcsrobust_filebase="mcs_tracks_robust_",
        mcsfinal_filebase="mcs_tracks_final_", pixeltracking_filebase="mcstrack_", pixeltracking_outpath=str(tmp_path / "pixel") + "/",
        tracks_dimname="tracks", times_dimname="times", trackstats_dense_netcdf=0)
    return info, config, tmp_path


@pytest.fixture
def run(tmp_path):
    """
    A small finished tracking run (see make_run).
    """
    return make_run(tmp_path)


def _stats_path(config, filebase):
    return f"{config['stats_outpath']}{filebase}{config['startdate']}_{config['enddate']}.nc"


def _write_nc(path, dims=None, variables=None, attrs=None):
    with netCDF4.Dataset(path, "w") as ds:
        for name, size in (dims or {}).items():
            ds.createDimension(name, size)
        for name, (dim, values) in (variables or {}).items():
            ds.createVariable(name, "f8", dim)[:] = values
        for key, value in (attrs or {}).items():
            ds.setncattr(key, value)


def test_idfeature_counts_cloudid_files_against_inputs(run):
    info, config, tmp_path = run
    assert so.check_step_outputs("idfeature", config)[0]
    os.remove(f"{config['tracking_outpath']}{info['names'][5]}")
    ok_strict, message = so.check_step_outputs("idfeature", config, strict=True)
    ok_after, _ = so.check_step_outputs("idfeature", config, strict=False)
    assert not ok_strict and ok_after and "11 cloudid files for 12 input files" in message


def test_tracksingle_finds_a_missing_track_file(run):
    info, config, _ = run
    assert so.check_step_outputs("tracksingle", config)[0]
    os.remove(f"{config['tracking_outpath']}track_{info['names'][7][len('cloudid_'):]}")
    ok, message = so.check_step_outputs("tracksingle", config)
    assert not ok and "1 pair(s) of cloudid files without a track file" in message


def test_gettracks_needs_the_last_file(run):
    info, config, _ = run
    ok, message = so.check_step_outputs("gettracks", config)
    assert not ok and "does not exist" in message
    gettracks.gettracknumbers(dict(config, startdate="t", enddate="t"))
    config = dict(config, startdate="t", enddate="t")
    assert so.check_step_outputs("gettracks", config)[0]
    # the track numbers end before the last cloudid file: the run was stopped
    os.remove(f"{config['tracking_outpath']}{info['names'][11]}")
    ok, message = so.check_step_outputs("gettracks", config)
    assert not ok and "last file of the track numbers" in message


def test_gettracks_accepts_a_last_cloudid_file_that_is_a_gap_and_not_a_missing_track_file(tmp_path):
    # Frames 9 and 10 do not exist: the last frame (11) is 15 min after frame 8, not closer than timegap (0.25 h), so no pair ends in it.
    # It is in no track file and not in the track numbers, which end in frame 8: not an incomplete run
    info, config, _ = make_run(tmp_path, skip_frames=(9, 10))
    config = dict(config, startdate="t", enddate="t")
    gettracks.gettracknumbers(config)
    ok, message = so.check_step_outputs("gettracks", config)
    assert ok and "(a gap): it starts no pair" in message
    # the same when the last file is in no pair because its track file is missing (closer than timegap): that one is an error
    info, config, _ = make_run(tmp_path / "second", skip_track_files=(11,))
    config = dict(config, startdate="t", enddate="t")
    gettracks.gettracknumbers(config)
    ok, message = so.check_step_outputs("gettracks", config)
    assert not ok and "last file of the track numbers" in message


def test_gettracks_check_asks_for_the_noleap_calendar_when_the_gap_is_29_february(tmp_path, caplog):
    # The February chunk of a model with no 29 February: hourly files from 28 February 20:00 to 1 March 00:00 (frame 4). The standard
    # calendar makes no track file for the pair 23:00 -> 00:00 (25 h apart), so the track numbers end at 23:00 and the last cloudid file
    # is 25 h later
    info, config, _ = make_run(tmp_path, dt=3600, t0="2012-02-28 20:00:00", timegap_hours=3.1, calendar="noleap",
                               skip_frames=tuple(range(5, 12)), skip_track_files=(4,))
    config = dict(config, startdate="t", enddate="t", timegap=3.1)
    gettracks.gettracknumbers(config)
    with caplog.at_level("WARNING"):
        ok, message = so.check_step_outputs("gettracks", config)
    assert ok and "a gap" in message
    assert "set `calendar: noleap`" in caplog.text
    # with the calendar of the data the pair 23:00 -> 00:00 is 1 h apart and has no track file: the run is incomplete, whatever the gap says
    config = dict(config, startdate="u", enddate="u", calendar="noleap")
    gettracks.gettracknumbers(config)
    ok, message = so.check_step_outputs("gettracks", config)
    assert not ok and "last file of the track numbers" in message


def test_statistics_files_robust_attributes_pixel_files_and_speed(run):
    info, config, tmp_path = run
    os.makedirs(config["stats_outpath"], exist_ok=True)
    dims = {"tracks": 3, "times": 2}
    # trackstats (sparse file) and its dense file when asked for
    assert not so.check_step_outputs("trackstats", config)[0]
    _write_nc(_stats_path(config, "trackstats_sparse_"), {"tracks": 3})
    assert so.check_step_outputs("trackstats", config)[0]
    assert not so.check_step_outputs("trackstats", dict(config, trackstats_dense_netcdf=1))[0]
    # identifymcs: the variables the later steps need
    _write_nc(_stats_path(config, "mcs_tracks_"), dims, {"is_mcs": (("tracks",), [1, 0, 0])})
    assert "variables ['track_class', 'track_duration'] are missing" in so.check_step_outputs("identifymcs", config)[1]
    _write_nc(_stats_path(config, "mcs_tracks_"), dims, {"is_mcs": (("tracks",), [1, 0, 0]), "track_class": (("tracks",), [1, 2, 3]),
                                                         "track_duration": (("tracks",), [4, 4, 4])})
    assert so.check_step_outputs("identifymcs", config)[0]
    # regionprops: every (track, time) pair has a merged mask whose area is total_area
    robust = _stats_path(config, "mcs_tracks_robust_")
    for attrs, expected in (({"n_pairs_not_computed": 0, "n_pairs_area_mismatch": 0}, True),
                            ({"n_pairs_not_computed": 0, "n_pairs_area_mismatch": 7}, False), ({}, False)):
        _write_nc(robust, dims, attrs=attrs)
        assert so.check_step_outputs("regionprops", config)[0] is expected
    # mapfeature: one pixel file per cloudid file
    os.makedirs(config["pixeltracking_outpath"], exist_ok=True)
    for frame in info["frames"][:-1]:
        open(f"{config['pixeltracking_outpath']}mcstrack_{info['names'][frame][len('cloudid_'):]}", "w").close()
    assert not so.check_step_outputs("mapfeature", config)[0]
    open(f"{config['pixeltracking_outpath']}mcstrack_{info['names'][info['frames'][-1]][len('cloudid_'):]}", "w").close()
    assert so.check_step_outputs("mapfeature", config)[0]
    # speed: the final file with finite movement speeds
    final = _stats_path(config, "mcs_tracks_final_")
    _write_nc(final, dims, {"movement_speed": (("tracks", "times"), np.full((3, 2), np.nan))})
    assert not so.check_step_outputs("speed", config)[0]
    _write_nc(final, dims, {"movement_speed": (("tracks", "times"), np.array([[1.0, np.nan]] * 3))})
    assert so.check_step_outputs("speed", config)[0]
    with pytest.raises(ValueError):
        so.check_step_outputs("nothing", config)


def test_run_step_skips_completed_steps_runs_later_steps_and_verifies(run):
    info, config, _ = run
    calls = []
    config = dict(config, run_tracksingle=True, run_gettracks=True, run_trackstats=True, skip_completed_steps=True)
    _write_nc(_stats_path(config, "trackstats_sparse_"), {"tracks": 3})                  # trackstats is complete already
    state = {"stale": False}

    def make_gettracks(cfg):
        calls.append("gettracks")
        gettracks.gettracknumbers(cfg)

    # tracksingle is complete: skipped. gettracks has no output: runs. trackstats is complete but comes after a step that ran: runs
    assert so.run_step("tracksingle", config, state, lambda cfg: calls.append("tracksingle")) is None
    assert calls == [] and state["stale"] is False
    so.run_step("gettracks", config, state, make_gettracks)
    so.run_step("trackstats", config, state, lambda cfg: calls.append("trackstats"))
    assert calls == ["gettracks", "trackstats"] and state["stale"] is True
    # a step that is switched off is not run
    assert so.run_step("speed", dict(config, run_speed=False), state, lambda cfg: calls.append("speed")) is None
    # a step that leaves incomplete outputs stops the run; without verification it does not
    with pytest.raises(SystemExit) as stop:
        so.run_step("identifymcs", dict(config, run_identifymcs=True), state, lambda cfg: None)
    assert stop.value.code == 1
    so.run_step("identifymcs", dict(config, run_identifymcs=True, verify_step_outputs=False), state, lambda cfg: calls.append("id"))
    assert calls[-1] == "id"


def _write_rows_tables(folder, tracks, entries=None, global_attrs=None, entries3d=None):
    """
    Write small row tables (three tracks of two entries each) for the checks of stats_format: rows.

    Args:
        folder: string
            Output folder.
        tracks: dictionary
            {name: values [3]} 1-D variables.
        entries: dictionary (default: None)
            {name: values [6]} [tracks, times] variables.
        global_attrs: dictionary (default: None)
            Global attributes.
        entries3d: dictionary (default: None)
            {name: values [6, n]} small [tracks, times, n] variables (the PF statistics).
    """
    from pyflextrkr.rowstats import TableBuilder, write_rows
    builder = TableBuilder(np.full(3, 2), 2, global_attrs=global_attrs)
    for name, values in tracks.items():
        builder.add_track_var(name, np.asarray(values))
    for name, values in (entries or {}).items():
        builder.add_entry_var(name, np.asarray(values, dtype=float), fill=np.nan)
    for name, values in (entries3d or {}).items():
        builder.add_entry3d_var(name, np.asarray(values, dtype=float), "nmaxpf", fill=np.nan)
    write_rows(builder.build(), folder)


def test_rows_mode_checks_read_the_row_folders(run):
    info, config, tmp_path = run
    config = dict(config, stats_format="rows")
    os.makedirs(config["stats_outpath"], exist_ok=True)
    assert so.rows_mode(config) and not so.rows_mode({})
    assert so.stats_output(config, "mcs_tracks_").endswith("mcs_tracks_" + config["startdate"] + "_" + config["enddate"] + ".rows")
    assert so.stats_output({"stats_outpath": "s/", "startdate": "a", "enddate": "b"}, "x_") == "s/x_a_b.nc"
    # identifymcs
    assert "does not exist" in so.check_step_outputs("identifymcs", config)[1]
    folder = so.stats_output(config, "mcs_tracks_")
    _write_rows_tables(folder, {"track_duration": [2, 2, 2], "is_mcs": [1, 0, 0]})
    assert "variables ['track_class'] are missing" in so.check_step_outputs("identifymcs", config)[1]
    _write_rows_tables(folder, {"track_duration": [2, 2, 2], "is_mcs": [1, 0, 0], "track_class": [1, 2, 3]})
    assert so.check_step_outputs("identifymcs", config)[0]
    # regionprops: the attributes in the meta data
    robust = so.stats_output(config, "mcs_tracks_robust_")
    for attrs, expected in (({"n_pairs_not_computed": 0, "n_pairs_area_mismatch": 0}, True),
                            ({"n_pairs_not_computed": 0, "n_pairs_area_mismatch": 7}, False), ({}, False)):
        _write_rows_tables(robust, {"track_duration": [2, 2, 2]}, global_attrs=attrs)
        assert so.check_step_outputs("regionprops", config)[0] is expected
    # speed: finite movement speeds in the entries
    final = so.stats_output(config, "mcs_tracks_final_")
    _write_rows_tables(final, {"track_duration": [2, 2, 2]}, {"base_time": np.arange(6)})
    assert so.check_step_outputs("speed", config)[1] == "movement_speed is not in the file"
    _write_rows_tables(final, {"track_duration": [2, 2, 2]}, {"movement_speed": np.full(6, np.nan)})
    assert not so.check_step_outputs("speed", config)[0]
    _write_rows_tables(final, {"track_duration": [2, 2, 2]}, {"movement_speed": [1.0, np.nan] * 3})
    assert so.check_step_outputs("speed", config)[0]


def test_idfeature_manifest_counts_the_frames_skipped_by_miss_thresh(run):
    info, config, _ = run
    names = [info["names"][f] for f in info["frames"]]
    os.remove(f"{config['tracking_outpath']}{names[5]}")                          # frame 5 was skipped: no cloudid file
    assert not so.check_step_outputs("idfeature", config, strict=True)[0]       # no manifest: it looks incomplete, as before
    returned = [None if i == 5 else name for i, name in enumerate(names)]         # what the calls of idfeature_driver returned
    so.write_idfeature_manifest(config, len(names), returned)
    ok, message = so.check_step_outputs("idfeature", config, strict=True)
    assert ok and "11 cloudid files for 12 input files (1 frames skipped by miss_thresh)" in message
    # a manifest that does not belong to this period or number of inputs is ignored
    assert so.read_idfeature_manifest(config, 12)["n_calls_skipped"] == 1
    assert so.read_idfeature_manifest(config, 13) is None
    assert so.read_idfeature_manifest(dict(config, start_basetime=config["start_basetime"] - 3600), 12) is None
    # a file that is missing without having been skipped is still found
    os.remove(f"{config['tracking_outpath']}{names[6]}")
    assert not so.check_step_outputs("idfeature", config, strict=True)[0]
    assert so.check_step_outputs("idfeature", config, strict=False)[0]


def test_a_period_without_features_passes_with_a_warning_and_skips_the_later_steps(tmp_path, caplog):
    info, config, _ = make_run(tmp_path, zero_frames=tuple(range(12)))
    config = dict(config, skip_completed_steps=True, **{f"run_{step}": True for step in so.STEPS})
    assert so.period_has_no_features(config)
    calls = []

    def empty_trackstats(cfg):
        calls.append("trackstats")
        write_empty_trackstats_sparse(cfg, _stats_path(cfg, "trackstats_sparse_"))

    def run_all(state):
        so.run_step("gettracks", config, state, gettracks.gettracknumbers)
        so.run_step("trackstats", config, state, empty_trackstats)
        for step in ("identifymcs", "regionprops", "mapfeature", "speed"):
            assert so.run_step(step, config, state, lambda cfg, step=step: calls.append(step)) is None

    state = {"stale": False}
    with caplog.at_level("WARNING"):
        run_all(state)
    assert calls == ["trackstats"] and state["empty"] is True
    assert any("no tracks: none of the cloudid files" in r.message and "identifymcs" in r.message for r in caplog.records)
    # submitted again: gettracks and trackstats are complete (the empty file counts), the later steps are skipped again
    calls.clear()
    state = {"stale": False}
    run_all(state)
    assert calls == [] and state["empty"] is True and state["stale"] is False


def test_zero_tracks_only_pass_when_no_cloudid_file_has_a_feature(run):
    info, config, _ = run
    os.makedirs(config["stats_outpath"], exist_ok=True)
    write_empty_trackstats_sparse(config, _stats_path(config, "trackstats_sparse_"))
    ok, message = so.check_step_outputs("trackstats", config)
    assert not ok and message == "0 tracks"                                       # the frames have features: no tracks is an error
    state = {"stale": False}
    so._note_empty("trackstats", config, state)
    assert "empty" not in state


def test_an_empty_period_needs_no_dense_trackstats_file(tmp_path):
    info, config, _ = make_run(tmp_path, zero_frames=tuple(range(12)))
    write_empty_trackstats_sparse(config, _stats_path(config, "trackstats_sparse_"))
    config = dict(config, trackstats_dense_netcdf=1)
    assert so.check_step_outputs("trackstats", config) == (True, so.EMPTY_NOTE)


def test_trackpcp_is_an_optional_step_that_is_off_by_default_and_skipped_for_an_empty_period():
    from pyflextrkr import step_outputs
    assert "trackpcp" in step_outputs.STEPS and "trackpcp" in step_outputs.STEPS_AFTER_TRACKSTATS
    calls = []
    func = lambda config: calls.append(config.get("tag"))
    # the flag is off: not run
    assert step_outputs.run_step("trackpcp", {"tag": "a", "verify_step_outputs": False}, {"stale": False}, func) is None and calls == []
    # on: runs once; an empty period (no tracked feature) skips it
    step_outputs.run_step("trackpcp", {"tag": "b", "run_trackpcp": True, "verify_step_outputs": False}, {"stale": False}, func)
    step_outputs.run_step("trackpcp", {"tag": "c", "run_trackpcp": True, "verify_step_outputs": False}, {"stale": False, "empty": True}, func)
    assert calls == ["b"]
    # the output check of an unknown step is still an error; the check of this one asks for the files of its steps
    with pytest.raises(ValueError):
        step_outputs.check_step_outputs("nothing", {})


########################################################################################################################
# Tb+PF MCS tracking (feature_type tb_pf): its own checks for identifymcs, matchpf and robustmcs, the idfeature manifest, and the
# steps that have no check
########################################################################################################################
def _tbpf(config, **extra):
    """
    The config of a Tb+PF run: the feature type and the file base names of its statistics steps.
    """
    return dict(config, feature_type="tb_pf", mcstbstats_filebase="mcs_tracks_", mcspfstats_filebase="mcs_tracks_pf_",
                mcsrobust_filebase="mcs_tracks_robust_", **extra)


def test_tbpf_checks_of_the_dense_statistics_files(run):
    info, config, _ = run
    config = _tbpf(config)
    os.makedirs(config["stats_outpath"], exist_ok=True)
    dims = {"tracks": 3, "times": 2, "nmaxpf": 2}
    tk, tt, tp = ("tracks",), ("tracks", "times"), ("tracks", "times", "nmaxpf")
    # identifymcs: the variables of identifymcs_tb (the radar check of the same step name asks for others)
    assert "does not exist" in so.check_step_outputs("identifymcs", config)[1]
    _write_nc(_stats_path(config, "mcs_tracks_"), dims, {"is_mcs": (tk, [1, 0, 0])})
    assert "variables ['mcs_status', 'ccs_area', 'track_duration'] are missing" in so.check_step_outputs("identifymcs", config)[1]
    _write_nc(_stats_path(config, "mcs_tracks_"), dims, {"track_duration": (tk, [2, 2, 2]), "mcs_status": (tt, np.ones((3, 2))),
                                                         "ccs_area": (tt, np.ones((3, 2)))})
    assert so.check_step_outputs("identifymcs", config) == (True, "3 MCS tracks")
    assert not so.check_step_outputs("identifymcs", dict(config, feature_type="radar_feature"))[0]
    # matchpf: the variables, and a non-zero fraction of the entries matched to a cloudid file (total_rain is NaN where nothing matched)
    pf = _stats_path(config, "mcs_tracks_pf_")
    assert "does not exist" in so.check_step_outputs("matchpf", config)[1]
    _write_nc(pf, dims, {"track_duration": (tk, [2, 2, 2]), "pf_area": (tp, np.ones((3, 2, 2)))})
    assert "variables ['total_rain', 'pf_npf'] are missing" in so.check_step_outputs("matchpf", config)[1]
    unmatched = {"track_duration": (tk, [2, 2, 2]), "pf_area": (tp, np.ones((3, 2, 2))), "pf_npf": (tt, np.zeros((3, 2))),
                 "total_rain": (tt, np.full((3, 2), np.nan))}
    _write_nc(pf, dims, unmatched)
    ok, message = so.check_step_outputs("matchpf", config)
    assert not ok and message == "0 of 6 entries (0.0 %) have PF statistics"
    matched = np.array([[1.0, 0.0], [np.nan, 2.0], [3.0, np.nan]])
    _write_nc(pf, dims, dict(unmatched, total_rain=(tt, matched)))
    assert so.check_step_outputs("matchpf", config) == (True, "4 of 6 entries (66.7 %) have PF statistics")
    # robustmcs
    robust = _stats_path(config, "mcs_tracks_robust_")
    assert "does not exist" in so.check_step_outputs("robustmcs", config)[1]
    _write_nc(robust, dims, {"track_duration": (tk, [2, 2, 2])})
    assert "variables ['pf_mcsstatus', 'pf_lifetime'] are missing" in so.check_step_outputs("robustmcs", config)[1]
    _write_nc(robust, dims, {"track_duration": (tk, [2, 2, 2]), "pf_mcsstatus": (tt, np.ones((3, 2))), "pf_lifetime": (tk, [1, 2, 3])})
    assert so.check_step_outputs("robustmcs", config) == (True, "3 robust MCS tracks")
    _write_nc(robust, {"tracks": 0, "times": 2, "nmaxpf": 2}, {"track_duration": (tk, []), "pf_mcsstatus": (tt, np.zeros((0, 2))),
                                                               "pf_lifetime": (tk, [])})
    assert not so.check_step_outputs("robustmcs", config)[0]                        # no robust MCS is an error, not an empty success


def test_tbpf_checks_of_the_row_folders(run):
    info, config, _ = run
    config = _tbpf(config, stats_format="rows")
    os.makedirs(config["stats_outpath"], exist_ok=True)
    pf3 = np.arange(12, dtype=float).reshape(6, 2)
    # identifymcs
    folder = so.stats_output(config, "mcs_tracks_")
    assert "does not exist" in so.check_step_outputs("identifymcs", config)[1]
    _write_rows_tables(folder, {"track_duration": [2, 2, 2]}, {"area": np.arange(6)})
    assert "variables ['mcs_status', 'ccs_area'] are missing" in so.check_step_outputs("identifymcs", config)[1]
    _write_rows_tables(folder, {"track_duration": [2, 2, 2]}, {"mcs_status": np.ones(6), "ccs_area": np.ones(6)})
    assert so.check_step_outputs("identifymcs", config) == (True, "3 MCS tracks")
    # matchpf: pf_area is a small [tracks, times, n] variable of the row folder
    pf = so.stats_output(config, "mcs_tracks_pf_")
    _write_rows_tables(pf, {"track_duration": [2, 2, 2]}, {"total_rain": np.full(6, np.nan), "pf_npf": np.zeros(6)}, entries3d={"pf_area": pf3})
    ok, message = so.check_step_outputs("matchpf", config)
    assert not ok and message == "0 of 6 entries (0.0 %) have PF statistics"
    _write_rows_tables(pf, {"track_duration": [2, 2, 2]}, {"total_rain": [1.0, 0, np.nan, 2, 3, np.nan], "pf_npf": np.zeros(6)},
                       entries3d={"pf_area": pf3})
    assert so.check_step_outputs("matchpf", config) == (True, "4 of 6 entries (66.7 %) have PF statistics")
    _write_rows_tables(pf, {"track_duration": [2, 2, 2]}, {"total_rain": np.ones(6), "pf_npf": np.zeros(6)})
    assert "variables ['pf_area'] are missing" in so.check_step_outputs("matchpf", config)[1]
    # robustmcs
    robust = so.stats_output(config, "mcs_tracks_robust_")
    _write_rows_tables(robust, {"track_duration": [2, 2, 2], "pf_lifetime": [1, 2, 3]}, {"pf_mcsstatus": np.ones(6)})
    assert so.check_step_outputs("robustmcs", config) == (True, "3 robust MCS tracks")
    # speed and mapfeature are the checks of the radar pipeline, rows-aware
    final = so.stats_output(config, "mcs_tracks_final_")
    _write_rows_tables(final, {"track_duration": [2, 2, 2]}, {"movement_speed": [1.0, np.nan] * 3})
    assert so.check_step_outputs("speed", config)[0]


def test_tbpf_steps_without_a_check_are_never_skipped_and_pass_after_they_ran(run):
    info, config, _ = run
    config = _tbpf(config, skip_completed_steps=True)
    for step in ("preprocess", "remap_healpix", "regrid_mask"):
        assert so.check_step_outputs(step, config, strict=True)[0] is False               # the skip question: no
        assert so.check_step_outputs(step, config, strict=False)[0] is True               # the verification after the run: yes
    calls = []
    state = {"stale": False}
    for step in ("preprocess", "remap_healpix", "regrid_mask"):
        so.run_step(step, dict(config, **{f"run_{step}": True}), state, lambda cfg, step=step: calls.append(step))
    assert calls == ["preprocess", "remap_healpix", "regrid_mask"]
    # a radar config does not know these steps
    with pytest.raises(ValueError):
        so.check_step_outputs("matchpf", dict(config, feature_type="radar_feature"))
    # after an empty period the Tb+PF steps are skipped like the radar steps
    for step in ("matchpf", "robustmcs", "remap_healpix", "regrid_mask"):
        assert so.run_step(step, dict(config, **{f"run_{step}": True}), {"stale": False, "empty": True}, lambda cfg: calls.append("x")) is None
    assert "x" not in calls


def test_idfeature_driver_writes_the_manifest_for_tbpf_so_that_a_skipped_frame_does_not_fail_the_check(run, monkeypatch):
    info, config, tmp_path = run
    names = [info["names"][f] for f in info["frames"]]
    skipped_frame = 5
    config = _tbpf(config, run_parallel=0, input_format="netcdf", area_method="fixed")
    os.remove(f"{config['tracking_outpath']}{names[skipped_frame]}")                 # the skipped frame has no cloudid file
    # idclouds_tbpf returns the cloudid file name, or None when the frame was skipped (miss_thresh), as the real function does
    calls = []

    def fake_idclouds(input_file, cfg):
        calls.append(os.path.basename(input_file))
        return None if len(calls) == skipped_frame + 1 else input_file

    monkeypatch.setattr("pyflextrkr.idclouds_tbpf.idclouds_tbpf", fake_idclouds)
    from pyflextrkr.idfeature_driver import idfeature_driver
    idfeature_driver(config)
    assert len(calls) == len(names)
    ok, message = so.check_step_outputs("idfeature", config, strict=True)
    assert ok and "(1 frames skipped by miss_thresh)" in message
