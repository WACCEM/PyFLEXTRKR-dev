"""
Are the outputs of a pipeline step complete? Used by run_mcs_radar3d.py in two ways (config keys, both optional):

  verify_step_outputs: True (default)   after every step, its outputs are checked and the run stops with exit code 1 if they are
                                        not complete. A Slurm job whose last step failed without an error message (the speed step
                                        of June 2025 ended in a lost worker and the job still showed COMPLETED) now fails.
  skip_completed_steps: False           a step whose outputs are already complete is not run again; the first step that has to
                                        run makes all later steps run (they depend on its output). With it a job that ran out of time or
                                        failed is continued by submitting it again, without editing the run_* flags.

The checks look at the files of the run (names from the config), not at the logs: cloudid files against the input files, the
pairwise track files against the pairs that tracksingle makes, the last file of the track numbers, the statistics files, the
attributes of the robust file (n_pairs_not_computed, n_pairs_area_mismatch), the pixel files and the speed variables.

Frames without features and skipped frames: idfeature_radar3d writes a cloudid file for a frame without features (it ends the tracks,
a missing file would look like a data gap); a frame skipped by the optional miss_thresh guard has no file. idfeature_driver records
the number of skipped frames in idfeature_manifest.json (radar_feature), so the skip decision "one cloudid file per input file" counts
them. A period in which no cloudid file has a feature has no tracks: trackstats writes an empty file, the checks pass with a warning
and the steps after trackstats are skipped (state["empty"]).

Author: Zhe Feng | zhe.feng@pnnl.gov
"""
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
from netCDF4 import Dataset

from pyflextrkr.ft_utilities import subset_files_timerange
from pyflextrkr.step_timing import timed_step

STEPS = ("idfeature", "tracksingle", "gettracks", "trackstats", "identifymcs", "regionprops", "mapfeature", "speed", "trackpcp")


def track_file_name(new_basetime, filebase="track_"):
    """
    Name of the pairwise track file of a pair, the way tracksingle names it: <filebase><date>_<time of the NEW file>.nc

    Args:
        new_basetime: int
            Epoch time [s] of the second (new) file of the pair.
        filebase: string (default: 'track_')
            Prefix of the track files.

    Returns:
        name: string
            e.g. 'track_20250601_000040.nc'.
    """
    stamp = pd.to_datetime(new_basetime, unit="s")
    return f"{filebase}{stamp.strftime('%Y%m%d')}_{stamp.strftime('%H%M%S')}.nc"


def find_missing_pairs(cloudid_files, cloudid_basetimes, existing_names, timegap_hours, filebase="track_"):
    """
    Pairs of consecutive cloudid files that should have a track file but have none.

    Args:
        cloudid_files: list of strings
            Cloudid files in time order.
        cloudid_basetimes: array-like of int
            Epoch time [s] of each file.
        existing_names: set of strings
            File names that exist in the tracking folder.
        timegap_hours: float
            Pairs further apart than this get no track file (tracksingle would not link them either).
        filebase: string (default: 'track_')
            Prefix of the track files.

    Returns:
        pairs: list of tuples
            (reference file, new file, reference basetime, new basetime) of each missing pair, in time order.
    """
    times = np.asarray(cloudid_basetimes)
    missing = []
    for i in range(len(cloudid_files) - 1):
        hours = (times[i + 1] - times[i]) / 3600.0
        if 0 < hours < timegap_hours and track_file_name(times[i + 1], filebase) not in existing_names:
            missing.append((cloudid_files[i], cloudid_files[i + 1], int(times[i]), int(times[i + 1])))
    return missing


def _stats_file(config, filebase):
    """
    Path of a statistics file: <stats_outpath><filebase><startdate>_<enddate>.nc
    """
    return f"{config['stats_outpath']}{filebase}{config['startdate']}_{config['enddate']}.nc"


def rows_mode(config):
    """
    Are the statistics of the MCS steps (identifymcs, regionprops, speed) row tables (config stats_format: rows)?

    Args:
        config: dictionary
            Loaded config.

    Returns:
        rows: bool
            False for the dense netCDF files (stats_format: dense, the default).
    """
    return config.get("stats_format", "dense") == "rows"


def stats_output(config, filebase):
    """
    Path of the statistics file or row folder of a step: <stats_outpath><filebase><startdate>_<enddate> plus .nc (dense) or
    .rows (stats_format: rows).

    Args:
        config: dictionary
            Loaded config.
        filebase: string
            File base name (mcsradarstats_filebase, mcsrobust_filebase or mcsfinal_filebase).

    Returns:
        path: string
    """
    return f"{config['stats_outpath']}{filebase}{config['startdate']}_{config['enddate']}" + (".rows" if rows_mode(config) else ".nc")


def _open_rows(folder):
    """
    Open a row folder for a check; (None, message) if it does not exist or cannot be read.
    """
    if not os.path.isdir(folder):
        return None, f"{os.path.basename(folder)} does not exist"
    try:
        from pyflextrkr.rowstats import RowStats
        return RowStats.open(folder), ""
    except (OSError, ValueError, KeyError) as err:
        return None, f"{os.path.basename(folder)} cannot be read ({err})"


def _cloudid_files(config):
    """
    Cloudid files of the period and their times.

    Returns:
        files, times: list of strings, numpy array of int
    """
    files, times, _, _ = subset_files_timerange(config["tracking_outpath"], config["cloudid_filebase"],
                                                config["start_basetime"], config["end_basetime"])
    return files, np.asarray(times)


def _open(path):
    """
    Open a netCDF file for a check; (None, message) if it does not exist or cannot be read.
    """
    if not os.path.isfile(path):
        return None, f"{os.path.basename(path)} does not exist"
    try:
        return Dataset(path), ""
    except OSError as err:
        return None, f"{os.path.basename(path)} cannot be read ({err})"


def idfeature_manifest_path(config):
    """
    Path of the manifest that idfeature_driver writes (radar_feature): <tracking_outpath>idfeature_manifest.json
    """
    return os.path.join(config["tracking_outpath"], "idfeature_manifest.json")


def write_idfeature_manifest(config, n_inputs, returned):
    """
    Record how many input frames idfeature wrote a cloudid file for and how many it skipped (miss_thresh), so that the check
    "one cloudid file per input file" can count the skipped frames.

    Args:
        config: dictionary
            Loaded config (tracking_outpath, miss_thresh, start_basetime, end_basetime).
        n_inputs: int
            Number of input files (calls of the feature identification function).
        returned: list
            What the calls returned: the cloudid file name(s), or None when every time of the call was skipped.
    """
    skipped = sum(1 for r in returned if r is None)
    as_number = lambda v: None if v is None else float(v)
    manifest = {"n_inputs": int(n_inputs), "n_calls_written": len(returned) - skipped, "n_calls_skipped": skipped,
                "miss_thresh": as_number(config.get("miss_thresh")), "start_basetime": as_number(config.get("start_basetime")),
                "end_basetime": as_number(config.get("end_basetime"))}
    with open(idfeature_manifest_path(config), "w") as f:
        json.dump(manifest, f)


def read_idfeature_manifest(config, n_inputs):
    """
    The manifest of idfeature if it belongs to this period.

    Args:
        config: dictionary
            Loaded config.
        n_inputs: int
            Number of input files in the period now.

    Returns:
        manifest: dictionary or None
            None if there is none, it cannot be read, or it was written for another period or number of inputs.
    """
    try:
        with open(idfeature_manifest_path(config)) as f:
            manifest = json.load(f)
    except (OSError, ValueError):
        return None
    same = lambda a, b: (a is None and b is None) or (a is not None and b is not None and float(a) == float(b))
    if manifest.get("n_inputs") != n_inputs or not same(manifest.get("start_basetime"), config.get("start_basetime")) \
            or not same(manifest.get("end_basetime"), config.get("end_basetime")):
        return None
    return manifest


def period_has_no_features(config):
    """
    Do all the cloudid files of the period have zero features? (A period or a small domain without any feature has no tracks.)

    Only the cloudid files are read, one after the other, until one has a feature, so a normal period costs one file.

    Args:
        config: dictionary
            Loaded config (nfeature_varname, default 'nfeatures').

    Returns:
        empty: bool
            True if there is at least one cloudid file and none has a feature; False if one has a feature or if a file cannot be
            read (cannot tell).
    """
    try:
        files, _ = _cloudid_files(config)
    except OSError:
        return False
    if not files:
        return False
    name = config.get("nfeature_varname", "nfeatures")
    for path in files:
        try:
            with Dataset(path) as ds:
                if int(np.max(np.ma.filled(ds[name][:], 0))) > 0:
                    return False
        except (OSError, KeyError, IndexError, ValueError):
            return False
    return True


def trackstats_ntracks(config):
    """
    Number of tracks in the sparse trackstats file, None if the file does not exist or cannot be read.
    """
    ds, _ = _open(_stats_file(config, config["trackstats_sparse_filebase"]))
    if ds is None:
        return None
    with ds:
        return ds.dimensions[config.get("tracks_dimname", "tracks")].size


EMPTY_NOTE = "no tracks: none of the cloudid files of the period has a feature"


def _check_idfeature(config, strict):
    files, _ = _cloudid_files(config)
    n_input = len(subset_files_timerange(
        config["clouddata_path"], config.get("databasename", ""), config["start_basetime"], config["end_basetime"],
        time_format=config.get("time_format", "yyyymodd_hhmmss"), subdir_format=config.get("clouddata_subdir_format"))[0])
    # Frames skipped by the miss_thresh guard have no cloudid file: the manifest of idfeature_driver says how many
    manifest = read_idfeature_manifest(config, n_input)
    skipped = int(manifest["n_calls_skipped"]) if manifest is not None else 0
    message = f"{len(files)} cloudid files for {n_input} input files" + (f" ({skipped} frames skipped by miss_thresh)" if skipped else "")
    # Inputs that cannot be read (or have no feature file) leave gaps, which the step logs; only an empty result is an error
    return (len(files) + skipped == n_input and n_input > 0) if strict else len(files) > 0, message


def _check_tracksingle(config, strict):
    files, times = _cloudid_files(config)
    base = config.get("singletrack_filebase", "track_")
    names = {n for n in os.listdir(config["tracking_outpath"]) if n.startswith(base)}
    missing = find_missing_pairs(files, times, names, config["timegap"], base)
    if missing:
        return False, f"{len(missing)} pair(s) of cloudid files without a track file, the first {os.path.basename(missing[0][1])}"
    return len(files) > 1, f"track files for all {max(len(files) - 1, 0)} pairs of the {len(files)} cloudid files"


def _check_gettracks(config, strict):
    ds, message = _open(_stats_file(config, config["tracknumbers_filebase"]))
    if ds is None:
        return False, message
    with ds:
        ds.set_auto_maskandscale(False)
        last = b"".join(np.asarray(ds["cloudid_files"][-1]).ravel().tolist()).decode().strip("\x00 ")
        n_breaks = int(ds.getncattr("n_chain_breaks")) if "n_chain_breaks" in ds.ncattrs() else 0
        ntracks = int(np.asarray(ds["ntracks"][:]).ravel()[0])
    files, _ = _cloudid_files(config)
    if not files or os.path.basename(files[-1]) != last:
        return False, f"the last file of the track numbers is {last}, the last cloudid file is {os.path.basename(files[-1]) if files else None}"
    if ntracks <= 1 and period_has_no_features(config):
        return True, f"{EMPTY_NOTE}, last file {last}"
    return ntracks > 1, f"{ntracks - 1} track numbers, last file {last}" + (f", {n_breaks} chain break(s)" if n_breaks else "")


def _check_trackstats(config, strict):
    ds, message = _open(_stats_file(config, config["trackstats_sparse_filebase"]))
    if ds is None:
        return False, message
    with ds:
        ntracks = ds.dimensions[config.get("tracks_dimname", "tracks")].size
    # A period without features has no tracks and no dense file: trackstats writes only the empty sparse file
    if ntracks == 0 and period_has_no_features(config):
        return True, EMPTY_NOTE
    if config.get("trackstats_dense_netcdf", 0) == 1 and not os.path.isfile(_stats_file(config, config["trackstats_filebase"])):
        return False, "the dense trackstats file does not exist"
    return ntracks > 0, f"{ntracks} tracks"


def _check_identifymcs(config, strict):
    if rows_mode(config):
        rs, message = _open_rows(stats_output(config, config.get("mcsradarstats_filebase", "mcs_tracks_")))
        if rs is None:
            return False, message
        missing = [v for v in ("is_mcs", "track_class", "track_duration") if v not in rs.tracks]
        ntracks = len(rs.duration)
        return (not missing and ntracks > 0), (f"variables {missing} are missing" if missing else f"{ntracks} tracks")
    ds, message = _open(_stats_file(config, config.get("mcsradarstats_filebase", "mcs_tracks_")))
    if ds is None:
        return False, message
    with ds:
        missing = [v for v in ("is_mcs", "track_class", "track_duration") if v not in ds.variables]
        ntracks = ds.dimensions[config.get("tracks_dimname", "tracks")].size
    return (not missing and ntracks > 0), (f"variables {missing} are missing" if missing else f"{ntracks} tracks")


def _check_regionprops(config, strict):
    if rows_mode(config):
        rs, message = _open_rows(stats_output(config, config["mcsrobust_filebase"]))
        if rs is None:
            return False, message
        attrs = {k: rs.meta["global_attrs"][k] for k in ("n_pairs_not_computed", "n_pairs_area_mismatch") if k in rs.meta["global_attrs"]}
    else:
        ds, message = _open(_stats_file(config, config["mcsrobust_filebase"]))
        if ds is None:
            return False, message
        with ds:
            attrs = {k: ds.getncattr(k) for k in ("n_pairs_not_computed", "n_pairs_area_mismatch") if k in ds.ncattrs()}
    bad = {k: int(v) for k, v in attrs.items() if int(v) != 0}
    if bad or len(attrs) < 2:
        return False, f"attributes {bad or 'n_pairs_*'}: pairs without a merged mask or with a merged mask area different from total_area"
    return True, "all (track, time) pairs have a merged mask"


def _check_mapfeature(config, strict):
    files, _ = _cloudid_files(config)
    pixel, _, _, _ = subset_files_timerange(config["pixeltracking_outpath"], config["pixeltracking_filebase"],
                                            config["start_basetime"], config["end_basetime"]) \
        if os.path.isdir(config["pixeltracking_outpath"]) else ([], 0, 0, 0)
    return (len(pixel) == len(files) and len(files) > 0), f"{len(pixel)} pixel files for {len(files)} cloudid files"


def _check_speed(config, strict):
    if rows_mode(config):
        rs, message = _open_rows(stats_output(config, config["mcsfinal_filebase"]))
        if rs is None:
            return False, message
        if "movement_speed" not in rs.meta["vars_2d"]:
            return False, "movement_speed is not in the file"
        n_finite = int(np.isfinite(np.asarray(rs["movement_speed"].flat, dtype=float)).sum())
        return n_finite > 0, f"{n_finite} finite speeds"
    ds, message = _open(_stats_file(config, config["mcsfinal_filebase"]))
    if ds is None:
        return False, message
    with ds:
        if "movement_speed" not in ds.variables:
            return False, "movement_speed is not in the file"
        n_finite = int(np.isfinite(np.asarray(ds["movement_speed"][:], dtype=float)).sum())
    return n_finite > 0, f"{n_finite} finite speeds"


def _check_trackpcp(config, strict):
    """The files of the steps in trackpcp_steps exist (the names come from trackpcp_driver.expected_outputs)."""
    from pyflextrkr.trackpcp_driver import expected_outputs
    files = expected_outputs(config)
    missing = [f for f in files if not os.path.isfile(f) or os.path.getsize(f) == 0]
    if missing:
        return False, f"{len(missing)} of {len(files)} files are missing or empty, e.g. {missing[0]}"
    return True, f"{len(files)} files"


CHECKS = {"idfeature": _check_idfeature, "tracksingle": _check_tracksingle, "gettracks": _check_gettracks,
          "trackstats": _check_trackstats, "identifymcs": _check_identifymcs, "regionprops": _check_regionprops,
          "mapfeature": _check_mapfeature, "speed": _check_speed, "trackpcp": _check_trackpcp}


def check_step_outputs(step, config, strict=True):
    """
    Are the outputs of a step complete?

    Args:
        step: string
            One of STEPS.
        config: dictionary
            Loaded config (load_config).
        strict: bool (default: True)
            True: the question "can this step be skipped" (idfeature must have a cloudid file for every input file).
            False: the check after a step has run (idfeature may have gaps that it logged).

    Returns:
        ok, message: bool, string
            Whether the outputs are complete and what was found.
    """
    if step not in CHECKS:
        raise ValueError(f"unknown step '{step}', the steps are {STEPS}")
    return CHECKS[step](config, strict)


def run_step(step, config, state, func, client=None, **kwargs):
    """
    Run one pipeline step if its run_<step> flag is on: time it, skip it if its outputs are complete (skip_completed_steps), verify
    its outputs afterwards (verify_step_outputs).

    Args:
        step: string
            One of STEPS.
        config: dictionary
            Loaded config.
        state: dictionary
            {'stale': bool}, shared by the steps of one run: once a step has run, the later steps are not skipped.
        func: callable
            func(config, **kwargs) runs the step.
        client: dask.distributed.Client (default: None)
            Client of the run (for the timing record).
        **kwargs:
            Passed to func.

    Returns:
        result: the return value of func, or None if the step is off or skipped.
    """
    logger = logging.getLogger(__name__)
    if not config.get(f"run_{step}", False):
        return None
    if state.get("empty") and step in STEPS_AFTER_TRACKSTATS:
        logger.warning(f"Step {step} skipped: {EMPTY_NOTE}")
        return None
    if config.get("skip_completed_steps", False) and not state["stale"]:
        ok, message = check_step_outputs(step, config, strict=True)
        if ok:
            logger.info(f"Step {step} skipped: its outputs are complete ({message})")
            _note_empty(step, config, state)
            return None
        logger.info(f"Step {step} runs: its outputs are not complete ({message})")
    state["stale"] = True
    with timed_step(step, config, client):
        result = func(config, **kwargs)
    if config.get("verify_step_outputs", True):
        ok, message = check_step_outputs(step, config, strict=False)
        if not ok:
            logger.critical(f"Step {step} did not make complete outputs: {message}. Tracking will now exit.")
            sys.exit(1)
        logger.info(f"Step {step} outputs verified: {message}")
    _note_empty(step, config, state)
    return result


STEPS_AFTER_TRACKSTATS = ("identifymcs", "regionprops", "mapfeature", "speed", "trackpcp")


def _note_empty(step, config, state):
    """
    After trackstats: a period without any tracked feature has nothing for the later steps; remember it in state["empty"].

    Args:
        step: string
            The step that just finished or was skipped.
        config: dictionary
            Loaded config.
        state: dictionary
            Shared state of the run; gets state["empty"] = True when trackstats made zero tracks and no cloudid file has a feature.
    """
    if step == "trackstats" and trackstats_ntracks(config) == 0 and period_has_no_features(config):
        logging.getLogger(__name__).warning(f"{EMPTY_NOTE}: the steps after trackstats are skipped")
        state["empty"] = True
