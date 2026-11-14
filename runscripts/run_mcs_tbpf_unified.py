"""
Run MCS tracking from brightness temperature (Tb) and precipitation (PF), feature_type: tb_pf. This script supersedes the run_mcs_tbpf*.py
scripts (they stay for old workflows): one script, one config, the differences between them are config keys.

Which MCS definition (the key robustmcs_definition, required when run_robustmcs is on; there is no default):

    mcsmip   MCSMIP / SAAG: a PF with a major axis in [mcs_pf_majoraxis_thresh, max_pf_majoraxis_thresh] for mcs_pf_durationthresh hours,
             a maximum rain rate above heavy_rainrate_thresh for that long and a rain volume above mcs_min_rainvol_thresh for
             mcs_volrain_duration_thresh hours (pyflextrkr/robustmcspf_saag.py).      Replaces run_mcs_tbpf_mcsmip.py, run_mcs_tbpf_saag.py.
    feng21   Feng et al. (2021): the PF area, rain rate, skewness and heavy rain ratio above values fitted to the duration (coefs_pf_area,
             coefs_pf_rr, coefs_pf_skew, coefs_pf_heavyratio) (pyflextrkr/robustmcspf.py).
                                                                                      Replaces run_mcs_tbpf.py, run_mcs_tbpf_wrf.py, run_mcs_tbpf_gpm.py.

What replaces the other differences of the old scripts: the WRF preprocessing (run_preprocess) and the regridding of the mask to the native
grid (run_regrid_mask) are optional steps; the remapping to HEALPix (run_remap_healpix) is an optional step; the year that run_mcs_tbpf_gpm.py
added to the paths is clouddata_subdir_format (e.g. '%Y') and a root_path per year.

Steps, each with its run_<step> flag: [preprocess] idfeature, tracksingle, gettracks, trackstats, identifymcs, matchpf, robustmcs, mapfeature,
speed, and the optional [rainmap] (monthly maps of the total and the MCS precipitation), [trackpcp] (precipitation accumulated at every pixel of
every MCS), [remap_healpix], [regrid_mask]. Every step runs through run_step (pyflextrkr/step_outputs.py): its wall time is logged (with
step_profile_dir also the step ledger), its outputs are verified afterwards (verify_step_outputs, default True: the run stops with exit code 1
if they are not complete), and with skip_completed_steps a step whose outputs are complete is not run again (the later steps then run in any
case). A step that ends the run with sys.exit() without a message (the old modules do that when they find nothing, e.g. no MCS) ends it with
exit code 1.

The statistics are dense netCDF files (stats_format: dense, the default, the old modules) or row tables (stats_format: rows, pyflextrkr/rowstats.py:
a Parquet folder per file whose size follows the number of entries, not tracks x longest track x slots).

Usage:
    python run_mcs_tbpf_unified.py config.yml                   # serial or LocalCluster (run_parallel: 0 or 1)
    python run_mcs_tbpf_unified.py config.yml scheduler_file    # Dask distributed (run_parallel: 2)

Author: Zhe Feng | zhe.feng@pnnl.gov
"""
import functools
import logging
import os
import sys

import dask
from dask.distributed import Client, LocalCluster

from pyflextrkr.ft_utilities import load_config, setup_logging
from pyflextrkr.gettracks import gettracknumbers
from pyflextrkr.idfeature_driver import idfeature_driver
from pyflextrkr.identifymcs import identifymcs_tb
from pyflextrkr.mapfeature_driver import mapfeature_driver
from pyflextrkr.matchtbpf_driver import match_tbpf_tracks
from pyflextrkr.movement_speed import movement_speed
from pyflextrkr.robustmcspf import define_robust_mcs_pf as robustmcs_feng21
from pyflextrkr.robustmcspf_saag import define_robust_mcs_pf as robustmcs_mcsmip
from pyflextrkr.step_outputs import rows_mode, run_step, stats_output
from pyflextrkr.tracksingle_driver import tracksingle_driver
from pyflextrkr.trackstats_driver import trackstats_driver

# The robust MCS definitions, the old scripts each one replaces and the config keys it needs (the text of the start-up error)
DEFINITION_HELP = (
    "robustmcs_definition chooses the robust MCS definition (there is no default):\n"
    "  mcsmip   MCSMIP / SAAG: PF major axis and duration, heavy rain rate duration, rain volume "
    "(replaces run_mcs_tbpf_mcsmip.py and run_mcs_tbpf_saag.py); keys: mcs_pf_majoraxis_thresh, max_pf_majoraxis_thresh, "
    "mcs_pf_durationthresh, mcs_pf_majoraxis_for_lifetime, mcs_pf_gap, heavy_rainrate_thresh, mcs_min_rainvol_thresh, "
    "mcs_volrain_duration_thresh\n"
    "  feng21   Feng et al. (2021): PF area, rain rate, skewness and heavy rain ratio fitted to the duration "
    "(replaces run_mcs_tbpf.py, run_mcs_tbpf_wrf.py and run_mcs_tbpf_gpm.py); keys: mcs_pf_majoraxis_thresh, max_pf_majoraxis_thresh, "
    "mcs_pf_durationthresh, mcs_pf_majoraxis_for_lifetime, mcs_pf_gap, coefs_pf_area, coefs_pf_rr, coefs_pf_skew, coefs_pf_heavyratio")


def fail_on_silent_exit(step, func):
    """
    Wrap the function of a step so that a sys.exit() without an error code ends the run with exit code 1.

    The old modules call sys.exit() when they find nothing (no files, no MCS): the process then ends with exit code 0, a Slurm job is
    COMPLETED and the jobs that wait for it start. sys.exit with a message or a code other than 0 is passed on as it is.

    Args:
        step: string
            Name of the step (for the message).
        func: callable
            func(config, **kwargs) runs the step.

    Returns:
        wrapped: callable
            Same arguments and return value as func; func is wrapped.__wrapped__.
    """
    @functools.wraps(func)
    def wrapped(config, **kwargs):
        try:
            return func(config, **kwargs)
        except SystemExit as err:
            # sys.exit() or sys.exit(0) after the step found nothing to do: that is a failed step, not a finished one
            if err.code in (None, 0):
                logging.getLogger(__name__).critical(f"Step {step} ended the run without making its outputs. Tracking will now exit.")
                sys.exit(1)
            raise
    return wrapped


def check_config(config):
    """
    Check the config before any work is done (and before any queue wait is wasted): the feature type, stats_format and the robust MCS
    definition with its keys.

    Args:
        config: dictionary
            Loaded config.

    Raises:
        SystemExit: with a message, if the config cannot be used.
    """
    logger = logging.getLogger(__name__)
    # This script is for Tb+PF tracking: the step checks and the row modules of the other feature types are different
    feature_type = str(config.get("feature_type"))
    if "tb_pf" not in feature_type:
        sys.exit(f"run_mcs_tbpf_unified.py is for Tb+PF tracking (feature_type containing tb_pf), not '{feature_type}'. "
                 f"The radar_feature pipeline runs with run_mcs_radar3d.py.")
    # A typo in stats_format would silently be the dense format
    stats_format = config.get("stats_format", "dense")
    if stats_format not in ("dense", "rows"):
        sys.exit(f"stats_format must be 'dense' or 'rows', not '{stats_format}'")
    # The robust MCS definition: required, with its keys, when the step runs; a wrong value is an error whenever it is given
    from pyflextrkr.robustmcs_rows import DEFINITIONS, robustmcs_params
    definition = config.get("robustmcs_definition")
    if config.get("run_robustmcs", False):
        try:
            robustmcs_params(config)
        except ValueError as err:
            sys.exit(f"{err}\n{DEFINITION_HELP}")
    elif definition is not None and definition not in DEFINITIONS:
        sys.exit(f"robustmcs_definition must be one of {DEFINITIONS}, not {definition!r}\n{DEFINITION_HELP}")
    if definition in DEFINITIONS:
        logger.info(f"Robust MCS definition: {definition}; statistics format: {stats_format}")


def build_steps(config):
    """
    The steps of the pipeline, in order, with the function of each: the old modules for stats_format: dense, the row modules for
    stats_format: rows, and the robust MCS function of robustmcs_definition. Every function is wrapped by fail_on_silent_exit.

    Args:
        config: dictionary
            Loaded config (stats_format, robustmcs_definition, mcsrobust_filebase and the optional run_* flags).

    Returns:
        steps: list of tuples
            (step name, function, keyword arguments of the function) for idfeature ... speed and the optional steps rainmap, trackpcp,
            remap_healpix and regrid_mask. Whether a step runs is decided by its run_<step> flag in run_step.
    """
    definition = config.get("robustmcs_definition")
    # The functions of the MCS steps. The row modules are imported here, not at the top: the Dask workers import this file too
    if rows_mode(config):
        from pyflextrkr.identifymcs_tb_rows import identifymcs_tb_rows as identifymcs_step
        from pyflextrkr.matchtbpf_rows import match_tbpf_tracks_rows as matchpf_step
        from pyflextrkr.robustmcs_rows import define_robust_mcs_pf_rows as robustmcs_step
        from pyflextrkr.mapfeature_rows import mapfeature_driver_rows as mapfeature_step
        from pyflextrkr.movement_speed_rows import movement_speed_rows as speed_step
    else:
        identifymcs_step, matchpf_step = identifymcs_tb, match_tbpf_tracks
        robustmcs_step = robustmcs_mcsmip if definition == "mcsmip" else robustmcs_feng21
        mapfeature_step, speed_step = mapfeature_driver, movement_speed
    steps = [
        ("idfeature", idfeature_driver, {}),
        ("tracksingle", tracksingle_driver, {}),
        ("gettracks", gettracknumbers, {}),
        ("trackstats", trackstats_driver, {}),
        ("identifymcs", identifymcs_step, {}),
        ("matchpf", matchpf_step, {}),
        ("robustmcs", robustmcs_step, {}),
        # The pixel-level tracking files are made from the robust MCS statistics
        ("mapfeature", mapfeature_step, {"trackstats_filebase": config["mcsrobust_filebase"]}),
        ("speed", speed_step, {}),
    ]
    # The optional steps are imported only when their flag is on, so that a run without them is unchanged (and needs nothing of them)
    if config.get("run_rainmap", False):
        from pyflextrkr.rainmap_driver import rainmap_driver
        steps.append(("rainmap", rainmap_driver, {}))
    if config.get("run_trackpcp", False):
        from pyflextrkr.trackpcp_driver import trackpcp_driver
        steps.append(("trackpcp", trackpcp_driver, {}))
    if config.get("run_remap_healpix", False):
        steps.append(("remap_healpix", remap_healpix, {}))
    if config.get("run_regrid_mask", False):
        from pyflextrkr.regrid_tracking_mask import regrid_tracking_mask
        steps.append(("regrid_mask", regrid_tracking_mask, {}))
    return [(step, fail_on_silent_exit(step, func), kwargs) for step, func, kwargs in steps]


def remap_healpix(config):
    """
    Remap the MCS mask to the HEALPix grid (remap_to_healpix_zarr).

    The tasks of this step are rechunked with the "tasks" method; the setting is scoped to this step so that the earlier steps keep the
    default rechunking.

    Args:
        config: dictionary
            Loaded config.

    Returns:
        The return value of remap_to_healpix_zarr.
    """
    from pyflextrkr.remap_healpix_zarr import remap_to_healpix_zarr
    with dask.config.set({"array.rechunk.method": "tasks"}):
        return remap_to_healpix_zarr(config)


def start_dask_client(config, argv):
    """
    Start the Dask client of the run: a LocalCluster (run_parallel: 1) or a connection to the scheduler of the scheduler file given as
    the second argument (run_parallel: 2); None in serial (run_parallel: 0).

    Args:
        config: dictionary
            Loaded config (run_parallel, nprocesses, dask_tmp_dir, timeout).
        argv: list of strings
            Command line; argv[2] is the scheduler file of run_parallel: 2.

    Returns:
        client: dask.distributed.Client or None
    """
    logger = logging.getLogger(__name__)
    run_parallel = config.get('run_parallel', 0)
    if run_parallel == 1:
        # Set Dask temporary directory for workers and the retry behavior for transient errors
        dask.config.set({'temporary-directory': config.get("dask_tmp_dir", "./")})
        dask.config.set({'distributed.scheduler.allowed-failures': 3})
        # Local cluster; report the requested and the connected workers for easier diagnostics on shared systems
        cluster = LocalCluster(n_workers=config['nprocesses'], threads_per_worker=1)
        client = Client(cluster)
        client.run(setup_logging)
        client.wait_for_workers(config["nprocesses"], timeout=60)
        logger.info(f"LocalCluster requested n_workers={config['nprocesses']}, actual connected workers="
                    f"{len(client.scheduler_info().get('workers', {}))}")
        return client
    if run_parallel == 2:
        # Dask scheduler: the scheduler file is the second argument; the connection has a timeout
        scheduler_file = argv[2]
        timeout = config.get("timeout", 600)
        dask.config.set({'distributed.scheduler.allowed-failures': 3})
        logger.info(f"Connecting to Dask scheduler at {scheduler_file} with timeout {timeout}")
        try:
            client = Client(scheduler_file=scheduler_file, timeout=timeout)
            client.run(setup_logging)
        except Exception as err:
            logger.error(f"Failed to connect to Dask scheduler: {err}")
            sys.exit(1)
        logger.info(f"Connected scheduler currently has {len(client.scheduler_info().get('workers', {}))} workers")
        return client
    logger.info("Running in serial.")
    return None


def run_steps(config, client, steps, state):
    """
    Run the steps in order through run_step.

    Args:
        config: dictionary
            Loaded config.
        client: dask.distributed.Client or None
            Client of the run (for the timing record).
        steps: list of tuples
            (step name, function, keyword arguments), see build_steps.
        state: dictionary
            {'stale': bool}, shared by the steps of the run (see run_step).
    """
    logger = logging.getLogger(__name__)
    # Loop over the steps in order; a step that is off, skipped or after an empty period is not run (run_step decides)
    for step, func, kwargs in steps:
        # The pixel-level files are made from the robust MCS statistics: stop before the step if the step that makes them did not run
        if step == "mapfeature" and config.get('run_mapfeature', False) and not state.get('empty'):
            robust_file = stats_output(config, config['mcsrobust_filebase'])
            if not os.path.exists(robust_file):
                logger.critical(f"{robust_file} not found: run_robustmcs writes it from the run_matchpf output. Tracking will now exit.")
                sys.exit(1)
        run_step(step, config, state, func, client, **kwargs)


def main(argv=None):
    """
    Run the pipeline of the config file in argv[1].

    Args:
        argv: list of strings (default: None = sys.argv)
            argv[1] config file, argv[2] the scheduler file for run_parallel: 2.
    """
    argv = sys.argv if argv is None else argv
    setup_logging()
    logger = logging.getLogger(__name__)
    config = load_config(argv[1])
    # Everything that can be wrong with the config is reported now
    check_config(config)
    steps = build_steps(config)
    state = {'stale': False}

    # Step 0 - Optional: preprocess WRF output to Tb and rain rate (before the Dask client, as run_mcs_tbpf_wrf.py does)
    if config.get('run_preprocess', False):
        from pyflextrkr.preprocess_wrf_tb_rainrate import preprocess_wrf_tb_rainrate
        run_step('preprocess', config, state, fail_on_silent_exit('preprocess', preprocess_wrf_tb_rainrate))

    # Parallel processing options: client stays None in serial; it is passed to run_step for the worker count and the Dask report
    client = start_dask_client(config, argv)

    # Steps 1 to 9 and the optional ones
    run_steps(config, client, steps, state)

    # Clean up resources
    if client is not None:
        logger.info("Shutting down Dask client...")
        client.close()
        logger.info("Dask client shutdown complete.")


if __name__ == '__main__':
    main()
