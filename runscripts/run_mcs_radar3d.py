import os
import sys
import logging
import dask
from dask.distributed import Client, LocalCluster
from pyflextrkr.ft_utilities import load_config, setup_logging
from pyflextrkr.step_outputs import rows_mode, run_step, stats_output
from pyflextrkr.idfeature_driver import idfeature_driver
from pyflextrkr.tracksingle_driver import tracksingle_driver
from pyflextrkr.gettracks import gettracknumbers
from pyflextrkr.trackstats_driver import trackstats_driver
from pyflextrkr.identifymcs_radar import identifymcs_radar
from pyflextrkr.regionprops_driver import compute_regionprops
from pyflextrkr.mapfeature_driver import mapfeature_driver
from pyflextrkr.movement_speed import movement_speed

"""
Run MCS tracking directly from 3D (or 2D) Cartesian radar reflectivity
(feature_type: radar_feature), e.g. MRMS composite reflectivity regridded to
the HRRR 3-km grid. No Tb, no PF-linking, no SL3D - the MCS definition and
convective/stratiform classification come entirely from the reflectivity
field itself (see pyflextrkr/idfeature_radar3d.py).

Usage:
    python run_mcs_radar3d.py config.yml                  # serial or LocalCluster (run_parallel: 0 or 1)
    python run_mcs_radar3d.py config.yml scheduler_file    # Dask distributed (run_parallel: 2)
"""

if __name__ == '__main__':

    # Set the logging message level
    setup_logging()
    logger = logging.getLogger(__name__)

    # Load configuration file
    config_file = sys.argv[1]
    config = load_config(config_file)

    ################################################################################################
    # Parallel processing options
    # client stays None in serial; it is passed to timed_step for the worker count and the Dask report
    client = None
    run_parallel = config.get('run_parallel', 0)
    if run_parallel == 1:
        # Set Dask temporary directory for workers
        dask_tmp_dir = config.get("dask_tmp_dir", "./")
        dask.config.set({'temporary-directory': dask_tmp_dir})
        # Configure retry behavior for transient errors
        dask.config.set({'distributed.scheduler.allowed-failures': 3})
        # Local cluster
        cluster = LocalCluster(n_workers=config['nprocesses'], threads_per_worker=1)
        client = Client(cluster)
        client.run(setup_logging)
    elif run_parallel == 2:
        # Dask scheduler
        # Get the scheduler filename from input argument
        scheduler_file = sys.argv[2]
        timeout = config.get("timeout", 120)
        # Configure retry behavior for transient errors
        dask.config.set({'distributed.scheduler.allowed-failures': 3})
        client = Client(scheduler_file=scheduler_file)
        client.run(setup_logging)
    else:
        logger.info(f"Running in serial.")

    # Statistics of the MCS steps (identifymcs ... speed): dense netCDF files (stats_format: dense, the default) or row tables
    # (stats_format: rows, pyflextrkr/rowstats.py: a Parquet folder per file, size follows the number of entries, not tracks x
    # longest track x slots). The rows modules are imported here, not at the top, because the Dask workers import this file too.
    if rows_mode(config):
        from pyflextrkr.identifymcs_rows import identifymcs_radar_rows as identifymcs_step
        from pyflextrkr.regionprops_rows import compute_regionprops_rows as regionprops_step
        from pyflextrkr.mapfeature_rows import mapfeature_driver_rows as mapfeature_step
        from pyflextrkr.movement_speed_rows import movement_speed_rows as speed_step
    else:
        identifymcs_step, regionprops_step = identifymcs_radar, compute_regionprops
        mapfeature_step, speed_step = mapfeature_driver, movement_speed

    # Every step runs through run_step (pyflextrkr/step_outputs.py): one INFO line with its wall time (with `step_profile_dir`
    # in the config also the step ledger, memory timeline and profiles, pyflextrkr/step_timing.py), its outputs are verified
    # afterwards (verify_step_outputs, default True: the run stops with exit code 1 if they are not complete), and with
    # skip_completed_steps a step whose outputs are complete is not run again (the later steps then run in any case)
    state = {'stale': False}

    # Step 1 - Identify features
    run_step('idfeature', config, state, idfeature_driver, client)

    # Step 2 - Link features in time adjacent files
    run_step('tracksingle', config, state, tracksingle_driver, client)

    # Step 3 - Track features through the entire dataset
    run_step('gettracks', config, state, gettracknumbers, client)

    # Step 4 - Calculate track statistics
    run_step('trackstats', config, state, trackstats_driver, client)

    # Step 5 - Identify MCS from radar statistics (MCS criteria, feature classes, merge/split folding)
    # (writes mcsradarstats_filebase, like the Tb pipelines' mcstbstats_filebase)
    run_step('identifymcs', config, state, identifymcs_step, client)

    # Step 6 - Recompute shape and extremes on the merged mask (own feature + folded merge/split fragments),
    # then stage the lifecycle (reads mcsradarstats_filebase, writes mcsrobust_filebase)
    run_step('regionprops', config, state, regionprops_step, client)

    # Step 7 - Map tracking to pixel files
    # (same file naming as the Tb+PF pipelines: reads mcsrobust_filebase)
    if config.get('run_mapfeature', False) and not state.get('empty'):
        robust_file = stats_output(config, config['mcsrobust_filebase'])
        if not os.path.exists(robust_file):
            logger.critical(f"{robust_file} not found: run_regionprops writes it from the run_identifymcs output. "
                            f"Tracking will now exit.")
            sys.exit(1)
    run_step('mapfeature', config, state, mapfeature_step, client, trackstats_filebase=config['mcsrobust_filebase'])

    # Step 8 - Movement speed calculation
    # (reads mcsrobust_filebase, writes mcsfinal_filebase)
    run_step('speed', config, state, speed_step, client)

    # Clean up resources
    if run_parallel >= 1:
        logger.info("Shutting down Dask client...")
        client.close()
        logger.info("Dask client shutdown complete.")
