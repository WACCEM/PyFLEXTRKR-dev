import os
import sys
import logging
import dask
from dask.distributed import Client, LocalCluster
from pyflextrkr.ft_utilities import load_config, setup_logging
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

    # Step 1 - Identify features
    if config.get('run_idfeature', False):
        idfeature_driver(config)

    # Step 2 - Link features in time adjacent files
    if config.get('run_tracksingle', False):
        tracksingle_driver(config)

    # Step 3 - Track features through the entire dataset
    if config.get('run_gettracks', False):
        tracknumbers_filename = gettracknumbers(config)

    # Step 4 - Calculate track statistics
    if config.get('run_trackstats', False):
        trackstats_filename = trackstats_driver(config)

    # Step 5 - Identify MCS from radar statistics (MCS criteria, feature classes, merge/split folding)
    # (writes mcsradarstats_filebase, like the Tb pipelines' mcstbstats_filebase)
    if config.get('run_identifymcs', False):
        mcsstats_filename = identifymcs_radar(config)

    # Step 6 - Recompute shape and extremes on the merged mask (own feature + folded merge/split fragments),
    # then stage the lifecycle (reads mcsradarstats_filebase, writes mcsrobust_filebase)
    if config.get('run_regionprops', False):
        robuststats_filename = compute_regionprops(config)

    # Step 7 - Map tracking to pixel files
    # (same file naming as the Tb+PF pipelines: reads mcsrobust_filebase)
    if config.get('run_mapfeature', False):
        robust_file = f"{config['stats_outpath']}{config['mcsrobust_filebase']}{config['startdate']}_{config['enddate']}.nc"
        if not os.path.isfile(robust_file):
            logger.critical(f"{robust_file} not found: run_regionprops writes it from the run_identifymcs output. "
                            f"Tracking will now exit.")
            sys.exit(1)
        mapfeature_driver(config, trackstats_filebase=config['mcsrobust_filebase'])

    # Step 8 - Movement speed calculation
    # (reads mcsrobust_filebase, writes mcsfinal_filebase)
    if config.get('run_speed', False):
        movement_speed(config)

    # Clean up resources
    if run_parallel >= 1:
        logger.info("Shutting down Dask client...")
        client.close()
        logger.info("Dask client shutdown complete.")
