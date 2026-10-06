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
from pyflextrkr.mapfeature_driver import mapfeature_driver

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

    # Step 5 - Identify MCS from radar statistics (lifecycle, merge/split folding)
    if config.get('run_identifymcs', False):
        mcsstats_filename = identifymcs_radar(config)

    # Step 6 - Map tracking to pixel files
    if config.get('run_mapfeature', False):
        mapfeature_driver(config, trackstats_filebase=config['finalstats_filebase'])

    # Clean up resources
    if run_parallel >= 1:
        logger.info("Shutting down Dask client...")
        client.close()
        logger.info("Dask client shutdown complete.")
