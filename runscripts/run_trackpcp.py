import logging
import argparse
import yaml
import dask
from dask.distributed import Client, LocalCluster
from pyflextrkr.ft_utilities import load_config, setup_logging
from pyflextrkr.trackpcp_driver import compute_track_precip, compute_track_pixel_accum, compute_class_maps

"""
Post-process a finished tracking with a gridded variable (e.g. hourly precipitation) that is not part of the tracking:

  step 1  statistics of the variable inside every track (added to a new copy of the track statistics file, and a Parquet table)
  step 2  accumulated depth at every pixel of every track + per-track metrics (Parquet)
  step 3  accumulated precipitation / counts per feature class on the grid (NetCDF, one month or the whole period)

Usage:
    python run_trackpcp.py tracking_config.yml trackpcp_config.yml [--steps 1 2 3] [--year Y --month M] [--scheduler FILE]

tracking_config.yml is the config of the tracking run (paths, dates); trackpcp_config.yml holds the trackpcp_* keys
(see pyflextrkr/trackpcp_driver.py and config/config_mrms_trackpcp_example.yml). Keys of the second file override the first.
run_parallel / nprocesses are read from the merged config: 0 serial, 1 LocalCluster, 2 an existing scheduler.
"""

if __name__ == '__main__':
    setup_logging()
    logger = logging.getLogger(__name__)
    ap = argparse.ArgumentParser(description="Track-matched precipitation statistics, pixel accumulations and class maps")
    ap.add_argument("tracking_config")
    ap.add_argument("trackpcp_config")
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--year", type=int, default=None, help="step 3: one calendar month of the tracking period")
    ap.add_argument("--month", type=int, default=None)
    ap.add_argument("--scheduler", default=None, help="Dask scheduler file (run_parallel 2)")
    args = ap.parse_args()

    config = load_config(args.tracking_config)
    with open(args.trackpcp_config) as f:
        config.update(yaml.full_load(f))

    run_parallel = config.get("run_parallel", 0)
    client = None
    if run_parallel == 1:
        dask.config.set({'temporary-directory': config.get("dask_tmp_dir", "./")})
        dask.config.set({'distributed.scheduler.allowed-failures': 3})
        cluster = LocalCluster(n_workers=config.get('nprocesses', 4), threads_per_worker=1)
        client = Client(cluster)
        client.run(setup_logging)
    elif run_parallel == 2:
        client = Client(scheduler_file=args.scheduler)
        client.run(setup_logging)
    else:
        logger.info("Running in serial.")

    if 1 in args.steps:
        compute_track_precip(config)
    if 2 in args.steps:
        compute_track_pixel_accum(config)
    if 3 in args.steps:
        compute_class_maps(config, year=args.year, month=args.month)

    if client is not None:
        client.close()
