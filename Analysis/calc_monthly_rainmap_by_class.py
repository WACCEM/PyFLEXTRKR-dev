"""
Monthly precipitation maps by tracked feature class (MCS, multi-cell, isolated cell, non-convective, ... : the classes
are read from the long_name of the class variable of the track statistics file, nothing is hard-coded).

For every month of tracking (one tracking config per month, as for the other Analysis scripts) and a precipitation data set
that is not part of the tracking (e.g. hourly MRMS QPE, own files and time axis), writes one NetCDF file with, per class:
  precip         accumulated precipitation while the pixel is in a feature of the class [mm]
  cloud_count    number of data windows in which the pixel is in a feature of the class
  precip_count   number of those windows with precipitation >= pcp_thresh (default 1 mm/h)
  precip_conv, precip_strat   the same accumulation split into the convective swath and the rest
and total_precip, total_precip_count, ntimes (windows in the month) for all precipitation.
Frequency = count / ntimes; share of the total = precip / total_precip.

A pixel in the feature of more than one track in a window belongs to one class only (best class, then the track that
covers it in most 5-min steps): classes add up to at most the total. Class values can be selected with trackpcp_classes.
The matching of 5-min track times to the data records (prior / after / nearest) and the definitions are those of
pyflextrkr/trackpcp_func.py. The work is the same as `run_trackpcp.py ... --steps 3`.

usage:
    python calc_monthly_rainmap_by_class.py tracking_config.yml trackpcp_config.yml [--year 2025 --month 5] [--workers 64]
"""
import argparse
import logging
import yaml
import dask
from dask.distributed import Client, LocalCluster
from pyflextrkr.ft_utilities import load_config, setup_logging
from pyflextrkr.trackpcp_driver import compute_class_maps


def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("tracking_config")
    ap.add_argument("trackpcp_config")
    ap.add_argument("--year", type=int, default=None, help="calendar month inside the tracking period (default: the whole period)")
    ap.add_argument("--month", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None, help="Dask workers (default: nprocesses of the configs, 0 = serial)")
    ap.add_argument("--outfile", default=None)
    args = ap.parse_args()

    config = load_config(args.tracking_config)
    with open(args.trackpcp_config) as f:
        config.update(yaml.full_load(f))
    nworkers = args.workers if args.workers is not None else (config.get("nprocesses", 4) if config.get("run_parallel", 0) else 0)
    client = None
    if nworkers > 0:
        config["run_parallel"] = 1
        config["nprocesses"] = nworkers
        dask.config.set({"temporary-directory": config.get("dask_tmp_dir", "./")})
        client = Client(LocalCluster(n_workers=nworkers, threads_per_worker=1))
        client.run(setup_logging)
    else:
        config["run_parallel"] = 0
        logger.info("Running in serial.")
    outfile = compute_class_maps(config, year=args.year, month=args.month, outfile=args.outfile)
    logger.info(f"Output saved: {outfile}")
    if client is not None:
        client.close()


if __name__ == "__main__":
    main()
