import sys
import glob
import logging
import xarray as xr
import pandas as pd
import dask
from dask.distributed import wait
from pyflextrkr.ft_utilities import subset_files_timerange, convert_to_cftime, precompute_grid_area

def idfeature_driver(config):
    """
    Driver for feature identification.

    Args:
        config: dictionary
            Dictionary containing config parameters.

    Returns:
        Feature identification are written to netCDF files.
    """

    logger = logging.getLogger(__name__)
    logger.info('Identifying features from raw data')

    clouddata_path = config["clouddata_path"]
    databasename = config.get("databasename", "")
    start_basetime = config.get("start_basetime", None)
    end_basetime = config.get("end_basetime", None)
    run_parallel = config["run_parallel"]
    feature_type = config["feature_type"]
    input_format = config.get("input_format", "netcdf")

    # Load function depending on feature_type
    if feature_type == "generic":
        from pyflextrkr.idfeature_generic import idfeature_generic as id_feature
    elif feature_type == "radar_cells":
        from pyflextrkr.idcells_reflectivity import idcells_reflectivity as id_feature
    elif feature_type == "radar_feature":
        from pyflextrkr.idfeature_radar3d import idfeature_radar3d as id_feature
    elif "tb_pf" in feature_type:
        from pyflextrkr.idclouds_tbpf import idclouds_tbpf as id_feature
    elif "coldpool" in feature_type:
        from pyflextrkr.idcoldpool import idcoldpool as id_feature
    else:
        logger.critical(f"ERROR: Unknown feature_type: {feature_type}")
        logger.critical("Tracking will now exit.")
        sys.exit()

    # Optional capability: combine multiple netcdf/zarr input files into one
    # lazily-read Dataset spanning the full time range, and dispatch dask
    # tasks per TIME FRAME (not per file) in batches, rather than building one
    # task per input file. This is additive: existing feature types keep
    # using the per-file/per-time dispatch below unchanged unless this flag
    # is explicitly set.
    if config.get("combine_files", False):
        _run_idfeature_combined(id_feature, config)
        logger.info('Done with features from raw data.')
        return

    if input_format.lower() == "zarr":

        if "catalog_file" in config:
            import intake     # For catalogs

            # Get catalog info from config
            catalog_file = config["catalog_file"]
            catalog_location = config.get("catalog_location", None)
            catalog_source = config["catalog_source"]
            catalog_params = config.get("catalog_params", {})
            olr_varname = config['olr_varname']
            pcp_varname = config['pcp_varname']
            start_date = config["startdate"]
            end_date = config["enddate"]

            # Load the catalog
            logger.info(f"Loading HEALPix catalog: {catalog_file}")
            in_catalog = intake.open_catalog(catalog_file)
            if catalog_location is not None:
                in_catalog = in_catalog[catalog_location]

            # Get the DataSet from the catalog
            ds = in_catalog[catalog_source](**catalog_params).to_dask()

        else:
            # Get HEALPix path and basename from config
            healpix_path = config["healpix_path"]
            healpix_basename = config["healpix_basename"]
            start_date = config["startdate"]
            end_date = config["enddate"]
            pass_varname = config.get("pass_varname", None)

            # TODO: make file search more robust
            in_file = sorted(glob.glob(f"{healpix_path}{healpix_basename}*.zarr"))[0]
            # Read in Zarr data directly
            # consolidated=None: auto-detect v2 (.zmetadata) or v3 (zarr.json) consolidated metadata
            zarr_consolidated = config.get("zarr_consolidated", None)
            ds = xr.open_zarr(in_file, consolidated=zarr_consolidated)

        # Make a list of required variables based on feature_type
        if "tb_pf" in feature_type:
            keep_vars = [olr_varname, pcp_varname]
        elif "radar_cells" in feature_type:
            reflectivity_varname = config["reflectivity_varname"]
            if pass_varname is not None:
                keep_vars = [reflectivity_varname] + list(pass_varname)
            else:
                keep_vars = [reflectivity_varname]
        else:
            # Keep all variables for other feature types
            keep_vars = list(ds.data_vars)

        # Subset to keep only the required variables
        all_vars = list(ds.data_vars)
        drop_vars = [var for var in all_vars if var not in keep_vars]
        ds = ds.drop_vars(drop_vars)
        
        # Check the calendar type of the time coordinate
        calendar = ds['time'].dt.calendar
        # Convert start_date and end_date to pandas.Timestamp
        start_datetime = pd.to_datetime(start_date, format='%Y%m%d.%H%M')
        end_datetime = pd.to_datetime(end_date, format='%Y%m%d.%H%M')
        # Convert pandas.Timestamp to cftime objects based on the calendar type
        start_datetime_cftime = convert_to_cftime(start_datetime, calendar)
        end_datetime_cftime = convert_to_cftime(end_datetime, calendar)
        # Subset the Dataset using the cftime objects
        ds = ds.sel(time=slice(start_datetime_cftime, end_datetime_cftime))

        # Get the number of time steps
        nfiles = ds.sizes['time']
        logger.info(f"Total number of time steps to process: {nfiles}")

        # Pre-compute grid area file if area_method is 'latlon'
        precompute_grid_area(config)

        # Serial
        if run_parallel == 0:
            for ifile in range(0, nfiles):
                # Subset one time from the DataSets and combine them
                id_feature(ds.isel(time=ifile), config)
        # Parallel
        elif run_parallel >= 1:
            results = []
            for ifile in range(0, nfiles):
                # Subset one time from the DataSets and combine them
                result = dask.delayed(id_feature)(ds.isel(time=ifile), config)
                results.append(result)
            final_result = dask.compute(*results)
            wait(final_result)
        else:
            sys.exit('Valid parallelization flag not provided')

    elif input_format.lower() == "netcdf":

        time_format = config["time_format"]

        # Identify files to process
        infiles_info = subset_files_timerange(
            clouddata_path,
            databasename,
            start_basetime=start_basetime,
            end_basetime=end_basetime,
            time_format=time_format,
        )
        # Get file list
        rawdatafiles = infiles_info[0]
        nfiles = len(rawdatafiles)
        logger.info(f"Total number of files to process: {nfiles}")

        # Pre-compute grid area file if area_method is 'latlon'
        precompute_grid_area(config, first_file=rawdatafiles[0])

        # Serial
        if run_parallel == 0:
            for ifile in range(0, nfiles):
                id_feature(rawdatafiles[ifile], config)
        # Parallel
        elif run_parallel >= 1:
            results = []
            for ifile in range(0, nfiles):
                result = dask.delayed(id_feature)(rawdatafiles[ifile], config)
                results.append(result)
            final_result = dask.compute(*results)
            wait(final_result)
        else:
            sys.exit('Valid parallelization flag not provided')

    logger.info('Done with features from raw data.')
    return


def _open_combined_dataset(config):
    """
    Lazily combine all input files within the configured time range into a
    single Xarray Dataset with the full time dimension, using Xarray's
    native multi-file lazy reading (xr.open_mfdataset for netcdf, xr.open_zarr
    for a single Cartesian zarr store).

    Args:
        config: dictionary
            Dictionary containing config parameters.

    Returns:
        ds: xr.Dataset
            Lazily-combined Dataset with a full time dimension.
        files: list of str
            The list of input files combined (netcdf only; empty list for zarr).
    """
    logger = logging.getLogger(__name__)
    input_format = config.get("input_format", "netcdf")
    time_dimname = config.get("time_dimname", "time")

    if input_format.lower() == "netcdf":
        clouddata_path = config["clouddata_path"]
        databasename = config.get("databasename", "")
        start_basetime = config.get("start_basetime", None)
        end_basetime = config.get("end_basetime", None)
        time_format = config["time_format"]

        infiles_info = subset_files_timerange(
            clouddata_path,
            databasename,
            start_basetime=start_basetime,
            end_basetime=end_basetime,
            time_format=time_format,
        )
        files = infiles_info[0]
        nfiles = len(files)
        if nfiles == 0:
            logger.critical("ERROR: No files found in clouddata_path within the specified time range.")
            logger.critical("Tracking will now exit.")
            sys.exit()
        logger.info(f"Combining {nfiles} files lazily for batched time-frame processing.")

        ds = xr.open_mfdataset(
            files,
            combine="nested",
            concat_dim=time_dimname,
            chunks={time_dimname: 1},
            data_vars="minimal",
            coords="minimal",
            compat="override",
            decode_timedelta=False,
        )
        return ds, files
    elif input_format.lower() == "zarr":
        clouddata_path = config["clouddata_path"]
        databasename = config.get("databasename", "")
        in_files = sorted(glob.glob(f"{clouddata_path}{databasename}*.zarr"))
        if len(in_files) == 0:
            logger.critical(f"ERROR: No zarr store found matching {clouddata_path}{databasename}*.zarr")
            logger.critical("Tracking will now exit.")
            sys.exit()
        ds = xr.open_zarr(in_files[0], consolidated=config.get("zarr_consolidated", None))
        return ds, []
    else:
        logger.critical(f"ERROR: Unsupported input_format for combine_files: {input_format}")
        logger.critical("Tracking will now exit.")
        sys.exit()


def _get_dask_batch_size(config):
    """
    Determine the number of time frames to dispatch to Dask workers per batch.

    Either an explicit 'dask_batch_size', or
    (total worker threads) * 'dask_batch_multiplier' (default 64).
    """
    dask_batch_size = config.get("dask_batch_size", None)
    if dask_batch_size is not None:
        return max(1, int(dask_batch_size))

    dask_batch_multiplier = config.get("dask_batch_multiplier", 64)
    run_parallel = config["run_parallel"]
    total_threads = config.get("nprocesses", 1)
    if run_parallel == 2:
        try:
            from dask.distributed import get_client
            client = get_client()
            # sum(client.nthreads().values()) gives the true total worker
            # thread count; avoid Client.scheduler_info(n_workers=...), which
            # can silently cap the worker count it reports.
            total_threads = sum(client.nthreads().values()) or total_threads
        except (ValueError, ImportError):
            pass
    return max(1, int(total_threads) * int(dask_batch_multiplier))


def _run_idfeature_combined(id_feature, config):
    """
    Driver branch for combine_files=True: combine all input files lazily into
    one Dataset spanning the full time range, then dispatch one task per time
    frame, in batches, waiting for each batch to finish before submitting the
    next one.

    Args:
        id_feature: function
            Feature identification function (accepts a Dataset time slice
            and config).
        config: dictionary
            Dictionary containing config parameters.

    Returns:
        None. Feature identification results are written to netCDF files.
    """
    logger = logging.getLogger(__name__)
    run_parallel = config["run_parallel"]
    time_dimname = config.get("time_dimname", "time")

    ds, files = _open_combined_dataset(config)
    ntimes = ds.sizes[time_dimname]
    logger.info(f"Total number of time steps to process: {ntimes}")

    # Pre-compute grid area file if area_method is 'latlon'
    first_file = files[0] if len(files) > 0 else None
    precompute_grid_area(config, first_file=first_file)

    if run_parallel == 0:
        for itime in range(ntimes):
            id_feature(ds.isel({time_dimname: [itime]}), config)
    elif run_parallel >= 1:
        batch_size = _get_dask_batch_size(config)
        nbatches = (ntimes + batch_size - 1) // batch_size
        logger.info(f"Dask batch size: {batch_size} ({nbatches} batch(es) total).")
        for ibatch in range(nbatches):
            istart = ibatch * batch_size
            iend = min(istart + batch_size, ntimes)
            results = []
            for itime in range(istart, iend):
                result = dask.delayed(id_feature)(ds.isel({time_dimname: [itime]}), config)
                results.append(result)
            final_result = dask.compute(*results)
            wait(final_result)
            logger.info(f"Batch {ibatch + 1}/{nbatches} done (time steps {istart}-{iend - 1}).")
    else:
        sys.exit('Valid parallelization flag not provided')
    return

