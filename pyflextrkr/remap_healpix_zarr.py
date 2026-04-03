import xarray as xr
import numpy as np
import os
import glob
import time
import threading
import warnings
import logging
import dask
import healpy as hp
from functools import partial
from pyflextrkr.ft_utilities import setup_logging


def _log_write_progress(out_zarr, logger, stop_event, interval=30):
    """
    Background thread that logs Zarr write progress every interval seconds.
    Calculates bytes written by walking the Zarr store directory.
    Exits cleanly when stop_event is set.
    """
    start = time.time()
    while not stop_event.wait(interval):
        elapsed = time.time() - start
        try:
            nbytes = sum(
                os.path.getsize(os.path.join(root, f))
                for root, _, files in os.walk(out_zarr)
                for f in files
            )
            logger.info(
                f"Writing Zarr ... elapsed: {elapsed:.0f}s, written: {nbytes / 1e9:.2f} GB"
            )
        except Exception:
            logger.info(f"Writing Zarr ... elapsed: {elapsed:.0f}s")

def remap_to_healpix_zarr(config):
    """
    Convert pixel-level tracking mask NetCDF files to HEALPix grid in Zarr format.
    This function combines the functionality of convert_mask_to_zarr and remap_mask_to_healpix.

    HEALPix cell coordinates are generated directly using healpy (nested ordering),
    removing the previous dependency on intake catalogs and easygems.healpix.
    The zoom level is specified via zarr_output_presets.healpix.zoom in config.

    Args:
        config: dictionary
            Dictionary containing config parameters.
            Required keys for HEALPix remapping:
                zarr_output_presets.healpix.zoom: int
                    HEALPix zoom level (order). nside = 2^zoom.
            Optional keys:
                alternative_healpix_chunks: bool, default False
                    If True, use optimize_healpix_chunks() instead of compute_chunksize().

    Returns:
        out_zarr: string
            Zarr store path for the HEALPix output.
    """
    # Set the logging message level
    setup_logging()
    logger = logging.getLogger(__name__)

    start_time = time.time()
    logger.info(f"Starting NetCDF to HEALPix Zarr conversion: ...")

    # Get client if available
    try:
        from dask.distributed import get_client
        client = get_client()
        parallel = True
        logger.info(f"Using existing Dask client with {len(client.scheduler_info()['workers'])} workers")
    except ValueError:
        logger.warning("No Dask client found, continuing without explicit client")
        client = None
        parallel = False

    # ---------- INPUT CONFIGURATION ----------
    pixeltracking_outpath = config.get("pixeltracking_outpath")
    pixeltracking_filebase = config.get("pixeltracking_filebase", "mcstrack_")
    startdate = config.get("startdate")
    enddate = config.get("enddate")
    outpath = os.path.dirname(os.path.normpath(pixeltracking_outpath)) + "/"

    # Get preset-specific configuration
    presets = config.get("zarr_output_presets", {})
    preset_mask = presets.get("mask", {})
    preset_healpix = presets.get("healpix", {})
    write_mask = preset_mask.get("write", False)
    
    # ---------- HEALPIX CONFIGURATION ----------
    # Get zoom level from healpix preset, fall back to catalog_params for backward compatibility
    hp_zoom = preset_healpix.get("zoom", config.get("catalog_params", {}).get("zoom"))
    hp_version = preset_healpix.get("version", "v1")
    if hp_zoom is None:
        logger.error("HEALPix zoom level not specified. Set zarr_output_presets.healpix.zoom in config.")
        return None
    
    # ---------- OUTPUT FILE PATHS ----------
    # Intermediate lat/lon Zarr (if needed)
    latlon_filebase = preset_mask.get("out_filebase", "mcs_mask_latlon_")
    latlon_zarr = f"{outpath}{latlon_filebase}{startdate}_{enddate}.zarr"
    
    # Final HEALPix Zarr output
    hp_filebase = preset_healpix.get("out_filebase", "mcs_mask_")
    out_zarr = f"{outpath}{hp_filebase}hp{hp_zoom}_{hp_version}_{startdate}_{enddate}.zarr"
    
    # Check if output exists and should be overwritten
    overwrite = preset_mask.get("overwrite", config.get("overwrite_zarr", True))
    if os.path.exists(out_zarr) and not overwrite:
        logger.warning(f"HEALPix Zarr store already exists at {out_zarr} and overwrite=False. Skipping.")
        return out_zarr

    # ---------- CHUNKING CONFIGURATION ----------
    chunksize_time = config.get("chunksize_time", "auto")
    chunksize_lat = config.get("chunksize_lat", "auto")
    chunksize_lon = config.get("chunksize_lon", "auto")
    chunksize_cell = config.get("chunksize_cell", "auto")
    # Zarr format version: 2 (v2, backward compatible) or 3 (v3, default)
    zarr_format = config.get("zarr_format", 3)
    
    # ---------- VARIABLE SELECTION ----------
    # Required coordinate variables
    required_vars = ["time", "lat", "lon"]
    
    # # Special case for "full" preset - include all variables
    # if output_preset == 'full':
    #     # Will be populated with all dataset variables later
    #     config_vars = []
    #     logger.info("Using 'full' preset: all variables will be included")        
    # Normal case - get variables from preset config or main config
    if "var_list" in preset_mask:
        config_vars = preset_mask["var_list"]
    else:
        config_vars = ["tracknumber"]
    
    if isinstance(config_vars, str):
        config_vars = [config_vars]  # Convert string to list if needed
    
    # ---------- READ INPUT FILES ----------
    logger.info("Reading input NetCDF files...")
    file_list = sorted(glob.glob(f"{pixeltracking_outpath}{pixeltracking_filebase}*.nc"))
    logger.info(f"Number of input files: {len(file_list)}")

    # Open as a lazy dataset
    ds = xr.open_mfdataset(
        file_list,
        combine="by_coords",
        parallel=parallel,
        chunks={},  # Defer chunking to later
        mask_and_scale=False,
    )
    logger.info(f"Finished reading input files.")
    
    # ---------- PROCESS VARIABLES ----------
    # # For 'full' preset, include all variables from the dataset
    # if output_preset == 'full':
    #     # Get all variable names
    #     all_vars = list(ds.data_vars)
    #     logger.info(f"Full preset: found {len(all_vars)} variables in dataset")
    #     config_vars = all_vars

    # Combine required vars with config vars, ensuring no duplicates
    keep_var_list = list(set(required_vars + config_vars))
    logger.info(f"Variables to include in output: {keep_var_list}")

    # Get variable rename mapping
    if "var_rename" in preset_mask:
        rename_dict = preset_mask["var_rename"]
    else:
        rename_dict = config.get("zarr_var_rename", {})
    
    if rename_dict:
        logger.info(f"Variable renaming map: {rename_dict}")
    else:
        logger.info("No variable renaming will be performed")

    # Filter to keep only existing variables
    variables_to_keep = [v for v in keep_var_list if v in ds.variables]
    
    # Create a rename dictionary containing only the variables that exist in dataset
    active_renames = {k: v for k, v in rename_dict.items() if k in variables_to_keep}
    
    # Select and rename variables in one step
    if active_renames:
        logger.info(f"Renaming variables: {active_renames}")
        ds = ds[variables_to_keep].rename(active_renames)
    else:
        # Simple subset without renaming
        ds = ds[variables_to_keep]
    
    # Apply initial chunking for processing
    ds = ds.chunk({
        "time": min(100, chunksize_time) if isinstance(chunksize_time, int) else "auto", 
        "lat": chunksize_lat, 
        "lon": chunksize_lon
    })
    
    # ---------- INTERMEDIATE ZARR (OPTIONAL) ----------
    if write_mask:
        logger.info(f"Writing intermediate lat/lon Zarr to: {latlon_zarr}")

        # Write lat/lon zarr with progress tracking
        if client:
            from dask.distributed import progress

            try:
                # Build and write under task-based rechunking to avoid P2PRechunkLayer
                # assertion in distributed shuffle culling.
                with dask.config.set({"array.rechunk.method": "tasks"}):
                    chunked_ds = ds.chunk({
                        "time": chunksize_time,
                        "lat": chunksize_lat,
                        "lon": chunksize_lon,
                    })
                    logger.info(f"Dataset dimensions: {dict(chunked_ds.sizes)}")
                    logger.info(f"Chunking scheme: time={chunksize_time}, lat={chunksize_lat}, lon={chunksize_lon}")

                    # Write intermediate Zarr
                    write_task = chunked_ds.to_zarr(
                        latlon_zarr,
                        mode="w",
                        consolidated=True,
                        zarr_format=zarr_format,
                        compute=False,
                    )

                    # Compute with progress tracking
                    future = client.compute(write_task)
                logger.info("Writing intermediate Zarr (this may take a while)...")
                progress(future)
                
                result = future.result()
                logger.info("Intermediate Zarr write completed successfully")
                
                # Reload from the Zarr store to ensure everything is consistent
                ds = xr.open_dataset(latlon_zarr, engine='zarr', consolidated=True)
                
            except Exception as e:
                logger.error(f"Intermediate Zarr write failed: {str(e)}")
                raise
        else:
            # Compute locally
            chunked_ds = ds.chunk({
                "time": chunksize_time,
                "lat": chunksize_lat,
                "lon": chunksize_lon,
            })
            logger.info(f"Dataset dimensions: {dict(chunked_ds.sizes)}")
            logger.info(f"Chunking scheme: time={chunksize_time}, lat={chunksize_lat}, lon={chunksize_lon}")
            chunked_ds.to_zarr(latlon_zarr, mode="w", consolidated=True, zarr_format=zarr_format)
            ds = xr.open_dataset(latlon_zarr, engine='zarr', consolidated=True)
    
    # ---------- HEALPIX REMAPPING ----------
    logger.info("Beginning HEALPix remapping...")
    
    # Save longitude sign info
    signed_lon = True if np.min(ds["lon"]) < 0 else False
    logger.info(f"Input data lon coordinate has negative values: {signed_lon}")
    
    # Generate HEALPix cell coordinates using healpy
    hp_nside = 2 ** hp_zoom
    hp_npix = hp.nside2npix(hp_nside)
    logger.info(f"Generating HEALPix grid: zoom={hp_zoom}, nside={hp_nside}, npix={hp_npix}")
    hp_lons, hp_lats = hp.pix2ang(
        nside=hp_nside, ipix=np.arange(hp_npix), lonlat=True, nest=True,
    )
    if signed_lon:
        hp_lons = np.where(hp_lons <= 180, hp_lons, hp_lons - 360)
    else:
        hp_lons = hp_lons % 360
    
    # Create DataArrays with cell coordinate and extra hp coordinates
    # (hp coordinates are needed for limiting extrapolation via is_valid)
    cell_coord = np.arange(hp_npix)
    lon_hp = xr.DataArray(
        hp_lons, dims="cell",
        coords={"cell": cell_coord, "lon_hp": ("cell", hp_lons)},
    )
    lat_hp = xr.DataArray(
        hp_lats, dims="cell",
        coords={"cell": cell_coord, "lat_hp": ("cell", hp_lats)},
    )
    
    # Make sure coordinates are fixed before remapping
    ds = fix_coords(ds)

    # Calculate appropriate tolerance based on zoom level
    tolerance = calculate_healpix_tolerance(hp_zoom)
    logger.info(f"Using HEALPix tolerance of {tolerance:.4f}° at zoom level {hp_zoom}")
    
    # Remap DataSet to HEALPix
    logger.info("Applying nearest neighbor remapping to HEALPix grid...")
    fill_value = 0
    dsout_hp = ds.sel(
        lon=lon_hp, lat=lat_hp, method="nearest",
    ).where(partial(is_valid, tolerance=tolerance), fill_value)
    
    # Drop lat/lon coordinates (not needed in HEALPix)
    dsout_hp = dsout_hp.drop_vars(["lat_hp", "lon_hp", "lat", "lon"])
    # Add CRS coordinate (consistent with HEALPix catalog convention)
    crs_var = xr.DataArray(
        data=np.array([], dtype=np.float64),
        dims=["crs"],
        coords={"crs": np.array([], dtype=np.float64)},
        attrs={
            "grid_mapping_name": "healpix",
            "healpix_nside": hp_nside,
            "healpix_order": "nest",
        },
    )
    dsout_hp = dsout_hp.assign_coords(crs=crs_var)
    # Update global attributes
    dsout_hp.attrs['Title'] = f"HEALPix remapped tracking mask data (zoom={hp_zoom})"
    dsout_hp.attrs['zoom'] = hp_zoom
    dsout_hp.attrs["Created_on"] = time.ctime(time.time())

    # Compute cell chunking for HEALPix grid
    if chunksize_cell == 'auto' or not isinstance(chunksize_cell, (int, float)):
        if config.get("alternative_healpix_chunks", False):
            chunksize_cell = optimize_healpix_chunks(hp_npix, chunksize_cell, logger)
        else:
            chunksize_cell = compute_chunksize(hp_zoom)
            logger.info(f"HEALPix chunk size: {chunksize_cell} (zoom={hp_zoom})")
    else:
        chunksize_cell = int(chunksize_cell)

    # Make time chunks more even if needed
    if isinstance(chunksize_time, (int, float)) and chunksize_time != 'auto':
        total_times = dsout_hp.sizes['time']
        chunks = total_times // chunksize_time
        if chunks * chunksize_time < total_times:
            # We have a remainder - try to make chunks more even
            if total_times % chunks == 0:
                chunksize_time = total_times // chunks
            elif total_times % (chunks + 1) == 0:
                chunksize_time = total_times // (chunks + 1)
    
    # For client mode, build chunk graph inside protected context later.
    # For serial mode, build it here.
    if not client:
        chunked_hp = dsout_hp.chunk({
            "time": chunksize_time,
            "cell": chunksize_cell,
        })

        # Report dataset size and chunking info
        logger.info(f"HEALPix dataset dimensions: {dict(chunked_hp.sizes)}")
        logger.info(f"HEALPix chunking scheme: time={chunksize_time}, cell={chunksize_cell}")
    
    # ---------- WRITE HEALPIX ZARR OUTPUT ----------
    logger.info(f"Starting HEALPix Zarr write to: {out_zarr}")

    # Start background thread that logs elapsed time and bytes written every 30 s.
    # Uses only stdlib (threading, os, time) — no extra dependencies.
    stop_event = threading.Event()
    progress_thread = threading.Thread(
        target=_log_write_progress,
        args=(out_zarr, logger, stop_event),
        kwargs={"interval": 30},
        daemon=True,
    )
    progress_thread.start()

    try:
        if client:
            import psutil

            try:
                memory_usage = client.run(lambda: psutil.Process().memory_info().rss / 1e9)
                logger.info(f"Current memory usage across workers (GB): {memory_usage}")
            except Exception as e:
                logger.warning(f"Could not get memory usage: {e}")

            # Suppress distributed.shuffle noise during write
            shuffle_logger = logging.getLogger('distributed.shuffle')
            original_level = shuffle_logger.level
            shuffle_logger.setLevel(logging.ERROR)
            try:
                # When a distributed Client is active, dask.array.rechunk._choose_rechunk_method
                # automatically selects "p2p" for large rechunks, injecting P2PRechunkLayer
                # into the graph. P2PRechunkLayer.cull() has a bug: it asserts all key
                # components after the layer name are ints, but xarray zarr-write keys
                # contain non-int components (e.g. variable names), triggering AssertionError.
                # Fix: force task-based rechunking via the 'array.rechunk.method' config key,
                # which is checked FIRST in _choose_rechunk_method before the distributed
                # client check. This must wrap both .chunk() (graph construction) and
                # to_zarr() (graph execution) to apply to all rechunk operations.
                with dask.config.set({"array.rechunk.method": "tasks"}):
                    chunked_hp_safe = dsout_hp.chunk({"time": chunksize_time, "cell": chunksize_cell})
                    logger.info(f"HEALPix dataset dimensions: {dict(chunked_hp_safe.sizes)}")
                    logger.info(f"HEALPix chunking scheme: time={chunksize_time}, cell={chunksize_cell}")
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message="Consolidated metadata",
                            category=UserWarning,
                        )
                        chunked_hp_safe.to_zarr(
                            out_zarr,
                            mode="w",
                            consolidated=True,
                            zarr_format=zarr_format,
                        )
                logger.info("HEALPix Zarr write completed successfully")
            except Exception as e:
                logger.error(f"HEALPix Zarr write failed: {str(e)}")
                raise
            finally:
                shuffle_logger.setLevel(original_level)
        else:
            # Serial / threaded scheduler path
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Consolidated metadata",
                    category=UserWarning,
                )
                chunked_hp.to_zarr(
                    out_zarr,
                    mode="w",
                    consolidated=True,
                    zarr_format=zarr_format,
                )
    finally:
        stop_event.set()
        progress_thread.join(timeout=5)
    
    # Cleanup intermediate file if created
    # if not skip_intermediate and os.path.exists(latlon_zarr) and out_zarr != latlon_zarr:
    # if (write_mask == True) and (os.path.exists(latlon_zarr)) and (out_zarr != latlon_zarr):
    #     if config.get("remove_intermediate", False):
    #         logger.info(f"Removing intermediate lat/lon Zarr: {latlon_zarr}")
    #         import shutil
    #         try:
    #             shutil.rmtree(latlon_zarr)
    #         except Exception as e:
    #             logger.warning(f"Failed to remove intermediate file: {str(e)}")
    
    logger.info(f"HEALPix conversion complete: {out_zarr}")
    
    # Log completion time
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    logger.info(f"Conversion completed in {int(hours):02}:{int(minutes):02}:{int(seconds):02} (hh:mm:ss).")
    
    return out_zarr


def fix_coords(ds, lat_dim="lat", lon_dim="lon", roll=False):
    """
    Fix coordinates in a dataset:
    1. Convert longitude from -180/+180 to 0-360 range (optional)
    2. Roll dataset to start at longitude 0 (optional)
    3. Ensure coordinates are in ascending order
    
    Parameters:
    -----------
    ds : xarray.Dataset or xarray.DataArray
        Dataset with lat/lon coordinates
    lat_dim : str, optional
        Name of latitude dimension, default "lat"
    lon_dim : str, optional
        Name of longitude dimension, default "lon"
    roll : bool, optional, default=False
        If True, convert longitude from -180/+180 to 0-360, and roll the dataset to start at longitude 0
        
    Returns:
    --------
    xarray.Dataset or xarray.DataArray
        Dataset with fixed coordinates
    """
    if roll:
        # Find where longitude crosses from negative to positive (approx. where lon=0)
        lon_0_index = (ds[lon_dim] < 0).sum().item()
        
        # Create indexers for the roll
        lon_indices = np.roll(np.arange(ds.sizes[lon_dim]), -lon_0_index)
        
        # Roll dataset and convert longitudes to 0-360 range
        ds = ds.isel({lon_dim: lon_indices})
        lon360 = xr.where(ds[lon_dim] < 0, ds[lon_dim] + 360, ds[lon_dim])
        ds = ds.assign_coords({lon_dim: lon360})
    
    # Ensure latitude and longitude are in ascending order if needed
    if np.all(np.diff(ds[lat_dim].values) < 0):
        ds = ds.isel({lat_dim: slice(None, None, -1)})
    if np.all(np.diff(ds[lon_dim].values) < 0):
        ds = ds.isel({lon_dim: slice(None, None, -1)})
    
    return ds


def is_valid(ds, tolerance=0.1):
    """
    Limit extrapolation distance to a certain tolerance.
    This is useful for preventing extrapolation of regional data to global HEALPix grid.

    Args:
        ds (xarray.Dataset):
            The dataset containing latitude and longitude coordinates.
        tolerance (float): default=0.1
            The maximum allowed distance in [degrees] for extrapolation.

    Returns:
        xarray.DataSet.
    """
    return (np.abs(ds.lat - ds.lat_hp) < tolerance) & (np.abs(ds.lon - ds.lon_hp) < tolerance)


def calculate_healpix_tolerance(zoom_level):
    """
    Calculate appropriate tolerance for is_valid function based on HEALPix zoom level.
    Returns approximately one grid cell size in degrees.
    
    Args:
        zoom_level (int): HEALPix zoom level
        
    Returns:
        float: Tolerance in degrees
    """
    # Calculate nside from zoom level (nside = 2^zoom)
    # nside determines HEALPix resolution - each increase in zoom doubles the resolution
    nside = 2 ** zoom_level
    
    # Calculate approximate pixel size in degrees
    # Mathematical derivation:
    # - Sphere has total area of 4π steradians (= 4π × (180/π)² sq. degrees)
    # - HEALPix divides sphere into 12 × nside² equal-area pixels
    # - Each pixel has area = 4π × (180/π)² / (12 × nside²) sq. degrees
    # - Linear size = √(pixel area) ≈ 58.6 / nside degrees
    # This gives approximately the angular width of one HEALPix cell
    pixel_size_degrees = 58.6 / nside
    
    return pixel_size_degrees


def compute_chunksize(zoom):
    """
    Compute HEALPix cell chunk size based on zoom level.

    For lower zoom levels (zoom < 8), the total number of cells is small enough
    to fit in a single chunk (npix = 12 * 4^zoom).
    For higher zoom levels (zoom >= 8), uses a fixed chunk size of 4^9 = 262144,
    which evenly divides npix since npix = 12 * 4^zoom and 4^9 divides 12 * 4^zoom
    for all zoom >= 8.

    Parameters:
    -----------
    zoom : int
        HEALPix zoom level (order). Determines resolution via nside = 2^zoom.

    Returns:
    --------
    int
        Chunk size for the cell dimension.
    """
    # Zoom level below which all cells fit in one chunk
    start_split = 8
    if zoom < start_split:
        # Total cells = 12 * 4^zoom, use a single chunk
        return 12 * 4**zoom
    else:
        # Fixed chunk size that evenly divides npix for zoom >= 8
        return 4 ** (start_split + 1)


def optimize_healpix_chunks(total_cells, chunksize_cell='auto', logger=None):
    """
    Optimize chunking for HEALPix grids based on their hierarchical structure.
    
    Parameters:
    -----------
    total_cells : int
        Total number of HEALPix cells (npix)
    chunksize_cell : int or 'auto', optional
        Requested chunk size or 'auto' for automatic optimization
    logger : logging.Logger, optional
        Logger for debug information
        
    Returns:
    --------
    int
        Optimized chunk size for the cell dimension
    """
    
    # If a specific chunk size is requested (not 'auto'), return it directly
    if chunksize_cell != 'auto' and isinstance(chunksize_cell, (int, float)):
        return int(chunksize_cell)
    
    # Estimate nside based on cell count (12 * nside^2 = total cells)
    nside = int(np.sqrt(total_cells / 12))
    
    if logger:
        logger.info(f"Total HEALPix cells: {total_cells}, estimated nside: {nside}")
    
    # For very small grids: divide into 12 chunks (one per base face)
    if total_cells <= 12288:
        chunksize_cell = max(1, total_cells // 12)
    else:
        # For larger grids, create chunks based on nested HEALPix structure
        base_face_size = total_cells // 12
        
        # Try to keep chunks around 262144 cells (good balance for most systems)
        target_chunk_size = 262144
        
        if base_face_size <= target_chunk_size:
            chunksize_cell = base_face_size
        else:
            # Find a divisor of base_face_size that creates chunks close to target size
            # Focus on powers of 4 since HEALPix has quad-tree structure
            for divisor in [2, 4, 8, 16, 32, 64, 128, 256]:
                if base_face_size // divisor <= target_chunk_size:
                    chunksize_cell = base_face_size // divisor
                    break
            else:
                # If no good divisor found, use a power of 4 near target size
                power = max(1, int(np.log(target_chunk_size) / np.log(4)))
                chunksize_cell = 4**power
    
    # Ensure chunk size isn't too small
    chunksize_cell = max(chunksize_cell, 4096)
    
    # Ensure chunk size divides evenly or leaves a smaller final chunk
    if total_cells % chunksize_cell > 0:
        # Try to find a divisor close to our calculated chunk size
        for factor in range(chunksize_cell, chunksize_cell // 2, -1):
            if total_cells % factor == 0:
                chunksize_cell = factor
                break
    
    if logger:
        logger.info(f"Optimized HEALPix chunk size: {chunksize_cell}")
        
    return int(chunksize_cell)