from __future__ import division, print_function
import sys
import os
import time
import shutil
import logging
import numpy as np
from netCDF4 import Dataset
import xarray as xr
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d
import dask
from dask.distributed import wait
from pyflextrkr.ft_utilities import subset_files_timerange, get_pixel_area

def movement_speed(
        config,
        trackstats_filebase=None,
        trackstats_outfilebase=None,
        pixelpath_basename=None,
        pixeltracking_filebase=None,
):
    """
    Calculate movement speed using pixel level tracked feature.

    Args:
        config: dictionary
            Dictionary containing config parameters.
        trackstats_filebase: string, default=None
            Input track statistics file basename.
        trackstats_outfilebase: string, default=None
            Output track statistics file basename.
        pixelpath_basename: string, default=None,
            Pixel-level files base path name.
            If None, pixeltracking_outpath defaults to config["pixeltracking_outpath"].
            Otherwise, pixeltracking_outpath is constructed using the config info:
                f'{config["root_path"]}/{pixelpath_basename}/{config["startdate"]}_{config["enddate"]}/'
        pixeltracking_filebase=None, default=None
            Pixel-level file basename.
            If None, defaults to config["pixeltracking_filebase"].

    Returns:
        statistics_outfile: string
            MCS track statistics file name.
    """

    stats_outpath = config["stats_outpath"]
    startdate = config["startdate"]
    enddate = config["enddate"]
    start_basetime = config["start_basetime"]
    end_basetime = config["end_basetime"]
    tracks_dimname = config["tracks_dimname"]
    times_dimname = config["times_dimname"]
    run_parallel = config["run_parallel"]
    feature_type = config["feature_type"]
    pixel_radius = config["pixel_radius"]
    area_method = config.get("area_method", "fixed")
    lag = config["lag_for_speed"]
    max_speed_thresh = config["max_speed_thresh"]

    logger = logging.getLogger(__name__)
    logger.info('Calculating movement speed using pixel-level tracked feature')

    # Set trackstats file basenames
    if trackstats_filebase is None:
        trackstats_filebase = config["mcsrobust_filebase"]
    if trackstats_outfilebase is None:
        trackstats_outfilebase = config["mcsfinal_filebase"]
    # Set pixel file path and basename
    if pixelpath_basename is None:
        pixeltracking_outpath = config["pixeltracking_outpath"]
    else:
        pixeltracking_outpath = f'{config["root_path"]}/{pixelpath_basename}/{config["startdate"]}_{config["enddate"]}/'
    if pixeltracking_filebase is None:
        pixeltracking_filebase = config["pixeltracking_filebase"]

    # Stats file name
    # if 'tb_pf' in feature_type:
    # Robust MCS track stats filename
    statistics_file = f"{stats_outpath}{trackstats_filebase}{startdate}_{enddate}.nc"
    # Output MCS track stats filename
    statistics_outfile = f"{stats_outpath}{trackstats_outfilebase}{startdate}_{enddate}.nc"

    # Identify pixel files to process
    filelist, \
    files_basetime, \
    files_datestring, \
    files_timestring = subset_files_timerange(pixeltracking_outpath,
                                              pixeltracking_filebase,
                                              start_basetime,
                                              end_basetime)
    nfiles = len(filelist)
    logger.info(f"Total number of files to process: {nfiles}")

    # Open stats file to get maximum number of storms to track.
    ds_stats = xr.open_dataset(statistics_file,
                               mask_and_scale=False,
                               decode_times=False)
    ntracks = ds_stats.sizes[tracks_dimname]
    ntimes = ds_stats.sizes[times_dimname]
    tracks_coord = ds_stats.coords[tracks_dimname].data
    times_coord = ds_stats.coords[times_dimname].data
    # Only the first time of each track is needed (aligns the movement values to the track times)
    stats_start_basetime = ds_stats.variables['base_time'][:, 0].values
    ds_stats.close()

    # Make file pairs
    filepairs = list(zip(filelist[0:-lag], filelist[lag::]))


    results = []
    # Serial
    if run_parallel == 0:
        for ifile in range(0, nfiles-1):
            result = movement_of_feature_fft(
                filepairs[ifile], ntracks,
                config, sparse=True,
            )
            results.append(result)
        final_result = results
    # Parallel
    elif run_parallel >= 1:
        for ifile in range(0, nfiles-1):
            result = dask.delayed(movement_of_feature_fft)(
                filepairs[ifile], ntracks,
                config, sparse=True,
            )
            results.append(result)
        final_result = dask.compute(*results)
        wait(final_result)
    else:
        sys.exit('Valid parallelization flag not provided')

    # Movement of the measured tracks of every file pair -> [tracks, times] arrays aligned to the track times.
    # (Only the few hundred tracks that are measured in a pair are kept, never an array over all pairs and tracks.)
    tracks_movement = align_movement_to_tracks(final_result, stats_start_basetime, ntimes, lag)

    # Put the movement variables in an Xarray Dataset
    # consistent with track statistics
    ds_vars = define_movement_dataset(tracks_movement, lag, times_coord, times_dimname, tracks_coord, tracks_dimname)

    # Run filter to interpolate over high movement speeds
    ds_vars_filt = filter_interp_speed(ds_vars, config)

    ###########################################################################
    # Write statistics to netcdf file: a copy of the input statistics file with the movement variables added.
    # (Merging the movement variables with the opened input and writing the result would read every variable of the
    # input into memory: 180 GB for a month of 240,000 tracks.)
    if statistics_file == statistics_outfile:
        logger.critical(f"The input and output statistics files are the same ({statistics_file}). Tracking will now exit.")
        sys.exit()

    # Delete file if it already exists
    if os.path.isfile(statistics_outfile):
        os.remove(statistics_outfile)
    shutil.copyfile(statistics_file, statistics_outfile)

    # Update global attributes
    ds_vars_filt.attrs["Created_on"] = time.ctime(time.time())
    ds_vars_filt.attrs["max_speed_thresh"] = max_speed_thresh

    # Set encoding/compression for the new variables
    comp = dict(zlib=True)
    encoding = {var: comp for var in ds_vars_filt.data_vars}

    # Append to the copy (the dimensions and their coordinates are in the file already)
    ds_vars_filt.drop_vars([tracks_dimname, times_dimname]).to_netcdf(
        path=statistics_outfile, mode="a", format="NETCDF4", encoding=encoding)
    logger.info(f"{statistics_outfile}")

    return statistics_outfile



def movement_of_feature_fft(
        filepairs,
        ntracks,
        config,
        optimize_sub_array=True,
        sparse=False,
):
    """
    Calculate movement of tracked features.

    Only the tracks that have at least min_size_thresh_for_speed pixels in both files are measured (a few hundred
    of the tracks of a month); the other tracks have NaN.

    Args:
        filepairs: tuple
            Pairs of pixel file names.
        ntracks: int
            Number of tracks.
        config: dictionary
            Dictionary containing config parameters.
        optimize_sub_array: boolean
            Flag to subset each tracked feature from the full image.
        sparse: boolean
            False (default): return arrays over all ntracks tracks. True: return only the measured tracks.

    Returns:
        sparse=False: (y_lag, x_lag, time_lag, base_time, pixel_length_per_feature)
        sparse=True: (track_index, y_lag, x_lag, time_lag, base_time, pixel_length) for the measured tracks
        y_lag: np.array
            Movement magnitude in y-direction.
        x_lag: np.array
            Movement magnitude in x-direction.
        time_lag: float
            Time difference between two pixel files.
        base_time: float
            Base time for the first pixel file.
        pixel_length: np.array
            Per-feature mean pixel length [km].
    """

    tracknumber = config["track_number_for_speed"]
    track_field = config["track_field_for_speed"]
    min_size_thresh = config["min_size_thresh_for_speed"]
    pixel_radius = config["pixel_radius"]
    area_method = config.get("area_method", "fixed")
    # Parameters for handling perdiodic boundary condition
    pbc_direction = config.get("pbc_direction", "none")
    max_feature_frac_x = 0.95   # Max fraction of domain size for a feature in x-direction
    max_feature_frac_y = 0.95   # Max fraction of domain size for a feature in y-direction

    # Load pixel area for per-feature pixel length computation
    if area_method == "latlon":
        pixel_area = get_pixel_area(config)
    else:
        pixel_area = None

    logger = logging.getLogger(__name__)
    logger.debug("Starting Storm File: %s" % filepairs[0])
    sys.stdout.flush()

    dset1 = Dataset(filepairs[0], 'r')
    dset2 = Dataset(filepairs[1], 'r')
    y_lag = np.full(ntracks, np.nan)
    x_lag = np.full(ntracks, np.nan)
    # Per-feature mean pixel length [km]
    if pixel_area is not None:
        pixel_length_per_feature = np.full(ntracks, np.nan)
    else:
        pixel_length_per_feature = np.full(ntracks, pixel_radius)

    # Get minimum size of feature from pixel files
    min_cloud_size = np.minimum(get_pixel_size_of_clouds(dset1, ntracks, tracknumber),
                                get_pixel_size_of_clouds(dset2, ntracks, tracknumber))
    # Get tracknumber and field values
    tracknumber_1 = dset1.variables[tracknumber][:].squeeze()
    tracknumber_2 = dset2.variables[tracknumber][:].squeeze()
    field_1 = dset1.variables[track_field][:].squeeze()
    field_2 = dset2.variables[track_field][:].squeeze()
    # Get dimensions of data
    ydim, xdim = np.shape(field_1)

    # Loop over the tracks that are big enough in both files
    measured = np.where(min_cloud_size >= min_size_thresh)[0]
    for track_number in measured:
        # In pixel mask files, track number need to +1
        # track indices start from 0, pixel mask track numbers start from 1
        track_number_pix = track_number + 1
        if optimize_sub_array:
            # Subset array to only contain the current track mask area
            # Calculate size of bounding box
            ymin, ymax, xmin, xmax = get_bounding_box_for_fft(tracknumber_1, tracknumber_2, track_number_pix)
            masked_field_1 = field_1[ymin:ymax, xmin:xmax].copy()
            masked_field_2 = field_2[ymin:ymax, xmin:xmax].copy()

            # Replace data outside of the current track mask area or NaN with 0
            masked_field_1[tracknumber_1[ymin:ymax, xmin:xmax] != track_number_pix] = 0
            masked_field_1[np.isnan(masked_field_1)] = 0

            masked_field_2[tracknumber_2[ymin:ymax, xmin:xmax] != track_number_pix] = 0
            masked_field_2[np.isnan(masked_field_2)] = 0

            # Check feature boundary span
            # If boundary span > X fraction of domain, and periodic boundary condition is set,
            # roll the data such that the feature does not span across the domain boundary
            if (((xmax - xmin) >= xdim * max_feature_frac_x) or \
                ((ymax - ymin) >= ydim * max_feature_frac_y)) and \
                (pbc_direction != 'none'):
                # Combine non-zero pixels from both time steps to detect
                # which boundaries the feature crosses.
                masked_fields = (np.abs(masked_field_1) > 0) | (np.abs(masked_field_2) > 0)
                # Sub-array dimensions; indices are 0-based within the sub-box.
                sub_ydim, sub_xdim = masked_fields.shape
                occ_y, occ_x = np.where(masked_fields)
                # Gap-based boundary detection:
                # The largest gap in sorted occupied indices > 50% of the
                # sub-array span indicates the feature wraps across that edge.
                # The shift amount is chosen so the right/top cluster rolls
                # to the left/bottom, compacting the feature away from the border.
                sorted_x = np.sort(np.unique(occ_x))
                gap_x = np.diff(sorted_x)
                max_gap_x_idx = np.argmax(gap_x) if len(gap_x) > 0 else 0
                if len(gap_x) > 0 and gap_x[max_gap_x_idx] > sub_xdim * 0.5:
                    shift_x_right = sub_xdim - int(sorted_x[max_gap_x_idx + 1])
                else:
                    shift_x_right = 0
                sorted_y = np.sort(np.unique(occ_y))
                gap_y = np.diff(sorted_y)
                max_gap_y_idx = np.argmax(gap_y) if len(gap_y) > 0 else 0
                if len(gap_y) > 0 and gap_y[max_gap_y_idx] > sub_ydim * 0.5:
                    shift_y_top = sub_ydim - int(sorted_y[max_gap_y_idx + 1])
                else:
                    shift_y_top = 0
                # Roll both sub-arrays by the detected shifts so the
                # feature is compact and does not straddle the boundary.
                masked_field_1 = np.roll(masked_field_1, (shift_y_top, shift_x_right), axis=(0, 1))
                masked_field_2 = np.roll(masked_field_2, (shift_y_top, shift_x_right), axis=(0, 1))
        else:
            masked_field_1 = field_1.copy()
            masked_field_2 = field_2.copy()

            # Replace data outside of the current track mask area or NaN with 0
            masked_field_1[tracknumber_1 != track_number_pix] = 0
            masked_field_1[np.isnan(masked_field_1)] = 0

            masked_field_2[tracknumber_2 != track_number_pix] = 0
            masked_field_2[np.isnan(masked_field_2)] = 0

        # Flip the second image, do an FFT convolution
        result = fftconvolve(masked_field_1, masked_field_2[::-1, ::-1], mode='same')
        # Get the index with max value (highest correlation)
        # then reshape it to 2D to get x, y index

        # ALTERNATIVE No. 1
        y_step, x_step = np.apply_along_axis(np.mean, 1,
            np.asarray(np.where(result>np.quantile(result, .995)))).round(0).astype('int')

        y_dim, x_dim = np.shape(masked_field_1)
        # Get the relative position from the center of the image
        # This is the movement in x, y direction
        y_lag[track_number] = np.floor(y_dim/2) - y_step
        x_lag[track_number] = np.floor(x_dim/2) - x_step

        # Compute per-feature mean pixel length for latlon grids
        if pixel_area is not None:
            if optimize_sub_array:
                sub_pa = pixel_area[ymin:ymax, xmin:xmax]
                feat_mask = (
                    (tracknumber_1[ymin:ymax, xmin:xmax] == track_number_pix) |
                    (tracknumber_2[ymin:ymax, xmin:xmax] == track_number_pix)
                )
            else:
                sub_pa = pixel_area
                feat_mask = (
                    (tracknumber_1 == track_number_pix) |
                    (tracknumber_2 == track_number_pix)
                )
            pixel_length_per_feature[track_number] = np.sqrt(
                np.nanmean(sub_pa[feat_mask])
            )

    # Get time difference between the file pair
    time_lag = dset2.variables['time'][0] - dset1.variables['time'][0]
    base_time = dset1.variables['time'][0].copy()

    dset1.close()
    dset2.close()
    if sparse:
        return (measured, y_lag[measured], x_lag[measured], time_lag, base_time,
                pixel_length_per_feature[measured])
    return y_lag, x_lag, time_lag, base_time, pixel_length_per_feature


def get_pixel_size_of_clouds(
        dataset,
        ntracks,
        tracknumber,
):
    """
    Calculate pixel size of each identified cloud in the file.

    Args:
        dataset: Dataset
            netcdf Dataset
        tracknumber: string
            variable that contains pixel level values.

    Returns:
        counts: array_like
            Pixel size of every cloud in file. Cloud 0 is stored at 0.
    """
    storm_sizes = np.zeros(ntracks + 1)

    track, counts = np.unique(dataset.variables[tracknumber][:], return_counts=True)
    storm_sizes[track] = counts
    # storm_sizes[0] = 0
    # Remove the first value (background, which is not storm)
    storm_sizes = storm_sizes[1:]
    return storm_sizes


def get_bounding_box_for_fft(in1, in2, track_number):
    """
    Given two masks and a track number, calculate the maximum bounding box to fit both

    Args:
        in1: np.array
            First mask array
        in2: np.array
            Second mask array
        track_number: int
            Track number for masking.

    Returns:
        ymin, ymax, xmin, xmax: int
            Bounding box x, y indices.
    """

    a = in1 == track_number
    b = in2 == track_number

    rows = np.any(a, axis=1)
    cols = np.any(a, axis=0)
    rmin1, rmax1 = np.where(rows)[0][[0, -1]]
    cmin1, cmax1 = np.where(cols)[0][[0, -1]]

    rows = np.any(b, axis=1)
    cols = np.any(b, axis=0)
    rmin2, rmax2 = np.where(rows)[0][[0, -1]]
    cmin2, cmax2 = np.where(cols)[0][[0, -1]]

    ymin = min(rmin1, rmin2)
    ymax = max(rmax1, rmax2)
    xmin = min(cmin1, cmin2)
    xmax = max(cmax1, cmax2)
    return ymin, ymax, xmin, xmax

def offset_to_speed(x, y, time_lag):
    """
    Return normalized speed assuming uniform grid.

    Args:
        x: np.array
            Movement in x-direction.
        y: np.array
            Movement in y-direction.
        time_lag: np.array
            Time lag of each file pair: one value per row for 2-D x and y [pairs, tracks],
            one value per element (same shape as x) for 1-D x and y.

    Returns:
        r_mag: np.array
            Movement magnitude.
        r_dir: np.array
            Movement direction.
        r_speed: np.array
            Movement speed.
    """
    # Movement in grid point units
    r_mag = np.sqrt(x**2 + y**2)
    # Movement direction
    # ATERNATIVE No.1 (effective but maybe.too.complex?)
    z = np.logical_and((y+x)==0, (y*x)==0)
    r_dir = -np.arctan2(y, x) *180/np.pi +90
    negative = r_dir < 0
    r_dir[negative] = r_dir[negative] + 360
    r_dir[z] = np.nan
    # Movement speed [n_grid / second]
    time_lag = np.asarray(time_lag)
    if r_mag.ndim == 2:
        time_lag = time_lag.reshape(-1, 1)
    r_speed = r_mag / time_lag
    return r_mag, r_dir, r_speed

def nearest_pair_index(pair_basetime, start_basetime):
    """
    Index of the file pair whose base time is nearest to each start time: the first minimum of
    np.abs(pair_basetime - start_time), i.e. np.nanargmin, for all start times at once.

    Args:
        pair_basetime: np.array, shape [npairs]
            Base time of each file pair [seconds].
        start_basetime: np.array, shape [ntracks]
            Start time of each track [seconds]; NaN gives -1.

    Returns:
        index: np.array(int64), shape [ntracks]
    """
    pair_basetime = np.asarray(pair_basetime, dtype=float)
    start_basetime = np.asarray(start_basetime, dtype=float)
    npairs = len(pair_basetime)
    index = np.full(start_basetime.shape, -1, dtype=np.int64)
    valid = np.isfinite(start_basetime)
    if npairs == 0 or not valid.any():
        return index
    order = np.argsort(pair_basetime, kind="stable")
    sorted_time = pair_basetime[order]
    pos = np.searchsorted(sorted_time, start_basetime[valid])
    left = np.clip(pos - 1, 0, npairs - 1)
    right = np.clip(pos, 0, npairs - 1)
    pick = np.where(np.abs(sorted_time[left] - start_basetime[valid]) <= np.abs(sorted_time[right] - start_basetime[valid]),
                    left, right)
    index[valid] = order[pick]
    return index


def align_movement_to_tracks(results, stats_start_basetime, ntimes, lag):
    """
    Movement of the tracks, from the measurements of every file pair to arrays [tracks, times] on the time axis of the
    track statistics (the first time of a track is the file pair nearest to its start time).

    Args:
        results: list of tuples
            One per file pair: (track_index, y_lag, x_lag, time_lag, base_time, pixel_length) of the tracks that
            were measured in that pair (movement_of_feature_fft with sparse=True).
        stats_start_basetime: np.array, shape [ntracks]
            Base time of the first time of every track.
        ntimes: int
            Length of the times dimension of the track statistics.
        lag: int
            Lag between the files of a pair (lag_for_speed).

    Returns:
        movement: dictionary of np.array(float32), shape [ntracks, ntimes], NaN where not measured:
            movement_distance [km], movement_speed [m/s], movement_theta [degrees],
            movement_distance_x, movement_distance_y [km]
    """
    ntracks = len(stats_start_basetime)
    npairs = len(results)
    out = {name: np.full((ntracks, ntimes), np.nan, dtype=np.float32) for name in
           ("movement_distance", "movement_speed", "movement_theta", "movement_distance_x", "movement_distance_y")}
    if npairs == 0:
        return out
    counts = np.array([len(r[0]) for r in results], dtype=np.int64)
    if counts.sum() == 0:
        return out
    pair_index = np.repeat(np.arange(npairs), counts)
    track = np.concatenate([np.asarray(r[0], dtype=np.int64) for r in results])
    y_lag = np.concatenate([np.asarray(r[1], dtype=float) for r in results])
    x_lag = np.concatenate([np.asarray(r[2], dtype=float) for r in results])
    pixel_length = np.concatenate([np.asarray(r[5], dtype=float) for r in results])
    time_lag = np.asarray([r[3] for r in results], dtype=float)[pair_index]
    pair_basetime = np.asarray([r[4] for r in results], dtype=float)

    # Movement magnitude [km], components [km], speed [m/s], direction (0 deg = North)
    r_mag, r_dir, r_speed = offset_to_speed(x_lag, y_lag, time_lag)
    movement_mag = r_mag * pixel_length / lag
    movement_x = x_lag * pixel_length / lag
    movement_y = y_lag * pixel_length / lag
    movement_speed = r_speed * pixel_length * 1000.

    # Time index of every entry in its track: pair index minus the pair index of the track start
    start = nearest_pair_index(pair_basetime, stats_start_basetime)
    k = pair_index - start[track]
    # A track's movement ends at its last finite speed (later pairs are not copied)
    finite = np.isfinite(movement_speed)
    last = np.full(ntracks, -1, dtype=np.int64)
    np.maximum.at(last, track[finite], pair_index[finite])
    keep = (start[track] >= 0) & (k >= 0) & (k < ntimes) & (pair_index <= last[track])
    t, k = track[keep], k[keep]
    out["movement_distance"][t, k] = movement_mag[keep]
    out["movement_speed"][t, k] = movement_speed[keep]
    out["movement_theta"][t, k] = r_dir[keep]
    out["movement_distance_x"][t, k] = movement_x[keep]
    out["movement_distance_y"][t, k] = movement_y[keep]
    return out


def define_movement_dataset(
        movement,
        lag,
        times_coord,
        times_dimname,
        tracks_coord,
        tracks_dimname,
):
    """
    Dataset of the movement variables ([tracks, times], same structure as the track statistics).

    Args:
        movement: dictionary
            Arrays of align_movement_to_tracks.
        lag: int
            Lag between the files of a pair (lag_for_speed).
        times_coord, tracks_coord: np.array
            Coordinates of the times and tracks dimensions.
        times_dimname, tracks_dimname: string
            Dimension names.
    """
    fillval_f = np.nan
    var_dict = movement
    var_attrs = {
        "movement_distance": {
            "long_name": "Movement distance along angle movement_theta",
            "units": "km",
            "_FillValue": fillval_f,
            "lag": lag,
            "comments": "This is the total movement along the angle theta between lag estimates.",
        },
        "movement_speed": {
            "long_name": "Movement speed along angle movement_theta",
            "units": "m/s",
            "_FillValue": fillval_f,
        },
        "movement_theta": {
            "long_name": "Movement direction",
            "units": "degrees",
            "_FillValue": fillval_f,
            "comments": "0:northward, 90:eastward, 180:southward, 270:westward.",
        },
        "movement_distance_x": {
            "long_name": "East-West component of movement distance",
            "units": "km",
            "_FillValue": fillval_f,
        },
        "movement_distance_y": {
            "long_name": "North-South component of movement distance",
            "units": "km",
            "_FillValue": fillval_f,
        },
    }
    # Define output variable dictionary
    varlist = {}
    for key, value in var_dict.items():
        if value.ndim == 2:
            varlist[key] = ([tracks_dimname, times_dimname], value, var_attrs[key])
    # Define coordinate list
    coordlist = {
        tracks_dimname: ([tracks_dimname], tracks_coord),
        times_dimname: ([times_dimname], times_coord),
    }
    # Define Dataset
    ds_vars = xr.Dataset(varlist, coords=coordlist)
    return ds_vars

def filter_interp_speed(dset, config):
    """
    Filter and interpolate high values of movement_speed.

    Tracks with fewer than 3 speed values below max_speed_thresh get NaN. For the others the values above the
    threshold are interpolated (quadratic) from the ones below it. Only the tracks that have any movement value are
    visited (the others are NaN already and stay NaN), with numpy slices.

    Args:
        dset: Xarray Dataset
            Dataset containing movement variables.
        config: dictionary
            Dictionary containing config parameters.

    Returns:
        dset: Xarray Dataset
            Update Dataset with filtered values.
    """

    tracks_dimname = config["tracks_dimname"]
    times_dimname = config["times_dimname"]
    max_speed_thresh = config["max_speed_thresh"]

    logger = logging.getLogger(__name__)

    fillval_f = np.nan
    times = dset[times_dimname].values
    m_mag = dset['movement_distance'].values
    m_speed = dset['movement_speed'].values
    m_theta = dset['movement_theta'].values
    m_x = dset['movement_distance_x'].values
    m_y = dset['movement_distance_y'].values
    m_mag_attrs = dset['movement_distance'].attrs
    m_speed_attrs = dset['movement_speed'].attrs
    m_theta_attrs = dset['movement_theta'].attrs
    m_x_attrs = dset['movement_distance_x'].attrs
    m_y_attrs = dset['movement_distance_y'].attrs

    for track in np.where(np.any(~np.isnan(m_speed), axis=1))[0]:
        speed = m_speed[track]
        mask_nan = ~np.isnan(speed)
        total_mask = mask_nan & (speed < max_speed_thresh)
        if np.count_nonzero(total_mask) < 3:
            logger.debug('Not enough values in Track %d' % track)
            m_mag[track] = fillval_f
            m_speed[track] = fillval_f
            m_theta[track] = fillval_f
            m_x[track] = fillval_f
            m_y[track] = fillval_f
        else:
            x = times[total_mask]
            xm = times[mask_nan]
            spd = speed[total_mask]
        # because THETA is now not being interpolated
            theta = m_theta[track][mask_nan]
            mag = m_mag[track][total_mask]
            intp_r = interp1d(x, spd, kind='quadratic', fill_value=fillval_f, bounds_error=False)
        # the interpolation of THETA is now not necessary (np.nan also implies no movement --NOW--)
            intp_mag = interp1d(x, mag, kind='quadratic', fill_value=fillval_f, bounds_error=False)

            # original formula from FELIPERIOSG (already checked as we're dealing with azimuths now)
            mov_x = intp_mag(xm) * np.sin(np.pi / 180.0 * theta)
            mov_y = intp_mag(xm) * np.cos(np.pi / 180.0 * theta)
            # NP.NAN here (in XM) means no movement (check ALTERNATIVE 1 in "offset_to_speed")
            mov_x[np.isnan(mov_x)] = 0
            mov_y[np.isnan(mov_y)] = 0

            # Interpolate values
            m_mag[track][mask_nan] = intp_mag(xm)
            m_speed[track][mask_nan] = intp_r(xm)
            # m_theta[track][mask_nan] = intp_theta(xm)
            m_x[track][mask_nan] = mov_x
            m_y[track][mask_nan] = mov_y

    # Update variables in dataset
    dset['movement_distance'] = ((tracks_dimname, times_dimname), m_mag, m_mag_attrs)
    dset['movement_speed'] = ((tracks_dimname, times_dimname), m_speed, m_speed_attrs)
    dset['movement_theta'] = ((tracks_dimname, times_dimname), m_theta, m_theta_attrs)
    dset['movement_distance_x'] = ((tracks_dimname, times_dimname), m_x, m_x_attrs)
    dset['movement_distance_y'] = ((tracks_dimname, times_dimname), m_y, m_y_attrs)

    return dset
