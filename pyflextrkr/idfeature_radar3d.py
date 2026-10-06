import logging
import numpy as np
import xarray as xr
from pyflextrkr import netcdf_io as net
from pyflextrkr.idcells_reflectivity import get_composite_reflectivity_generic, subset_domain
from pyflextrkr.label_and_grow_features import label_and_grow_features
from pyflextrkr.steiner_func import make_dilation_step_func, mod_steiner_classification, label_cells_fast
from pyflextrkr.echotop_func import echotop_height, echotop_height_fast
from pyflextrkr.ft_utilities import get_pixel_area


def idfeature_radar3d(
    input_data,
    config,
):
    """
    Identifies MCS candidate convective-system features from Cartesian 3D (or 2D)
    radar reflectivity.

    This follows the same detect-and-grow approach used for Tb cold-cloud-shield
    tracking (label_and_grow_features), applied to column-maximum (or 2D composite)
    radar reflectivity, plus the Steiner convective/stratiform classification
    (following the cell-tracking identification parameters) and echo-top heights.

    Args:
        input_data: string or Xarray DataSet
            For input_format=='netcdf': filename of the input data (may contain
            multiple times), or an already-open Xarray DataSet (e.g., a single-time
            slice handed in by idfeature_driver's "combine_files" batching path).
            For input_format=='zarr': filename of a Cartesian Zarr store, or an
            already-open Xarray DataSet.
        config: dictionary
            Dictionary containing config parameters.

    Returns:
        cloudid_outfile: string, list of strings, or None
            Cloudid file name(s) written. None if every time in this call was
            skipped (e.g., by the optional miss_thresh guard).
    """
    logger = logging.getLogger(__name__)
    logger.debug(f"Processing {input_data}.")

    # If input_data is an already-open Dataset (e.g. a single-time slice
    # handed in by idfeature_driver's combine_files batching path), it may be
    # backed by dask arrays (from xr.open_mfdataset). Materialize it to plain
    # numpy before running the scipy/astropy-based feature identification
    # below, which requires in-memory arrays. The synchronous scheduler keeps
    # this simple and avoids any nested-scheduler complications when this
    # function runs inside a dask worker task (run_parallel>=1).
    if isinstance(input_data, xr.Dataset):
        import dask
        with dask.config.set(scheduler="synchronous"):
            input_data = input_data.load()

    tracking_outpath = config["tracking_outpath"]
    cloudid_filebase = config["cloudid_filebase"]
    pixel_radius = config["pixel_radius"]
    geolimits = config.get("geolimits", None)
    dx = config["dx"]
    dy = config["dy"]
    radar_sensitivity = config["radar_sensitivity"]
    z_dimname = config.get("z_dimname", "z")

    # Feature labeling and growth parameters (detect-and-grow on column-max
    # reflectivity, same approach as idclouds_tbpf/label_and_grow_features)
    feature_dbz_threshs = config.get("feature_dbz_threshs", [30, 10])
    core_thresh = feature_dbz_threshs[0]
    secondary_thresh = feature_dbz_threshs[1]
    smooth_size = config.get("smooth_size", 5)
    min_core_npix = config.get("min_core_npix", 4)
    area_thresh = config.get("area_thresh", 36)
    core_operator = config.get("core_operator", "gt")
    growth_method = config.get("growth_method", "edt")
    expand_to_tertiary = config.get("feature_expand_tertiary", 0)
    feature_field_floor = config.get("feature_field_floor", None)
    miss_thresh = config.get("miss_thresh", None)

    # Steiner convective/stratiform classification parameters (follow the
    # cell-tracking identification config parameters)
    absConvThres = config["absConvThres"]
    minZdiff = config["minZdiff"]
    truncZconvThres = config["truncZconvThres"]
    mindBZuse = config["mindBZuse"]
    dBZforMaxConvRadius = config["dBZforMaxConvRadius"]
    conv_rad_increment = config["conv_rad_increment"]
    conv_rad_start = config["conv_rad_start"]
    bkg_refl_increment = config["bkg_refl_increment"]
    maxConvRadius = config["maxConvRadius"]
    weakEchoThres = config["weakEchoThres"]
    bkgrndRadius = config["bkgrndRadius"]
    min_corearea = config.get("min_corearea", 0)
    min_cellarea = config.get("min_cellarea", 0)
    remove_smallcores = config.get("remove_smallcores", True)
    remove_smallcells = config.get("remove_smallcells", False)
    return_diag = config.get("return_diag", False)
    convolve_method = config.get("convolve_method", "fft")
    dilate_method = config.get("dilate_method", "orig")
    # cell_number is a labeled, size-sorted convective cell ID array derived
    # from a Steiner diagnostic mask (see Design decision 1 in the project plan):
    # 'score_orig' (pre small-core-area filter) or 'score_dilate' (post min_cellarea
    # filter + convective-radius dilation).
    cell_number_source = config.get("cell_number_source", "score_orig")
    cell_min_npix = config.get("cell_min_npix", 0)

    # Echo-top height parameters
    echotop_gap = config["echotop_gap"]
    echotop_method = config.get("echotop_method", "orig")

    # Set echo classification type values (same convention as idcells_reflectivity.py)
    types_steiner = {
        'NO_SURF_ECHO': 1,
        'WEAK_ECHO': 2,
        'STRATIFORM': 3,
        'CONVECTIVE': 4,
    }

    # Get composite reflectivity using the shared generic reader (reused from
    # idcells_reflectivity.py: handles 3D/2D input, vertical-coordinate
    # standardization, terrain, netcdf/HEALPix-zarr/in-memory-Dataset input)
    comp_dict = get_composite_reflectivity_generic(input_data, config)
    is_3d = comp_dict.get('is_3d', True)

    # Subset domain
    if geolimits is not None:
        comp_dict = subset_domain(comp_dict, geolimits, dx, dy)

    # Get time coordinate to check if we need to loop over multiple times
    time_coords_all = comp_dict['time_coords']
    ntimes = get_length(time_coords_all)

    cloudid_outfiles = []
    for itime in range(ntimes):
        # Extract variables for this time step
        if ntimes > 1:
            time_coords = time_coords_all.isel(time=itime)
            dbz3d_filt = comp_dict['dbz3d_filt'].isel(time=itime)
            dbz_comp = comp_dict['dbz_comp'].isel(time=itime)
            dbz_lowlevel = comp_dict['dbz_lowlevel'].isel(time=itime)
            refl = comp_dict['refl'][itime]
        else:
            time_coords = time_coords_all
            dbz3d_filt = comp_dict['dbz3d_filt']
            dbz_comp = comp_dict['dbz_comp']
            dbz_lowlevel = comp_dict['dbz_lowlevel']
            refl = comp_dict['refl']

        x_coords = comp_dict['x_coords']
        y_coords = comp_dict['y_coords']
        grid_lat = comp_dict['grid_lat']
        grid_lon = comp_dict['grid_lon']
        mask_goodvalues = comp_dict['mask_goodvalues']

        # Handle height coordinate (can be 1D, 3D, or 4D)
        height = comp_dict['height']
        if ntimes > 1 and height.ndim == 4:
            height = height[itime, :, :, :]

        dbz_comp_vals = dbz_comp.data if hasattr(dbz_comp, 'data') else dbz_comp
        shape_2d = dbz_comp_vals.shape

        # Optional missing-data guard: skip frames with too much missing coverage
        # rather than crashing (see feedback_guardrails_nan_not_crash)
        if miss_thresh is not None:
            missingfrac = np.count_nonzero(np.isnan(dbz_comp_vals)) / dbz_comp_vals.size
            if missingfrac > miss_thresh:
                logger.info(
                    f"Frame skipped: missing data fraction {missingfrac:.3f} "
                    f"exceeds miss_thresh {miss_thresh}."
                )
                continue

        # Get pixel area (scalar for 'fixed', 2D array for 'latlon')
        pixel_area = get_pixel_area(config)

        #####################################################################
        # 1. Feature identification: detect-and-grow on column-max reflectivity
        #####################################################################
        # Floor the no-echo/below-sensitivity value (keep NaN as-is; NaN must
        # not be averaged into the smoothing window). Defaults to
        # radar_sensitivity, matching the floor already applied to the
        # Steiner "refl" input by get_composite_reflectivity_generic.
        floor_val = feature_field_floor if feature_field_floor is not None else radar_sensitivity
        field = np.copy(dbz_comp_vals).astype(float)
        nan_mask = np.isnan(field)
        field[~nan_mask & (field < floor_val)] = floor_val

        featuredata = label_and_grow_features(
            field,
            pixel_radius,
            [core_thresh, secondary_thresh, secondary_thresh, secondary_thresh],
            area_thresh,
            min_core_npix,
            smooth_size,
            expand_to_tertiary,
            config,
            pixel_area=pixel_area,
            core_operator=core_operator,
            growth_method=growth_method,
        )
        feature_number = featuredata["final_Feature_Number"]
        feature_pixel_type = featuredata["final_Feature_Type"]
        nfeatures = int(featuredata["final_nFeature"])
        # Recompute npix_feature directly from the labeled array (bincount),
        # rather than trusting the core/secondary-flag-derived counts, since
        # pixels exactly at the secondary threshold are included by growth
        # but not necessarily counted by the flags (see plan Callout under
        # Finding 1, "caveat b").
        if nfeatures > 0:
            npix_feature = np.bincount(
                feature_number.ravel(), minlength=nfeatures + 1
            )[1:nfeatures + 1].astype(int)
        else:
            npix_feature = np.zeros(0, dtype=int)

        #####################################################################
        # 2. Convective/stratiform classification (Steiner) + convective cells
        #####################################################################
        bkg_bin, conv_rad_bin = make_dilation_step_func(
            mindBZuse,
            dBZforMaxConvRadius,
            bkg_refl_increment,
            conv_rad_increment,
            conv_rad_start,
            maxConvRadius,
        )
        steiner_result = mod_steiner_classification(
            types_steiner, refl, mask_goodvalues, dx, dy,
            bkg_rad=bkgrndRadius * 1000,
            minZdiff=minZdiff,
            absConvThres=absConvThres,
            truncZconvThres=truncZconvThres,
            weakEchoThres=weakEchoThres,
            bkg_bin=bkg_bin,
            conv_rad_bin=conv_rad_bin,
            min_corearea=min_corearea,
            min_cellarea=min_cellarea,
            remove_smallcores=remove_smallcores,
            remove_smallcells=remove_smallcells,
            # score_orig (the diagnostic pre-small-core-filter core mask) is
            # always needed to build cell_number, regardless of the
            # user-facing 'return_diag' config flag (which only controls
            # whether refl_bkg/peakedness are ALSO written to the output file).
            return_diag=True,
            convolve_method=convolve_method,
            dilate_method=dilate_method,
        )
        convsf = steiner_result['sclass']
        conv_core = steiner_result['score']
        score_orig = steiner_result['score_orig']
        score_dilate = steiner_result['score_dilate']

        cell_source_array = score_orig if cell_number_source == 'score_orig' else score_dilate
        cell_number, _ = label_cells_fast(cell_source_array, cell_min_npix)

        diag_kwargs = {}
        if return_diag:
            diag_kwargs['refl_bkg'] = steiner_result['refl_bkg']
            diag_kwargs['peakedness'] = steiner_result['peakedness']

        #####################################################################
        # 3. Echo-top heights
        #####################################################################
        if not is_3d:
            logger.info("Input data is 2D, skipping echo-top height calculations")
            echotop10 = np.full(shape_2d, np.nan, dtype=np.float32)
            echotop20 = np.full(shape_2d, np.nan, dtype=np.float32)
            echotop30 = np.full(shape_2d, np.nan, dtype=np.float32)
            echotop40 = np.full(shape_2d, np.nan, dtype=np.float32)
            echotop50 = np.full(shape_2d, np.nan, dtype=np.float32)
        else:
            _echotop_fn = echotop_height_fast if echotop_method == 'fast' else echotop_height
            echotop10 = _echotop_fn(dbz3d_filt, height, z_dimname, shape_2d,
                                     dbz_thresh=10, gap=echotop_gap, min_thick=0)
            echotop20 = _echotop_fn(dbz3d_filt, height, z_dimname, shape_2d,
                                     dbz_thresh=20, gap=echotop_gap, min_thick=0)
            echotop30 = _echotop_fn(dbz3d_filt, height, z_dimname, shape_2d,
                                     dbz_thresh=30, gap=echotop_gap, min_thick=0)
            echotop40 = _echotop_fn(dbz3d_filt, height, z_dimname, shape_2d,
                                     dbz_thresh=40, gap=echotop_gap, min_thick=0)
            echotop50 = _echotop_fn(dbz3d_filt, height, z_dimname, shape_2d,
                                     dbz_thresh=50, gap=echotop_gap, min_thick=0)

        del dbz3d_filt

        feature_params = {
            'core_dbz_thresh': core_thresh,
            'secondary_dbz_thresh': secondary_thresh,
            'smooth_size': smooth_size,
            'min_core_npix': min_core_npix,
            'area_thresh': area_thresh,
            'growth_method': growth_method,
            'cell_number_source': cell_number_source,
        }
        steiner_params = {
            'dx': dx,
            'dy': dy,
            'absConvThres': absConvThres,
            'minZdiff': minZdiff,
            'truncZconvThres': truncZconvThres,
            'mindBZuse': mindBZuse,
            'dBZforMaxConvRadius': dBZforMaxConvRadius,
            'conv_rad_increment': conv_rad_increment,
            'conv_rad_start': conv_rad_start,
            'bkg_refl_increment': bkg_refl_increment,
            'maxConvRadius': maxConvRadius,
            'weakEchoThres': weakEchoThres,
            'bkgrndRadius': bkgrndRadius,
            'min_corearea': min_corearea,
            'min_cellarea': min_cellarea,
            'echotop_gap': echotop_gap,
        }

        #####################################################################
        # Get date/time and make output filename (same pattern as
        # idcells_reflectivity.py / idclouds_tbpf.py)
        #####################################################################
        if ntimes > 1:
            timestamp = time_coords
        else:
            timestamp = time_coords[0]
        iTimestamp = pd_to_datetime(timestamp)
        file_basetime = np.array([iTimestamp])
        file_datestring = timestamp.dt.strftime("%Y%m%d").item()
        file_timestring = timestamp.dt.strftime("%H%M%S").item()
        cloudid_outfile = (
            tracking_outpath +
            cloudid_filebase +
            file_datestring +
            "_" +
            file_timestring +
            ".nc"
        )

        out_basetime = np.zeros(1, dtype=float)
        out_basetime[0] = file_basetime[0]
        out_nfeatures = np.zeros(1, dtype=int)
        out_nfeatures[0] = nfeatures

        ds_pass = comp_dict.get('ds_pass', None)
        if ds_pass is not None and ntimes > 1:
            ds_pass = ds_pass.isel(time=itime)

        #####################################################################
        # Write output to netCDF file (always written, even if nfeatures=0 -
        # see Design decision 3: a missing/skipped file would otherwise be
        # treated by gettracks/tracksingle as a bridgeable data gap, while an
        # echo-free timestep should cleanly end tracks instead)
        #####################################################################
        net.write_radar_featureid(
            cloudid_outfile,
            out_basetime,
            file_datestring,
            file_timestring,
            grid_lon,
            grid_lat,
            x_coords,
            y_coords,
            dbz_comp,
            dbz_lowlevel,
            convsf,
            conv_core,
            cell_number,
            feature_pixel_type,
            echotop10,
            echotop20,
            echotop30,
            echotop40,
            echotop50,
            feature_number,
            npix_feature,
            out_nfeatures,
            input_data,
            config,
            feature_params=feature_params,
            steiner_params=steiner_params,
            ds_pass=ds_pass,
            **diag_kwargs,
        )
        logger.info(f"{cloudid_outfile}")
        cloudid_outfiles.append(cloudid_outfile)

    if len(cloudid_outfiles) == 0:
        return None
    elif ntimes == 1:
        return cloudid_outfiles[0]
    else:
        return cloudid_outfiles


def get_length(var):
    try:
        return len(var)
    except TypeError:
        return 1


def pd_to_datetime(timestamp):
    """
    Convert a time coordinate value to Epoch seconds (float), in a way that
    is robust to non-standard (e.g. cftime) calendars.
    """
    import pandas as pd
    iTimestamp = pd.to_datetime(timestamp.dt.strftime("%Y-%m-%dT%H:%M:%S").item())
    return (iTimestamp - pd.Timestamp('1970-01-01T00:00:00')).total_seconds()
