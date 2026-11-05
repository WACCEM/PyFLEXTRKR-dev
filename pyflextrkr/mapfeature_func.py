import numpy as np
import time
import os
import logging
import xarray as xr

from pyflextrkr.regionprops_func import conv_dilate_overwrite_km, redilate_cell_clusters, cluster_global_attrs, cluster_radii_from_cloudid

def map_feature(
        cloudid_filename,
        filebasetime,
        file_trackindex,
        file_cloudnumber,
        file_trackstatus,
        file_mergetracknumber,
        file_splittracknumber,
        file_mergecloudnumber,
        file_splitcloudnumber,
        trackstats_comments,
        config,
        pixeltracking_outpath,
        pixeltracking_filebase,
        file_featureclass=None,
        file_trackclass=None,
        file_mcsnumber=None,
):
    """
    Map track numbers to pixel level files for all feature tracking.

    For feature_type radar_feature with conv_dilate_overwrite_radius_km set, the cell_cluster_number pixel variable
    is recomputed at that radius (the clusters dilated further and labeled again: the groups regionprops_driver used
    for the statistics) instead of copied from the cloudid file.

    Args:
        cloudid_filename: string
            Cloudid file name.
        filebasetime: int
            Cloudid file base time.
        file_trackindex: np.array
            Track indices for the features in the cloudid file.
        file_cloudnumber: np.array
            Matched feature numbers in the cloudid file.
        file_trackstatus: np.array
            Track status for the features in the cloudid file.
        file_mergetracknumber: np.array
            Merge feature track number.
        file_splittracknumber: np.array
            Split feature track number.
        file_mergecloudnumber: np.array
            Merge feature cloud number in the cloudid file.
        file_splitcloudnumber: np.array
            Split feature cloud number in the cloudid file.
        trackstats_comments: string
            Track status explanation.
        config: dictionary
            Dictionary containing config parameters.
        pixeltracking_outpath: string
            Output directory for pixel-level files.
        pixeltracking_filebase: string
            Output pixel-level file basename.
        file_featureclass: np.array, optional, default=None
            Per-cloud feature classification for this time (e.g.
            identifymcs_radar.py's feature_class: 1=MCS, 2=multi-cell,
            3=single-cell, 4=stratiform). When provided, a
            feature_class pixel map is written, with merge/split fragment
            pixels taking their parent track's class.
        file_trackclass: np.array, optional, default=None
            Class of the track of each cloud (identifymcs_radar.py's track_class, the same value
            at every time of a track). When provided, a track_class pixel map is written in the
            same way as feature_class.
        file_mcsnumber: np.array, optional, default=None
            MCS number of the track of each cloud (identifymcs_radar.py's mcs_number: 1, 2, 3, ... for the MCS tracks of the
            statistics file, 0 for the other tracks). When provided, a mcs_number pixel map is written on the mask of every MCS
            track (own feature and the merge/split fragments folded into it), 0 (the _FillValue) elsewhere.

    Returns:
        tracksmap_outfile: string
            Track number pixel-level file name.
    """
    feature_varname = config.get("feature_varname", "feature_number")
    feature_type = config.get("feature_type", None)
    time_dimname = "time"
    y_dimname = "lat"
    x_dimname = "lon"
    fillval = config.get("fillval", -9999)

    np.set_printoptions(threshold=np.inf)
    logger = logging.getLogger(__name__)

    #########################################################################
    # Get cloudid file associated with this time
    file_datetime = time.strftime("%Y%m%d_%H%M%S", time.gmtime(np.copy(filebasetime)))
    # Load cloudid data
    ds_in = xr.open_dataset(
        cloudid_filename,
        decode_times=False,
        mask_and_scale=False
    )
    # Get data dimensions
    ny = ds_in.sizes[y_dimname]
    nx = ds_in.sizes[x_dimname]
    # Get dimension names from the file
    dims_file = []
    for key in ds_in.dims: dims_file.append(key)
    # Find extra dimensions beyond [time, y, x]
    dims_keep = [time_dimname, y_dimname, x_dimname]
    dims_drop = list(set(dims_file) - set(dims_keep))
    # Drop extra dimensions
    ds_in = ds_in.drop_dims(dims_drop)
    # Get necessary variables
    feature_number = ds_in[feature_varname].data.squeeze()


    ################################################################
    # Create map of status and track number for every feature in this file
    statusmap = np.full((1, ny, nx), fillval, dtype=int)
    trackmap = np.zeros((1, ny, nx), dtype=int)
    allmergemap = np.zeros((1, ny, nx), dtype=int)
    allsplitmap = np.zeros((1, ny, nx), dtype=int)

    trackmap_include_ms = np.zeros((1, ny, nx), dtype=int)
    trackmap_merge = np.zeros((1, ny, nx), dtype=int)
    trackmap_split = np.zeros((1, ny, nx), dtype=int)

    # Optional class maps, painted like the track numbers: the class of the feature at this time (feature_class) and the
    # constant class of its track (track_class); fragment pixels take the class of the track they are folded into
    class_values = {}
    if file_featureclass is not None:
        class_values["feature_class"] = file_featureclass
    if file_trackclass is not None:
        class_values["track_class"] = file_trackclass
    class_maps = {name: np.full((1, ny, nx), fillval, dtype=int) for name in class_values}
    # Optional MCS number map: the number of the MCS track on its mask (the extent of cloudtracknumber), 0 elsewhere
    mcsnumber_map = np.zeros((1, ny, nx), dtype=int) if file_mcsnumber is not None else None

    # Check number of matched features
    nmatchcloud = len(file_cloudnumber)
    if nmatchcloud > 0:
        ##############################################################
        # Loop over each instance matching the trackstats time
        for jj in range(0, nmatchcloud):
            # Get cloud number
            jjcloudnumber = file_cloudnumber[jj]
            jjstatus = file_trackstatus[jj]

            # Get the mask matching this feature number
            cmask = feature_number == jjcloudnumber

            # Label this feature with the track number.
            # Need to add one to the cloud number since have the index number and we want the track number
            if np.count_nonzero(cmask) > 0:
                trackmap[0, cmask] = file_trackindex[jj] + 1
                trackmap_include_ms[0, cmask] = file_trackindex[jj] + 1
                statusmap[0, cmask] = jjstatus
                for name, values in class_values.items():
                    class_maps[name][0, cmask] = values[jj]
                if mcsnumber_map is not None:
                    mcsnumber_map[0, cmask] = file_mcsnumber[jj]
            else:
                logger.warning(f"Warning: No matching cloud pixel found: {jjcloudnumber}")

            ###########################################################
            # Find splitting clouds
            jjsplit = np.where(file_splitcloudnumber[jj, :] > 0)[0]
            # Loop through splitting clouds if present
            if len(jjsplit) > 0:
                for isplit in jjsplit:
                    # Get mask matching the splitting cloud
                    is_number = file_splitcloudnumber[jj, isplit]
                    s_cmask = feature_number == is_number
                    # Label this cloud with the track number
                    if np.count_nonzero(s_cmask) > 0:
                        trackmap_include_ms[0, s_cmask] = file_trackindex[jj] + 1
                        trackmap_split[0, s_cmask] = file_trackindex[jj] + 1
                        # Split fragment pixels take their parent track's class
                        for name, values in class_values.items():
                            class_maps[name][0, s_cmask] = values[jj]
                        if mcsnumber_map is not None:
                            mcsnumber_map[0, s_cmask] = file_mcsnumber[jj]
                    else:
                        logger.warning(f"Warning: No matching splitting cloud found: {is_number}")

            ###########################################################
            # Find merging clouds
            jjmerge = np.where(file_mergecloudnumber[jj, :] > 0)[0]
            # Loop through merging clouds if present
            if len(jjmerge) > 0:
                for imerge in jjmerge:
                    # Get mask matching the merging cloud
                    im_number = file_mergecloudnumber[jj, imerge]
                    m_cmask = feature_number == im_number
                    # Label this cloud with the track number
                    if np.count_nonzero(m_cmask) > 0:
                        trackmap_include_ms[0, m_cmask] = file_trackindex[jj] + 1
                        trackmap_merge[0, m_cmask] = file_trackindex[jj] + 1
                        # Merge fragment pixels take their parent track's class
                        for name, values in class_values.items():
                            class_maps[name][0, m_cmask] = values[jj]
                        if mcsnumber_map is not None:
                            mcsnumber_map[0, m_cmask] = file_mcsnumber[jj]
                    else:
                        logger.warning(f"Warning: No matching merging cloud found: {im_number}")

            ###########################################################
            # Get split tracknumber within this time
            jjallsplit = file_splittracknumber[jj]
            # Count valid split cloudnumbers (> 0)
            splitpresent = np.count_nonzero(jjallsplit > 0)
            if splitpresent > 0:
                # Find valid split tracknumbers (> 0)
                splittracks = jjallsplit[jjallsplit > 0]
                splitcloudid = jjcloudnumber[jjallsplit > 0]
                if len(splittracks) > 0:
                    for isplit in range(0, len(splittracks)):
                        s_cmask = feature_number == splitcloudid[isplit]
                        allsplitmap[0, s_cmask] = splittracks[isplit]

            ###########################################################
            # Get merge tracknumber within this time
            jjallmerge = file_mergetracknumber[jj]
            # Count valid split cloudnumbers (> 0)
            mergepresent = np.count_nonzero(jjallmerge > 0)
            if mergepresent > 0:
                # Find valid merge tracknumbers (> 0)
                mergetracks = jjallmerge[jjallmerge > 0]
                mergecloudid = jjcloudnumber[jjallmerge > 0]
                if len(mergetracks) > 0:
                    for imerge in range(0, len(mergetracks)):
                        m_cmask = feature_number == mergecloudid[imerge]
                        allmergemap[0, m_cmask] = mergetracks[imerge]


    # Handle special variables for specific feature_type
    if "tb_pf" in feature_type:
        pcp_thresh = config["pcp_thresh"]
        precipitation = ds_in["precipitation"]
        pcptrackmap = (precipitation.data > pcp_thresh) * trackmap_include_ms
    if "radar3d" in feature_type:
        dbz_thresh = config["dbz_thresh"]
        reflectivity = ds_in["reflectivity_comp"]
        dbztrackmap = (reflectivity.data > dbz_thresh) * trackmap_include_ms
    # Exact match (not substring), to avoid any ambiguity with the "radar3d"
    # substring check above, which belongs to the separate Tb+SL3D pipeline.
    if feature_type == "radar_feature":
        dbz_thresh = config.get("dbz_thresh", 10)
        ref_varname = config.get("ref_varname", "dbz_comp")
        reflectivity = ds_in[ref_varname]
        dbztrackmap = (reflectivity.data > dbz_thresh) * trackmap_include_ms



    #####################################################################
    # Add track number variables to output dataset

    # Make a dictionary new variable coordinates
    coords = {
        time_dimname: ([time_dimname], ds_in[time_dimname].data, ds_in[time_dimname].attrs),
        y_dimname: ([y_dimname], ds_in[y_dimname].data, ds_in[y_dimname].attrs),
        x_dimname: ([x_dimname], ds_in[x_dimname].data, ds_in[x_dimname].attrs),
    }
    # Make attributes for new variables
    trackmap_attrs = {
        "long_name": "Track number in this file at a given pixel",
        "units": "unitless",
        "_FillValue": 0,
    }
    allmergemap_attrs = {
        "long_name": "Track number where this track merges with",
        "units": "unitless",
        "_FillValue": 0,
    }
    allsplitmap_attrs = {
        "long_name": "Track number where this track splits from",
        "units": "unitless",
        "_FillValue": 0,
    }
    statusmap_attrs = {
        "long_name": "Flag indicating history of track",
        "units": "unitless",
        "_FillValue": fillval,
        "comments": trackstats_comments,
    }
    trackmap_ms_attrs = {
        "long_name": "Track number including merge/split",
        "units": "unitless",
        "_FillValue": 0,
        "comments": "This mask renumbers small merge/split tracks to the main track (e.g., MCS)",
    }
    trackmap_m_attrs = {
        "long_name": "Track number for small merge tracks",
        "units": "unitless",
        "_FillValue": 0,
        "comments": "This mask marks small merge tracks renumbered to the main track (e.g., MCS)",
    }
    trackmap_s_attrs = {
        "long_name": "Track number for small split tracks",
        "units": "unitless",
        "_FillValue": 0,
        "comments": "This mask marks small split tracks renumbered to the main track (e.g., MCS)",
    }
    # Convert numpy arrays to Xarray DataArrays
    trackmap = xr.DataArray(trackmap, coords=coords, dims=dims_keep, attrs=trackmap_attrs)
    allmergemap = xr.DataArray(allmergemap, coords=coords, dims=dims_keep, attrs=allmergemap_attrs)
    allsplitmap = xr.DataArray(allsplitmap, coords=coords, dims=dims_keep, attrs=allsplitmap_attrs)
    statusmap = xr.DataArray(statusmap, coords=coords, dims=dims_keep, attrs=statusmap_attrs)
    trackmap_include_ms = xr.DataArray(trackmap_include_ms, coords=coords, dims=dims_keep, attrs=trackmap_ms_attrs)
    trackmap_merge = xr.DataArray(trackmap_merge, coords=coords, dims=dims_keep, attrs=trackmap_m_attrs)
    trackmap_split = xr.DataArray(trackmap_split, coords=coords, dims=dims_keep, attrs=trackmap_s_attrs)
    if "tb_pf" in feature_type:
        pcptrackmap_attrs = {
            "long_name": "Track number including merge/split for precipitation",
            "units": "unitless",
            "_FillValue": 0,
            "precipitation_threshold": pcp_thresh,
        }
        pcptrackmap = xr.DataArray(pcptrackmap, coords=coords, dims=dims_keep, attrs=pcptrackmap_attrs)
    if "radar3d" in feature_type:
        dbztrackmap_attrs = {
            "long_name": "Track number including merge/split for composite reflectivity",
            "units": "unitless",
            "_FillValue": 0,
            "reflectivity_threshold": dbz_thresh,
        }
        dbztrackmap = xr.DataArray(dbztrackmap, coords=coords, dims=dims_keep, attrs=dbztrackmap_attrs)
    if feature_type == "radar_feature":
        dbztrackmap_attrs = {
            "long_name": "Track number including merge/split for composite reflectivity",
            "units": "unitless",
            "_FillValue": 0,
            "reflectivity_threshold": dbz_thresh,
        }
        dbztrackmap = xr.DataArray(dbztrackmap, coords=coords, dims=dims_keep, attrs=dbztrackmap_attrs)
        class_attrs = {
            "feature_class": {
                "long_name": "Feature class at this time: 1=MCS, 2=multi-cell, 3=single-cell, 4=stratiform",
                "comments": "The class of the feature at this time: 1 only at the times of a qualifying MCS period. "
                            "Merge/split fragment pixels take their parent track's class.",
            },
            "track_class": {
                "long_name": "Track class: 1=MCS, 2=multi-cell, 3=single-cell, 4=stratiform",
                "comments": "The most organized class of the track at any time, the same value at every time of the track. "
                            "Merge/split fragment pixels take their parent track's class.",
            },
        }
        class_maps = {
            name: xr.DataArray(arr, coords=coords, dims=dims_keep, attrs=dict(class_attrs[name], units="unitless", _FillValue=fillval))
            for name, arr in class_maps.items()
        }
        if mcsnumber_map is not None:
            mcsnumber_map = xr.DataArray(mcsnumber_map, coords=coords, dims=dims_keep, attrs={
                "long_name": "MCS number: the sequential number of the MCS track (mcs_number of the statistics file)",
                "units": "unitless", "_FillValue": 0,
                "comments": "The number of the MCS track on its mask (own feature and the merge/split fragments folded into it, "
                            "the extent of cloudtracknumber); 0 (the _FillValue) elsewhere."})

    # Create a copy of the input dataset
    ds_out = ds_in.copy(deep=True)

    # Assign new variables to output dataset
    ds_out = ds_out.assign(tracknumber=trackmap)
    ds_out = ds_out.assign(merge_tracknumber=allmergemap)
    ds_out = ds_out.assign(split_tracknumber=allsplitmap)
    ds_out = ds_out.assign(track_status=statusmap)
    ds_out = ds_out.assign(cloudtracknumber=trackmap_include_ms)
    ds_out = ds_out.assign(cloudmerge_tracknumber=trackmap_merge)
    ds_out = ds_out.assign(cloudsplit_tracknumber=trackmap_split)
    if "tb_pf" in feature_type:
        ds_out = ds_out.assign(pcptracknumber=pcptrackmap)
    if "radar3d" in feature_type:
        ds_out = ds_out.assign(dbztracknumber=dbztrackmap)
    if feature_type == "radar_feature":
        ds_out = ds_out.assign(dbztracknumber=dbztrackmap)
        for name, class_map in class_maps.items():
            ds_out = ds_out.assign({name: class_map})
        if mcsnumber_map is not None:
            ds_out = ds_out.assign(mcs_number=mcsnumber_map)
        # The statistics were computed with the cell clusters dilated further (regionprops_driver): give the pixel files the
        # same groups, not the clusters of the cloudid file
        overwrite_km = conv_dilate_overwrite_km(config)
        if overwrite_km is not None:
            if "cell_cluster_number" not in ds_in:
                raise ValueError(f"conv_dilate_overwrite_radius_km needs cell_cluster_number, which {cloudid_filename} does not "
                                 f"have: run idfeature with radii_expand (default [1, 2, 3, 4, 5, 6, 7] km, null = off).")
            groups = redilate_cell_clusters(ds_in["cell_cluster_number"].values.squeeze(), overwrite_km, config)
            groups_attrs = dict(ds_in["cell_cluster_number"].attrs)
            groups_attrs["dilation_radius_km"] = overwrite_km
            groups_attrs["comments"] = (f"Recomputed from the cell_cluster_number of the cloudid file by mapfeature "
                                        f"(conv_dilate_overwrite_radius_km = {overwrite_km:g} km): the clusters dilated further "
                                        f"by that radius and labeled again (8-connected, sorted by size)")
            ds_out = ds_out.assign(cell_cluster_number=xr.DataArray(
                groups[np.newaxis], coords=coords, dims=dims_keep, attrs=groups_attrs))
            ds_out.attrs.update(cluster_global_attrs(cluster_radii_from_cloudid(ds_in), overwrite_km))
    # Update global attributes
    ds_out.attrs["Title"] = "Pixel-level feature tracking data"
    ds_out.attrs["Created_on"] = time.ctime(time.time())

    #####################################################################
    # Output to netcdf file

    # Define output filename
    tracksmap_outfile = (
        pixeltracking_outpath +
        pixeltracking_filebase +
        file_datetime + ".nc"
    )

    # Delete file if it already exists
    if os.path.isfile(tracksmap_outfile):
        os.remove(tracksmap_outfile)

    # Set encoding/compression for all variables
    comp = dict(zlib=True)
    encoding = {var: comp for var in ds_out.data_vars}
    # Write to netCDF file
    # Touch output file so HDF5 1.14.x H5Fis_accessible() probe does not
    # fail with ENOENT and emit spurious diagnostics to stderr
    open(tracksmap_outfile, 'ab').close()
    ds_out.to_netcdf(
        path=tracksmap_outfile,
        mode="w",
        format="NETCDF4",
        unlimited_dims="time",
        encoding=encoding,
    )
    logger.info(f"{tracksmap_outfile}")

    return tracksmap_outfile
