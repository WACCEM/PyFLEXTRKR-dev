"""
Per-cloudid-file worker of the second pass of the radar_feature MCS statistics (see regionprops_driver.py).

identifymcs_radar folds the merge/split fragments into the track they belong to, so the mask of a track at one time is
its own feature plus the fragment features: the "merged mask" that mapfeature paints as cloudtracknumber. The shape and
extreme statistics of trackstats_func.py describe the track's own feature only. This worker recomputes them on the merged
mask, with the same definitions as trackstats_func.py, plus a convective length that groups nearby convective cells
(cell_cluster_number, see idfeature_radar3d.py: the connected groups of the inflated cells). The groups are those idfeature
wrote, or, with conv_dilate_overwrite_radius_km set, the clusters dilated further by that radius and labeled again.
"""
import logging
import warnings
import numpy as np
import xarray as xr
from skimage.measure import regionprops

from pyflextrkr.ftfunctions import circular_mean
from pyflextrkr.ft_utilities import get_pixel_area
from pyflextrkr.steiner_func import dilate_label_conv_mask

ECHOTOP_THRESHS = (10, 20, 30, 40, 50)


def conv_dilate_overwrite_km(config):
    """
    The conv_dilate_overwrite_radius_km option: the radius [km] by which the cell clusters (cell_cluster_number) of the
    cloudid files are dilated further and labeled again, to connect adjacent clusters, instead of using them as they are.

    Args:
        config: dictionary
            Dictionary containing config parameters.

    Returns:
        radius_km: float, or None if the key is missing or null (use the clusters idfeature_radar3d wrote).

    Raises:
        ValueError: the radius is not a non-negative number, or dx/dy (grid spacing [m], the sampling of the dilation)
            are missing from the config.
    """
    radius = config.get("conv_dilate_overwrite_radius_km", None)
    if radius is None:
        return None
    if isinstance(radius, bool) or not isinstance(radius, (int, float, np.integer, np.floating)) \
            or not np.isfinite(radius) or radius < 0:
        raise ValueError(f"conv_dilate_overwrite_radius_km must be a non-negative number [km] or null, got {radius!r}.")
    if config.get("dx") is None or config.get("dy") is None:
        raise ValueError("conv_dilate_overwrite_radius_km needs dx and dy (grid spacing [m]) in the config.")
    return float(radius)


def track_classes_from_config(config):
    """
    The regionprops_track_classes option: the track_class values whose tracks are measured on the merged mask.

    Args:
        config: dictionary
            Dictionary containing config parameters.

    Returns:
        classes: list of int, or None if the key is missing or null (measure all tracks).

    Raises:
        ValueError: the value is not a non-empty list of integers.
    """
    classes = config.get("regionprops_track_classes", None)
    if classes is None:
        return None
    if isinstance(classes, (list, tuple)) and len(classes) > 0 and all(
            isinstance(c, (int, np.integer)) and not isinstance(c, bool) for c in classes):
        return sorted(int(c) for c in classes)
    raise ValueError(f"regionprops_track_classes must be a non-empty list of track_class values or null, got {classes!r}.")


def redilate_cell_clusters(cell_cluster_number, radius_km, config):
    """
    The cell clusters at a larger distance: the binary cell_cluster_number (> 0) dilated by a disc of radius_km and labeled
    (8-connected, sorted by size), so clusters that are less than 2 x radius_km apart become one group. Its valid-data gate
    (mask_goodvalues, the range mask of the radar) is not stored in the cloudid files and is not applied here.

    Args:
        cell_cluster_number: 2D np.array, cell clusters of the cloudid file (0 = not in a cluster).
        radius_km: float, dilation radius [km] (0 = only clusters that touch are joined, the same labeling as the file's).
        config: dictionary with dx and dy [m].

    Returns:
        2D np.array(int32), group number at every pixel of the dilated mask, sorted by size (largest = 1), 0 = none.
    """
    return dilate_label_conv_mask(cell_cluster_number > 0, radius_km * 1000.0, config["dx"], config["dy"])


def cluster_radii_from_cloudid(ds_cloudid):
    """
    radii_expand [km] that a cloudid file was made with.

    Args:
        ds_cloudid: xarray.Dataset
            An open cloudid file.

    Returns:
        radii_expand_km: np.array, or None if the file does not record it (the global attribute, else the attribute of
            cell_number_inflated).
    """
    radii = ds_cloudid.attrs.get("radii_expand_km", None)
    if radii is None and "cell_number_inflated" in ds_cloudid:
        radii = ds_cloudid["cell_number_inflated"].attrs.get("radii_expand_km", None)
    return None if radii is None else np.atleast_1d(np.asarray(radii, dtype=float))


def cluster_groups_text(radii_expand_km, overwrite_km):
    """
    The words that say what the convective groups of conv_majoraxis are, for the comments of the variable.

    Args:
        radii_expand_km: array-like, or None
            radii_expand [km] the cloudid files were made with (the attribute of the files; None if unknown).
        overwrite_km: float, or None
            conv_dilate_overwrite_radius_km (None = off).

    Returns:
        text: string
    """
    text = "cell_cluster_number (the 8-connected groups of the inflated convective cells"
    if radii_expand_km is not None:
        radii = ", ".join(f"{float(r):g}" for r in np.atleast_1d(radii_expand_km))
        text += f": the cells grown by radii_expand = [{radii}] km"
    text += ")"
    if overwrite_km is not None:
        text += (f", dilated further by a disc of radius {float(overwrite_km):g} km and labeled again "
                 f"(conv_dilate_overwrite_radius_km, run by run_regionprops, not read from the cloudid files)")
    return text


def cluster_global_attrs(radii_expand_km, overwrite_km):
    """
    The global attributes that record how the convective groups were made.

    Args:
        radii_expand_km: array-like, or None
            radii_expand [km] of the cloudid files (None = unknown, no attribute).
        overwrite_km: float, or None
            conv_dilate_overwrite_radius_km (None = off, no attribute).

    Returns:
        attrs: dictionary with radii_expand_km (float array) and conv_dilate_overwrite_radius_km (float), the ones that are set.
    """
    attrs = {}
    if radii_expand_km is not None:
        attrs["radii_expand_km"] = np.atleast_1d(np.asarray(radii_expand_km, dtype=float))
    if overwrite_km is not None:
        attrs["conv_dilate_overwrite_radius_km"] = float(overwrite_km)
    return attrs


def recomputed_variable_names(dbz_area_threshs, with_conv_diameter_sum=False):
    """
    Names of the track statistics that are recomputed on the merged mask (same names as in trackstats_func.py).

    Args:
        dbz_area_threshs: list
            dBZ thresholds of the area_<T>dbz / conv_area_<T>dbz variables.
        with_conv_diameter_sum: bool (default: False)
            Add conv_diameter_sum, the sum of the circular-equivalent diameters of the convective cells (needs cell_number in
            the cloudid files). False for first-pass files from before that variable existed; regionprops_singlefile returns it
            in either case.

    Returns:
        names: list of strings
    """
    names = [
        "meanlat", "meanlon", "mean_x", "mean_y",
        "lat_weightedcentroid", "lon_weightedcentroid", "x_weightedcentroid", "y_weightedcentroid",
        "max_dbz", "majoraxis", "minoraxis", "aspectratio", "orientation", "eccentricity", "conv_majoraxis",
    ]
    if with_conv_diameter_sum:
        names.append("conv_diameter_sum")
    names += [f"maxETH_{t}dbz" for t in ECHOTOP_THRESHS]
    for t in dbz_area_threshs:
        names += [f"area_{t}dbz", f"conv_area_{t}dbz"]
    return names


def regionprops_singlefile(
        cloudidfile,
        file_cloudnumber,
        file_mergecloudnumber,
        file_splitcloudnumber,
        config,
        compute_track=None,
):
    """
    Statistics of the merged mask of every track that has a feature in one cloudid file.

    Args:
        cloudidfile: string
            Cloudid file name.
        file_cloudnumber: np.array(int), shape [n]
            Feature number (cloudnumber) of each of the n tracks' own feature in this file.
        file_mergecloudnumber, file_splitcloudnumber: np.array(int), shape [n, m]
            Feature numbers of the merge/split fragments folded into each track at this time
            (<= 0 = no fragment in that slot), as in mapfeature_driver.
        config: dictionary
            Dictionary containing config parameters.
        compute_track: np.array(bool), shape [n], optional
            Tracks to measure (default: all). The merged masks are still built from the claims of all n tracks,
            so a feature claimed by two tracks goes to the same track as in mapfeature; the other tracks are
            not measured and keep NaN in the output (including merged_area).

    Returns:
        None if the file cannot be read (the caller counts the tracks as not computed), else a tuple
        out_dict: dictionary of np.array(float32), shape [n]
            The variables of recomputed_variable_names(), plus merged_area [km^2], the area of the merged mask.
        diag: dictionary
            n_conflict: feature numbers claimed by more than one track; n_out_of_range: feature numbers that
            are not in the file.
    """
    logger = logging.getLogger(__name__)
    feature_varname = config.get("feature_varname", "feature_number")
    ref_varname = config.get("ref_varname", "dbz_comp")
    pixel_radius = config["pixel_radius"]
    area_method = config.get("area_method", "fixed")
    pbc_direction = config.get("pbc_direction", "none")
    dbz_area_threshs = config.get("trackstats_dbz_area_threshs", [45, 50])
    overwrite_km = conv_dilate_overwrite_km(config)

    n = len(file_cloudnumber)
    out = {name: np.full(n, np.nan, dtype=np.float32)
           for name in recomputed_variable_names(dbz_area_threshs, with_conv_diameter_sum=True)}
    out["merged_area"] = np.full(n, np.nan, dtype=np.float32)
    diag = {"n_conflict": 0, "n_out_of_range": 0}
    if n == 0:
        return out, diag

    try:
        with xr.open_dataset(cloudidfile, mask_and_scale=False, decode_times=False) as ds:
            feature_number = ds[feature_varname].squeeze().values.astype(np.int64)
            convsf = ds["convsf"].squeeze().values
            dbz = ds[ref_varname].squeeze().values
            cell_cluster_number = ds["cell_cluster_number"].squeeze().values.astype(np.int64)
            cell_number = ds["cell_number"].squeeze().values.astype(np.int64) if "cell_number" in ds else None
            echotops = {t: ds[f"echotop{t}"].squeeze().values / 1000. for t in ECHOTOP_THRESHS}
            latitude = ds["latitude"].values
            longitude = ds["longitude"].values
            x_coords = ds["x"].values / 1000.
            y_coords = ds["y"].values / 1000.
    except OSError as err:
        logger.warning(f"Cannot read cloudid file {cloudidfile}: {err}")
        return None

    if overwrite_km is not None:
        # clusters joined at the overwrite radius, from the clusters of this file (the cloudid file is not modified)
        cell_cluster_number = redilate_cell_clusters(cell_cluster_number, overwrite_km, config).astype(np.int64)

    ny, nx = feature_number.shape
    pixel_area = get_pixel_area(config)
    mean_pixelength = np.sqrt(np.nanmean(pixel_area)) if area_method == "latlon" else pixel_radius

    #########################################################################
    # Merged mask: one label per track (its position in the file arrays + 1) for the track's own feature
    # and the fragment features folded into it
    pair_labels = np.arange(1, n + 1, dtype=np.int32)
    # one row per track: own feature, merge fragments, split fragments (as mapfeature paints them, in track order)
    claims = np.concatenate([np.reshape(file_cloudnumber, (n, 1)), np.reshape(file_mergecloudnumber, (n, -1)),
                             np.reshape(file_splitcloudnumber, (n, -1))], axis=1).astype(np.int64)
    claim_labels = np.repeat(pair_labels[:, None], claims.shape[1], axis=1)
    keep = claims > 0
    claims, claim_labels = claims[keep], claim_labels[keep]
    lut_size = int(feature_number.max()) + 1
    in_file = claims < lut_size
    diag["n_out_of_range"] = int(np.count_nonzero(~in_file))
    claims, claim_labels = claims[in_file], claim_labels[in_file]
    # a feature claimed by more than one track goes to the last one (np.unique on the reversed claims returns the
    # last claim of each feature; the order of repeated fancy-index assignments is not guaranteed)
    unique_claims = np.unique(np.stack([claims, claim_labels]), axis=1)
    _, claimed_by = np.unique(unique_claims[0], return_counts=True)
    diag["n_conflict"] = int(np.count_nonzero(claimed_by > 1))
    features, last = np.unique(claims[::-1], return_index=True)
    lut = np.zeros(lut_size, dtype=np.int32)
    lut[features] = claim_labels[::-1][last]
    merged_label = lut[feature_number]
    if compute_track is not None:
        # measure only the selected tracks: the others' masks are removed from the label image (their features were
        # resolved above, so the selected masks are the same as with all tracks)
        label_selected = np.zeros(n + 1, dtype=bool)
        label_selected[1:] = np.asarray(compute_track, dtype=bool)
        merged_label = np.where(label_selected[merged_label], merged_label, np.int32(0))

    #########################################################################
    # Convective length: the pixels of the merged mask that belong to one cell cluster (cell_cluster_number: the
    # connected groups of the inflated convective cells; read from the file, or joined further at
    # conv_dilate_overwrite_radius_km) form a group; the group's
    # major axis is measured on that region inside the mask, and conv_majoraxis is the longest group of a track.
    # Groups without a convective pixel (Steiner class 4) of the track do not count.
    conv_major_by_label = np.zeros(n + 1)
    in_mask = (merged_label > 0) & (cell_cluster_number > 0)
    if in_mask.any():
        stride = int(cell_cluster_number.max()) + 1
        keys = merged_label[in_mask].astype(np.int64) * stride + cell_cluster_number[in_mask]
        unique_keys, inverse = np.unique(keys, return_inverse=True)
        inverse = inverse.reshape(-1)
        has_conv = np.zeros(len(unique_keys), dtype=bool)
        has_conv[inverse[convsf[in_mask] == 4]] = True
        group_id = np.where(has_conv, np.arange(1, len(unique_keys) + 1), 0).astype(np.int32)
        group_map = np.zeros(merged_label.shape, dtype=np.int32)
        group_map[in_mask] = group_id[inverse]
        group_parent = (unique_keys // stride).astype(np.int64)
        for gp in regionprops(group_map):
            parent = group_parent[gp.label - 1]
            conv_major_by_label[parent] = max(conv_major_by_label[parent], gp.axis_major_length * mean_pixelength)

    #########################################################################
    # Sum of the circular-equivalent diameters 2*sqrt(A_i/pi) of the convective cells (cell_number labels) in the merged mask
    # (conv_diameter_sum of trackstats_func.py): the area of each (track, cell) pair from one bincount. A cell that lies in the
    # track's own feature and in a folded fragment is one cell of the merged mask.
    diameter_by_label = np.zeros(n + 1)
    if cell_number is not None:
        in_cell = (merged_label > 0) & (cell_number > 0)
        if in_cell.any():
            cell_stride = int(cell_number.max()) + 1
            cell_keys = merged_label[in_cell].astype(np.int64) * cell_stride + cell_number[in_cell]
            unique_cell_keys, cell_inverse = np.unique(cell_keys, return_inverse=True)
            cell_inverse = cell_inverse.reshape(-1)
            if area_method == "latlon":
                cell_area = np.bincount(cell_inverse, weights=np.nan_to_num(pixel_area[in_cell]), minlength=len(unique_cell_keys))
            else:
                cell_area = np.bincount(cell_inverse, minlength=len(unique_cell_keys)) * pixel_radius ** 2
            cell_track = (unique_cell_keys // cell_stride).astype(np.int64)
            diameter_by_label = np.bincount(cell_track, weights=2.0 * np.sqrt(cell_area / np.pi), minlength=n + 1)

    #########################################################################
    # Statistics of the merged mask (definitions of trackstats_func.py, radar_feature)
    dbz_filled = np.copy(dbz)
    dbz_filled[np.isnan(dbz_filled)] = -9999.
    longitude_min, longitude_max = np.nanmin(longitude), np.nanmax(longitude)
    latitude_min, latitude_max = np.nanmin(latitude), np.nanmax(latitude)
    x_min, x_max = np.nanmin(x_coords), np.nanmax(x_coords)
    y_min, y_max = np.nanmin(y_coords), np.nanmax(y_coords)

    for rp in regionprops(merged_label, intensity_image=dbz_filled):
        j = rp.label - 1
        rows, cols = rp.coords[:, 0], rp.coords[:, 1]
        if area_method == "latlon":
            pix_area = pixel_area[rows, cols]
            out["merged_area"][j] = np.nansum(pix_area)
        else:
            out["merged_area"][j] = rows.size * pixel_radius ** 2

        # Mean position
        lon, lat = longitude[rows, cols], latitude[rows, cols]
        feature_x, feature_y = x_coords[cols], y_coords[rows]
        if pbc_direction in ['x', 'both']:
            out["meanlon"][j] = circular_mean(lon, longitude_min, longitude_max)
            out["mean_x"][j] = circular_mean(feature_x, x_min, x_max)
        else:
            out["meanlon"][j] = np.nanmean(lon)
            out["mean_x"][j] = np.nanmean(feature_x)
        if pbc_direction in ['y', 'both']:
            out["meanlat"][j] = circular_mean(lat, latitude_min, latitude_max)
            out["mean_y"][j] = circular_mean(feature_y, y_min, y_max)
        else:
            out["meanlat"][j] = np.nanmean(lat)
            out["mean_y"][j] = np.nanmean(feature_y)

        # Reflectivity extremes and areas above the dBZ thresholds
        values = dbz[rows, cols]
        is_conv = convsf[rows, cols] == 4
        out["max_dbz"][j] = np.nanmax(values)
        for t in dbz_area_threshs:
            above = values > t
            above_conv = above & is_conv
            if area_method == "latlon":
                out[f"area_{t}dbz"][j] = np.nansum(pix_area[above]) if above.any() else 0
                out[f"conv_area_{t}dbz"][j] = np.nansum(pix_area[above_conv]) if above_conv.any() else 0
            else:
                out[f"area_{t}dbz"][j] = np.count_nonzero(above) * pixel_radius ** 2
                out[f"conv_area_{t}dbz"][j] = np.count_nonzero(above_conv) * pixel_radius ** 2

        # Echo-top heights over the convective pixels
        if is_conv.any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                for t in ECHOTOP_THRESHS:
                    out[f"maxETH_{t}dbz"][j] = np.nanmax(echotops[t][rows[is_conv], cols[is_conv]])

        out["conv_majoraxis"][j] = conv_major_by_label[rp.label]
        if cell_number is not None:
            out["conv_diameter_sum"][j] = diameter_by_label[rp.label]

        # Shape statistics and reflectivity-weighted centroid
        out["eccentricity"][j] = rp.eccentricity
        out["majoraxis"][j] = rp.axis_major_length * mean_pixelength
        try:
            out["minoraxis"][j] = rp.axis_minor_length * mean_pixelength
        except ValueError:
            pass
        if (not np.isnan(out["minoraxis"][j])) and (out["minoraxis"][j] > 0):
            out["aspectratio"][j] = out["majoraxis"][j] / out["minoraxis"][j]
        out["orientation"][j] = rp.orientation * (180. / np.pi)
        cw = rp.centroid_weighted
        if not (np.isnan(cw[0]) or np.isnan(cw[1])):
            yc, xc = int(round(cw[0])), int(round(cw[1]))
            if (0 <= yc < ny) and (0 <= xc < nx):
                out["lat_weightedcentroid"][j] = latitude[yc, xc]
                out["lon_weightedcentroid"][j] = longitude[yc, xc]
                out["y_weightedcentroid"][j] = y_coords[yc]
                out["x_weightedcentroid"][j] = x_coords[xc]

    return out, diag
