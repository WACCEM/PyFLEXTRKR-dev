"""
Rows-mode robust MCS for Tb+PF tracking (config stats_format: rows).

The robust MCS identification from the PF statistics of robustmcspf.py (Feng et al. 2021: criteria on the PF area, rain rate, skewness
and heavy rain ratio fitted to the duration) and of robustmcspf_saag.py (MCSMIP: PF major axis, heavy rain rate and rain volume), with
row tables as input (the output of matchtbpf_rows) and as output. The definition is chosen by the config key robustmcs_definition:

    feng21   the definition of robustmcspf.define_robust_mcs_pf   (coefs_pf_area, coefs_pf_rr, coefs_pf_skew, coefs_pf_heavyratio, ...)
    mcsmip   the definition of robustmcspf_saag.define_robust_mcs_pf   (mcs_min_rainvol_thresh, heavy_rainrate_thresh, ...)

The tests of the two dense steps are applied to the flat arrays of every track (one position for every valid (track, time) entry of
every track), expression by expression as in the dense modules, so the result is that of the dense step including its behaviour for
NaN values and for a 0 / 0 heavy rain ratio (not an MCS), and a PF major axis exactly equal to mcs_pf_majoraxis_for_lifetime is counted
as that value in pf_lifetime (neither 0 nor 1). tests/test_robustmcs_rows.py checks that the tables equal what rowstats.dense_to_rows
makes of the dense file of the dense step. The robust tracks are selected from all the tracks and renumbered.

Author: Zhe Feng | zhe.feng@pnnl.gov
"""
import logging
import os
import shutil
import time
import warnings

import numpy as np

from pyflextrkr.rowstats import TableBuilder, gather_index, read_tables, subset_tracks, write_rows

DEFINITIONS = ("feng21", "mcsmip")


def feng21_mcs_periods(pf_majoraxis, pf_area, pf_rainrate, pf_skewness, volrain_all, volrain_heavy, time_res, params):
    """
    The robust MCS test of one track by the definition of Feng et al. (2021): at which times is it an MCS?

    The times with a PF major axis in [mcs_pf_majoraxis_thresh, max_pf_majoraxis_thresh] are split into periods at gaps of more than
    mcs_pf_gap time steps; a period lasting at least mcs_pf_durationthresh hours is an MCS period if the PF area, rain rate and
    skewness are above the values fitted to its duration (coefs_pf_area, coefs_pf_rr, coefs_pf_skew) for at least that many hours and
    the heavy rain makes more than the fitted fraction (coefs_pf_heavyratio) of the rain volume.

    Args:
        pf_majoraxis, pf_area, pf_rainrate, pf_skewness: float32 arrays [duration]
            Statistics of the largest PF at every time of the track.
        volrain_all, volrain_heavy: float32 arrays [duration]
            Rain volume of all the PF (pf_accumrain summed over the PF) and of their heavy rain (pf_accumrainheavy summed).
        time_res: float
            Hours per time step.
        params: dictionary
            mcs_pf_majoraxis_thresh, max_pf_majoraxis_thresh, mcs_pf_durationthresh, mcs_pf_gap, coefs_pf_area, coefs_pf_rr,
            coefs_pf_skew, coefs_pf_heavyratio.

    Returns:
        status: bool array [duration]
            True at the times of the MCS periods.
    """
    status = np.zeros(len(pf_majoraxis), dtype=bool)
    # PF major axis length criteria
    ipfmcs = np.flatnonzero((pf_majoraxis >= params["mcs_pf_majoraxis_thresh"]) & (pf_majoraxis <= params["max_pf_majoraxis_thresh"]))
    nipfmcs = len(ipfmcs)
    # Duration threshold on the whole set of times (strictly longer)
    if nipfmcs == 0 or not nipfmcs * time_res > params["mcs_pf_durationthresh"]:
        return status
    # Periods of consecutive times
    groups = np.split(ipfmcs, np.where(np.diff(ipfmcs) > params["mcs_pf_gap"])[0] + 1)
    # Loop over the periods: the duration test and the criteria fitted to the duration
    for group in groups:
        # Duration of the period: last index - first index + 1 time steps, in hours
        igroup_duration = np.multiply(group[-1] - group[0] + 1, time_res)
        if not igroup_duration >= params["mcs_pf_durationthresh"]:
            continue
        # The PF area, rain rate, skewness and heavy rain ratio that a period of this duration has to exceed
        coefs_area, coefs_rr = params["coefs_pf_area"], params["coefs_pf_rr"]
        coefs_skew, coefs_heavy = params["coefs_pf_skew"], params["coefs_pf_heavyratio"]
        mcs_pfarea = coefs_area[0] + coefs_area[1] * igroup_duration
        mcs_rrskew = coefs_skew[0] + coefs_skew[1] * igroup_duration
        mcs_rravg = coefs_rr[0] + coefs_rr[1] * igroup_duration
        mcs_heavyratio = coefs_heavy[0] + coefs_heavy[1] * igroup_duration
        # Number of hours when the PF exceeds the MCS criteria
        ct_pftimes = np.count_nonzero((pf_area[group] > mcs_pfarea) & (pf_rainrate[group] > mcs_rravg) & (pf_skewness[group] > mcs_rrskew))
        dur_pf = float(ct_pftimes) * time_res
        # Volumetric heavy rain ratio during this period (0 / 0 is NaN: the comparison is False, not an MCS)
        with np.errstate(divide="ignore", invalid="ignore"):
            heavyrain_ratio = 100 * np.nansum(volrain_heavy[group]) / np.nansum(volrain_all[group])
        # Duration of PF satisfying the criteria >= the duration threshold and the heavy rain ratio above the fitted value
        if (dur_pf >= params["mcs_pf_durationthresh"]) & (heavyrain_ratio > mcs_heavyratio):
            status[group] = True
    return status


def mcsmip_mcs_periods(pf_majoraxis, pf_maxrainrate, volrain_all, time_res, params):
    """
    The robust MCS test of one track by the MCSMIP definition: at which times is it an MCS?

    The times with a PF major axis in [mcs_pf_majoraxis_thresh, max_pf_majoraxis_thresh] are split into periods at gaps of more than
    mcs_pf_gap time steps; a period lasting at least mcs_pf_durationthresh hours is an MCS period if the maximum rain rate is above
    heavy_rainrate_thresh for at least mcs_pf_durationthresh hours and the rain volume is above mcs_min_rainvol_thresh for at least
    mcs_volrain_duration_thresh hours.

    Args:
        pf_majoraxis: float32 array [duration]
            Major axis of the largest PF at every time of the track.
        pf_maxrainrate: float32 array [duration]
            Maximum rain rate of the PF (pf_maxrainrate, largest over the PF).
        volrain_all: float32 array [duration]
            Total rain volume (total_volrain).
        time_res: float
            Hours per time step.
        params: dictionary
            mcs_pf_majoraxis_thresh, max_pf_majoraxis_thresh, mcs_pf_durationthresh, mcs_pf_gap, heavy_rainrate_thresh,
            mcs_min_rainvol_thresh, mcs_volrain_duration_thresh.

    Returns:
        status: bool array [duration]
            True at the times of the MCS periods.
    """
    status = np.zeros(len(pf_majoraxis), dtype=bool)
    # PF major axis length criteria
    ipfmcs = np.flatnonzero((pf_majoraxis >= params["mcs_pf_majoraxis_thresh"]) & (pf_majoraxis <= params["max_pf_majoraxis_thresh"]))
    nipfmcs = len(ipfmcs)
    # Duration threshold on the whole set of times (strictly longer)
    if nipfmcs == 0 or not nipfmcs * time_res > params["mcs_pf_durationthresh"]:
        return status
    # Periods of consecutive times
    groups = np.split(ipfmcs, np.where(np.diff(ipfmcs) > params["mcs_pf_gap"])[0] + 1)
    # Loop over the periods: the duration test and the rain rate and rain volume criteria
    for group in groups:
        igroup_duration = np.multiply(group[-1] - group[0] + 1, time_res)
        if not igroup_duration >= params["mcs_pf_durationthresh"]:
            continue
        # Hours with a heavy maximum rain rate and hours with a large rain volume
        ct_maxrrtimes = np.count_nonzero(pf_maxrainrate[group] > params["heavy_rainrate_thresh"])
        ct_volrain = np.count_nonzero(volrain_all[group] > params["mcs_min_rainvol_thresh"])
        dur_maxrr = float(ct_maxrrtimes) * time_res
        dur_volrain = float(ct_volrain) * time_res
        if (dur_maxrr >= params["mcs_pf_durationthresh"]) & (dur_volrain >= params["mcs_volrain_duration_thresh"]):
            status[group] = True
    return status


def robustmcs_params(config):
    """
    The config values of the robust MCS definition that is chosen by robustmcs_definition.

    Args:
        config: dictionary
            Config (robustmcs_definition: 'feng21' or 'mcsmip', and the keys of that definition).

    Returns:
        definition, params: string, dictionary
            The definition and its parameters.

    Raises:
        ValueError: robustmcs_definition is missing or not one of DEFINITIONS, or a key of the definition is missing.
    """
    definition = config.get("robustmcs_definition")
    if definition not in DEFINITIONS:
        raise ValueError(f"robustmcs_definition must be one of {DEFINITIONS}, not {definition!r}")
    keys = ["mcs_pf_majoraxis_thresh", "max_pf_majoraxis_thresh", "mcs_pf_durationthresh", "mcs_pf_majoraxis_for_lifetime", "mcs_pf_gap"]
    if definition == "feng21":
        keys += ["coefs_pf_area", "coefs_pf_rr", "coefs_pf_skew", "coefs_pf_heavyratio"]
    else:
        keys += ["mcs_min_rainvol_thresh", "heavy_rainrate_thresh", "mcs_volrain_duration_thresh"]
    missing = [k for k in keys if k not in config]
    if missing:
        raise ValueError(f"robustmcs_definition {definition!r} needs the config keys {missing}")
    return definition, {k: config[k] for k in keys}


def define_robust_mcs_pf_rows(config):
    """
    Identify robust MCS based on PF statistics: robustmcspf.define_robust_mcs_pf (robustmcs_definition: feng21) or
    robustmcspf_saag.define_robust_mcs_pf (robustmcs_definition: mcsmip) with row tables as the input (the output of
    matchtbpf_rows) and as the output.

    Args:
        config: dictionary
            Config parameters (robustmcs_definition and the keys of the definition, see robustmcs_params).

    Returns:
        statistics_outfolder: string
            Folder of the row tables, f"{stats_path}{mcsrobust_filebase}{startdate}_{enddate}.rows": the PF tables of the robust MCS
            tracks (renumbered) and their pf_lifetime and pf_mcsstatus. With no robust MCS the tables are empty (a warning is logged)
            and the output check of the run script stops the run.
    """
    logger = logging.getLogger(__name__)
    definition, params = robustmcs_params(config)
    logger.info(f"Identifying robust MCS based on PF statistics (row tables, definition {definition})")

    #########################################################################
    # Read the PF statistics of the MCS
    #########################################################################
    stats_path = config["stats_outpath"]
    startdate, enddate = config["startdate"], config["enddate"]
    statistics_outfolder = f"{stats_path}{config['mcsrobust_filebase']}{startdate}_{enddate}.rows"
    tables = read_tables(f"{stats_path}{config['mcspfstats_filebase']}{startdate}_{enddate}.rows")
    meta, entries = tables.meta, tables.entries
    offsets, duration = tables.offsets, tables.duration
    ntracks, nentries = len(duration), int(offsets[-1])
    time_res = float(meta["global_attrs"]["time_resolution_hour"])
    fillval = meta["attrs"]["mcs_status"]["_FillValue"]
    # The statistics of the largest PF (the first of the nmaxpf), and the PF variables summed or maximized over the PF
    pf_columns = lambda name: np.stack([entries[f"{name}__{j}"] for j in range(meta["vars_entry3d"][name])], axis=1)
    pf_majoraxis = entries["pf_majoraxis__0"]
    if definition == "feng21":
        pf_area, pf_rainrate, pf_skewness = entries["pf_area__0"], entries["pf_rainrate__0"], entries["pf_skewness__0"]
        # Rain volume of all the PF and of their heavy rain: summed over the PF, NaN skipped
        volrain_all, volrain_heavy = np.nansum(pf_columns("pf_accumrain"), axis=1), np.nansum(pf_columns("pf_accumrainheavy"), axis=1)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            pf_maxrainrate = np.nanmax(pf_columns("pf_maxrainrate"), axis=1)
        volrain_all = entries["total_volrain"]

    #########################################################################
    # The robust MCS test, per track
    #########################################################################
    status_flat = np.zeros(nentries, dtype=bool)
    # Loop over the tracks: the times of its MCS periods
    for nt in range(ntracks):
        sl = slice(offsets[nt], offsets[nt + 1])
        if duration[nt] <= 0:
            continue
        if definition == "feng21":
            status_flat[sl] = feng21_mcs_periods(pf_majoraxis[sl], pf_area[sl], pf_rainrate[sl], pf_skewness[sl], volrain_all[sl],
                                                 volrain_heavy[sl], time_res, params)
        else:
            status_flat[sl] = mcsmip_mcs_periods(pf_majoraxis[sl], pf_maxrainrate[sl], volrain_all[sl], time_res, params)
    # Robust MCS: the tracks with at least one MCS time
    has_status = np.zeros(ntracks, dtype=bool)
    nonempty = duration > 0
    if nonempty.any():
        has_status[nonempty] = np.add.reduceat(status_flat.astype(np.int64), offsets[:-1][nonempty]) > 0
    trackid_mcs = np.flatnonzero(has_status)
    nmcs = len(trackid_mcs)
    if nmcs == 0:
        logger.warning("No robust MCS found. The tables are empty.")
    else:
        logger.info(f"Number of robust MCS: {nmcs}")

    #########################################################################
    # The robust tracks, their lifetime with a significant PF, and the PF status of every time
    #########################################################################
    sub = subset_tracks(tables, trackid_mcs)
    # Positions of the entries of the robust tracks in the flat columns, and where each robust track starts in them
    index, sub_offsets = gather_index(offsets, trackid_mcs)
    # Lifetime when a significant PF is present: the largest PF major axis counts 1 above mcs_pf_majoraxis_for_lifetime and 0 below it
    # (a value exactly equal to it stays as it is)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        pf_maxmajoraxis = np.nanmax(pf_columns("pf_majoraxis")[index], axis=1) if nmcs else np.zeros(0, dtype=np.float32)
    pf_maxmajoraxis[pf_maxmajoraxis < params["mcs_pf_majoraxis_for_lifetime"]] = 0
    pf_maxmajoraxis[pf_maxmajoraxis > params["mcs_pf_majoraxis_for_lifetime"]] = 1
    pf_lifetime = np.zeros(0, dtype=np.float32)
    if nmcs:
        pf_lifetime = np.multiply(np.add.reduceat(np.nan_to_num(pf_maxmajoraxis, nan=0.0), sub_offsets[:-1]), time_res)
    pf_mcsstatus = np.where(status_flat[index], 1, fillval).astype(np.int64)

    #########################################################################
    # Assemble the row tables
    #########################################################################
    builder = TableBuilder.from_tables(sub)
    # (the dense step gives pf_lifetime no _FillValue, and the netCDF writer adds NaN to a float variable that has none: the file has it)
    builder.add_track_var("pf_lifetime", pf_lifetime, {
        "long_name": "MCS lifetime when a significant PF is present", "units": "hour", "_FillValue": np.nan})
    builder.add_entry_var("pf_mcsstatus", pf_mcsstatus, {
        "long_name": "Flag indicating the status of MCS based on PF. 1 = Yes, 0 = No", "units": "unitless", "_FillValue": fillval},
        fill=fillval)
    builder.global_attrs.update({
        "MCS_PF_majoraxis_thresh": params["mcs_pf_majoraxis_thresh"], "MCS_PF_duration_thresh": params["mcs_pf_durationthresh"],
        "PF_PF_min_majoraxis_thresh": params["mcs_pf_majoraxis_for_lifetime"]})
    if definition == "feng21":
        builder.global_attrs.update({k: params[k] for k in ("coefs_pf_area", "coefs_pf_rr", "coefs_pf_skew", "coefs_pf_heavyratio")})
    else:
        builder.global_attrs.update({k: params[k] for k in ("heavy_rainrate_thresh", "mcs_min_rainvol_thresh",
                                                           "mcs_volrain_duration_thresh")})
    builder.global_attrs["Created_on"] = time.ctime(time.time())
    out_tables = builder.build()

    #########################################################################
    # Write output
    #########################################################################
    logger.info(f"Writing {statistics_outfolder}")
    if os.path.isdir(statistics_outfolder):
        shutil.rmtree(statistics_outfolder)
    write_rows(out_tables, statistics_outfolder)
    logger.info(f"{statistics_outfolder}")
    return statistics_outfolder
