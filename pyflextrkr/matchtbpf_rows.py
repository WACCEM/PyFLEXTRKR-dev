"""
Rows-mode matchpf for Tb tracking (config stats_format: rows).

The same matching of the Tb tracked MCS with the precipitation as matchtbpf_driver.match_tbpf_tracks (the precipitation feature (PF)
statistics inside the cold cloud shield of every track and time), with row tables as input and output: the Tb statistics are read
as the row tables of identifymcs_tb_rows, the PF statistics are written as columns of the entries table, one value per (track, time
index) for the [tracks, times] variables (total_rain, pf_npf, ...) and nmaxpf values per entry for the [tracks, times, nmaxpf]
variables (pf_area, pf_lon, ...; meta "vars_entry3d" of pyflextrkr.rowstats). The statistics themselves are calculated by the
unchanged matchtbpf_func.matchtbpf_singlefile, once for every cloudid file, with the same arguments as in the dense step, so the
values are those of the dense step; tests/test_matchtbpf_rows.py checks that the tables equal what rowstats.dense_to_rows makes of the
dense file of the dense step.

Author: Zhe Feng | zhe.feng@pnnl.gov
"""
import logging
import os
import shutil
import sys
import time

import dask
import numpy as np
import xarray as xr

from pyflextrkr.ft_utilities import match_pairs_to_files, subset_files_timerange
from pyflextrkr.regionprops_rows import fragment_slots
from pyflextrkr.rowstats import FRAGMENT_KINDS, TableBuilder, read_tables, write_rows
from pyflextrkr.trackstats_driver import adjust_position_continuous_simple


def pbc_thresholds(first_cloudid_file, pbc_max_domain_fraction):
    """
    The jump of a longitude / latitude between two times above which a position crossed the periodic boundary: the fraction
    pbc_max_domain_fraction of the domain size, as trackstats_driver and matchtbpf_driver calculate it.

    Args:
        first_cloudid_file: string
            Cloudid file (its longitude and latitude give the size of the domain).
        pbc_max_domain_fraction: float
            Fraction of the domain (config pbc_max_domain_fraction).

    Returns:
        thresh_lon, thresh_lat: float, float
            Thresholds in degrees.
    """
    with xr.open_dataset(first_cloudid_file) as dspix:
        lon2d, lat2d = dspix["longitude"].values, dspix["latitude"].values
    # Number of columns / rows and the median grid spacing along the first row / column
    nx = lon2d.shape[1] if lon2d.ndim == 2 else len(lon2d)
    ny = lat2d.shape[0] if lat2d.ndim == 2 else len(lat2d)
    dx = float(np.abs(np.median(np.diff(lon2d[0, :] if lon2d.ndim == 2 else lon2d))))
    dy = float(np.abs(np.median(np.diff(lat2d[:, 0] if lat2d.ndim == 2 else lat2d))))
    return pbc_max_domain_fraction * nx * dx, pbc_max_domain_fraction * ny * dy


def match_tbpf_tracks_rows(config):
    """
    Match Tb tracked MCS with precipitation to calculate PF statistics: matchtbpf_driver.match_tbpf_tracks with row tables as the
    input (the output of identifymcs_tb_rows) and as the output.

    Args:
        config: dictionary
            Config parameters (those of match_tbpf_tracks: nmaxpf, match_pixel_dt_thresh, pbc_direction, ...).

    Returns:
        statistics_outfolder: string
            Folder of the row tables, f"{stats_path}{mcspfstats_filebase}{startdate}_{enddate}.rows": the Tb tables and the PF
            statistics, the input of the rows-mode robust MCS step.
    """
    logger = logging.getLogger(__name__)
    logger.info("Matching Tb tracked MCS with precipitation to calculate PF statistics (row tables)")

    #########################################################################
    # Settings
    #########################################################################
    feature_type = config["feature_type"]
    stats_path = config["stats_outpath"]
    tracking_outpath = config["tracking_outpath"]
    cloudid_filebase = config["cloudid_filebase"]
    startdate, enddate = config["startdate"], config["enddate"]
    nmaxpf = config["nmaxpf"]
    pf_dimname = config["pf_dimname"]
    run_parallel = config["run_parallel"]
    fillval = config["fillval"]
    match_pixel_dt_thresh = config["match_pixel_dt_thresh"]
    pbc_direction = config.get("pbc_direction", "none")
    pbc_max_domain_fraction = config.get("pbc_max_domain_fraction", 0.5)
    # The function that calculates the PF statistics of one cloudid file depends on the feature type
    if "radar3d" in feature_type:
        from pyflextrkr.matchtbradar_func import matchtbpf_singlefile
    else:
        from pyflextrkr.matchtbpf_func import matchtbpf_singlefile
    statistics_folder = f"{stats_path}{config['mcstbstats_filebase']}{startdate}_{enddate}.rows"
    statistics_outfolder = f"{stats_path}{config['mcspfstats_filebase']}{startdate}_{enddate}.rows"

    #########################################################################
    # Read the Tb statistics of the MCS
    #########################################################################
    tables = read_tables(statistics_folder)
    meta, entries, fragments = tables.meta, tables.entries, tables.fragments
    offsets = tables.offsets
    ir_basetime, ir_cloudnumber = entries["base_time"], entries["cloudnumber"]
    nentries = len(ir_basetime)
    # Merge / split cloud numbers of the entries, from the fragments table (the layout of the dense [tracks, times, slots] variable)
    by_kind = {}
    if {"merge_cloudnumber", "split_cloudnumber"}.issubset(meta["vars_3d"]):
        nslots = meta["dims"][meta["slots_dim"]]
        slot_fill = meta["fill"]["merge_cloudnumber"]
        frag_entry = offsets[fragments["track"]] + fragments["k"]
        for code, kind in enumerate(FRAGMENT_KINDS):
            use = fragments["kind"] == code
            by_kind[kind] = (frag_entry[use], fragments["slot"][use], fragments["cloudnumber"][use])
    else:
        nslots, slot_fill = int(config["nmaxlinks"]), fillval
        for kind in FRAGMENT_KINDS:
            by_kind[kind] = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int32))

    #########################################################################
    # Cloudid files, the entries of each of them, and the thresholds of the periodic boundary
    #########################################################################
    infiles_info = subset_files_timerange(tracking_outpath, cloudid_filebase, config["start_basetime"], config["end_basetime"])
    cloudidfile_list, cloudidfile_basetime = infiles_info[0], infiles_info[1]
    nfiles = len(cloudidfile_list)
    thresh_lon = thresh_lat = None
    if pbc_direction != "none" and nfiles > 0:
        thresh_lon, thresh_lat = pbc_thresholds(cloudidfile_list[0], pbc_max_domain_fraction)
        logger.info(f"PBC adjustment thresholds: lon={thresh_lon:.4f} lat={thresh_lat:.4f} deg (pbc_direction={pbc_direction})")
    # The flat entries of each cloudid file (all entries are in one row of a [1, nentries] array: the index in it is the entry)
    file_pairs = match_pairs_to_files(ir_basetime[None, :], cloudidfile_basetime, match_pixel_dt_thresh)

    #########################################################################
    # PF statistics of every cloudid file
    #########################################################################
    entry_of_file, results = [], []
    # Loop over the cloudid files: the entries that belong to it and the call that calculates their PF statistics
    for ifile in range(nfiles):
        entry = file_pairs[ifile][1]
        entry_of_file.append(entry)
        args = (cloudidfile_list[ifile], ir_cloudnumber[entry], fragment_slots(*by_kind["merge"], entry, nslots, slot_fill),
                fragment_slots(*by_kind["split"], entry, nslots, slot_fill), config)
        if run_parallel == 0:
            results.append(matchtbpf_singlefile(*args))
        elif run_parallel >= 1:
            results.append(dask.delayed(matchtbpf_singlefile)(*args))
        else:
            sys.exit('Valid parallelization flag not provided.')
    if run_parallel >= 1:
        from dask.distributed import wait       # only the parallel mode needs the `distributed` package (the CI environment has none)
        results = dask.compute(*results)
        wait(results)

    #########################################################################
    # Collect the results as flat columns (one value or nmaxpf values per entry)
    #########################################################################
    first = next((r for r in results if r is not None), None)
    if first is None:
        logger.critical("No PF statistics: none of the cloudid files could be matched to the MCS. Tracking will now exit.")
        sys.exit(1)
    var_names, var_attrs, names_per_entry = list(first[0].keys()), first[1], list(first[2])
    # [tracks, times] variables: one float32 value per entry; the others: nmaxpf float32 values per entry
    flat = {n: np.full(nentries, np.nan, dtype=np.float32) for n in names_per_entry}
    flat3 = {n: np.full((nentries, nmaxpf), np.nan, dtype=np.float32) for n in var_names if n not in names_per_entry}
    # Loop over the cloudid files that gave a result: put the values of its entries into the columns
    for ifile in range(nfiles):
        if results[ifile] is None:
            continue
        entry = entry_of_file[ifile]
        for name in var_names:
            values = results[ifile][0][name]
            if name in flat and values.ndim == 1:
                flat[name][entry] = values
            elif name in flat3 and values.ndim == 2:
                flat3[name][entry, :] = values
            else:
                raise ValueError(f"{name}: a result of {values.ndim} dimensions does not fit the variable")

    #########################################################################
    # Periodic boundary: the longitude and latitude of every PF along every track must be continuous
    #########################################################################
    ntracks = len(tables.duration)
    # Loop over the longitude and the latitude variables
    for thresh, suffix in ((thresh_lon, "_lon"), (thresh_lat, "_lat")):
        if thresh is None:
            continue
        names = [v for v in flat3 if suffix in v]
        logger.info(f"Applying PBC {suffix} adjustment to: {names}")
        # Loop over the variables, the tracks and the PF slots: unwrap the series of the valid entries of one track
        for name in names:
            arr = flat3[name]
            for itrack in range(ntracks):
                sl = slice(offsets[itrack], offsets[itrack + 1])
                for ipf in range(nmaxpf):
                    series = arr[sl, ipf]
                    if np.any(~np.isnan(series)):
                        arr[sl, ipf], _ = adjust_position_continuous_simple(series, thresh, pbc_max_domain_fraction)

    #########################################################################
    # Assemble the row tables: the Tb tables and the PF statistics
    #########################################################################
    builder = TableBuilder.from_tables(tables)
    for name in names_per_entry:
        builder.add_entry_var(name, flat[name], var_attrs[name], fill=np.nan)
    for name, values in flat3.items():
        builder.add_entry3d_var(name, values, pf_dimname, var_attrs[name], fill=np.nan)
    builder.global_attrs.update({
        "nmaxpf": nmaxpf, "PF_rainrate_thresh": config["pf_rr_thres"], "heavy_rainrate_thresh": config["heavy_rainrate_thresh"],
        "landfrac_thresh": config["landfrac_thresh"], "Created_on": time.ctime(time.time())})
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
