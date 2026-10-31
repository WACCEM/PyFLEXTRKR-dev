"""
Stitch the track numbers of consecutive periods into one tracknumbers file, as if gettracks had run over all periods at once.

gettracknumbers only carries the track number of every cloud and a counter from one file to the next (the decisions of the
linking use the links and the sizes of the clouds, never the numbers), so the result of a period can be renumbered
afterwards. Each period is tracked on its own with local numbers 1, 2, ...; neighbouring periods share exactly one cloudid file,
the SEAM: the pair (n-1, n) belongs to period A, the pair (n, n+1) to period B. Then

  - the local numbers 1..n0 of the clouds in the first column of B (the seam file) are the clouds of the last column of A, so they
    get the global numbers of A's last column; the other local numbers j > n0 get G + (j - n0 - 1), G being the first unused
    global number after A;
  - the seam column is built from both sides: track_status is the sum (the codes of the pair before and of the pair after the file
    are disjoint), the merge numbers come from B and the split numbers from A, track_reset is 0 where the track starts in the
    file and -9999 elsewhere, as in an unbroken run;
  - boundaries without a shared file are joined as gettracknumbers joins a gap: the last column of A keeps its end flag, B starts
    new tracks after the last number of A. If the periods are closer than timegap although no pair links them (a track file is
    missing), that is a restart too and is counted in n_chain_breaks, as gettracknumbers does for a missing track file.

The result is identical to a run over all files at once (tests/test_stitch_tracks.py compares every variable for every cut of
random data). Create the missing pair of two months that were tracked apart with make_bridge_pairs.py and track the seam as a
period of its own, or run the periods with one shared file.

Author: Zhe Feng | zhe.feng@pnnl.gov
"""
import logging
import os

import netCDF4
import numpy as np

from pyflextrkr.gettracks import write_tracknumbers

FILLVAL = -9999
# Variables that hold track numbers and must be renumbered
NUMBER_VARS = ("track_numbers", "track_mergenumbers", "track_splitnumbers")


def read_period(path):
    """
    Read a tracknumbers file as written by gettracknumbers.

    Args:
        path: string
            tracknumbers_*.nc

    Returns:
        period: dictionary
            ntracks (next unused number), basetimes (int64 seconds, [nfiles]), cloudid ([nfiles, ncharacters] of S1),
            track_numbers / track_status / track_mergenumbers / track_splitnumbers / track_reset ([nfiles, nclouds] int),
            attrs (global attributes), path.
    """
    with netCDF4.Dataset(path) as ds:
        ds.set_auto_maskandscale(False)
        cloudid = np.array(ds["cloudid_files"][:])
        if cloudid.ndim == 3:                      # char arrays come with a trailing dimension of length 1
            cloudid = cloudid[:, :, 0]
        period = {
            "path": path, "ntracks": int(ds["ntracks"][:].ravel()[0]), "basetimes": np.array(ds["basetimes"][:], dtype=np.int64),
            "cloudid": cloudid, "attrs": {k: ds.getncattr(k) for k in ds.ncattrs()},
        }
        for name in NUMBER_VARS + ("track_status", "track_reset"):
            period[name] = np.array(ds[name][0])
    return period


def _timegap_hours(attrs):
    """
    The timegap [hour] stored in the attributes of a tracknumbers file ('0.25-hours').
    """
    return float(str(attrs["timegap"]).split("-")[0])


def _pad(arr, nclouds, fill, flag_rows=False):
    """
    Pad the last axis of [nfiles, nclouds_period] to nclouds with a fill value.

    Args:
        arr: numpy array [nfiles, nclouds_period]
        nclouds: int
            New length of the last axis.
        fill: int
            Value of the padded elements.
        flag_rows: bool (default: False)
            For track_reset: gettracknumbers flags the start (1) and the end (2) of the data for all slots of a column, so a
            column that has one of these flags in every slot gets it in the padded slots too.

    Returns:
        padded: numpy array [nfiles, nclouds]
    """
    if arr.shape[1] == nclouds:
        return arr
    out = np.full((arr.shape[0], nclouds), fill, dtype=arr.dtype)
    out[:, : arr.shape[1]] = arr
    if flag_rows:
        for flag in (1, 2):
            out[(arr == flag).all(axis=1)] = flag
    return out


def _column_before_last(blocks):
    """
    The column before the last one of a list of column blocks ([ncolumns, nclouds] arrays).

    Args:
        blocks: list of numpy arrays
            Consecutive blocks of columns.

    Returns:
        column: numpy array or None
            None if there is only one column.
    """
    if blocks[-1].shape[0] > 1:
        return blocks[-1][-2]
    return blocks[-2][-1] if len(blocks) > 1 else None


def _relabel(arr, mapping):
    """
    Replace the track numbers (values > 0) of an array by mapping[number]; fill values stay.
    """
    out = arr.copy()
    positive = arr > 0
    out[positive] = mapping[arr[positive]]
    return out


def stitch_arrays(periods):
    """
    Join the track numbers of consecutive periods (see the module description).

    Args:
        periods: list of dictionaries
            Output of read_period, in time order.

    Returns:
        result: dictionary
            ntracks, basetimes (datetime64[s]), cloudid, and the arrays track_numbers / track_status / track_mergenumbers /
            track_splitnumbers / track_reset of shape [nfiles, nclouds], and n_chain_breaks (breaks inside the periods plus the
            unlinked boundaries between them).

    Raises:
        ValueError: if neighbouring periods overlap by more than the seam file, or the seam files differ.
    """
    logger = logging.getLogger(__name__)
    nclouds = max(p["track_numbers"].shape[1] for p in periods)
    if len({p["cloudid"].shape[1] for p in periods}) != 1:
        raise ValueError("the file names of the periods have different lengths")
    blocks = {name: [] for name in NUMBER_VARS + ("track_status", "track_reset")}
    basetimes, cloudid = [], []
    g_end = None                                                  # first unused global track number
    n_breaks = sum(int(p["attrs"].get("n_chain_breaks", 0)) for p in periods)

    for index, p in enumerate(periods):
        padded = {name: _pad(p[name], nclouds, 0 if name == "track_status" else FILLVAL, flag_rows=(name == "track_reset"))
                  for name in NUMBER_VARS + ("track_status", "track_reset")}
        if index == 0:
            for name in padded:
                blocks[name].append(padded[name])
            basetimes.append(p["basetimes"])
            cloudid.append(p["cloudid"])
            g_end = p["ntracks"]
            continue

        # What joins this period to the previous one: a seam file, a gap, or a restart
        last_time, first_time = basetimes[-1][-1], p["basetimes"][0]
        last_name, first_name = cloudid[-1][-1].tobytes(), p["cloudid"][0].tobytes()
        timegap = _timegap_hours(p["attrs"]) * 3600.0
        if first_time == last_time and first_name == last_name:
            kind = "seam"
        elif first_time == last_time or first_time < last_time:
            raise ValueError(f"periods {index - 1} and {index} overlap ({periods[index - 1]['path']}, {p['path']}): they must "
                             f"share exactly one cloudid file or none")
        elif first_time - last_time < timegap:
            kind = "restart"
            n_breaks += 1
            logger.warning(f"Periods {index - 1} and {index} are {(first_time - last_time) / 60.0:.1f} min apart but no pair "
                           f"links them (make_bridge_pairs.py makes it): the tracks restart at the boundary.")
        else:
            kind = "gap"

        # Map of the local track numbers of this period to global numbers (valid local numbers are 1 .. ntracks - 1)
        mapping = np.zeros(p["ntracks"], dtype=np.int64)
        if kind == "seam":
            n0 = int((p["track_numbers"][0] > 0).sum())
            prev_numbers = blocks["track_numbers"][-1][-1]               # last column of the previous period, global numbers
            if int((prev_numbers > 0).sum()) != n0:
                raise ValueError(f"the seam file has {int((prev_numbers > 0).sum())} clouds in period {index - 1} and {n0} in "
                                 f"period {index}")
            mapping[1: n0 + 1] = prev_numbers[:n0]
            mapping[n0 + 1:] = g_end + np.arange(p["ntracks"] - n0 - 1)
            g_end = g_end + p["ntracks"] - (n0 + 1)
            first_kept = 1                                               # the seam column is rebuilt, B's own first column is dropped
        else:
            mapping[1:] = g_end + np.arange(p["ntracks"] - 1)
            g_end = g_end + p["ntracks"] - 1
            first_kept = 0

        relabeled = {name: _relabel(padded[name], mapping) for name in NUMBER_VARS}
        if kind == "seam":
            # The seam column from both sides: the last column of the previous period and the first column of this one
            a_status = blocks["track_status"][-1][-1]
            a_merge, a_split = blocks["track_mergenumbers"][-1][-1], blocks["track_splitnumbers"][-1][-1]
            if (a_merge > 0).any() or (relabeled["track_splitnumbers"][0] > 0).any():
                raise ValueError(f"the seam column of periods {index - 1} and {index} has merge numbers in the last column of "
                                 f"the first or split numbers in the first column of the second: not tracked by gettracknumbers")
            numbers = prev_numbers
            if not np.array_equal(numbers[:n0], relabeled["track_numbers"][0][:n0]):
                raise ValueError("the clouds of the seam file do not carry the same tracks in both periods")
            merge = np.where(relabeled["track_mergenumbers"][0] > 0, relabeled["track_mergenumbers"][0], a_merge)
            status = a_status + padded["track_status"][0]
            earlier = _column_before_last(blocks["track_numbers"])      # the column before the seam column
            if earlier is None:
                earlier = np.full(nclouds, FILLVAL)
            reset = np.where((numbers > 0) & ~np.isin(numbers, earlier[earlier > 0]), 0, FILLVAL).astype(padded["track_reset"].dtype)
            for name, column in (("track_numbers", numbers), ("track_status", status), ("track_mergenumbers", merge),
                                 ("track_splitnumbers", a_split), ("track_reset", reset)):
                blocks[name][-1] = np.concatenate([blocks[name][-1][:-1], column[None, :]], axis=0)

        for name in NUMBER_VARS:
            blocks[name].append(relabeled[name][first_kept:])
        blocks["track_status"].append(padded["track_status"][first_kept:])
        blocks["track_reset"].append(padded["track_reset"][first_kept:])
        basetimes.append(p["basetimes"][first_kept:])
        cloudid.append(p["cloudid"][first_kept:])

    result = {name: np.concatenate(blocks[name], axis=0) for name in blocks}
    result.update({"ntracks": g_end, "basetimes": np.concatenate(basetimes).astype("datetime64[s]"),
                   "cloudid": np.concatenate(cloudid, axis=0), "n_chain_breaks": n_breaks})
    return result


def stitch_tracknumbers(paths, outfile, tracking_dirs=None, featuresize_varname="npix_feature"):
    """
    Stitch the tracknumbers files of consecutive periods into one file with global track numbers.

    Args:
        paths: list of strings
            tracknumbers_*.nc of the periods, in time order. Neighbours share exactly one cloudid file (the seam) or none.
        outfile: string
            Output tracknumbers file (replaced if it exists). Its dates are the first start and the last end.
        tracking_dirs: list of strings (default: None = no check)
            The folder with the cloudid files of every period. With it the seam files of the two periods are compared
            (feature sizes and, if present, the feature numbers must be identical).
        featuresize_varname: string (default: 'npix_feature')
            Variable with the feature sizes in the cloudid files.

    Returns:
        outfile: string
            The file written.
    """
    periods = [read_period(p) for p in paths]
    if tracking_dirs is not None:
        for i in range(1, len(periods)):
            if periods[i]["basetimes"][0] == periods[i - 1]["basetimes"][-1]:
                _check_seam_file(periods[i - 1], tracking_dirs[i - 1], periods[i], tracking_dirs[i], featuresize_varname)
    result = stitch_arrays(periods)
    first, last = periods[0]["attrs"], periods[-1]["attrs"]
    if os.path.isfile(outfile):
        os.remove(outfile)
    write_tracknumbers(
        outfile, result["ntracks"], result["basetimes"], result["cloudid"],
        result["track_numbers"][None], result["track_status"][None], result["track_mergenumbers"][None],
        result["track_splitnumbers"][None], result["track_reset"][None], first["singletrack_filebase"], first["startdate"],
        last["enddate"], _timegap_hours(first), result["n_chain_breaks"])
    return outfile


def _check_seam_file(period_a, dir_a, period_b, dir_b, featuresize_varname):
    """
    The seam cloudid file is made by both periods; it must be identical.

    Args:
        period_a, period_b: dictionary
            Output of read_period of the two periods.
        dir_a, dir_b: string
            Folders with the cloudid files of the two periods.
        featuresize_varname: string
            Variable with the feature sizes.

    Raises:
        ValueError: if the feature sizes or feature numbers differ.
    """
    name = period_b["cloudid"][0].tobytes().decode().strip("\x00 ")
    with netCDF4.Dataset(os.path.join(dir_a, name)) as a, netCDF4.Dataset(os.path.join(dir_b, name)) as b:
        for var in (featuresize_varname, "feature_number"):
            if var in a.variables and var in b.variables:
                if not np.array_equal(np.asarray(a[var][:]), np.asarray(b[var][:])):
                    raise ValueError(f"the seam file {name} differs between {dir_a} and {dir_b} in {var}")
