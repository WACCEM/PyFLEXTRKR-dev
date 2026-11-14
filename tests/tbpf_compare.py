"""
Comparison of the outputs of two Tb+PF tracking runs, file by file, for the tests that the unified workflow gives the results of the old
scripts (tests/test_tbpf_unified_demo.py) and that a stitched run gives the results of a continuous one (tests/test_tbpf_stitch_demo.py).

A run is a root folder with the folders tracking/ (cloudid_*.nc, track_*.nc, the pairwise links), stats/ (tracknumbers, trackstats, the MCS
statistics files, as dense netCDF files or as row folders .rows) and mcstracking/<startdate>_<enddate>/ (the pixel-level files). Files are
the same if every variable has the same dimensions, dtype, values (NaN equals NaN) and attributes and the global attributes are the same,
except the time stamps Created_on and Created (the time the step ran). A row folder is compared with a dense file or another row folder through the
tables that rowstats makes of them.

    python tests/tbpf_compare.py ROOT_A ROOT_B [--rows-b]       # prints the differences, exit code 1 if there are any
"""
import argparse
import glob
import os
import re
import sys
from datetime import datetime

import numpy as np
import xarray as xr

from pyflextrkr.rowstats import dense_to_rows, read_tables, table_differences

# Time stamps of the global attributes: the time the step ran (Created_on; the tracknumbers file calls it Created)
IGNORED_GLOBAL_ATTRS = ("Created_on", "Created")
# The statistics files of a run, by the beginning of their names (the dates follow): a dense file is <name><dates>.nc, row tables
# <name><dates>.rows. The dense trackstats and the files of the steps are compared when both runs have them.
STATS_FILES = ("tracknumbers_", "trackstats_sparse_", "trackstats_", "mcs_tracks_", "mcs_tracks_pf_", "mcs_tracks_robust_",
               "mcs_tracks_final_")
ROW_STATS_FILES = ("mcs_tracks_", "mcs_tracks_pf_", "mcs_tracks_robust_", "mcs_tracks_final_")


def _attrs_equal(a, b):
    """
    Two attribute dictionaries are equal: the same keys and the same values (arrays by value, NaN equals NaN).
    """
    if set(a) != set(b):
        return False
    for key in a:
        x, y = np.asarray(a[key]), np.asarray(b[key])
        if x.shape != y.shape or not np.array_equal(x, y, equal_nan=x.dtype.kind == "f" and y.dtype.kind == "f"):
            return False
    return True


def netcdf_differences(path_a, path_b, ignore_global_attrs=IGNORED_GLOBAL_ATTRS):
    """
    What differs between two netCDF files.

    Args:
        path_a, path_b: string
            The files.
        ignore_global_attrs: tuple of strings (default: IGNORED_GLOBAL_ATTRS)
            Global attributes that may differ.

    Returns:
        differences: list of strings
            Variables that are in one file only, and variables with other dimensions, dtype, values or attributes; the global attributes.
            Empty if the files are the same.
    """
    out = []
    with xr.open_dataset(path_a, mask_and_scale=False, decode_times=False) as a, \
            xr.open_dataset(path_b, mask_and_scale=False, decode_times=False) as b:
        names_a, names_b = set(a.variables), set(b.variables)
        for name in sorted(names_a ^ names_b):
            out.append(f"{name}: only in the {'first' if name in names_a else 'second'} file")
        # Loop over the variables of both files: dimensions, dtype, values and attributes
        for name in sorted(names_a & names_b):
            x, y = a[name], b[name]
            if x.dims != y.dims or x.dtype != y.dtype or x.shape != y.shape:
                out.append(f"{name}: dimensions / dtype differ ({x.dims} {x.dtype} {x.shape} / {y.dims} {y.dtype} {y.shape})")
                continue
            xv, yv = x.values, y.values
            if not np.array_equal(xv, yv, equal_nan=xv.dtype.kind in "fc"):
                n_diff = int((xv != yv).sum()) if xv.dtype.kind not in "fc" else int(((xv != yv) & ~(np.isnan(xv) & np.isnan(yv))).sum())
                out.append(f"{name}: {n_diff} of {xv.size} values differ")
            if not _attrs_equal(x.attrs, y.attrs):
                out.append(f"{name}: attributes differ ({dict(x.attrs)} / {dict(y.attrs)})")
        attrs_a = {k: v for k, v in a.attrs.items() if k not in ignore_global_attrs}
        attrs_b = {k: v for k, v in b.attrs.items() if k not in ignore_global_attrs}
        if not _attrs_equal(attrs_a, attrs_b):
            keys = sorted(k for k in set(attrs_a) | set(attrs_b) if not _attrs_equal({k: attrs_a.get(k, "")}, {k: attrs_b.get(k, "")}))
            out.append(f"global attributes differ: {keys}")
    return out


def _tables(path):
    """
    The row tables of a statistics file: of a dense netCDF file (dense_to_rows) or of a row folder.
    """
    return read_tables(path) if os.path.isdir(path) else dense_to_rows(path)


def stats_differences(path_a, path_b):
    """
    What differs between two statistics files, each a dense netCDF file or a row folder: two netCDF files file as they are, otherwise the
    row tables are compared (columns, dtypes, values, fill values, attributes, dimensions).

    Args:
        path_a, path_b: string
            The files or folders.

    Returns:
        differences: list of strings
    """
    if not os.path.isdir(path_a) and not os.path.isdir(path_b):
        return netcdf_differences(path_a, path_b)
    return table_differences(_tables(path_a), _tables(path_b))


def _period_length(path):
    """
    The length of the period in the name of a statistics file (<name><start>_<end>.<extension>, the dates are YYYYMMDD.HHMM).

    Args:
        path: string
            Path of the file or row folder.

    Returns:
        length: datetime.timedelta or None
            None if the name does not hold two dates.
    """
    dates = re.findall(r"\d{8}\.\d{4}", os.path.basename(path))
    if len(dates) < 2:
        return None
    return datetime.strptime(dates[1], "%Y%m%d.%H%M") - datetime.strptime(dates[0], "%Y%m%d.%H%M")


def _stats_path(root, name, rows):
    """
    The statistics file of a run: <root>/stats/<name><dates>.nc, or .rows for the row tables (None if there is none).

    The root of a stitched campaign holds the tracknumbers file of every period next to the stitched one; with several files the one of the
    longest period (the file of the whole run) is taken, and None if two of them have that length or a name has no dates.
    """
    pattern = f"{root}/stats/{name}[0-9]*.{'rows' if rows else 'nc'}"
    found = sorted(glob.glob(pattern))
    if len(found) <= 1:
        return found[0] if found else None
    # Several files: the one of the longest period
    lengths = [_period_length(p) for p in found]
    if None in lengths:
        return None
    longest = [p for p, length in zip(found, lengths) if length == max(lengths)]
    return longest[0] if len(longest) == 1 else None


def compare_runs(root_a, root_b, rows_b=False, rows_a=False, tracking_a="tracking", tracking_b="tracking"):
    """
    Compare two runs: the pairwise links and cloudid files, the statistics files and the pixel-level files.

    Args:
        root_a, root_b: string
            Roots of the runs.
        rows_b: bool (default: False)
            The MCS statistics files of the second run are row folders (stats_format: rows); the first run has dense files unless
            rows_a. The files up to the trackstats are netCDF files in both.
        rows_a: bool (default: False)
            The MCS statistics files of the first run are row folders.
        tracking_a, tracking_b: string (default: 'tracking')
            Folder of the cloudid and track files of each run below its root (the folder of links that the stats job of a stitched campaign
            makes, e.g. 'tracking/campaign_p0_p2', for a stitched run).

    Returns:
        report: dictionary
            {'compared': {group: number of files compared}, 'differences': list of strings (file: difference)}. A file that one run has
            and the other has not is a difference.
    """
    compared, differences = {"tracking": 0, "stats": 0, "pixel": 0}, []

    # The cloudid files and the pairwise links of the tracking
    names_a = sorted(os.path.basename(p) for pattern in ("cloudid_*.nc", "track_*.nc") for p in glob.glob(f"{root_a}/{tracking_a}/{pattern}"))
    names_b = sorted(os.path.basename(p) for pattern in ("cloudid_*.nc", "track_*.nc") for p in glob.glob(f"{root_b}/{tracking_b}/{pattern}"))
    for name in sorted(set(names_a) ^ set(names_b)):
        differences.append(f"tracking/{name}: only in {'the first' if name in names_a else 'the second'} run")
    # Loop over the files of both runs
    for name in sorted(set(names_a) & set(names_b)):
        compared["tracking"] += 1
        differences += [f"tracking/{name}: {d}" for d in netcdf_differences(f"{root_a}/{tracking_a}/{name}", f"{root_b}/{tracking_b}/{name}")]

    # The statistics files (the MCS files are row folders in a run with stats_format: rows)
    for name in STATS_FILES:
        in_rows_a = rows_a and name in ROW_STATS_FILES
        in_rows_b = rows_b and name in ROW_STATS_FILES
        path_a, path_b = _stats_path(root_a, name, in_rows_a), _stats_path(root_b, name, in_rows_b)
        # trackstats_ matches the sparse file too: the dense file has a date right after the name (the pattern needs a digit)
        if path_a is None and path_b is None:
            continue
        if path_a is None or path_b is None:
            differences.append(f"stats/{name}: only in the {'second' if path_a is None else 'first'} run")
            continue
        compared["stats"] += 1
        differences += [f"stats/{name}: {d}" for d in stats_differences(path_a, path_b)]

    # The pixel-level files of the MCS
    pixels_a = sorted(glob.glob(f"{root_a}/mcstracking/*/mcstrack_*.nc"))
    pixels_b = sorted(glob.glob(f"{root_b}/mcstracking/*/mcstrack_*.nc"))
    base_a, base_b = [os.path.basename(p) for p in pixels_a], [os.path.basename(p) for p in pixels_b]
    for name in sorted(set(base_a) ^ set(base_b)):
        differences.append(f"mcstracking/{name}: only in {'the first' if name in base_a else 'the second'} run")
    lookup_b = dict(zip(base_b, pixels_b))
    for path, name in zip(pixels_a, base_a):
        if name in lookup_b:
            compared["pixel"] += 1
            differences += [f"mcstracking/{name}: {d}" for d in netcdf_differences(path, lookup_b[name])]
    return {"compared": compared, "differences": differences}


def main():
    """
    Command line: compare two runs and print the differences.
    """
    parser = argparse.ArgumentParser(description="Compare the outputs of two Tb+PF tracking runs.")
    parser.add_argument("root_a", help="root of the first run (dense statistics files)")
    parser.add_argument("root_b", help="root of the second run")
    parser.add_argument("--rows-b", action="store_true", help="the MCS statistics of the second run are row folders")
    args = parser.parse_args()
    report = compare_runs(args.root_a, args.root_b, rows_b=args.rows_b)
    print(f"compared: {report['compared']}")
    for line in report["differences"]:
        print(line)
    print(f"{len(report['differences'])} differences")
    return 1 if report["differences"] else 0


if __name__ == "__main__":
    sys.exit(main())
