"""
Convert a track statistics file between the dense netCDF layout and the row tables (pyflextrkr/rowstats.py, docs/rowstats.md).

    python runscripts/convert_stats.py to-rows  mcs_tracks_robust_X.nc    mcs_tracks_robust_X.rows
    python runscripts/convert_stats.py to-dense mcs_tracks_robust_X.rows  mcs_tracks_robust_X.nc
    python runscripts/convert_stats.py to-dense mcs_tracks_robust_X.rows                  # target: the name of the old dense file (.rows -> .nc)
    python runscripts/convert_stats.py to-dense mcs_tracks_robust_X.rows  mcs_mcs_X.nc --select is_mcs=1
    python runscripts/convert_stats.py to-dense mcs_tracks_robust_X.rows  subset.nc    --select track_class=1,2 --variables area majoraxis

to-rows reads the dense file block by block (memory of one block, not of the file) and checks that the dense layout can be
rebuilt exactly. to-dense writes the dense file block by block too (rowstats.write_dense_netcdf, --block-tracks tracks at a
time; the result does not depend on the block size), so a month (180 GB of dense arrays) needs the memory of one block; a whole
year as dense arrays is too big for any node: --select NAME=V1,V2 keeps the tracks whose 1-D variable NAME has one of the values,
--variables limits the [tracks, times] variables. The tracks coordinate of the dense file keeps the track numbers of the whole
file (pixel track number = index + 1). The dense file is written with zlib compression and an unlimited tracks dimension, as the
pipeline steps do, and is the same file the dense step would have written (values, dtypes, fill values, attributes). An existing
target is an error unless --overwrite is given.

Author: Zhe Feng | zhe.feng@pnnl.gov
"""
import argparse
import sys

import numpy as np


def select_tracks(stats, select):
    """
    Tracks whose 1-D variable has one of the given values.

    Args:
        stats: pyflextrkr.rowstats.RowStats
            Opened row folder.
        select: string or None
            'NAME=V1,V2,...' (values are compared as numbers); None: all tracks.

    Returns:
        tracks: int array or None
            Track indices, None for all tracks.
    """
    if select is None:
        return None
    name, _, values = select.partition("=")
    if not values or name not in stats.meta["vars_1d"]:
        raise SystemExit(f"--select needs NAME=V1,V2 with NAME a 1-D variable of the file: {sorted(stats.meta['vars_1d'])}")
    return np.nonzero(np.isin(stats.tracks[name], [float(v) for v in values.split(",")]))[0]


def main(argv=None):
    """
    Command line of the converter.

    Args:
        argv: list of strings (default: None = sys.argv[1:])
            Arguments.

    Returns:
        status: int
            0 if the file was written.
    """
    parser = argparse.ArgumentParser(description="Convert a track statistics file between dense netCDF and row tables.")
    parser.add_argument("direction", choices=["to-rows", "to-dense"], help="to-rows: dense .nc -> row folder; to-dense: row folder -> dense .nc")
    parser.add_argument("source", help="input file (to-rows) or row folder (to-dense)")
    parser.add_argument("target", nargs="?", default=None,
                        help="output row folder (to-rows) or dense file (to-dense); default: the name of the dense file of the old "
                             "step (mcs_tracks_X.rows -> mcs_tracks_X.nc), or mcs_tracks_X.nc -> mcs_tracks_X.rows")
    parser.add_argument("--block-tracks", type=int, default=4000, help="tracks per block, to-rows and to-dense (default: %(default)s)")
    parser.add_argument("--select", default=None, help="to-dense: keep the tracks with NAME=V1,V2 (a 1-D variable), e.g. is_mcs=1")
    parser.add_argument("--variables", nargs="+", default=None, help="to-dense: the [tracks, times] variables to write (default: all)")
    parser.add_argument("--overwrite", action="store_true", help="to-dense: replace an existing target file")
    args = parser.parse_args(argv)

    from pyflextrkr.rowstats import RowStats, dense_netcdf_name, dense_to_rows, rows_folder, write_dense_netcdf, write_rows
    if args.direction == "to-rows":
        target = args.target or (args.source[:-3] + ".rows" if args.source.endswith(".nc") else args.source + ".rows")
        write_rows(dense_to_rows(args.source, block_tracks=args.block_tracks), target)
        print(f"{args.source} -> {target}")
        return 0
    source = rows_folder(args.source)
    stats = RowStats.open(source)
    tracks = select_tracks(stats, args.select)
    target = args.target or dense_netcdf_name(source)
    try:
        write_dense_netcdf(source, target, tracks=tracks, variables=args.variables, block_tracks=args.block_tracks,
                           overwrite=args.overwrite)
    except FileExistsError as error:
        raise SystemExit(f"{error}")
    ntracks = len(stats.duration) if tracks is None else len(tracks)
    print(f"{source} -> {target}: {ntracks} tracks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
