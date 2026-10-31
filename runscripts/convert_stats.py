"""
Convert a track statistics file between the dense netCDF layout and the row tables (pyflextrkr/rowstats.py, docs/rowstats.md).

    python runscripts/convert_stats.py to-rows  mcs_tracks_robust_X.nc    mcs_tracks_robust_X.rows
    python runscripts/convert_stats.py to-dense mcs_tracks_robust_X.rows  mcs_tracks_robust_X.nc
    python runscripts/convert_stats.py to-dense mcs_tracks_robust_X.rows  mcs_mcs_X.nc --select is_mcs=1
    python runscripts/convert_stats.py to-dense mcs_tracks_robust_X.rows  subset.nc    --select track_class=1,2 --variables area majoraxis

to-rows reads the dense file block by block (memory of one block, not of the file) and checks that the dense layout can be
rebuilt exactly. to-dense builds the dense arrays of the selected tracks only (a whole year does not fit in memory as dense
arrays): --select NAME=V1,V2 keeps the tracks whose 1-D variable NAME has one of the values, --variables limits the
[tracks, times] variables. The tracks coordinate of the dense file keeps the track numbers of the whole file (pixel track number =
index + 1). The dense file is written with zlib compression and an unlimited tracks dimension, as the pipeline steps do.

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
    parser.add_argument("target", help="output row folder (to-rows) or dense file (to-dense)")
    parser.add_argument("--block-tracks", type=int, default=4000, help="to-rows: tracks per block (default: %(default)s)")
    parser.add_argument("--select", default=None, help="to-dense: keep the tracks with NAME=V1,V2 (a 1-D variable), e.g. is_mcs=1")
    parser.add_argument("--variables", nargs="+", default=None, help="to-dense: the [tracks, times] variables to write (default: all)")
    args = parser.parse_args(argv)

    from pyflextrkr.rowstats import RowStats, dense_to_rows, write_rows
    if args.direction == "to-rows":
        write_rows(dense_to_rows(args.source, block_tracks=args.block_tracks), args.target)
        print(f"{args.source} -> {args.target}")
        return 0
    stats = RowStats.open(args.source)
    tracks = select_tracks(stats, args.select)
    ds = stats.to_xarray(tracks=tracks, variables=args.variables)
    tracks_dim = stats.meta["tracks_dim"]
    ds.to_netcdf(args.target, mode="w", format="NETCDF4", unlimited_dims=tracks_dim,
                 encoding={v: dict(zlib=True) for v in ds.data_vars})
    print(f"{args.source} -> {args.target}: {ds.sizes[tracks_dim]} tracks, times dimension {ds.sizes[stats.meta['times_dim']]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
