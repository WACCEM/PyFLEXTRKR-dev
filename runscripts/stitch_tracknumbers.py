"""
Stitch the tracknumbers files of consecutive periods into one file, as if gettracks had run over all of them at once
(pyflextrkr/stitch_tracks.py, docs/stitching.md).

    python runscripts/stitch_tracknumbers.py OUT.nc P1/tracknumbers_A.nc P2/tracknumbers_B.nc ... [--tracking-dirs D1 D2 ...]

The periods are given in time order. Neighbouring periods share exactly one cloudid file (the seam: the last file of one is the
first file of the next, which is what the monthly configs give when the end date of a month is the start date of the next) or none
(a data gap, or a restart if no track file links them). With --tracking-dirs (one folder of cloudid files per period) the seam file
of the two periods is compared; it must be identical.

Author: Zhe Feng | zhe.feng@pnnl.gov
"""
import argparse
import logging
import sys

from pyflextrkr.ft_utilities import setup_logging
from pyflextrkr.stitch_tracks import stitch_tracknumbers

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stitch the tracknumbers files of consecutive periods.')
    parser.add_argument('outfile', help='output tracknumbers file')
    parser.add_argument('tracknumbers', nargs='+', help='tracknumbers files of the periods, in time order')
    parser.add_argument('--tracking-dirs', nargs='+', default=None,
                        help='folder with the cloudid files of every period (same order); compares the seam files')
    parser.add_argument('--featuresize-varname', default='npix_feature', help='feature size variable (default: %(default)s)')
    args = parser.parse_args()

    setup_logging()
    if args.tracking_dirs is not None and len(args.tracking_dirs) != len(args.tracknumbers):
        sys.exit('--tracking-dirs needs one folder per period')
    out = stitch_tracknumbers(args.tracknumbers, args.outfile, args.tracking_dirs, args.featuresize_varname)
    logging.getLogger(__name__).info(f"Stitched {len(args.tracknumbers)} periods into {out}")
