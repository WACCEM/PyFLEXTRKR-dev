"""
FROZEN COPY of the overlap loops of pyflextrkr/tracksingle_drift.py as of commit e457d07 (before overlap_links.py replaced them):
for every feature of the reference file a scan of the whole grid for the overlapping features of the new file, and the same
backwards. Reference for tests/test_overlap_links.py. Do not edit.
"""
import logging
import sys

import numpy as np


def overlap_links_reference(reference_convcold_cloudnumber, new_convcold_cloudnumber, nreference, nnew, nmaxlinks, othresh,
                            fillval):
    """
    The original loops. Returns reference_forward_index, reference_forward_size, new_backward_index, new_backward_size.
    """
    logger = logging.getLogger(__name__)
    reference_file = new_file = ""

    #######################################################
    # Initialize matrices
    reference_forward_index = (
        np.ones((1, int(nreference), int(nmaxlinks)), dtype=int) * fillval
    )
    reference_forward_size = (
        np.ones((1, int(nreference), int(nmaxlinks)), dtype=int) * fillval
    )
    new_backward_index = (
        np.ones((1, int(nnew), int(nmaxlinks)), dtype=int) * fillval
    )
    new_backward_size = np.ones((1, int(nnew), int(nmaxlinks)), dtype=int) * fillval

    ######################################################
    # Loop through each cloud / feature in reference time and look for overlaping clouds / features in the new file
    for refindex in np.arange(1, nreference + 1):
        # Locate where the cloud in the reference file overlaps with any cloud in the new file
        forward_matchindices = np.where(
            (reference_convcold_cloudnumber == refindex)
            & (new_convcold_cloudnumber != 0)
        )

        # Get the convcold_cloudnumber of the clouds in the new file that overlap the cloud in the reference file
        forward_newindex = new_convcold_cloudnumber[forward_matchindices]
        unique_forwardnewindex = np.unique(forward_newindex)

        # Calculate size of reference cloud in terms of number of pixels
        sizeref = len(
            np.extract(
                reference_convcold_cloudnumber == refindex,
                reference_convcold_cloudnumber,
            )
        )

        # Loop through the overlapping clouds in the new file, determining if they statisfy the overlap requirement
        forward_nmatch = 0  # Initialize overlap counter
        for matchindex in unique_forwardnewindex:
            sizematch = len(
                np.extract(forward_newindex == matchindex, forward_newindex)
            )

            if sizematch / float(sizeref) > othresh:
                if forward_nmatch > nmaxlinks:
                    logger.debug(
                        ("reference: " + reference_file)
                    )
                    logger.debug(("new: " + new_file))
                    sys.exit(
                        "More than "
                        + str(int(nmaxlinks))
                        + " clouds in new file match with reference cloud?!"
                    )
                else:
                    reference_forward_index[
                        0, int(refindex) - 1, forward_nmatch
                    ] = matchindex
                    reference_forward_size[
                        0, int(refindex) - 1, forward_nmatch
                    ] = len(
                        np.extract(
                            new_convcold_cloudnumber == matchindex,
                            new_convcold_cloudnumber,
                        )
                    )

                    forward_nmatch = forward_nmatch + 1

    ######################################################
    # Loop through each cloud / feature at new time and look for overlaping clouds / features in the reference file
    for newindex in np.arange(1, nnew + 1):
        # Locate where the cloud in the new file overlaps with any cloud in the reference file
        backward_matchindices = np.where(
            (new_convcold_cloudnumber == newindex)
            & (reference_convcold_cloudnumber != 0)
        )

        # Get the convcold_cloudnumber of the clouds in the reference file that overlap the cloud in the new file
        backward_refindex = reference_convcold_cloudnumber[backward_matchindices]
        unique_backwardrefindex = np.unique(backward_refindex)

        # Calculate size of reference cloud in terms of number of pixels
        sizenew = len(
            np.extract(
                new_convcold_cloudnumber == newindex, new_convcold_cloudnumber
            )
        )

        # Loop through the overlapping clouds in the new file, determining if they statisfy the overlap requirement
        backward_nmatch = 0  # Initialize overlap counter
        for matchindex in unique_backwardrefindex:
            sizematch = len(
                np.extract(backward_refindex == matchindex, backward_refindex)
            )

            if sizematch / float(sizenew) > othresh:
                if backward_nmatch > nmaxlinks:
                    logger.debug(
                        ("reference: " + reference_file)
                    )
                    logger.debug(("new: " + new_file))
                    sys.exit(
                        "More than "
                        + str(int(nmaxlinks))
                        + " clouds in reference file match with new cloud?!"
                    )
                else:
                    new_backward_index[
                        0, int(newindex) - 1, backward_nmatch
                    ] = matchindex
                    new_backward_size[0, int(newindex) - 1, backward_nmatch] = len(
                        np.extract(
                            reference_convcold_cloudnumber == matchindex,
                            reference_convcold_cloudnumber,
                        )
                    )

                    backward_nmatch = backward_nmatch + 1

    return reference_forward_index, reference_forward_size, new_backward_index, new_backward_size
