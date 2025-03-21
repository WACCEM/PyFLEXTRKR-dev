def cloud_type_tracking(ct, pixel_radius, area_thresh, smoothsize, mincorecoldpix):
    ######################################################################
    # Import modules
    import numpy as np
    from scipy.ndimage import label, binary_dilation, generate_binary_structure, filters
    import logging

    logger = logging.getLogger(__name__)

    ######################################################################

    # Determine dimensions
    ny, nx = np.shape(ct)

    # Calculate area of one pixel. Assumed to be a circle.
    pixel_area = pixel_radius ** 2

    # Calculate minimum number of pixels based on area threshold
    nthresh = area_thresh / pixel_area

    # Threshold for deep ('core')
    thresh_core = 4
    thresh_concu = 3

    ######################################################################
    # Use thresholds identify pixels containing cold core, cold anvil, and warm anvil. Also create arrays with a flag for each type and fill in cloudid array. Cores = 1. Cold anvils = 2. Warm anvils = 3. Other = 4. Clear = 5. Areas do not overlap
    final_cloudid = np.zeros((ny, nx), dtype=int)

    core_flag = np.zeros((ny, nx), dtype=int)
    core_indices = np.where(ct == thresh_core)
    ncorepix = np.shape(core_indices)[1]
    logger.debug("ncorepix at 1503: ", ncorepix)
    if ncorepix > 0:
        core_flag[core_indices] = 1
        final_cloudid[core_indices] = 1

    coldanvil_flag = np.zeros((ny, nx), dtype=int)
    coldanvil_indices = np.where(ct == thresh_concu)
    ncoldanvilpix = np.shape(coldanvil_indices)[1]
    if ncoldanvilpix > 0:
        coldanvil_flag[coldanvil_indices] = 1
        final_cloudid[coldanvil_indices] = 2

    clear_flag = np.zeros((ny, nx), dtype=int)
    clear_indices = np.where(ct < thresh_concu)
    nclearpix = np.shape(clear_indices)[1]
    if nclearpix > 0:
        clear_flag[clear_indices] = 1
        final_cloudid[clear_indices] = 0

    #################################################################
    # Smooth IR data prior to identifying cores using a boxcar filter. Along the edges the boundary elements come from the nearest edge pixel
    smoothct = filters.uniform_filter(ct, size=smoothsize, mode="nearest")

    smooth_cloudid = np.zeros((ny, nx), dtype=int)

    core_indices = np.where(smoothct == thresh_core)
    ncorepix = np.shape(core_indices)[1]
    if ncorepix > 0:
        smooth_cloudid[core_indices] = 1

    coldanvil_indices = np.where(smoothct == thresh_concu)
    ncoldanvilpix = np.shape(coldanvil_indices)[1]
    if ncoldanvilpix > 0:
        smooth_cloudid[coldanvil_indices] = 2

    clear_indices = np.where(smoothct < thresh_concu)
    nclearpix = np.shape(clear_indices)[1]
    if nclearpix > 0:
        smooth_cloudid[clear_indices] = 5

    #################################################################
    # Find cold cores in smoothed data
    smoothcore_flag = np.zeros((ny, nx), dtype=int)
    smoothcore_indices = np.where(smoothct >= thresh_core)
    nsmoothcorepix = np.shape(smoothcore_indices)[1]
    if ncorepix > 0:
        smoothcore_flag[smoothcore_indices] = 1

    ##############################################################
    # Label cold cores in smoothed data
    labelcore_number2d, nlabelcores = label(smoothcore_flag)

    # Check is any cores have been identified
    if nlabelcores > 0:
        logger.info("entered nlabelcores > 0 loop")

        #############################################################
        # Check if cores satisfy size threshold
        labelcore_npix = np.ones(nlabelcores, dtype=int) * -9999
        for ilabelcore in range(1, nlabelcores + 1):
            temp_labelcore_npix = len(
                np.array(np.where(labelcore_number2d == ilabelcore))[0, :]
            )
            if temp_labelcore_npix > mincorecoldpix:
                labelcore_npix[ilabelcore - 1] = np.copy(temp_labelcore_npix)

        # Check if any of the cores passed the size threshold test
        ivalidcores = np.array(np.where(labelcore_npix > 0))[0, :]
        ncores = len(ivalidcores)
        if ncores > 0:
            # Isolate cores that satisfy size threshold
            labelcore_number1d = (
                np.copy(ivalidcores) + 1
            )  # Add one since label numbers start at 1 and indices, which validcores reports starts at 0
            labelcore_npix = labelcore_npix[ivalidcores]

            #########################################################3
            # Sort sizes largest to smallest
            order = np.argsort(labelcore_npix)
            order = order[::-1]
            sortedcore_npix = np.copy(labelcore_npix[order])

            # Re-number cores
            sortedcore_number1d = np.copy(labelcore_number1d[order])

            sortedcore_number2d = np.zeros((ny, nx), dtype=int)
            corestep = 0
            for isortedcore in range(0, ncores):
                sortedcore_indices = np.where(
                    labelcore_number2d == sortedcore_number1d[isortedcore]
                )
                nsortedcoreindices = np.shape(sortedcore_indices)[1]
                if nsortedcoreindices == sortedcore_npix[isortedcore]:
                    corestep = corestep + 1
                    sortedcore_number2d[sortedcore_indices] = np.copy(corestep)

            #####################################################
            # Spread cold cores outward until reach cold anvil threshold. Generates cold anvil.
            labelcorecold_number2d = np.copy(sortedcore_number2d)
            labelcorecold_npix = np.copy(sortedcore_npix)

            keepspreading = 1
            # Keep looping through dilating code as long as at least one feature is growing. At this point limit it to 20 dilations. Remove this once use real data.
            while keepspreading > 0:
                keepspreading = 0

                # Loop through each feature
                for ifeature in range(1, ncores + 1):
                    # Create map of single feature
                    featuremap = np.copy(labelcorecold_number2d)
                    featuremap[labelcorecold_number2d != ifeature] = 0
                    featuremap[labelcorecold_number2d == ifeature] = 1

                    # Find maximum extent of the of the feature
                    extenty = np.nansum(featuremap, axis=1)
                    extenty = np.array(np.where(extenty > 0))[0, :]
                    miny = extenty[0]
                    maxy = extenty[-1]

                    extentx = np.nansum(featuremap, axis=0)
                    extentx = np.array(np.where(extentx > 0))[0, :]
                    minx = extentx[0]
                    maxx = extentx[-1]

                    # Subset ir and map data to smaller region around feature. This reduces computation time. Add a 10 pixel buffer around the edges of the feature.
                    if minx <= 10:
                        minx = 0
                    else:
                        minx = minx - 10

                    if maxx >= nx - 10:
                        maxx = nx
                    else:
                        maxx = maxx + 11

                    if miny <= 10:
                        miny = 0
                    else:
                        miny = miny - 10

                    if maxy >= ny - 10:
                        maxy = ny
                    else:
                        maxy = maxy + 11

                    ctsubset = np.copy(ct[miny:maxy, minx:maxx])
                    fullsubset = np.copy(labelcorecold_number2d[miny:maxy, minx:maxx])
                    featuresubset = np.copy(featuremap[miny:maxy, minx:maxx])

                    # Dilate cloud region
                    dilationstructure = generate_binary_structure(
                        2, 1
                    )  # Defines shape of growth. This grows one pixel as a cross

                    dilatedsubset = binary_dilation(
                        featuresubset, structure=dilationstructure, iterations=1
                    ).astype(featuremap.dtype)

                    # Isolate region that was dilated.
                    expansionzone = dilatedsubset - featuresubset

                    # Only keep pixels in dilated regions that are below the warm anvil threshold and are not associated with another feature
                    expansionzone[
                        np.where((expansionzone == 1) & (fullsubset != 0))
                    ] = 0
                    expansionzone[
                        np.where((expansionzone == 1) & (ctsubset >= thresh_cold))
                    ] = 0

                    # Find indices of accepted dilated regions
                    expansionindices = np.column_stack(np.where(expansionzone == 1))

                    # Add the accepted dilated region to the map of the cloud numbers
                    labelcorecold_number2d[
                        expansionindices[:, 0] + miny, expansionindices[:, 1] + minx
                    ] = ifeature

                    # Add the number of expanded pixels to pixel count
                    labelcorecold_npix[ifeature - 1] = (
                        len(expansionindices[:, 0]) + labelcorecold_npix[ifeature - 1]
                    )

                    # Count the number of dilated pixels. Add to the keepspreading variable. As long as this variables is > 0 the code continues to run the dilating portion. Also at this point have a requirement that can't dilate more than 20 times. This shoudl be removed when have actual data.
                    keepspreading = keepspreading + len(
                        np.extract(expansionzone == 1, expansionzone)
                    )

        #############################################################
        # Create blank core and cold anvil arrays if no cores present
        elif ncores == 0:
            logger.info("ncores was equal to 0 at line 1677")
            labelcorecold_number2d = np.zeros((ny, nx), dtype=int)
            labelcorecold_npix = []
            sortedcore_npix = []
            sortedcorecold_number2d = []  # KB TESTING
            sortedcore_npix = []
            sortedcold_npix = []

        ############################################################
        # Label cold anvils that do not have a cold core

        # Find indices that satisfy cold anvil threshold or convective core threshold and is not labeled
        isolated_flag = np.zeros((ny, nx), dtype=int)
        isolated_indices = np.where(
            (labelcorecold_number2d == 0) & ((coldanvil_flag > 0) | (core_flag > 0))
        )
        nisolated = np.shape(isolated_indices)[1]
        if nisolated > 0:
            isolated_flag[isolated_indices] = 1

        # Label isolated cold cores or cold anvils
        labelisolated_number2d, nlabelisolated = label(isolated_flag)

        # Check if any features have been identified
        if nlabelisolated > 0:

            #############################################################
            # Check if features satisfy size threshold
            labelisolated_npix = np.ones(nlabelisolated, dtype=int) * -9999
            for ilabelisolated in range(1, nlabelisolated + 1):
                temp_labelisolated_npix = len(
                    np.array(np.where(labelisolated_number2d == ilabelisolated))[0, :]
                )
                if temp_labelisolated_npix > nthresh:
                    labelisolated_npix[ilabelisolated - 1] = np.copy(
                        temp_labelisolated_npix
                    )

            ###############################################################
            # Check if any of the features are retained
            ivalidisolated = np.array(np.where(labelisolated_npix > 0))[0, :]
            nlabelisolated = len(ivalidisolated)
            if nlabelisolated > 0:
                # Isolate cores that satisfy size threshold
                labelisolated_number1d = (
                    np.copy(ivalidisolated) + 1
                )  # Add one since label numbers start at 1 and indices, which validcores reports starts at 0
                labelisolated_npix = labelisolated_npix[ivalidisolated]

                ###########################################################
                # Sort sizes largest to smallest
                order = np.argsort(labelisolated_npix)
                order = order[::-1]
                sortedisolated_npix = np.copy(labelisolated_npix[order])

                # Re-number cores
                sortedisolated_number1d = np.copy(labelisolated_number1d[order])

                sortedisolated_number2d = np.zeros((ny, nx), dtype=int)
                isolatedstep = 0
                for isortedisolated in range(0, nlabelisolated):
                    sortedisolated_indices = np.where(
                        labelisolated_number2d
                        == sortedisolated_number1d[isortedisolated]
                    )
                    nsortedisolatedindices = np.shape(sortedisolated_indices)[1]
                    if nsortedisolatedindices == sortedisolated_npix[isortedisolated]:
                        isolatedstep = isolatedstep + 1
                        sortedisolated_number2d[sortedisolated_indices] = np.copy(
                            isolatedstep
                        )
            else:
                sortedisolated_number2d = np.zeros((ny, nx), dtype=int)
                sortedisolated_npix = []
        else:
            sortedisolated_number2d = np.zeros((ny, nx), dtype=int)
            sortedisolated_npix = []

        ##############################################################
        # Combine cases with cores and cold anvils with those that those only have cold anvils

        # Add feature to core - cold anvil map giving it a number one greater that tne number of valid cores. These cores are after those that have a cold anvil.
        labelcorecoldisolated_number2d = np.copy(labelcorecold_number2d)

        sortedisolated_indices = np.where(sortedisolated_number2d > 0)
        nsortedisolatedindices = np.shape(sortedisolated_indices)[1]
        if nsortedisolatedindices > 0:
            labelcorecoldisolated_number2d[sortedisolated_indices] = np.copy(
                sortedisolated_number2d[sortedisolated_indices]
            ) + np.copy(ncores)

        # Combine the npix data for cases with cores and cold anvils with those that those only have cold anvils
        labelcorecoldisolated_npix = np.hstack(
            (labelcorecold_npix, sortedisolated_npix)
        )
        ncorecoldisolated = len(labelcorecoldisolated_npix)

        # Initialize cloud numbers
        labelcorecoldisolated_number1d = np.arange(1, ncorecoldisolated + 1)

        # Sort clouds by size
        order = np.argsort(labelcorecoldisolated_npix)
        order = order[::-1]
        sortedcorecoldisolated_npix = np.copy(labelcorecoldisolated_npix[order])
        sortedcorecoldisolated_number1d = np.copy(labelcorecoldisolated_number1d[order])

        # Re-number cores
        sortedcorecoldisolated_number2d = np.zeros((ny, nx), dtype=int)
        final_ncorepix = np.ones(ncorecoldisolated, dtype=int) * -9999
        final_ncoldpix = np.ones(ncorecoldisolated, dtype=int) * -9999
        featurecount = 0
        for ifeature in range(0, ncorecoldisolated):
            feature_indices = np.where(
                labelcorecoldisolated_number2d
                == sortedcorecoldisolated_number1d[ifeature]
            )
            nfeatureindices = np.shape(feature_indices)[1]

            if nfeatureindices == sortedcorecoldisolated_npix[ifeature]:
                featurecount = featurecount + 1
                sortedcorecoldisolated_number2d[feature_indices] = np.copy(featurecount)

                final_ncorepix[featurecount - 1] = np.nansum(core_flag[feature_indices])
                final_ncoldpix[featurecount - 1] = np.nansum(
                    coldanvil_flag[feature_indices]
                )

        ##############################################
        # Save final matrices
        final_corecoldnumber = np.copy(sortedcorecoldisolated_number2d)
        final_ncorecold = np.copy(ncorecoldisolated)

        final_ncorepix = final_ncorepix[0:featurecount]
        final_ncoldpix = final_ncoldpix[0:featurecount]

        final_ncorecoldpix = final_ncorepix + final_ncoldpix

    ######################################################################
    # If no core is found, use cold anvil threshold to identify features
    else:
        logger.info("there was no core found line 1795")
        #################################################
        # Label regions with cold anvils and cores
        corecold_flag = core_flag + coldanvil_flag
        corecold_number2d, ncorecold = label(coldanvil_flag)

        ##########################################################
        # Loop through clouds and only keep those where core + cold anvil exceed threshold
        if ncorecold > 0:
            labelcorecold_number2d = np.zeros((ny, nx), dtype=int)
            labelcore_npix = np.ones(ncorecold, dtype=int) * -9999
            labelcold_npix = np.ones(ncorecold, dtype=int) * -9999
            featurecount = 0

            for ifeature in range(1, ncorecold + 1):
                feature_indices = np.where(corecold_number2d == ifeature)
                nfeatureindices = np.shape(feature_indices)[1]

                if nfeatureindices > 0:
                    temp_core = np.copy(core_flag[feature_indices])
                    temp_corenpix = np.nansum(temp_core)

                    temp_cold = np.copy(coldanvil_flag[feature_indices])
                    temp_coldnpix = np.nansum(temp_cold)

                    if temp_corenpix + temp_coldnpix >= nthresh:
                        featurecount = featurecount + 1

                        labelcorecold_number2d[feature_indices] = np.copy(featurecount)
                        labelcore_npix[featurecount - 1] = np.copy(temp_corenpix)
                        labelcold_npix[featurecount - 1] = np.copy(temp_coldnpix)

            ###############################
            # Update feature count
            ncorecold = np.copy(featurecount)
            labelcorecold_number1d = (
                np.array(np.where(labelcore_npix + labelcold_npix > 0))[0, :] + 1
            )

            ###########################################################
            # Reduce size of final arrays so only as long as number of valid features
            if ncorecold > 0:
                labelcore_npix = labelcore_npix[0:ncorecold]
                labelcold_npix = labelcold_npix[0:ncorecold]

                ##########################################################
                # Reorder base on size, largest to smallest
                labelcorecold_npix = labelcore_npix + labelcold_npix
                order = np.argsort(labelcorecold_npix)
                order = order[::-1]
                sortedcore_npix = np.copy(labelcore_npix[order])
                sortedcold_npix = np.copy(labelcold_npix[order])

                sortedcorecold_npix = np.add(sortedcore_npix, sortedcold_npix)

                # Re-number cores
                sortedcorecold_number1d = np.copy(labelcorecold_number1d[order])

                sortedcorecold_number2d = np.zeros((ny, nx), dtype=int)
                corecoldstep = 0
                for isortedcorecold in range(0, ncorecold):
                    sortedcorecold_indices = np.where(
                        labelcorecold_number2d
                        == sortedcorecold_number1d[isortedcorecold]
                    )
                    nsortedcorecoldindices = np.shape(sortedcorecold_indices)[1]
                    if nsortedcorecoldindices == sortedcorecold_npix[isortedcorecold]:
                        corecoldstep = corecoldstep + 1
                        sortedcorecold_number2d[sortedcorecold_indices] = np.copy(
                            corecoldstep
                        )

            ##############################################
            # Save final matrices
            final_corecoldnumber = np.copy(sortedcorecold_number2d)
            final_ncorecold = np.copy(ncorecold)
            final_ncorepix = np.copy(sortedcore_npix)
            final_ncoldpix = np.copy(sortedcold_npix)
            final_cloudid = np.copy(final_cloudid)

            final_ncorecoldpix = final_ncorepix + final_ncoldpix

            logger.info(f"final_cloudid.shape:  {final_cloudid.shape}")

    ##################################################################
    # Output data. Only done if core-cold exist in this file
    return {
        "final_nclouds": final_ncorecold,
        "final_ncorepix": final_ncorepix,
        "final_ncoldpix": final_ncoldpix,
        "final_ncorecoldpix": final_ncorecoldpix,
        "final_cloudtype": final_cloudid,
        "final_convcold_cloudnumber": final_corecoldnumber,
    }
