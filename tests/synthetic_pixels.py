"""
Synthetic pixel-level files of a Tb+PF run (mcstrack_<yyyymodd>_<hhmmss>.nc: the track-number mask of the cold cloud shield, the mask of the
precipitation feature and the precipitation) and the row-table statistics of the three tracks in them, for the tests of the monthly rain
maps and of the accumulation of precipitation along the tracks. Six hourly files, 21:00 UTC on 31 January 2009 to 02:00 on 1 February (so the
period has two calendar months). Track 1 moves right and its path crosses the one of track 2, which comes later (a pixel two tracks cover at
different hours); track 3 has a second blob with the same number (a merged fragment); NaN precipitation at the border and inside a track.
"""
import calendar
import datetime
import os

import numpy as np
import xarray as xr

from pyflextrkr.rowstats import TableBuilder, write_rows

NY, NX, NHOUR = 30, 40, 6
T0 = calendar.timegm(datetime.datetime(2009, 1, 31, 21, 0, 0).timetuple())
STARTDATE, ENDDATE = "20090131.2100", "20090201.0300"


def hour_time(h):
    """Epoch time [s] of hour h."""
    return T0 + 3600 * h


def blobs(track, h):
    """
    The pixels of a track at hour h as a list of (rows, columns) slices (empty if the track does not exist at that hour).

    Args:
        track: int
            Track number 1 to 3.
        h: int
            Hour index 0 to 5.

    Returns:
        blobs: list of tuples of slices
    """
    if track == 1:
        return [(slice(10, 14), slice(2 + 2 * h, 6 + 2 * h))]
    if track == 2 and 2 <= h <= 5:
        return [(slice(10, 14), slice(6, 10))]
    if track == 3 and 0 <= h <= 3:
        return [(slice(20, 24), slice(30, 34)), (slice(25, 27), slice(30, 34))]
    return []


def make_fields():
    """
    The masks and the precipitation of the six hours.

    Returns:
        cloudtracknumber, pcptracknumber: int32 arrays [NHOUR, NY, NX]
            The track number at every pixel (0 outside the tracks); a later track overwrites an earlier one where they overlap. The
            precipitation feature is the part of the track with a rain rate above 8 mm/h.
        precipitation: float32 array [NHOUR, NY, NX]
            Rain rate [mm/h]: NaN in the two top rows (no data) and in one block inside track 3 at hour 1.
    """
    masks = np.zeros((NHOUR, NY, NX), dtype=np.int32)
    for h in range(NHOUR):
        for track in (1, 2, 3):
            for rows, cols in blobs(track, h):
                masks[h][rows, cols] = track
    yy, xx = np.mgrid[0:NY, 0:NX]
    precipitation = np.stack([1.0 + 0.1 * xx + 0.2 * yy + 2.0 * h for h in range(NHOUR)]).astype(np.float32)
    precipitation[:, 0:2, :] = np.nan
    precipitation[1, 21:23, 31:33] = np.nan
    pcpmask = np.where((masks > 0) & (np.nan_to_num(precipitation) > 8.0), masks, 0).astype(np.int32)
    return masks, pcpmask, precipitation


def write_pixel_files(folder):
    """
    Write the six pixel-level files.

    Args:
        folder: string
            Output folder (exists, with or without the trailing slash).

    Returns:
        fields: dictionary
            cloudtracknumber, pcptracknumber, precipitation (the arrays of make_fields).
    """
    masks, pcpmask, precipitation = make_fields()
    lon = np.tile(-100.0 + 0.1 * np.arange(NX)[None, :], (NY, 1))
    lat = np.tile(10.0 + 0.1 * np.arange(NY)[:, None], (1, NX))
    # Loop over the hours: one file each, with a time dimension of length 1 as the files of the tracking have
    for h in range(NHOUR):
        stamp = datetime.datetime.fromtimestamp(hour_time(h), datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        ds = xr.Dataset(
            {"cloudtracknumber": (("time", "lat", "lon"), masks[h][None]), "pcptracknumber": (("time", "lat", "lon"), pcpmask[h][None]),
             "precipitation": (("time", "lat", "lon"), precipitation[h][None], {"units": "mm/h"}),
             "longitude": (("lat", "lon"), lon), "latitude": (("lat", "lon"), lat)},
            coords={"time": [hour_time(h)], "lat": np.arange(NY), "lon": np.arange(NX)})
        ds.to_netcdf(os.path.join(folder, f"mcstrack_{stamp}.nc"))
    return {"cloudtracknumber": masks, "pcptracknumber": pcpmask, "precipitation": precipitation}


def write_track_statistics(stats_folder, filebase="mcs_tracks_robust_"):
    """
    Write the row tables of the three tracks (base_time, meanlon, meanlat at the hours they exist).

    Args:
        stats_folder: string
            Folder of the statistics files (with the trailing slash).
        filebase: string (default: 'mcs_tracks_robust_')
            File base name of the folder: <filebase><startdate>_<enddate>.rows.
    """
    hours = {track: [h for h in range(NHOUR) if blobs(track, h)] for track in (1, 2, 3)}
    duration = np.array([len(hours[t]) for t in (1, 2, 3)])
    base_time = np.array([hour_time(h) for t in (1, 2, 3) for h in hours[t]], dtype=np.float64)
    builder = TableBuilder(duration, int(duration.max()))
    builder.add_track_var("track_duration", duration.astype(np.int32), {"long_name": "duration"})
    builder.add_entry_var("base_time", base_time, {"_FillValue": np.nan}, fill=np.nan)
    builder.add_entry_var("meanlon", np.full(len(base_time), -99.0, dtype=np.float32), {"_FillValue": np.float32(np.nan)}, fill=np.nan)
    builder.add_entry_var("meanlat", np.full(len(base_time), 11.0, dtype=np.float32), {"_FillValue": np.float32(np.nan)}, fill=np.nan)
    write_rows(builder.build(), f"{stats_folder}{filebase}{STARTDATE}_{ENDDATE}.rows")
