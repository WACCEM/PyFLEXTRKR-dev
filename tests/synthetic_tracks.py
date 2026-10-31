"""
Synthetic cloudid and track files for the tests of gettracks and of the stitching of track numbers.

The files have the variables, encodings and attributes that tracksingle_drift.trackclouds writes and gettracknumbers reads
(npix_feature in the cloudid files; basetime_ref / basetime_new, refcloud_forward_index, newcloud_backward_index and the attributes
ref_file / new_file / ref_date / new_date in the track files), with random clouds and random links: continuations, mergers,
splits and combinations of both, forward and backward links that need not agree (as in the real files), clouds without a link, and
optional gaps in the data or missing track files. Not a test module (no test_ prefix); imported by the tests.
"""
import os

import netCDF4
import numpy as np
import pandas as pd
import xarray as xr

FILLVAL = -9999
TIME_ENCODING = {"dtype": "int64", "zlib": True, "units": "seconds since 1970-01-01"}
INDEX_ENCODING = {"dtype": "int", "zlib": True, "_FillValue": FILLVAL}
# Variables of a tracknumbers file that carry the tracking result
TRACK_VARIABLES = ("ntracks", "basetimes", "cloudid_files", "track_numbers", "track_status", "track_mergenumbers",
                   "track_splitnumbers", "track_reset")


def _stamp(epoch):
    """
    Date and time text of an epoch time, as in the file names.

    Args:
        epoch: int
            Seconds since 1970-01-01.

    Returns:
        text: string
            'YYYYMMDD_HHMMSS'.
    """
    t = pd.to_datetime(epoch, unit="s")
    return f"{t.strftime('%Y%m%d')}_{t.strftime('%H%M%S')}"


def _random_links(rng, n_from, n_to, nlinks, density):
    """
    Random links of the clouds of one file to the clouds of the other file, in the layout of the track files: row r (cloud r + 1)
    holds the numbers (1-based) of the linked clouds, filled with FILLVAL; one extra row for the "cloud" n + 1 that
    trackclouds always adds.

    Args:
        rng: numpy.random.Generator
        n_from, n_to: int
            Number of clouds in the file the rows belong to and in the other file.
        nlinks: int
            Maximum number of links per cloud (third dimension).
        density: float
            Probability that a cloud has links at all.

    Returns:
        links: numpy.ndarray [1, n_from + 1, nlinks] of int
    """
    links = np.full((1, n_from + 1, nlinks), FILLVAL, dtype=int)
    for r in range(1, n_from + 1):
        if rng.random() >= density:
            continue
        k = int(rng.integers(1, 3))                                             # one or two links
        targets = rng.choice(np.arange(1, n_to + 1), size=min(k, n_to), replace=False)
        if r <= n_to and rng.random() < 0.5:
            targets[0] = r                                                      # often the same index continues
        for j, q in enumerate(np.unique(targets)[:nlinks]):
            links[0, r - 1, j] = q
    return links


def write_synthetic_tracking(folder, nframes=12, seed=0, max_clouds=8, nlinks=4, dt=300, t0="2020-01-01 00:00:00",
                             skip_frames=(), skip_track_files=(), timegap_hours=0.25, link_density=0.8):
    """
    Write cloudid_*.nc and track_*.nc files of random clouds.

    Args:
        folder: string
            Output folder (created).
        nframes: int (default: 12)
            Number of frames, dt seconds apart.
        seed: int (default: 0)
            Seed of the random generator.
        max_clouds: int (default: 8)
            Clouds per frame: random between 1 and this.
        nlinks: int (default: 4)
            Maximum links per cloud.
        dt: int (default: 300)
            Seconds between frames.
        t0: string (default: '2020-01-01 00:00:00')
            Time of frame 0, UTC.
        skip_frames: tuple of int (default: none)
            Frames that do not exist (data gap).
        skip_track_files: tuple of int (default: none)
            Frames whose track file (the pair ending in that frame) is not written, although both frames exist.
        timegap_hours: float (default: 0.25)
            Frames further apart than this get no track file (as in tracksingle).
        link_density: float (default: 0.8)
            Probability that a cloud has links.

    Returns:
        info: dictionary
            frames (existing frame indices), epoch {frame: seconds}, names {frame: cloudid file name}, ncl {frame: clouds},
            npix {frame: sizes}, pairs [(frame a, frame b)] that have a track file, and the folder.
    """
    rng = np.random.default_rng(seed)
    os.makedirs(folder, exist_ok=True)
    t0_epoch = int(pd.Timestamp(t0, tz="UTC").timestamp())
    frames = [i for i in range(nframes) if i not in skip_frames]
    epoch = {i: t0_epoch + i * dt for i in frames}
    ncl = {i: int(rng.integers(1, max_clouds + 1)) for i in frames}
    npix = {i: rng.integers(1, 60, ncl[i]).astype(np.int32) for i in frames}
    names = {i: f"cloudid_{_stamp(epoch[i])}.nc" for i in frames}
    for i in frames:
        xr.Dataset({"npix_feature": (["feature"], npix[i]), "nfeatures": ([], np.int32(ncl[i]))}).to_netcdf(
            os.path.join(folder, names[i]))

    pairs = []
    for a, b in zip(frames[:-1], frames[1:]):
        hours = (epoch[b] - epoch[a]) / 3600.0
        if not (0 < hours < timegap_hours) or b in skip_track_files:
            continue
        forward = _random_links(rng, ncl[a], ncl[b], nlinks, link_density)
        backward = _random_links(rng, ncl[b], ncl[a], nlinks, link_density)
        ds = xr.Dataset(
            {
                "basetime_new": (["time"], np.array([pd.to_datetime(epoch[b], unit="s")], dtype="datetime64[ns]")),
                "basetime_ref": (["time"], np.array([pd.to_datetime(epoch[a], unit="s")], dtype="datetime64[ns]")),
                "newcloud_backward_index": (["time", "nclouds_new", "nlinks"], backward),
                "refcloud_forward_index": (["time", "nclouds_ref", "nlinks"], forward),
            },
            coords={"time": np.arange(1), "nclouds_new": np.arange(0, ncl[b] + 1), "nclouds_ref": np.arange(0, ncl[a] + 1),
                    "nlinks": np.arange(nlinks)},
            attrs={"ref_file": names[a], "new_file": names[b], "ref_date": _stamp(epoch[a]), "new_date": _stamp(epoch[b])},
        )
        ds.to_netcdf(os.path.join(folder, f"track_{_stamp(epoch[b])}.nc"), mode="w", format="NETCDF4", unlimited_dims="time",
                     encoding={"basetime_new": TIME_ENCODING, "basetime_ref": TIME_ENCODING,
                               "newcloud_backward_index": INDEX_ENCODING, "refcloud_forward_index": INDEX_ENCODING})
        pairs.append((a, b))
    return {"frames": frames, "epoch": epoch, "names": names, "ncl": ncl, "npix": npix, "pairs": pairs, "folder": folder,
            "timegap_hours": timegap_hours}


def make_config(info, stats_dir, first=None, last=None, maxnclouds=30, tag="run", auto_update_maxnclouds=False):
    """
    Config dictionary for gettracknumbers over the pairs that end in the frames after `first` up to `last`.

    The first pair of the period is the one that has `first` as its reference file, so two periods [a, k] and [k, b] share frame k
    (the seam) and every pair belongs to exactly one of them.

    Args:
        info: dictionary
            From write_synthetic_tracking.
        stats_dir: string
            Folder for the tracknumbers file (created).
        first, last: int (default: first and last existing frame)
            Frames of the period.
        maxnclouds: int (default: 30)
            Maximum number of clouds per file in the output arrays.
        tag: string (default: 'run')
            Part of the dates in the output file name, to keep several runs apart.
        auto_update_maxnclouds: bool (default: False)
            Let gettracknumbers raise maxnclouds to the data (rounded up to 100).

    Returns:
        config: dictionary
    """
    first = info["frames"][0] if first is None else first
    last = info["frames"][-1] if last is None else last
    os.makedirs(stats_dir, exist_ok=True)
    return {
        "tracking_outpath": info["folder"].rstrip("/") + "/", "stats_outpath": stats_dir.rstrip("/") + "/",
        "singletrack_filebase": "track_", "tracknumbers_filebase": "tracknumbers_",
        "startdate": f"{tag}-{first}", "enddate": f"{tag}-{last}",
        "start_basetime": info["epoch"][first] + 1 if first in info["epoch"] else int(first),
        "end_basetime": info["epoch"][last] if last in info["epoch"] else int(last),
        "timegap": info["timegap_hours"], "maxnclouds": maxnclouds, "featuresize_varname": "npix_feature",
        "fillval": FILLVAL, "auto_update_maxnclouds": auto_update_maxnclouds,
    }


def read_tracknumbers(path):
    """
    Raw arrays of a tracknumbers file (no masking, no time decoding).

    Args:
        path: string
            tracknumbers_*.nc.

    Returns:
        data: dictionary
            {variable: numpy array} for TRACK_VARIABLES, and the global attributes under 'attrs'.
    """
    with netCDF4.Dataset(path) as ds:
        ds.set_auto_maskandscale(False)
        data = {name: np.array(ds[name][:]) for name in TRACK_VARIABLES}
        data["attrs"] = {k: ds.getncattr(k) for k in ds.ncattrs()}
    return data


def file_names(data):
    """
    The cloudid file name of every column of a tracknumbers file.

    Args:
        data: dictionary
            Output of read_tracknumbers.

    Returns:
        names: list of strings
    """
    return ["".join(c.decode() for c in row.ravel()) for row in data["cloudid_files"]]


def same_tracknumbers(a, b):
    """
    Whether two tracknumbers files hold the same tracking result.

    Args:
        a, b: dictionary or string
            Output of read_tracknumbers, or the file path.

    Returns:
        problems: list of strings
            One entry per variable that differs (shape or values); empty if identical.
    """
    a = read_tracknumbers(a) if isinstance(a, str) else a
    b = read_tracknumbers(b) if isinstance(b, str) else b
    problems = []
    for name in TRACK_VARIABLES:
        if a[name].shape != b[name].shape:
            problems.append(f"{name}: shape {a[name].shape} vs {b[name].shape}")
        elif not np.array_equal(a[name], b[name]):
            problems.append(f"{name}: {int(np.count_nonzero(a[name] != b[name]))} elements differ")
    return problems
