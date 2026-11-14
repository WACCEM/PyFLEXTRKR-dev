"""
Synthetic Tb+PF data for the tests of the rows-mode MCS steps: a sparse trackstats file with the variables of a real Tb+PF run
(core / cold cloud area, merge / split links, ...), the PF statistics file that matchpf makes from it (random PF variables with the
dimensions of the real file) and the config of the steps. Every track has an entry at every time index (as real tracks do).
"""
import os

import numpy as np
import xarray as xr

T0 = 1746200000.0           # epoch time of time step 0 [s]
DT = 3600.0                 # hourly data
NMAXPF = 3                  # precipitation features per time
STARTDATE, ENDDATE = "20250101.0000", "20250103.0000"

# Attributes of the variables of the sparse file, as in a real file (a float fill is NaN except base_time, which a real file
# labels with -9999 although the dense step fills it with NaN)
_FLOAT_VARS = ("meanlat", "meanlon", "area", "core_area", "cold_area", "corecold_mintb", "corecold_meantb", "core_meantb",
               "lat_mintb", "lon_mintb")
_INT_VARS = ("cloudnumber", "track_status", "track_interruptions", "merge_tracknumbers", "split_tracknumbers")


def tb_config(stats_path, **overrides):
    """
    The config keys of the Tb+PF MCS steps (identifymcs, matchpf, robust MCS) for the synthetic files.

    Args:
        stats_path: string
            Folder of the statistics files (with the trailing slash).
        **overrides:
            Config keys to change or add.

    Returns:
        config: dictionary
    """
    config = {
        "stats_outpath": stats_path, "startdate": STARTDATE, "enddate": ENDDATE, "trackstats_sparse_filebase": "trackstats_sparse_",
        "mcstbstats_filebase": "mcs_tracks_", "mcspfstats_filebase": "mcs_tracks_pf_", "mcsrobust_filebase": "mcs_tracks_robust_",
        "datatimeresolution": 1.0, "duration_range": [2, 40], "duration_range_auto_update": True, "tracks_dimname": "tracks",
        "times_dimname": "times", "pf_dimname": "nmaxpf", "nmaxlinks": 4, "fillval": -9999, "nmaxpf": NMAXPF,
        "mcs_tb_area_thresh": 40000, "mcs_tb_duration_thresh": 4, "mcs_tb_split_duration": 3, "mcs_tb_merge_duration": 3,
        "mcs_tb_gap": 1, "max_speed_for_trajectory_smoothing": 50,
        "mcs_pf_majoraxis_thresh": 0, "max_pf_majoraxis_thresh": 100000, "mcs_pf_durationthresh": 4, "mcs_pf_majoraxis_for_lifetime": 20,
        "mcs_pf_gap": 1, "pf_rr_thres": 2.0, "heavy_rainrate_thresh": 10, "landfrac_thresh": [0, 90], "mcs_min_rainvol_thresh": 20000,
        "mcs_volrain_duration_thresh": 1.0,
        "coefs_pf_area": [4.2, 0.2], "coefs_pf_rr": [0.5, 0.01], "coefs_pf_skew": [0.2, 0.01], "coefs_pf_heavyratio": [10, 0.5],
    }
    config.update(overrides)
    return config


def random_tb_sparse_file(stats_path, seed, ntracks=60, longest=28, nlong=0, crowd=0):
    """
    Write a sparse trackstats file of random tracks: long roots (a third of them with a cold cloud shield above 40,000 km^2 most of
    the time, with dips that split their MCS periods), roots that are never MCS (small shield, or no cold core), and short tracks
    that merge into and / or split from a root, most of them while the root exists.

    Args:
        stats_path: string
            Folder to write to (exists); the file is trackstats_sparse_<STARTDATE>_<ENDDATE>.nc.
        seed: int
            Random seed.
        ntracks: int (default: 60)
            Number of tracks.
        longest: int (default: 28)
            Longest duration of a root [time steps].
        nlong: int (default: 0)
            Number of MCS-sized roots made longer than duration_range (to test the width of the dense layout).
    crowd: int (default: 0)
        Number of the last tracks that are one-step fragments merging into the first MCS-sized root at the same time (they fill the
        slots 0, 1, 2, ... of that time and, with fewer slots than fragments, do not all fit).

    Returns:
        info: dictionary
            duration, start, merge_into (track number, 1-based, -9999 for none), split_from, is_big (MCS-sized roots).
    """
    rng = np.random.default_rng(seed)
    nroots = ntracks // 2
    duration = np.zeros(ntracks, dtype=np.int32)
    start = np.zeros(ntracks, dtype=np.int64)
    is_big = np.zeros(ntracks, dtype=bool)
    merge_into = np.full(ntracks, -9999, dtype=np.int32)
    split_from = np.full(ntracks, -9999, dtype=np.int32)

    # Roots: long tracks, a third of them MCS-sized; the others stay below the shield threshold or have no cold core
    for i in range(nroots):
        is_big[i] = rng.random() < 0.35
        duration[i] = rng.integers(8, longest + 1)
        start[i] = rng.integers(0, 30)
    big = np.flatnonzero(is_big)
    duration[big[:nlong]] = 45
    # Fragments: short tracks, most of them linked to an MCS-sized root (the Tb step uses only the links to MCS), the others to any
    # track. The start offsets inside the root come from a small set, so that several fragments share a time of one root and
    # fill the slots 0, 1, 2, ... (and, with few slots, do not fit)
    big_roots = np.flatnonzero(is_big)
    for i in range(nroots, ntracks):
        parent = int(rng.choice(big_roots)) if len(big_roots) and rng.random() < 0.75 else int(rng.integers(0, i))
        duration[i] = rng.integers(1, 4)
        inside = rng.random() < 0.9 and duration[parent] > duration[i]
        offset = min(int(rng.choice([0, 1, 2, 3])), max(int(duration[parent]) - int(duration[i]), 0))
        start[i] = start[parent] + offset if inside else rng.integers(0, 60)
        kind = rng.integers(0, 4)
        if kind in (0, 2):
            merge_into[i] = parent + 1
        if kind in (1, 2):
            # a split from another MCS-sized root (a fragment that merges into one MCS and splits from another)
            other = int(rng.choice(big_roots)) if len(big_roots) and kind == 2 else parent
            split_from[i] = other + 1
    # The crowd: one-step fragments that all merge into the first MCS-sized root at its time index 2
    for i in range(ntracks - crowd, ntracks):
        duration[i], start[i] = 1, start[big_roots[0]] + 2
        merge_into[i], split_from[i] = big_roots[0] + 1, -9999
    width = int(duration.max())
    shape = (ntracks, width)
    valid = np.arange(width)[None, :] < duration[:, None]

    # Areas: the shield of an MCS-sized root is mostly above the threshold; fragments and the other roots are small
    core = np.where(valid, rng.uniform(0, 4000, shape), 0.0)
    cold = np.where(valid, rng.uniform(1000, 30000, shape), 0.0)
    core[rng.random(ntracks) < 0.1, :] = 0.0                                   # tracks without a cold core
    big_rows = np.flatnonzero(is_big)
    core[big_rows] = np.where(valid[big_rows], rng.uniform(3000, 15000, (len(big_rows), width)), 0.0)
    cold[big_rows] = np.where(valid[big_rows], rng.uniform(35000, 90000, (len(big_rows), width)), 0.0)
    dips = (rng.random(shape) < 0.12) & valid
    cold[dips & is_big[:, None]] = 3000.0                                      # periods split by dips, some too short
    core, cold = core.astype(np.float32), cold.astype(np.float32)

    # Positions, cloud numbers and the other variables of the sparse file
    base_time = T0 + (start[:, None] + np.arange(width)[None, :]) * DT
    lon = (-98 + np.cumsum(rng.normal(0, 0.1, shape), axis=1)).astype(np.float32)
    lat = (10 + np.cumsum(rng.normal(0, 0.05, shape), axis=1)).astype(np.float32)
    lon[rng.random(shape) < 0.03] += 3.0                                       # an outlier for the trajectory smoothing
    lon[(rng.random(shape) < 0.03) & valid] = np.nan
    rows, cols = np.nonzero(valid)
    values = {
        "base_time": base_time, "meanlat": lat, "meanlon": lon, "area": (core + cold).astype(np.float32),
        "cloudnumber": rng.integers(1, 800, shape).astype(np.int32), "track_status": rng.integers(0, 40, shape).astype(np.int32),
        "track_interruptions": np.zeros(shape, dtype=np.int32),
        "merge_tracknumbers": np.full(shape, -9999, dtype=np.int32), "split_tracknumbers": np.full(shape, -9999, dtype=np.int32),
        "core_area": core, "cold_area": cold,
        "corecold_mintb": rng.uniform(180, 230, shape).astype(np.float32), "corecold_meantb": rng.uniform(200, 240, shape).astype(np.float32),
        "core_meantb": rng.uniform(190, 225, shape).astype(np.float32), "lat_mintb": lat, "lon_mintb": lon,
    }
    variables = {name: (["sparse_index"], arr[rows, cols]) for name, arr in values.items()}
    variables["tracks_indices"] = (["sparse_index"], rows.astype(np.int32))
    variables["times_indices"] = (["sparse_index"], cols.astype(np.int32))
    # 1-D variables of the tracks
    ints = lambda v: np.asarray(v, dtype=np.int32)
    variables["track_duration"] = (["tracks"], duration, {"long_name": "Duration of each track"})
    variables["start_status"] = (["tracks"], ints(rng.integers(0, 40, ntracks)))
    variables["end_status"] = (["tracks"], ints(rng.integers(0, 40, ntracks)))
    variables["start_basetime"] = (["tracks"], T0 + start * DT, {"units": "seconds since 1970-01-01"})
    variables["end_basetime"] = (["tracks"], T0 + (start + duration - 1) * DT, {"units": "seconds since 1970-01-01"})
    variables["start_split_tracknumber"] = (["tracks"], split_from)
    variables["start_split_timeindex"] = (["tracks"], ints(np.where(split_from > 0, 1, -9999)))
    variables["start_split_cloudnumber"] = (["tracks"], ints(np.where(split_from > 0, 7, -9999)))
    variables["end_merge_tracknumber"] = (["tracks"], merge_into)
    variables["end_merge_timeindex"] = (["tracks"], ints(np.where(merge_into > 0, 2, -9999)))
    variables["end_merge_cloudnumber"] = (["tracks"], ints(np.where(merge_into > 0, 9, -9999)))
    ds = xr.Dataset(variables, coords={"tracks": np.arange(ntracks), "sparse_index": np.arange(len(rows))},
                    attrs={"Title": "Statistics of each track", "startdate": STARTDATE, "enddate": ENDDATE,
                           "timegap_hour": 3.1, "time_resolution_hour": 1.0, "pixel_radius_km": 10.0})
    # Fill values as the attributes of a real file
    for name in _FLOAT_VARS:
        ds[name].attrs["_FillValue"] = np.float32(np.nan)
    for name in _INT_VARS:
        ds[name].attrs["_FillValue"] = np.int32(-9999)
    ds["base_time"].attrs["_FillValue"] = -9999.0
    ds["area"].attrs["long_name"] = "Area of the feature"
    ds.to_netcdf(os.path.join(stats_path, f"trackstats_sparse_{STARTDATE}_{ENDDATE}.nc"))
    return {"duration": duration, "start": start, "merge_into": merge_into, "split_from": split_from, "is_big": is_big}


def write_cloudid_files(folder, basetimes, ny=4, nx=360):
    """
    Write minimal cloudid files (only the longitude and latitude that the periodic-boundary thresholds are read from) named like the
    files of the tracking: cloudid_<yyyymodd>_<hhmmss>.nc.

    Args:
        folder: string
            Output folder (with the trailing slash, exists).
        basetimes: array-like of float
            Epoch time [s] of every file.
        ny, nx: int (default: 4, 360)
            Grid size; the grid spacing is 1 degree, so the longitude jump of a periodic boundary is half of nx.
    """
    import pandas as pd
    lon = np.tile(np.arange(nx) - nx / 2 + 0.5, (ny, 1))
    lat = np.tile((np.arange(ny) - ny / 2 + 0.5)[:, None], (1, nx))
    for t in basetimes:
        name = pd.to_datetime(float(t), unit="s").strftime("cloudid_%Y%m%d_%H%M%S.nc")
        xr.Dataset({"longitude": (["y", "x"], lon), "latitude": (["y", "x"], lat)}).to_netcdf(os.path.join(folder, name))


def fake_matchtbpf_singlefile(cloudid_filename, cloudnumber, merge_cloudnumber, split_cloudnumber, config):
    """
    A deterministic stand-in of matchtbpf_func.matchtbpf_singlefile: the same arguments and the same kind of result (a dictionary of
    [n] and [n, nmaxpf] arrays, their attributes and the names of the [n] variables), values that are simple functions of the cloud
    numbers, the file and the merge / split slots (so that a wrong slot width, a wrong entry or a wrong order shows in the values),
    NaN where a PF is missing and longitudes that cross the periodic boundary. None for every fifth file (a file that could not be
    read) and for a file that none of the entries belongs to.

    Args:
        cloudid_filename: string
            Cloudid file name.
        cloudnumber: int array [n]
            Cloud numbers of the n entries that belong to the file.
        merge_cloudnumber, split_cloudnumber: int arrays [n, nslots]
            Cloud numbers that merge into / split from the MCS of each entry (-9999 for an empty slot).
        config: dictionary
            Config (nmaxpf).

    Returns:
        result: tuple or None
            (out_dict, out_dict_attrs, var_names_2d), or None.
    """
    code = sum(map(ord, os.path.basename(cloudid_filename)))
    if len(cloudnumber) == 0 or code % 5 == 0:
        return None
    k = float(code % 17)
    base = np.asarray(cloudnumber, dtype=np.float64)
    nmaxpf = config["nmaxpf"]
    pf = np.arange(nmaxpf)[None, :]
    nmerge, nsplit = (merge_cloudnumber > 0).sum(axis=1), (split_cloudnumber > 0).sum(axis=1)
    per_pf = lambda scale, shift: base[:, None] * scale + k * shift + pf * 1.5
    area = per_pf(3.0, 2.0)
    area[base % 5 == 0, nmaxpf - 1] = np.nan                                       # a PF that does not exist
    out = {
        "pf_npf": (np.asarray(cloudnumber) % (nmaxpf + 1)).astype(np.int16),
        "pf_landfrac": (base % 100) * 1.0,
        "total_rain": base * 0.5 + k,
        # the merge / split slots: how many are filled, their sum, and the width of the array that was passed
        "total_heavyrain": base + nmerge + 10.0 * nsplit + 0.001 * merge_cloudnumber.shape[1] + (merge_cloudnumber.sum(axis=1) % 7),
        "total_volrain": base * 100.0 + k, "total_heavyvolrain": base * 10.0, "rainrate_heavyrain": base * 0.1 + 12.0,
        "rainrate_p95": base * 0.2,
        "pf_area": area, "pf_rainrate": per_pf(0.1, 0.5), "pf_majoraxis": per_pf(1.0, 0.25), "pf_skewness": per_pf(0.01, 0.1),
        "pf_accumrain": per_pf(5.0, 1.0), "pf_accumrainheavy": per_pf(1.0, 0.5), "pf_maxrainrate": per_pf(0.3, 0.2),
        # longitudes that jump across the periodic boundary from one time to the next; latitudes that jump by more than a few degrees
        "pf_lon": (base[:, None] * 37.0 + k * 61.0 + pf * 13.0) % 360.0 - 180.0,
        "pf_lat": (base[:, None] * 7.0 + k * 3.0 + pf * 2.0) % 12.0 - 6.0,
        "pf_lon_centroid": (base[:, None] * 41.0 + k * 59.0) % 360.0 - 180.0 + 0 * pf,
        "pf_lat_centroid": (base[:, None] * 5.0 + k * 3.0) % 12.0 - 6.0 + 0 * pf,
        "pf_lon_maxrainrate": (base[:, None] * 43.0 + k * 53.0 + pf) % 360.0 - 180.0,
        "pf_lat_maxrainrate": (base[:, None] * 11.0 + k * 3.0 + pf) % 12.0 - 6.0,
    }
    names_1d = ["pf_npf", "pf_landfrac", "total_rain", "total_heavyrain", "total_volrain", "total_heavyvolrain", "rainrate_heavyrain",
                "rainrate_p95"]
    # The attributes of the real function: every variable has a _FillValue (the integer fill value on the count of PF, NaN on the others)
    attrs = {name: {"long_name": f"synthetic {name}", "units": "unitless", "_FillValue": np.nan} for name in out}
    attrs["pf_npf"]["_FillValue"] = config["fillval"]
    return out, attrs, names_1d


def random_pf_stats_file(stats_path, seed, config, ntracks=150):
    """
    The PF statistics file of a Tb+PF run, mcs_tracks_pf_<dates>.nc, and the row folder of the same name (.rows): the Tb statistics of
    the MCS of random tracks (identifymcs_tb) with random precipitation-feature variables. A third of the MCS have a strong PF all the
    time, a third a weak one, a third a strong one with dips; the smaller PF (slots 1 and 2) are often NaN, some times have no PF at
    all, some tracks have no rain at all (heavy rain ratio 0 / 0) and some PF major axes are exactly the lifetime threshold.

    Args:
        stats_path: string
            Folder (exists, with the trailing slash) for the sparse file and the statistics files.
        seed: int
            Random seed.
        config: dictionary
            Config of the steps (tb_config).
        ntracks: int (default: 150)
            Number of random tracks before the MCS selection.

    Returns:
        dense_file, rows_folder: string, string
            The PF statistics as a dense netCDF file and as row tables.
    """
    from pyflextrkr.identifymcs import identifymcs_tb
    from pyflextrkr.rowstats import dense_to_rows, write_rows
    random_tb_sparse_file(stats_path[:-1], seed, ntracks=ntracks)
    tb_file = identifymcs_tb(config)
    ds = xr.open_dataset(tb_file, mask_and_scale=False, decode_times=False).load()
    rng = np.random.default_rng(seed + 500)
    nt, nk = ds.sizes["tracks"], ds.sizes["times"]
    valid = np.arange(nk)[None, :] < ds["track_duration"].values[:, None]
    style = rng.choice([0, 1, 2], nt, p=[0.25, 0.35, 0.4])[:, None, None]      # 0 weak, 1 strong, 2 strong with dips
    dip = (rng.random((nt, nk, 1)) < 0.3) & (style == 2)
    strong = ((style >= 1) & ~dip).astype(np.float64)
    shape = (nt, nk, NMAXPF)
    slot_scale = np.array([1.0, 0.5, 0.25])[None, None, :]                    # the largest PF is slot 0

    def pf(low, high, strong_low, strong_high):
        """A float32 [tracks, times, nmaxpf] array: values between low and high (weak) or strong_low and strong_high (strong)."""
        weak_values = rng.uniform(low, high, shape)
        strong_values = rng.uniform(strong_low, strong_high, shape)
        return (np.where(strong > 0, strong_values, weak_values) * slot_scale).astype(np.float32)

    majoraxis = pf(5, 40, 25, 90)
    area = pf(10, 800, 1500, 6000)
    rainrate = pf(0.1, 3, 2, 10)
    skewness = pf(-0.5, 0.5, 0.5, 3.0)
    accumrain = pf(0, 5000, 20000, 60000)
    heavy_fraction = rng.uniform(0.0, 0.6, (nt, 1, 1))
    accumheavy = (accumrain * heavy_fraction).astype(np.float32)
    maxrainrate = pf(0.5, 12, 8, 40)
    # No rain at all on some tracks (heavy rain ratio 0 / 0), no PF at some times, the smaller PF often missing
    norain = rng.random(nt) < 0.1
    accumrain[norain], accumheavy[norain] = 0.0, 0.0
    nopf = rng.random((nt, nk, 1)) < 0.08
    missing_small = (rng.random(shape) < 0.35) & (np.arange(NMAXPF)[None, None, :] > 0)
    # A PF major axis exactly at the lifetime threshold on some entries
    exact = rng.random((nt, nk)) < 0.05
    majoraxis[exact, 0] = np.float32(config["mcs_pf_majoraxis_for_lifetime"])
    arrays = {"pf_majoraxis": majoraxis, "pf_area": area, "pf_rainrate": rainrate, "pf_skewness": skewness,
              "pf_accumrain": accumrain, "pf_accumrainheavy": accumheavy, "pf_maxrainrate": maxrainrate}
    ds["nmaxpf"] = ("nmaxpf", np.arange(NMAXPF))
    for name, values in arrays.items():
        values[missing_small | nopf] = np.nan
        values[~valid] = np.nan
        ds[name] = (["tracks", "times", "nmaxpf"], values, {"long_name": name, "units": "unitless", "_FillValue": np.float32(np.nan)})
    # [tracks, times] PF variables (the count of PF is float32 with NaN fill and the integer fill value in its attributes, as in a file of the step)
    volrain = np.where(strong[:, :, 0] > 0, rng.uniform(15000, 60000, (nt, nk)), rng.uniform(0, 15000, (nt, nk))).astype(np.float32)
    volrain[norain] = 0.0
    volrain[~valid | nopf[:, :, 0]] = np.nan
    ds["total_volrain"] = (["tracks", "times"], volrain, {"long_name": "volrain", "units": "mm/h km^2", "_FillValue": np.float32(np.nan)})
    npf = np.where(valid & ~nopf[:, :, 0], rng.integers(0, NMAXPF + 1, (nt, nk)), np.nan).astype(np.float32)
    ds["pf_npf"] = (["tracks", "times"], npf, {"long_name": "npf", "units": "unitless", "_FillValue": np.float32(-9999)})
    ds.attrs.update({"nmaxpf": NMAXPF, "PF_rainrate_thresh": 2.0, "heavy_rainrate_thresh": 10.0, "landfrac_thresh": np.array([0, 90])})
    dense_file = f"{stats_path}mcs_tracks_pf_{STARTDATE}_{ENDDATE}.nc"
    ds.to_netcdf(dense_file, format="NETCDF4", unlimited_dims="tracks")
    rows_folder = f"{stats_path}mcs_tracks_pf_{STARTDATE}_{ENDDATE}.rows"
    write_rows(dense_to_rows(dense_file), rows_folder)
    return dense_file, rows_folder


def assert_same_dense(got, want):
    """
    Two dense Datasets are the same: variables, dimensions, dtypes, values (NaN equals NaN), attributes and the global attributes,
    except the time stamp Created_on (it is the time the step ran: two steps that run in different seconds differ in it).

    Args:
        got, want: xarray.Dataset
            The datasets (e.g. rows_to_dense of the row tables, and the dense file of the dense step opened with mask_and_scale=False).
    """
    from test_rowstats import _assert_same_dataset
    got, want = got.copy(), want.copy()
    for ds in (got, want):
        ds.attrs.pop("Created_on", None)
    _assert_same_dataset(got, want)
