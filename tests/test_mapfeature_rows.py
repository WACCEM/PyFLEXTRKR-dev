"""
Tests of the rows mode of mapfeature (pyflextrkr/mapfeature_rows.py): from the same robust statistics, the pixel files of
mapfeature_driver_rows are identical (every variable and attribute) to those of mapfeature_driver. Uses the synthetic case of
test_regionprops_radar_feature.py (a track with a merge fragment folded into it, a second track, 40 pixel files).
"""
import os

import numpy as np
import pytest
import xarray as xr

from pyflextrkr.mapfeature_driver import mapfeature_driver
from pyflextrkr.mapfeature_rows import mapfeature_driver_rows
from pyflextrkr.regionprops_driver import compute_regionprops
from pyflextrkr.rowstats import dense_to_rows, write_rows
from test_regionprops_radar_feature import NSTEPS, build_run
from test_rowstats import _same_attrs

DATES = "20250502.1900_20250502.2000"
ROBUST = f"mcs_tracks_robust_{DATES}"


def prepare(tmp_path, radius_km=10.0):
    """
    Cloudid files and a robust statistics file with the track status (dense, and the same as row tables).

    Args:
        tmp_path: pathlib.Path
            Folder to work in.
        radius_km: float (default: 10.0)
            Dilation radius of the convective groups of the synthetic case.

    Returns:
        config, arrays: dictionary, dictionary
            Config of the run and the arrays of build_run.
    """
    config, arrays = build_run(tmp_path, radius_km=radius_km)
    compute_regionprops(config)
    path = f"{config['stats_outpath']}{ROBUST}.nc"
    ds = xr.open_dataset(path, decode_times=False, mask_and_scale=False).load()
    ds.close()
    status = np.full(ds["cloudnumber"].shape, -9999, dtype=np.int32)
    status[:, :NSTEPS] = 1
    status[0, 10:20] = 3
    ds["track_status"] = (["tracks", "times"], status, {"comments": "test status"})
    os.remove(path)
    ds.to_netcdf(path)
    write_rows(dense_to_rows(path), f"{config['stats_outpath']}{ROBUST}.rows")
    config.update({'pixeltracking_filebase': 'mcstrack_', 'nmaxlinks': 3, 'feature_type': 'radar_feature', 'dbz_thresh': 10,
                   'fillval': -9999})
    return config, arrays


def pixel_files_differ(folder_a, folder_b):
    """
    Names of the pixel files that differ (or exist in one folder only), comparing every variable (dimensions, dtype, values)
    and the attributes of the variables and of the file (NaN equals NaN; the Created_on time stamp is ignored).

    Args:
        folder_a, folder_b: string
            Folders with the pixel files.

    Returns:
        names: list of strings
    """
    names_a, names_b = sorted(os.listdir(folder_a)), sorted(os.listdir(folder_b))
    differ = sorted(set(names_a) ^ set(names_b))
    for name in sorted(set(names_a) & set(names_b)):
        with xr.open_dataset(os.path.join(folder_a, name), decode_times=False, mask_and_scale=False) as a, \
                xr.open_dataset(os.path.join(folder_b, name), decode_times=False, mask_and_scale=False) as b:
            same = set(a.variables) == set(b.variables) and all(
                a[v].dims == b[v].dims and a[v].dtype == b[v].dtype and _same_attrs(a[v].attrs, b[v].attrs)
                and np.array_equal(a[v].values, b[v].values, equal_nan=a[v].dtype.kind == "f") for v in a.variables)
            ignore = {"Created_on"}
            same = same and _same_attrs({k: v for k, v in a.attrs.items() if k not in ignore},
                                        {k: v for k, v in b.attrs.items() if k not in ignore})
            if not same:
                differ.append(name)
    return differ


def run_both(tmp_path, **overrides):
    """Run the dense and the rows step on the same statistics, return the two pixel folders."""
    config, arrays = prepare(tmp_path)
    config.update(overrides)
    dense_dir, rows_dir = str(tmp_path / "pixel_dense") + "/", str(tmp_path / "pixel_rows") + "/"
    mapfeature_driver(dict(config, pixeltracking_outpath=dense_dir), trackstats_filebase="mcs_tracks_robust_")
    mapfeature_driver_rows(dict(config, pixeltracking_outpath=rows_dir), trackstats_filebase="mcs_tracks_robust_")
    return dense_dir, rows_dir, arrays


def test_pixel_files_of_the_rows_step_equal_those_of_the_dense_step(tmp_path):
    dense_dir, rows_dir, arrays = run_both(tmp_path)
    assert len(os.listdir(rows_dir)) == NSTEPS
    assert pixel_files_differ(dense_dir, rows_dir) == []
    # and they carry what the dense test expects: the merged mask of track 1 (feature 1 + folded fragment 2) and track 2
    with xr.open_dataset(rows_dir + sorted(os.listdir(rows_dir))[17], decode_times=False, mask_and_scale=False) as px:
        ctn, fn = px["cloudtracknumber"].values.squeeze(), arrays["feature_number"]
        assert np.all(ctn[np.isin(fn, [1, 2])] == 1) and np.all(ctn[fn == 3] == 2) and np.all(ctn[fn == 0] == 0)


def test_the_optional_feature_class_map(tmp_path):
    dense_dir, rows_dir, _ = run_both(tmp_path, mapfeature_class_varname="track_status")
    assert pixel_files_differ(dense_dir, rows_dir) == []
