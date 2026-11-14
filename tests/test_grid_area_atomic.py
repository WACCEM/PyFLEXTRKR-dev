"""
save_grid_area writes the grid area file atomically: the monthly jobs of a campaign start at the same time, all find no file and all write it
(area_method: latlon), and no reader may ever see a half-written file. Several processes write the same file again and again while the main
process opens it as often as it can.
"""
import multiprocessing
import os

import numpy as np
import xarray as xr

from pyflextrkr.ft_utilities import load_grid_area, save_grid_area

NY, NX = 300, 400


def _grid():
    lat = np.tile(np.linspace(-60, 60, NY)[:, None], (1, NX))
    lon = np.tile(np.linspace(-180, 180, NX)[None, :], (NY, 1))
    return 100.0 * np.cos(np.deg2rad(lat)), lat, lon


def _write_many(path, n):
    area, lat, lon = _grid()
    for _ in range(n):
        save_grid_area(area, lat, lon, path)


def test_concurrent_writers_never_expose_a_partial_file(tmp_path):
    path = str(tmp_path / "stats" / "grid_area_from_latlon.nc")
    area, lat, lon = _grid()
    context = multiprocessing.get_context("fork")
    writers = [context.Process(target=_write_many, args=(path, 6)) for _ in range(6)]
    for process in writers:
        process.start()
    # While the writers work, open the file whenever it exists: it must always be complete
    n_read = 0
    while any(process.is_alive() for process in writers):
        if os.path.isfile(path):
            with xr.open_dataset(path) as ds:
                assert ds["grid_area"].shape == (NY, NX) and bool(np.isfinite(ds["grid_area"].values).all())
            n_read += 1
    for process in writers:
        process.join()
        assert process.exitcode == 0
    # The file has the content, and no temporary file of a writer is left
    assert np.allclose(load_grid_area(path), area.astype(np.float32))
    assert sorted(os.listdir(os.path.dirname(path))) == ["grid_area_from_latlon.nc"]
    assert n_read > 0
