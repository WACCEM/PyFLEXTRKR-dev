"""
The python recipes of docs/rowstats.md run on a row folder (made from the synthetic dense file of test_rowstats.py): the cookbook
cannot go stale. The first block is the setup (it names a file of a real run); the test replaces its path.
"""
import os
import re

import numpy as np

from pyflextrkr.rowstats import dense_to_rows, write_rows
from test_rowstats import _dense, _write


def _write_path(tmp_path):
    """The dense file that test_the_cookbook_recipes_run made (named dense.nc by test_rowstats._write)."""
    return str(tmp_path / "dense.nc")

DOC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "rowstats.md")


def python_blocks(path):
    """
    The fenced python blocks of a markdown file.

    Args:
        path: string
            Markdown file.

    Returns:
        blocks: list of strings
            The code of each ```python block, in order.
    """
    return re.findall(r"```python\n(.*?)```", open(path, encoding="utf-8").read(), flags=re.S)


def test_the_cookbook_recipes_run(tmp_path):
    folder = str(tmp_path / "mcs_tracks_robust.rows")
    write_rows(dense_to_rows(_write(tmp_path, _dense())), folder)
    blocks = python_blocks(DOC)
    assert len(blocks) >= 8 and blocks[0].startswith("# setup")
    namespace = {}
    for number, code in enumerate(blocks):
        code = re.sub(r'^path = ".*"$', f'path = "{folder}"', code, flags=re.M)
        try:
            exec(compile(code, f"rowstats.md block {number}", "exec"), namespace)
        except Exception as err:
            raise AssertionError(f"block {number} of docs/rowstats.md failed: {err!r}\n{code}") from err
    # a few of the results, checked against the dense file
    dense = _dense()
    assert namespace["ntracks"] == dense.sizes["tracks"] and namespace["nentries"] == int(dense["track_duration"].sum())
    selected = np.nonzero(dense["is_mcs"].values == 1)[0]
    assert namespace["selected"].tolist() == selected.tolist()
    assert np.allclose(namespace["dense_area"], dense["area"].values[selected][:, :namespace["dense_area"].shape[1]], equal_nan=True)
    assert namespace["tracks"].shape[0] == dense.sizes["tracks"] and len(namespace["mcs"]) == len(selected)
    assert namespace["peak"].index.tolist() == [t for t in selected if dense["track_duration"].values[t] > 0]
    assert np.isclose(namespace["value"], dense["area"].values[1, 0], equal_nan=True)
    # the dense file written by the cookbook is the dense file again
    import xarray as xr
    back = xr.open_dataset(namespace["dense_file"], mask_and_scale=False, decode_times=False)
    want = xr.open_dataset(_write_path(tmp_path), mask_and_scale=False, decode_times=False)
    assert os.path.basename(namespace["dense_file"]) == "mcs_tracks_robust.nc" and set(back.variables) == set(want.variables)
    assert all(np.array_equal(back[v].values, want[v].values, equal_nan=want[v].dtype.kind == "f") for v in want.variables)
