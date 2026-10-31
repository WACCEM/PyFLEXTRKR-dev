"""
steiner_func.remove_small_regions replaces the loops of mod_steiner_classification that looked for every connected region with
np.where(labels == region) (a scan of the whole grid per region). It must remove exactly the same pixels.
"""
import numpy as np
import pytest
import scipy.ndimage as ndi

from pyflextrkr.steiner_func import remove_small_regions


def _loop(field, min_npix):
    """
    The original loops (copy of the code that was in mod_steiner_classification).
    """
    field = np.copy(field)
    regions, num_regions = ndi.label(field)
    for rr in range(1, num_regions + 1):
        rid = np.where(regions == rr)
        if len(rid[0]) < min_npix:
            field[rid] = 0
    return field


@pytest.mark.parametrize("min_npix", [0, 1, 2, 3, 9, 40, 100000])
@pytest.mark.parametrize("seed", range(6))
def test_same_pixels_removed_as_the_loop(seed, min_npix):
    rng = np.random.default_rng(seed)
    smooth = ndi.gaussian_filter(rng.standard_normal((120, 150)), [1.0, 2.0, 3.0][seed % 3])
    field = (smooth > [0.0, 0.05, 0.15][seed % 3] * smooth.std()).astype(int)
    want = _loop(field, min_npix)
    got = remove_small_regions(np.copy(field), min_npix)
    assert np.array_equal(got, want) and got.dtype == field.dtype
    if min_npix in (9, 40):
        assert 0 < want.sum() < field.sum()                       # some regions go, some stay


def test_in_place_and_empty_field():
    field = np.zeros((10, 10), dtype=int)
    assert remove_small_regions(field, 5) is field and not field.any()
    field[2, 2] = 1
    field[6:9, 6:9] = 1
    out = remove_small_regions(field, 4)
    assert out is field and field[2, 2] == 0 and field[6:9, 6:9].all()
