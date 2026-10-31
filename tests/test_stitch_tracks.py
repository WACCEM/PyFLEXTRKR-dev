"""
Tests of pyflextrkr.stitch_tracks: the track numbers of consecutive periods, tracked apart and stitched, are identical to a
single run over all periods, for every position of the cut(s) on random data with continuations, mergers and splits
(tests/synthetic_tracks.py). The comparison covers every variable of the tracknumbers file: numbers, status codes, merge and
split numbers, start / end flags, file names and times.

Two kinds of random links: symmetric (the backward links mirror the forward links, like clean tracksingle output; the real
c4 and c7 tracking behaves like this) must give a bit-identical file. Asymmetric links (forward and backward drawn
independently, a harsher test) give the same file except for "stale" start flags that gettracknumbers leaves when it reaches a
cloud from two groups; the end flag written over the last column of a period hides them (synthetic_tracks.stale_reset_zeros).
"""
import itertools
import shutil

import netCDF4
import pytest
import xarray as xr

import synthetic_tracks as syn
from pyflextrkr import gettracks, stitch_tracks

BOTH_LINK_KINDS = pytest.mark.parametrize("symmetric", [True, False], ids=["symmetric", "asymmetric"])


def _valid_cuts(info):
    """
    Frames that can be the seam of two periods: a pair ends in them and a pair starts in them.
    """
    ends = {b for a, b in info["pairs"]}
    starts = {a for a, b in info["pairs"]}
    return sorted(ends & starts)


def _run_period(info, tmp_path, first, last, tag, maxnclouds=30):
    """
    gettracknumbers over the pairs from the one that has `first` as its reference file to the one that ends in `last`.
    """
    config = syn.make_config(info, str(tmp_path / f"stats_{tag}"), first=first, last=last, tag=tag, maxnclouds=maxnclouds)
    return gettracks.gettracknumbers(config)


def _stitched_and_continuous(info, tmp_path, cuts, maxnclouds=None, tag="x"):
    """
    Track the periods between the cuts apart, stitch them, and track everything at once.

    Args:
        info: dictionary
            Synthetic data.
        tmp_path: pathlib.Path
            Folder for the outputs.
        cuts: list of int
            Seam frames in time order.
        maxnclouds: list of int (default: 30 for every period)
            maxnclouds of every period; the continuous run uses the largest.
        tag: string (default: 'x')
            Part of the output names.

    Returns:
        stitched, continuous: dictionary, dictionary
            Output of syn.read_tracknumbers for the stitched and the continuous result.
    """
    bounds = [info["frames"][0]] + list(cuts) + [info["frames"][-1]]
    sizes = maxnclouds or [30] * (len(bounds) - 1)
    paths = [_run_period(info, tmp_path, a, b, f"{tag}p{i}", sizes[i]) for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:]))]
    out = str(tmp_path / f"stitched_{tag}.nc")
    stitch_tracks.stitch_tracknumbers(paths, out)
    whole = _run_period(info, tmp_path, bounds[0], bounds[-1], f"{tag}all", max(sizes))
    return syn.read_tracknumbers(out), syn.read_tracknumbers(whole)


@BOTH_LINK_KINDS
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_every_single_cut_equals_the_continuous_run(tmp_path, seed, symmetric):
    info = syn.write_synthetic_tracking(str(tmp_path / "data"), nframes=14, seed=seed, symmetric_links=symmetric)
    cuts = _valid_cuts(info)
    assert len(cuts) >= 10
    for cut in cuts:
        stitched, whole = _stitched_and_continuous(info, tmp_path, [cut], tag=f"s{cut}")
        assert syn.same_tracknumbers(stitched, whole, allow_stale_reset=not symmetric) == [], f"cut at frame {cut}"
        assert stitched["attrs"]["startdate"] == f"s{cut}p0-{info['frames'][0]}"


@BOTH_LINK_KINDS
def test_every_pair_of_cuts_equals_the_continuous_run(tmp_path, symmetric):
    info = syn.write_synthetic_tracking(str(tmp_path / "data"), nframes=13, seed=3, symmetric_links=symmetric)
    for cut_a, cut_b in itertools.combinations(_valid_cuts(info), 2):
        stitched, whole = _stitched_and_continuous(info, tmp_path, [cut_a, cut_b], tag=f"s{cut_a}_{cut_b}")
        assert syn.same_tracknumbers(stitched, whole, allow_stale_reset=not symmetric) == [], f"cuts at {cut_a} and {cut_b}"


@BOTH_LINK_KINDS
def test_periods_of_one_pair_each(tmp_path, symmetric):
    # Every frame is a seam: periods with a single track file
    info = syn.write_synthetic_tracking(str(tmp_path / "data"), nframes=11, seed=4, symmetric_links=symmetric)
    stitched, whole = _stitched_and_continuous(info, tmp_path, _valid_cuts(info), tag="many")
    assert syn.same_tracknumbers(stitched, whole, allow_stale_reset=not symmetric) == []


@BOTH_LINK_KINDS
@pytest.mark.parametrize("seed", [0, 1])
def test_boundary_at_a_data_gap_equals_the_continuous_run(tmp_path, seed, symmetric):
    # Frames 7, 8, 9 do not exist: 20 min between frames 6 and 10, more than timegap, so no pair links them
    info = syn.write_synthetic_tracking(str(tmp_path / "data"), nframes=16, seed=seed, skip_frames=(7, 8, 9),
                                        symmetric_links=symmetric)
    pa = _run_period(info, tmp_path, 0, 6, "gapA")
    pb = _run_period(info, tmp_path, 10, 15, "gapB")
    out = str(tmp_path / "gap_stitched.nc")
    stitch_tracks.stitch_tracknumbers([pa, pb], out)
    whole = _run_period(info, tmp_path, 0, 15, "gapall")
    assert syn.same_tracknumbers(syn.read_tracknumbers(out), syn.read_tracknumbers(whole), allow_stale_reset=not symmetric) == []
    # one more cut on each side of the gap
    stitched, whole = _stitched_and_continuous(info, tmp_path, [3, 12], tag="gap2")
    assert syn.same_tracknumbers(stitched, whole, allow_stale_reset=not symmetric) == []


def test_unlinked_but_close_boundary_restarts_like_a_missing_track_file(tmp_path):
    # No pair (5, 6): the periods end in frame 5 and start in frame 6, 5 min apart
    info = syn.write_synthetic_tracking(str(tmp_path / "data"), nframes=40, seed=2, skip_track_files=(6,), symmetric_links=True)
    pa = _run_period(info, tmp_path, 0, 5, "unlA")
    pb = _run_period(info, tmp_path, 6, 39, "unlB")
    out = str(tmp_path / "unl_stitched.nc")
    stitch_tracks.stitch_tracknumbers([pa, pb], out)
    whole = _run_period(info, tmp_path, 0, 39, "unlall")
    assert syn.same_tracknumbers(out, whole) == []
    assert int(syn.read_tracknumbers(out)["attrs"]["n_chain_breaks"]) == 1 == int(syn.read_tracknumbers(whole)["attrs"]["n_chain_breaks"])


def test_different_maxnclouds_are_padded_including_the_start_flag(tmp_path):
    info = syn.write_synthetic_tracking(str(tmp_path / "data"), nframes=14, seed=1, symmetric_links=True)
    cut = _valid_cuts(info)[3]
    stitched, whole = _stitched_and_continuous(info, tmp_path, [cut], maxnclouds=[20, 35], tag="pad")
    assert stitched["track_numbers"].shape[2] == 35
    assert syn.same_tracknumbers(stitched, whole) == []                  # the 1 flag of the first column covers the padded slots


def test_overlap_of_more_than_the_seam_file_is_refused(tmp_path):
    info = syn.write_synthetic_tracking(str(tmp_path / "data"), nframes=14, seed=1)
    pa = _run_period(info, tmp_path, 0, 8, "ovA")
    pb = _run_period(info, tmp_path, 5, 13, "ovB")
    with pytest.raises(ValueError, match="overlap"):
        stitch_tracks.stitch_tracknumbers([pa, pb], str(tmp_path / "ov.nc"))


def test_seam_file_that_differs_between_the_periods_is_refused(tmp_path):
    info_a = syn.write_synthetic_tracking(str(tmp_path / "dir_a"), nframes=14, seed=1)
    shutil.copytree(str(tmp_path / "dir_a"), str(tmp_path / "dir_b"))
    info_b = dict(info_a, folder=str(tmp_path / "dir_b"))
    seam = 6
    pa = _run_period(info_a, tmp_path, 0, seam, "sA")
    pb = _run_period(info_b, tmp_path, seam, 13, "sB")
    out = str(tmp_path / "ok.nc")
    stitch_tracks.stitch_tracknumbers([pa, pb], out, tracking_dirs=[info_a["folder"], info_b["folder"]])      # identical seam files
    # change the feature sizes of the seam file in one folder
    path = f"{info_b['folder']}/{info_b['names'][seam]}"
    ds = xr.open_dataset(path).load()
    ds.close()
    ds["npix_feature"] = ds["npix_feature"] + 1
    ds.to_netcdf(path)
    with pytest.raises(ValueError, match="differs between"):
        stitch_tracks.stitch_tracknumbers([pa, pb], str(tmp_path / "bad.nc"), tracking_dirs=[info_a["folder"], info_b["folder"]])


def test_stitched_file_is_a_normal_tracknumbers_file(tmp_path):
    info = syn.write_synthetic_tracking(str(tmp_path / "data"), nframes=14, seed=0, symmetric_links=True)
    cut = _valid_cuts(info)[4]
    stitched, whole = _stitched_and_continuous(info, tmp_path, [cut], tag="norm")
    out = str(tmp_path / "stitched_norm.nc")
    with netCDF4.Dataset(out) as ds:
        assert int(ds["ntracks"][0]) == int(whole["ntracks"][0])
        assert ds.getncattr("timegap") == "0.25-hours" and ds.getncattr("singletrack_filebase") == "track_"
        assert ds.getncattr("enddate") == f"normp1-{info['frames'][-1]}"
    # the stitched ids are the ones of the continuous run: the largest number used is ntracks - 1
    assert stitched["track_numbers"].max() == int(stitched["ntracks"][0]) - 1
