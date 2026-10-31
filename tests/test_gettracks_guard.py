"""
Regression tests of gettracks.py against the frozen original (tests/gettracks_reference.py): factoring out write_tracknumbers
and adding the chain check change nothing on consistent input; a missing track file restarts the tracks instead of carrying the
track numbers of the previous file to the wrong clouds. Synthetic files (tests/synthetic_tracks.py), no data needed.
"""
import numpy as np
import pytest

import gettracks_reference as ref
import synthetic_tracks as syn
from pyflextrkr import gettracks


def _run(info, tmp_path, name, module, **kwargs):
    """
    Run gettracknumbers of a module over the data of `info` and read the result.
    """
    config = syn.make_config(info, str(tmp_path / name), tag=name, **kwargs)
    path = module.gettracknumbers(config)
    return syn.read_tracknumbers(path)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("skip_frames", [(), (7,), (7, 8, 9)])           # none, a gap shorter than timegap, a real gap
def test_same_result_as_the_original_on_consistent_input(tmp_path, seed, skip_frames):
    info = syn.write_synthetic_tracking(str(tmp_path / "data"), nframes=16, seed=seed, skip_frames=skip_frames)
    new = _run(info, tmp_path, "new", gettracks)
    old = _run(info, tmp_path, "old", ref)
    assert syn.same_tracknumbers(new, old) == []
    assert int(new["attrs"]["n_chain_breaks"]) == 0
    if len(skip_frames) == 3:                                             # the real gap shows as an end (2) and a restart (1)
        assert (new["track_reset"] == 2).any(axis=2).sum() == 2 and (new["track_reset"] == 1).any(axis=2).sum() == 2


def test_missing_track_file_restarts_the_tracks(tmp_path):
    # the track file of the pair (5, 6) is missing; (4, 5) and (6, 7) are 10 min apart, closer than timegap, so only the chain check sees it
    # (40 frames: one missing track file is 2.6 % of them, below the 5 % limit at which gettracknumbers stops)
    info = syn.write_synthetic_tracking(str(tmp_path / "data"), nframes=40, seed=5, skip_track_files=(6,))
    assert (5, 6) not in info["pairs"] and (4, 5) in info["pairs"] and (6, 7) in info["pairs"]
    new = _run(info, tmp_path, "new", gettracks)
    assert int(new["attrs"]["n_chain_breaks"]) == 1
    resets = new["track_reset"][0]
    ends = np.nonzero((resets == 2).any(axis=1))[0]
    starts = np.nonzero((resets == 1).any(axis=1))[0]
    # the tracks end in the column of frame 5 and restart in a new column that holds frame 6 again
    names = syn.file_names(new)
    assert names[ends[0]] == info["names"][5] and names[ends[0] + 1] == info["names"][6]
    assert list(starts) == [0, ends[0] + 1] and names[ends[0] + 1] == info["names"][6]
    # all clouds of frame 6 are new tracks: their numbers are larger than every number before
    tn = new["track_numbers"][0]
    assert tn[ends[0] + 1][tn[ends[0] + 1] > 0].min() > tn[: ends[0] + 1].max()
    # the original carries the numbers of frame 5 to frame 6 without notice: no restart
    old = _run(info, tmp_path, "old", ref)
    assert not (old["track_reset"][0, 1:-1] == 2).any()


def test_too_many_missing_track_files_stop(tmp_path):
    info = syn.write_synthetic_tracking(str(tmp_path / "data"), nframes=20, seed=1, skip_track_files=tuple(range(2, 20, 2)))
    with pytest.raises(SystemExit):
        _run(info, tmp_path, "new", gettracks)
