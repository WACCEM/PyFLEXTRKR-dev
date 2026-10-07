#!/usr/bin/env python
"""
Prepare a work directory to compare two code revisions of the merge/split handling of
identifymcs_radar on the SAME tracking. Nothing is run here, files are only set up:

  1. the two revisions (pyflextrkr/ and runscripts/) are exported with `git archive` into
     <workdir>/code_old and <workdir>/code_new
       old = the parent of the commit titled "...keep merge/split fragments that do not coexist with their root"
             (found with `git log --grep`, so it survives a rebase; override with --old-rev)
       new = HEAD (override with --new-rev)
  2. one root per version, <workdir>/old and <workdir>/new, each with
       stats/     the track statistics written by the tracking steps (copied: they are small)
       tracking/  the cloudid and track files (hard-linked: they are large and only read here)
  3. <workdir>/{old,new}/config.yml: the given MCS tracking config with root_path pointing at that root,
     only run_identifymcs and run_mapfeature on, and duration_range given explicitly (the old code has no
     guard for a duration_range that is shorter than the longest track)

Then run each version with run_tree.py (on a compute node: mapfeature is parallel):

  python run_tree.py <workdir>/code_old <workdir>/old/config.yml
  python run_tree.py <workdir>/code_new <workdir>/new/config.yml

and analyse with analyze_mask_difference.py, make_synthetic_figure.py and make_realdata_keyframes.py.

usage: setup_comparison.py --workdir DIR --source-root DIR --config MCS_CONFIG.yml
"""
import os
import sys
import glob
import shutil
import argparse
import subprocess
from pathlib import Path
import yaml

FIX_TITLE = "keep merge/split fragments that do not coexist with their root"


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def resolve_old_rev(repo, title):
    sha = git(repo, "log", "--format=%H", "-1", f"--grep={title}")
    if not sha:
        sys.exit(f"no commit with '{title}' in its title found in {repo}; pass --old-rev")
    return git(repo, "rev-parse", f"{sha}^")


def export_revision(repo, rev, dest):
    """git archive <rev> pyflextrkr runscripts | tar -x -C dest"""
    dest.mkdir(parents=True, exist_ok=True)
    arc = subprocess.Popen(["git", "-C", str(repo), "archive", rev, "pyflextrkr", "runscripts"], stdout=subprocess.PIPE)
    subprocess.run(["tar", "-x", "-C", str(dest)], stdin=arc.stdout, check=True)
    arc.stdout.close()
    if arc.wait() != 0:
        sys.exit(f"git archive {rev} failed")
    print(f"  {dest}  <-  {rev[:10]}  {git(repo, 'log', '-1', '--format=%s', rev)}")


def link_tree(src, dst):
    """Hard-link every file of src into dst (copy when linking across file systems is not possible)."""
    dst.mkdir(parents=True, exist_ok=True)
    n_link = n_copy = 0
    for entry in os.scandir(src):
        if not entry.is_file():
            continue
        target = dst / entry.name
        if target.exists():
            continue
        try:
            os.link(entry.path, target)
            n_link += 1
        except OSError:
            shutil.copy2(entry.path, target)
            n_copy += 1
    return n_link, n_copy


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workdir", required=True, help="directory to create (code_old, code_new, old, new)")
    ap.add_argument("--source-root", required=True,
                    help="tracking root that already holds stats/ and tracking/ (the root_path of the MCS config)")
    ap.add_argument("--config", required=True, help="MCS tracking config (feature_type: radar_feature)")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]), help="PyFLEXTRKR git checkout")
    ap.add_argument("--old-rev", default=None, help="previous revision (default: parent of the fix commit)")
    ap.add_argument("--new-rev", default="HEAD", help="current revision")
    ap.add_argument("--nprocesses", type=int, default=48, help="Dask workers used when each version is run")
    ap.add_argument("--duration-range-max", type=int, default=None,
                    help="max of duration_range [time steps]; default: the next multiple of 10 above the longest track")
    args = ap.parse_args()

    work, repo, src = Path(args.workdir).resolve(), Path(args.repo), Path(args.source_root).resolve()
    cfg_in = yaml.full_load(open(args.config))
    dates = f"{cfg_in['startdate']}_{cfg_in['enddate']}"
    work.mkdir(parents=True, exist_ok=True)

    old_rev = args.old_rev or resolve_old_rev(repo, FIX_TITLE)
    print("Code revisions:")
    export_revision(repo, old_rev, work / "code_old")
    export_revision(repo, args.new_rev, work / "code_new")

    # longest track in the sparse stats -> duration_range (the old code cannot expand it by itself)
    sparse = src / "stats" / f"trackstats_sparse_{dates}.nc"
    if not sparse.is_file():
        sys.exit(f"missing {sparse}: run the tracking steps (through run_trackstats) first")
    dmax = args.duration_range_max
    if dmax is None:
        import numpy as np
        import xarray as xr
        with xr.open_dataset(sparse, decode_times=False, mask_and_scale=False) as ds:
            longest = int(np.max(ds["times_indices"].values)) + 1
        dmax = int(np.ceil(longest / 10.0) * 10)
        print(f"Longest track: {longest} steps -> duration_range max {dmax}")

    robust = cfg_in.get("mcsrobust_filebase", "mcs_tracks_robust_")
    for version in ("old", "new"):
        root = work / version
        (root / "stats").mkdir(parents=True, exist_ok=True)
        for name in (f"trackstats_sparse_{dates}.nc", f"trackstats_{dates}.nc", f"tracknumbers_{dates}.nc"):
            if (src / "stats" / name).is_file():
                shutil.copy2(src / "stats" / name, root / "stats" / name)
        n_link, n_copy = link_tree(src / "tracking", root / "tracking")
        cfg = dict(cfg_in)
        cfg.update({
            "root_path": f"{root}/",
            "run_idfeature": False, "run_tracksingle": False, "run_gettracks": False, "run_trackstats": False,
            "run_identifymcs": True, "run_mapfeature": True, "run_speed": False,
            "run_parallel": 1, "nprocesses": args.nprocesses,
            "duration_range": [cfg_in.get("duration_range", [2, 10])[0], dmax],
            # the old code reads finalstats_filebase, the current code mcsrobust_filebase: same file name in both roots
            "finalstats_filebase": robust, "mcsrobust_filebase": robust,
        })
        with open(root / "config.yml", "w") as fh:
            yaml.dump(cfg, fh)
        print(f"{version}: {root}  stats copied, tracking: {n_link} hard-linked, {n_copy} copied; config.yml written")

    print("\nNext (on a compute node):")
    print(f"  python {Path(__file__).parent}/run_tree.py {work}/code_old {work}/old/config.yml")
    print(f"  python {Path(__file__).parent}/run_tree.py {work}/code_new {work}/new/config.yml")


if __name__ == "__main__":
    main()
