"""
Make one folder of symbolic links to the cloudid / track files of several folders (metadata only, nothing is copied or read).

The steps after gettracks (trackstats, matchpf, mapfeature, ...) open every cloudid file by the file name stored in the tracknumbers file,
relative to ONE tracking folder. The jobs of a campaign track every month in its own folder (tracking/2009-01, tracking/2009-02, ...), so a
run over several months needs a folder that holds all of them: this script builds it from links (generate_tbpf_mcstracking_scripts.py puts
it in front of the stitched stats job). A file name that exists in more than one source (the seam file of two periods: both periods wrote it)
is linked once, from the first source; if the two copies differ in size the script stops (the seam file must be identical).

    python runscripts/link_period_folders.py OUT_DIR SRC_DIR [SRC_DIR ...] [--patterns 'cloudid_*.nc' 'track_*.nc']

Only directory listings and os.symlink are used (no stat of the targets except the size of a duplicate), so it works while a file system
has a hung target.

Author: Zhe Feng | zhe.feng@pnnl.gov
"""
import argparse
import fnmatch
import os
import sys


def link_folders(out_dir, sources, patterns=('cloudid_*.nc', 'track_*.nc'), check_duplicates=True):
    """
    Link the files of several folders into one.

    Args:
        out_dir: string
            Folder to fill (created). Existing links with the right target are left alone.
        sources: list of strings
            Folders with the files, in time order; the first one wins for a name that is in several.
        patterns: tuple of strings (default: ('cloudid_*.nc', 'track_*.nc'))
            File name patterns to link.
        check_duplicates: bool (default: True)
            Compare the sizes of the copies of a file name that is in more than one source.

    Returns:
        counts: dictionary
            {'linked': new links, 'existing': links that were already there, 'duplicates': names found in more than one source}.

    Raises:
        SystemExit: the copies of a file name in two sources differ in size.
    """
    os.makedirs(out_dir, exist_ok=True)
    first_target = {}
    counts = {'linked': 0, 'existing': 0, 'duplicates': 0}
    # Loop over the source folders in time order, and over their entries that match a pattern
    for source in sources:
        with os.scandir(source) as entries:
            for entry in sorted(entries, key=lambda e: e.name):
                if not any(fnmatch.fnmatch(entry.name, p) for p in patterns):
                    continue
                target = os.path.join(os.path.abspath(source), entry.name)
                # A name that an earlier source has: the earlier one stays, the copies must have the same size
                if entry.name in first_target:
                    counts['duplicates'] += 1
                    if check_duplicates and os.path.getsize(target) != os.path.getsize(first_target[entry.name]):
                        raise SystemExit(f"ERROR: {entry.name} differs in size between {first_target[entry.name]} and {target}")
                    continue
                first_target[entry.name] = target
                # Make the link, or leave it if it is there with the right target
                link = os.path.join(out_dir, entry.name)
                if os.path.islink(link) and os.readlink(link) == target:
                    counts['existing'] += 1
                    continue
                if os.path.lexists(link):
                    os.remove(link)
                os.symlink(target, link)
                counts['linked'] += 1
    return counts


def main():
    """
    Command line: link the files of the source folders into the output folder.

    Returns:
        status: int
            0.
    """
    parser = argparse.ArgumentParser(description='Link the files of several folders into one folder.')
    parser.add_argument('out_dir', help='folder to fill with links')
    parser.add_argument('sources', nargs='+', help='folders with the files, in time order')
    parser.add_argument('--patterns', nargs='+', default=['cloudid_*.nc', 'track_*.nc'], help='file name patterns (default: %(default)s)')
    parser.add_argument('--no-size-check', action='store_true', help='do not compare the sizes of duplicate names')
    args = parser.parse_args()
    counts = link_folders(args.out_dir, args.sources, tuple(args.patterns), not args.no_size_check)
    print(f"{counts['linked']} links made, {counts['existing']} already there, {counts['duplicates']} duplicate names skipped")
    return 0


if __name__ == '__main__':
    sys.exit(main())
