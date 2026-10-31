# Stitching the track numbers of consecutive periods

A long tracking job can be split into periods (for example months) that are tracked at the same time, each in its own job, and the
track numbers of the periods can be joined afterwards. The joined file is **identical** to the one that a single `gettracknumbers`
run over all files would make: a track or MCS that crosses the boundary of two periods is one track with one number, and no step
needs to be run again because of the joining.

`pyflextrkr/stitch_tracks.py` does the joining, `runscripts/stitch_tracknumbers.py` is the command line.

## Why the joining is exact

`gettracknumbers` carries two things from one file to the next: the track number of every cloud of the previous file, and the
counter of the next unused number. Every decision it takes (continuation, merger, split, the largest cloud keeps the number) uses
the links of `tracksingle` and the sizes of the clouds, never the values of the numbers; the numbers are only copied or counted up.
A period that is tracked with its own numbers 1, 2, 3, ... can therefore be renumbered afterwards, and the result is the one of a run
that had carried the numbers over.

## Periods and seams

Neighbouring periods **share exactly one cloudid file**, the seam: the last file of period A is the first file of period B. The pair
of files that ends in the seam belongs to A, the pair that starts in it to B, so every pair of files is tracked once. The monthly
configs of `run_mcs_radar3d.py` give this when the end date of a month is the start date of the next one, because both dates include
the file with exactly that time. How a boundary is joined depends on what is between the last file of A and the first of B:

| boundary | what is between the periods | what the joining does |
|---|---|---|
| seam | the last cloudid file of A is the first of B | maps the numbers of B to those of A through the clouds of the seam file and rebuilds the seam column from both sides |
| gap | no shared file, further apart than timegap | B starts new tracks after the last number of A, as gettracknumbers does after a data gap |
| restart | no shared file, closer than timegap (no pair links them) | the same as a gap, with a warning, counted in n_chain_breaks (as for a missing track file) |
| error | overlap of more than the seam file, or seam files that differ, or a different number of clouds in the seam file | stops with a message |

Two months that were tracked without a shared file have no pair between the last file of one and the first of the next. The pair
can be made with the same function that tracksingle uses (`conus_tracking/scripts/make_bridge_pairs.py`) and a period of its own
that holds just that pair joins the two months with a seam on each side.

## Use

```bash
python runscripts/stitch_tracknumbers.py OUT/tracknumbers_A_B.nc P1/tracknumbers_*.nc P2/tracknumbers_*.nc \
    --tracking-dirs P1/tracking P2/tracking
```

The periods are given in time order. With `--tracking-dirs` (one folder with the cloudid files per period) the seam file is read in
both folders and must be identical (feature sizes and feature numbers). In Python: `stitch_tracks.stitch_tracknumbers(paths, outfile)`.

The steps after gettracks (`trackstats`, `identifymcs`, `regionprops`, `mapfeature`) open the cloudid files by the names stored in
the tracknumbers file, relative to one tracking folder. For periods that kept their files in separate folders, make one folder of
links (`conus_tracking/scripts/link_period_folders.py`) and run those steps over the stitched range: the whole-track logic (the
duration criterion of the MCS, the folding of merge and split fragments, the lifecycle stages) then sees the complete tracks.

## How the numbers are joined

1. The local numbers 1..n0 of the clouds in the first column of period B (the seam file) are the clouds of the last column of A, so
   they get the global numbers of A's last column. Every other local number j gets `G + (j - n0 - 1)`, where G is the first unused
   global number after A.
2. The seam column is built from both sides: `track_status` is the sum of the two columns (the codes of the pair before the file and of
   the pair after it never overlap), the merge numbers come from B and the split numbers from A, and `track_reset` is 0 where the
   track starts in the file and the fill value elsewhere, as in an unbroken run. The end flag of A and the start flag of B are dropped.
3. A period with a larger `maxnclouds` than another is padded; the start and end flags of whole columns cover the padded slots.
4. The output is written by the same function as the one of gettracknumbers (`write_tracknumbers`), with the dates of the first start
   and the last end and the number of chain breaks.

## Checks

| test | data | result |
|---|---|---|
| `tests/test_stitch_tracks.py` | random clouds with continuations, mergers, splits, combined mergers and splits, dissipation, gaps | every single cut, every pair of cuts, periods of one pair, a gap, a missing pair and different `maxnclouds` give the file of the continuous run, variable by variable |
| `tests/test_stitch_idealized.py` | the four idealized demo scenarios | every single cut and 25 pairs of cuts per scenario are identical, as is the restart at a missing pair |
| `tests/test_gettracks_guard.py` | random clouds | the refactored gettracknumbers equals the frozen original (`tests/gettracks_reference.py`) on consistent input |

With links that are consistent in both directions (as in the real tracking of CONUS cases) all variables are bit-identical. With
random links that disagree between the forward and the backward direction, `gettracknumbers` sometimes reaches a cloud from two
groups of linked clouds and leaves a stale start flag in `track_reset`; the end flag that closes a period hides it, so a stitched
file cannot reproduce it. All other variables are still identical. None of the real tracking checked has this (the CONUS cases c4 and c7:
0 of 98,615 and 0 of 23,393 start flags).

## The chain check in gettracknumbers

`gettracknumbers` assumes that the reference file of every track file is the new file of the previous track file. If a track file is
missing and the two neighbours are closer than `timegap`, the test on the time difference does not see it and the track numbers of the
previous file used to be carried to the clouds of another file. Now the tracks restart there, as at a data gap, a warning is logged,
`n_chain_breaks` is stored in the file, and the run stops if more than 5 % of the track files are missing.
