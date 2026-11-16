# Performance of the Tb+PF MCS tracking workflow: a year of global tracking

What a year of global Tb + precipitation tracking costs on Perlmutter CPU nodes (128 cores, 512 GB), measured on two data sets that were each run as a campaign of
[tbpf_mcs_workflow.md](tbpf_mcs_workflow.md): one track job per month (idfeature, tracksingle, gettracks, one node each, all months at once), then one stats job on 4 nodes that
stitches the months and runs trackstats, identifymcs, matchpf, robustmcs, mapfeature and speed, and a post-processing job on 4 nodes for the monthly rain maps and the precipitation accumulated along the tracks (trackpcp), then the dense netCDF file of the final statistics (step `dense_netcdf`, one process on one node). The numbers
are measured (from `sacct`, the step ledgers and the job logs); a number that is a projection says so. **Times are compute times** (queue waits are not counted and are listed apart). Run on 2026-11-14/15 with the code of the branch `perf-workflow`
at commit 088ba06; the post-processing jobs, which the stats job did not finish (IMERG) or ran with the old rain-map step (E3SM), with commit 1e7d2c4, and the dense netCDF step, which did not exist then, with the code of commit a4b8ad8 ([Method](#method)).

## Summary

| question | answer |
|---|---|
| what does a year cost? | **E3SM ne120 2009: 4.6 node-hours of work** (1.7 in 12 track jobs of one node, 2.1 in the stats job up to speed on 4 nodes, 0.8 in the post-processing job on 4 nodes and the dense file on one). **IMERG V07B 2019: 7.0 node-hours of work** (2.9, 2.5 and 1.6 the same way), and 12.3 node-hours lost in the incident below |
| how long does it take? | **Compute time**, queue waits not counted: E3SM 0.95 h = the longest of the 12 track jobs (12 min, side by side) + the stats job up to speed (31 min on 4 nodes) + the post-processing job (13 min on 4 nodes) + the dense netCDF file (0.4 min); IMERG 1.36 h = 17 + 38 + 24 + 2.1 min. The queue waits differ from day to day (the stats job 56 min for E3SM and 53 min for IMERG) and are listed apart |
| what does the dense netCDF file cost? | `mcs_tracks_final_<dates>.nc`, the file that most users expect, is written from the row tables by a serial step: **0.4 min (E3SM, 0.10 GB) and 2.1 min (IMERG, 0.43 GB)** on one core with 4.3 and 6.6 GB of memory; 0.01 and 0.04 node-hours on one node, 0.03 and 0.14 inside a 4-node stats job (the other nodes wait). The time is the compression of the 26 GB of mostly empty arrays, not the reading of the rows ([the step](#the-dense-netcdf-file-of-the-final-statistics)) |
| where does it go? | by node-hours, E3SM: idfeature 30 %, trackstats 16 %, matchpf 15 %, trackpcp 15 %, mapfeature 7 %; IMERG: idfeature 37 %, trackpcp 20 %, trackstats 16 %, matchpf 8 %, mapfeature 6 % ([E3SM](#e3sm-ne120-2009), [IMERG](#imerg-v07b-2019)) |
| what was projected? | from 3 days on one node (a month is 10 times, a year 122 times): the track job within 3 % for E3SM and 16 % for IMERG; the stats job on 4 nodes ran 2.5 times faster than the projected one node (E3SM) |
| can the stats job ask for less than 4 h? | Yes: **2 h** (`--stats-time 02:00:00`) is enough on 4 nodes: the stats steps took 0.75 h (E3SM) and 1.07 h (IMERG) with the fixed rain maps and the dense file ([settings](#what-follows-for-the-settings)) |
| what went wrong? | **The rain-map step hung** in the IMERG campaign, twice (4 nodes, then 2 nodes): a netCDF lock was left locked by xarray's garbage collection of a file on a Dask worker (`CombinedLock.acquire(blocking=False)` is not all-or-nothing), and the next read of that worker waited for ever. The step now reads the files inside the tasks: identical files (47 of 47), 13-15 s per month on one node and 8-10 s on 4 nodes, instead of 46-66 s, no hang in 72 months, including the two 4-node jobs ([the rain-map step](#the-rain-map-step-before-and-after-the-fix)) |
| is every campaign output checked? | Yes: every step ran its output check (the frames, the tracks, the matched PF fraction, the pixel files, the rain maps): the checks of all steps cost 0.16 s per track job and 0.3 s in the E3SM stats job |
| how big is a year? | 211 GB (E3SM) and 251 GB (IMERG) on `$SCRATCH` |
| what to request? | `--track-time 01:00:00` (default), `--stats-nodes 4 --stats-time 02:00:00` ([settings](#what-follows-for-the-settings)) |

## Method

**The two campaigns.** Both use the same settings, so that only the data differ: the MCSMIP definition of the robust MCS, the thresholds of `config_mcs_tbpf_imergv7_mcsmip.yml`, Tb at the full hour,
`stats_format: rows`, the monthly rain maps and the accumulation along the tracks switched on, the output checks on.

| | E3SM ne120, 2009 | IMERG V07B, 2019 |
|---|---|---|
| input | `ne120_olr_pcp_YYYYMMDDHH.nc`, 8,760 hourly files of 5.4 MB (FLUT, PRECT), 1200 x 3600 | `merg_YYYYMMDDHH_10km-pixel.nc`, 8,760 hourly files of 24 MB (Tb and precipitation at :00 and :30), 1200 x 3600 |
| config template | [config_mcs_tbpf_e3sm_0p1deg_template.yml](../config/config_mcs_tbpf_e3sm_0p1deg_template.yml) | [config_mcs_tbpf_imergv7_global_template.yml](../config/config_mcs_tbpf_imergv7_global_template.yml) |
| track jobs | 12, 1 node, 128 Dask workers (a LocalCluster), 1 h asked | 12, 1 node, 128 workers, 2 h asked |
| stats job | 1, 4 nodes, a Dask scheduler and 64 workers per node, 4 h asked | 1, 4 nodes, 4 h asked (cancelled in the rain maps: [IMERG](#imerg-v07b-2019)) |
| post-processing job | 1, 4 nodes, 2 h asked: rain maps and trackpcp with the fixed rain-map step | 1, 4 nodes, 2 h asked |
| dense netCDF file | 1 run on my interactive node (one Python process, 4000 tracks at a time): the `dense_netcdf` step of the unified script on the final row tables | the same |

The two templates differ only in the data keys (a test pins that). The campaigns were made and submitted with
[generate_tbpf_mcstracking_scripts.py](../config/generate_tbpf_mcstracking_scripts.py) (`--stats-nodes 4 --trackpcp on --profile-dir ... --pin-code`).

**What is measured, and what is not.**

- `sacct`: the start, end, elapsed time and nodes of every job, and `MaxRSS` of the batch step (the main process and, in a track job, the 128 workers) and of the worker step of the stats job.
- The step ledger of every job (`step_profile_dir`, one folder per job: the wall time of every step, the effective cores, the memory of the main process and of the workers of a LocalCluster).
  **The ledger does not see the workers of the 4-node stats job**: they run on other nodes, so its cores and memory columns are those of the main process. The memory of the workers of the stats job
  is the one of `sacct`.
- The job logs: the time between `Step X finished` and `Step X outputs verified` (the output check of the step), the end of the stitching, the counts of the checks.
- [collect_campaign_performance.py](../Analysis/collect_campaign_performance.py) reads these into one JSON file per campaign, [make_performance_tbpf_figures.py](../Analysis/make_performance_tbpf_figures.py)
  draws the figures from it. **The wall-clock times are compute times**: the run times of the jobs; the longest of the 12 track jobs, then the stats job, then the post-processing job. **Queue waits are not counted** (they differ from day to day, and the 12 track jobs of a campaign start whenever nodes are free); they are listed apart. A step that was run again (the post steps of the stats job, the 1-node run) is drawn from its last run; the earlier runs stay in the node-hours charged.
  A run by hand on an interactive node (the last two steps of the IMERG campaign, the dense netCDF file of both) is a line of `submissions.jsonl` with its own start, end and nodes: `sacct` knows only the allocation, which lives longer than the run.
- **The code.** The campaigns ran from a separate `git worktree` (`--pin-code` puts it in `PYTHONPATH` of every job): the conda environment has an editable install of the working copy, so queued
  jobs would otherwise import whatever the working copy holds when they start. All track jobs and stats steps (trackstats to speed) of both campaigns, and the E3SM rain maps and trackpcp of the campaign, ran with commit 088ba06;
  the post-processing jobs of both campaigns (rain maps and trackpcp, 4 nodes), the IMERG run of these steps on one node and the E3SM repeat, with 1e7d2c4 (the rain-map step reads the files inside the tasks, [see below](#the-rain-map-step-before-and-after-the-fix));
  the `dense_netcdf` step (it did not exist when the campaigns ran) with the code of commit a4b8ad8 (the commit of the step), on the finished row tables of both campaigns.
- The 12 IMERG track jobs started at once at 15:03, the time the last eight E3SM jobs also started: the E3SM months that ran alone (1 to 4) and those that shared the file system with the IMERG jobs (5 to 12) took the same time.

## E3SM ne120, 2009

### The campaign

![The E3SM 2009 campaign by step](https://github.com/FlexTRKR/PyFLEXTRKR/releases/download/docs-assets/perf_tbpf_e3sm_2009_campaign.png)

*Figure 1. The E3SM campaign: the 12 track jobs, the stats job up to speed and, below the dashed line, the post-processing steps (the dense netCDF file of the final statistics, on one node; rain maps and precipitation accumulated along the tracks), which were run again in a job of their own on 4 nodes with the fixed rain-map step. The bars are node-hours (axis at the top), the axes at the bottom
the run time (wall-clock) of each row; the numbers in the segments are the wall-clock minutes of the step and its node-hours; the braces give the compute time and the node-hours of the tracking (rows 1 and 2) and of the post-processing.*

The year took **4.6 node-hours of work** (1.7 in the 12 track jobs, 2.1 in the stats job up to speed, 0.8 in the post-processing job and the dense file) and **0.95 h of compute time**:
the longest of the 12 track jobs (11.8 min; the 12 run side by side), the stats job on 4 nodes (31.3 min up to speed), the post-processing job on 4 nodes (13.2 min) and the dense netCDF file (0.43 min on one node).
**Queue waits are not counted**, they differ from day to day: the stats job waited 56 min for its four nodes, the post-processing job 18 min, and the track jobs started in three groups over 17 min.
The campaign was charged 6.3 node-hours in all, 1.3 of them for the first run of the two post steps in the stats job (with the old rain-map code), which the re-run replaces. The dense file is a run of its own (0.007 node-hours): the campaign's stats job ran before the step existed.
The jobs: the track jobs took 8.9 - 11.8 min (mean 10.1), the stats job 51.4 min on 4 nodes with all its steps, the post-processing job 13.2 min. All 14 jobs completed at the first attempt, every step passed its output check.

The tracking found 8,760 frames (the file of 01 UTC on 1 January has no OLR: 1 frame is skipped, as in the 3-day test), 1,271,383 tracks, 89,464 Tb MCS and
7,608 robust MCS (MCSMIP); the PF statistics matched all 1,526,703 MCS entries. The files: 103 GB of cloudid and track files, 106 GB of pixel-level
files and 1.7 GB of statistics (the row tables of the robust MCS are 55 MB), rain maps (189 MB) and the accumulation along the tracks (102.9 million (MCS, pixel) rows in a 0.86 GB Parquet file): 211 GB.

#### The track jobs

| | mean ± SD | min - max |
|---|---|---|
| idfeature | 6.93 ± 0.64 min | 6.02 - 8.16 |
| tracksingle | 1.17 ± 0.13 min | 1.07 - 1.54 |
| gettracks | 0.54 ± 0.04 min | 0.49 - 0.60 |
| setup (module load, the Dask workers start) | 1.07 ± 0.23 min | 0.78 - 1.70 |
| the job (elapsed time) | 10.06 ± 0.82 min | 8.95 - 11.78 |

The months that ran alone (1 to 4: 8.9 - 11.8 min, mean 10.4) and the months that ran at the same time as the 12 IMERG track jobs (5 to 12: 9.2 - 10.9, mean 9.9) took
the same time: no slowdown from the shared file system is visible. About 1.4 min of a job (13 %) is not a step: the start of the Python environment and of the 128 workers before idfeature and the shutdown after gettracks.

#### The stats job

| step | wall-clock | node-hours | share | output check | projected, one node |
|---|---|---|---|---|---|
| setup + stitch | 1.5 min | 0.10 | 3 % | | |
| trackstats | 10.8 min | 0.72 | 21 % | 0.01 s | 23.1 min |
| identifymcs | 1.8 min | 0.12 | 4 % | 0.11 s | 2.0 min |
| matchpf | 10.2 min | 0.68 | 20 % | 0.03 s | 34.7 min |
| robustmcs | 0.1 min | 0.00 | 0 % | 0.00 s | 1.2 min |
| mapfeature | 4.7 min | 0.31 | 9 % | 0.10 s | 24.1 min |
| speed | 1.8 min | 0.12 | 3 % | 0.01 s | 8.9 min |
| rainmap | 9.8 min | 0.65 | 19 % | 0.07 s | 20.7 min |
| trackpcp | 10.3 min | 0.69 | 20 % | 0.00 s | 12.0 min |
| between the steps, shutdown | 0.5 min | 0.04 | 1 % | | |
| **the job** | **51.4 min** | **3.43** | | | **2.11 h** |

*The last column is the projection of the 3-day test run (one node, 128 workers, 72 frames: every step scaled by 8,760 / 72), made before the campaign. The rain maps and trackpcp rows are the first run of these steps, with the code of the first commit (088ba06);
the figures show their re-run ([below](#the-two-last-steps-run-again)).*

The three steps of the stats job that read every frame (trackstats, matchpf, mapfeature) and the two post-processing steps that do (rainmap, trackpcp) are 91 % of its time; identifymcs and robustmcs, the serial steps, are 4 %. The step times
add to 50.9 of the 51.4 min of the job; the rest (0.5 min) is the time between the steps and the shutdown. The setup is the stitching of the 12 months
(38 s from the start of the job, 12 months, 8,760 cloudid and 8,748 track files linked) and the start of the Dask scheduler and the 256 workers (53 s).

#### Projected and measured

| | projected from the 3 days | measured |
|---|---|---|
| a track job (one month, 1 node) | 10.3 min | 10.1 min (8.9 - 11.8), -3 % |
| the stats job with all its steps | 2.11 h on one node | 51.4 min on 4 nodes (3.4 node-hours: 1.6 times the node-hours of the projected one-node job) |
| tracks in the year | about 1.2 million (the size used for the row-table benchmark) | 1,271,383 |
| output | about 200 GB | 211 GB |

The projection of the track job is within 3 %. Four nodes made the stats job 2.5 times faster than the projected one node at 1.6 times the node-hours: the steps with
a task per frame (mapfeature, speed, matchpf) divide over 256 workers (the 3-day test had only 72 tasks for 128 workers, so its steps were limited by their longest task, and the projection overstates them), the steps with
few tasks or a serial part (trackstats, trackpcp, identifymcs) gain little from the further nodes.

#### The two last steps run again

The rain maps of the campaign's stats job were made with the code of the first commit (088ba06), which opened the files in the main process (the step needed 9.8 min; [what was wrong](#the-rain-map-step-before-and-after-the-fix)).
After the fix the two post steps were run again, with the config of the stats job and only these steps on, in a job of their own (4 nodes, 256 workers, 2 h asked, commit 1e7d2c4; the files went to a folder of their own),
and once more on one interactive node (64 workers) as a like-for-like partner of the IMERG run:

| step | campaign: stats job, 4 nodes, 088ba06 | re-run: own job, 4 nodes, 1e7d2c4 | repeat: 1 node, 64 workers, 1e7d2c4 |
|---|---|---|---|
| rainmap | 9.8 min | 2.1 min (8 - 10 s a month, the first month 27 s with cold files) | 2.8 min |
| trackpcp | 10.3 min | 10.0 min | 9.7 min (13 GB in the main process) |
| the job | | 13.2 min, 0.88 node-hours | |
| the files against the campaign's | | rain maps identical bit by bit (12 of 12); trackpcp: the same 102,903,541 rows, keys, `n_windows`, `max_1h_mm`; 7,476,867 accumulated depths differ by at most 3.0e-07 relative; the threshold-based columns (areas above 25, 50 and 100 mm, the cluster area) are identical | rain maps identical (12 of 12); trackpcp: the same rows and keys, differences at most 3.3e-7 relative |

trackpcp gains nothing from the three further nodes (10.0 min on 4 nodes against 9.7 min on one): the main process, which merges the records of the windows, is its bottleneck (while the IMERG step ran, the workers were mostly idle and the main process used 70 % of a core).
Its sums are float32 and the order of the additions follows the order in which the windows finish, so a repeat is equal to the float32 rounding and not bit by bit; the rain maps, whose graph fixes the order of the sums, are identical.

### The campaign in compute time, and the cost of each step

![The E3SM 2009 campaign in compute time](https://github.com/FlexTRKR/PyFLEXTRKR/releases/download/docs-assets/perf_tbpf_e3sm_2009_timeline.png)

*Figure 2. The same campaign in compute time: the 12 track jobs from 0 (their own length; gray = setup and shutdown), the stats job from the end of the longest of them, then the two post steps of the re-run job and the dense netCDF file (violet, the last segment: it ran afterwards on one node). Queue waits are not on the axis: the stats job waited 56 min for its 4 nodes,
the post-processing job 18 min, and the track jobs started in three groups over 17 min.*

![Node-hours of each step of the E3SM 2009 campaign](https://github.com/FlexTRKR/PyFLEXTRKR/releases/download/docs-assets/perf_tbpf_e3sm_2009_steps.png)

*Figure 3. Left: node-hours of each step, largest first (the steps up to speed of the stats job, the re-run of the two post steps and the dense netCDF file, 4.6 node-hours). Right: the three steps of the track jobs in the 12 months (dots)
with their mean ± SD: the time of a job varies by about 8 % from month to month (SD 0.8 min of 10.1).*

## IMERG V07B, 2019

### The campaign

![The IMERG 2019 campaign by step](https://github.com/FlexTRKR/PyFLEXTRKR/releases/download/docs-assets/perf_tbpf_imerg_2019_campaign.png)

*Figure 4. The IMERG campaign. Rows 1 and 2: the 12 track jobs and the stats job up to speed. Row 3 (below the dashed line): the rain maps and the precipitation accumulated along the tracks, which ran in a job of their own on 4 nodes after the stats job was cancelled in the rain maps, and the dense netCDF file of the final statistics (on one node afterwards)
(the hung time, the debug job and the earlier 1-node run are not drawn: they are charged as failed or superseded). The numbers in the segments are the wall-clock minutes of the step and its node-hours.*

**This campaign did not run through as planned.** The 12 track jobs ran side by side, all started at once, and took 14.5 - 16.9 min (mean 15.9); the stats job waited 53 min for its four nodes, ran
trackstats to speed in 38.0 min (including 1.7 min of setup) and then **hung in the rain-map step for 2.83 h until it was cancelled** (job 61298650, 13.8 node-hours charged,
2.5 of them for the steps that finished). A resumed stats job (skip of the finished steps) was cancelled while it waited in the queue, and the rain maps were run as a 2-node job in the debug QOS (30 min limit,
job 61320439): it reproduced the hang, which gave the diagnosis ([the rain-map step](#the-rain-map-step-before-and-after-the-fix)), and timed out after 30 min (1.0 node-hours).
With the step fixed (commit 1e7d2c4) the two last steps ran first on one interactive node of mine (23.1 min, 0.38 node-hours) and then in the production layout, **a job of its own on 4 nodes with a 2 h limit**
(job 61331895: rain maps 2.2 min, trackpcp 20.8 min, the job 24.1 min, 1.61 node-hours). The figures show the 4-node run; the 1-node run is superseded.

The year took **7.0 node-hours of work** (2.9 in the track jobs, 2.5 in the stats steps up to speed, 1.6 in the post-processing job and the dense file) and **1.36 h of compute time**: the longest track job (16.9 min),
the stats job up to speed (38.0 min), the post-processing job (24.1 min) and the dense netCDF file (2.1 min on one node). Queue waits are not counted: the stats job waited 53 min for its four nodes, the post-processing job 18 min.
Charged in all: 20.0 node-hours, of them 12.3 lost (the hung stats job 11.3, the debug job 1.0) and 0.38 for the superseded 1-node run.

The tracking found 8,755 frames (6 of the 8,761 input files, the 8,760 hours of the year and the first hour of the next, were skipped for too much missing data), 2,237,492 tracks, 62,305 Tb MCS and 41,501 robust MCS (MCSMIP); the PF
statistics matched all 999,751 MCS entries. The files: 121 GB of cloudid and track files, 126 GB of pixel-level files and 3.7 GB of statistics, rain maps
and 243 million (MCS, pixel) rows of the accumulation along the tracks (2.4 GB Parquet): 251 GB.

#### The track jobs

| | mean ± SD | min - max |
|---|---|---|
| idfeature | 12.72 ± 0.82 min | 10.90 - 13.92 |
| tracksingle | 1.14 ± 0.05 min | 1.08 - 1.21 |
| gettracks | 0.60 ± 0.04 min | 0.53 - 0.67 |
| setup (module load, the Dask workers start) | 1.07 ± 0.39 min | 0.77 - 2.27 |
| the job (elapsed time) | 15.86 ± 0.67 min | 14.53 - 16.92 |

#### The stats job up to speed

| step | wall-clock | node-hours | share | output check | projected, one node |
|---|---|---|---|---|---|
| setup + stitch | 1.7 min | 0.11 | 4 % | | |
| trackstats | 16.3 min | 1.08 | 43 % | 0.01 s | 31.0 min |
| identifymcs | 1.3 min | 0.09 | 3 % | 0.15 s | 1.8 min |
| matchpf | 8.8 min | 0.59 | 23 % | 0.02 s | 34.4 min |
| robustmcs | 0.1 min | 0.01 | 0 % | 0.01 s | 2.0 min |
| mapfeature | 5.9 min | 0.39 | 15 % | 0.10 s | 30.4 min |
| speed | 4.1 min | 0.27 | 11 % | 0.02 s | 19.0 min |
| (the steps above) | 38.0 min | 2.54 | 100 % | | 118.6 min |
| hung in the rain maps until the job was cancelled | 2.83 h | 11.31 | | | |
| **the job** | **3.46 h** | **13.84** | | | |

*The last column is the projection of the 3-day test run (one node, 128 workers, 73 frames: every step scaled by 8,760 / 73), made before the campaign; the stats job on 4 nodes was 3.1 times faster than the projected one node.*

#### The rain maps and the precipitation accumulation: 4 nodes and one node

| step | 4 nodes (own job, 256 workers): wall-clock | node-hours | output check | 1 node (interactive, 64 workers) | projected, one node (3 days) |
|---|---|---|---|---|---|
| rainmap | 2.2 min | 0.15 | 0.07 s | 3.0 min | 22.0 min |
| trackpcp | 20.8 min | 1.38 | 0.00 s | 20.1 min | 15.0 min |
| the job (setup, steps, shutdown) | 24.1 min | 1.61 | | 23.4 min | |

*Both with the code of commit 1e7d2c4 (the rain maps read the files inside the tasks; on 4 nodes 9 - 10 s a month, the first month 29 s with cold files). The projection of the rain maps is that of the 3-day run with the old reading (088ba06).
The files of the 4-node job against those of the 1-node run: rain maps identical bit by bit (12 of 12); trackpcp the same 243,051,250 rows and keys, 13,100,947 accumulated depths differ by at most 2.7e-07 relative (float32 sums in another order).*

**trackpcp is reproducible to float32 rounding, not bit by bit, and its threshold-based columns can change.** The accumulated depth of a pixel is a float32 sum over the hours of the MCS, and the order of the additions follows the order in which the windows finish, which differs from run to run.
The depths and the metrics that are sums, maxima or means differ in the last digit only (at most 2.7e-07 relative). The columns that count pixels above a threshold (`area_ge25mm_km2`, `area_ge50mm_km2`, `area_ge100mm_km2`) and the connected region above 50 mm (`cluster_area_km2`, `cluster_vol_1e6m3`)
can change when a pixel is at the threshold. Many IMERG accumulations are exactly 25.0, 50.0 or 100.0 mm (in the 1-node run 1,129 / 288 / 48 pixels of the MCSs concerned: the IMERG rain rates are coarse values), and a float32 step below the threshold in the other run puts the pixel on the other side. Between the 1-node run and the 4-node run
833 / 242 / 36 of the 243 million pixels cross 25 / 50 / 100 mm differently; that changes the area above the threshold for
771 / 233 / 36 of the 41,501 MCSs (46 / 29 / 6 by more than 1 %)
and the cluster area for 107 (20 by more than 1 %). The E3SM runs (model rain rates) have no such change: 0 of 7,608 MCSs. Accumulating in float64, or in the order of the times, should make the step reproducible; that was not done here.

#### Projected and measured

| | projected from the 3 days | measured |
|---|---|---|
| a track job (one month, 1 node) | 18.9 min | 15.9 min (14.5 - 16.9), -16 % |
| the stats steps trackstats to speed | 118.6 min on one node | 38.0 min on 4 nodes (2.5 node-hours) |
| rain maps | 22.0 min on one node (old reading) | 2.2 min on 4 nodes, 3.0 min on one (fixed reading) |
| trackpcp | 15.0 min on one node | 20.8 min on 4 nodes, 20.1 min on one |
| tracks in the year | 2.2 million (122 x the 18,518 of the 3 days) | 2,237,492 |

### The campaign in compute time, and the cost of each step

![The IMERG 2019 campaign in compute time](https://github.com/FlexTRKR/PyFLEXTRKR/releases/download/docs-assets/perf_tbpf_imerg_2019_timeline.png)

*Figure 5. The campaign in compute time: the 12 track jobs from 0, the stats job from the end of the longest of them, then the two post steps of the 4-node job and the dense netCDF file (violet, the last segment, one node). Queue waits and the 2.8 h that the stats job was hung before it was cancelled are not on the axis
(the stats job waited 53 min for its 4 nodes, the post-processing job 18 min); the debug job and the 1-node run are not drawn.*

![Node-hours of each step of the IMERG 2019 campaign](https://github.com/FlexTRKR/PyFLEXTRKR/releases/download/docs-assets/perf_tbpf_imerg_2019_steps.png)

*Figure 6. Left: node-hours of each step, largest first (the steps up to speed of the stats job, the two post steps of the 4-node job and the dense netCDF file, 7.0 node-hours). Right: the three steps of the track jobs in the 12 months (dots) with their mean ± SD.*

## The dense netCDF file of the final statistics

The steps of the workflow write the statistics as row tables. Most users, and the old pipeline, work with the dense netCDF file `mcs_tracks_final_<dates>.nc`, so the unified script has a step for it: `dense_netcdf`, after `speed`, **on by default with `stats_format: rows`**
(`run_dense_netcdf`, [config keys](tbpf_mcs_workflow.md#config-keys-of-this-work)). It writes the file from the row tables of the final statistics, with the layout, names, dtypes, fill values and attributes of the old dense file, under a temporary name that is renamed when it is complete.
The campaigns ran before the step existed, so it was run afterwards on the finished row tables of both campaigns, as one serial Python process on my interactive node (job 61356886), once each:

| | E3SM ne120, 2009 | IMERG V07B, 2019 |
|---|---|---|
| row tables of the final statistics | 7,608 tracks, 251,611 entries, 390,551 fragments | 41,501 tracks, 785,380 entries, 1,260,565 fragments |
| the same as dense arrays (tracks x 400 times, 50 merge / split slots) | 4.8 GB in memory | 26.4 GB in memory |
| **wall time of the step** | **25.7 s (0.43 min)** | **128.2 s (2.14 min)** |
| cores in use (ledger) | 1.0 | 1.0 |
| peak memory of the process | 4.4 GB | 6.5 GB (4000 tracks at a time) |
| the dense file | 0.10 GB | 0.43 GB |
| node-hours: one node (as run) / inside a 4-node stats job | 0.007 / 0.03 | 0.036 / 0.14 |
| the file against the row tables | 70 variables, 0 differences | 70 variables, 0 differences |

*The comparison reads the file block by block and compares every variable (dimensions, dtype, values with NaN equal NaN, attribute names) with `load_trackstats` of the row tables, and checks that what follows the longest track of the block is fill. The step is serial: one core of the node works and, inside the 4-node stats job, the other
three nodes wait (the last column). Recorded in both campaign folders as the job `post_dense` (a line of `submissions.jsonl`, the ledger in `profile/post_dense`) and collected like the other jobs; the figures show it as the violet segment.*

**Where the time goes.** For the IMERG year: reading the three tables 0.8 s, building the dense arrays of the 11 blocks 3.8 s, **writing them 121 s**. The step is the compression and writing of 26 GB of arrays that are 95 % empty (the entries are 4.7 % of the tracks x 400 times cells), not the reading of the rows. Measured on one block of 4000 tracks (1.88 GB as arrays):

| netCDF compression of the block | time | size of the block |
|---|---|---|
| zlib level 4 (what the step uses, the level of the old file) | 12.8 s | 42 MB |
| zlib level 4 with the shuffle filter | 11.7 s | 42 MB |
| zlib level 1 | 6.3 s | 49 MB |
| zlib level 9 | 18.8 s | 42 MB |
| no compression | 5.5 s | 1,890 MB |

The block size does not change the time (the whole year: 127 s with blocks of 1,000 tracks, 126 s with 4,000, 125 s with 12,000) and sets the memory (peak 4.5, 7.6 and 18.7 GB). The step keeps the level of the old file; level 1 would halve the time for a 17 % larger file
and is not used because it changes the file that every reader of the old pipeline expects. A year of IMERG tracks takes 2 min, so there was no reason to parallelise it; with millions of tracks (the all-track files) the time grows with the arrays, see [the other statistics files](tbpf_mcs_workflow.md#dense-netcdf-files-of-the-other-statistics).

## The rain-map step: before and after the fix

**What happened.** The rain maps are the step after speed: the 12 monthly maps of the year, from the pixel-level files. In the E3SM campaign they took 9.8 min of the 4-node stats job. In the IMERG campaign the step **hung twice**:
in the stats job on 4 nodes (the file of January was created at 16:52, then nothing for 2.8 h: two of the 256 workers each had a task that did not move, all four nodes were idle; the job was cancelled at 19:40), and again in the
2-node debug job (January took 62 s, then February stalled; 19 minutes later the evidence below was collected). The E3SM step had run without a problem through its 12 months; IMERG hung in 2 of 2 attempts.

**Evidence** (the files are in `/pscratch/sd/f/feng045/tbpf_perf_results/imerg_stats_hang/`, `README.txt` lists them):

1. The Python stack of the stuck worker (`faulthandler` run on it with `client.run`): its only pool thread waits in `SerializableLock.__enter__`, called by `CombinedLock.__enter__` in `NetCDF4ArrayWrapper._getitem`: a read of `mcstrack_20190205_060000.nc` waits for a netCDF lock of xarray.
2. **Nobody holds that lock.** The worker has one pool thread, and it is the thread that waits. The lock is one of the two netCDF locks of the *main process*, as the worker received them with the lazily opened dataset (one locked, one free); the worker's own locks are free.
3. Releasing the locked lock by hand from the worker's event loop let the task go on at once: February finished after 1,165 s. The same was needed 5 more times in the next 8 minutes (a watcher released each one: 6 leaks in the 11 months that finished, in 5 of the months); the months took 46-66 s each.
4. A ring buffer of all lock calls of every worker caught one in the act: the worker's event loop thread (`MainThread`) called `acquire(blocking=False)` on the `CombinedLock` of a dataset: the first lock was taken (`True`), the second was busy (`False`, the pool thread
   held it), and the first one was never released.

**Cause.** xarray's `CachingFileManager.__del__` closes a file only if it can take the dataset's lock without waiting, `acquire(self._lock, blocking=False)`, and `CombinedLock.acquire(blocking=False)` is
`all(acquire(lock) for lock in self.locks)`: **it is not all-or-nothing**. If the first lock is free and a later one is busy it returns `False` and keeps the first one locked. The rain-map step opened the 744 files of a month lazily in the
main process (`xr.open_mfdataset`) and let the workers write the maps (`to_netcdf` of Dask arrays), so the workers held the netCDF locks of the main process, and a Dask worker frees the lazily opened datasets on its event loop thread, `__del__`
included, while its pool thread reads or writes. A write takes the two locks and a distributed lock (a round trip to the scheduler of 40-100 ms) and releases them in that order, so for that time the first lock is free and the second is held; if
a `__del__` runs then, the first lock stays locked for ever and the next read of that worker waits for it for ever. It leaked 6 times in the 11 months that finished (in 5 of the months). A second finding: only 5 of the 128 workers of the debug job had the main process's locks, that is
did all the reading of a month, so the step did not use the other workers (why the scheduler placed the work this way was not examined).

**The fix** (commit 1e7d2c4, `pyflextrkr/rainmap_driver.py`): every pixel-level file is read by a task of its own that opens and closes it (`read_pixel_file`), so the graph holds paths and arrays only and no xarray file handle or lock leaves a
task; the main process writes the computed maps. The expressions of the maps are unchanged. Tests: the graph holds no xarray backend object (the `open_mfdataset` graph as the control), a `LocalCluster` run equals the threaded run, files with another
grid or with several times are refused, the files still equal those of `Analysis/calc_tbpf_mcs_monthly_rainmap.py`.

**Verification on the real files** (one node, 64 workers, the code of 1e7d2c4):

| | before (088ba06) | after (1e7d2c4) |
|---|---|---|
| E3SM 2009, 12 monthly files | made by the campaign | **identical bit by bit** (12 of 12) |
| IMERG 2019, 11 monthly files (January to November of the debug job; December was cut by its time limit) | made with the hangs and 6 manual lock releases | **identical bit by bit** (11 of 11) |
| time of a month | 46-66 s (IMERG, 128 workers on 2 nodes), 49 s (E3SM, 256 workers on 4 nodes) | 13-15 s (first month 30 s: cold file cache) |
| the 12 months | E3SM 9.8 min; IMERG hung in 2 of 2 attempts | E3SM 2.8 min, IMERG 3.0 min (the step of the unified run, outputs verified) |
| the 12 months on 4 nodes (256 workers) | | E3SM 2.1 min (8 - 10 s a month, the first month 27 s with cold files), IMERG 2.2 min (9 - 10 s a month, the first month 29 s with cold files): the files equal the earlier ones bit by bit (E3SM 12 of 12, IMERG 12 of 12) |
| hangs | 2 of 2 IMERG attempts; 5 of the 11 months finished only because a lock was released by hand | none in 72 months (24 in the tests and 24 more on one node, 12 + 12 on 4 nodes in Slurm jobs) |
| locks on the workers after the run | 5 of 128 workers had the main process's locks | no lock locked, 2 locks on every worker (their own) |

**Open.** The 4-node jobs above are the first multi-node Slurm runs of the fixed step (no hang, no lock left locked). `Analysis/calc_tbpf_mcs_monthly_rainmap.py` still has the old way: it is safe with the threaded scheduler it
uses, not with a distributed client. The same pattern (a lazily opened dataset of the main process used by the workers of a distributed client, with a write by the workers) is in a few other places of the package (`idfeature_driver` for the native
multi-file reading, `convert_to_zarr`, `remap_healpix_zarr`); they were not examined here.

## E3SM and IMERG compared

Same settings, same grid (1200 x 3600), same number of files (8,760), so the differences are the data:

| | E3SM ne120, 2009 | IMERG V07B, 2019 | IMERG / E3SM |
|---|---|---|---|
| input files | 8,760 | 8,760 |  |
| frames skipped | 1 | 6 |  |
| tracks | 1,271,383 | 2,237,492 | 1.76 |
| Tb MCS | 89,464 | 62,305 | 0.70 |
| robust MCS (MCSMIP) | 7,608 | 41,501 | 5.45 |
| idfeature, one month job [min] | 6.93 | 12.72 | 1.84 |
| tracksingle, one month job [min] | 1.17 | 1.14 | 0.97 |
| gettracks, one month job [min] | 0.54 | 0.60 | 1.10 |
| a track job [min] | 10.06 | 15.86 | 1.58 |
| trackstats, stats job on 4 nodes [min] | 10.76 | 16.25 | 1.51 |
| identifymcs, stats job on 4 nodes [min] | 1.84 | 1.28 | 0.69 |
| matchpf, stats job on 4 nodes [min] | 10.15 | 8.78 | 0.86 |
| robustmcs, stats job on 4 nodes [min] | 0.06 | 0.09 | 1.40 |
| mapfeature, stats job on 4 nodes [min] | 4.71 | 5.89 | 1.25 |
| speed, stats job on 4 nodes [min] | 1.76 | 4.09 | 2.32 |
| rainmap, own job on 4 nodes [min] | 2.12 | 2.20 | 1.04 |
| trackpcp, own job on 4 nodes [min] | 10.04 | 20.77 | 2.07 |
| rainmap, 1 node [min] | 2.81 | 2.97 | 1.06 |
| trackpcp, 1 node [min] | 9.72 | 20.13 | 2.07 |
| dense netCDF file, 1 node, serial [min] | 0.43 | 2.14 | 5.0 |
| output: cloudid and track files [GB] | 102.82 | 120.97 | 1.18 |
| output: pixel-level files [GB] | 106.27 | 126.35 | 1.19 |
| output: statistics, rain maps, accumulation [GB] | 1.74 | 3.72 | 2.15 |
| output: dense netCDF file of the final statistics [GB] | 0.10 | 0.43 | 4.5 |

IMERG has 1.8 times the tracks and 5.5 times the robust MCS of the model year, but fewer Tb MCS (0.70 times).
idfeature, the largest step, takes 1.8 times as long per month (it did in the 3-day runs too: 2.0; the IMERG files are 4.5 times larger, 24 MB with Tb and precipitation at :00 and :30
against 5.4 MB), trackstats 1.5 times, speed 2.3 times; matchpf, mapfeature, identifymcs and the rain maps cost about the same, and the dense netCDF file takes 5 times as long (5.5 times the tracks, so the same time per track). The post steps are those of the 4-node re-run jobs (both data sets, the same layout, the fixed rain-map step) and of the
runs on one node. trackpcp grows with the (MCS, pixel) rows: 243 million for IMERG against 103 million for E3SM.

## Dask health

The messages of the logs of all jobs (the job logs, which hold the `LocalCluster` of a track job, and the logs of the scheduler and the workers of the stats jobs), counted by `collect_campaign_performance.py`:

| message | E3SM track jobs (12) | E3SM stats job | E3SM post job (4 nodes) | IMERG track jobs (12) | IMERG stats job (cancelled) | IMERG debug job (my diagnostics ran in it) | IMERG post job (4 nodes) |
|---|---|---|---|---|---|---|---|
| `Event loop was unresponsive` (INFO) | 0 | 472 | 494 | 0 | 368 | 198 | 486 |
| `Refreshing an unknown lease ID` (CRITICAL) | 1 | 2 | 0 | 13 | 0 | 0 | 0 |
| `Removing worker` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `KilledWorker` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| memory 95 % / `memory use is high` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `Couldn't gather` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| other `unresponsive` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `TimeoutError` / `timed out` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `CommClosedError` | 0 | 0 | 0 | 0 | 0 | 28 | 0 |
| traceback after `Dask client shutdown complete` (teardown noise) | 16 | 0 | 0 | 6 | 0 | 0 | 0 |

No worker was removed or killed, no worker reached its memory limit, no gather failed. The `Event loop was unresponsive` lines are tasks that hold the GIL for a few seconds (the event loop of the worker is then not answering); they are common in the stats
jobs. The `lease` lines name `cloudid_*.nc` files (idfeature) and `mcstrack_*.nc` files (mapfeature): each file is written under a distributed lock of its own (one lock per file, nobody waits for it) whose lease was not renewed in time because the worker's event loop was blocked: harmless. The
`CommClosedError` lines of the debug job are those of my diagnostics (I connected to the cluster as a client to read the stacks and to release locks).

**A hung step is silent.** The stuck rain maps wrote no warning at all, neither in the job log nor in the Dask logs; the only signs were that the log did not grow, the scheduler had two tasks in `processing` for hours and all nodes were idle (the CPU of the nodes
and `client.processing()` showed it). A watch of a long run should therefore also look at the age of the last log line, not only at the state of the jobs and the output checks.

## What follows for the settings

Measured for a year of 8,760 hourly files of 1200 x 3600 on Perlmutter CPU nodes (the settings are options of [generate_tbpf_mcstracking_scripts.py](../config/generate_tbpf_mcstracking_scripts.py) or keys of the template):

| setting | recommended | measured |
|---|---|---|
| `nprocesses` / workers per node | 128 on a one-node job, 64 per node of a multi-node stats job (the templates) | the largest worker of a track job needs 1.8 GB, the stats job's largest worker 3.2 GB (`sacct` MaxRSS of the worker step), the 7 GB limit was never reached |
| `--track-time` | `01:00:00` (the default) for both data sets | E3SM 8.9 - 11.8 min, IMERG 14.5 - 16.9 min: the default leaves a factor of 5.1 and 3.5 |
| `--stats-nodes` | 4 for a year | E3SM: 0.86 h on 4 nodes with all its steps (3.4 node-hours) against a projected 2.1 h on one node: 2.5 times faster for 1.6 times the node-hours; trackstats, identifymcs, robustmcs and trackpcp gain little from nodes (serial parts), mapfeature, speed and matchpf gain most. The queue made the stats job wait 56 min (E3SM) and 53 min (IMERG) for its 4 nodes: one node starts earlier |
| `--stats-time` | **`02:00:00`** on 4 nodes (the campaigns asked 4 h, the default `06:00:00` is for one node) | the stats work with the fixed rain maps and the dense file: E3SM 0.52 h up to speed + 0.22 h post-processing + 0.01 h dense file = **0.75 h**; IMERG 0.63 h + 0.40 h + 0.04 h = **1.07 h**. 2 h leaves a factor of 2.7 and 1.9; a shorter request would also be scheduled sooner but leaves little for a slow node or a year with more MCS (trackpcp grows with the rows) |
| `--chunk-months` | 12 | one stats job per year was run for both data sets |
| `run_dense_netcdf` | on (the template; always on by default for `stats_format: rows`) | 0.4 min (E3SM) and 2.1 min (IMERG) of one core, 0.10 and 0.43 GB; inside a 4-node stats job the other nodes wait for it (0.03 and 0.14 node-hours); memory 4.4 and 6.5 GB with `dense_netcdf_block_tracks: 4000` |
| `run_rainmap` | on (the template) | the 12 months take 2.1 min (E3SM) and 2.2 min (IMERG) on 4 nodes, about 3 min on one node (13 - 15 s a month); with the old reading the 4-node E3SM stats job needed 9.8 min |
| `run_trackpcp` | off (the template); on for a study of the rain of the tracks | E3SM 10.0 min on 4 nodes (9.7 on one), 103 million rows; IMERG 20.8 min on 4 nodes (20.1 on one), 243 million rows (2.4 GB Parquet); the main process of the step holds the rows (IMERG: 34 GB) |
| `skip_completed_steps` | `True` for a job that is submitted again | it skipped the six finished steps of the IMERG stats job in 0.3 s (their outputs were checked) |
| `step_profile_dir` | one folder per job (the generator's `--profile-dir`) | the ledgers of this campaign are the source of the step times above |
| output checks | on (the default) | the checks of all steps cost 0.16 s per track job and 0.3 s in the E3SM stats job |

The output of a year is 211 GB (E3SM) and 251 GB (IMERG) on `$SCRATCH`, which is purged: copy what you keep.

## Repeating a campaign

```bash
# a checkout of a commit that no later change can reach (see the method), and the files of the campaign (nothing is submitted without --submit)
git worktree add --detach /path/to/code HEAD
python /path/to/code/config/generate_tbpf_mcstracking_scripts.py -s 2009-01 -e 2009-12 -c /path/to/code/config/config_mcs_tbpf_e3sm_0p1deg_template.yml -o campaign_2009 \
    --input-dir /path/to/olr_pcp --root-path /path/to/mcs_tbpf/2009 --account <account> --stats-nodes 4 --trackpcp on --profile-dir /path/to/mcs_tbpf/2009/profile --pin-code --submit
# when all jobs have ended: the numbers and the figures
python Analysis/collect_campaign_performance.py --campaign-dir campaign_2009 --root /path/to/mcs_tbpf/2009 -o campaign.json --tag "E3SM ne120 2009" --save-sacct sacct.txt
python Analysis/make_performance_tbpf_figures.py --data campaign.json --outdir figures --name e3sm_2009
```

A campaign made with the current code has the dense netCDF step inside its stats job (after `speed`), so the figures show it as the first of the post steps of that job. `--save-sacct` keeps the text of `sacct`, so that the collection can be repeated (`--sacct-file`) after the job records have aged out. The figures are not committed (the images of the documentation are assets of the
`docs-assets` release, [docs_assets.md](docs_assets.md)).
