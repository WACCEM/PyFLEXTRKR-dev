# Radar-based MCS tracking (`feature_type: radar_feature`): checkpoint of the experimental work

> **Status: experimental.** The workflow is new, still being developed and evaluated, and its interfaces (config keys, variable names, the lifecycle definition) may still change.
> This page is a checkpoint written on 2026-10-19 for the branch `mcs_radar3d` (26 commits, `853f3ac` ... `192b762`, on top of the public `main` at `feea483`). It records what exists, how it was checked
> and what is still open, so that the development can continue from here. It has been run on one kind of input only: MRMS composite reflectivity on the HRRR 3 km grid, a 2-day development case
> (2-4 May 2025) and monthly runs for April-September 2025.

MCS are defined and tracked from 3D or 2D Cartesian radar reflectivity alone, with no Tb, precipitation features or SL3D. The page has three parts: the radar_feature workflow, `trackpcp` (precipitation inside the tracks of a finished
tracking) and the scalability fixes of shared steps that the monthly runs made necessary.

## 1. The radar_feature workflow

| Step | Code | What it does |
|---|---|---|
| feature identification | `idfeature_radar3d.py` | column-maximum reflectivity, contiguous features (30 and 10 dBZ), Steiner convective / stratiform classification, labeled convective cells (`cell_number`), echo-top heights at 10-50 dBZ (3D input), `conv_feature_number`: convective cells grouped by a disc dilation of `conv_dilate_radius_km` (0 = only touching pixels) |
| batching | `idfeature_driver.py` | optional `combine_files`: one lazy dataset over the period and Dask tasks per time frame in batches (opt-in for any feature type, the per-file dispatch is unchanged) |
| track statistics | `trackstats_func.py` (`radar_feature` branch) | centroid and reflectivity-weighted centroid, axes, aspect ratio, orientation, `max_dbz`, convective / stratiform area, `ncells`, echo tops of the convective pixels, areas above dBZ thresholds, `conv_majoraxis` (longest convective group) |
| first pass | `identifymcs_radar.py` | MCS test, 4-way class of every track (MCS, multi-cell, isolated cell, non-convective), short merge / split tracks folded into the track they belong to, `total_area`, `total_conv_area`, `total_strat_area` that match the folded mask; written to `mcsradarstats_filebase` |
| second pass | `regionprops_driver.py`, `regionprops_func.py`, `mcs_lifecycle.py` | shape and extreme statistics recomputed on the merged mask (own feature + folded fragments, the mask `mapfeature` paints), then the lifecycle staging; written to `mcsrobust_filebase` |
| pixel maps | `mapfeature_func.py` (`radar_feature` branch) | `dbztracknumber`, `feature_class` (class of the parent track also on fragment pixels), `conv_feature_number` |
| speed | `movement_speed.py` | the existing step on the new statistics (see section 3) |

`runscripts/run_mcs_radar3d.py` runs the chain. It exits with status 1 when `run_mapfeature` finds no `mcs_tracks_robust` file, so a batch job does not end as COMPLETED.

### Design choices

- **Keep every tracked feature**, classify it (4 classes), and fold only short merge / split fragments into their root, like `link_mergesplit_tracks.py` and unlike `identifymcs_tb`, which keeps only MCS tracks. A fragment is folded only if
  all its time steps exist on the root's time axis; otherwise it stays a standalone track (this changed the 2-day case from 12,566 to 15,758 kept tracks, with no unmatched fragment time points left and the same 31 MCS).
- **Two passes**, because `trackstats` describes the own feature of a track while the mask that `mapfeature` paints includes the folded fragments. The second pass recomputes the morphology on that mask; the MCS flags stay as the first pass
  decided (the log reports how many tracks the new major axis would flip).
- **Lifecycle** (cloud only, unorganized, genesis, mature, decay; after Coniglio et al. 2010), in `mcs_lifecycle.py`: every organized-line episode is staged independently (multi-cycle tracks), gaps and lags are in hours and applied to the real
  sample times (MRMS steps alternate between 4 and 6 minutes), a running median (`mcs_lifecycle_smooth_hour`, default 0.5 h) removes spikes and dropouts, and the staging cannot fail on tracks without stages (the older `robustmcs_radar.py`
  raises a `NameError` there; it is untouched).
- **`conv_dilate_overwrite_radius_km`** (default off) regroups the convective pixels at another radius inside the second pass, so a new radius does not need `idfeature` again.

### What was checked

- Unit tests for each module on synthetic data (`test_idfeature_radar3d_synthetic.py`, `test_trackstats_radar_feature.py`, `test_identifymcs_radar_synthetic.py`, `test_mcs_lifecycle_synthetic.py` with Monte-Carlo scenarios,
  `test_regionprops_radar_feature.py`, `test_mapfeature_radar_feature.py`, `test_steiner_func.py`); real-data parity of `idfeature_radar3d` against the validated `radar_cells` output for one MRMS snapshot (`convsf` and `conv_core`
  identical, reflectivity and echo tops allclose; marked `local`).
- 2-day MRMS case (576 files, 31,151 tracks): merged-mask area equals `total_area` at all 100,741 (track, time) pairs; every second-pass variable except `conv_majoraxis` (which is the point of the second pass) equals its `trackstats`
  value at all 83,704 pairs without fragments; `total_*area` equal a pixel recount at all pairs; 31 MCS, 17 of them with a lifetime of 8 h or more.
- Lifecycle: with a 100 km line threshold, 17 evaluated MCS (lifetime of 8 h or more) have a complete genesis-mature-decay cycle in 0 (no dilation), 7 (5 km), 13 (9 and 10 km) and 14 (20 km) of the cases; on synthetic MCS with known stages
  (5-minute data) the 0.5 h smoothing finds the right stages and onsets in 0.99 (moderate noise) and 0.95 (severe noise) of the realizations, against 0.62 and 0.32 without it.
- The May 2025 month (8,872 files, 239,963 tracks) and the other months of April-September 2025 were run through the chain, one job per month.

### Open issues

- The dilation radius and the line threshold are a choice, not a result: the 0.5 h smoothing was tuned on synthetic series only, and at 20 km most tracks are organized from the start.
- A merge / split fragment that overlaps its MCS only partly is detached from it completely (11 % fewer time points attached to MCS roots at 4 h); attaching only the overlap needs track truncation.
- `total_*area` and the painted mask undercount when more fragments coexist than `nmaxlinks` (a warning is logged; 50 removed the effect in the test case).
- A mature stage can last only 2 steps and the track still counts as complete.
- Not tested: periodic boundaries (`pbc_direction`) in the second pass; other inputs than MRMS; other grids than 3 km. `idfeature` limits the dilation to the valid-data mask, which is not stored in the cloudid files, so
  `conv_dilate_overwrite_radius_km` does not apply it (no effect for MRMS).
- 9 multi-cell tracks would meet the MCS criteria with the merged-mask major axis; their flags are not changed.
- `movement_speed` aligns each track to the frame pair nearest its start time, so it gives meaningful values only when the pair list covers the whole tracking period.

## 2. `trackpcp`: precipitation inside tracks

`runscripts/run_trackpcp.py`, `trackpcp_driver.py`, `trackpcp_func.py`, `trackvar_io.py` and `parallel_utils.py`, documented in [track_precipitation.md](track_precipitation.md) with an example config (`config/config_mrms_trackpcp_example.yml`).
It matches a gridded variable that is not part of the tracking (hourly QPE in the verification case; NetCDF files or a Zarr store, same grid) with the tracks of any finished tracking, in three steps:

1. **track statistics**: mean / median / max / sum / percentiles / area of the variable inside the swath of each track over the data window, split into convective and stratiform part, and the precipitation variables of
   `matchtbradar_func` (`total_rain`, `total_volrain`, `pf_*`); the input statistics file is copied with the new variables appended, and a Parquet table is written;
2. **pixel accumulation**: depth, mm/day and largest hourly depth at every (track, pixel) as a Parquet table; each pixel-window has one owner track (best class, then most steps, then lowest track number);
3. **class maps** of precipitation, cloud count and precipitation count per track class.

`parallel_utils.run_bounded` (at most 2 x threads tasks in flight, inputs scattered once, memory released after every task) runs the Dask work.

Checked on the MRMS `radar_feature` tracking, April-September 2025: per-class sums of the accumulations equal the class maps (relative difference at most 2.3e-10), per-track volumes of step 1 equal step 2 (1.5e-7), and one day was
checked against an independent union of the 12 masks. MCS carry 44-61 % of the precipitation inside the reflectivity coverage (mean 53 %), multi-cell tracks 36-46 %, isolated and non-convective features less than 1.2 %.
Cost for May 2025 (240,000 tracks, 744 windows, 64 workers): 7.2 / 1.4 / 1.2 minutes for the three steps, 72-79 GB peak on the node.

Limits: the data must be on the tracking grid (checked, the run stops otherwise); periodic boundaries are not handled; the class of a track is its `track_class` (the per-time class path is tested on synthetic data only);
12-15 % of the pixel-windows are contested by more than one track (the owner rule is documented); the run times for a 7-month continuous run are estimates.

## 3. Scalability fixes of shared steps

Found on the May 2025 month (8,872 files, 239,963 tracks). Each change has a test against the previous code, and the outputs are identical.

| Change | Before | After |
|---|---|---|
| `ft_utilities.match_pairs_to_files` (all (track, time) pairs of every file in one sorted pass) in `regionprops_driver` and `mapfeature_driver` | 1.78 s per file, 265 min serial for the month | 0.7 s for all 1.68 M pairs |
| lazy opening of the dense statistics file, only the used variables are read | 180 GB file loaded, 245 GB of memory | the second pass runs in 37 min with a 211 GB peak (it did not finish in 63 min) |
| `movement_speed`: sparse results per file pair, appended to a copy of the statistics file | 48 min, 443 GB peak | 8 min 45 s, 69 GB peak; the five movement variables identical (0 of 132 million elements differ) |
| `ft_utilities.resolve_sparse_max_trackduration` | a step run alone with a stale `duration_range` failed with `ValueError: axis 1 index 282 exceeds matrix dimension 10` | the size comes from the file (expanded with `duration_range_auto_update`, otherwise the run stops with a message, as `trackstats_driver` does); used by `load_sparse_trackstats` and `convert_trackstats_sparse2dense` |

`movement_speed`, `load_sparse_trackstats` and `convert_trackstats_sparse2dense` are shared with the Tb + PF pipelines. They are covered by `tests/test_movement_speed.py` (against a frozen copy of the old module),
`tests/test_load_sparse_trackstats.py` and the comparisons above; the end-to-end demo registry (`tests/run_demo_tests.py --all`) was not run for these changes.

## 4. What is not in the repository

- The MRMS production config of the radar workflow (site-specific) and the documentation of the `radar_feature` config keys: there is no example config and no user documentation for the workflow yet.
- The lifecycle diagnostic plots (`plot_mcs_radar_lifecycle.py`, `select_mcs_lifecycle_tracks.py`) and the notebooks of the trackpcp results: developed on this branch, then moved to a separate analysis repository; they import
  `pyflextrkr.mcs_lifecycle`, which stays here.
- Not run: the radar workflow on another dataset than MRMS; `tests/run_demo_tests.py --all` on the branch.

## 5. What the public CI said about this branch (found and fixed on 2026-10-20)

The public CI (`.github/workflows/test.yml`) runs `pytest tests/` after `pip install -r requirements-ci.txt` and `pip install -e .`. A venv built from the same two requirements files (Python 3.12, cartopy 0.26, matplotlib 3.11)
ran the tests of this branch: 549 passed, 94 skipped, 2 collection errors and 6 failures, all from two packages that the CI environment does not install (a third module with the same import, `mapfeature_driver.py`, showed up once the
collection errors were gone). Both are fixed:

- **`distributed`**: `regionprops_driver.py`, `mapfeature_driver.py` and `movement_speed.py` (and the frozen copy `tests/movement_speed_reference.py`) imported `dask.distributed` at the top of the module. `dask` alone is installed in the CI, so
  `tests/test_movement_speed.py` and `tests/test_regionprops_radar_feature.py` could not be collected, which stops the whole run. `wait` is now imported in the parallel branch, where it is used (as in
  `pyflextrkr/visualization/runner.py`); `tests/test_imports_without_distributed.py` imports the modules of this workflow with `distributed` blocked. Eight other drivers (`idfeature_driver.py`, `trackstats_driver.py`, ...) still import it at
  the top; no CI test imports them (`test_idfeature_driver_batching.py` has its own `importorskip`).
- **`pyarrow`**: the 6 failures of `tests/test_trackpcp_synthetic.py` were "pyarrow is required for parquet support" (the Parquet tables of `trackpcp`). `pyarrow` is now in `requirements.txt` and `environment.yml`; the conda-forge
  recipe needs it too when this work reaches the public repository.

With both, the same venv (with `pyarrow` installed from `requirements.txt`, as the CI does) gives 644 passed, 94 skipped, no error and no failure; the development environment gives 698 passed (`-m "not local and not demo"`). The about 90 tests of the two
modules that the collection errors had hidden now run in the CI.

Still to expect when this branch becomes a public pull request: Windows and macOS (these tests have run on Linux only). The CI of the visualization pull request found test-side problems there that are worth checking: text files read
without `encoding='utf-8'` (the default on Windows is cp1252), expected strings with `/` where Windows has `\`, and `bash` on Windows paths.
