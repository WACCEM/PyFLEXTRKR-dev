# Track-matched precipitation (trackpcp)

Post-processing of a finished tracking with a gridded variable that was **not** part of the tracking: precipitation statistics inside
the tracked features, accumulated precipitation at every pixel of every track, and monthly precipitation maps by feature class.
Written for hourly MRMS QPE and 5-minute radar-based tracks; the data may have its own files, naming, time axis (NetCDF files or a Zarr
store) and any time resolution, but must be on the tracking grid (checked; the step aborts otherwise).

| step | function | output |
|---|---|---|
| 1 | `trackpcp_driver.compute_track_precip` | a **new copy of the track statistics file** with the new variables appended (dense `tracks x times`, NaN where there is no record), and the same rows as a Parquet table |
| 2 | `trackpcp_driver.compute_track_pixel_accum` | `trackpcp_pixels_START_END.parquet` (one row per track and pixel: accumulated depth, rate in mm/day, largest hourly depth, number of windows), `trackpcp_metrics_START_END.parquet` (one row per track), `trackpcp_grid.nc` |
| 3 | `trackpcp_driver.compute_class_maps` | `trackpcp_classmaps_START_END.nc`: `precip`, `cloud_count`, `precip_count` (and the convective / stratiform split) per class on the grid, `total_precip` |

```bash
python runscripts/run_trackpcp.py tracking_config.yml config/config_mrms_trackpcp_example.yml --steps 1 2 3 [--year 2025 --month 5]
python Analysis/calc_monthly_rainmap_by_class.py tracking_config.yml config/config_mrms_trackpcp_example.yml   # step 3 only
```

`tracking_config.yml` supplies the paths and dates of the tracking run; the second YAML holds the `trackpcp_*` keys (override the first).
`run_parallel` / `nprocesses` as in the tracking run scripts (0 serial, 1 LocalCluster, 2 existing scheduler).

## How data records and tracks are matched

A data record valid at time T represents a window of tracking times (`trackpcp_time_match`, `trackpcp_window_sec`):

| mode | window | use |
|---|---|---|
| `prior` | (T - W, T] | accumulation that ends at T (MRMS QPE: the file stamp is the end of the hour) |
| `after` | [T, T + W) | accumulation that starts at T |
| `nearest` | [T - tol, T + tol) | instantaneous data matched to the nearest track time (5-min rain rates with 5-min tracks; tol defaults to W / 2) |

`ft_utilities.match_pairs_to_windows` assigns all (track, time) pairs of a month to windows in one sorted pass.
The data time is parsed from the file name (`trackpcp_data_time_format`, as for the tracking inputs), or read from a time variable / Zarr coordinate.

Definitions (`trackpcp_func.py`)
- **swath**: union over the steps of a window of a track's mask (`cloudtracknumber`, merge/split fragments included);
- **convective swath**: pixels of the swath that are convective (`convsf == 4`) in at least one step of the window, inside the track's mask at that step; all data in it is
  convective, the rest of the swath is stratiform (so the convective mask takes precedence over stratiform for a data record that is coarser than the tracking);
- **owner**: a pixel claimed by more than one track in a window belongs to one track only: the best class (ascending class value, or `trackpcp_class_priority`), then the track
  that covers it in most steps, then the lowest track number. All three steps use the same rule, so per-track sums add up to the class maps and never exceed the total.
  The number of contested pixels is logged and stored (`n_pixels_contested`).

## Where a record is stored (step 1)

The output keeps the tracking time axis. The value of a data record is stored at **one** step of each track: the step nearest the data time inside the window
(the last step for `prior`, the first for `after`); the other steps stay NaN. This was chosen over repeating the value in all steps of the window (a sum over the lifetime
would count it 12 times) and over a separate hourly axis (no direct join with area, class or lifecycle). Companion variables: `pcp_time` (valid time of the record),
`pcp_nsteps` (steps of the track in the window; fewer than a full window means the track covers only part of the hour), per-track totals `track_total_rainvolume`,
`track_conv_rainvolume`, `track_strat_rainvolume`, `track_pcp_nwindows`, `track_max_window_rainvolume`. Use nan-aware reductions for sums over time. NaN compresses to nothing
(zlib), and the Parquet table has only the rows that exist.

Variables (precipitation module, `trackpcp_precip: true`): `total_rain` (sum of the rain rates of the swath, mm/h), `total_volrain` (mm/h km2), `total_rainvolume` (10^6 m3 that fell on
the swath during the window), `conv_*` and `strat_*` of the same, `pcp_area`, `pcp_conv_area`, `pcp_strat_area`, `total_heavyrain`, `total_heavyvolrain`, `rainrate_heavyrain`
(above `trackpcp_heavy_thresh`), `pcp_valid_frac`, and up to `trackpcp_nmaxpf` precipitation features (connected swath pixels above `trackpcp_rain_thresh`, largest first):
`pf_npf, pf_lon, pf_lat, pf_area, pf_rainrate, pf_maxrainrate, pf_skewness, pf_majoraxis, pf_minoraxis, pf_aspectratio, pf_orientation, pf_eccentricity, pf_perimeter,
pf_lon/lat_centroid, pf_lon/lat_weightedcentroid, pf_lon/lat_maxrainrate, pf_accumrain, pf_accumrainheavy, pf_ccarea, pf_sfarea, pf_ccrainrate, pf_sfrainrate, pf_ccrainamount,
pf_sfrainamount` (the names of `matchtbradar_func.py`). Generic statistics of the data variable in the swath, in the convective part and in the stratiform part:
`trackpcp_stats` (count, area, mean, median, min, max, sum, std, `pNN` percentiles), named `<prefix>_<stat>`, `<prefix>_conv_<stat>`, `<prefix>_strat_<stat>`.

## Pixel accumulation and flood metrics (step 2)

`accum_mm` is the depth at the pixel over the track's life (rate x window length, one owner per pixel and window), `rate_mm_day` = accum / lifetime in days,
`max_1h_mm` the largest single-window depth, `pix_index` the flat index `iy * nx + ix` of the tracking grid. The table is sorted by track, so one track is read with
`pq.read_table(file, filters=[("track_number", "==", k)])`, and `trackpcp_driver.track_pixels_to_map(df, k, ny, nx)` returns the 2-D map.
Parquet is used (not a padded NetCDF array) because the number of pixels differs by orders of magnitude between tracks; a ragged NetCDF would need manual
slicing by offsets in xarray, a padded one would be mostly NaN.

Metrics per track: `accum_max_mm`, `accum_vol_1e6m3`, `swath_area_km2`, `area_ge<T>mm_km2`, `cluster_area_km2` / `cluster_vol_1e6m3` (largest connected region at or above
`trackpcp_cluster_thresh_mm`), `local_depth_mm_<L>km` (largest mean depth in an L x L km box over the whole track: concentrated high volume), `local_rate_mm_h_<L>km`
(largest box-mean hourly rain over time: high volume in a short time). Dry pixels count as 0 in the box mean.

## Class maps (step 3)

`trackpcp_coverage_varname` (e.g. `dbz_comp`) names a pixel variable that is missing where the tracking input has no data; precipitation of the data set at such pixels (in all steps
of the window) is accumulated in `precip_uncovered` instead of being counted as "outside tracked features" (MRMS QPE has values offshore and over Mexico where the composite reflectivity has none:
14-15 % of the August and September precipitation of the domain, 3-4 % in the other months).

Classes (values and names) are read from the class variable of the statistics file (`trackpcp_class_varname`, default `track_class`; its `long_name` lists `value=name`).
`trackpcp_classes` selects values; without a class variable all tracked features form one class. `precip_count` counts windows with the class present and a value of at
least `trackpcp_pcp_thresh` (default 1 mm/h); frequency = count / `ntimes`.

## Parallel processing

Windows are tasks of a bounded Dask run (`parallel_utils.run_bounded`): at most 2 x threads submitted at a time, read-only inputs scattered once, `gc` + `malloc_trim`
after every task (the memory glibc keeps from freed work arrays otherwise grows every task), a failed task is retried once, then counted (abort above 5 %). Results are small
arrays per window; step 3 reduces one block of windows per worker and sums the blocks as they arrive.

Measured on one Perlmutter CPU node, 64 workers, one month of 5-minute MRMS tracking (8,600-8,900 frames, 190,000-240,000 tracks) and hourly QPE (720-744 windows):
step 1 5.2-6.8 min (about 1.5 min for the windows, the rest writing the new dense variables and the Parquet table), step 2 about 1 min (38-46 million (track, pixel) rows),
step 3 0.8 min. Six months: 48 minutes in total.

## Checks that were made (Apr-Sep 2025, MRMS)

- brute-force reference on a synthetic grid (`tests/test_trackpcp_synthetic.py`, `tests/test_match_pairs_to_windows.py`): window matching, swath, convective swath, owner rule,
  statistics incl. percentiles, NetCDF and Zarr sources, a run without a class variable, 5-minute data, pixel table -> map, class maps against pixel accumulations;
- real data, all six months: per-class sums of the pixel accumulations (step 2) equal the class maps (step 3), and per-track volumes of step 1 equal those of step 2, to float
  precision (relative differences below 1e-6);
- the owner rule: 12-15 % of the owner pixel-windows were claimed by more than one track (ID hand-overs inside an hour and overlapping features).

## Limits

Same grid as the tracking (no remapping yet); periodic boundaries are not handled; a track's class is `track_class` (whole lifetime) unless a per-time class variable
(2-D) is given; the owner rule trades some swath area of overlapping tracks for sums that add up; the dense step-1 variables are `tracks x times` like the existing statistics
files (the PF variables add `nmaxpf` columns), so a month adds 0.15-0.18 GB to the file (1.1-1.3 GB for the MRMS months); use the Parquet table for analysis of the records themselves.
