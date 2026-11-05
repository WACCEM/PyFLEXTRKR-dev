"""
Lifecycle staging of tracked MCS (feature_type='radar_feature').

Hardened, multi-cycle-aware port of robustmcs_radar.py's Coniglio-et-al.-(2010) staging. It works on
the time series of ONE track and has no I/O and no dask import, so it is directly unit-testable with
synthetic series (tests/test_mcs_lifecycle_synthetic.py).

Stages: 1=cloud only, 2=unorganized convection (convective initiation), 3=genesis, 4=mature, 5=decay and, in the
single-pass method, 6=redeveloping. They follow the lifecycle of Feng et al. (2019) (after Coniglio et al. 2010).
"After" means a later sample, not a duration, so the definition works for any time resolution. The only time
constants are mcs_lifecycle_gap_hour and mcs_lifecycle_smooth_hour (a running median that removes isolated spikes
and short dropouts before the thresholds are applied): durations in hours compared with the real sample times, so
the staging does not depend on the time resolution of the data (5 min ... 1 h) or on a regular time spacing (the
MRMS steps alternate between about 4 and 6 min, and samples may be missing).

Two methods (params['mcs_lifecycle_method']):

single (the default of the current configs, classify_lifecycle_single): one lifecycle per track.
  line      the convective size series (params['mcs_lifecycle_line_method']: the sum of the circular-equivalent
            diameters of the convective cells conv_diameter_sum, the variable of the MCS test, or the cell-cluster
            major axis conv_majoraxis) at or above mcs_lifecycle_line_thresh (100 km);
  floor     the median (mcs_lifecycle_sf_stat 'median', or the mean) of the track's positive stratiform areas;
  mature condition  the line is at or above the threshold and the stratiform area is at or above the floor.
  genesis   happens once: the first sample with the line, kept while the line holds, until the first later sample
            that meets the mature condition; if the line falls below the threshold before that, the stage falls back
            to unorganized and there is no second genesis;
  redeveloping (6)  the line is at or above the threshold after the first genesis but the stratiform area is below
            the floor: before the first mature stage (after a fall-back), between two mature segments, or without any
            mature stage; not after the last mature segment;
  mature    from the first to the last sample that meets the mature condition, interruptions included while the
            system is still there (line at or above the threshold or stratiform at or above the floor); a stretch
            where neither holds for longer than mcs_lifecycle_gap_hour ends the segment, the next sample that meets
            the condition starts another one (a restart, counted by mature_segments);
  decay     after the last mature sample of a segment until the next segment or the end of the track; never before
            the first mature stage.

episodes (the rules of the earlier configs, the rest of this module): every organized-line episode (line samples
less than mcs_lifecycle_gap_hour apart) is staged on its own, genesis = the first sample of an episode (or
mcs_lifecycle_genesis_lag_hour later), mature = a later sample (mcs_lifecycle_mature_lag_hour after genesis) while
the line and the stratiform floor hold, decay = after the last mature sample until the next episode. The lags
(1 h and 2 h) are those of the earlier hourly design; with 0 the episodes have no lag.
"""
import logging
import warnings
import numpy as np

# Stage codes
STAGE_CLOUD_ONLY = 1
STAGE_UNORGANIZED = 2
STAGE_GENESIS = 3
STAGE_MATURE = 4
STAGE_DECAY = 5
STAGE_REDEVELOPING = 6      # single-pass method only
LIFECYCLE_METHODS = ("single", "episodes")
LIFECYCLE_LINE_VARNAMES = ("conv_diameter_sum", "conv_majoraxis")


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------

def _steps_from_hours(hours, time_resolution):
    """Convert an hour duration to a (minimum 1) number of time steps."""
    return max(1, int(round(hours / time_resolution)))


def _lag_hours(hours, time_resolution):
    """A lag [hour] rounded to whole steps of the nominal resolution; 0 stays 0 (no lag), any other value is at least one step."""
    return 0.0 if hours <= 0 else _steps_from_hours(hours, time_resolution) * time_resolution


def lifecycle_method(params):
    """The staging method of a params dict ('single' or 'episodes'; a dict without the key is the earlier 'episodes')."""
    method = params.get('mcs_lifecycle_method', 'episodes')
    if method not in LIFECYCLE_METHODS:
        raise ValueError(f"mcs_lifecycle_method '{method}' is not one of {list(LIFECYCLE_METHODS)}")
    return method


def lifecycle_line_varname(params):
    """The statistics variable that is the line series of the staging (a dict without the key: conv_majoraxis, the earlier one)."""
    name = params.get('mcs_lifecycle_line_method', 'conv_majoraxis')
    if name not in LIFECYCLE_LINE_VARNAMES:
        raise ValueError(f"mcs_lifecycle_line_method '{name}' is not one of {list(LIFECYCLE_LINE_VARNAMES)}")
    return name


def n_lifecycle_stages(params):
    """Number of stages of a method: 6 for 'single' (with redeveloping), 5 for 'episodes' (the columns of lifecycle_index)."""
    return 6 if lifecycle_method(params) == 'single' else 5


def count_runs(stage, code):
    """Number of separate runs of one stage code in a stage array."""
    s = np.asarray(stage) == code
    return int(np.sum(s[1:] & ~s[:-1]) + int(s[0])) if s.size else 0


def mature_segments(stage):
    """Number of separate mature runs of a stage array: 0 = none, 1 = one lifecycle, 2 or more = restarts (decay or redeveloping between)."""
    return count_runs(stage, STAGE_MATURE)


# Bits of lifecycle_upscale_flag: a quick filter for MCS that grow upscale naturally (the track starts as cloud or unorganized
# convection, then forms its genesis and reaches the mature stage, optionally not at once). A track meets all the requirements when
# (flag & UPSCALE_REQUIRED_MASK) == UPSCALE_REQUIRED_MASK; the redeveloping bit is information only.
UPSCALE_STARTS_UNORGANIZED = 1
UPSCALE_GENESIS_OBSERVED = 2
UPSCALE_MATURE_OBSERVED = 4
UPSCALE_MATURE_AFTER_MIN_TIME = 8
UPSCALE_REDEVELOPING_BEFORE_MATURE = 16
UPSCALE_REQUIRED_MASK = UPSCALE_STARTS_UNORGANIZED | UPSCALE_GENESIS_OBSERVED | UPSCALE_MATURE_OBSERVED | UPSCALE_MATURE_AFTER_MIN_TIME
UPSCALE_FLAG_MASKS = (UPSCALE_STARTS_UNORGANIZED, UPSCALE_GENESIS_OBSERVED, UPSCALE_MATURE_OBSERVED,
                      UPSCALE_MATURE_AFTER_MIN_TIME, UPSCALE_REDEVELOPING_BEFORE_MATURE)
UPSCALE_FLAG_MEANINGS = ("starts_unorganized genesis_observed mature_observed mature_after_min_time redeveloping_before_mature")


def upscale_growth(stage, time_hours, min_mature_hour=1.0):
    """
    The natural-upscale-growth flag of ONE staged track (method 'single') and the times of its stages.

    Args:
        stage: 1D np.array(int), the lifecycle stage of every time of the track (classify_lifecycle_single).
        time_hours: 1D np.array(float), sample times [hour], same length (only differences matter: the times returned are
            counted from the first sample).
        min_mature_hour: float (default: 1.0)
            Shortest time from the first sample to the first mature stage that sets the bit UPSCALE_MATURE_AFTER_MIN_TIME.

    Returns:
        flag: int, the sum of the UPSCALE_* bits that hold:
            1 the first stage is cloud only or unorganized (the track does not begin already organized),
            2 a genesis stage occurs, 4 a mature stage occurs,
            8 the first mature stage is at least min_mature_hour after the first sample,
            16 a redeveloping stage occurs before the first mature stage (or, without a mature stage, at all): information only.
        time_to_genesis, time_to_mature, time_to_redeveloping: float, hours from the first sample to the first sample of that
            stage; NaN if the stage does not occur.
    """
    stage = np.asarray(stage)
    time_hours = np.asarray(time_hours, dtype=float)
    nan = float("nan")
    if stage.size == 0:
        return 0, nan, nan, nan
    first = {code: (int(np.flatnonzero(stage == code)[0]) if np.any(stage == code) else None)
             for code in (STAGE_GENESIS, STAGE_MATURE, STAGE_REDEVELOPING)}
    since_start = lambda i: float(time_hours[i] - time_hours[0]) if i is not None else nan
    t_genesis, t_mature, t_redev = (since_start(first[c]) for c in (STAGE_GENESIS, STAGE_MATURE, STAGE_REDEVELOPING))
    flag = 0
    if stage[0] in (STAGE_CLOUD_ONLY, STAGE_UNORGANIZED):
        flag |= UPSCALE_STARTS_UNORGANIZED
    if first[STAGE_GENESIS] is not None:
        flag |= UPSCALE_GENESIS_OBSERVED
    if first[STAGE_MATURE] is not None:
        flag |= UPSCALE_MATURE_OBSERVED
        if t_mature >= min_mature_hour - 1e-6:
            flag |= UPSCALE_MATURE_AFTER_MIN_TIME
    if first[STAGE_REDEVELOPING] is not None and (first[STAGE_MATURE] is None or first[STAGE_REDEVELOPING] < first[STAGE_MATURE]):
        flag |= UPSCALE_REDEVELOPING_BEFORE_MATURE
    return flag, t_genesis, t_mature, t_redev


def effective_constants(params, time_resolution):
    """
    What the time constants of the staging really are at a time resolution (the staging rounds them to whole steps).

    Args:
        params: dict, see classify_lifecycle.
        time_resolution: float [hour], nominal time step of the data.

    Returns:
        constants: dict with gap_hour, genesis_lag_hour, mature_lag_hour (as applied; the single-pass method has no lags and uses the
            gap in hours as given), smooth_hour, smooth_samples (samples of the nominal time axis in the smoothing window; below 3 the
            smoothing does nothing) and time_resolution_hour.
    """
    smooth_hour = params.get('mcs_lifecycle_smooth_hour', 0.0)
    smooth_samples = 2 * int(np.floor(0.5 * smooth_hour / time_resolution + 1e-6)) + 1 if smooth_hour > 0 else 1
    single = lifecycle_method(params) == 'single'
    return {
        "time_resolution_hour": time_resolution,
        "gap_hour": params['mcs_lifecycle_gap_hour'] if single else _steps_from_hours(params['mcs_lifecycle_gap_hour'], time_resolution) * time_resolution,
        "genesis_lag_hour": _lag_hours(params['mcs_lifecycle_genesis_lag_hour'], time_resolution),
        "mature_lag_hour": _lag_hours(params['mcs_lifecycle_mature_lag_hour'], time_resolution),
        "smooth_hour": smooth_hour, "smooth_samples": smooth_samples,
    }


def effective_constants_text(params, time_resolution):
    """One line for the log: the method, the constants of effective_constants and whether the smoothing does anything."""
    c = effective_constants(params, time_resolution)
    smooth = (f"running median {c['smooth_hour']:g} h = {c['smooth_samples']} sample(s)" +
              (" (no smoothing)" if c['smooth_samples'] < 3 else ""))
    if lifecycle_method(params) == 'single':
        return (f"Lifecycle staging at {60 * c['time_resolution_hour']:g} min (single genesis, line {lifecycle_line_varname(params)} >= "
                f"{params['mcs_lifecycle_line_thresh']:g} km, stratiform floor {params.get('mcs_lifecycle_sf_stat', 'mean')}): "
                f"interruptions bridged up to {c['gap_hour']:g} h, {smooth}")
    return (f"Lifecycle staging at {60 * c['time_resolution_hour']:g} min (episodes, line {lifecycle_line_varname(params)}): episodes merge over "
            f"{c['gap_hour']:g} h, genesis lag {c['genesis_lag_hour']:g} h, mature lag {c['mature_lag_hour']:g} h, {smooth}")


def lifecycle_global_attrs(params, time_resolution):
    """
    Global attributes that record the lifecycle settings of a run (the stats file says what staged it).

    Args:
        params: dict, from mcs_params_from_config.
        time_resolution: float [hour], nominal time step of the data.

    Returns:
        attrs: dict, mcs_lifecycle_* (thresholds in km / hours, the method and line variable, the stratiform statistic, the classes if
            set, the resolution and the number of samples in the smoothing window).
    """
    c = effective_constants(params, time_resolution)
    attrs = {
        "mcs_lifecycle_thresh_hour": params['mcs_lifecycle_thresh'], "mcs_lifecycle_smooth_hour": params['mcs_lifecycle_smooth_hour'],
        "mcs_lifecycle_gap_hour": params['mcs_lifecycle_gap_hour'], "mcs_lifecycle_genesis_lag_hour": params['mcs_lifecycle_genesis_lag_hour'],
        "mcs_lifecycle_mature_lag_hour": params['mcs_lifecycle_mature_lag_hour'], "mcs_lifecycle_line_thresh_km": params['mcs_lifecycle_line_thresh'],
        "mcs_lifecycle_sf_stat": params['mcs_lifecycle_sf_stat'], "mcs_lifecycle_time_resolution_hour": time_resolution,
        "mcs_lifecycle_smooth_samples": c['smooth_samples'], "mcs_lifecycle_method": lifecycle_method(params),
        "mcs_lifecycle_line_method": lifecycle_line_varname(params),
        "mcs_upscale_min_mature_hour": params.get('mcs_upscale_min_mature_hour', 1.0),
    }
    if params.get('mcs_lifecycle_classes') is not None:
        attrs["mcs_lifecycle_classes"] = np.array(params['mcs_lifecycle_classes'], dtype=np.int32)
    return attrs


def lifecycle_variable_attrs(params, fillval, fillval_f=np.nan):
    """
    The attributes of the lifecycle_* output variables of a run (the same texts in the dense and the row-table second pass).

    Args:
        params: dict, from mcs_params_from_config (the method decides the texts and the two extra variables of 'single').
        fillval: int, the _FillValue of the integer variables.
        fillval_f: float (default: NaN), the _FillValue of the time variables.

    Returns:
        attrs: dict {variable name: attribute dict}; 'lifecycle_nmature', 'lifecycle_erratic_flag', 'lifecycle_upscale_flag' and the
            three 'lifecycle_time_to_*_hr' variables only for the 'single' method.
        comment: string, the 'comments' attribute of every variable.
    """
    single = lifecycle_method(params) == 'single'
    base = {"units": "unitless", "_FillValue": fillval}
    attrs = {
        "lifecycle_stage": dict(long_name="MCS lifecycle stage: 1=cloud only, 2=unorganized convection, 3=genesis, 4=mature, 5=decay"
                                + (", 6=redeveloping (line back, not yet mature)" if single else ""), **base),
        "lifecycle_cycle": dict(long_name=("Lifecycle number (1-based): 1 from genesis, plus 1 at the start of every later mature segment; "
                                           "0 = before genesis" if single else
                                           "Which organized-convective-line episode (1-based) a time belongs to; 0 = not part of any episode"), **base),
        "lifecycle_index": dict(long_name="Time index of the first occurrence of each lifecycle stage "
                                          f"(1-{n_lifecycle_stages(params)}); -1 = that stage was not observed", **base),
        "lifecycle_ncycles": dict(long_name="Number of organized-convective-line episodes found", **base),
        "lifecycle_complete_flag": dict(long_name="Flag indicating genesis, mature, and decay stages were all observed", **base),
        "lifecycle_evaluated": dict(long_name="Flag indicating lifecycle staging was evaluated for this track "
                                              "(track lifetime >= mcs_lifecycle_thresh and track_class in mcs_lifecycle_classes)", **base),
    }
    if single:
        attrs["lifecycle_nmature"] = dict(long_name="Number of separate mature segments: 0 = none, 1 = one lifecycle, "
                                                    "2 or more = the mature stage restarted after a decay or a redeveloping stretch", **base)
        attrs["lifecycle_erratic_flag"] = dict(long_name="Flag: the lifecycle has more than one mature segment (lifecycle_nmature >= 2)", **base)
        min_hour = params.get('mcs_upscale_min_mature_hour', 1.0)
        attrs["lifecycle_upscale_flag"] = dict(
            long_name="Natural upscale growth bit mask: the track starts as cloud or unorganized convection, then genesis and mature",
            flag_masks=np.array(UPSCALE_FLAG_MASKS, dtype=np.int16), flag_meanings=UPSCALE_FLAG_MEANINGS,
            flag_description=(f"Bits: 1 starts_unorganized (first stage is cloud only or unorganized), 2 genesis_observed, 4 mature_observed, "
                      f"8 mature_after_min_time (first mature stage at least {min_hour:g} h after the first sample, "
                      f"mcs_upscale_min_mature_hour), 16 redeveloping_before_mature (information only). All requirements: "
                      f"(flag & {UPSCALE_REQUIRED_MASK}) == {UPSCALE_REQUIRED_MASK}. Not a split-born test: see start_split_cloudnumber."),
            **base)
        time_base = {"units": "hour", "_FillValue": fillval_f}
        for stage_name in ("genesis", "mature", "redeveloping"):
            attrs[f"lifecycle_time_to_{stage_name}_hr"] = dict(
                long_name=f"Time from the first sample of the track to the first {stage_name} stage; NaN if the stage does not occur",
                **time_base)
    comment = (f"Classified from total_conv_area, total_strat_area and the merged-mask {lifecycle_line_varname(params)} "
               "(mcs_lifecycle.classify_lifecycle)")
    return attrs, comment


def lifecycle_tracks(track_lifetime_hr, track_class, params, select_track):
    """
    The tracks that get a lifecycle staging.

    Args:
        track_lifetime_hr: 1D np.array, lifetime of every track [hour].
        track_class: 1D np.array, track_class of every track (only used when mcs_lifecycle_classes is set).
        params: dict with mcs_lifecycle_thresh (minimum lifetime [hour]) and mcs_lifecycle_classes (list of track_class values, or None).
        select_track: 1D np.array(bool), the tracks that the second pass handles (regionprops_track_classes).

    Returns:
        indices: 1D np.array(int), the tracks to stage.
    """
    selected = (track_lifetime_hr >= params['mcs_lifecycle_thresh']) & select_track
    if params.get('mcs_lifecycle_classes') is not None:
        selected &= np.isin(track_class, params['mcs_lifecycle_classes'])
    return np.where(selected)[0]


def running_median(values, time_hours, window_hours, skipna=False):
    """
    Centered running median over a time window.

    Args:
        values: 1D np.array
        time_hours: 1D np.array, strictly increasing sample times [hour]; irregular spacing and
            missing samples are fine.
        window_hours: float, full width of the window [hour]. <= 0 returns the values unchanged.
            Each output is the median of the samples within +-window_hours/2 of its time; at the
            ends of the series the window is truncated.
        skipna: bool, default False. False: a NaN in the window makes the output NaN (the staging
            replaces NaN by 0 before it smooths). True: NaN are left out of the median, and a window
            of only NaN gives NaN (for display of series that have real gaps, e.g. echo-top height).

    Returns:
        1D np.array(float), same length as values.
    """
    values = np.asarray(values, dtype=float)
    if window_hours <= 0 or len(values) < 3:
        return values.copy()
    # small tolerance (3.6 ms) so that a sample exactly at the window edge is not lost to round-off
    half = 0.5 * window_hours + 1e-6
    lo = np.searchsorted(time_hours, time_hours - half, side='left')
    hi = np.searchsorted(time_hours, time_hours + half, side='right')
    median = np.nanmedian if skipna else np.median
    out = np.empty(len(values), dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)       # nanmedian of an all-NaN window
        for i in range(len(values)):
            out[i] = median(values[lo[i]:hi[i]])
    return out


def _time_axis(time_hours, n, time_resolution):
    """Sample times [hour] as a strictly increasing float array; a regular axis if none/invalid is given."""
    if time_hours is not None:
        t = np.asarray(time_hours, dtype=float)
        if t.shape == (n,) and np.all(np.isfinite(t)) and np.all(np.diff(t) > 0):
            return t
        logging.getLogger(__name__).warning(
            "time_hours is not a finite, strictly increasing array of the series length; "
            "using a regular time axis from the time resolution instead."
        )
    return np.arange(n) * time_resolution


def smooth_inputs(conv_area, conv_majoraxis, strat_area, time_hours, params):
    """
    The three input series of the staging exactly as classify_lifecycle thresholds them: NaN -> 0, then
    a centered running median over params['mcs_lifecycle_smooth_hour'] (default 0 = no smoothing).
    The plots of the staging use this function, so what they show is what the thresholds see.

    Args:
        conv_area, conv_majoraxis, strat_area: 1D np.array, one track's own time series.
        time_hours: 1D np.array, strictly increasing sample times [hour].
        params: dict, see classify_lifecycle.

    Returns:
        conv_area, conv_majoraxis, strat_area: 1D np.array(float), same length.
    """
    smooth_hour = params.get('mcs_lifecycle_smooth_hour', 0.0)
    out = []
    for series in (conv_area, conv_majoraxis, strat_area):
        series = np.nan_to_num(np.asarray(series, dtype=float), nan=0.0)
        out.append(running_median(series, time_hours, smooth_hour) if smooth_hour > 0 else series)
    return tuple(out)


def stratiform_floor(strat_area, params):
    """
    Stratiform area [km^2] that a time needs, together with an organized convective line, to count
    towards the mature stage: the mean (params['mcs_lifecycle_sf_stat'] 'mean', the default) or the median
    ('median') of the positive values of the track's (smoothed) stratiform area, at least
    params['mcs_lifecycle_sf_area_min'] (default 0).

    Args:
        strat_area: 1D np.array, as returned by smooth_inputs. NaN are ignored.
        params: dict, see classify_lifecycle.

    Returns:
        float
    """
    strat_area = np.asarray(strat_area, dtype=float)
    positive = strat_area[strat_area > 0]
    stat = params.get('mcs_lifecycle_sf_stat', 'mean')
    if stat not in ('mean', 'median'):
        raise ValueError(f"mcs_lifecycle_sf_stat '{stat}' is not 'mean' or 'median'")
    value = (float(np.mean(positive)) if stat == 'mean' else float(np.median(positive))) if positive.size > 0 else 0.0
    return max(value, params.get('mcs_lifecycle_sf_area_min', 0))


# ---------------------------------------------------------------------------
# Lifecycle staging
# ---------------------------------------------------------------------------

def _line_episodes(t, line_idx, gap_hour, time_resolution):
    """
    Groups of line samples: consecutive line samples are one episode if the time without a line sample between them (their spacing
    minus one step) is at most gap_hour (a quarter step of tolerance for the round-off of the time axis).

    Args:
        t: 1D np.array, strictly increasing sample times [hour].
        line_idx: 1D np.array(int), indices of the samples with the line.
        gap_hour: float, the longest interruption inside an episode [hour].
        time_resolution: float [hour], nominal time step.

    Returns:
        episodes: list of 1D np.array(int), the indices of each episode in time order (empty list without line samples).
    """
    if line_idx.size == 0:
        return []
    breaks = np.where(np.diff(t[line_idx]) > gap_hour + time_resolution + 0.25 * time_resolution)[0] + 1
    return [e for e in np.split(line_idx, breaks) if len(e) > 0]


def _false_runs(mask):
    """(first, last) index pairs of the runs of False in a bool array that have True on both sides."""
    runs, n, i = [], len(mask), 0
    while i < n:
        if mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and not mask[j + 1]:
            j += 1
        if i > 0 and j < n - 1:
            runs.append((i, j))
        i = j + 1
    return runs


def classify_lifecycle_single(conv_area, line_length, strat_area, time_resolution, params, time_hours=None):
    """
    Lifecycle staging of ONE track with one lifecycle per track (method 'single', see the module docstring for the rules).

    Stages: 1=cloud only, 2=unorganized convection, 3=genesis (once), 4=mature, 5=decay, 6=redeveloping (line back, not yet mature).

    Args:
        conv_area, line_length, strat_area: 1D np.array, this track's own time series (length = track_duration). line_length is the
            line series: conv_diameter_sum or conv_majoraxis (params['mcs_lifecycle_line_method'] names it; the caller passes it).
            NaN is treated as 0. For a track with merge/split fragments the caller passes total_conv_area, the line series of the
            merged mask and total_strat_area.
        time_resolution: float [hour], nominal time step of the data.
        params: dict with mcs_lifecycle_line_thresh [km], mcs_lifecycle_gap_hour (the longest stretch without line and without
            stratiform area at the floor that is bridged inside a mature segment; 0 = no bridging), mcs_lifecycle_sf_area_min,
            mcs_lifecycle_smooth_hour (default 0), mcs_lifecycle_sf_stat (default 'mean'); no lag keys are used.
        time_hours: 1D np.array, optional, sample times [hour]. Strictly increasing; irregular spacing and missing samples are fine.
            Default: a regular axis with step time_resolution.

    Returns:
        stage: 1D np.array(int16), length = track_duration.
        cycle: 1D np.array(int16), 0 before genesis, 1 from genesis, plus 1 at the start of every later mature segment.
        lifecycle_index: 1D np.array(int64), length 6, the first time INDEX of stages 1..6; -1 if never observed.
        ncycles: int, number of organized-line episodes (line samples separated by more than mcs_lifecycle_gap_hour).
        complete: bool, True if stages 3, 4 and 5 are all observed.
    """
    n = len(conv_area)
    stage = np.full(n, STAGE_CLOUD_ONLY, dtype=np.int16)
    cycle = np.zeros(n, dtype=np.int16)
    if n == 0:
        return stage, cycle, np.full(6, -1, dtype=np.int64), 0, False

    t = _time_axis(time_hours, n, time_resolution)
    conv_area, line, strat_area = smooth_inputs(conv_area, line_length, strat_area, t, params)
    stage[conv_area > 0] = STAGE_UNORGANIZED

    # The three conditions: the line, the stratiform area at the floor, and both together (the mature condition)
    gap_hour = params['mcs_lifecycle_gap_hour']
    on = line >= params['mcs_lifecycle_line_thresh']
    floor = stratiform_floor(strat_area, params)
    strat_ok = (strat_area > 0) & (strat_area >= floor)
    mature_ok = on & strat_ok
    on_idx = np.where(on)[0]
    ncycles = len(_line_episodes(t, on_idx, gap_hour, time_resolution))

    if on_idx.size > 0:
        # Genesis, once: the first line sample, kept while the line holds until the first later mature sample
        g = int(on_idx[0])
        candidates = np.where(mature_ok)[0]
        candidates = candidates[candidates > g]
        first_mature = int(candidates[0]) if candidates.size else n
        e = g
        while e + 1 < min(first_mature, n) and on[e + 1]:
            e += 1
        stage[g:e + 1] = STAGE_GENESIS
        cycle[g:] = 1

        if candidates.size == 0:
            # never mature: the line coming back after the genesis run is redeveloping, there is no decay
            after = np.arange(e + 1, n)
            stage[after[on[after]]] = STAGE_REDEVELOPING
        else:
            # Mature segments: the system is "present" where the line or the stratiform floor holds; a stretch without it that is
            # no longer than the gap is bridged, and each connected stretch that holds a mature sample is one segment
            present = on | strat_ok
            closed = present.copy()
            for i, j in _false_runs(present):
                if (t[j + 1] - t[i - 1]) - time_resolution <= gap_hour + 0.25 * time_resolution:
                    closed[i:j + 1] = True
            segments, i = [], 0
            while i < n:
                if not closed[i]:
                    i += 1
                    continue
                j = i
                while j + 1 < n and closed[j + 1]:
                    j += 1
                inside = candidates[(candidates >= i) & (candidates <= j)]
                if inside.size > 0:
                    segments.append((int(inside[0]), int(inside[-1])))
                i = j + 1
            # Decay after every segment (the line coming back before the next segment is redeveloping), mature on top
            for k, (first, last) in enumerate(segments):
                following = segments[k + 1][0] if k + 1 < len(segments) else n
                stage[last + 1:following] = STAGE_DECAY
                if k + 1 < len(segments):
                    between = np.arange(last + 1, following)
                    stage[between[on[between]]] = STAGE_REDEVELOPING
                stage[first:last + 1] = STAGE_MATURE
                if k > 0:
                    cycle[first:] = k + 1
            before = np.arange(e + 1, segments[0][0])
            stage[before[on[before]]] = STAGE_REDEVELOPING

    lifecycle_index = np.full(6, -1, dtype=np.int64)
    for s in range(1, 7):
        where_s = np.where(stage == s)[0]
        if where_s.size > 0:
            lifecycle_index[s - 1] = where_s[0]
    complete = bool(lifecycle_index[2] != -1 and lifecycle_index[3] != -1 and lifecycle_index[4] != -1)
    return stage, cycle, lifecycle_index, ncycles, complete


def classify_lifecycle(conv_area, conv_majoraxis, strat_area, time_resolution, params, time_hours=None):
    """
    Lifecycle staging of ONE track, from its own time series (already trimmed to its track_duration).

    params['mcs_lifecycle_method'] chooses the rules: 'single' (one lifecycle per track, classify_lifecycle_single, which also has
    stage 6 and a six-element lifecycle_index) or 'episodes' (this function's own rules below, the default of a params dict without
    the key, i.e. of the earlier configs). The second argument is the line series of either method (conv_majoraxis of the
    earlier configs, conv_diameter_sum with the current defaults: params['mcs_lifecycle_line_method']).

    Stages of the 'episodes' method: 1=cloud only, 2=unorganized convection, 3=genesis, 4=mature, 5=decay.

    Hardening vs. robustmcs_radar.py:
    - Every contiguous "organized convective line" episode
      (conv_majoraxis >= mcs_lifecycle_line_thresh, gaps up to
      mcs_lifecycle_gap_hour bridged) is staged independently, in time
      order, without overwriting earlier stage assignments - this replaces
      robustmcs_radar.py's approach of keeping only the single largest
      contiguous group, which silently drops re-intensification episodes
      (the "odd multi-cycle tracks" the lifecycle logic needed to be made
      robust against).
    - All constants are expressed in hours and compared with the sample times, instead of
      robustmcs_radar.py's hard-coded hourly-step constants (e.g. "+2"), which would silently
      misbehave at 5-minute (or any non-hourly) time resolution. A constant is rounded to a whole
      number of time_resolution steps (as before), then applied to the real time axis, so irregular
      spacing and missing samples shift the stage boundaries by at most the spacing.
    - With mcs_lifecycle_smooth_hour > 0 the three input series are first smoothed by a centered
      running median of that width, so single-sample spikes and short dropouts of conv_majoraxis,
      conv_area and strat_area no longer start or end an episode.
    - "Stage not observed" is -1 in lifecycle_index, distinct from a track
      not being evaluated at all (handled by the caller).
    - This function is always called and always returns a result (never
      raises/exits for a short or poorly-organized track) - the caller
      decides whether a track is long enough to bother evaluating.

    Args:
        conv_area, conv_majoraxis, strat_area: 1D np.array, this track's own
            time series (length = track_duration). NaN treated as 0/absent. For a track with merge/split
            fragments the caller passes total_conv_area, the convective length of the merged mask and
            total_strat_area.
        time_resolution: float [hour], nominal time step of the data.
        params: dict with mcs_lifecycle_line_thresh, mcs_lifecycle_gap_hour,
            mcs_lifecycle_genesis_lag_hour, mcs_lifecycle_mature_lag_hour,
            mcs_lifecycle_sf_area_min, and optionally mcs_lifecycle_smooth_hour (default 0 = no smoothing).
        time_hours: 1D np.array, optional, sample times [hour] (e.g. (base_time - base_time[0]) / 3600).
            Strictly increasing; irregular spacing and missing samples are fine. Default: a regular axis
            with step time_resolution.

    Returns:
        stage: 1D np.array(int16), length = track_duration.
        cycle: 1D np.array(int16), which organized-line episode (1-based) a
            time belongs to; 0 for cloud-only/unorganized-baseline times not
            part of any episode's genesis-through-decay span.
        lifecycle_index: 1D np.array(int64), length 5, the first time INDEX
            of stages 1..5 within this track's own trimmed array; -1 if that
            stage was never observed.
        ncycles: int, number of organized-line episodes found.
        complete: bool, True if stages 3, 4 and 5 are all observed.
    """
    if lifecycle_method(params) == 'single':
        return classify_lifecycle_single(conv_area, conv_majoraxis, strat_area, time_resolution, params, time_hours=time_hours)
    n = len(conv_area)
    stage = np.full(n, STAGE_CLOUD_ONLY, dtype=np.int16)
    cycle = np.zeros(n, dtype=np.int16)
    if n == 0:
        return stage, cycle, np.full(5, -1, dtype=np.int64), 0, False

    t = _time_axis(time_hours, n, time_resolution)
    conv_area, conv_majoraxis, strat_area = smooth_inputs(conv_area, conv_majoraxis, strat_area, t, params)

    C = conv_area > 0
    stage[C] = STAGE_UNORGANIZED

    Lf = conv_majoraxis >= params['mcs_lifecycle_line_thresh']
    sf_floor = stratiform_floor(strat_area, params)
    S = Lf & (strat_area > 0) & (strat_area >= sf_floor)

    # Constants in hours, rounded to whole steps of the nominal resolution (as before), then compared with
    # the real time axis; a quarter step of tolerance absorbs round-off of the time axis
    gap_h = _steps_from_hours(params['mcs_lifecycle_gap_hour'], time_resolution) * time_resolution
    genesis_lag_h = _lag_hours(params['mcs_lifecycle_genesis_lag_hour'], time_resolution)
    mature_lag_h = _lag_hours(params['mcs_lifecycle_mature_lag_hour'], time_resolution)
    tol = 0.25 * time_resolution

    # Line episodes: consecutive line samples belong to one episode if the time without a line sample
    # between them (their spacing minus one step) is at most gap_h
    episodes = _line_episodes(t, np.where(Lf)[0], gap_h, time_resolution)
    ncycles = len(episodes)

    for ic, episode in enumerate(episodes, start=1):
        a, b = int(episode[0]), int(episode[-1])
        # genesis: the first sample of the episode (the first time of an organized line), or, with a lag set, the first
        # sample at least genesis_lag_h after the line first appeared
        g = int(np.searchsorted(t, t[a] + genesis_lag_h - tol, side='left'))
        if g > b:
            # Episode too short for even genesis to complete (less than
            # mcs_lifecycle_genesis_lag_hour of organized line) - leave the
            # R0 baseline (cloud-only/unorganized) stage for this episode.
            continue

        # mature: from a later sample than genesis (and at least mature_lag_h after it, if a lag is set)
        mature_search_start = max(g + 1, int(np.searchsorted(t, t[g] + mature_lag_h - tol, side='left')))
        mature_candidates = np.where(S[mature_search_start:b + 1])[0]
        if mature_candidates.size > 0:
            s0 = mature_search_start + int(mature_candidates[0])
            s1 = g + int(np.where(S[g:b + 1])[0][-1])
            stage[g:s0] = STAGE_GENESIS
            stage[s0:s1 + 1] = STAGE_MATURE
            decay_start = s1 + 1
        else:
            stage[g:b + 1] = STAGE_GENESIS
            decay_start = b + 1

        decay_end = int(episodes[ic][0]) if ic < len(episodes) else n
        if decay_start < decay_end:
            stage[decay_start:decay_end] = STAGE_DECAY
        cycle[g:max(decay_end, g + 1)] = ic

    lifecycle_index = np.full(5, -1, dtype=np.int64)
    for s in range(1, 6):
        where_s = np.where(stage == s)[0]
        if where_s.size > 0:
            lifecycle_index[s - 1] = where_s[0]
    complete = bool(
        (lifecycle_index[2] != -1) and (lifecycle_index[3] != -1) and (lifecycle_index[4] != -1)
    )
    return stage, cycle, lifecycle_index, ncycles, complete
