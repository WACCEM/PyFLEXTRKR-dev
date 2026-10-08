"""
Lifecycle staging of tracked MCS (feature_type='radar_feature').

Hardened, multi-cycle-aware port of robustmcs_radar.py's Coniglio-et-al.-(2010) staging. It works on
the time series of ONE track and has no I/O and no dask import, so it is directly unit-testable with
synthetic series (tests/test_mcs_lifecycle_synthetic.py).

Stages: 1=cloud only, 2=unorganized convection, 3=genesis (convective line organizing, not yet
mature), 4=mature (organized line + developed stratiform area), 5=decay.

Every constant is a duration in hours and is compared with the real sample times, so the staging does
not depend on the time resolution of the data (5 min ... 1 h) or on a regular time spacing (the MRMS
steps alternate between about 4 and 6 min, and samples may be missing). An optional running median over
mcs_lifecycle_smooth_hour removes isolated spikes and short dropouts of the input series before the
thresholds are applied; the window is in hours as well, so it covers more samples in 5-min data and
none in hourly data.
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


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------

def _steps_from_hours(hours, time_resolution):
    """Convert an hour duration to a (minimum 1) number of time steps."""
    return max(1, int(round(hours / time_resolution)))


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
    towards the mature stage: the mean of the positive values of the track's (smoothed) stratiform area,
    at least params['mcs_lifecycle_sf_area_min'] (default 0).

    Args:
        strat_area: 1D np.array, as returned by smooth_inputs. NaN are ignored.
        params: dict, see classify_lifecycle.

    Returns:
        float
    """
    strat_area = np.asarray(strat_area, dtype=float)
    positive = strat_area[strat_area > 0]
    return max(float(np.mean(positive)) if positive.size > 0 else 0.0, params.get('mcs_lifecycle_sf_area_min', 0))


# ---------------------------------------------------------------------------
# Lifecycle staging
# ---------------------------------------------------------------------------

def classify_lifecycle(conv_area, conv_majoraxis, strat_area, time_resolution, params, time_hours=None):
    """
    Lifecycle staging of ONE track, from its own time series (already trimmed to its track_duration).

    Stages: 1=cloud only, 2=unorganized convection, 3=genesis, 4=mature, 5=decay.

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
    genesis_lag_h = _steps_from_hours(params['mcs_lifecycle_genesis_lag_hour'], time_resolution) * time_resolution
    mature_lag_h = _steps_from_hours(params['mcs_lifecycle_mature_lag_hour'], time_resolution) * time_resolution
    tol = 0.25 * time_resolution

    # Line episodes: consecutive line samples belong to one episode if the time without a line sample
    # between them (their spacing minus one step) is at most gap_h
    line_idx = np.where(Lf)[0]
    if line_idx.size > 0:
        breaks = np.where(np.diff(t[line_idx]) > gap_h + time_resolution + tol)[0] + 1
        episodes = [e for e in np.split(line_idx, breaks) if len(e) > 0]
    else:
        episodes = []
    ncycles = len(episodes)

    for ic, episode in enumerate(episodes, start=1):
        a, b = int(episode[0]), int(episode[-1])
        # first sample at least genesis_lag_h after the line first appeared
        g = int(np.searchsorted(t, t[a] + genesis_lag_h - tol, side='left'))
        if g > b:
            # Episode too short for even genesis to complete (less than
            # mcs_lifecycle_genesis_lag_hour of organized line) - leave the
            # R0 baseline (cloud-only/unorganized) stage for this episode.
            continue

        mature_search_start = int(np.searchsorted(t, t[g] + mature_lag_h - tol, side='left'))
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
