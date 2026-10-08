"""
Synthetic MCS series with a known lifecycle, to test and benchmark pyflextrkr.mcs_lifecycle.classify_lifecycle.

- CASES: smooth noise-free series (convective area, convective length, stratiform area) of an idealized
  single-cycle MCS, a two-cycle MCS and four incomplete ones (no stratiform area, line too short for genesis,
  track truncated before decay, never organized).
- truth(): the stage onsets of a case, from the frozen pre-hardening classifier (classify_lifecycle_v1) run
  on the dense 1-minute noise-free series.
- sample_case(): the same case sampled at 5/15/30/60 minutes or at the irregular 4/6-minute MRMS cadence,
  with log-normal noise, isolated spikes and dropped samples.
- run_case(): scores a classifier on many noisy realizations against the truth.

classify_lifecycle_v1 is a verbatim copy of classify_lifecycle as it was before it became time-based
(index steps, no smoothing); it is the regression reference and the baseline of
Analysis/mcs_radar_lifecycle_robustness.py. Do not edit it.
"""
import numpy as np

# Production defaults of the staging (config mcs_lifecycle_*)
PARAMS = {
    'mcs_lifecycle_line_thresh': 100.0,       # km
    'mcs_lifecycle_gap_hour': 1.0,
    'mcs_lifecycle_genesis_lag_hour': 1.0,
    'mcs_lifecycle_mature_lag_hour': 2.0,
    'mcs_lifecycle_sf_area_min': 0,
}

# ---------------------------------------------------------------------------
# Frozen copy of the pre-hardening classifier (index steps, no smoothing)
# ---------------------------------------------------------------------------

STAGE_CLOUD_ONLY, STAGE_UNORGANIZED, STAGE_GENESIS, STAGE_MATURE, STAGE_DECAY = 1, 2, 3, 4, 5

def _steps_from_hours_v1(hours, time_resolution):
    """Convert an hour duration to a (minimum 1) number of time steps."""
    return max(1, int(round(hours / time_resolution)))


def _group_contiguous_indices_v1(idx, max_gap_steps):
    """
    Group sorted integer indices into contiguous runs, bridging up to
    max_gap_steps missing intermediate indices.

    Args:
        idx: 1D np.array(int), sorted ascending.
        max_gap_steps: int
            Maximum number of missing intermediate indices still bridged
            into the same group (0 = indices must be perfectly consecutive).

    Returns:
        list of 1D np.array(int): the groups, in order, each non-empty.
    """
    if len(idx) == 0:
        return []
    breaks = np.where(np.diff(idx) > max_gap_steps + 1)[0] + 1
    return [g for g in np.split(idx, breaks) if len(g) > 0]


def classify_lifecycle_v1(conv_area, conv_majoraxis, strat_area, time_resolution, params):
    """
    Hardened, multi-cycle-aware port of robustmcs_radar.py's Coniglio-et-al.
    -2010 lifecycle staging, for ONE track's own time series (already
    trimmed to its own track_duration).

    Stages: 1=cloud only, 2=unorganized convection, 3=genesis (line
    organizing, not yet mature), 4=mature (organized line + developed
    stratiform area), 5=decay.

    Hardening vs. robustmcs_radar.py:
    - Every contiguous "organized convective line" episode
      (conv_majoraxis >= mcs_lifecycle_line_thresh, gaps up to
      mcs_lifecycle_gap_hour bridged) is staged independently, in time
      order, without overwriting earlier stage assignments - this replaces
      robustmcs_radar.py's approach of keeping only the single largest
      contiguous group, which silently drops re-intensification episodes
      (the "odd multi-cycle tracks" the lifecycle logic needed to be made
      robust against).
    - All constants are expressed in hours and converted to time steps via
      time_resolution, instead of robustmcs_radar.py's hard-coded hourly-
      step constants (e.g. "+2"), which would silently misbehave at
      5-minute (or any non-hourly) time resolution.
    - "Stage not observed" is -1 in lifecycle_index, distinct from a track
      not being evaluated at all (handled by the caller).
    - This function is always called and always returns a result (never
      raises/exits for a short or poorly-organized track) - the caller
      decides whether a track is long enough to bother evaluating.

    Args:
        conv_area, conv_majoraxis, strat_area: 1D np.array, this track's own
            time series (length = track_duration). NaN treated as 0/absent.
        time_resolution: float [hour]
        params: dict with mcs_lifecycle_line_thresh, mcs_lifecycle_gap_hour,
            mcs_lifecycle_genesis_lag_hour, mcs_lifecycle_mature_lag_hour,
            mcs_lifecycle_sf_area_min.

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
    conv_area = np.nan_to_num(np.asarray(conv_area, dtype=float), nan=0.0)
    conv_majoraxis = np.nan_to_num(np.asarray(conv_majoraxis, dtype=float), nan=0.0)
    strat_area = np.nan_to_num(np.asarray(strat_area, dtype=float), nan=0.0)

    stage = np.full(n, STAGE_CLOUD_ONLY, dtype=np.int16)
    cycle = np.zeros(n, dtype=np.int16)
    if n == 0:
        return stage, cycle, np.full(5, -1, dtype=np.int64), 0, False

    C = conv_area > 0
    stage[C] = STAGE_UNORGANIZED

    Lf = conv_majoraxis >= params['mcs_lifecycle_line_thresh']
    sf_present = strat_area[strat_area > 0]
    sf_floor = max(
        float(np.mean(sf_present)) if sf_present.size > 0 else 0.0,
        params.get('mcs_lifecycle_sf_area_min', 0),
    )
    S = Lf & (strat_area > 0) & (strat_area >= sf_floor)

    gap_steps = _steps_from_hours_v1(params['mcs_lifecycle_gap_hour'], time_resolution)
    genesis_lag_steps = _steps_from_hours_v1(params['mcs_lifecycle_genesis_lag_hour'], time_resolution)
    mature_lag_steps = _steps_from_hours_v1(params['mcs_lifecycle_mature_lag_hour'], time_resolution)

    line_idx = np.where(Lf)[0]
    episodes = _group_contiguous_indices_v1(line_idx, gap_steps)
    ncycles = len(episodes)

    for ic, episode in enumerate(episodes, start=1):
        a, b = int(episode[0]), int(episode[-1])
        g = a + genesis_lag_steps
        if g > b:
            # Episode too short for even genesis to complete (less than
            # mcs_lifecycle_genesis_lag_hour of organized line) - leave the
            # R0 baseline (cloud-only/unorganized) stage for this episode.
            continue

        mature_search_start = g + mature_lag_steps
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


# ---------------------------------------------------------------------------
# Noise-free series with known truth
# ---------------------------------------------------------------------------

LINE_PEAK_KM = 220.0      # convective length at the peak of an organized line
STRAT_PEAK_KM2 = 25000.0  # stratiform area at the peak of a stratiform bump

# duration T [h]; organized-line episodes (on, off) [h]; stratiform bumps (rise start, rise end, fall start, fall end) [h]
CASES = {
    'textbook':        dict(T=12.0, lines=[(2.0, 9.5)], strat=[(3.0, 5.5, 8.0, 11.0)]),
    'two_cycles':      dict(T=15.0, lines=[(1.5, 6.0), (9.0, 14.0)],
                            strat=[(2.5, 4.5, 5.0, 7.5), (10.0, 12.0, 12.5, 14.8)]),
    'missing_mature':  dict(T=10.0, lines=[(2.0, 8.0)], strat=[]),
    'short_episode':   dict(T=8.0, lines=[(3.0, 3.8)], strat=[]),
    'truncated':       dict(T=7.0, lines=[(2.0, 12.0)], strat=[(3.0, 5.0, 12.0, 13.0)]),
    'never_organized': dict(T=10.0, lines=[], strat=[]),
}

# Sample-level disturbances: log-normal sigma, probability of an isolated spike (per sample and variable),
# probability of a dropped sample, length of one dropped block [h]
NOISE = {
    'clean':    dict(sigma=0.0, p_spike=0.0, p_drop=0.0, block_h=0.0),
    'moderate': dict(sigma=0.15, p_spike=0.02, p_drop=0.05, block_h=0.0),
    'severe':   dict(sigma=0.30, p_spike=0.05, p_drop=0.10, block_h=0.5),
}

# Sampling: regular steps in minutes, or the alternating 4/6-minute cadence of the MRMS data
RESOLUTIONS = [5, 15, 30, 60, 'irregular']


def _ramp(t, t0, t1):
    """Smooth 0 -> 1 between t0 and t1."""
    x = np.clip((np.asarray(t, dtype=float) - t0) / (t1 - t0), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def true_series(case, t):
    """Noise-free (conv_area [km2], conv_majoraxis [km], strat_area [km2]) of a case at times t [h]."""
    c = CASES[case]
    t = np.asarray(t, dtype=float)
    w = np.zeros_like(t)                      # 1 inside an organized-line episode
    for on, off in c['lines']:
        w = np.maximum(w, _ramp(t, on - 0.25, on + 0.25) * (1.0 - _ramp(t, off - 0.25, off + 0.25)))
    length = 45.0 + 15.0 * np.sin(2 * np.pi * t / 3.7) + (LINE_PEAK_KM - 45.0) * w      # 30-60 km without a line
    conv = 800.0 + 2400.0 * w + 150.0 * np.sin(2 * np.pi * t / 2.3)                      # always > 0
    strat = np.zeros_like(t)
    for r0, r1, f0, f1 in c['strat']:
        strat += STRAT_PEAK_KM2 * _ramp(t, r0, r1) * (1.0 - _ramp(t, f0, f1))
    return conv, length, strat


def nominal_step_hour(resolution):
    """Nominal time step [h] of a sampling (the irregular cadence averages 5 minutes)."""
    return (5 if resolution == 'irregular' else resolution) / 60.0


def truth(case, params=PARAMS):
    """
    Stage onsets of a case from the frozen classifier on the dense (1-minute) noise-free series.

    Returns:
        onset: np.array(3), time [h] of the first genesis, mature and decay sample; NaN = stage not observed.
        ncycles: int, number of organized-line episodes.
    """
    onset, ncycles, _, _ = truth_stages(case, params)
    return onset, ncycles


def truth_stages(case, params=PARAMS):
    """truth() plus the dense 1-minute time axis [h] and stage series of the case."""
    t = np.arange(0.0, CASES[case]['T'], 1.0 / 60.0)
    conv, length, strat = true_series(case, t)
    stage, _, idx, ncycles, _ = classify_lifecycle_v1(conv, length, strat, 1.0 / 60.0, params)
    onset = np.where(idx[2:5] >= 0, t[np.maximum(idx[2:5], 0)], np.nan)
    return onset, ncycles, t, stage


def sample_times(case, resolution, rng):
    """Sample times [h] of a case at a resolution (minutes, or 'irregular' for alternating 4/6-min steps)."""
    T = CASES[case]['T']
    if resolution == 'irregular':
        steps = np.where(np.arange(int(T * 12) + 4) % 2 == rng.integers(2), 4.0, 6.0) / 60.0
        t = np.concatenate([[0.0], np.cumsum(steps)])
        return t[t < T]
    return np.arange(0.0, T, resolution / 60.0)


def sample_case(case, resolution, noise, rng):
    """
    One noisy realization of a case.

    Returns:
        t [h], conv_area, conv_majoraxis, strat_area (1D arrays; dropped samples are removed).
    """
    k = NOISE[noise]
    t = sample_times(case, resolution, rng)
    conv, length, strat = true_series(case, t)
    n = len(t)
    if k['sigma'] > 0:
        conv = conv * np.exp(k['sigma'] * rng.standard_normal(n))
        length = length * np.exp(k['sigma'] * rng.standard_normal(n))
        strat = strat * np.exp(k['sigma'] * rng.standard_normal(n))
    if k['p_spike'] > 0:
        spike = rng.random(n) < k['p_spike']
        length[spike] = rng.uniform(0.0, 300.0, int(spike.sum()))
        spike = rng.random(n) < k['p_spike']
        strat[spike] *= rng.uniform(0.1, 3.0, int(spike.sum()))
    keep = rng.random(n) >= k['p_drop']
    if k['block_h'] > 0:
        start = rng.uniform(0.0, max(CASES[case]['T'] - k['block_h'], 0.0))
        keep &= ~((t >= start) & (t < start + k['block_h']))
    return t[keep], conv[keep], length[keep], strat[keep]


def run_case(classify, case, resolution, noise, n_real, seed, params=PARAMS):
    """
    Score a classifier on n_real noisy realizations of a case.

    Args:
        classify: callable(conv, length, strat, dt_hour, params, t_hours) -> classify_lifecycle's 5-tuple.
        case, resolution, noise: keys of CASES, RESOLUTIONS, NOISE.
        n_real: int, number of realizations. seed: int.

    Returns:
        dict with arrays over realizations:
            err (n_real, 3): onset error [h] (estimate - truth) of genesis/mature/decay; 0 where the stage is
                absent in both, +-inf where it is invented / missed
            invented, missed (n_real, 3) bool: stage observed only in the estimate / only in the truth
            ncycles_ok (n_real,) bool
            agreement (n_real,): fraction of the samples whose stage equals the truth stage at that time
            tol: float, tolerance [h] = max(1 h, 2 samples)
    """
    true_onset, true_ncycles, t_dense, stage_dense = truth_stages(case, params)
    rng = np.random.default_rng(seed)
    dt = nominal_step_hour(resolution)
    err = np.zeros((n_real, 3))
    ncycles_ok = np.zeros(n_real, dtype=bool)
    agreement = np.zeros(n_real)
    for i in range(n_real):
        t, conv, length, strat = sample_case(case, resolution, noise, rng)
        stage, _, idx, ncycles, _ = classify(conv, length, strat, dt, params, t)
        agreement[i] = np.mean(stage == stage_dense[np.minimum(np.round(t * 60.0).astype(int), len(t_dense) - 1)])
        est = np.where(idx[2:5] >= 0, t[np.maximum(idx[2:5], 0)], np.nan)
        both = ~np.isnan(est) & ~np.isnan(true_onset)
        err[i, both] = est[both] - true_onset[both]
        err[i, np.isnan(est) & ~np.isnan(true_onset)] = -np.inf
        err[i, ~np.isnan(est) & np.isnan(true_onset)] = np.inf
        ncycles_ok[i] = ncycles == true_ncycles
    return dict(
        err=err,
        invented=np.isposinf(err),
        missed=np.isneginf(err),
        ncycles_ok=ncycles_ok,
        agreement=agreement,
        tol=max(1.0, 2.0 * dt),
    )


def pass_rate(result):
    """Fraction of realizations with all three onsets within tol (and none invented/missed) and the right cycle count."""
    ok = np.all(np.abs(result['err']) <= result['tol'], axis=1) & result['ncycles_ok']
    return float(ok.mean())
