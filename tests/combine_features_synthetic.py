"""
Synthetic radar reflectivity frames for testing combine_features_with_share_boundary (numpy only).

The storms mimic radar echo on a 3-km grid: intense convective cores (45-55 dBZ) inside a stratiform shield
(about 20 dBZ) that falls off to 0-10 dBZ at the edge of the storm. Fields combine by the pixel-wise maximum. Three
kinds of storms:
  - "oval": a large oval storm with 4-6 cores, the shape that detect-and-spread cuts into several pieces,
  - "pair": two round storms whose edges overlap over a short neck (barely touching, must stay separate),
  - "single": an isolated storm with one core.
The 10 dBZ contour is the outline of a storm (shield ramp: 20 dBZ inside 0.65, 10 dBZ at 0.9, 0 dBZ at 1.15 of the
semi-axes). Used by test_combine_features.py and plot_combine_features_synthetic.py.
"""
import numpy as np

# Grid and field
NY, NX = 300, 400
DX_KM = 3.0
BACKGROUND_DBZ = -10.0     # no echo (clean: the noise is added only where there is echo)
NOISE_DBZ = 1.0
# Shield: 20 dBZ inside SHIELD_FLAT (in units of the semi-axes), linear ramp to 0 dBZ at SHIELD_END
SHIELD_DBZ = 20.0
SHIELD_FLAT = 0.65
SHIELD_END = 1.15
# Flat-topped convective core: peak * exp(-(d / radius)^4), radius in pixels
CORE_RADIUS_PX = 5.5


def storm_field(shape, center, semi_axes, angle, cores, rng, irregularity=0.12):
    """
    Reflectivity of one storm: an oval shield with an irregular edge plus convective cores.

    Args:
        shape: tuple (ny, nx)
            Grid shape.
        center: tuple (cy, cx)
            Storm center [pixels].
        semi_axes: tuple (a, b)
            Semi-major and semi-minor axes of the 10 dBZ outline [pixels].
        angle: float
            Rotation of the major axis from the x axis [radians].
        cores: list of (dy, dx, peak_dbz)
            Cores: offset from the center [pixels] and peak reflectivity [dBZ].
        rng: np.random.Generator
            Random numbers for the irregular edge.
        irregularity: float (optional, default=0.12)
            Amplitude of the lumpiness of the edge (fraction of the semi-axes).

    Returns:
        field: np.ndarray (ny, nx)
            Reflectivity of the storm [dBZ] (BACKGROUND_DBZ outside).
    """
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    dy, dx = yy - center[0], xx - center[1]
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    along = dx * cos_a + dy * sin_a
    across = -dx * sin_a + dy * cos_a
    # Normalized elliptical distance, scaled by a smooth random function of the angle (lumpy edge)
    theta = np.arctan2(across / semi_axes[1], along / semi_axes[0])
    lumpy = np.ones_like(theta)
    for harmonic in (2, 3, 4):
        lumpy += irregularity / harmonic * np.cos(harmonic * theta + rng.uniform(0.0, 2.0 * np.pi))
    r = np.hypot(along / semi_axes[0], across / semi_axes[1]) / lumpy
    # Stratiform shield and the cores, combined by the maximum
    shield = SHIELD_DBZ * np.clip((SHIELD_END - r) / (SHIELD_END - SHIELD_FLAT), 0.0, 1.0)
    field = np.where(shield > 0, shield, BACKGROUND_DBZ)
    for (cdy, cdx, peak) in cores:
        d = np.hypot(yy - (center[0] + cdy), xx - (center[1] + cdx))
        core = peak * np.exp(-(d / CORE_RADIUS_PX) ** 4)
        field = np.maximum(field, np.where(core >= 1.0, core, BACKGROUND_DBZ))      # no tail of the core below 1 dBZ
    return field


def make_frame(seed, kind="cluster", shape=(NY, NX)):
    """
    One synthetic reflectivity frame.

    Args:
        seed: int
            Random seed (the frame is reproducible).
        kind: string (optional, default="cluster")
            "cluster": an oval storm with 4-6 cores, a pair of barely touching storms and two single storms.
            "spread": five single storms far apart (a baseline where nothing should be combined).
        shape: tuple (optional, default=(NY, NX))
            Grid shape.

    Returns:
        dbz: np.ndarray (ny, nx)
            Reflectivity [dBZ], with noise, clipped to -10..70.
        storms: list of dictionaries
            Truth for every storm: name, kind, ncores, mask (the pixels where its own field is >= 10 dBZ).
    """
    rng = np.random.default_rng(seed)
    ny, nx = shape
    fields, storms = [], []

    def add(name, kind_, center, semi_axes, angle, cores):
        field = storm_field(shape, center, semi_axes, angle, cores, rng)
        fields.append(field)
        storms.append({"name": name, "kind": kind_, "ncores": len(cores), "mask": field >= 10.0})

    if kind == "cluster":
        # Oval storm, 4-6 cores along the major axis (>= 20 px apart: the 10-px smoothing keeps them separate)
        a, b = 58.0, 30.0
        ncores = int(rng.integers(4, 7))
        along = np.linspace(-0.62, 0.62, ncores) * a
        cores = [(0.0, al, rng.uniform(45.0, 55.0)) for al in along]
        # Alternate the cores across the storm so that the cuts between them are diagonal
        cores = [(cy + (-1) ** i * rng.uniform(5.0, 9.0), cx, pk) for i, (cy, cx, pk) in enumerate(cores)]
        add("oval", "oval", (ny * 0.30, nx * 0.30), (a, b), np.deg2rad(rng.uniform(-20.0, 20.0)), cores)
        # Two round storms whose 10 dBZ outlines overlap over a neck of about 9 % of their perimeter
        radius = 16.0
        dist = 2.0 * radius * 0.84
        cy, cx = ny * 0.70, nx * 0.30
        add("pair_west", "pair", (cy, cx - dist / 2), (radius, radius), 0.0, [(0.0, 0.0, 50.0)])
        add("pair_east", "pair", (cy, cx + dist / 2), (radius, radius), 0.0, [(0.0, 0.0, 50.0)])
        # Single storms
        add("single_1", "single", (ny * 0.30, nx * 0.78), (24.0, 17.0), 0.5, [(0.0, 0.0, 52.0)])
        add("single_2", "single", (ny * 0.72, nx * 0.74), (20.0, 20.0), 0.0, [(0.0, 0.0, 48.0)])
    elif kind == "spread":
        centers = [(0.25, 0.2), (0.25, 0.7), (0.5, 0.45), (0.75, 0.2), (0.75, 0.72)]
        for i, (fy, fx) in enumerate(centers):
            semi = (rng.uniform(18.0, 26.0), rng.uniform(15.0, 20.0))
            add(f"single_{i + 1}", "single", (ny * fy, nx * fx), semi, rng.uniform(0.0, np.pi), [(0.0, 0.0, rng.uniform(45.0, 55.0))])
    else:
        raise ValueError(f"kind must be 'cluster' or 'spread', got '{kind}'")

    # Combine by the maximum, then noise where there is echo
    dbz = np.maximum.reduce(fields)
    noise = rng.normal(0.0, NOISE_DBZ, size=dbz.shape)
    dbz = np.clip(np.where(dbz > BACKGROUND_DBZ, dbz + noise, dbz), -10.0, 70.0)
    return dbz, storms
