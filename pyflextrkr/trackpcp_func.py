"""
Worker functions of the track post-processing steps (trackpcp_driver.py): measure a gridded variable (e.g. hourly
precipitation) inside tracked features whose masks come at a higher time resolution than the variable.

Definitions (see trackpcp_driver.py for how windows are made)

  window   the track times that one data record represents, e.g. the 12 five-minute steps of the hour an accumulation covers
  swath    union over the steps of a window of a track's mask (cloudtracknumber, merge/split fragments included)
  owner    a pixel claimed by more than one track in a window belongs to one track only: the best class (class_rank), then
           the track that covers the pixel in most steps, then the lowest track number. All results of the three steps
           use this, so per-track sums add up to the class maps and never exceed the total
  convective swath   pixels of the swath that are convective (convective_value of the convective variable) in at least one step
                     of the window, inside the track's mask at that step. All data in it is convective, the rest of the swath
                     is stratiform

The functions take a task (small, one window or a block of windows) and a static dictionary (read-only data sent once to
every worker) and return small arrays, so they can run serially or on Dask workers unchanged.
"""
import numpy as np
import xarray as xr
from math import pi
from scipy import ndimage
from scipy.stats import skew
from skimage.measure import regionprops


PF_NAMES = [
    "pf_npf", "pf_lon", "pf_lat", "pf_area", "pf_rainrate", "pf_maxrainrate", "pf_skewness", "pf_majoraxis",
    "pf_minoraxis", "pf_aspectratio", "pf_orientation", "pf_eccentricity", "pf_perimeter", "pf_lon_centroid",
    "pf_lat_centroid", "pf_lon_weightedcentroid", "pf_lat_weightedcentroid", "pf_lon_maxrainrate", "pf_lat_maxrainrate",
    "pf_accumrain", "pf_accumrainheavy", "pf_ccarea", "pf_sfarea", "pf_ccrainrate", "pf_sfrainrate", "pf_ccrainamount",
    "pf_sfrainamount",
]


########################################################################################################################
# Reading and ownership
########################################################################################################################
def read_pixel_file(filename, mask_varname, conv_varname=None, cov_varname=None):
    """Track-number mask (int32, 0 = none), optional convective variable and optional data-coverage mask (bool: the
    coverage variable, e.g. the radar reflectivity, has a value) of one pixel-level file."""
    with xr.open_dataset(filename, decode_times=False, mask_and_scale=False) as ds:
        mask = np.asarray(ds[mask_varname].values).squeeze()
        conv = np.asarray(ds[conv_varname].values).squeeze() if conv_varname else None
        cov = np.isfinite(np.asarray(ds[cov_varname].values).squeeze()) if cov_varname else None
    mask = np.where(np.isfinite(mask) & (mask > 0), mask, 0).astype(np.int32)
    return mask, conv, cov


def class_luts(nlut, tracks, class_at, static):
    """
    Class value, selection and priority rank of every track number for one window.

    class_at: class value of the tracks in `tracks` (1-D class variable: the static table is used)
    Returns (class_lut int16 or None, select_lut bool, rank_lut int32).
    """
    class_values = static.get("class_values")           # None = every tracked feature
    cls = static.get("class_lut")
    if class_at is not None:
        cls = np.zeros(nlut, dtype=np.int16)
        cls[np.asarray(tracks, dtype=np.int64)] = class_at
    if cls is None or class_values is None:
        select = np.ones(nlut, dtype=bool)
        select[0] = False
    else:
        select = np.isin(cls, np.asarray(class_values)) & (np.arange(nlut) > 0)
    rank = np.zeros(nlut, dtype=np.int32)
    if cls is not None:
        order = static.get("class_priority") or sorted(set(np.unique(cls).tolist()))
        rank[:] = len(order)
        for r, v in enumerate(order):
            rank[cls == v] = r
    return cls, select, rank


def window_owner(masks, convs, select_lut, rank_lut, conv_value=4):
    """
    One owner per pixel for a window of K track-number masks.

    Args:
        masks: list of K 2-D int arrays (0 = no track).
        convs: list of K 2-D arrays of the convective variable, or None.
        select_lut, rank_lut: arrays over track numbers (selected tracks; priority rank, lower = better).
        conv_value: value of the convective variable that is convective.

    Returns:
        dict with pix (int64 flat index), trk (int32), conv (bool: in the convective swath), nsteps (int: steps in which
        the owner covers the pixel), n_pixels_contested (pixels claimed by more than one track),
        track_nsteps: (tracks, number of steps with a mask pixel) of every selected track present in any step.
    """
    nlut = select_lut.size
    if len(masks) == 0:                                  # a data record with no track step in its window
        return {"track_nsteps": (np.zeros(0, np.int32), np.zeros(0, np.int32)), "n_pixels_contested": 0,
                "pix": np.zeros(0, np.int64), "trk": np.zeros(0, np.int32), "conv": np.zeros(0, bool),
                "nsteps": np.zeros(0, np.int16)}
    npix = masks[0].size
    pix_l, trk_l, cv_l, present = [], [], [], []
    for k, m in enumerate(masks):
        flat = m.ravel()
        idx = np.flatnonzero((flat > 0) & (flat < nlut))
        t = flat[idx]
        keep = select_lut[t]
        idx, t = idx[keep], t[keep]
        pix_l.append(idx.astype(np.int64))
        trk_l.append(t.astype(np.int32))
        cv_l.append((convs[k].ravel()[idx] == conv_value) if convs is not None else np.zeros(idx.size, dtype=bool))
        present.append(np.unique(t))
    tr, cnt = np.unique(np.concatenate(present), return_counts=True) if present else (np.zeros(0, np.int32), np.zeros(0, int))
    out = {"track_nsteps": (tr.astype(np.int32), cnt.astype(np.int32)), "n_pixels_contested": 0}
    if len(masks) == 1:
        out.update(pix=pix_l[0], trk=trk_l[0], conv=cv_l[0], nsteps=np.ones(pix_l[0].size, dtype=np.int16))
        return out
    pix, trk, cv = np.concatenate(pix_l), np.concatenate(trk_l), np.concatenate(cv_l)
    if pix.size == 0:
        out.update(pix=pix, trk=trk, conv=cv, nsteps=np.zeros(0, dtype=np.int16))
        return out
    key = trk.astype(np.int64) * npix + pix
    uniq, inv = np.unique(key, return_inverse=True)
    inv = inv.reshape(-1)
    counts = np.bincount(inv)
    convany = np.bincount(inv, weights=cv.astype(float)) > 0
    pu = uniq % npix
    tu = (uniq // npix).astype(np.int32)
    order = np.lexsort((tu, -counts, rank_lut[tu], pu))
    pu_o = pu[order]
    first = np.r_[True, pu_o[1:] != pu_o[:-1]]
    pick = order[first]
    group_size = np.diff(np.flatnonzero(np.r_[first, True]))
    out.update(pix=pu[pick], trk=tu[pick], conv=convany[pick], nsteps=counts[pick].astype(np.int16),
               n_pixels_contested=int(np.count_nonzero(group_size > 1)))
    return out


########################################################################################################################
# Statistics of a variable over the swath of every track
########################################################################################################################
def parse_stat(name):
    """'mean' -> ('mean', None); 'p95' -> ('percentile', 95.0)"""
    if len(name) > 1 and name[0] == "p" and name[1:].replace(".", "", 1).isdigit():
        return "percentile", float(name[1:])
    if name in ("count", "area", "mean", "median", "min", "max", "sum", "std"):
        return name, None
    raise ValueError(f"unknown statistic '{name}': use count, area, mean, median, min, max, sum, std or pNN (e.g. p95)")


def group_by_track(trk, pix, conv):
    """Owner entries sorted by track; returns trk_s, pix_s, conv_s, starts, ends."""
    order = np.argsort(trk, kind="stable")
    trk_s, pix_s, conv_s = trk[order], pix[order], conv[order]
    if trk_s.size == 0:
        z = np.zeros(0, dtype=np.int64)
        return trk_s, pix_s, conv_s, z, z
    starts = np.flatnonzero(np.r_[True, trk_s[1:] != trk_s[:-1]])
    ends = np.r_[starts[1:], trk_s.size]
    return trk_s, pix_s, conv_s, starts, ends


def reduce_by_track(trk_s, pix_s, conv_s, starts, ends, field_flat, area, stats, have_conv):
    """
    Statistics of field over the swath pixels of each track, for the whole swath ("") and, if have_conv, for its
    convective ("conv_") and stratiform ("strat_") parts.

    Returns dict {(subset, stat): array over the groups}; stat 'count' is the number of valid (finite) values and 'area' the
    area of the part of the swath [km2] (finite or not).
    """
    n = starts.size
    vals = field_flat[pix_s].astype(np.float64)
    area_pix = area[pix_s] if np.ndim(area) else np.full(vals.shape, float(area))
    valid = np.isfinite(vals)
    parsed = [(s, *parse_stat(s)) for s in stats]
    out = {}
    subsets = {"": np.ones(vals.shape, dtype=bool)}
    if have_conv:
        subsets["conv_"] = conv_s
        subsets["strat_"] = ~conv_s
    need_pct = any(kind in ("median", "percentile") for _, kind, _ in parsed)
    for sub, sel in subsets.items():
        m = sel & valid
        cnt = np.add.reduceat(m.astype(np.int64), starts) if n else np.zeros(0, dtype=np.int64)
        v0 = np.where(m, vals, 0.0)
        total = np.add.reduceat(v0, starts) if n else np.zeros(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(cnt > 0, total / np.maximum(cnt, 1), np.nan)
        for name, kind, q in parsed:
            if kind == "count":
                out[(sub, name)] = cnt.astype(np.float32)
            elif kind == "area":
                out[(sub, name)] = (np.add.reduceat(np.where(sel, area_pix, 0.0), starts) if n else np.zeros(0)).astype(np.float32)
            elif kind == "sum":
                out[(sub, name)] = np.where(cnt > 0, total, np.nan).astype(np.float32)
            elif kind == "mean":
                out[(sub, name)] = mean.astype(np.float32)
            elif kind == "max":
                r = np.maximum.reduceat(np.where(m, vals, -np.inf), starts) if n else np.zeros(0)
                out[(sub, name)] = np.where(cnt > 0, r, np.nan).astype(np.float32)
            elif kind == "min":
                r = np.minimum.reduceat(np.where(m, vals, np.inf), starts) if n else np.zeros(0)
                out[(sub, name)] = np.where(cnt > 0, r, np.nan).astype(np.float32)
            elif kind == "std":
                sq = np.add.reduceat(np.where(m, vals * vals, 0.0), starts) if n else np.zeros(0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    var = np.where(cnt > 0, sq / np.maximum(cnt, 1) - mean ** 2, np.nan)
                out[(sub, name)] = np.sqrt(np.maximum(var, 0.0)).astype(np.float32)
        if need_pct:
            for name, kind, q in parsed:
                if kind in ("median", "percentile"):
                    out[(sub, name)] = np.full(n, np.nan, dtype=np.float32)
            for j in range(n):
                v = vals[starts[j]:ends[j]][m[starts[j]:ends[j]]]
                if v.size == 0:
                    continue
                for name, kind, q in parsed:
                    if kind == "median":
                        out[(sub, name)][j] = np.median(v)
                    elif kind == "percentile":
                        out[(sub, name)][j] = np.percentile(v, q)
    return out


def track_patches(pix_s, starts, ends, nx):
    """Yield (j, y0, x0, iy, ix) with the bounding box origin and the local indices of the pixels of group j."""
    for j in range(starts.size):
        p = pix_s[starts[j]:ends[j]]
        iy, ix = p // nx, p % nx
        y0, x0 = int(iy.min()), int(ix.min())
        yield j, y0, x0, iy - y0, ix - x0


def precip_features(rain, swath, conv, y0, x0, lon, lat, pixel_area, pixel_length, rr_thresh, heavy_thresh, nmaxpf,
                    min_npix):
    """
    Precipitation features (PFs) of one track in one window: connected pixels of the swath with rain rate above
    rr_thresh, largest first. rain, swath, conv are patches (bounding box of the swath, origin y0, x0 in the grid).
    Returns dict of arrays (nmaxpf,) with the names of PF_NAMES (pf_npf is the number of PFs found).
    """
    out = {name: np.full(nmaxpf, np.nan, dtype=np.float32) for name in PF_NAMES}
    out["pf_npf"] = np.array([0.0], dtype=np.float32)
    with np.errstate(invalid="ignore"):
        cand = swath & (rain > rr_thresh)
    labels, n = ndimage.label(cand)
    if n == 0:
        return out
    sizes = np.bincount(labels.ravel(), minlength=n + 1)[1:]
    keep = np.flatnonzero(sizes >= min_npix) + 1
    if keep.size == 0:
        return out
    order = keep[np.argsort(-sizes[keep - 1], kind="stable")]
    out["pf_npf"] = np.array([float(order.size)], dtype=np.float32)
    for r, lab in enumerate(order[:nmaxpf]):
        m = labels == lab
        iy, ix = np.nonzero(m)
        gy, gx = iy + y0, ix + x0
        v = rain[m]
        area = pixel_area[gy, gx] if np.ndim(pixel_area) else np.full(iy.size, float(pixel_area))
        out["pf_area"][r] = area.sum()
        out["pf_lon"][r], out["pf_lat"][r] = np.nanmean(lon[gy, gx]), np.nanmean(lat[gy, gx])
        out["pf_rainrate"][r] = np.nanmean(v)
        out["pf_maxrainrate"][r] = np.nanmax(v)
        out["pf_skewness"][r] = skew(v) if v.size > 2 else np.nan
        out["pf_accumrain"][r] = np.nansum(v)
        heavy = v > heavy_thresh
        out["pf_accumrainheavy"][r] = np.nansum(v[heavy]) if heavy.any() else np.nan
        cc = conv[m]
        if cc.any():
            out["pf_ccarea"][r] = area[cc].sum()
            out["pf_ccrainrate"][r] = np.nanmean(v[cc])
            out["pf_ccrainamount"][r] = np.nansum(v[cc])
        if (~cc).any():
            out["pf_sfarea"][r] = area[~cc].sum()
            out["pf_sfrainrate"][r] = np.nanmean(v[~cc])
            out["pf_sfrainamount"][r] = np.nansum(v[~cc])
        imax = int(np.nanargmax(v))
        out["pf_lon_maxrainrate"][r], out["pf_lat_maxrainrate"][r] = lon[gy[imax], gx[imax]], lat[gy[imax], gx[imax]]
        rp = regionprops(m.astype(np.uint8), intensity_image=np.where(m, np.nan_to_num(rain), 0.0))[0]
        out["pf_eccentricity"][r] = rp.eccentricity
        out["pf_majoraxis"][r] = rp.axis_major_length * pixel_length
        try:
            out["pf_minoraxis"][r] = rp.axis_minor_length * pixel_length
        except ValueError:
            pass
        if out["pf_minoraxis"][r] > 0:
            out["pf_aspectratio"][r] = out["pf_majoraxis"][r] / out["pf_minoraxis"][r]
        out["pf_orientation"][r] = rp.orientation * (180.0 / pi)
        out["pf_perimeter"][r] = rp.perimeter * pixel_length
        for (cy, cx), (nlon, nlat) in (
                (rp.centroid, ("pf_lon_centroid", "pf_lat_centroid")),
                (rp.centroid_weighted, ("pf_lon_weightedcentroid", "pf_lat_weightedcentroid"))):
            if np.isfinite(cy) and np.isfinite(cx):
                yy, xx = int(round(cy)) + y0, int(round(cx)) + x0
                if 0 <= yy < lon.shape[0] and 0 <= xx < lon.shape[1]:
                    out[nlon][r], out[nlat][r] = lon[yy, xx], lat[yy, xx]
    return out


def _window_inputs(task, static):
    """Data field, owner of one window (with owner["covered"]: bool map of pixels with a coverage value in at least one
    step, or None), and the class lookup of the window."""
    source = static["source"]
    field = source.read(task["data_index"])
    masks, convs, covs = [], ([] if static.get("conv_varname") else None), []
    for f in task["files"]:
        m, c, v = read_pixel_file(f, static["mask_varname"], static.get("conv_varname"), static.get("cov_varname"))
        masks.append(m)
        if convs is not None:
            convs.append(c)
        if v is not None:
            covs.append(v)
    if masks and field.shape != masks[0].shape:
        raise ValueError(f"data grid {field.shape} differs from the mask grid {masks[0].shape}: the data must be on the tracking grid")
    nlut = static["nlut"]
    cls, select, rank = class_luts(nlut, task["tracks"], task.get("class_at"), static)
    owner = window_owner(masks, convs, select, rank, static.get("conv_value", 4))
    owner["covered"] = np.any(np.stack(covs), axis=0).ravel() if covs else None
    return field, owner, cls


########################################################################################################################
# Step 1: statistics per (track, window)
########################################################################################################################
def trackpcp_stats_window(task, static):
    """
    Statistics of the data inside the swath of every track of one window.

    task: dict with data_index, files (pixel files of the window), tracks (track numbers present in the window),
          class_at (optional class of each track at this window).
    static: dict (see trackpcp_driver.build_static).

    Returns dict: tracks (as in the task), vars {name: float32 array over tracks} and diag.
    """
    field, owner, _ = _window_inputs(task, static)
    tracks = np.asarray(task["tracks"], dtype=np.int32)
    ntr = tracks.size
    nx = field.shape[1]
    field_flat = field.ravel()
    area = static["pixel_area"]
    stats = static["stats"]
    have_conv = bool(static.get("conv_varname"))
    prefix = static["stats_prefix"]
    window_h = static["window_sec"] / 3600.0
    out = {}
    tr_nsteps, nsteps = owner["track_nsteps"]
    pos = np.searchsorted(tracks, tr_nsteps)
    ok = (pos < ntr) & (tracks[np.minimum(pos, ntr - 1)] == tr_nsteps)
    v = np.full(ntr, np.nan, dtype=np.float32)
    v[pos[ok]] = nsteps[ok]
    out["pcp_nsteps"] = v

    trk_s, pix_s, conv_s, starts, ends = group_by_track(owner["trk"], owner["pix"], owner["conv"])
    gtracks = trk_s[starts] if starts.size else np.zeros(0, dtype=np.int32)
    gpos = np.searchsorted(tracks, gtracks)
    gok = (gpos < ntr) & (tracks[np.minimum(gpos, ntr - 1)] == gtracks) if ntr else np.zeros(0, dtype=bool)

    def put(name, values):
        arr = np.full(ntr, np.nan, dtype=np.float32)
        arr[gpos[gok]] = np.asarray(values, dtype=np.float32)[gok]
        out[name] = arr

    red = reduce_by_track(trk_s, pix_s, conv_s, starts, ends, field_flat, area, stats, have_conv)
    for (sub, stat), values in red.items():
        put(f"{prefix}_{sub}{stat}", values)

    if static.get("precip", False):
        extra = reduce_by_track(trk_s, pix_s, conv_s, starts, ends, field_flat, area, ["sum", "area", "count"], have_conv)
        # volumetric rain: sum(rate x area), computed with the pixel areas
        vals = field_flat[pix_s].astype(np.float64)
        area_pix = area[pix_s] if np.ndim(area) else np.full(vals.shape, float(area))
        fin = np.isfinite(vals)
        subsets = {"": np.ones(vals.shape, bool)}
        if have_conv:
            subsets.update({"conv_": conv_s, "strat_": ~conv_s})
        for sub, sel in subsets.items():
            m = sel & fin
            vol = np.add.reduceat(np.where(m, vals * area_pix, 0.0), starts) if starts.size else np.zeros(0)
            cnt = extra[(sub, "count")]
            tag = {"": "total", "conv_": "conv", "strat_": "strat"}[sub]
            put(f"{tag}_rain", np.where(cnt > 0, extra[(sub, "sum")], np.nan))
            put(f"{tag}_volrain", np.where(cnt > 0, vol, np.nan))
            put(f"{tag}_rainvolume", np.where(cnt > 0, vol * window_h * 1e-3, np.nan))
            put(f"pcp_{tag}_area" if sub else "pcp_area", extra[(sub, "area")])
        heavy_thresh = static["heavy_thresh"]
        hv = fin & (vals > heavy_thresh)
        hcnt = np.add.reduceat(hv.astype(np.int64), starts) if starts.size else np.zeros(0, dtype=np.int64)
        hsum = np.add.reduceat(np.where(hv, vals, 0.0), starts) if starts.size else np.zeros(0)
        hvol = np.add.reduceat(np.where(hv, vals * area_pix, 0.0), starts) if starts.size else np.zeros(0)
        put("total_heavyrain", np.where(hcnt > 0, hsum, np.nan))
        put("total_heavyvolrain", np.where(hcnt > 0, hvol, np.nan))
        put("rainrate_heavyrain", np.where(hcnt > 0, hsum / np.maximum(hcnt, 1), np.nan))
        if starts.size:
            ntot = np.add.reduceat(np.ones(vals.shape), starts)
            put("pcp_valid_frac", np.add.reduceat(fin.astype(float), starts) / np.maximum(ntot, 1))
        # precipitation features
        nmaxpf = static["nmaxpf"]
        pf = {name: np.full((ntr, nmaxpf), np.nan, dtype=np.float32) for name in PF_NAMES}
        pf["pf_npf"] = np.full((ntr, 1), np.nan, dtype=np.float32)
        ny = field.shape[0]
        lon, lat = static["lon"], static["lat"]
        pixel_length = static["pixel_length"]
        min_npix = static["pf_min_npix"]
        for j, y0, x0, iy, ix in track_patches(pix_s, starts, ends, nx):
            if not gok[j]:
                continue
            h, w = int(iy.max()) + 1, int(ix.max()) + 1
            rain = np.full((h, w), np.nan, dtype=np.float32)
            swath = np.zeros((h, w), dtype=bool)
            cmask = np.zeros((h, w), dtype=bool)
            sl = slice(starts[j], ends[j])
            rain[iy, ix] = field_flat[pix_s[sl]]
            swath[iy, ix] = True
            cmask[iy, ix] = conv_s[sl]
            res = precip_features(rain, swath, cmask, y0, x0, lon, lat, area, pixel_length, static["rain_thresh"],
                                  heavy_thresh, nmaxpf, min_npix)
            for name in PF_NAMES:
                if name == "pf_npf":
                    pf[name][gpos[j], 0] = res[name][0]
                else:
                    pf[name][gpos[j], :] = res[name]
        for name in PF_NAMES:
            out[name] = pf[name][:, 0] if name == "pf_npf" else pf[name]
    diag = {"n_pixels_contested": owner["n_pixels_contested"], "n_pixels": int(owner["pix"].size)}
    return {"tracks": tracks, "vars": out, "diag": diag}


########################################################################################################################
# Step 2: accumulation per (track, pixel)
########################################################################################################################
def local_box_mean_max(depth_patch, sizes_px):
    """Maximum over space of the mean of depth_patch in a box of s x s pixels (zero outside the patch), for each s."""
    out = []
    for s in sizes_px:
        pad = s // 2 + 1
        padded = np.pad(depth_patch, pad, mode="constant")
        out.append(float(ndimage.uniform_filter(padded, size=s, mode="constant", cval=0.0).max()))
    return out


def trackpcp_accum_window(task, static):
    """
    Depth [mm] at every pixel of every track's swath in one window (rate x window length), and the largest box-mean
    hourly depth of every track in the window.

    Returns dict: trk (int32), pix (int32), depth (float32), nsteps, tracks (int32 of the local-rate arrays),
    local_rate {box label: float32 array over tracks [mm/h]}.
    """
    field, owner, _ = _window_inputs(task, static)
    nx = field.shape[1]
    window_h = static["window_sec"] / 3600.0
    field_flat = field.ravel()
    pix, trk = owner["pix"], owner["trk"]
    rate = field_flat[pix]
    good = np.isfinite(rate)
    depth = (rate[good] * window_h).astype(np.float32)
    out = {"trk": trk[good], "pix": pix[good].astype(np.int32), "depth": depth, "nsteps": owner["nsteps"][good],
           "n_pixels_contested": owner["n_pixels_contested"]}
    boxes = static.get("boxes", [])
    sizes = [px for _, px in boxes]
    trk_s, pix_s, conv_s, starts, ends = group_by_track(trk[good], pix[good], owner["conv"][good])
    local = {label: np.full(starts.size, np.nan, dtype=np.float32) for label, _ in boxes}
    if boxes:
        d_flat = rate[good] * window_h
        # same order as group_by_track (stable sort by track)
        order = np.argsort(trk[good], kind="stable")
        d_s = d_flat[order]
        for j, y0, x0, iy, ix in track_patches(pix_s, starts, ends, nx):
            h, w = int(iy.max()) + 1, int(ix.max()) + 1
            patch = np.zeros((h, w), dtype=np.float32)
            patch[iy, ix] = d_s[starts[j]:ends[j]]
            for (label, _), val in zip(boxes, local_box_mean_max(patch, sizes)):
                local[label][j] = val / window_h
    out["local_tracks"] = trk_s[starts] if starts.size else np.zeros(0, dtype=np.int32)
    out["local_rate"] = local
    return out


def merge_accumulations(parts):
    """
    Reduce partial (track, pixel) results to one row per (track, pixel): depth summed, hourly maximum, number of windows.

    parts: list of dicts with trk, pix and either depth (one window's value of each pixel) or accum, max_1h, nwin
    (already merged, e.g. the result of an earlier call); returns dict trk, pix, accum, max_1h, nwin sorted by (track, pixel).
    """
    parts = [p for p in parts if p is not None and len(p["trk"]) > 0]
    if not parts:
        return {"trk": np.zeros(0, np.int32), "pix": np.zeros(0, np.int32), "accum": np.zeros(0, np.float32),
                "max_1h": np.zeros(0, np.float32), "nwin": np.zeros(0, np.int16)}
    trk = np.concatenate([p["trk"] for p in parts]).astype(np.int64)
    pix = np.concatenate([p["pix"] for p in parts]).astype(np.int64)
    accum = np.concatenate([p["accum"] if "accum" in p else p["depth"] for p in parts]).astype(np.float64)
    mx = np.concatenate([p["max_1h"] if "max_1h" in p else p["depth"] for p in parts]).astype(np.float64)
    nwin = np.concatenate([p["nwin"] if "nwin" in p else np.ones(len(p["trk"]), dtype=np.int64) for p in parts]).astype(np.int64)
    npix = int(pix.max()) + 1
    key = trk * npix + pix
    order = np.argsort(key, kind="stable")
    key_s = key[order]
    starts = np.flatnonzero(np.r_[True, key_s[1:] != key_s[:-1]])
    return {"trk": (key_s[starts] // npix).astype(np.int32), "pix": (key_s[starts] % npix).astype(np.int32),
            "accum": np.add.reduceat(accum[order], starts).astype(np.float32),
            "max_1h": np.maximum.reduceat(mx[order], starts).astype(np.float32),
            "nwin": np.add.reduceat(nwin[order], starts).astype(np.int16)}


########################################################################################################################
# Step 2, second stage: per-track metrics from the accumulated pixels
########################################################################################################################
def trackpcp_metrics_block(task, static):
    """
    Metrics of a block of tracks from their accumulated pixels.

    task: dict with tracks (int32), offsets (len + 1), pix (int32), accum (float32) of the block.
    static: nx, pixel_area (km2 scalar or 2-D), boxes (list of (label, pixels)), accum_thresholds (mm list), cluster_thresh (mm).
    """
    nx = static["nx"]
    area = static["pixel_area"]
    boxes = static["boxes"]
    sizes = [px for _, px in boxes]
    thresholds = static["accum_thresholds"]
    cthresh = static["cluster_thresh"]
    tracks, offsets, pix, accum = task["tracks"], task["offsets"], task["pix"], task["accum"]
    n = tracks.size
    out = {"accum_max_mm": np.full(n, np.nan, np.float32), "accum_vol_1e6m3": np.full(n, np.nan, np.float32),
           "swath_area_km2": np.full(n, np.nan, np.float32), "cluster_area_km2": np.full(n, np.nan, np.float32),
           "cluster_vol_1e6m3": np.full(n, np.nan, np.float32)}
    for t in thresholds:
        out[f"area_ge{t:g}mm_km2"] = np.full(n, np.nan, np.float32)
    for label, _ in boxes:
        out[f"local_depth_mm_{label}"] = np.full(n, np.nan, np.float32)
    for j in range(n):
        a, b = offsets[j], offsets[j + 1]
        p = pix[a:b].astype(np.int64)
        v = accum[a:b].astype(np.float64)
        iy, ix = p // nx, p % nx
        area_p = area[iy, ix] if np.ndim(area) else np.full(p.size, float(area))
        out["accum_max_mm"][j] = v.max()
        out["accum_vol_1e6m3"][j] = float(np.sum(v * area_p)) * 1e-3
        out["swath_area_km2"][j] = area_p.sum()
        for t in thresholds:
            out[f"area_ge{t:g}mm_km2"][j] = area_p[v >= t].sum()
        y0, x0 = iy.min(), ix.min()
        h, w = int(iy.max() - y0) + 1, int(ix.max() - x0) + 1
        patch = np.zeros((h, w), dtype=np.float32)
        patch[iy - y0, ix - x0] = v
        for (label, _), val in zip(boxes, local_box_mean_max(patch, sizes)):
            out[f"local_depth_mm_{label}"][j] = val
        labels, nl = ndimage.label(patch >= cthresh)
        if nl > 0:
            cvol = ndimage.sum(patch, labels, index=np.arange(1, nl + 1))
            cnpix = np.bincount(labels.ravel(), minlength=nl + 1)[1:]
            best = int(np.argmax(cvol))
            amean = float(area_p.mean())
            out["cluster_area_km2"][j] = cnpix[best] * amean
            out["cluster_vol_1e6m3"][j] = cvol[best] * amean * 1e-3
        else:
            out["cluster_area_km2"][j] = 0.0
            out["cluster_vol_1e6m3"][j] = 0.0
    return {"tracks": tracks, "metrics": out}


########################################################################################################################
# Step 3: class maps for a block of windows
########################################################################################################################
def trackpcp_classmap_block(task, static):
    """
    Accumulate class maps over a block of windows (one result per worker).

    task: dict with windows (list of window tasks as in trackpcp_stats_window).
    static: class_values (list), pcp_thresh [mm/h], conv_varname, window_sec ...

    Returns dict of arrays: precip (nclass, ny, nx) [mm], cloud_count, precip_count (nclass, ny, nx), optional precip_conv,
    precip_strat, total_precip (ny, nx), total_precip_count, nwindows, n_pixels_contested.
    """
    class_values = list(static["class_values"])
    cindex = {v: i for i, v in enumerate(class_values)}
    ncls = len(class_values)
    window_h = static["window_sec"] / 3600.0
    thresh = static["pcp_thresh"]
    have_conv = bool(static.get("conv_varname"))
    acc = None
    nwin = 0
    contested = 0
    for wtask in task["windows"]:
        field, owner, cls = _window_inputs(wtask, static)
        ny, nx = field.shape
        if acc is None:
            shape = (ncls, ny * nx)
            acc = {"precip": np.zeros(shape, np.float64), "cloud_count": np.zeros(shape, np.int32),
                   "precip_count": np.zeros(shape, np.int32), "total_precip": np.zeros(ny * nx, np.float64),
                   "total_precip_count": np.zeros(ny * nx, np.int32), "precip_uncovered": np.zeros(ny * nx, np.float64)}
            if have_conv:
                acc["precip_conv"] = np.zeros(shape, np.float64)
                acc["precip_strat"] = np.zeros(shape, np.float64)
        flat = field.ravel()
        fin = np.isfinite(flat)
        acc["total_precip"] += np.where(fin, flat, 0.0) * window_h
        acc["total_precip_count"] += (fin & (flat >= thresh))
        if owner["covered"] is not None:
            acc["precip_uncovered"] += np.where(fin & ~owner["covered"], flat, 0.0) * window_h
        pix, trk = owner["pix"], owner["trk"]
        if pix.size:
            if cls is None:
                ci = np.zeros(pix.size, dtype=np.int64)
            else:
                lookup = np.full(int(max(cls.max(), 0)) + 1, -1, dtype=np.int64)
                for v, i in cindex.items():
                    if 0 <= v < lookup.size:
                        lookup[v] = i
                cv = cls[trk]
                ci = np.where((cv >= 0) & (cv < lookup.size), lookup[np.clip(cv, 0, lookup.size - 1)], -1)
            sel = ci >= 0
            ci, p, cvv = ci[sel], pix[sel], owner["conv"][sel]
            r = flat[p]
            rf = np.isfinite(r)
            d = np.where(rf, r, 0.0) * window_h
            acc["precip"][ci, p] += d
            acc["cloud_count"][ci, p] += 1
            acc["precip_count"][ci, p] += (rf & (r >= thresh))
            if have_conv:
                acc["precip_conv"][ci, p] += np.where(cvv, d, 0.0)
                acc["precip_strat"][ci, p] += np.where(cvv, 0.0, d)
        nwin += 1
        contested += owner["n_pixels_contested"]
    if acc is None:
        return None
    out = {k: v for k, v in acc.items()}
    out["nwindows"] = nwin
    out["n_pixels_contested"] = contested
    return out
