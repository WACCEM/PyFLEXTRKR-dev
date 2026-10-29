#!/usr/bin/env python
"""
Visual check of combine_features_with_share_boundary on synthetic radar reflectivity frames.

Each frame has convective cores (45-55 dBZ) inside stratiform shields (about 20 dBZ) that fall off to 0-10 dBZ at the
storm edge (tests/combine_features_synthetic.py). "cluster" frames hold an oval storm with 4-6 cores, which
detect-and-spread (label_and_grow_features, the [25, 10] dBZ settings of the MRMS configs) cuts into one piece per
core, a pair of storms whose edges barely touch, and two single storms. "spread" frames hold five single storms far
apart. Three panels per frame: reflectivity, the features without the combine step, the features with it.

This script is for visual inspection; test_combine_features.py has the assertions.

Usage
-----
  python tests/plot_combine_features_synthetic.py --outdir /path/to/output

Requirements: numpy, matplotlib (no map projection needed)
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pyflextrkr.combine_features import combine_features_with_share_boundary
from pyflextrkr.label_and_grow_features import label_and_grow_features
from combine_features_synthetic import make_frame, DX_KM

# Detect-and-spread settings of the MRMS configs ([25, 10] dBZ test)
THRESHOLDS = [25.0, 10.0, 10.0, 10.0]
AREA_THRESH = 81.0       # [km^2]
MIN_CORE_NPIX = 4
SMOOTH_SIZE = 10
# Reflectivity color scale: gist_ncar, -10 to 70 dBZ in 5 dB bins
DBZ_LEVELS = np.arange(-10, 71, 5)
DBZ_CMAP = "gist_ncar"
RADAR_SENSITIVITY = 0.0    # [dBZ] echo below this is not drawn (as in the track maps)


def parse_args():
    """Command line: --outdir (default "."), --nframes (default 4), --seed (default 0)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outdir", default=".", help="Directory of the figures (default: .)")
    parser.add_argument("--nframes", type=int, default=4, help="Number of frames, cluster and spread alternate (default: 4)")
    parser.add_argument("--seed", type=int, default=0, help="Base seed, frame i uses seed+i (default: 0)")
    return parser.parse_args()


def detect_and_spread(dbz):
    """
    Features of a reflectivity frame with label_and_grow_features.

    Args:
        dbz: np.ndarray (ny, nx), reflectivity [dBZ]

    Returns:
        labels: np.ndarray (ny, nx), feature numbers (0 = none)
    """
    result = label_and_grow_features(
        np.maximum(dbz, 0.0), DX_KM, THRESHOLDS, AREA_THRESH, MIN_CORE_NPIX, SMOOTH_SIZE, 0,
        {"pbc_direction": "none"}, core_operator="gt", growth_method="bfs",
    )
    return result["final_Feature_Number"]


def outline(labels):
    """Pixels on the inner edge of each feature, 2 pixels wide (for drawing the outlines), as a masked label array."""
    inner = np.zeros(labels.shape, dtype=bool)
    for label in np.unique(labels[labels > 0]):
        mask = labels == label
        inner |= mask & ~ndimage.binary_erosion(mask, iterations=2, border_value=0)
    return np.ma.masked_where(~inner, labels)


def plot_frame(dbz, labels_before, labels_after, title, figname):
    """
    Three panels: reflectivity, features without and with the combine step (outlines, one color per feature).

    Args:
        dbz: np.ndarray (ny, nx), reflectivity [dBZ]
        labels_before, labels_after: np.ndarray (ny, nx), feature numbers without / with the combine step
        title: string, figure title
        figname: string, PNG file name
    """
    ny, nx = dbz.shape
    extent = [0, nx * DX_KM, 0, ny * DX_KM]
    norm = matplotlib.colors.BoundaryNorm(DBZ_LEVELS, ncolors=plt.get_cmap(DBZ_CMAP).N, clip=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.6), constrained_layout=True)
    panels = [(None, "Reflectivity"), (labels_before, f"Without combine: {labels_before.max()} features"),
              (labels_after, f"With combine: {labels_after.max()} features")]
    for ax, (labels, label_title) in zip(axes, panels):
        field = np.ma.masked_where(dbz < RADAR_SENSITIVITY, dbz)     # no echo below the radar sensitivity
        cf = ax.imshow(field, origin="lower", extent=extent, cmap=DBZ_CMAP, norm=norm, interpolation="nearest",
                       alpha=1.0 if labels is None else 0.55)
        if labels is not None:
            ax.imshow(outline(labels) % 10, origin="lower", extent=extent, cmap="tab10", vmin=0, vmax=10,
                      interpolation="nearest")
        ax.set_title(label_title)
        ax.set_xlabel("x [km]")
    axes[0].set_ylabel("y [km]")
    fig.colorbar(cf, ax=axes, label="Reflectivity (dBZ)", ticks=DBZ_LEVELS[::2], extend="both", shrink=0.9, pad=0.01)
    fig.suptitle(title)
    fig.savefig(figname, dpi=150, facecolor="w")
    plt.close(fig)


def main():
    """Make the frames, run detect-and-spread with and without the combine step, save the figures and print a summary."""
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    print(f"\n{'=' * 60}\n  Combine features with a shared boundary: synthetic frames\n{'=' * 60}")
    for i in range(args.nframes):
        kind = "cluster" if i % 2 == 0 else "spread"
        dbz, storms = make_frame(args.seed + i, kind)
        labels = detect_and_spread(dbz)
        combined, nfeatures, info = combine_features_with_share_boundary(labels)
        name = f"synthetic_{i:02d}_{kind}"
        plot_frame(dbz, labels, combined, name, os.path.join(args.outdir, f"{name}.png"))
        cores = sum(s["ncores"] for s in storms)
        print(f"  {name}: {len(storms)} storms, {cores} cores | features {info['nfeatures_before']} -> {nfeatures} "
              f"(passes {info['npasses']}), merged groups {list(info['groups'].values())}")
    print(f"\n  Figures saved to: {args.outdir}")


if __name__ == "__main__":
    main()
