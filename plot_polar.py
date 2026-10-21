"""
Figures from the terrain-aware polar products -> data/figures/polar_<pole>.png

Usage:  python plot_polar.py [north|south ...]
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap

from plot_overview import INK, OUT, SEQ

SHADOW = LinearSegmentedColormap.from_list("shadow", ["#f4f3ef", "#0d366b"])


def hillshade(z, res_m):
    gy, gx = np.gradient(z, res_m)
    return -(gx - gy) / np.sqrt(1 + gx**2 + gy**2)


def plot(pole):
    ds = xr.open_dataset(os.path.join("data", f"mercury_polar_{pole}_2km.nc"))
    x, y = ds["x"].values, ds["y"].values
    ext = [x[0], x[-1], y[-1], y[0]]
    T = ds["temperature"]
    inside = np.isfinite(ds["sky_view_factor"].values)
    hs = np.where(inside, hillshade(ds["elevation"].values, 1000 * float(ds.attrs["resolution_km"])),
                  np.nan)
    panels = [
        (hs, "Shaded relief", "gray", None),
        (T.isel(depth=0).max("local_time").values, "Maximum surface temperature (K)", SEQ, None),
        (ds["permanent_shadow"].where(inside).values, "Permanent shadow (dark)", SHADOW, (0, 1)),
        (T.sel(depth=0.5).mean("local_time").values, "Mean temperature at 0.5 m (K)", SEQ, None),
        (ds["sun_visible_fraction"].values, "Fraction of time Sun visible", SEQ, (0, 1)),
        (T.sel(depth=0.5).max("local_time").values - T.sel(depth=0.5).min("local_time").values,
         "Day–night swing at 0.5 m (K)", SEQ, None),
    ]
    fig, axs = plt.subplots(2, 3, figsize=(15, 9.6))
    for ax, (a, title, cmap, lim) in zip(axs.flat, panels):
        im = ax.imshow(a, cmap=cmap, extent=ext, interpolation="nearest",
                       vmin=None if lim is None else lim[0], vmax=None if lim is None else lim[1])
        ax.set_title(title, loc="left", color=INK)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        if title != "Shaded relief":
            fig.colorbar(im, ax=ax, shrink=0.8)
        # latitude rings and 0-degree meridian for orientation
        for latr in (82, 84, 86, 88):
            r = 2 * 2439.4 * np.tan(np.deg2rad(90 - latr) / 2)
            ax.add_patch(plt.Circle((0, 0), r, fill=False, ec="#eb6834", lw=0.5, alpha=0.6))
        r80 = 2 * 2439.4 * np.tan(np.deg2rad(10) / 2)
        sgn = -1 if pole == "north" else 1
        ax.plot([0, 0], [0, sgn * r80], color="#eb6834", lw=0.6, alpha=0.6)
        ax.text(0, sgn * r80 * 1.03, "0°E", color="#eb6834", ha="center",
                va="top" if pole == "north" else "bottom", fontsize=8)
    if "dem_quality" in ds and (ds["dem_quality"].values == 0).any():
        for ax in axs.flat:
            r = 2 * 2439.4 * np.tan(np.deg2rad(2.5) / 2)
            ax.add_patch(plt.Circle((0, 0), r, fill=False, ec="#e34948", lw=1, ls="--"))
        fig.text(0.01, 0.005, "Dashed red circle: no stereo topography (interpolated DEM); "
                 "terrain results inside are unreliable.", color=INK, fontsize=9)
    fig.suptitle(f"Mercury {pole} polar region (≥80°), 2 km terrain-aware model; rings every 2°",
                 x=0.01, ha="left", color=INK)
    fig.tight_layout()
    out = os.path.join(OUT, f"polar_{pole}.png")
    fig.savefig(out, dpi=110)
    print("wrote", out)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for pole in sys.argv[1:] or ["north", "south"]:
        plot(pole)
