"""
Diagnostic figures from data/mercury_temperature.nc -> data/figures/*.png

Usage:  python plot_overview.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from mercury_lookup import MercuryTemperature

OUT = os.path.join("data", "figures")
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3
SEQ = LinearSegmentedColormap.from_list("seq_blue", [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2, "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb", "legend.frameon": False,
})


def surface_vs_local_time(mt):
    lt = np.linspace(0, 24, 385)
    fig, ax = plt.subplots(figsize=(7, 4))
    for c, lon, name in zip(SERIES, (0, 45, 90), ("0° E (hot pole)", "45° E", "90° E (warm pole)")):
        ax.plot(lt, mt(lon, 0, lt, 0.0), color=c, label=name)
    ax.set(xlabel="Mean local solar time (h; 24 h = 176 Earth days)",
           ylabel="Surface temperature (K)", xlim=(0, 24), xticks=range(0, 25, 3))
    ax.set_title("Equatorial surface temperature", loc="left", color=INK)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "surface_vs_local_time_equator.png"), dpi=150)


def depth_envelopes(mt):
    z = mt.depth
    lt = mt.ds["local_time"].values
    fig, ax = plt.subplots(figsize=(6, 5))
    for c, (lat, lon, name) in zip(SERIES, [(0, 0, "Equator, 0° E"), (0, 90, "Equator, 90° E"),
                                             (70, 0, "70° N, 0° E")]):
        T = mt(lon, lat, lt[:, None], z[None, :])
        ax.fill_betweenx(z, T.min(0), T.max(0), color=c, alpha=0.18, linewidth=0)
        ax.plot(T.mean(0), z, color=c, label=f"{name} (mean; band = min–max)")
    ax.invert_yaxis()
    ax.set(xlabel="Temperature (K)", ylabel="Depth (m)", ylim=(z[-1], 0))
    ax.set_title("Temperature range vs depth over a solar day", loc="left", color=INK)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "depth_envelopes.png"), dpi=150)


def maps(mt):
    lon = np.arange(-180, 180.1, 2.0)
    lat = np.arange(-90, 90.1, 2.0)
    lt = mt.ds["local_time"].values
    LON, LAT = np.meshgrid(lon, lat)
    Ts = mt(LON[..., None], LAT[..., None], lt[None, None, :], 0.0)
    T1 = mt(LON[..., None], LAT[..., None], lt[None, None, :], 1.0)
    panels = [(Ts.max(-1), "Maximum surface temperature"),
              (Ts.min(-1), "Minimum surface temperature"),
              (T1.mean(-1), "Mean temperature at 1 m depth"),
              (T1.max(-1) - T1.min(-1), "Day–night swing at 1 m depth")]
    fig, axs = plt.subplots(2, 2, figsize=(11, 6.5))
    for ax, (data, title) in zip(axs.flat, panels):
        im = ax.pcolormesh(lon, lat, data, cmap=SEQ, shading="nearest")
        ax.set_title(title, loc="left", color=INK)
        ax.set(xticks=range(-180, 181, 90), yticks=range(-90, 91, 45), aspect="equal")
        ax.grid(False)
        fig.colorbar(im, ax=ax, label="K", shrink=0.85)
    for ax in axs[1]:
        ax.set_xlabel("Longitude (° E)")
    for ax in axs[:, 0]:
        ax.set_ylabel("Latitude (°)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "global_maps.png"), dpi=150)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    mt = MercuryTemperature()
    surface_vs_local_time(mt)
    depth_envelopes(mt)
    maps(mt)
    print("figures written to", OUT)
