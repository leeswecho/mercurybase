"""
Time on-demand generation of a single (lat, lon) temperature column and
check its accuracy against a brute-force (no acceleration, tight tolerance)
reference solution.

Usage:  python benchmark.py
"""

import os
import time

import numpy as np

from mercury_thermal import Model, timed_column

SITES = [  # (lat, lon, description)
    (0, 0, "equator, hot pole"),
    (0, 90, "equator, warm pole"),
    (0, 45, "equator, intermediate"),
    (30, 20, "mid-latitude"),
    (60, 0, "60N, hot longitude"),
    (60, 90, "60N, warm longitude"),
    (80, 0, "80N"),
    (85, 120, "85N"),
    (89, 0, "89N (near pole)"),
]


def main():
    t0 = time.perf_counter()
    model = Model()
    t_setup = time.perf_counter() - t0
    t0 = time.perf_counter()
    model.run_column(0, 0, max_days=1)  # JIT compile / load cached machine code
    t_jit = time.perf_counter() - t0
    print(f"model setup (grid + orbit): {t_setup*1e3:.1f} ms")
    print(f"first call (numba compile or cache load): {t_jit:.2f} s\n")

    hdr = (f"{'site':26s} {'time ms':>8s} {'coarse d':>8s} {'fine d':>6s} "
           f"{'max|err| K':>10s} {'Tsurf min/max K':>16s} {'T(1m) K':>8s}")
    print(hdr)
    print("-" * len(hdr))
    times = []
    i1 = np.searchsorted(model.z, 1.0)
    for lat, lon, name in SITES:
        runs = [timed_column(model, lat, lon) for _ in range(5)]
        r = runs[-1]
        dt = np.median([x["seconds"] for x in runs])
        times.append(dt)
        ref = model.run_column(lat, lon, accelerate=False, tol=1e-5, max_days=5000)
        err = np.abs(r["T"] - ref["T"]).max()
        T = r["T"]
        print(f"{name:26s} {dt*1e3:8.1f} {r['coarse_days']:8d} {r['days']:6d} "
              f"{err:10.3f} {T[:,0].min():7.1f}/{T[:,0].max():6.1f}   {T[:,i1].mean():7.1f}")
    print(f"\nmedian per-column time: {np.median(times)*1e3:.1f} ms, "
          f"max: {np.max(times)*1e3:.1f} ms (single core)")

    # throughput when many columns run in parallel on all cores
    lats = np.repeat(np.arange(0, 90, 10.0), 8)
    lons = np.tile(np.arange(0, 180, 22.5), 9)
    model.run_batch(lats[:2], lons[:2], max_days=1)
    t0 = time.perf_counter()
    model.run_batch(lats, lons)
    dt = time.perf_counter() - t0
    print(f"parallel batch: {lats.size} columns in {dt:.2f} s "
          f"= {dt/lats.size*1e3:.1f} ms/column effective")


POLAR_SITES = [  # (lat, lon, description)
    (85.85, 63.0, "Prokofiev floor"),
    (86.4, 60.0, "Prokofiev surroundings"),
    (87.9, 78.8, "Kandinsky"),
    (88.8, 148.9, "Tolkien"),
    (82.0, 30.0, "82N plains"),
    (77.0, 120.0, "77N (cap edge)"),
    (-85.0, 100.0, "85S"),
    (-81.0, 250.0, "81S"),
    (-72.0, 300.0, "72S (cap edge)"),
]


def polar():
    """Time terrain-aware on-demand runs at native DEM resolution."""
    from polar_site import PolarSite
    sites = {}
    hdr = (f"{'site':24s} {'first ms':>9s} {'repeat ms':>9s} {'PSR':>5s} {'sun vis':>7s} "
           f"{'Tsurf min/max K':>16s} {'T(0.5m) K':>9s}")
    print("\nterrain-aware polar sites (native DEM resolution)")
    print(hdr)
    print("-" * len(hdr))
    for lat, lon, name in POLAR_SITES:
        pole = "north" if lat > 0 else "south"
        if not os.path.exists(os.path.join("data", f"mercury_polar_{pole}_2km.nc")):
            continue
        if pole not in sites:
            t0 = time.perf_counter()
            sites[pole] = PolarSite(pole)
            sites[pole].run(lat, lon)  # compile + build far-field grid
            print(f"[{pole}: setup + first query incl. compile "
                  f"{time.perf_counter() - t0:.1f} s]")
            sites[pole].far_grid = None
        s = sites[pole]
        s.far_grid = None
        first = s.run(lat, lon)
        rep = [s.run(lat, lon)["seconds"] for _ in range(3)]
        T = first["T"]
        print(f"{name:24s} {first['seconds']*1e3:9.0f} {np.median(rep)*1e3:9.0f} "
              f"{str(first['permanent_shadow']):>5s} {first['sun_visible_fraction']:7.3f} "
              f"{T[:,0].min():7.1f}/{T[:,0].max():6.1f}   {T[:,7].mean():8.1f}")
    print("first = new neighbourhood (builds 2 km far-field grid); "
          "repeat = within an already-cached neighbourhood")


GLOBAL_SITES = [  # (lat, lon, description)
    (0.0, 0.0, "equator, hot pole"),
    (0.0, 90.0, "equator, warm pole"),
    (45.0, 45.0, "mid-latitude"),
    (-30.0, 200.0, "30S, 200E"),
    (70.0, 300.0, "70N, 300E"),
]


def global_sites():
    """Time full terrain-aware on-demand runs outside the polar caps, against the slope table."""
    import global_terrain as gt
    from mercury_lookup import MercuryTemperature
    if not os.path.exists(gt.TABLE_PATH):
        return
    mt = MercuryTemperature()
    t0 = time.perf_counter()
    site = mt.site_model(0.0)
    site.run(10.0, 10.0)  # compile + build far-field grid
    print(f"\nterrain-aware sites outside the polar caps (665 m DEM)\n[setup + first query incl. compile "
          f"{time.perf_counter() - t0:.1f} s]")
    hdr = (f"{'site':22s} {'first ms':>9s} {'repeat ms':>9s} {'slope':>6s} {'horizon max':>11s} "
           f"{'exact vs table, mean |dT| K':>27s}")
    print(hdr)
    print("-" * len(hdr))
    for lat, lon, name in GLOBAL_SITES:
        site.far_grid = None
        first = site.run(lat, lon)
        rep = [site.run(lat, lon)["seconds"] for _ in range(3)]
        si = mt.slope_info(lat, lon)
        tab = mt.slope_table.profile(lat, lon, si["gx"], si["gy"])
        print(f"{name:22s} {first['seconds'] * 1e3:9.0f} {np.median(rep) * 1e3:9.0f} "
              f"{first['slope_deg']:5.1f}° {first['horizon_deg'].max():10.1f}° "
              f"{np.abs(first['T'] - tab).mean():27.2f}")


if __name__ == "__main__":
    main()
    polar()
    global_sites()
