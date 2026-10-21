"""
Generate the global Mercury subsurface temperature dataset.

Output: data/mercury_temperature.nc with
  temperature(lon, lat, local_time, depth)   [K]
on 0 <= lon < 180 E and 0 <= lat <= 90 N. Use mercury_lookup.py to query any
longitude/latitude (the model is symmetric under lon -> lon+180 and N <-> S).

Usage:  python generate_grid.py [--dlon 2] [--dlat 2] [--nlt 96]
"""

import argparse
import os
import time

import numpy as np
import xarray as xr

from mercury_thermal import Model, resample_local_time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dlon", type=float, default=2.0)
    ap.add_argument("--dlat", type=float, default=2.0)
    ap.add_argument("--nlt", type=int, default=96, help="local-time samples per day")
    ap.add_argument("--out", default=os.path.join("data", "mercury_temperature.nc"))
    args = ap.parse_args()

    model = Model()
    lons = np.arange(0.0, 180.0, args.dlon)
    lats = np.arange(0.0, 90.0 + 1e-9, args.dlat)
    lt = np.arange(args.nlt) * 24.0 / args.nlt
    LON, LAT = np.meshgrid(lons, lats, indexing="ij")

    # symmetry check: lon and lon+180 must give identical local-time histories
    a = model.column_vs_local_time(20, 30, lt)
    b = model.column_vs_local_time(20, 210, lt)
    print(f"symmetry check |T(lon) - T(lon+180)| max = {np.abs(a-b).max():.3f} K")

    model.run_batch(LAT.ravel()[:2], LON.ravel()[:2], max_days=1)  # compile
    t0 = time.perf_counter()
    T, days, conv = model.run_batch(LAT.ravel(), LON.ravel())
    elapsed = time.perf_counter() - t0
    print(f"{T.shape[0]} columns in {elapsed:.1f} s "
          f"({elapsed / T.shape[0] * 1e3:.1f} ms/column effective); "
          f"unconverged: {(~conv).sum()}")

    out = np.empty((lons.size, lats.size, lt.size, model.z.size), dtype=np.float32)
    for j, (lo, la) in enumerate(zip(LON.ravel(), LAT.ravel())):
        out[np.searchsorted(lons, lo), np.searchsorted(lats, la)] = \
            resample_local_time(T[j], lo, lt)

    # geometry on (lon, local_time): true solar hour angle and Sun distance
    # (with the default grid these samples fall exactly on model time steps,
    # so no interpolation across the +-180 deg wrap occurs)
    hour_angle = np.stack([resample_local_time(model.true_hour_angle_deg(lo)[:, None],
                                               lo, lt)[:, 0] for lo in lons])
    r_sun = np.stack([resample_local_time(model.r_sun[:, None], lo, lt)[:, 0]
                      for lo in lons])

    p, r, g = model.planet, model.regolith, model.grid
    ds = xr.Dataset(
        {
            "temperature": (("lon", "lat", "local_time", "depth"), out,
                            {"units": "K", "long_name": "regolith temperature"}),
            "true_hour_angle": (("lon", "local_time"), hour_angle.astype(np.float32),
                                {"units": "degree",
                                 "long_name": "true solar hour angle (+ = afternoon)"}),
            "sun_distance": (("lon", "local_time"), r_sun.astype(np.float32),
                             {"units": "AU"}),
            "density": (("depth",), model.rho, {"units": "kg m-3"}),
            "contact_conductivity": (("depth",), model.kc, {"units": "W m-1 K-1"}),
        },
        coords={
            "lon": ("lon", lons, {"units": "degree_east"}),
            "lat": ("lat", lats, {"units": "degree_north"}),
            "local_time": ("local_time", lt, {
                "units": "hour",
                "long_name": "mean local solar time (24 h = 175.94 Earth days, 12 = mean noon)"}),
            "depth": ("depth", model.z, {"units": "m", "positive": "down"}),
        },
        attrs={
            "title": "Mercury near-surface temperature, 1D thermal model",
            "symmetry": "T(lon+180) = T(lon); T(-lat) = T(lat) (zero obliquity)",
            "method": "1D heat conduction, backward Euler, Hayne et al. 2017 regolith "
                      "formulation; forced by eccentric orbit + 3:2 resonance; flat "
                      "terrain, no topographic shadowing or scattering",
            "albedo0": r.albedo0, "emissivity": r.emis, "rho_s": r.rho_s,
            "rho_d": r.rho_d, "H_m": r.H, "ks": r.ks, "kd": r.kd, "chi": r.chi,
            "q_geo_W_m2": r.q_geo, "solar_constant_1AU": p.S0,
            "semi_major_axis_AU": p.a_au, "eccentricity": p.ecc,
            "steps_per_solar_day": g.steps_per_day,
        },
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    enc = {"temperature": {"zlib": True, "complevel": 4}}
    ds.to_netcdf(args.out, encoding=enc)
    print(f"wrote {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
