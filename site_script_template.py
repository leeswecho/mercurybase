#!/usr/bin/env python3
"""
Mercury regolith temperature at one site: a standalone copy of the
"Run exact model here" calculation from the Mercury temperature explorer.

What this script does
  1. Computes Mercury's orbit (eccentric, 3:2 spin-orbit resonance) over one
     solar day (175.94 Earth days), and the Sun's direction as seen from the site.
  2. Computes the sunlight absorbed by the ground there, allowing for the
     local tilt of the ground, the terrain horizon (the Sun's finite disk can
     be partly hidden), and an angle-dependent albedo.
  3. Adds the heat arriving from surrounding terrain (sunlight it scatters and
     infrared it emits), which is supplied below as a time series.
  4. Solves the 1D heat-conduction equation down to 3 m, repeating solar days
     until the temperatures settle into a repeating daily cycle.

Inputs inlined below were extracted from the explorer's terrain-aware run for
this site (see SITE["provenance"] for the datasets they come from):
  - the surface normal (tilt and facing direction of the ground)
  - the horizon elevation in 128 directions, ray-traced over up to 250 km of
    terrain on a MESSENGER digital elevation model
  - the heat absorbed from surrounding terrain over the day, summed over every
    terrain cell the site can see, weighted by its view factor
Everything else (orbit, sunlight, regolith properties, heat flow) is computed
here from scratch.

Requirements: Python 3.8+ and numpy. Optional: numba (faster: about a second
after compiling), matplotlib (for --plot). Without numba it takes about 5-30 s.

Usage
  python THIS_FILE.py                  # run and print a summary
  python THIS_FILE.py --csv out.csv    # also write T(local time, depth) to CSV
  python THIS_FILE.py --plot           # also plot (needs matplotlib)
  python THIS_FILE.py --no-terrain     # same site as flat open ground, for comparison

Physics follows Vasavada, Paige & Wood (1999), Icarus 141, 179-193 and the
regolith properties of Hayne et al. (2017), JGR Planets 122, 2371-2400.
"""

import argparse
import math
import time

import numpy as np

try:
    from numba import njit
except ImportError:  # pure Python fallback: same code, just slower
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return lambda f: f

# ---------------------------------------------------------------- site inputs
# @@INPUTS@@
SITE = dict(lat_deg=0.0, lon_deg=0.0, cell_lat_deg=0.0, cell_lon_deg=0.0,
            normal_enu=(0.0, 0.0, 1.0), elevation_m=0.0, slope_deg=0.0,
            sky_view_factor=1.0, provenance=[])
HORIZON_AZIMUTH_DEG = [0.0, 90.0, 180.0, 270.0]
HORIZON_ELEVATION_DEG = [-90.0, -90.0, -90.0, -90.0]
INDIRECT_FLUX = [0.0] * 180
EXPECTED = None
# @@END INPUTS@@

# ------------------------------------------------------------------ constants
SIGMA = 5.670374419e-8         # Stefan-Boltzmann constant, W m-2 K-4
DAY = 86400.0                  # s
S0 = 1361.0                    # solar constant at 1 AU, W m-2
A_AU = 0.387098                # semi-major axis, AU
ECC = 0.205630                 # eccentricity
P_ORB = 87.9691 * DAY          # orbital (sidereal) period, s
P_ROT = P_ORB * 2.0 / 3.0      # sidereal rotation (exact 3:2 resonance)
P_SOL = 2.0 * P_ORB            # solar day = 2 orbits
SUN_RADIUS_RAD_1AU = 0.0046524 # angular radius of the Sun at 1 AU

# regolith (Hayne et al. 2017)
ALBEDO0, ALB_A, ALB_B = 0.07, 0.06, 0.25  # normal albedo and angle law (Keihm 1984)
EMIS = 0.95                    # infrared emissivity
RHO_S, RHO_D, H = 1100.0, 1800.0, 0.07    # density at surface / depth (kg m-3), e-folding (m)
KS, KD = 7.4e-4, 3.4e-3        # contact conductivity at surface / depth, W m-1 K-1
CHI = 2.7                      # radiative conductivity parameter (at 350 K)
Q_GEO = 0.02                   # geothermal heat flux, W m-2

# numerical grid
Z_MAX, DZ0, GROWTH = 3.0, 0.002, 1.08     # depth grid: 2 mm top layer, growing 8% per layer
STEPS_PER_DAY = 2880           # time steps per solar day
COARSE_STRIDE = 16             # spin up first with 180 steps per day
N_TC = len(INDIRECT_FLUX)      # samples per day of the terrain heat series
OUT_LT = np.arange(48) * 0.5   # output mean local solar times, hours
OUT_DEPTHS = np.array([0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])


def depth_grid():
    z = [0.0]
    dz = DZ0
    while z[-1] < Z_MAX:
        z.append(z[-1] + dz)
        dz *= GROWTH
    return np.array(z)


def orbit(steps):
    """Time (s), Sun distance (AU) and subsolar east longitude (rad) over one solar day.
    t = 0 is a perihelion at which the Sun is overhead at 0 deg E."""
    t = np.arange(steps) * P_SOL / steps
    M = 2 * np.pi * t / P_ORB
    E = M.copy()
    for _ in range(50):  # Kepler's equation by Newton's method
        E -= (E - ECC * np.sin(E) - M) / (1 - ECC * np.cos(E))
    nu = 2 * np.arctan2(np.sqrt(1 + ECC) * np.sin(E / 2), np.sqrt(1 - ECC) * np.cos(E / 2))
    r = A_AU * (1 - ECC * np.cos(E))
    lss = np.unwrap(nu) - 2 * np.pi * t / P_ROT
    return t, r, lss


def albedo(mu):
    """Albedo vs cosine of the incidence angle (Keihm 1984 form)."""
    th = math.acos(min(max(mu, 0.0), 1.0))
    return ALBEDO0 + ALB_A * (th / (math.pi / 4)) ** 3 + ALB_B * (th / (math.pi / 2)) ** 8


def horizon_at(az):
    """Horizon elevation (rad) toward compass azimuth az (rad), periodic interpolation."""
    return math.radians(np.interp(math.degrees(az) % 360.0, HORIZON_AZIMUTH_DEG,
                                  HORIZON_ELEVATION_DEG, period=360.0))


def sunlight(lat, lon, normal, lss, r_au, terrain=True):
    """
    For each time step: fraction of the solar disk above the horizon, and
    absorbed direct sunlight (W m-2) on ground with unit normal `normal`
    (east, north, up components).
    """
    n = lss.size
    vis = np.zeros(n)
    F = np.zeros(n)
    for s in range(n):
        h = lon - lss[s]                         # hour angle
        se = -math.sin(h)                        # Sun direction: east, north, up
        sn = -math.sin(lat) * math.cos(h)
        su = math.cos(lat) * math.cos(h)
        hz = horizon_at(math.atan2(se, sn)) if terrain else -math.pi / 2
        x = (math.asin(su) - hz) / (SUN_RADIUS_RAD_1AU / r_au[s])
        if x <= -1.0:
            continue                             # whole disk below the horizon
        f = 1.0 if x >= 1.0 else 1.0 - (math.acos(x) - x * math.sqrt(1 - x * x)) / math.pi
        vis[s] = f
        mu = normal[0] * se + normal[1] * sn + normal[2] * su
        if mu > 0.0:
            F[s] = (1.0 - albedo(mu)) * S0 / r_au[s] ** 2 * f * mu
    return vis, F


@njit(cache=True)
def cp(T):
    """Heat capacity, J kg-1 K-1 (Hayne et al. 2017 polynomial)."""
    return -3.6125 + T * (2.7431 + T * (2.3616e-3 + T * (-1.2340e-5 + T * 8.9093e-9)))


@njit(cache=True)
def step(T, Tn, z, V, rho, kc, F_abs, dt, k, P, Q):
    """
    One backward-Euler time step of the heat equation. Conductivity and heat
    capacity are taken at the start of the step; surface radiation is fully
    implicit. The tridiagonal system is eliminated from the bottom up, leaving
    one quartic equation for the surface temperature (solved by Newton's method).
    """
    nz = z.size
    for i in range(nz):
        k[i] = kc[i] * (1.0 + CHI * (T[i] / 350.0) ** 3)
    G = 0.5 * (k[nz - 1] + k[nz - 2]) / (z[nz - 1] - z[nz - 2])
    m = rho[nz - 1] * cp(T[nz - 1]) * V[nz - 1] / dt
    bb = m + G
    P[nz - 1] = (m * T[nz - 1] + Q_GEO) / bb   # geothermal heat enters at the bottom
    Q[nz - 1] = G / bb
    for i in range(nz - 2, 0, -1):
        Gp = G
        G = 0.5 * (k[i] + k[i - 1]) / (z[i] - z[i - 1])
        m = rho[i] * cp(T[i]) * V[i] / dt
        bb = m + G + Gp - Gp * Q[i + 1]
        P[i] = (m * T[i] + Gp * P[i + 1]) / bb
        Q[i] = G / bb
    m = rho[0] * cp(T[0]) * V[0] / dt
    B = m + G - G * Q[1]
    D = m * T[0] + F_abs + G * P[1]
    es = EMIS * SIGMA
    Ts = T[0]
    for _ in range(20):  # B*Ts + es*Ts^4 = D
        dT = (B * Ts + es * Ts ** 4 - D) / (B + 4.0 * es * Ts ** 3)
        Ts -= dT
        if abs(dT) < 1e-6:
            break
    Tn[0] = Ts
    for i in range(1, nz):
        Tn[i] = P[i] + Q[i] * Tn[i - 1]


@njit(cache=True)
def spin_up(T, Tday, F, dt, z, rho, kc, tol, max_days):
    """
    Repeat the solar day under absorbed flux F until the daily-mean temperature
    at depth changes by less than tol (K) per day. Once the deep temperatures
    approach equilibrium geometrically, the remaining change is extrapolated
    (Aitken acceleration). T is updated in place; Tday gets the last day.
    """
    nz = z.size
    nt = F.size
    V = np.empty(nz)  # layer thicknesses (finite volumes)
    V[0] = 0.5 * (z[1] - z[0])
    V[nz - 1] = 0.5 * (z[nz - 1] - z[nz - 2])
    for i in range(1, nz - 1):
        V[i] = 0.5 * (z[i + 1] - z[i - 1])
    Tn = np.empty(nz); k = np.empty(nz); P = np.empty(nz); Q = np.empty(nz)
    mean_prev = np.zeros(nz)
    delta_prev = np.zeros(nz)
    r_prev = 0.0
    since_jump = 0
    day = 0
    for day in range(1, max_days + 1):
        mean = np.zeros(nz)
        for s in range(nt):
            step(T, Tn, z, V, rho, kc, F[s], dt, k, P, Q)
            for i in range(nz):
                T[i] = Tn[i]
                Tday[s, i] = Tn[i]
                mean[i] += Tn[i]
        mean /= nt
        delta = mean - mean_prev
        since_jump += 1
        if day > 2 and abs(delta[nz - 1]) < tol and abs(delta[nz // 2]) < tol:
            break
        r = delta[nz - 1] / delta_prev[nz - 1] if delta_prev[nz - 1] != 0.0 else 0.0
        if since_jump >= 4 and 0.5 < r < 0.99 and 0.5 < r_prev < 0.99 and abs(r - r_prev) < 0.01:
            jump = min(r / (1.0 - r), 30.0)
            for i in range(nz):
                T[i] += delta[i] * jump
            mean = mean + delta * jump
            since_jump = 0
            r = 0.0
            delta = np.zeros(nz)
        mean_prev = mean
        delta_prev = delta
        r_prev = r
    return day


def run(terrain=True):
    """Temperature (48 local times x 13 depths), plus illumination statistics."""
    z = depth_grid()
    rho = RHO_D - (RHO_D - RHO_S) * np.exp(-z / H)
    kc = KD - (KD - KS) * (RHO_D - rho) / (RHO_D - RHO_S)
    _, r_au, lss = orbit(STEPS_PER_DAY)
    lat = math.radians(SITE["cell_lat_deg"])
    lon = math.radians(SITE["cell_lon_deg"])
    normal = SITE["normal_enu"] if terrain else (0.0, 0.0, 1.0)
    vis, F = sunlight(lat, lon, normal, lss, r_au, terrain)
    lit = F > 0
    if terrain:  # heat from surrounding terrain, linearly interpolated in time
        tf = np.arange(STEPS_PER_DAY) * N_TC / STEPS_PER_DAY
        F = F + np.interp(tf, np.arange(N_TC + 1), list(INDIRECT_FLUX) + [INDIRECT_FLUX[0]])
    dt = P_SOL / STEPS_PER_DAY
    T = np.full(z.size, 1.05 * ((F.mean() + Q_GEO) / (EMIS * SIGMA)) ** 0.25)
    Fc = F[::COARSE_STRIDE].copy()
    spin_up(T, np.empty((Fc.size, z.size)), Fc, dt * COARSE_STRIDE, z, rho, kc, 0.01, 2000)
    Tday = np.empty((STEPS_PER_DAY, z.size))
    days = spin_up(T, Tday, F, dt, z, rho, kc, 0.01, 2000)
    # sample at mean local solar times: LT = 12 h + (lon + 360 deg * t / P_SOL) / 15
    s = ((15.0 * (OUT_LT - 12.0) - math.degrees(lon)) / 360.0 * STEPS_PER_DAY) % STEPS_PER_DAY
    s0 = np.floor(s).astype(int)
    w = (s - s0)[:, None]
    Tlt = (1 - w) * Tday[s0 % STEPS_PER_DAY] + w * Tday[(s0 + 1) % STEPS_PER_DAY]
    out = np.stack([np.interp(OUT_DEPTHS, z, row) for row in Tlt])
    return dict(T=out, days=days, sun_visible_fraction=vis.mean(),
                direct_lit_fraction=lit.mean(), permanent_shadow=not lit.any())


def main():
    ap = argparse.ArgumentParser(description="Mercury regolith temperature at one site")
    ap.add_argument("--csv", help="write T(local time, depth) in kelvin to this CSV file")
    ap.add_argument("--plot", action="store_true", help="plot the result (needs matplotlib)")
    ap.add_argument("--no-terrain", action="store_true",
                    help="ignore slope, horizon and terrain heat (flat open ground)")
    a = ap.parse_args()
    terrain = not a.no_terrain

    print(f"Site {SITE['lat_deg']:.3f} N, {SITE['lon_deg']:.3f} E "
          f"(model cell {SITE['cell_lat_deg']:.4f} N, {SITE['cell_lon_deg']:.4f} E)")
    print(f"elevation {SITE['elevation_m']:.0f} m, slope {SITE['slope_deg']:.1f} deg, "
          f"sky view factor {SITE['sky_view_factor']:.3f}"
          + ("" if terrain else "  [ignored: --no-terrain]"))
    t0 = time.perf_counter()
    r = run(terrain)
    T = r["T"]
    print(f"solved in {time.perf_counter() - t0:.1f} s ({r['days']} solar days to converge)")
    print(f"Sun disk visible {100 * r['sun_visible_fraction']:.1f}% of the day, "
          f"directly lit {100 * r['direct_lit_fraction']:.1f}%"
          + (" -- PERMANENT SHADOW" if r["permanent_shadow"] else ""))
    print("\n depth     min K    mean K     max K")
    for k, d in enumerate(OUT_DEPTHS):
        print(f"{d:5.2f} m  {T[:, k].min():8.1f}  {T[:, k].mean():8.1f}  {T[:, k].max():8.1f}")
    if EXPECTED is not None and terrain:
        diff = np.abs(T - np.asarray(EXPECTED))
        print(f"\nvs the explorer's result: max |difference| {diff.max():.2f} K, "
              f"mean {diff.mean():.3f} K")

    if a.csv:
        with open(a.csv, "w") as f:
            f.write("local_time_h," + ",".join(f"T_{d:g}m_K" for d in OUT_DEPTHS) + "\n")
            for i, lt in enumerate(OUT_LT):
                f.write(f"{lt:g}," + ",".join(f"{v:.2f}" for v in T[i]) + "\n")
        print(f"wrote {a.csv}")
    if a.plot:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
        for k in (0, 4, 7, 9, 12):
            ax1.plot(OUT_LT, T[:, k], label=f"{OUT_DEPTHS[k]:g} m")
        ax1.set(xlabel="mean local solar time (h)", ylabel="temperature (K)", xlim=(0, 24))
        ax1.legend()
        im = ax2.pcolormesh(OUT_LT, OUT_DEPTHS, T.T, shading="nearest", cmap="inferno")
        ax2.set(xlabel="mean local solar time (h)", ylabel="depth (m)", yscale="symlog")
        ax2.set_yscale("symlog", linthresh=0.05)
        ax2.invert_yaxis()
        fig.colorbar(im, ax=ax2, label="K")
        fig.suptitle(f"Mercury regolith, {SITE['lat_deg']:.2f} N {SITE['lon_deg']:.2f} E")
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
