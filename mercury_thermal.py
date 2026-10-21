"""
1D thermal model of Mercury's regolith.

Temperature as a function of (longitude, latitude, local time, depth) is
obtained by solving the 1D heat equation in each column, forced by
insolation computed from Mercury's eccentric orbit and 3:2 spin-orbit
resonance.

Physics follows the standard lunar/Mercury regolith formulation:
  - Vasavada, Paige & Wood (1999), Icarus 141, 179-193
  - Hayne et al. (2017), JGR Planets 122, 2371-2400 (density/conductivity
    profile, radiative conductivity, heat capacity polynomial, albedo law)

Conventions
  - East longitude, degrees. Subsolar point is at 0 deg longitude at the
    reference perihelion (t = 0), so 0/180 E are the "hot poles" and
    90/270 E the "warm poles".
  - Obliquity is taken as zero (actual ~0.03 deg), so the solar declination
    is always zero and the model is symmetric north/south.
  - Because of the 3:2 resonance, the temperature history at longitude
    lon + 180 is identical to that at lon (shifted by one orbit), so only
    0 <= lon < 180 needs to be computed.
  - "Local time" means MEAN local solar time (uniform clock, 24 Mercury
    hours per 175.94-day solar day, 12h = local mean noon). True solar time
    is not monotonic near perihelion (the Sun briefly reverses in the sky),
    so it cannot be used as an unambiguous time coordinate.
"""

import time
from dataclasses import dataclass, field

import numpy as np
from numba import njit, prange

SIGMA = 5.670374419e-8  # Stefan-Boltzmann, W m-2 K-4
DAY = 86400.0  # s


@dataclass
class Planet:
    S0: float = 1361.0  # solar constant at 1 AU, W m-2
    a_au: float = 0.387098  # semi-major axis, AU
    ecc: float = 0.205630  # eccentricity
    P_orb: float = 87.9691 * DAY  # orbital (sidereal) period, s

    @property
    def P_rot(self):  # sidereal rotation, exact 3:2 resonance
        return self.P_orb * 2.0 / 3.0

    @property
    def P_sol(self):  # solar day = 2 orbits
        return 2.0 * self.P_orb


@dataclass
class Regolith:
    albedo0: float = 0.07  # normal-incidence albedo (Bond ~0.09 with angle law)
    alb_a: float = 0.06  # albedo angle-dependence coefficients (Keihm 1984)
    alb_b: float = 0.25
    emis: float = 0.95  # infrared emissivity
    rho_s: float = 1100.0  # surface density, kg m-3
    rho_d: float = 1800.0  # density at depth, kg m-3
    H: float = 0.07  # density e-folding scale, m
    ks: float = 7.4e-4  # surface contact conductivity, W m-1 K-1
    kd: float = 3.4e-3  # deep contact conductivity, W m-1 K-1
    chi: float = 2.7  # radiative conductivity parameter (at 350 K)
    q_geo: float = 0.02  # geothermal heat flux, W m-2


@dataclass
class Grid:
    z_max: float = 3.0  # m
    dz0: float = 0.002  # top-layer thickness, m
    growth: float = 1.08  # geometric growth factor of layer thickness
    steps_per_day: int = 2880  # time steps per solar day (0.125 deg each)
    spinup_steps_per_day: int = 180  # coarse steps used to approach equilibrium

    def depths(self):
        z = [0.0]
        dz = self.dz0
        while z[-1] < self.z_max:
            z.append(z[-1] + dz)
            dz *= self.growth
        z = np.array(z)
        z[-1] = max(z[-1], self.z_max)
        return z


@dataclass
class Model:
    planet: Planet = field(default_factory=Planet)
    regolith: Regolith = field(default_factory=Regolith)
    grid: Grid = field(default_factory=Grid)

    def __post_init__(self):
        self.z = self.grid.depths()
        r = self.regolith
        self.rho = r.rho_d - (r.rho_d - r.rho_s) * np.exp(-self.z / r.H)
        self.kc = r.kd - (r.kd - r.ks) * (r.rho_d - self.rho) / (r.rho_d - r.rho_s)
        self.t, self.r_sun, self.lss = orbit(self.planet, self.grid.steps_per_day)
        _, self.r_sun_c, self.lss_c = orbit(self.planet, self.grid.spinup_steps_per_day)

    def mean_local_time_hours(self, lon_deg):
        """Mean local solar time (hours) at each model time step for a longitude."""
        h_mean = np.deg2rad(lon_deg) + 2 * np.pi * self.t / self.planet.P_sol
        return (12.0 + np.rad2deg(h_mean) / 15.0) % 24.0

    def true_hour_angle_deg(self, lon_deg):
        """True solar hour angle (deg, -180..180, + = afternoon) at each time step."""
        return (lon_deg - np.rad2deg(self.lss) + 180.0) % 360.0 - 180.0

    def _args(self, coarse=False):
        p, r = self.planet, self.regolith
        r_sun, lss = (self.r_sun_c, self.lss_c) if coarse else (self.r_sun, self.lss)
        dt = p.P_sol / lss.size
        return (self.z, self.rho, self.kc, r.chi, r.emis, r.albedo0, r.alb_a,
                r.alb_b, r.q_geo, p.S0 / r_sun**2, lss, dt)

    def initial_temperature(self, lat_deg, lon_deg):
        """Uniform starting temperature from radiative balance with mean flux."""
        r = self.regolith
        h = np.deg2rad(lon_deg) - self.lss
        mu = np.maximum(np.cos(np.deg2rad(lat_deg)) * np.cos(h), 0.0)
        F = (1 - r.albedo0) * self.planet.S0 / self.r_sun**2 * mu
        T0 = ((F.mean() + r.q_geo) / (r.emis * SIGMA)) ** 0.25
        # nonlinear radiative conductivity makes the subsurface warmer than
        # this estimate; ~1.1x is a good empirical starting point
        return np.full(self.z.size, 1.1 * T0)

    def run_column(self, lat_deg, lon_deg, tol=0.01, max_days=2000,
                   T_init=None, accelerate=True):
        """
        Spin up one column to a periodic steady state, return final solar day.

        Returns dict with
          T: (steps_per_day, nz) temperatures over the final solar day,
             time index 0 = reference perihelion (subsolar lon 0)
          days: solar days of spin-up used, converged: bool
        """
        if T_init is None:
            T_init = self.initial_temperature(lat_deg, lon_deg)
        T = np.asarray(T_init, dtype=np.float64).copy()
        lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
        coarse_days = 0
        if self.grid.spinup_steps_per_day:
            Tc = np.empty((self.grid.spinup_steps_per_day, self.z.size))
            coarse_days, _ = _spin_up(lat, lon, T, Tc, tol, max_days, accelerate,
                                      *self._args(coarse=True))
        Tday = np.empty((self.grid.steps_per_day, self.z.size))
        days, conv = _spin_up(lat, lon, T, Tday, tol, max_days, accelerate,
                              *self._args())
        return dict(T=Tday, days=days, coarse_days=coarse_days, converged=conv)

    def column_vs_local_time(self, lat_deg, lon_deg, local_time_hours, **kw):
        """
        On-demand temperatures for one site: array (n_local_times, nz) at the
        requested mean local solar times (hours), depths self.z.
        """
        T = self.run_column(lat_deg, lon_deg, **kw)["T"]
        return resample_local_time(T, lon_deg, local_time_hours)

    def run_batch(self, lats_deg, lons_deg, tol=0.01, max_days=2000,
                  accelerate=True, out_stride=1):
        """Run many columns in parallel. Returns T (ncol, nt_out, nz), days, conv."""
        lats = np.deg2rad(np.asarray(lats_deg, dtype=np.float64))
        lons = np.deg2rad(np.asarray(lons_deg, dtype=np.float64))
        T0 = np.stack([self.initial_temperature(la, lo)
                       for la, lo in zip(lats_deg, lons_deg)])
        nt = self.grid.steps_per_day
        Tout = np.empty((lats.size, nt // out_stride, self.z.size))
        days = np.empty(lats.size, dtype=np.int64)
        conv = np.empty(lats.size, dtype=np.bool_)
        if self.grid.spinup_steps_per_day:
            Tc = np.empty((lats.size, self.grid.spinup_steps_per_day, self.z.size))
            _batch(lats, lons, T0, Tc, days, conv, tol, max_days, accelerate,
                   1, *self._args(coarse=True))
            T0 = Tc[:, -1, :].copy()
        _batch(lats, lons, T0, Tout, days, conv, tol, max_days, accelerate,
               out_stride, *self._args())
        return Tout, days, conv


def resample_local_time(T, lon_deg, local_time_hours):
    """
    Convert a solar-day history T (nt, ...) with time index 0 at the reference
    perihelion into values at mean local solar times (periodic interpolation).
    """
    nt = T.shape[0]
    # mean hour angle advances 360 deg per solar day: LT = 12 + (lon + 360 s/nt)/15
    s = ((15.0 * (np.asarray(local_time_hours) - 12.0) - lon_deg) / 360.0 * nt) % nt
    i0 = np.floor(s).astype(int)
    w = (s - i0).reshape((-1,) + (1,) * (T.ndim - 1))
    return (1 - w) * T[i0 % nt] + w * T[(i0 + 1) % nt]


def orbit(planet, steps_per_day):
    """Sun distance (AU) and subsolar east longitude (rad) over one solar day."""
    t = np.arange(steps_per_day) * planet.P_sol / steps_per_day
    M = 2 * np.pi * t / planet.P_orb
    e = planet.ecc
    E = M.copy()
    for _ in range(50):
        E -= (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
    nu = np.unwrap(nu)
    r = planet.a_au * (1 - e * np.cos(E))
    lss = nu - 2 * np.pi * t / planet.P_rot  # subsolar longitude, moves westward
    return t, r, lss


@njit(cache=True)
def _cp(T):
    # Hayne et al. (2017) polynomial (J kg-1 K-1), extrapolated above ~350 K
    return -3.6125 + T * (2.7431 + T * (2.3616e-3 + T * (-1.2340e-5 + T * 8.9093e-9)))


@njit(cache=True)
def _step(T, Tn, z, V, rho, kc, chi, emis, q, F_abs, dt, k, P, Q):
    """
    One backward-Euler step. Conductivity and heat capacity are lagged at the
    start of the step; surface radiation is fully implicit. The tridiagonal
    system is eliminated bottom-up, leaving a scalar quartic for the surface
    temperature which is solved by Newton's method.
    """
    nz = z.size
    for i in range(nz):
        k[i] = kc[i] * (1.0 + chi * (T[i] / 350.0) ** 3)
    # bottom row: geothermal flux in
    G = 0.5 * (k[nz - 1] + k[nz - 2]) / (z[nz - 1] - z[nz - 2])
    m = rho[nz - 1] * _cp(T[nz - 1]) * V[nz - 1] / dt
    bb = m + G
    P[nz - 1] = (m * T[nz - 1] + q) / bb
    Q[nz - 1] = G / bb
    for i in range(nz - 2, 0, -1):
        Gp = G
        G = 0.5 * (k[i] + k[i - 1]) / (z[i] - z[i - 1])
        m = rho[i] * _cp(T[i]) * V[i] / dt
        bb = m + G + Gp - Gp * Q[i + 1]
        P[i] = (m * T[i] + Gp * P[i + 1]) / bb
        Q[i] = G / bb
    # surface: B*T0 + es*T0^4 = D
    m = rho[0] * _cp(T[0]) * V[0] / dt
    B = m + G - G * Q[1]
    D = m * T[0] + F_abs + G * P[1]
    es = emis * SIGMA
    Ts = T[0]
    for _ in range(20):
        dT = (B * Ts + es * Ts**4 - D) / (B + 4.0 * es * Ts**3)
        Ts -= dT
        if abs(dT) < 1e-6:
            break
    Tn[0] = Ts
    for i in range(1, nz):
        Tn[i] = P[i] + Q[i] * Tn[i - 1]


@njit(cache=True)
def albedo(mu, A0, aa, ab):
    """Albedo vs cosine of incidence angle (Keihm 1984 form)."""
    th = np.arccos(min(max(mu, 0.0), 1.0))
    return A0 + aa * (th / (np.pi / 4)) ** 3 + ab * (th / (np.pi / 2)) ** 8


@njit(cache=True)
def _flat_flux(lat, lon, A0, aa, ab, flux_scale, lss):
    """Absorbed solar flux over the solar day for flat ground."""
    nt = lss.size
    F = np.zeros(nt)
    coslat = np.cos(lat)
    for s in range(nt):
        mu = coslat * np.cos(lon - lss[s])
        if mu > 0.0:
            F[s] = (1.0 - albedo(mu, A0, aa, ab)) * flux_scale[s] * mu
    return F


@njit(cache=True)
def _spin_up(lat, lon, T, Tday, tol, max_days, accelerate,
             z, rho, kc, chi, emis, A0, aa, ab, q, flux_scale, lss, dt):
    F = _flat_flux(lat, lon, A0, aa, ab, flux_scale, lss)
    return spin_up_flux(T, Tday, F, tol, max_days, accelerate,
                        z, rho, kc, chi, emis, q, dt)


@njit(cache=True)
def spin_up_flux(T, Tday, F, tol, max_days, accelerate,
                 z, rho, kc, chi, emis, q, dt):
    """
    Spin one column up to a periodic steady state under a prescribed absorbed
    flux history F (one solar day, len(F) steps of length dt). T is updated in
    place; Tday receives the final solar day.
    """
    nz = z.size
    nt = F.size
    V = np.empty(nz)
    V[0] = 0.5 * (z[1] - z[0])
    V[nz - 1] = 0.5 * (z[nz - 1] - z[nz - 2])
    for i in range(1, nz - 1):
        V[i] = 0.5 * (z[i + 1] - z[i - 1])
    Tn = np.empty(nz); k = np.empty(nz); P = np.empty(nz); Q = np.empty(nz)
    mean_prev = np.zeros(nz)
    delta_prev = np.zeros(nz)
    r_prev = 0.0
    since_jump = 0
    converged = False
    day = 0
    for day in range(1, max_days + 1):
        mean = np.zeros(nz)
        for s in range(nt):
            _step(T, Tn, z, V, rho, kc, chi, emis, q, F[s], dt, k, P, Q)
            for i in range(nz):
                T[i] = Tn[i]
                Tday[s, i] = Tn[i]
                mean[i] += Tn[i]
        mean /= nt
        delta = mean - mean_prev
        since_jump += 1
        if day > 2 and abs(delta[nz - 1]) < tol and abs(delta[nz // 2]) < tol:
            converged = True
            break
        # Extrapolate the slow deep-equilibration mode once the daily-mean
        # profile is converging geometrically at a steady rate.
        r = delta[nz - 1] / delta_prev[nz - 1] if delta_prev[nz - 1] != 0.0 else 0.0
        if (accelerate and since_jump >= 4 and 0.5 < r < 0.99 and 0.5 < r_prev < 0.99
                and abs(r - r_prev) < 0.01):
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
    return day, converged


@njit(parallel=True, cache=True)
def _batch(lats, lons, T0, Tout, days, conv, tol, max_days, accelerate, stride,
           z, rho, kc, chi, emis, A0, aa, ab, q, flux_scale, lss, dt):
    nt = lss.size
    for j in prange(lats.size):
        Tday = np.empty((nt, z.size))
        T = T0[j].copy()
        dj, cj = _spin_up(lats[j], lons[j], T, Tday, tol, max_days, accelerate,
                          z, rho, kc, chi, emis, A0, aa, ab, q, flux_scale, lss, dt)
        days[j] = dj
        conv[j] = cj
        for s in range(nt // stride):
            Tout[j, s, :] = Tday[s * stride, :]


def timed_column(model, lat, lon, **kw):
    t0 = time.perf_counter()
    res = model.run_column(lat, lon, **kw)
    res["seconds"] = time.perf_counter() - t0
    return res
