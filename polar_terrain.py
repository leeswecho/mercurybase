"""
Terrain-aware thermal model for Mercury's polar regions.

Extends the flat 1D model (mercury_thermal.py) with real MESSENGER topography:
  - horizon shadowing by surrounding terrain (ray-traced, planet curvature
    included exactly via 3D body-fixed positions)
  - local slope and aspect
  - finite solar disk (partial visibility at the horizon)
  - indirect illumination: sunlight scattered by, and thermal infrared emitted
    from, surrounding terrain, using exact Lambertian view factors from the
    same rays. Infrared from the terrain depends on its temperature, so the
    thermal solution is iterated (two passes).

Topography
  north: MESSENGER MLA gridded DEM HDEM_75N_250M (250 m/pix, 75-90 N)
  south: MESSENGER MDIS stereo DEM MSGR_DEM_USG_SP_I_V02 (665 m/pix, 55-90 S).
         The south polar DEM has no stereo coverage within ~2.5 deg of the
         pole; that area is interpolated and terrain results there are not
         trustworthy (flagged by `dem_quality` in the output).

Usage
  python polar_terrain.py --pole north [--res 2] [--lat-min 80]
  python polar_terrain.py --pole south
"""

import argparse
import os
import time

import numpy as np
import xarray as xr
from numba import njit, prange
from scipy import ndimage

from mercury_thermal import SIGMA, Grid, Model, albedo, orbit, serial_launch, spin_up_flux

R_KM = 2439.4  # projection sphere radius of both DEMs
SUN_RADIUS_RAD_1AU = 0.0046524  # angular radius of the Sun at 1 AU
HERE = os.path.dirname(os.path.abspath(__file__))
DEM_DIR = os.path.join(HERE, "data", "dem")

# Array-index mapping, with map coordinates x = rho sin(lon),
# y = -rho cos(lon) (north) / +rho cos(lon) (south), east longitude:
#   row = center - y / scale ; col = center + col_sign * x / scale
# Verified against the USGS global DEM (correlation 0.95-0.98 north, 0.999
# south) and against IAU crater centres, which the Gazetteer gives in WEST
# longitude for Mercury: Prokofiev 86.0 N 63.7 E, Kandinsky 87.9 N 78.8 E.
# (An earlier col_sign = -1 for the north mirrored it east-west.)
#
# cap_lat: equatorward edge of each polar product. The north MLA DEM stops
# at 75 N, so terrain equatorward of it (needed for horizons near the cap
# edge) is filled from the global DEM (fill_global), blended over 0.5 deg.
DEMS = {
    "north": dict(
        file="hdem_75n_250m.img", n=5120, scale_km=0.25, center=2559.5,
        ref_radius_m=2440000.0, col_sign=+1, lat_limit=75.0, cap_lat=75.0, fill_global=True,
        url="https://pds-geosciences.wustl.edu/MESSENGER/mess-e_v_h-mla-3_4-cdr_rdr-data-v2/"
            "messmla_2101/gdr/img/hdem_75n_250m.img"),
    "south": dict(
        file="MSGR_DEM_USG_SP_I_V02.IMG", n=4625, scale_km=0.66524315270546, center=2312.0,
        ref_radius_m=2439400.0, col_sign=+1, lat_limit=55.0, cap_lat=70.0, gap_colat_deg=2.5,
        url="https://pds-imaging.jpl.nasa.gov/data/messenger/MESSDEM_1001/DEM/POLAR/IMG/"
            "MSGR_DEM_USG_SP_I_V02.IMG"),
}

OUT_DEPTHS = np.array([0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
N_LT = 48  # output local times per solar day
N_TC = 180  # coarse time samples per solar day for indirect fluxes


# ----------------------------------------------------------------------------
# DEM and working grid
# ----------------------------------------------------------------------------

def load_dem(pole):
    d = DEMS[pole]
    path = os.path.join(DEM_DIR, d["file"])
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing; download from {d['url']}")
    dn = np.fromfile(path, dtype="<i2").reshape(d["n"], d["n"])
    radius = (dn.astype(np.float64) * 0.5 + d["ref_radius_m"]) / 1000.0  # km
    bad = dn == -32768
    if bad.any():
        idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
        radius = radius[tuple(idx)]
    return radius


_GLOBAL_SLOPES = []


def global_slope_map():
    """The global 2 km elevation/slope map (loaded once)."""
    if not _GLOBAL_SLOPES:
        from global_terrain import SlopeMap  # global_terrain imports this module
        _GLOBAL_SLOPES.append(SlopeMap())
    return _GLOBAL_SLOPES[0]


def stereo_inverse(x, y, pole):
    rho = np.hypot(x, y)
    colat = 2 * np.arctan(rho / (2 * R_KM))
    if pole == "north":
        return np.pi / 2 - colat, np.arctan2(x, -y) % (2 * np.pi)
    return -(np.pi / 2 - colat), np.arctan2(x, y) % (2 * np.pi)


def stereo_forward(lat, lon, pole):
    rho = 2 * R_KM * np.tan((np.pi / 2 - np.abs(lat)) / 2)
    sgn = -1.0 if pole == "north" else 1.0
    return rho * np.sin(lon), sgn * rho * np.cos(lon)


class TerrainGrid:
    """
    Square polar-stereographic working grid (cell size `res_km`) centred on
    map point (x0, y0) with half-width `half_km`, sampled from the DEM after
    box-filtering to the working resolution.
    """

    def __init__(self, pole, res_km, half_km, x0=0.0, y0=0.0, dem=None):
        d = DEMS[pole]
        self.pole, self.res = pole, res_km
        n = 2 * int(np.ceil(half_km / res_km))
        self.shape = (n, n)
        self.x = x0 + (np.arange(n) - (n - 1) / 2) * res_km
        self.y = y0 + ((n - 1) / 2 - np.arange(n)) * res_km
        X, Y = np.meshgrid(self.x, self.y)
        self.lat, self.lon = stereo_inverse(X, Y, pole)

        if dem is None:
            dem = load_dem(pole)
        f = res_km / d["scale_km"]
        if f > 1.5:
            dem = ndimage.uniform_filter(dem, size=int(round(f)), mode="nearest")
        rows = d["center"] - Y / d["scale_km"]
        cols = d["center"] + d["col_sign"] * X / d["scale_km"]
        self.radius = ndimage.map_coordinates(dem, [rows, cols], order=1, mode="nearest")
        if d.get("fill_global"):
            self._fill_global(d["lat_limit"])
        co, so = np.cos(self.lon), np.sin(self.lon)
        if pole == "north":  # map-frame (x right, y up) unit vectors of local east / north
            EM, NM = np.stack([co, so], -1), np.stack([-so, co], -1)
        else:
            EM, NM = np.stack([co, -so], -1), np.stack([so, co], -1)
        self._derive(EM, NM)
        self.dem_quality = np.ones(self.shape, dtype=np.int8)
        if "gap_colat_deg" in d:
            self.dem_quality[90 - np.rad2deg(np.abs(self.lat)) < d["gap_colat_deg"]] = 0

    def _fill_global(self, lat_limit, blend_deg=0.5):
        """Terrain equatorward of the polar DEM from the global DEM (2 km), blended at the edge."""
        latd = np.rad2deg(np.abs(self.lat))
        w = np.clip((latd - lat_limit) / blend_deg, 0.0, 1.0)  # 1 = polar DEM only
        m = w < 1.0
        if not m.any():
            return
        rg = R_KM + global_slope_map().elevation_bilinear(latd[m] * np.sign(self.lat[m]),
                                                          np.rad2deg(self.lon[m])) / 1000.0
        self.radius[m] = w[m] * self.radius[m] + (1.0 - w[m]) * rg

    def _derive(self, EM, NM):
        """Frames, 3D positions and surface normals from lat, lon (rad) and radius (km)."""
        self.elevation = (self.radius - R_KM) * 1000.0  # m, relative to 2439.4 km
        cl, sl = np.cos(self.lat), np.sin(self.lat)
        co, so = np.cos(self.lon), np.sin(self.lon)
        self.U = np.stack([cl * co, cl * so, sl], -1)
        self.E = np.stack([-so, co, np.zeros_like(co)], -1)
        self.N = np.stack([-sl * co, -sl * so, cl], -1)
        self.P = self.radius[..., None] * self.U
        self.EM, self.NM = EM, NM

        # surface normal from 3D positions of neighbouring cells
        dPx = np.gradient(self.P, axis=1)
        dPy = -np.gradient(self.P, axis=0)
        nrm = np.cross(dPx, dPy)
        nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
        nrm *= np.sign(np.sum(nrm * self.U, -1))[..., None]
        self.NRM = nrm
        self.n_enu = np.stack([np.sum(nrm * self.E, -1), np.sum(nrm * self.N, -1),
                               np.sum(nrm * self.U, -1)], -1)
        self.slope_deg = np.rad2deg(np.arccos(np.clip(self.n_enu[..., 2], -1, 1)))

    def flat(self, a):
        return a.reshape((-1,) + a.shape[2:])


# ----------------------------------------------------------------------------
# Geometry: horizons and view factors
# ----------------------------------------------------------------------------

@njit(cache=True)
def _G(e, m, nu):
    # integral over elevation of (d.n) cos(e) for a receiver with normal
    # components m (horizontal, along the ray azimuth) and nu (vertical)
    return m * (0.5 * e + 0.25 * np.sin(2 * e)) + 0.5 * nu * np.sin(e) ** 2


@serial_launch
@njit(parallel=True, cache=True)
def _geometry(rec, P, U, E3, N3, NRM, EM, NM, naz, nsteps, wmin, do_vf, fill,
              counts, indptr, src, wts, horizon, sky):
    ny, nx = P.shape[0], P.shape[1]
    dphi = 2 * np.pi / naz
    for r in prange(rec.size):
        i0 = rec[r] // nx
        j0 = rec[r] % nx
        p0 = P[i0, j0]
        up = U[i0, j0]
        nv = NRM[i0, j0]
        ne = nv[0] * E3[i0, j0, 0] + nv[1] * E3[i0, j0, 1] + nv[2] * E3[i0, j0, 2]
        nn = nv[0] * N3[i0, j0, 0] + nv[1] * N3[i0, j0, 1] + nv[2] * N3[i0, j0, 2]
        nu = nv[0] * up[0] + nv[1] * up[1] + nv[2] * up[2]
        cnt = 0
        pos = indptr[r] if fill else 0
        sk = 0.0
        for k in range(naz):
            a = k * dphi
            dx = np.cos(a)
            dy = np.sin(a)
            ex = dx * EM[i0, j0, 0] + dy * EM[i0, j0, 1]
            nx_ = dx * NM[i0, j0, 0] + dy * NM[i0, j0, 1]
            phi = np.arctan2(ex, nx_)
            m = ne * np.sin(phi) + nn * np.cos(phi)
            run = np.arctan2(-m, nu)  # receiver's own tangent plane
            hmax = -np.pi / 2
            last = -1
            for st in range(1, nsteps + 1):
                fi = i0 - dy * st
                fj = j0 + dx * st
                if fi < 0.0 or fj < 0.0 or fi > ny - 1.0 or fj > nx - 1.0:
                    break
                ii = min(int(fi), ny - 2)
                jj = min(int(fj), nx - 2)
                wi = fi - ii
                wj = fj - jj
                v0 = 0.0
                v1 = 0.0
                v2 = 0.0
                for c in range(3):
                    val = ((1 - wi) * ((1 - wj) * P[ii, jj, c] + wj * P[ii, jj + 1, c])
                           + wi * ((1 - wj) * P[ii + 1, jj, c] + wj * P[ii + 1, jj + 1, c])
                           - p0[c])
                    if c == 0:
                        v0 = val
                    elif c == 1:
                        v1 = val
                    else:
                        v2 = val
                dist = np.sqrt(v0 * v0 + v1 * v1 + v2 * v2)
                e = np.arcsin((v0 * up[0] + v1 * up[1] + v2 * up[2]) / dist)
                if e > hmax:
                    hmax = e
                if do_vf and e > run:
                    w = (_G(e, m, nu) - _G(run, m, nu)) * dphi / np.pi
                    run = e
                    s = int(fi + 0.5) * nx + int(fj + 0.5)
                    if s == last:
                        if fill:
                            wts[pos - 1] += w
                    elif w > wmin:
                        if fill:
                            src[pos] = s
                            wts[pos] = w
                            pos += 1
                        cnt += 1
                        last = s
            horizon[r, k] = hmax
            if do_vf:
                sk += (_G(np.pi / 2, m, nu) - _G(run, m, nu)) * dphi / np.pi
        counts[r] = cnt
        sky[r] = sk


def compute_geometry(tg, rec, naz=64, max_dist_km=250.0, do_vf=True, wmin=1e-7):
    """Horizons (rad, by map-angle bin) and sparse view factors for receivers `rec`."""
    nsteps = int(max_dist_km / tg.res)
    n = rec.size
    counts = np.zeros(n, dtype=np.int64)
    horizon = np.empty((n, naz), dtype=np.float32)
    sky = np.zeros(n)
    dummy_i = np.zeros(1, dtype=np.int64)
    dummy_w = np.zeros(1, dtype=np.float32)
    args = (tg.P, tg.U, tg.E, tg.N, tg.NRM, tg.EM, tg.NM, naz, nsteps, wmin)
    _geometry(rec, *args, do_vf, False, counts, np.zeros(n + 1, dtype=np.int64),
              dummy_i, dummy_w, horizon, sky)
    if not do_vf:
        return dict(horizon=horizon)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    src = np.empty(indptr[-1], dtype=np.int32 if tg.P.shape[0] ** 2 < 2**31 else np.int64)
    wts = np.empty(indptr[-1], dtype=np.float32)
    _geometry(rec, *args, True, True, counts, indptr, src, wts, horizon, sky)
    return dict(horizon=horizon, indptr=indptr, src=src, wts=wts, sky=sky)


# ----------------------------------------------------------------------------
# Illumination
# ----------------------------------------------------------------------------

@njit(cache=True)
def _sun(lat, lon, n_enu, em, nm, hor, lss, fs, rsun):
    """Sun-disk visible fraction, cosine of incidence and incident direct flux."""
    naz = hor.size
    h = lon - lss
    se = -np.sin(h)
    sn = -np.sin(lat) * np.cos(h)
    su = np.cos(lat) * np.cos(h)
    vx = se * em[0] + sn * nm[0]
    vy = se * em[1] + sn * nm[1]
    kf = (np.arctan2(vy, vx) % (2 * np.pi)) / (2 * np.pi / naz)
    k0 = int(kf) % naz
    w = kf - np.floor(kf)
    hz = (1 - w) * hor[k0] + w * hor[(k0 + 1) % naz]
    x = (np.arcsin(su) - hz) / rsun
    if x <= -1.0:
        return 0.0, 0.0, 0.0
    f = 1.0 if x >= 1.0 else 1.0 - (np.arccos(x) - x * np.sqrt(1 - x * x)) / np.pi
    mu = n_enu[0] * se + n_enu[1] * sn + n_enu[2] * su
    if mu <= 0.0:
        return f, 0.0, 0.0
    return f, mu, fs * f * mu


@serial_launch
@njit(parallel=True, cache=True)
def _reflected_source(lat, lon, n_enu, EM, NM, hor, lss, fs, rsun, A0, aa, ab, absorb,
                      incident=False):
    """
    Scattered sunlight leaving each cell, pre-multiplied by receiver
    absorptivity (or, with incident=True, the incident direct flux itself).
    """
    n = lat.size
    nt = lss.size
    out = np.zeros((n, nt), dtype=np.float32)
    for i in prange(n):
        for s in range(nt):
            f, mu, Q = _sun(lat[i], lon[i], n_enu[i], EM[i], NM[i], hor[i], lss[s], fs[s], rsun[s])
            if Q > 0.0:
                out[i, s] = Q if incident else absorb * albedo(mu, A0, aa, ab) * Q
    return out


@serial_launch
@njit(parallel=True, cache=True)
def _radiance(R, T, e2s):
    """Source term leaving each cell: scattered part R + eps^2 sigma T^4 (float32)."""
    X = np.empty_like(R)
    for i in prange(R.shape[0]):
        for t in range(R.shape[1]):
            X[i, t] = R[i, t] + e2s * T[i, t] ** 4
    return X


@serial_launch
@njit(parallel=True, cache=True)
def _sparse_apply(indptr, src, wts, X):
    """Y[r, :] = sum_j w_rj X[src_j, :]"""
    n = indptr.size - 1
    nt = X.shape[1]
    Y = np.zeros((n, nt), dtype=np.float32)
    for r in prange(n):
        for p in range(indptr[r], indptr[r + 1]):
            s = src[p]
            w = wts[p]
            for t in range(nt):
                Y[r, t] += w * X[s, t]
    return Y


# ----------------------------------------------------------------------------
# Thermal solution with terrain
# ----------------------------------------------------------------------------

@serial_launch
@njit(parallel=True, cache=True)
def _terrain_batch(lat, lon, n_enu, EM, NM, hor, Find, lss, fs, rsun, stride, tol,
                   max_days, z, rho, kc, chi, emis, A0, aa, ab, q, dt,
                   out_lt, dep_i, dep_w, ntc,
                   Tsurf_c, Tout, sun_frac, lit_frac, Qmean, days):
    n = lat.size
    nt = lss.size
    nz = z.size
    nfc = Find.shape[1]
    for i in prange(n):
        F = np.empty(nt)
        fsum = 0.0
        lit = 0
        qsum = 0.0
        for s in range(nt):
            f, mu, Q = _sun(lat[i], lon[i], n_enu[i], EM[i], NM[i], hor[i], lss[s], fs[s], rsun[s])
            Fd = (1.0 - albedo(mu, A0, aa, ab)) * Q if Q > 0.0 else 0.0
            tf = s * nfc / nt
            c0 = int(tf)
            wc = tf - c0
            F[s] = Fd + (1 - wc) * Find[i, c0 % nfc] + wc * Find[i, (c0 + 1) % nfc]
            fsum += f
            qsum += Q
            if Q > 0.0:
                lit += 1
        sun_frac[i] = fsum / nt
        lit_frac[i] = lit / nt
        Qmean[i] = qsum / nt
        T0 = 1.05 * ((np.mean(F) + q) / (emis * SIGMA)) ** 0.25
        T = np.full(nz, T0)
        Fc = F[::stride].copy()
        Tc = np.empty((Fc.size, nz))
        spin_up_flux(T, Tc, Fc, tol, max_days, True, z, rho, kc, chi, emis, q, dt * stride)
        Tday = np.empty((nt, nz))
        d, _ = spin_up_flux(T, Tday, F, tol, max_days, True, z, rho, kc, chi, emis, q, dt)
        days[i] = d
        for c in range(ntc):
            Tsurf_c[i, c] = Tday[c * nt // ntc, 0]
        lon_deg = lon[i] * 180.0 / np.pi
        for l in range(out_lt.size):
            sf = ((15.0 * (out_lt[l] - 12.0) - lon_deg) / 360.0 * nt) % nt
            s0 = int(sf)
            ws = sf - s0
            s1 = (s0 + 1) % nt
            s0 = s0 % nt
            for k in range(dep_i.size):
                a = dep_i[k]
                b = min(a + 1, nz - 1)
                Ta = (1 - ws) * Tday[s0, a] + ws * Tday[s1, a]
                Tb = (1 - ws) * Tday[s0, b] + ws * Tday[s1, b]
                Tout[i, l, k] = (1 - dep_w[k]) * Ta + dep_w[k] * Tb


def depth_weights(z, depths):
    i = np.clip(np.searchsorted(z, depths) - 1, 0, z.size - 2)
    w = np.clip((depths - z[i]) / (z[i + 1] - z[i]), 0.0, 1.0)
    return i.astype(np.int64), w


def flat_surface_temperature(lat, lon, t_frac, table_path=None):
    """Surface temperature from the flat-terrain global table (nearest grid point)."""
    from mercury_lookup import DEFAULT_PATH
    ds = xr.open_dataset(table_path or DEFAULT_PATH)
    Ts = ds["temperature"].isel(depth=0).values  # lon, lat, lt
    lons, lats, lts = ds["lon"].values, ds["lat"].values, ds["local_time"].values
    ds.close()
    lon_deg = np.rad2deg(lon) % 180.0
    il = np.rint(lon_deg / (lons[1] - lons[0])).astype(int) % lons.size
    ia = np.clip(np.rint(np.abs(np.rad2deg(lat)) / (lats[1] - lats[0])).astype(int), 0, lats.size - 1)
    out = np.empty((lat.size, t_frac.size), dtype=np.float32)
    dlt = lts[1] - lts[0]
    for c, tf in enumerate(t_frac):
        lt = (12.0 + (np.rad2deg(lon) + 360.0 * tf) / 15.0) % 24.0
        it = np.rint(lt / dlt).astype(int) % lts.size
        out[:, c] = Ts[il, ia, it]
    return out


class PolarModel:
    """Run the terrain-aware model for receivers on a TerrainGrid."""

    def __init__(self, tg, naz=64, max_dist_km=250.0):
        self.tg = tg
        self.naz = naz
        self.max_dist = max_dist_km
        self.model = Model()
        self.model_fast = Model(grid=Grid(steps_per_day=720, spinup_steps_per_day=180))
        p = self.model.planet
        _, r_c, self.lss_c = orbit(p, N_TC)
        self.fs_c = p.S0 / r_c**2
        self.rsun_c = SUN_RADIUS_RAD_1AU / r_c
        reg = self.model.regolith
        # hemispherically averaged absorptivity for scattered light
        mu = np.linspace(1e-4, 1, 2001)
        A = np.array([albedo(m, reg.albedo0, reg.alb_a, reg.alb_b) for m in mu])
        self.absorb_vis = float(1 - np.trapezoid(A * 2 * mu, mu))

    def cells(self, mask):
        return np.flatnonzero(mask.ravel()).astype(np.int64)

    def run(self, receivers, sources_T=None, n_pass=2, fast_first=True, log=print,
            checkpoint_dir=None, chunk=20000):
        """
        receivers: flat cell indices to solve. Other cells within reach act as
        sources only; their temperature comes from `sources_T` (array over all
        cells x N_TC) or, by default, the flat-terrain table.

        With checkpoint_dir, thermal results are saved per chunk of receivers
        and reused if the run is restarted with the same configuration.
        """
        tg = self.tg
        F = lambda a: tg.flat(a)
        ncell = tg.lat.size
        reg, g = self.model.regolith, self.model.grid
        tl = {}
        t0 = time.perf_counter()
        allc = np.arange(ncell, dtype=np.int64)
        geo_all = compute_geometry(tg, allc, self.naz, self.max_dist, do_vf=False)
        tl["horizons (all cells)"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        geo = compute_geometry(tg, receivers, self.naz, self.max_dist, do_vf=True)
        tl["view factors (receivers)"] = time.perf_counter() - t0
        log(f"  view-factor entries: {geo['wts'].size:,} "
            f"({geo['wts'].size / receivers.size:.0f} per receiver)")

        lat, lon = F(tg.lat), F(tg.lon)
        n_enu, EM, NM = F(tg.n_enu), F(tg.EM), F(tg.NM)
        t0 = time.perf_counter()
        R = _reflected_source(lat, lon, n_enu, EM, NM, geo_all["horizon"], self.lss_c,
                              self.fs_c, self.rsun_c, reg.albedo0, reg.alb_a, reg.alb_b,
                              self.absorb_vis)
        tl["scattered-light sources"] = time.perf_counter() - t0

        # initial source temperatures
        t0 = time.perf_counter()
        t_frac = np.arange(N_TC) / N_TC
        if sources_T is None:
            Tsrc = flat_surface_temperature(lat, lon, t_frac)
        else:
            Tsrc = sources_T.astype(np.float32).copy()
        # first guess for receivers: instantaneous radiative equilibrium with
        # their own direct sunlight (floored), since the flat table knows
        # nothing about shadows
        r = receivers
        Qr = _reflected_source(lat[r], lon[r], n_enu[r], EM[r], NM[r], geo_all["horizon"][r],
                               self.lss_c, self.fs_c, self.rsun_c, reg.albedo0, reg.alb_a,
                               reg.alb_b, self.absorb_vis, True)
        Qr *= np.float32((1 - reg.albedo0) / (reg.emis * SIGMA))
        Tsrc[receivers] = np.maximum(Qr ** np.float32(0.25), np.float32(60.0))
        del Qr
        hor_rec = geo_all["horizon"][receivers]
        R_rec = R[receivers]
        del geo_all
        tl["initial source temperatures"] = time.perf_counter() - t0

        e2s = reg.emis**2 * SIGMA
        res = None
        for ip in range(n_pass):
            t0 = time.perf_counter()
            X = _radiance(R, Tsrc, np.float32(e2s))
            Find = _sparse_apply(geo["indptr"], geo["src"], geo["wts"], X)
            del X
            last = ip == n_pass - 1
            full = last or not fast_first
            model = self.model if full else self.model_fast
            tol = 0.01 if full else 0.05
            ck = None if checkpoint_dir is None else os.path.join(checkpoint_dir, f"pass{ip + 1}")
            res_new = self._thermal(model, receivers, geo["horizon"], Find, tol,
                                    keep_T=last, ckpt=ck, chunk=chunk, log=log)
            if res is not None:
                d = res_new["Tsurf_c"] - res["Tsurf_c"]
                log(f"  pass {ip + 1}: surface T change vs previous pass "
                    f"mean {np.abs(d).mean():.2f} K, 99th pct {np.percentile(np.abs(d), 99):.2f} K, "
                    f"max {np.abs(d).max():.2f} K")
            res = res_new
            res["Find"] = Find
            Tsrc[receivers] = res["Tsurf_c"]
            tl[f"thermal pass {ip + 1} ({'full' if full else 'fast'})"] = \
                time.perf_counter() - t0
        res.update(horizon=geo["horizon"], sky=geo["sky"], R=R_rec, timing=tl,
                   horizon_rec=hor_rec)
        return res

    def _thermal(self, model, rec, horizon, Find, tol, keep_T=True, ckpt=None,
                 chunk=20000, log=print):
        """Thermal solution for receivers in chunks (optionally checkpointed)."""
        parts = []
        t0 = time.perf_counter()
        for a in range(0, rec.size, chunk):
            b = min(a + chunk, rec.size)
            path = None if ckpt is None else os.path.join(ckpt, f"{a:07d}_{b:07d}.npz")
            if path and os.path.exists(path):
                with np.load(path) as f:
                    part = {k: f[k] for k in f.files}
            else:
                part = self._thermal_chunk(model, rec[a:b], horizon[a:b], Find[a:b], tol)
                if not keep_T:
                    del part["Tout"]
                if path:
                    os.makedirs(ckpt, exist_ok=True)
                    np.savez(path + ".tmp.npz", **part)
                    os.replace(path + ".tmp.npz", path)
            parts.append(part)
            el = time.perf_counter() - t0
            log(f"    {b:,}/{rec.size:,} receivers, {el:.0f} s elapsed, "
                f"~{el / b * (rec.size - b):.0f} s remaining")
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}

    def _thermal_chunk(self, model, rec, horizon, Find, tol):
        tg, reg = self.tg, model.regolith
        F = lambda a: tg.flat(a)[rec].copy()
        n = rec.size
        p = model.planet
        lss, fs = model.lss, p.S0 / model.r_sun**2
        rsun = SUN_RADIUS_RAD_1AU / model.r_sun
        dt = p.P_sol / lss.size
        stride = model.grid.steps_per_day // model.grid.spinup_steps_per_day
        di, dw = depth_weights(model.z, OUT_DEPTHS)
        out_lt = np.arange(N_LT) * 24.0 / N_LT
        Tsurf_c = np.empty((n, N_TC), dtype=np.float32)
        Tout = np.empty((n, N_LT, OUT_DEPTHS.size), dtype=np.float32)
        sun_frac = np.empty(n)
        lit_frac = np.empty(n)
        Qmean = np.empty(n)
        days = np.empty(n, dtype=np.int64)
        _terrain_batch(F(tg.lat), F(tg.lon), F(tg.n_enu), F(tg.EM), F(tg.NM), horizon, Find,
                       lss, fs, rsun, stride, tol, 2000, model.z, model.rho, model.kc,
                       reg.chi, reg.emis, reg.albedo0, reg.alb_a, reg.alb_b, reg.q_geo, dt,
                       out_lt, di, dw, N_TC, Tsurf_c, Tout, sun_frac, lit_frac, Qmean, days)
        return dict(Tsurf_c=Tsurf_c, Tout=Tout, sun_frac=sun_frac, lit_frac=lit_frac,
                    Qmean=Qmean, days=days)


# ----------------------------------------------------------------------------
# Driver: polar product
# ----------------------------------------------------------------------------

def pack(a, scale):
    """Pack to int16 with the given scale factor (NaN -> -32768), in chunks."""
    a = np.asarray(a, dtype=np.float32)
    out = np.empty(a.shape, dtype=np.int16)
    fo, fa = out.reshape(-1), a.reshape(-1)
    for i in range(0, fa.size, 1 << 24):
        c = np.rint(fa[i:i + (1 << 24)] / np.float32(scale))
        c = np.where(np.isfinite(c), np.clip(c, -32767, 32767), -32768)
        fo[i:i + (1 << 24)] = c
    return out


def build_polar_product(pole, res_km=2.0, lat_min=None, margin_deg=5.0, out=None, log=print):
    t_start = time.perf_counter()
    d = DEMS[pole]
    lat_min = d["cap_lat"] if lat_min is None else lat_min
    # sources beyond the receivers, out to ~ray reach (250 km ~ 5.9 deg)
    lat_edge = lat_min - margin_deg
    if not d.get("fill_global"):
        lat_edge = max(lat_edge, d["lat_limit"])
    half = 2 * R_KM * np.tan(np.deg2rad(90 - lat_edge) / 2)
    tg = TerrainGrid(pole, res_km, half)
    log(f"{pole}: grid {tg.shape[0]}x{tg.shape[1]} at {res_km} km "
        f"({time.perf_counter() - t_start:.1f} s)")
    pm = PolarModel(tg)
    rec_mask = np.abs(np.rad2deg(tg.lat)) >= lat_min
    rec = pm.cells(rec_mask)
    log(f"  receivers (|lat| >= {lat_min}): {rec.size:,}")
    ckpt = os.path.join(HERE, "data", "checkpoints", f"{pole}_{res_km:g}km_{lat_min:g}")
    res = pm.run(rec, log=log, checkpoint_dir=ckpt)
    for k, v in res["timing"].items():
        log(f"  {k:32s} {v:8.1f} s")

    # crop to the receiver region and assemble gridded output
    ny, nx = tg.shape
    rows = np.flatnonzero(rec_mask.any(1))
    cols = np.flatnonzero(rec_mask.any(0))
    r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    sl = (slice(r0, r1), slice(c0, c1))

    def grid_of(vals, extra=()):
        g = np.full((ny * nx,) + extra, np.nan, dtype=np.float32)
        g[rec] = vals
        return g.reshape((ny, nx) + extra)[sl]

    lt = np.arange(N_LT) * 24.0 / N_LT
    T = grid_of(res["Tout"], (N_LT, OUT_DEPTHS.size))
    hz = grid_of(np.rad2deg(res["horizon"]), (pm.naz,))
    Tsc = grid_of(res["Tsurf_c"], (N_TC,))
    Rg = grid_of(res["R"], (N_TC,))
    ds = xr.Dataset(
        {
            "temperature": (("y", "x", "local_time", "depth"), pack(T, 0.02),
                            {"units": "K", "scale_factor": 0.02, "_FillValue": np.int16(-32768),
                             "long_name": "regolith temperature with terrain"}),
            "elevation": (("y", "x"), tg.elevation[sl].astype(np.float32),
                          {"units": "m", "long_name": "height above 2439.4 km sphere"}),
            "slope": (("y", "x"), tg.slope_deg[sl].astype(np.float32), {"units": "degree"}),
            "lat": (("y", "x"), np.rad2deg(tg.lat[sl]).astype(np.float32), {"units": "degree_north"}),
            "lon": (("y", "x"), np.rad2deg(tg.lon[sl]).astype(np.float32), {"units": "degree_east"}),
            "sky_view_factor": (("y", "x"), grid_of(res["sky"])),
            "sun_visible_fraction": (("y", "x"), grid_of(res["sun_frac"]), {
                "long_name": "time-mean fraction of the solar disk above the horizon "
                             "(independent of local slope)"}),
            "direct_lit_fraction": (("y", "x"), grid_of(res["lit_frac"]), {
                "long_name": "fraction of the solar day the surface receives direct sunlight"}),
            "mean_direct_flux": (("y", "x"), grid_of(res["Qmean"]), {
                "units": "W m-2", "long_name": "time-mean direct solar flux on the local surface"}),
            "permanent_shadow": (("y", "x"), (grid_of(res["lit_frac"]) == 0).astype(np.int8)),
            "dem_quality": (("y", "x"), tg.dem_quality[sl], {
                "long_name": "1 = measured topography, 0 = interpolated (unreliable)"}),
            "horizon": (("y", "x", "azimuth_bin"), pack(hz, 0.01), {
                "units": "degree", "scale_factor": 0.01, "_FillValue": np.int16(-32768),
                "long_name": "terrain horizon elevation by map-frame direction "
                             "(bin k: angle 360k/naz from +x, counterclockwise)"}),
            "surface_temperature_t": (("y", "x", "day_phase"), pack(Tsc, 0.02), {
                "units": "K", "scale_factor": 0.02, "_FillValue": np.int16(-32768),
                "long_name": "surface temperature vs time since perihelion (day_phase)"}),
            "scattered_source_t": (("y", "x", "day_phase"), pack(Rg, 0.05), {
                "units": "W m-2", "scale_factor": 0.05, "_FillValue": np.int16(-32768),
                "long_name": "scattered sunlight leaving the surface x absorptivity"}),
        },
        coords={
            "x": ("x", tg.x[c0:c1], {"units": "km", "long_name": "polar stereographic x"}),
            "y": ("y", tg.y[r0:r1], {"units": "km", "long_name": "polar stereographic y"}),
            "local_time": ("local_time", lt, {"units": "hour", "long_name": "mean local solar time"}),
            "depth": ("depth", OUT_DEPTHS, {"units": "m"}),
            "day_phase": ("day_phase", np.arange(N_TC) / N_TC,
                          {"long_name": "time since reference perihelion / solar day"}),
        },
        attrs={
            "title": f"Mercury {pole} polar regolith temperature with terrain",
            "pole": pole, "resolution_km": res_km, "lat_min": lat_min,
            "projection": "polar stereographic, sphere R=2439.4 km, true scale at pole; "
                          "north: x=rho sin(lon), y=-rho cos(lon); south: y=+rho cos(lon)",
            "dem": DEMS[pole]["file"] + (" (+ USGS global DEM equatorward of "
                                         f"{d['lat_limit']:g} deg)" if d.get("fill_global") else ""),
            "n_azimuth": pm.naz, "max_ray_km": pm.max_dist,
            "method": "1D heat conduction per cell; direct sunlight with ray-traced horizons, "
                      "finite solar disk, slopes; single-scattered sunlight and thermal IR "
                      "from visible terrain via Lambertian view factors; 2 passes",
        },
    )
    enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    out = out or os.path.join(HERE, "data", f"mercury_polar_{pole}_{res_km:g}km.nc")
    ds.to_netcdf(out, encoding=enc)
    log(f"  wrote {out} ({os.path.getsize(out) / 1e6:.0f} MB); total "
        f"{time.perf_counter() - t_start:.0f} s; unconverged: {(res['days'] >= 2000).sum()}")
    return ds, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pole", choices=["north", "south"], required=True)
    ap.add_argument("--res", type=float, default=2.0, help="grid resolution, km")
    ap.add_argument("--lat-min", type=float, default=None,
                    help="equatorward edge of the product (default: per pole, 75 N / 70 S)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    build_polar_product(a.pole, a.res, a.lat_min, out=a.out)


if __name__ == "__main__":
    main()
