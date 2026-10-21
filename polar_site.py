"""
On-demand terrain-aware temperature for a single polar site.

    from polar_site import PolarSite
    site = PolarSite("north")
    r = site.run(lat=85.9, lon=297.0)    # r["T"]: (48 local times, 13 depths)

The site is solved at the DEM's native resolution (250 m north, 665 m south):
  - near field (default +-25 km): horizons and view factors on a native-
    resolution grid around the site
  - far field (to 250 km): horizons and view factors on a 2 km grid
  - radiance of surrounding terrain (scattered sunlight + thermal IR) is taken
    from the precomputed polar product (mercury_polar_<pole>_2km.nc), or from
    the flat-terrain table outside the product's coverage.
"""

import os
import time

import numpy as np
import xarray as xr
from numba import njit, prange

from mercury_thermal import SIGMA, Model, serial_launch
from polar_terrain import (DEMS, HERE, N_LT, N_TC, OUT_DEPTHS, SUN_RADIUS_RAD_1AU,
                           TerrainGrid, _G, _terrain_batch, depth_weights, load_dem,
                           stereo_forward)


@serial_launch
@njit(parallel=True, cache=True)
def _ray_geometry(fi0, fj0, p0s, ups, nes, nns, nus, ems, nms, P, naz, step0, nsteps,
                  wmin, fill, counts, indptr, src, wts, horizon, viewed):
    """
    Like polar_terrain._geometry but for arbitrary ray origins (fractional grid
    position fi0/fj0, 3D position p0s, frame vectors) and starting `step0`
    cells out. Returns per-azimuth horizon and the elevation up to which the
    receiver's view is blocked (`viewed`), so near and far fields can be chained.
    """
    ny, nx = P.shape[0], P.shape[1]
    dphi = 2 * np.pi / naz
    for r in prange(fi0.size):
        p0 = p0s[r]
        up = ups[r]
        ne, nn, nu = nes[r], nns[r], nus[r]
        cnt = 0
        pos = indptr[r] if fill else 0
        for k in range(naz):
            a = k * dphi
            dx = np.cos(a)
            dy = np.sin(a)
            phi = np.arctan2(dx * ems[r, 0] + dy * ems[r, 1], dx * nms[r, 0] + dy * nms[r, 1])
            m = ne * np.sin(phi) + nn * np.cos(phi)
            run = max(np.arctan2(-m, nu), viewed[r, k])
            hmax = horizon[r, k]
            last = -1
            for st in range(step0, nsteps + 1):
                fi = fi0[r] - dy * st
                fj = fj0[r] + dx * st
                if fi < 0.0 or fj < 0.0 or fi > ny - 1.0 or fj > nx - 1.0:
                    break
                ii = min(int(fi), ny - 2)
                jj = min(int(fj), nx - 2)
                wi = fi - ii
                wj = fj - jj
                v = np.empty(3)
                for c in range(3):
                    v[c] = ((1 - wi) * ((1 - wj) * P[ii, jj, c] + wj * P[ii, jj + 1, c])
                            + wi * ((1 - wj) * P[ii + 1, jj, c] + wj * P[ii + 1, jj + 1, c])
                            - p0[c])
                dist = np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
                e = np.arcsin((v[0] * up[0] + v[1] * up[1] + v[2] * up[2]) / dist)
                if e > hmax:
                    hmax = e
                if e > run:
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
            if fill:
                horizon[r, k] = hmax
                viewed[r, k] = run
        counts[r] = cnt


def ray_geometry(tg, x, y, origin, naz, step0, nsteps, horizon, viewed, wmin=1e-7):
    """Run _ray_geometry for origins at map coords (x, y); updates horizon/viewed."""
    fi0 = np.atleast_1d((tg.y[0] - y) / tg.res).astype(np.float64)
    fj0 = np.atleast_1d((x - tg.x[0]) / tg.res).astype(np.float64)
    n = fi0.size
    args = (fi0, fj0, origin["p0"], origin["up"], origin["ne"], origin["nn"], origin["nu"],
            origin["em"], origin["nm"], tg.P, naz, step0, nsteps, wmin)
    counts = np.zeros(n, dtype=np.int64)
    # counting pass reads but does not modify horizon/viewed
    _ray_geometry(*args, False, counts, np.zeros(n + 1, dtype=np.int64),
                  np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.float32), horizon, viewed)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    src = np.empty(indptr[-1], dtype=np.int64)
    wts = np.empty(indptr[-1], dtype=np.float32)
    _ray_geometry(*args, True, counts, indptr, src, wts, horizon, viewed)
    return indptr, src, wts


def sky_fraction(origin, viewed, naz):
    """Sky view factor left above the final blocked elevation of each ray."""
    dphi = 2 * np.pi / naz
    a = np.arange(naz) * dphi
    dx, dy = np.cos(a), np.sin(a)
    em, nm = origin["em"][0], origin["nm"][0]
    phi = np.arctan2(dx * em[0] + dy * em[1], dx * nm[0] + dy * nm[1])
    m = origin["ne"][0] * np.sin(phi) + origin["nn"][0] * np.cos(phi)
    nu = origin["nu"][0]
    G = lambda e: m * (0.5 * e + 0.25 * np.sin(2 * e)) + 0.5 * nu * np.sin(e) ** 2
    return float(np.sum(G(np.pi / 2) - G(viewed[0])) * dphi / np.pi)


class PolarSite:
    def __init__(self, pole, product=None, near_km=25.0, far_km=250.0, far_res_km=2.0,
                 naz=128):
        t0 = time.perf_counter()
        self.pole = pole
        self.near_km, self.far_km, self.far_res, self.naz = near_km, far_km, far_res_km, naz
        self.dem = load_dem(pole)
        self.res = DEMS[pole]["scale_km"]
        self.model = Model()
        product = product or os.path.join(HERE, "data", f"mercury_polar_{pole}_2km.nc")
        self.prod = None
        if os.path.exists(product):
            ds = xr.open_dataset(product, mask_and_scale=False)
            self.prod = dict(
                x=ds["x"].values, y=ds["y"].values,
                Ts=ds["surface_temperature_t"].values,  # int16, scale 0.02
                Rs=ds["scattered_source_t"].values)  # int16, scale 0.05
            ds.close()
        from mercury_lookup import DEFAULT_PATH
        ds = xr.open_dataset(DEFAULT_PATH)
        self.flat = dict(Ts=ds["temperature"].isel(depth=0).values,
                         dlon=float(ds["lon"][1] - ds["lon"][0]),
                         dlat=float(ds["lat"][1] - ds["lat"][0]),
                         dlt=float(ds["local_time"][1] - ds["local_time"][0]))
        ds.close()
        self.far_grid = None  # far-field grid is cached per site neighbourhood
        self.setup_seconds = time.perf_counter() - t0

    # ------------------------------------------------------------------
    def _radiance(self, x, y, lat, lon):
        """Source term X (cells x N_TC) for cells at map coords x, y (flat arrays)."""
        X = np.empty((lat.size, N_TC), dtype=np.float32)
        have = np.zeros(lat.size, dtype=bool)
        reg = self.model.regolith
        e2s = reg.emis**2 * SIGMA
        if self.prod is not None:
            p = self.prod
            dx = p["x"][1] - p["x"][0]
            j = np.rint((x - p["x"][0]) / dx).astype(int)
            i = np.rint((p["y"][0] - y) / dx).astype(int)
            ok = (i >= 0) & (i < p["y"].size) & (j >= 0) & (j < p["x"].size)
            Ts = np.full((lat.size, N_TC), -32768, dtype=np.int16)
            Rs = np.zeros((lat.size, N_TC), dtype=np.int16)
            Ts[ok] = p["Ts"][i[ok], j[ok]]
            Rs[ok] = p["Rs"][i[ok], j[ok]]
            have = Ts[:, 0] != -32768
            T = Ts[have].astype(np.float32) * 0.02
            X[have] = Rs[have].astype(np.float32) * 0.05 + e2s * T**4
        if (~have).any():
            # outside the product: flat-terrain surface temperature, no scattering
            X[~have] = e2s * self._flat_T(lat[~have], lon[~have]) ** 4
        return X

    def _flat_T(self, lat, lon):
        """Flat-terrain surface temperature (cells x N_TC), nearest table point."""
        f = self.flat
        nlon, nlat, nlt = f["Ts"].shape
        il = np.rint((np.rad2deg(lon) % 180.0) / f["dlon"]).astype(int) % nlon
        ia = np.clip(np.rint(np.abs(np.rad2deg(lat)) / f["dlat"]).astype(int), 0, nlat - 1)
        lt = (12.0 + (np.rad2deg(lon)[:, None] + 360.0 * np.arange(N_TC) / N_TC) / 15.0) % 24.0
        it = np.rint(lt / f["dlt"]).astype(int) % nlt
        return f["Ts"][il[:, None], ia[:, None], it].astype(np.float32)

    def run(self, lat, lon):
        """Temperature (N_LT x depths) and illumination statistics for one site."""
        tm = {}
        t0 = time.perf_counter()
        la, lo = np.deg2rad(lat), np.deg2rad(lon % 360)
        x, y = stereo_forward(la, lo, self.pole)
        near = TerrainGrid(self.pole, self.res, self.near_km, x, y, dem=self.dem)
        # origin frame from the near grid at the site (central cell)
        fi, fj = (near.y[0] - y) / self.res, (x - near.x[0]) / self.res
        ic, jc = int(round(fi)), int(round(fj))
        o = dict(p0=near.P[ic, jc][None].copy(), up=near.U[ic, jc][None].copy(),
                 ne=near.n_enu[ic, jc, 0:1].copy(), nn=near.n_enu[ic, jc, 1:2].copy(),
                 nu=near.n_enu[ic, jc, 2:3].copy(),
                 em=near.EM[ic, jc][None].copy(), nm=near.NM[ic, jc][None].copy())
        tm["near grid"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        if (self.far_grid is None or np.hypot(x - self.far_grid[1], y - self.far_grid[2])
                > 0.2 * self.far_km):
            far = TerrainGrid(self.pole, self.far_res, self.far_km * 1.2, x, y, dem=self.dem)
            X, Y = np.meshgrid(far.x, far.y)
            self.far_grid = (far, x, y, self._radiance(X.ravel(), Y.ravel(),
                                                       far.lat.ravel(), far.lon.ravel()))
        far, _, _, X_far = self.far_grid
        tm["far grid (cached nearby)"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        naz = self.naz
        horizon = np.full((1, naz), -np.pi / 2, dtype=np.float32)
        viewed = np.full((1, naz), -np.pi / 2, dtype=np.float32)
        n_near = int(self.near_km / self.res) - 1
        _, src_n, w_n = ray_geometry(near, x, y, o, naz, 1, n_near, horizon, viewed)
        step0 = int(np.ceil(self.near_km / self.far_res))
        _, src_f, w_f = ray_geometry(far, x, y, o, naz, step0,
                                        int(self.far_km / self.far_res), horizon, viewed)
        sky = sky_fraction(o, viewed, naz)
        tm["rays (near + far)"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        X_near = self._radiance_subset(near, src_n)
        Find = (w_n[:, None] * X_near).sum(0) + (w_f[:, None] * X_far[src_f]).sum(0)
        tm["terrain radiance"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        m, reg = self.model, self.model.regolith
        p = m.planet
        di, dw = depth_weights(m.z, OUT_DEPTHS)
        out_lt = np.arange(N_LT) * 24.0 / N_LT
        Tsurf_c = np.empty((1, N_TC), dtype=np.float32)
        Tout = np.empty((1, N_LT, OUT_DEPTHS.size), dtype=np.float32)
        sf, lf, qm = np.empty(1), np.empty(1), np.empty(1)
        days = np.empty(1, dtype=np.int64)
        _terrain_batch(np.array([near.lat[ic, jc]]), np.array([near.lon[ic, jc]]),
                       near.n_enu[ic, jc][None].copy(), o["em"], o["nm"], horizon,
                       Find[None].astype(np.float32), m.lss, p.S0 / m.r_sun**2,
                       SUN_RADIUS_RAD_1AU / m.r_sun,
                       m.grid.steps_per_day // m.grid.spinup_steps_per_day, 0.01, 2000,
                       m.z, m.rho, m.kc, reg.chi, reg.emis, reg.albedo0, reg.alb_a,
                       reg.alb_b, reg.q_geo, p.P_sol / m.lss.size, out_lt, di, dw, N_TC,
                       Tsurf_c, Tout, sf, lf, qm, days)
        tm["thermal"] = time.perf_counter() - t0
        return dict(T=Tout[0], local_time=out_lt, depth=OUT_DEPTHS,
                    horizon_deg=np.rad2deg(horizon[0]), sky_view_factor=sky,
                    sun_visible_fraction=sf[0], direct_lit_fraction=lf[0],
                    mean_direct_flux=qm[0], permanent_shadow=bool(lf[0] == 0),
                    elevation_m=float(near.elevation[ic, jc]),
                    slope_deg=float(near.slope_deg[ic, jc]), timing=tm,
                    seconds=sum(tm.values()),
                    # inputs to the final column solve (used by site_script.py)
                    site_inputs=dict(lat=float(near.lat[ic, jc]), lon=float(near.lon[ic, jc]),
                                     n_enu=near.n_enu[ic, jc].astype(float), em=o["em"][0].astype(float),
                                     nm=o["nm"][0].astype(float), horizon=horizon[0].astype(float),
                                     Find=Find.astype(float)))

    def _radiance_subset(self, tg, src):
        """Radiance of only the near-grid cells that the site actually sees."""
        if src.size == 0:
            return np.zeros((0, N_TC), dtype=np.float32)
        u, inv = np.unique(src, return_inverse=True)
        nx = tg.shape[1]
        return self._radiance(tg.x[u % nx], tg.y[u // nx], tg.lat.ravel()[u],
                              tg.lon.ravel()[u])[inv]
