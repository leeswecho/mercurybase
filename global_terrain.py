"""
Slope-aware temperatures for the whole planet outside the polar products.

Outside the polar caps the Sun climbs high, so distant terrain rarely shadows a
site; the main terrain effect is each surface patch's own tilt. This module
models that with a precomputed *slope table*: 1D thermal runs for a tilted
surface over a grid of (latitude, longitude, east slope, north slope),
including
  - direct sunlight on the tilted surface (finite solar disk, self-shadowing)
  - reduced sky view: the patch sees a fraction (1 - cos s)/2 of the
    surrounding plain, which returns scattered sunlight and thermal infrared
    (taken from the flat-terrain model)
and a *slope map* from the USGS/MESSENGER global stereo DEM (665 m/pix),
block-averaged to ~2 km. A temperature anywhere is then a table lookup using
the local slope.

Symmetries (zero obliquity, 3:2 resonance): T(lon + 180) = T(lon) at the same
local time, and T(-lat, gx, gy) = T(lat, gx, -gy).

Usage
  python global_terrain.py slopes         # data/mercury_slopes_2km.nc from the DEM
  python global_terrain.py table          # data/mercury_slope_table.nc (~30 min)
  python global_terrain.py check          # accuracy of table interpolation
"""

import argparse
import os
import time

import numpy as np
import xarray as xr
from numba import njit, prange

from mercury_thermal import SIGMA, Model, albedo, serial_launch
from polar_terrain import (DEM_DIR, HERE, N_LT, N_TC, OUT_DEPTHS, SUN_RADIUS_RAD_1AU,
                           _reflected_source, _terrain_batch, depth_weights,
                           flat_surface_temperature, pack)

R_KM = 2439.4
GLOBAL_DEM = dict(
    file="Mercury_Messenger_USGS_DEM_Global_665m_v2.tif", ppd=64, ref_radius_m=2439400.0,
    url="https://planetarymaps.usgs.gov/mosaic/Mercury_Messenger_USGS_DEM_Global_665m_v2.tif")
DATA = os.path.join(HERE, "data")
SLOPES_PATH = os.path.join(DATA, "mercury_slopes_2km.nc")
TABLE_PATH = os.environ.get("MERCURY_SLOPE_TABLE", os.path.join(DATA, "mercury_slope_table.nc"))
BLOCK = 3  # 64 ppd / 3 = 21.33 ppd, ~2.0 km cells at the equator

# table axes
T_LAT = np.arange(0.0, 84.0, 2.0)    # deg; the polar products take over at 80
T_LON = np.arange(0.0, 180.0, 2.0)   # deg E, T(lon + 180) = T(lon)
T_G = np.array([-0.4, -0.2, -0.1, -0.04, 0.0, 0.04, 0.1, 0.2, 0.4])  # dz/dx east, dz/dy north


# ----------------------------------------------------------------------------
# DEM and slope map
# ----------------------------------------------------------------------------

def load_global_dem():
    """Memory-mapped int16 DEM (rows from 90 N, columns from 0 E; 0.5 m per unit)."""
    import tifffile
    path = os.path.join(DEM_DIR, GLOBAL_DEM["file"])
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing; download from {GLOBAL_DEM['url']}")
    return tifffile.memmap(path, mode="r")


def build_slopes(path=SLOPES_PATH):
    """Block-average the DEM to ~2 km and take east/north slopes (dz/dx, dz/dy)."""
    t0 = time.perf_counter()
    g = load_global_dem()
    ny, nx = g.shape[0] // BLOCK, g.shape[1] // BLOCK
    z = np.empty((ny, nx), dtype=np.float32)
    for i in range(ny):  # row blocks, to keep memory low
        blk = g[i * BLOCK:(i + 1) * BLOCK].astype(np.float32)
        blk[blk == -32768] = np.nan
        z[i] = np.nanmean(blk.reshape(BLOCK, nx, BLOCK), axis=(0, 2)) * 0.5
    nanfrac = np.isnan(z).mean()
    if nanfrac:
        from scipy import ndimage
        idx = ndimage.distance_transform_edt(np.isnan(z), return_distances=False, return_indices=True)
        z = z[tuple(idx)]
    ppd = GLOBAL_DEM["ppd"] / BLOCK
    lat = 90.0 - (np.arange(ny) + 0.5) / ppd
    lon = (np.arange(nx) + 0.5) / ppd
    d = np.deg2rad(1.0 / ppd) * (GLOBAL_DEM["ref_radius_m"])  # metres per cell (north-south)
    gx = (np.roll(z, -1, 1) - np.roll(z, 1, 1)) / (2 * d * np.cos(np.deg2rad(lat))[:, None])
    gy = np.empty_like(z)
    gy[1:-1] = (z[:-2] - z[2:]) / (2 * d)
    gy[0], gy[-1] = gy[1], gy[-2]
    gx[np.abs(lat) > 88.0] = 0.0  # longitude spacing collapses at the poles
    ds = xr.Dataset(
        {"elevation": (("lat", "lon"), np.rint(z).astype(np.int16),
                       {"units": "m", "long_name": "elevation above 2439.4 km sphere, ~2 km block mean"}),
         "gx": (("lat", "lon"), pack(np.clip(gx, -3, 3), 1e-4), {"scale_factor": 1e-4, "long_name": "dz/dx east"}),
         "gy": (("lat", "lon"), pack(np.clip(gy, -3, 3), 1e-4), {"scale_factor": 1e-4, "long_name": "dz/dy north"})},
        coords={"lat": lat, "lon": lon},
        attrs={"source": GLOBAL_DEM["file"], "block": BLOCK, "ppd": ppd, "nodata_fraction": float(nanfrac)})
    enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(path, encoding=enc)
    s = np.rad2deg(np.arctan(np.hypot(gx, gy)))
    print(f"wrote {path}: {ny}x{nx}, {time.perf_counter() - t0:.0f} s; slope percentiles "
          f"50/90/99/99.9: {np.percentile(s, [50, 90, 99, 99.9]).round(1)} deg; nodata {nanfrac:.2%}")


# ----------------------------------------------------------------------------
# Slope table
# ----------------------------------------------------------------------------

def normal_enu(gx, gy):
    """Unit surface normal (east, north, up) for slopes dz/dx = gx, dz/dy = gy."""
    gx, gy = np.broadcast_arrays(np.asarray(gx, float), np.asarray(gy, float))
    n = np.stack([-gx, -gy, np.ones_like(gx)], -1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


class SlopeColumns:
    """Thermal runs for tilted patches surrounded by flat plains (no horizon)."""

    def __init__(self):
        self.model = m = Model()
        reg = m.regolith
        mu = np.linspace(1e-4, 1, 2001)
        A = np.array([albedo(x, reg.albedo0, reg.alb_a, reg.alb_b) for x in mu])
        self.absorb_vis = float(1 - np.trapezoid(A * 2 * mu, mu))
        from mercury_thermal import orbit
        _, r_c, self.lss_c = orbit(m.planet, N_TC)
        self.fs_c = m.planet.S0 / r_c**2
        self.rsun_c = SUN_RADIUS_RAD_1AU / r_c
        self.flat_table = xr.open_dataset(os.path.join(DATA, "mercury_temperature.nc"))

    def plain_source(self, lat, lon):
        """Absorbed-ready source leaving the flat plain (sites x N_TC): scattered + IR."""
        n = lat.size
        up = np.tile([0.0, 0.0, 1.0], (n, 1))
        em, nm = np.tile([1.0, 0.0], (n, 1)), np.tile([0.0, 1.0], (n, 1))
        hor = np.zeros((n, 4), dtype=np.float32)
        reg = self.model.regolith
        R = _reflected_source(lat, lon, up, em, nm, hor, self.lss_c, self.fs_c, self.rsun_c,
                              reg.albedo0, reg.alb_a, reg.alb_b, self.absorb_vis)
        Tf = flat_surface_temperature(lat, lon, np.arange(N_TC) / N_TC)
        return R + np.float32(reg.emis**2 * SIGMA) * Tf**4

    def run(self, lat, lon, gx, gy, source=None):
        """lat/lon in radians, slopes per site. Returns Tout (n, N_LT, depths) and stats."""
        m, reg, p = self.model, self.model.regolith, self.model.planet
        n = lat.size
        nrm = normal_enu(gx, gy).reshape(n, 3)
        if source is None:
            source = self.plain_source(lat, lon)
        Find = ((1.0 - nrm[:, 2:3]) / 2.0 * source).astype(np.float32)
        di, dw = depth_weights(m.z, OUT_DEPTHS)
        out_lt = np.arange(N_LT) * 24.0 / N_LT
        Tsurf_c = np.empty((n, N_TC), dtype=np.float32)
        Tout = np.empty((n, N_LT, OUT_DEPTHS.size), dtype=np.float32)
        sf, lf, qm = np.empty(n), np.empty(n), np.empty(n)
        days = np.empty(n, dtype=np.int64)
        _terrain_batch(lat, lon, nrm, np.tile([1.0, 0.0], (n, 1)), np.tile([0.0, 1.0], (n, 1)),
                       np.zeros((n, 4), dtype=np.float32), Find, m.lss, p.S0 / m.r_sun**2,
                       SUN_RADIUS_RAD_1AU / m.r_sun,
                       m.grid.steps_per_day // m.grid.spinup_steps_per_day, 0.01, 2000,
                       m.z, m.rho, m.kc, reg.chi, reg.emis, reg.albedo0, reg.alb_a,
                       reg.alb_b, reg.q_geo, p.P_sol / m.lss.size, out_lt, di, dw, N_TC,
                       Tsurf_c, Tout, sf, lf, qm, days)
        return dict(T=Tout, Tsurf_c=Tsurf_c, sun_frac=sf, lit_frac=lf, Qmean=qm, days=days)


def build_table(path=TABLE_PATH, ckpt_dir=os.path.join(DATA, "checkpoints", "slope_table")):
    """Run every (lat, lon, gy, gx) node; one checkpoint file per latitude row."""
    os.makedirs(ckpt_dir, exist_ok=True)
    sc = SlopeColumns()
    GY, GX = np.meshgrid(T_G, T_G, indexing="ij")
    ng = T_G.size
    t_start = time.perf_counter()
    rows = []
    for ia, la in enumerate(T_LAT):
        f = os.path.join(ckpt_dir, f"lat{la:04.1f}.npy")
        if os.path.exists(f):
            rows.append(np.load(f))
            continue
        t0 = time.perf_counter()
        lat = np.full(T_LON.size, np.deg2rad(la))
        lon = np.deg2rad(T_LON)
        src = sc.plain_source(lat, lon)                                 # (nlon, N_TC)
        latc = np.repeat(lat, ng * ng)
        lonc = np.repeat(lon, ng * ng)
        srcc = np.repeat(src, ng * ng, axis=0)
        r = sc.run(latc, lonc, np.tile(GX.ravel(), T_LON.size), np.tile(GY.ravel(), T_LON.size), srcc)
        T = pack(r["T"], 0.02).reshape(T_LON.size, ng, ng, N_LT, OUT_DEPTHS.size)
        np.save(f + ".tmp.npy", T)
        os.replace(f + ".tmp.npy", f)
        rows.append(T)
        done = ia + 1
        el = time.perf_counter() - t_start
        print(f"  lat {la:4.1f}: {latc.size} columns in {time.perf_counter() - t0:.1f} s "
              f"({done}/{T_LAT.size}, ~{el / done * (T_LAT.size - done) / 60:.0f} min left)", flush=True)
    T = np.stack(rows)                                                   # lat, lon, gy, gx, lt, depth
    ds = xr.Dataset(
        {"temperature": (("lat", "lon", "gy", "gx", "local_time", "depth"), T,
                         {"units": "K", "scale_factor": 0.02, "_FillValue": np.int16(-32768)})},
        coords={"lat": T_LAT, "lon": T_LON, "gy": T_G, "gx": T_G,
                "local_time": np.arange(N_LT) * 24.0 / N_LT, "depth": OUT_DEPTHS},
        attrs={"title": "Mercury regolith temperature for tilted surfaces (slope table)",
               "method": "1D heat conduction per node; direct sunlight on the tilted surface "
                         "(finite disk, self-shadowing, flat horizon); the patch sees (1-cos s)/2 "
                         "of a flat plain returning scattered sunlight and IR from the flat model",
               "symmetry": "T(lon+180)=T(lon); T(-lat,gx,gy)=T(lat,gx,-gy)",
               "gx_gy": "dz/dx east and dz/dy north (tangent of slope components)"})
    ds.to_netcdf(path, encoding={"temperature": {"zlib": True, "complevel": 4}})
    print(f"wrote {path} ({os.path.getsize(path) / 1e6:.0f} MB) in "
          f"{(time.perf_counter() - t_start) / 60:.1f} min")


# ----------------------------------------------------------------------------
# Lookups
# ----------------------------------------------------------------------------

@njit(cache=True)
def _node(v, nodes):
    """Lower node index and weight for v on a (possibly non-uniform) axis, clamped."""
    n = nodes.size
    if v <= nodes[0]:
        return 0, 0.0
    if v >= nodes[n - 1]:
        return n - 2, 1.0
    i = 0
    while nodes[i + 1] < v:
        i += 1
    return i, (v - nodes[i]) / (nodes[i + 1] - nodes[i])


@serial_launch
@njit(parallel=True, cache=True)
def _table_points(T, scale, g_nodes, lat, lon, gx, gy, lt, depth, d_nodes, out):
    """
    Interpolate the slope table at points (degrees, slopes, local time h, depth m).
    T: int16 (lat, lon, gy, gx, lt, depth) with 2-deg lat/lon nodes from 0.
    """
    nla, nlo, nlt = T.shape[0], T.shape[1], T.shape[4]
    for p in prange(lat.size):
        a = min(abs(lat[p]), 2.0 * (nla - 1)) / 2.0
        ia = min(int(a), nla - 2)
        wa = a - ia
        o = (lon[p] % 180.0) / 2.0
        io = int(o) % nlo
        wo = o - np.floor(o)
        sgn = 1.0 if lat[p] >= 0 else -1.0
        iy, wy = _node(sgn * gy[p], g_nodes)
        ix, wx = _node(gx[p], g_nodes)
        f = (lt[p] % 24.0) / (24.0 / nlt)
        il = int(f) % nlt
        wl = f - np.floor(f)
        ik, wk = _node(depth[p], d_nodes)
        acc = 0.0
        for da in range(2):
            fa = wa if da else 1.0 - wa
            for do in range(2):
                fo = wo if do else 1.0 - wo
                jo = (io + do) % nlo
                for dy in range(2):
                    fy = wy if dy else 1.0 - wy
                    for dx in range(2):
                        fx = wx if dx else 1.0 - wx
                        w = fa * fo * fy * fx
                        if w == 0.0:
                            continue
                        for dl in range(2):
                            fl = wl if dl else 1.0 - wl
                            jl = (il + dl) % nlt
                            for dk in range(2):
                                fk = wk if dk else 1.0 - wk
                                acc += w * fl * fk * T[ia + da, jo, iy + dy, ix + dx, jl, ik + dk]
        out[p] = acc * scale


@serial_launch
@njit(parallel=True, cache=True)
def _table_series(T, scale, g_nodes, lat, lon, gx, gy, k_depth, ntc, out):
    """Temperature at one table depth index vs time of day (ntc samples from tau = 0)."""
    nla, nlo, nlt = T.shape[0], T.shape[1], T.shape[4]
    for p in prange(lat.size):
        a = min(abs(lat[p]), 2.0 * (nla - 1)) / 2.0
        ia = min(int(a), nla - 2)
        wa = a - ia
        o = (lon[p] % 180.0) / 2.0
        io = int(o) % nlo
        wo = o - np.floor(o)
        sgn = 1.0 if lat[p] >= 0 else -1.0
        iy, wy = _node(sgn * gy[p], g_nodes)
        ix, wx = _node(gx[p], g_nodes)
        for c in range(ntc):
            lt = (12.0 + lon[p] / 15.0 + 24.0 * c / ntc) % 24.0
            f = lt / (24.0 / nlt)
            il = int(f) % nlt
            wl = f - np.floor(f)
            acc = 0.0
            for da in range(2):
                fa = wa if da else 1.0 - wa
                for do in range(2):
                    fo = wo if do else 1.0 - wo
                    jo = (io + do) % nlo
                    for dy in range(2):
                        fy = wy if dy else 1.0 - wy
                        for dx in range(2):
                            fx = wx if dx else 1.0 - wx
                            w = fa * fo * fy * fx
                            for dl in range(2):
                                fl = wl if dl else 1.0 - wl
                                acc += w * fl * T[ia + da, jo, iy + dy, ix + dx, (il + dl) % nlt, k_depth]
            out[p, c] = acc * scale


class SlopeMap:
    """The ~2 km slope/elevation map (nearest-cell queries)."""

    def __init__(self, path=SLOPES_PATH):
        ds = xr.open_dataset(path, mask_and_scale=False)
        self.gx = ds["gx"].values
        self.gy = ds["gy"].values
        self.scale = float(ds["gx"].attrs["scale_factor"])
        self.elevation = ds["elevation"].values
        self.ppd = float(ds.attrs["ppd"])
        ds.close()
        self.ny, self.nx = self.gx.shape

    def index(self, lat, lon):
        i = np.clip(((90.0 - np.asarray(lat)) * self.ppd).astype(int), 0, self.ny - 1)
        j = ((np.asarray(lon) % 360.0) * self.ppd).astype(int) % self.nx
        return i, j

    def __call__(self, lat, lon):
        """(gx, gy, elevation_m) at points."""
        i, j = self.index(lat, lon)
        return (self.gx[i, j] * self.scale, self.gy[i, j] * self.scale,
                self.elevation[i, j].astype(np.float64))

    def elevation_bilinear(self, lat, lon):
        """Elevation (m) with bilinear interpolation (for coarse terrain grids)."""
        fy = (90.0 - lat) * self.ppd - 0.5
        fx = (lon % 360.0) * self.ppd - 0.5
        i0 = np.clip(np.floor(fy).astype(int), 0, self.ny - 2)
        wy = np.clip(fy - i0, 0, 1)
        j0 = np.floor(fx).astype(int)
        wx = fx - j0
        j0 %= self.nx
        j1 = (j0 + 1) % self.nx
        e = self.elevation
        return ((1 - wy) * ((1 - wx) * e[i0, j0] + wx * e[i0, j1])
                + wy * ((1 - wx) * e[i0 + 1, j0] + wx * e[i0 + 1, j1]))


class SlopeTable:
    """Interpolation in the slope table (kept packed in memory, ~0.4 GB)."""

    def __init__(self, path=TABLE_PATH):
        ds = xr.open_dataset(path, mask_and_scale=False)
        self.T = np.ascontiguousarray(ds["temperature"].values)
        self.scale = float(ds["temperature"].attrs["scale_factor"])
        self.g = ds["gx"].values.astype(np.float64)
        self.lat, self.lon = ds["lat"].values, ds["lon"].values
        self.local_time, self.depth = ds["local_time"].values, ds["depth"].values.astype(np.float64)
        ds.close()

    def __call__(self, lat, lon, gx, gy, lt, depth):
        args = np.broadcast_arrays(*(np.asarray(v, np.float64) for v in (lat, lon, gx, gy, lt, depth)))
        shape = args[0].shape
        flat = [np.ascontiguousarray(a.ravel()) for a in args]
        out = np.empty(flat[0].size)
        _table_points(self.T, self.scale, self.g, *flat, self.depth, out)
        return out.reshape(shape)

    def profile(self, lat, lon, gx, gy):
        """T (n_local_time, n_depth) at one site on the table's own grids."""
        L, D = np.meshgrid(self.local_time, self.depth, indexing="ij")
        return self(lat, lon, gx, gy, L, D)

    def surface_series(self, lat, lon, gx, gy, ntc=N_TC):
        """Surface temperature vs time of day (sites x ntc), for terrain radiance."""
        out = np.empty((lat.size, ntc), dtype=np.float32)
        _table_series(self.T, self.scale, self.g, *(np.ascontiguousarray(np.asarray(v, np.float64))
                      for v in (lat, lon, gx, gy)), 0, ntc, out)
        return out


# ----------------------------------------------------------------------------
# Full terrain-aware single site anywhere (for "exact" runs outside the polar caps)
# ----------------------------------------------------------------------------

def oblique_inverse(x, y, lat0, lon0):
    """Inverse oblique stereographic projection (sphere R_KM) centred on lat0, lon0 (rad)."""
    rho = np.hypot(x, y)
    c = 2 * np.arctan(rho / (2 * R_KM))
    sc, cc = np.sin(c), np.cos(c)
    with np.errstate(invalid="ignore", divide="ignore"):
        lat = np.arcsin(np.clip(cc * np.sin(lat0) + np.where(rho > 0, y * sc * np.cos(lat0) / rho, 0), -1, 1))
    lon = lon0 + np.arctan2(x * sc, rho * np.cos(lat0) * cc - y * np.sin(lat0) * sc)
    return lat, lon % (2 * np.pi)


def oblique_forward(lat, lon, lat0, lon0):
    dl = lon - lon0
    k = 2 * R_KM / (1 + np.sin(lat0) * np.sin(lat) + np.cos(lat0) * np.cos(lat) * np.cos(dl))
    return k * np.cos(lat) * np.sin(dl), k * (np.cos(lat0) * np.sin(lat) - np.sin(lat0) * np.cos(lat) * np.cos(dl))


class LocalGrid:
    """
    Square grid in an oblique stereographic projection centred on (lat0, lon0),
    with the same attributes as polar_terrain.TerrainGrid so the ray tracer
    and site model can use it. Grid centre is map point (x0, y0).
    """

    def __init__(self, lat0, lon0, res_km, half_km, x0=0.0, y0=0.0, elev_fn=None):
        from polar_terrain import TerrainGrid
        self.res = res_km
        n = 2 * int(np.ceil(half_km / res_km))
        self.shape = (n, n)
        self.x = x0 + (np.arange(n) - (n - 1) / 2) * res_km
        self.y = y0 + ((n - 1) / 2 - np.arange(n)) * res_km
        X, Y = np.meshgrid(self.x, self.y)
        self.lat, self.lon = oblique_inverse(X, Y, lat0, lon0)
        self.radius = R_KM + elev_fn(np.rad2deg(self.lat), np.rad2deg(self.lon)) / 1000.0
        d = 1e-6  # map-frame directions of local east and north, numerically
        xe, ye = oblique_forward(self.lat, self.lon + d / np.maximum(np.cos(self.lat), 1e-6), lat0, lon0)
        xn, yn = oblique_forward(self.lat + d, self.lon, lat0, lon0)
        EM = np.stack([xe - X, ye - Y], -1)
        NM = np.stack([xn - X, yn - Y], -1)
        EM /= np.linalg.norm(EM, axis=-1, keepdims=True)
        NM /= np.linalg.norm(NM, axis=-1, keepdims=True)
        TerrainGrid._derive(self, EM, NM)


class DemWindow:
    """Bilinear elevation (m) from the native 64 ppd DEM, reading only a window."""

    def __init__(self, dem, lat_deg, lon_deg):
        ppd = GLOBAL_DEM["ppd"]
        r0 = max(0, int((90.0 - lat_deg.max()) * ppd) - 2)
        r1 = min(dem.shape[0], int((90.0 - lat_deg.min()) * ppd) + 3)
        lo = lon_deg % 360.0
        # longitudes of the window, relative to its centre, to handle wrap-around
        c = np.rad2deg(np.angle(np.mean(np.exp(1j * np.deg2rad(lo)))))
        rel = (lo - c + 180.0) % 360.0 - 180.0
        c0 = int(np.floor((c + rel.min()) * ppd)) - 2
        c1 = int(np.ceil((c + rel.max()) * ppd)) + 3
        cols = np.arange(c0, c1) % dem.shape[1]
        self.w = dem[r0:r1][:, cols].astype(np.float32) * 0.5
        self.r0, self.c0, self.c, self.ppd = r0, c0, c, ppd

    def __call__(self, lat, lon):
        from scipy import ndimage
        rel = ((lon % 360.0) - self.c + 180.0) % 360.0 - 180.0
        fy = (90.0 - lat) * self.ppd - 0.5 - self.r0
        fx = (self.c + rel) * self.ppd - 0.5 - self.c0
        return ndimage.map_coordinates(self.w, [fy, fx], order=1, mode="nearest")


class GlobalSite:
    """
    Terrain-aware on-demand run for any site below the polar caps, mirroring
    polar_site.PolarSite: native-resolution near field (+-near_km, 665 m),
    2 km far field (to far_km), ray-traced horizons, view factors, and
    radiance of the visible terrain from the slope table.
    """

    def __init__(self, near_km=25.0, far_km=250.0, far_res_km=2.0, naz=128,
                 table=None, slopes=None):
        t0 = time.perf_counter()
        self.near_km, self.far_km, self.far_res, self.naz = near_km, far_km, far_res_km, naz
        self.dem = load_global_dem()
        self.res = 2 * np.pi * R_KM / (360 * GLOBAL_DEM["ppd"])  # 0.665 km
        self.model = Model()
        self.table = table or SlopeTable()
        self.slopes = slopes or SlopeMap()
        reg = self.model.regolith
        mu = np.linspace(1e-4, 1, 2001)
        A = np.array([albedo(x, reg.albedo0, reg.alb_a, reg.alb_b) for x in mu])
        self.absorb_vis = float(1 - np.trapezoid(A * 2 * mu, mu))
        from mercury_thermal import orbit
        _, r_c, self.lss_c = orbit(self.model.planet, N_TC)
        self.fs_c = self.model.planet.S0 / r_c**2
        self.rsun_c = SUN_RADIUS_RAD_1AU / r_c
        self.far_grid = None
        self.setup_seconds = time.perf_counter() - t0

    def _radiance(self, tg, idx):
        """Source term (cells x N_TC) leaving grid cells idx: scattered sunlight + IR."""
        lat, lon = tg.lat.ravel()[idx], tg.lon.ravel()[idx]
        nrm = tg.n_enu.reshape(-1, 3)[idx]
        n = idx.size
        reg = self.model.regolith
        R = _reflected_source(lat, lon, np.ascontiguousarray(nrm), np.tile([1.0, 0.0], (n, 1)),
                              np.tile([0.0, 1.0], (n, 1)), np.zeros((n, 4), dtype=np.float32),
                              self.lss_c, self.fs_c, self.rsun_c, reg.albedo0, reg.alb_a,
                              reg.alb_b, self.absorb_vis)
        gx, gy = -nrm[:, 0] / nrm[:, 2], -nrm[:, 1] / nrm[:, 2]
        Ts = self.table.surface_series(np.rad2deg(lat), np.rad2deg(lon), gx, gy)
        return R + np.float32(reg.emis**2 * SIGMA) * Ts**4

    def run(self, lat, lon):
        from polar_site import ray_geometry, sky_fraction
        tm = {}
        t0 = time.perf_counter()
        la, lo = np.deg2rad(lat), np.deg2rad(lon % 360)
        if self.far_grid is not None:
            c_la, c_lo = self.far_grid[1:3]
            x, y = oblique_forward(la, lo, c_la, c_lo)
        if self.far_grid is None or np.hypot(x, y) > 0.2 * self.far_km:
            far = LocalGrid(la, lo, self.far_res, self.far_km * 1.2,
                            elev_fn=self.slopes.elevation_bilinear)
            self.far_grid = (far, la, lo, self._radiance(far, np.arange(far.lat.size)))
            x, y = 0.0, 0.0
        far, c_la, c_lo, X_far = self.far_grid
        tm["far grid (cached nearby)"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        n = 2 * int(np.ceil(self.near_km / self.res))
        r = (np.arange(n) - (n - 1) / 2) * self.res
        bx, by = np.meshgrid(x + r[[0, -1]], y + r[[0, -1]])
        blat, blon = oblique_inverse(bx, by, c_la, c_lo)
        # DEM window covering the near grid (corners plus margin)
        span = np.rad2deg(self.near_km * 1.5 / R_KM)
        wl = np.rad2deg(blat).ravel()
        wo = np.rad2deg(blon).ravel()
        win = DemWindow(self.dem, np.array([wl.min() - span / 4, wl.max() + span / 4]),
                        np.concatenate([wo, [lon]]))
        near = LocalGrid(c_la, c_lo, self.res, self.near_km, x, y, elev_fn=win)
        fi, fj = (near.y[0] - y) / self.res, (x - near.x[0]) / self.res
        ic, jc = int(round(fi)), int(round(fj))
        o = dict(p0=near.P[ic, jc][None].copy(), up=near.U[ic, jc][None].copy(),
                 ne=near.n_enu[ic, jc, 0:1].copy(), nn=near.n_enu[ic, jc, 1:2].copy(),
                 nu=near.n_enu[ic, jc, 2:3].copy(),
                 em=near.EM[ic, jc][None].copy(), nm=near.NM[ic, jc][None].copy())
        tm["near grid"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        naz = self.naz
        horizon = np.full((1, naz), -np.pi / 2, dtype=np.float32)
        viewed = np.full((1, naz), -np.pi / 2, dtype=np.float32)
        _, src_n, w_n = ray_geometry(near, x, y, o, naz, 1, int(self.near_km / self.res) - 1,
                                     horizon, viewed)
        step0 = int(np.ceil(self.near_km / self.far_res))
        _, src_f, w_f = ray_geometry(far, x, y, o, naz, step0, int(self.far_km / self.far_res),
                                     horizon, viewed)
        sky = sky_fraction(o, viewed, naz)
        tm["rays (near + far)"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        if src_n.size:
            u, inv = np.unique(src_n, return_inverse=True)
            X_near = self._radiance(near, u)[inv]
        else:
            X_near = np.zeros((0, N_TC), dtype=np.float32)
        Find = (w_n[:, None] * X_near).sum(0) + (w_f[:, None] * X_far[src_f]).sum(0)
        tm["terrain radiance"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        m, reg, p = self.model, self.model.regolith, self.model.planet
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
        # horizon bins are map-frame angles; convert to compass azimuths here
        a = 2 * np.pi * np.arange(naz) / naz
        em, nm = o["em"][0], o["nm"][0]
        az = np.rad2deg(np.arctan2(np.cos(a) * em[0] + np.sin(a) * em[1],
                                   np.cos(a) * nm[0] + np.sin(a) * nm[1])) % 360.0
        return dict(T=Tout[0], local_time=out_lt, depth=OUT_DEPTHS,
                    horizon_deg=np.rad2deg(horizon[0]), horizon_azimuth_deg=az,
                    sky_view_factor=sky, sun_visible_fraction=sf[0], direct_lit_fraction=lf[0],
                    mean_direct_flux=qm[0], permanent_shadow=bool(lf[0] == 0),
                    elevation_m=float(near.elevation[ic, jc]),
                    slope_deg=float(near.slope_deg[ic, jc]), timing=tm,
                    seconds=sum(tm.values()),
                    # inputs to the final column solve (used by site_script.py)
                    site_inputs=dict(lat=float(near.lat[ic, jc]), lon=float(near.lon[ic, jc]),
                                     n_enu=near.n_enu[ic, jc].astype(float), em=o["em"][0].astype(float),
                                     nm=o["nm"][0].astype(float), horizon=horizon[0].astype(float),
                                     Find=Find.astype(float)))


def check_table(n=400, seed=0):
    """Compare table interpolation against direct runs at random off-node points."""
    rng = np.random.default_rng(seed)
    st, sm, sc = SlopeTable(), SlopeMap(), SlopeColumns()
    lat = rng.uniform(-79.9, 79.9, n)
    lon = rng.uniform(0, 360, n)
    # slopes drawn from the real slope map at random places
    i = rng.integers(0, sm.ny, n * 20)
    j = rng.integers(0, sm.nx, n * 20)
    g = np.stack([sm.gx[i, j], sm.gy[i, j]], -1) * sm.scale
    g = g[np.abs(sm.ny / 2 - i) < sm.ny * 80 / 180][:n]
    t0 = time.perf_counter()
    d = sc.run(np.deg2rad(lat), np.deg2rad(lon), g[:, 0], g[:, 1])["T"]
    print(f"{n} direct runs in {time.perf_counter() - t0:.1f} s")
    for k, z in enumerate(OUT_DEPTHS):
        if z not in (0.0, 0.1, 0.5, 1.0):
            continue
        tab = np.stack([st.profile(lat[p], lon[p], g[p, 0], g[p, 1])[:, k] for p in range(n)])
        e = np.abs(tab - d[:, :, k])
        print(f"depth {z:4.2f} m: |table - direct| mean {e.mean():.2f} K, 99th pct "
              f"{np.percentile(e, 99):.2f} K, max {e.max():.1f} K; daily mean bias "
              f"{np.abs(tab.mean(1) - d[:, :, k].mean(1)).max():.2f} K max")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["slopes", "table", "check"])
    a = ap.parse_args()
    if a.what == "slopes":
        build_slopes()
    elif a.what == "table":
        build_table()
    elif a.what == "check":
        check_table()
