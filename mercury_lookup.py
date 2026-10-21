"""
Query Mercury regolith temperature at any (lon, lat, local time, depth).

    from mercury_lookup import MercuryTemperature
    mt = MercuryTemperature()                    # loads data/mercury_temperature.nc
    mt(lon=250.0, lat=-35.0, local_time=6.5, depth=0.4)   # K, from the table
    mt.on_demand(lat=-35.0, lon=250.0)           # exact 1D run for that site (~0.1 s)

Poleward of 80 deg, values come from the terrain-aware polar products
(data/mercury_polar_<pole>_2km.nc) when present, which include shadowing,
slopes and heat exchange with surrounding terrain. Elsewhere they come from
the slope table (data/mercury_slope_table.nc) at the local ~2 km slope of the
global DEM (data/mercury_slopes_2km.nc), when present. Pass terrain=False to
use the flat-terrain model everywhere. on_demand() runs the full
terrain-aware single-site model (ray-traced horizons, heat from visible
terrain) at native DEM resolution: polar_site.PolarSite poleward of 80 deg,
global_terrain.GlobalSite elsewhere.

Inputs broadcast like numpy arrays. Longitude is degrees east (any range),
latitude -90..90, local_time is mean local solar time in hours (0-24, 12 =
mean noon), depth in metres (0..~3.16).
"""

import os

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DEFAULT_PATH = os.path.join(DATA, "mercury_temperature.nc")
R_KM = 2439.4


class PolarTable:
    """Interpolation in a terrain-aware polar product (kept packed in memory)."""

    def __init__(self, path):
        ds = xr.open_dataset(path, mask_and_scale=False)
        self.pole = ds.attrs["pole"]
        self.lat_min = float(ds.attrs["lat_min"])
        self.T = ds["temperature"].values  # int16, (y, x, lt, depth)
        self.scale = float(ds["temperature"].attrs["scale_factor"])
        self.x, self.y = ds["x"].values, ds["y"].values
        self.lt, self.depth = ds["local_time"].values, ds["depth"].values
        ds.close()

    def __call__(self, lon, lat, lt, depth):
        """Temperatures (NaN where the point is outside the product)."""
        la, lo = np.deg2rad(np.abs(lat)), np.deg2rad(lon % 360)
        rho = 2 * R_KM * np.tan((np.pi / 2 - la) / 2)
        x = rho * np.sin(lo)
        y = (-1.0 if self.pole == "north" else 1.0) * rho * np.cos(lo)
        dx = self.x[1] - self.x[0]
        fx = (x - self.x[0]) / dx
        fy = (self.y[0] - y) / dx
        ny, nx = self.T.shape[:2]
        j0 = np.clip(np.floor(fx).astype(int), 0, nx - 2)
        i0 = np.clip(np.floor(fy).astype(int), 0, ny - 2)
        wx, wy = np.clip(fx - j0, 0, 1), np.clip(fy - i0, 0, 1)
        nlt = self.lt.size
        fl = (lt % 24.0) / (24.0 / nlt)
        l0 = np.floor(fl).astype(int) % nlt
        wl = fl - np.floor(fl)
        d = np.clip(depth, 0, self.depth[-1])
        k0 = np.clip(np.searchsorted(self.depth, d) - 1, 0, self.depth.size - 2)
        wk = (d - self.depth[k0]) / (self.depth[k0 + 1] - self.depth[k0])
        out = np.zeros(np.shape(x))
        bad = np.zeros(np.shape(x), dtype=bool)
        for di, wi in ((0, 1 - wy), (1, wy)):
            for dj, wj in ((0, 1 - wx), (1, wx)):
                for dl, wll in ((0, 1 - wl), (1, wl)):
                    for dk, wkk in ((0, 1 - wk), (1, wk)):
                        v = self.T[i0 + di, j0 + dj, (l0 + dl) % nlt, k0 + dk]
                        bad |= v == -32768
                        out += wi * wj * wll * wkk * v
        out = out * self.scale
        out[bad | (fx < 0) | (fy < 0) | (fx > nx - 1) | (fy > ny - 1)] = np.nan
        return out


class MercuryTemperature:
    def __init__(self, path=DEFAULT_PATH, terrain=True):
        ds = xr.load_dataset(path)
        self.ds = ds
        T = ds["temperature"].values
        lon, lat, lt, z = (ds[c].values for c in ("lon", "lat", "local_time", "depth"))
        # close the periodic axes: lon 180 == lon 0, local time 24 == 0
        T = np.concatenate([T, T[:1]], axis=0)
        T = np.concatenate([T, T[:, :, :1]], axis=2)
        lon = np.append(lon, 180.0)
        lt = np.append(lt, 24.0)
        self.depth = z
        self._interp = RegularGridInterpolator((lon, lat, lt, z), T)
        self._model = None
        self._sites = {}
        self.polar = {}
        self.slope_table = self.slope_map = None
        if terrain:
            for pole in ("north", "south"):
                p = os.path.join(DATA, f"mercury_polar_{pole}_2km.nc")
                if os.path.exists(p):
                    self.polar[pole] = PolarTable(p)
            import global_terrain as gt
            if os.path.exists(gt.TABLE_PATH) and os.path.exists(gt.SLOPES_PATH):
                self.slope_table = gt.SlopeTable()
                self.slope_map = gt.SlopeMap()

    def __call__(self, lon, lat, local_time, depth):
        lon, lat, local_time, depth = np.broadcast_arrays(
            np.asarray(lon, float), np.asarray(lat, float),
            np.asarray(local_time, float), np.asarray(depth, float))
        pts = np.stack([lon % 180.0, np.abs(lat), local_time % 24.0,
                        np.clip(depth, 0.0, self.depth[-1])], axis=-1)
        if self.slope_table is not None:
            gx, gy, _ = self.slope_map(lat, lon)
            out = self.slope_table(lat, lon, gx, gy, local_time, depth)
        else:
            out = self._interp(pts)
        for pole, tab in self.polar.items():
            m = (lat >= tab.lat_min) if pole == "north" else (lat <= -tab.lat_min)
            if m.any():
                v = tab(lon[m], lat[m], local_time[m], depth[m])
                ok = np.isfinite(v)
                sub = out[m]
                sub[ok] = v[ok]
                out[m] = sub
        return out

    def slope_info(self, lat, lon):
        """Local ~2 km terrain from the global DEM: elevation (m), slope (deg), facing azimuth (deg)."""
        if self.slope_map is None:
            return None
        gx, gy, elev = (float(v) for v in self.slope_map(lat, lon))
        return dict(elevation_m=elev, slope_deg=float(np.rad2deg(np.arctan(np.hypot(gx, gy)))),
                    facing_deg=float(np.rad2deg(np.arctan2(-gx, -gy)) % 360.0), gx=gx, gy=gy)

    def site_model(self, lat):
        """The full terrain-aware single-site model for this latitude (cached), or None."""
        if abs(lat) >= 80.0:
            from polar_site import PolarSite
            key = "north" if lat > 0 else "south"
            if key not in self._sites:
                self._sites[key] = PolarSite(key)
        else:
            if self.slope_table is None:
                return None
            from global_terrain import GlobalSite
            key = "global"
            if key not in self._sites:
                self._sites[key] = GlobalSite(table=self.slope_table, slopes=self.slope_map)
        return self._sites[key]

    def on_demand(self, lat, lon, local_time=None, terrain=True):
        """
        Run the model for exactly this site. Returns (local_time, depth, T)
        with T shaped (n_local_time, n_depth). With terrain=True the full
        terrain-aware site model is used where its data are present; its output
        is on its own fixed local-time and depth grid.
        """
        site = self.site_model(lat) if terrain else None
        if site is not None:
            r = site.run(lat, lon)
            return r["local_time"], r["depth"], r["T"]
        from mercury_thermal import Model
        if self._model is None:
            self._model = Model()
        if local_time is None:
            local_time = self.ds["local_time"].values
        local_time = np.atleast_1d(local_time)
        T = self._model.column_vs_local_time(abs(lat), lon, local_time)
        return local_time, self._model.z, T
