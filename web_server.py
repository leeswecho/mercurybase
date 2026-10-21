"""
Local web server for interactively exploring the Mercury temperature model.

Usage:  python web_server.py [--port 8000] [--host 127.0.0.1]
then open http://127.0.0.1:8000/

Endpoints (all GET):
  /api/meta                    grids, colour scale, orbit (true subsolar longitude)
  /api/frame?depth=D&k=K       globe texture atlas (8-bit PNG) for frame K
  /api/point?lat=..&lon=..     table temperatures T(local_time, depth) + site info
  /api/ondemand?lat=..&lon=..  exact terrain-aware model run for the site
  /api/sample?lat&lon&t&depth  one temperature (globe hover readout)
  /api/script?lat=..&lon=..    standalone Python script reproducing the exact run
  /api/slopes.png              static ~2 km slope texture for the globe

Globe frames: frame k shows the planet at time tau = k / N_FRAMES of a solar
day after the reference perihelion (mean subsolar longitude -360 tau). The
whole day is rendered: the flat model satisfies T(lon + 180, tau) =
T(lon, tau + 1/2), but terrain in the polar caps does not.

Outside the polar caps the globe shades every ~2 km pixel by its own slope: the frame
holds the slope table's *response* at that moment, i.e. T at each
(latitude node, longitude node, gy node, gx node), and the GPU interpolates it
with the pixel's slope from /api/slopes.png.

Atlas layout (width 9 * 180, height 9 * 42 + CAP):
  top: 9 x 9 tiles; tile (row = gy node, column = gx node) is 180 x 42 texels,
       texel (column = lon node 0..358 step 2, row = lat node 0..82 step 2)
  below (row offset 9 * 42): north polar cap at column 0, south cap at the
       column after it (meta caps[pole].col), each its 2 km polar product grid
       (row 0 = largest map y); the caps can differ in size.
Pixel value v: 0 = no data, else T = T_LO + (v-1) * T_STEP.
Slope texture: R, G = 128 + 127 * g / G_MAX for gx (east), gy (north).
"""

import argparse
import io
import json
import mimetypes
import os
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import xarray as xr
from PIL import Image

from mercury_lookup import DATA, MercuryTemperature
import site_script
from mercury_thermal import Model

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
N_FRAMES = 96           # frames per solar day (3.75 deg of mean subsolar motion)
T_LO, T_HI = 20.0, 740.0
T_STEP = (T_HI - T_LO) / 254
GLOBE_DEPTHS = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0]
MAX_DEPTH = 3.0
R_KM = 2439.4
G_MAX = 0.4         # slope texture range (tangent)
TILE_W, TILE_H = 180, 42


def _encode(T):
    v = np.rint((np.nan_to_num(T, nan=-1e9) - T_LO) / T_STEP) + 1
    v = np.clip(v, 1, 255)
    v[~np.isfinite(T)] = 0
    return v.astype(np.uint8)


class Backend:
    def __init__(self):
        t0 = time.perf_counter()
        self.mt = MercuryTemperature()
        ds = self.mt.ds
        self.flatT = ds["temperature"].values           # (lon 90, lat 46, lt 96, depth 64)
        self.lon_t, self.lat_t = ds["lon"].values, ds["lat"].values
        self.lt_t, self.z_t = ds["local_time"].values, ds["depth"].values
        self.caps = {}
        col = 0
        for pole in ("north", "south"):
            p = os.path.join(DATA, f"mercury_polar_{pole}_2km.nc")
            if os.path.exists(p):
                tab = self.mt.polar.get(pole)  # share its packed temperatures (largest array)
                c = self._load_cap(p, T=tab.T if tab is not None else None)
                c["col"] = col
                col += c["x"].size
                self.caps[pole] = c
        self.cap_cols = col
        self.cap_rows = max([c["y"].size for c in self.caps.values()], default=0)
        self.model = Model()
        st = self.mt.slope_table
        self.g_nodes = st.g.tolist() if st is not None else [0.0]
        self.slopes_png = None
        # frames are browser-cached, so their URLs carry a version that changes with the data
        stamp = [os.path.getmtime(p) for p in (self.mt.ds.encoding.get("source", ""),) if p]
        import global_terrain as gt
        for p in [gt.TABLE_PATH, gt.SLOPES_PATH] + [os.path.join(DATA, f"mercury_polar_{q}_2km.nc")
                                                   for q in ("north", "south")]:
            if os.path.exists(p):
                stamp.append(os.path.getmtime(p))
        self.version = f"{N_FRAMES}.{TILE_W}x{TILE_H}.{int(sum(stamp)) % 10**8}"
        self.frames = OrderedDict()
        self.frame_lock = threading.Lock()
        self.run_lock = threading.Lock()
        self.exact_cache = OrderedDict()
        print(f"backend loaded in {time.perf_counter() - t0:.1f} s "
              f"(polar products: {', '.join(self.caps) or 'none'}; "
              f"slope table: {'yes' if st is not None else 'no, flat terrain outside the caps'})")

    @staticmethod
    def _load_cap(path, T=None):
        ds = xr.open_dataset(path, mask_and_scale=False)
        c = dict(T=ds["temperature"].values if T is None else T,      # int16 (y, x, lt, depth)
                 scale=float(ds["temperature"].attrs["scale_factor"]),
                 lat=ds["lat"].values.astype(np.float64), lon=ds["lon"].values.astype(np.float64),
                 x=ds["x"].values, y=ds["y"].values, depth=ds["depth"].values,
                 nlt=ds.sizes["local_time"], pole=ds.attrs["pole"],
                 lat_min=float(ds.attrs.get("lat_min", 80.0)))
        for k in ("elevation", "slope", "sky_view_factor", "sun_visible_fraction",
                  "direct_lit_fraction", "mean_direct_flux", "permanent_shadow", "dem_quality"):
            v = ds[k]
            a = v.values.astype(np.float64)
            if "_FillValue" in v.attrs:
                a[v.values == v.attrs["_FillValue"]] = np.nan
            if "scale_factor" in v.attrs:
                a *= float(v.attrs["scale_factor"])
            c[k] = a
        hz = ds["horizon"]
        c["horizon"] = hz.values                                       # int16, 0.01 deg
        c["horizon_scale"] = float(hz.attrs.get("scale_factor", 0.01))
        ds.close()
        return c

    # ------------------------------------------------------------------ meta
    def meta(self):
        m = self.model
        tau = m.t / m.planet.P_sol
        step = max(1, tau.size // 960)
        caps = {p: dict(x0=float(c["x"][0]), y0=float(c["y"][0]),
                        dx=float(c["x"][1] - c["x"][0]), n=int(c["x"].size),
                        col=int(c["col"]), lat_min=c["lat_min"])
                for p, c in self.caps.items()}
        return dict(
            version=self.version, n_frames=N_FRAMES, t_lo=T_LO, t_step=T_STEP, globe_depths=GLOBE_DEPTHS,
            solar_day_earth_days=float(m.planet.P_sol / 86400.0), radius_km=R_KM,
            caps=caps, g_nodes=self.g_nodes, g_max=G_MAX,
            tile_w=TILE_W, tile_h=TILE_H, cap_y0=TILE_H * len(self.g_nodes),
            slopes=self.mt.slope_map is not None,
            orbit=dict(tau=np.round(tau[::step], 5).tolist(),
                       subsolar_lon=np.round(np.rad2deg(m.lss[::step]) % 360, 3).tolist(),
                       sun_distance_au=np.round(m.r_sun[::step], 5).tolist()))

    # ---------------------------------------------------------------- frames
    def frame_png(self, depth, k):
        key = (depth, k)
        with self.frame_lock:
            if key in self.frames:
                self.frames.move_to_end(key)
                return self.frames[key]
        png = self._render_frame(depth, k)
        with self.frame_lock:
            self.frames[key] = png
            while len(self.frames) > 400:
                self.frames.popitem(last=False)
        return png

    def _response(self, depth, tau):
        """T at (lat node, lon node 0..358, gy node, gx node) at time tau."""
        lon = np.arange(TILE_W) * 2.0
        lt = (12.0 + lon / 15.0 + 24.0 * tau) % 24.0
        st = self.mt.slope_table
        if st is not None:
            T, d, nlt, scale = st.T, st.depth, st.local_time.size, st.scale
        else:  # flat table only: one slope node
            T = self.flatT.transpose(1, 0, 2, 3)[:TILE_H, :, None, None]   # (lat, lon, 1, 1, lt, z)
            d, nlt, scale = self.z_t, self.lt_t.size, 1.0
        jl = np.arange(TILE_W) % T.shape[1]
        k0 = int(np.clip(np.searchsorted(d, depth) - 1, 0, d.size - 2))
        wk = (depth - d[k0]) / (d[k0 + 1] - d[k0])
        f = lt / (24.0 / nlt)
        l0 = np.floor(f).astype(int) % nlt
        l1 = (l0 + 1) % nlt
        wl = (f - np.floor(f))[None, :, None, None]
        out = 0.0
        for kk, w2 in ((k0, 1 - wk), (k0 + 1, wk)):
            # paired fancy indices (lon, lt) -> that axis goes first; move it back to position 1
            a = np.moveaxis(T[:, jl, :, :, l0, kk], 0, 1).astype(np.float32)
            b = np.moveaxis(T[:, jl, :, :, l1, kk], 0, 1).astype(np.float32)
            out = out + w2 * ((1 - wl) * a + wl * b)
        return out * scale                                               # (lat, lon, gy, gx)

    def _render_frame(self, depth, k):
        tau = k / N_FRAMES
        G = len(self.g_nodes)
        R = self._response(depth, tau)
        tiles = R.transpose(2, 0, 3, 1).reshape(G * TILE_H, G * TILE_W)  # rows: gy, lat; cols: gx, lon
        y0 = G * TILE_H
        atlas = np.zeros((y0 + self.cap_rows, max(G * TILE_W, self.cap_cols)), dtype=np.uint8)
        atlas[:y0, :G * TILE_W] = _encode(tiles)
        for c in self.caps.values():
            off = c["col"]
            atlas[y0:y0 + c["T"].shape[0], off:off + c["T"].shape[1]] = \
                _encode(self._cap_slice(c, depth, tau))
        buf = io.BytesIO()
        Image.fromarray(atlas, mode="L").save(buf, format="PNG", compress_level=6)
        return buf.getvalue()

    def slope_texture(self):
        """PNG of the ~2 km slope map (built once, cached in data/web_cache)."""
        if self.slopes_png is None:
            path = os.path.join(DATA, "web_cache", "slopes.png")
            if os.path.exists(path):
                with open(path, "rb") as f:
                    self.slopes_png = f.read()
            else:
                sm = self.mt.slope_map
                rgb = np.zeros(sm.gx.shape + (3,), dtype=np.uint8)
                for c, g in ((0, sm.gx), (1, sm.gy)):
                    rgb[..., c] = np.clip(np.rint(128 + 127 * g * sm.scale / G_MAX), 1, 255)
                buf = io.BytesIO()
                Image.fromarray(rgb, mode="RGB").save(buf, format="PNG", compress_level=6)
                self.slopes_png = buf.getvalue()
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as f:
                    f.write(self.slopes_png)
        return self.slopes_png

    def sample(self, lat, lon, tau, depth):
        lt = (12.0 + lon / 15.0 + 24.0 * tau) % 24.0
        return float(np.asarray(self.mt(lon, lat, lt, depth)).ravel()[0])

    @staticmethod
    def _cap_slice(c, depth, tau):
        d = c["depth"]
        k0 = int(np.clip(np.searchsorted(d, depth) - 1, 0, d.size - 2))
        wk = (depth - d[k0]) / (d[k0 + 1] - d[k0])
        nlt = c["nlt"]
        lt = (12.0 + c["lon"] / 15.0 + 24.0 * tau) % 24.0
        fl = lt / (24.0 / nlt)
        l0 = np.floor(np.nan_to_num(fl)).astype(int) % nlt
        l1 = (l0 + 1) % nlt
        wl = fl - np.floor(fl)
        T = c["T"]
        iy, ix = np.indices(l0.shape)
        out = np.zeros(l0.shape)
        bad = ~np.isfinite(c["lon"])
        for ll, w1 in ((l0, 1 - wl), (l1, wl)):
            for kk, w2 in ((k0, 1 - wk), (k0 + 1, wk)):
                v = T[iy, ix, ll, kk]
                bad |= v == -32768
                out += w1 * w2 * v
        out *= c["scale"]
        out[bad] = np.nan
        return out

    # ----------------------------------------------------------------- sites
    def _sun_path(self, lat, lon, lt):
        m = self.model
        H = np.deg2rad(m.true_hour_angle_deg(lon))
        nt = H.size
        s = np.rint(((15.0 * (lt - 12.0) - lon) / 360.0 * nt) % nt).astype(int) % nt
        H, r = H[s], m.r_sun[s]
        phi = np.deg2rad(lat)
        elev = np.rad2deg(np.arcsin(np.clip(np.cos(phi) * np.cos(H), -1, 1)))
        az = np.rad2deg(np.arctan2(-np.sin(H), -np.cos(H) * np.sin(phi))) % 360.0
        disk = np.rad2deg(0.0046524 / r)
        return dict(elevation_deg=np.round(elev, 3).tolist(), azimuth_deg=np.round(az, 2).tolist(),
                    disk_radius_deg=np.round(disk, 4).tolist(),
                    distance_au=np.round(r, 5).tolist(),
                    hour_angle_deg=np.round(np.rad2deg(H), 2).tolist())

    @staticmethod
    def _horizon_compass(h_deg, lon, pole):
        """Map-frame horizon bins (angle 360k/n from +x, CCW) -> compass azimuths."""
        naz = h_deg.size
        th = 2 * np.pi * np.arange(naz) / naz
        lo = np.deg2rad(lon)
        if pole == "north":
            em, nm = (np.cos(lo), np.sin(lo)), (-np.sin(lo), np.cos(lo))
        else:
            em, nm = (np.cos(lo), -np.sin(lo)), (np.sin(lo), np.cos(lo))
        e = np.cos(th) * em[0] + np.sin(th) * em[1]
        nn = np.cos(th) * nm[0] + np.sin(th) * nm[1]
        az = np.rad2deg(np.arctan2(e, nn)) % 360.0
        o = np.argsort(az)
        return dict(azimuth_deg=np.round(az[o], 2).tolist(), elevation_deg=np.round(h_deg[o], 3).tolist())

    def _cap_cell(self, lat, lon):
        pole = "north" if lat > 0 else "south"
        c = self.caps.get(pole)
        if c is None or abs(lat) < c["lat_min"]:
            return None, None
        la, lo = np.deg2rad(abs(lat)), np.deg2rad(lon % 360)
        rho = 2 * R_KM * np.tan((np.pi / 2 - la) / 2)
        x = rho * np.sin(lo)
        y = (-1.0 if pole == "north" else 1.0) * rho * np.cos(lo)
        j = int(np.rint((x - c["x"][0]) / (c["x"][1] - c["x"][0])))
        i = int(np.rint((c["y"][0] - y) / (c["x"][1] - c["x"][0])))
        if not (0 <= i < c["y"].size and 0 <= j < c["x"].size) or not np.isfinite(c["sky_view_factor"][i, j]):
            return None, None
        return c, (i, j)

    def point(self, lat, lon):
        lt = self.lt_t
        st = self.mt.slope_table
        depth = st.depth if st is not None else self.z_t[self.z_t <= MAX_DEPTH + 1e-9]
        LT, Z = np.meshgrid(lt, depth, indexing="ij")
        T = self.mt(lon, lat, LT, Z)
        c, ij = self._cap_cell(lat, lon)
        si = self.mt.slope_info(lat, lon)
        if si is not None:
            info = dict(source="slope-aware table (local 2 km slope, flat horizon)", terrain=False,
                        slope_aware=True, elevation_m=si["elevation_m"], slope_deg=si["slope_deg"],
                        facing_deg=si["facing_deg"])
        else:
            info = dict(source="flat 1D table (2° grid)", terrain=False)
        horizon = None
        if c is not None:
            i, j = ij
            info = dict(
                source="terrain-aware polar product (2 km cell)", terrain=True,
                elevation_m=c["elevation"][i, j], slope_deg=c["slope"][i, j],
                sky_view_factor=c["sky_view_factor"][i, j],
                sun_visible_fraction=c["sun_visible_fraction"][i, j],
                direct_lit_fraction=c["direct_lit_fraction"][i, j],
                mean_direct_flux=c["mean_direct_flux"][i, j],
                permanent_shadow=bool(c["permanent_shadow"][i, j] == 1),
                dem_quality=int(c["dem_quality"][i, j]))
            h = c["horizon"][i, j].astype(np.float64) * c["horizon_scale"]
            horizon = self._horizon_compass(h, lon, c["pole"])
        info["sources"] = site_script.sources(lat, "table", terrain=si is not None or c is not None,
                                              polar=c is not None)
        info["script"] = self.mt.is_polar(lat) or self.mt.slope_table is not None
        return self._payload(lat, lon, lt, depth, T, info, horizon)

    def exact_run(self, lat, lon):
        """Terrain-aware run for a site (None without terrain data); recent runs are cached."""
        key = (round(lat, 6), round(lon % 360, 6))
        with self.run_lock:
            if key not in self.exact_cache:
                site = self.mt.site_model(lat)
                if site is None:
                    return None
                self.exact_cache[key] = site.run(lat, lon)
                while len(self.exact_cache) > 16:
                    self.exact_cache.popitem(last=False)
            return self.exact_cache[key]

    def script(self, lat, lon):
        r = self.exact_run(lat, lon)
        if r is None:
            raise ValueError("terrain data not available on this server")
        return site_script.render(lat, lon, r)

    def ondemand(self, lat, lon):
        t0 = time.perf_counter()
        r = self.exact_run(lat, lon)
        with self.run_lock:
            if r is not None:
                info = dict(source="exact terrain-aware run (native DEM resolution)", terrain=True,
                            elevation_m=r["elevation_m"], slope_deg=r["slope_deg"],
                            sky_view_factor=float(r["sky_view_factor"]),
                            sun_visible_fraction=float(r["sun_visible_fraction"]),
                            direct_lit_fraction=float(r["direct_lit_fraction"]),
                            mean_direct_flux=float(r["mean_direct_flux"]),
                            permanent_shadow=bool(r["permanent_shadow"]))
                h = np.asarray(r["horizon_deg"], float)
                if "horizon_azimuth_deg" in r:  # GlobalSite: already compass azimuths
                    o = np.argsort(r["horizon_azimuth_deg"])
                    horizon = dict(azimuth_deg=np.round(r["horizon_azimuth_deg"][o], 2).tolist(),
                                   elevation_deg=np.round(h[o], 3).tolist())
                else:
                    horizon = self._horizon_compass(h, lon, "north" if lat > 0 else "south")
                lt, depth, T = r["local_time"], np.asarray(r["depth"]), r["T"]
            else:
                lt, depth, T = self.mt.on_demand(lat, lon, terrain=False)
                keep = depth <= MAX_DEPTH + 1e-9
                depth, T = depth[keep], T[:, keep]
                info = dict(source="exact flat 1D run", terrain=False)
                horizon = None
        info["sources"] = site_script.sources(lat, "exact", terrain=r is not None,
                                              polar=r is not None and "horizon_azimuth_deg" not in r)
        info["script"] = r is not None
        info["seconds"] = time.perf_counter() - t0
        return self._payload(lat, lon, np.asarray(lt), np.asarray(depth), np.asarray(T), info, horizon)

    def _payload(self, lat, lon, lt, depth, T, info, horizon):
        info = {k: (None if isinstance(v, float) and not np.isfinite(v) else
                    (float(v) if isinstance(v, (np.floating, float)) else v)) for k, v in info.items()}
        return dict(lat=lat, lon=lon % 360, local_time=np.round(lt, 4).tolist(),
                    depth=np.round(depth, 5).tolist(), T=np.round(T, 2).tolist(),
                    info=info, horizon=horizon, sun=self._sun_path(lat, lon % 360, np.asarray(lt)))


class Handler(BaseHTTPRequestHandler):
    backend: Backend = None

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype, cache=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if cache:
            self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, allow_nan=False).encode(), "application/json")

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        b = self.backend
        try:
            if u.path == "/api/meta":
                return self._json(b.meta())
            if u.path == "/api/frame":
                depth = float(q.get("depth", 0))
                k = int(q.get("k", 0)) % N_FRAMES
                if depth not in GLOBE_DEPTHS:
                    return self._json({"error": "unsupported depth"}, 400)
                return self._send(200, b.frame_png(depth, k), "image/png", "max-age=86400")
            if u.path == "/api/slopes.png":
                if b.mt.slope_map is None:
                    return self._json({"error": "no slope map"}, 404)
                return self._send(200, b.slope_texture(), "image/png", "max-age=86400")
            if u.path == "/api/sample":
                lat, lon = float(q["lat"]), float(q["lon"])
                if not (-90 <= lat <= 90 and np.isfinite(lon)):
                    return self._json({"error": "lat must be in -90..90"}, 400)
                return self._json({"T": b.sample(lat, lon, float(q.get("t", 0)), float(q.get("depth", 0)))})
            if u.path in ("/api/point", "/api/ondemand"):
                lat = float(q["lat"])
                lon = float(q["lon"])
                if not (-90 <= lat <= 90 and np.isfinite(lon)):
                    return self._json({"error": "lat must be in -90..90"}, 400)
                fn = b.point if u.path == "/api/point" else b.ondemand
                return self._json(fn(lat, lon))
            if u.path == "/api/script":
                lat, lon = float(q["lat"]), float(q["lon"])
                if not (-90 <= lat <= 90 and np.isfinite(lon)):
                    return self._json({"error": "lat must be in -90..90"}, 400)
                name = f"mercury_site_{abs(lat):.3f}{'N' if lat >= 0 else 'S'}_{lon % 360:.3f}E.py"
                body = b.script(lat, lon).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/x-python; charset=utf-8")
                self.send_header("Content-Disposition", f'attachment; filename="{name}"')
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            return self._static(u.path)
        except (KeyError, ValueError) as e:
            return self._json({"error": f"bad request: {e}"}, 400)
        except Exception as e:  # report server-side failures to the page
            import traceback
            traceback.print_exc()
            return self._json({"error": str(e)}, 500)

    def _static(self, path):
        rel = "index.html" if path in ("", "/") else path.lstrip("/")
        full = os.path.realpath(os.path.join(WEB, rel))
        if not full.startswith(os.path.realpath(WEB) + os.sep) or not os.path.isfile(full):
            return self._send(404, b"not found", "text/plain")
        ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
        if full.endswith(".js"):
            ctype = "text/javascript"
        with open(full, "rb") as f:
            self._send(200, f.read(), ctype, "no-cache")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    Handler.backend = Backend()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"serving on http://{a.host}:{a.port}/")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
