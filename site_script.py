"""
Data provenance for a site, and a downloadable standalone Python script that
reproduces the exact (terrain-aware) model run there.

The script is site_script_template.py with its inputs block replaced by the
site's surface normal, horizon profile and terrain heat series, which come
from PolarSite.run / GlobalSite.run ("site_inputs").
"""

import os

import numpy as np

from polar_terrain import DEMS
from global_terrain import GLOBAL_DEM

TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "site_script_template.py")

PHYSICS = [
    dict(name="Hayne et al. (2017), Global regolith thermophysical properties of the Moon "
              "from the Diviner Lunar Radiometer Experiment, JGR Planets 122, 2371-2400",
         role="Regolith density, conductivity, heat capacity and albedo law",
         url="https://doi.org/10.1002/2017JE005387", label="paper"),
    dict(name="Vasavada, Paige & Wood (1999), Near-surface temperatures on Mercury and the Moon "
              "and the stability of polar ice deposits, Icarus 141, 179-193",
         role="Heat-conduction model formulation for Mercury",
         url="https://doi.org/10.1006/icar.1999.6175", label="paper"),
    dict(name="NASA Mercury Fact Sheet",
         role="Orbit: semi-major axis, eccentricity, orbital period",
         url="https://nssdc.gsfc.nasa.gov/planetary/factsheet/mercuryfact.html",
         label="fact sheet"),
]
GLOBAL = dict(name="USGS MESSENGER global DEM v2, 665 m/pixel (MDIS stereo)",
              role="Elevation and slope outside the polar caps",
              url="https://astrogeology.usgs.gov/search/map/mercury_messenger_global_dem_665m",
              download=GLOBAL_DEM["url"])
POLAR = {
    "north": dict(name="MESSENGER MLA north polar gridded DEM HDEM_75N_250M, 250 m/pixel (laser altimetry)",
                  role="Terrain of the north polar cap (poleward of 75° N)",
                  url="https://pds-geosciences.wustl.edu/missions/messenger/mla.htm",
                  download=DEMS["north"]["url"]),
    "south": dict(name="MESSENGER MDIS south polar stereo DEM MSGR_DEM_USG_SP_I_V02, 665 m/pixel",
                  role="Terrain of the south polar cap (poleward of 70° S)",
                  url="https://planetarydata.jpl.nasa.gov/img/data/messenger/MESSDEM_1001/",
                  download="https://planetarydata.jpl.nasa.gov/img/data/messenger/MESSDEM_1001/"
                           "DEM/POLAR/IMG/MSGR_DEM_USG_SP_I_V02.IMG"),
}
GENERATED = "Computed by this project from the datasets above; not published online"


def sources(lat, kind, terrain=True, polar=None):
    """
    Datasets behind the values shown for a site.
    kind: "table" (precomputed lookup) or "exact" (on-demand run).
    polar: whether the terrain-aware polar model covers the site (default: |lat| >= 80).
    """
    polar = abs(lat) >= 80.0 if polar is None else polar
    pole = "north" if lat > 0 else "south"
    out = []
    if terrain and polar:
        out.append(dict(POLAR[pole]))
        if pole == "north":  # the MLA DEM stops at 75 N
            out.append(dict(GLOBAL, role="Terrain equatorward of 75° N, for horizons near the cap edge"))
        if kind == "table":
            out.append(dict(name=f"Terrain-aware {pole} polar product, 2 km cells",
                            role="Temperatures with horizons, slopes and terrain heat", url=None,
                            note=GENERATED + " (polar_terrain.py)"))
        else:
            out[0]["role"] = "Terrain at native resolution within 25 km, 2 km out to 250 km"
            out.append(dict(name=f"Terrain-aware {pole} polar product, 2 km cells", url=None,
                            role="Temperatures of surrounding terrain (its infrared heat)",
                            note=GENERATED + " (polar_terrain.py)"))
    elif terrain:
        out.append(dict(GLOBAL))
        if kind == "table":
            out.append(dict(name="Slope-aware temperature table (2 km slope, flat horizon)", url=None,
                            role="Temperatures for the local slope",
                            note=GENERATED + " (global_terrain.py)"))
        else:
            out[-1]["role"] = "Terrain at 665 m within 25 km, 2 km out to 250 km"
            out.append(dict(name="Slope-aware temperature table", url=None,
                            role="Temperatures of surrounding terrain (its infrared heat)",
                            note=GENERATED + " (global_terrain.py)"))
    else:
        out.append(dict(name="Flat-terrain temperature table, 2° grid", url=None,
                        role="Temperatures for level ground", note=GENERATED + " (generate_grid.py)"))
    return out + [dict(s) for s in PHYSICS]


def _fmt(values, fmt, per_line=8, indent="    "):
    vals = [format(float(v), fmt) for v in values]
    lines = [", ".join(vals[i:i + per_line]) for i in range(0, len(vals), per_line)]
    return "[\n" + ",\n".join(indent + ln for ln in lines) + ",\n]"


def render(lat, lon, r):
    """Standalone script text for the site, from an exact run result r."""
    si = r["site_inputs"]
    naz = si["horizon"].size
    # horizon bins are map-frame angles; express them as compass azimuths
    a = 2 * np.pi * np.arange(naz) / naz
    em, nm = si["em"], si["nm"]
    az = np.rad2deg(np.arctan2(np.cos(a) * em[0] + np.sin(a) * em[1],
                               np.cos(a) * nm[0] + np.sin(a) * nm[1])) % 360.0
    o = np.argsort(az)
    polar = "horizon_azimuth_deg" not in r  # GlobalSite reports compass azimuths itself
    prov = sources(lat, "exact", polar=polar)
    prov_txt = ",\n".join(
        "        dict(name={!r},\n             role={!r},\n             url={!r})".format(
            p["name"], p["role"], p.get("download") or p.get("url") or p.get("note"))
        for p in prov)
    T = np.asarray(r["T"])
    expected = "[\n" + "\n".join("    " + _fmt(row, ".2f", per_line=13, indent="")
                                 .replace("[\n", "[").replace(",\n]", "],") for row in T) + "\n]"
    pole_note = ("USGS global DEM (665 m)" if not polar else
                 "north polar MLA DEM (250 m)" if lat > 0 else "south polar MDIS DEM (665 m)")
    inputs = f'''\
# Generated by the Mercury temperature explorer for {lat:.4f} N, {lon % 360:.4f} E.
SITE = dict(
    lat_deg={lat:.6f}, lon_deg={lon % 360:.6f},
    # centre of the DEM cell the model ran on
    cell_lat_deg={np.rad2deg(si["lat"]):.6f}, cell_lon_deg={np.rad2deg(si["lon"]) % 360:.6f},
    # unit normal of the ground (east, north, up), from the {pole_note}
    normal_enu=({si["n_enu"][0]:.8f}, {si["n_enu"][1]:.8f}, {si["n_enu"][2]:.8f}),
    elevation_m={r["elevation_m"]:.1f}, slope_deg={r["slope_deg"]:.3f},
    sky_view_factor={float(r["sky_view_factor"]):.5f},
    provenance=[
{prov_txt},
    ],
)

# Horizon: elevation angle (deg) of the skyline toward each compass azimuth
# (deg, clockwise from north), ray-traced from the site across the DEM:
# native resolution within 25 km, 2 km cells out to 250 km, with the
# planet's curvature included.
HORIZON_AZIMUTH_DEG = {_fmt(az[o], ".3f")}
HORIZON_ELEVATION_DEG = {_fmt(np.rad2deg(si["horizon"][o]), ".4f")}

# Heat absorbed from surrounding terrain (W m-2), at {si["Find"].size} equally spaced times
# through the solar day, starting at the reference perihelion (Sun overhead
# at 0 deg E). It is the sum over every terrain cell the site can see of
# (view factor) x (sunlight that cell scatters + infrared it emits), using
# surface temperatures of those cells from the explorer's precomputed tables.
INDIRECT_FLUX = {_fmt(si["Find"], ".3f")}

# The explorer's result for this site (K), 48 local times (00:00, 00:30, ...)
# x 13 depths (OUT_DEPTHS), used only to check that this script reproduces it.
EXPECTED = {expected}
'''
    src = open(TEMPLATE, encoding="utf-8").read()
    i0 = src.index("# @@INPUTS@@")
    i1 = src.index("# @@END INPUTS@@") + len("# @@END INPUTS@@")
    return src[:i0] + inputs.rstrip("\n") + src[i1:]
