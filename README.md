# Mercury near-surface temperature model

1D thermal model of Mercury's regolith giving temperature as a function of
**longitude, latitude, local time and depth (0–3.16 m)**, intended as an
environmental input for a Mercury base design simulator.

## Files

| File | Purpose |
|---|---|
| `mercury_thermal.py` | The model: orbit/insolation, regolith properties, numba-compiled 1D heat-conduction solver |
| `generate_grid.py` | Runs the model globally and writes `data/mercury_temperature.nc` |
| `mercury_lookup.py` | `MercuryTemperature` — interpolated lookup at any point, plus exact on-demand runs |
| `polar_terrain.py` | Terrain-aware polar model; writes `data/mercury_polar_<pole>_2km.nc` |
| `polar_site.py` | `PolarSite` — terrain-aware on-demand run for one polar site at native DEM resolution |
| `global_terrain.py` | Outside the polar caps: global slope map, slope-aware table, and `GlobalSite` (full terrain-aware on-demand run anywhere) |
| `site_script.py`, `site_script_template.py` | Per-site data sources, and the downloadable standalone script that reproduces an exact run |
| `setup_data.py` | Downloads the DEMs and builds all data products (skips what exists) |
| `benchmark.py` | Timing and accuracy of on-demand single-site runs (flat, polar and global terrain) |
| `plot_overview.py`, `plot_polar.py` | Diagnostic figures in `data/figures/` |
| `web_server.py`, `web/` | Interactive 3D explorer (local web server) |

The data are not in the repository: about 625 MB of public MESSENGER elevation
models are downloaded, and about 1.3 GB of products are built from them.
`setup_data.py` does both.

```
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python setup_data.py        # all of the steps below: downloads ~625 MB of DEMs,
                                          # then builds the data (~5 h on 8 cores, resumable)
.venv/Scripts/python web_server.py        # http://127.0.0.1:8000/
```

Or step by step:

```
.venv/Scripts/python generate_grid.py      # ~65 s on 8 cores
.venv/Scripts/python polar_terrain.py --pole north   # ~1.2 h (north), ~2.2 h (south) on 8 cores; resumable
.venv/Scripts/python polar_terrain.py --pole south
.venv/Scripts/python global_terrain.py slopes      # ~16 s; needs the global DEM (see below)
.venv/Scripts/python global_terrain.py table       # ~80 min on 8 cores, resumable
.venv/Scripts/python global_terrain.py check       # table interpolation error
.venv/Scripts/python benchmark.py
```

```python
from mercury_lookup import MercuryTemperature
mt = MercuryTemperature()
mt(lon=250.0, lat=-35.0, local_time=6.5, depth=0.4)   # K
lt, z, T = mt.on_demand(lat=-35.0, lon=250.0)         # exact run for one site
```

## Interactive explorer

```
.venv/Scripts/python web_server.py          # then open http://127.0.0.1:8000/
```

Add `--interaction-log FILE` to record site selections, exact runs and script
downloads as JSON lines. Each line has the UTC time, event (`select`,
`exact_run` or `script_download`), visitor, lat, lon, HTTP status, seconds and
the browser's User-Agent string. `visitor` is a random 16-hex-digit ID the page
creates once per browser (kept in localStorage), so visitors can be counted.
Client addresses are not recorded. The file rotates at 10 MB, keeping 10 old files.

The page shows a 3D globe coloured by temperature at a chosen depth
(surface to 3 m). The polar caps are drawn at the 2 km resolution of the
terrain products. Outside them, each ~2 km pixel is coloured using its own
slope. The GPU combines a static slope texture (31 MB PNG, downloaded once
and then cached) with a small per-frame table of slope responses.

- **Time:** play or scrub through the 176-Earth-day solar day. The marker
  shows where the Sun is overhead.
- **Site panel:** click any point, type coordinates, or pick a preset site.
  The sidebar then shows:
  - depth vs local-time heatmap, coloured against human habitability: grey
    at 20 °C, blue saturating at −40 °C, red saturating at +50 °C (`HABIT`
    in `web/charts.js`)
  - temperature over the day at four depths
  - depth profile, now and over the day
  - Sun path, against the terrain horizon at polar sites
  - current values, plus illumination facts at polar sites, or elevation and
    local slope outside the polar caps
- **Linked time:** click or drag any time chart to set the global time.
- **Exact run:** "Run exact model here" calls the full terrain-aware
  on-demand model everywhere: `PolarSite` inside the polar caps, `GlobalSite`
  (665 m DEM) below. It adds real horizons, shadowing and heat from
  surrounding terrain, which the slope-only table leaves out.
- **Data sources:** each site lists the datasets and papers behind its values, with
  links to the dataset pages and downloads. Precomputed tables made by this project
  are marked as such.
- **Python script:** "Download Python script" gives a standalone script (numpy only;
  numba optional) that re-runs the exact model for that site. It computes the
  orbit, sunlight and heat conduction itself. Inlined into it are the site's
  terrain-derived inputs: surface normal, 128-direction horizon, and the
  180-sample series of heat absorbed from surrounding terrain. It checks itself
  against the explorer's result; tested matches are within 0.01 K. It takes about 1 s with numba
  and 5–30 s in pure Python.
- **Units:** temperatures display in °F (default), °C or K. The choice is
  remembered per browser; the data and API stay in kelvin.
- **Saving:** download a CSV of the selected site, in the chosen unit. The URL keeps the site,
  time, depth and view, so it works as a link.

The server binds to localhost by default (`--host 0.0.0.0` to share on a
LAN). Globe frames are rendered from the tables on request: about 0.1 s and
170 KB per frame, 96 frames per solar day. The whole day is rendered,
because the T(lon + 180°) symmetry of the flat model does not hold for
polar terrain. three.js r160 is vendored
in `web/vendor/`, so the page works offline.

## Dataset

`data/mercury_temperature.nc`: `temperature(lon, lat, local_time, depth)`,
2° × 2° grid over 0–178° E, 0–90° N, 96 local times, 64 depth levels
(2 mm spacing at the surface, growing geometrically). Other longitudes and the
southern hemisphere follow from symmetry (handled by `mercury_lookup.py`):

- **T(lon + 180°) = T(lon)** — consequence of the 3:2 spin-orbit resonance.
- **T(−lat) = T(lat)** — obliquity (~0.03°) taken as zero.

**Local time** is *mean* local solar time: 24 "hours" = one 175.94-Earth-day
solar day, 12 h = mean noon. True solar time is not usable as a coordinate
because near perihelion the Sun reverses in the sky (double sunrise/sunset
near 90° and 270° E). The true hour angle and Sun distance are stored as
`true_hour_angle(lon, local_time)` and `sun_distance(lon, local_time)`.

## Physics and parameters

- Insolation from Mercury's eccentric orbit (a = 0.3871 AU, e = 0.2056) with
  exact 3:2 resonance; subsolar point at 0° E at perihelion (IAU convention:
  hot poles at 0°/180° E, warm poles at 90°/270° E). Solar constant 1361 W m⁻².
- Regolith following Hayne et al. (2017), as commonly applied to Mercury
  (cf. Vasavada et al. 1999): density 1100 → 1800 kg m⁻³ with e-folding depth
  H = 0.07 m; contact conductivity 7.4×10⁻⁴ → 3.4×10⁻³ W m⁻¹ K⁻¹; radiative
  conductivity k = k_c(1 + 2.7 (T/350 K)³); temperature-dependent heat
  capacity; albedo 0.07 at normal incidence rising with incidence angle;
  emissivity 0.95; geothermal flux 0.02 W m⁻².
- Backward-Euler finite volumes, 2880 steps per solar day, surface radiation
  fully implicit. Spin-up to periodic steady state (daily-mean change
  < 0.01 K/day) using a coarse-time-step stage and extrapolation of the slow
  deep-equilibration mode.

## Accuracy (numerical)

- vs. brute-force spin-up: ≤ 0.45 K at all tested sites and depths
- time-step / depth-grid refinement: ≤ 1 K (≤ 0.2 K at the surface)
- table interpolation vs. exact on-demand run (off-grid site): ≤ 0.2 K

Physical uncertainty is much larger than this — regolith properties (H,
conductivity, radiative term χ at Mercury's high temperatures, heat-capacity
extrapolation above ~350 K) plausibly shift subsurface temperatures by tens
of K. Sanity check against the literature: equatorial surface max 706 K (hot
pole) / 573 K (warm pole), night minimum ~95–110 K.

## Limitations

- **Slope only outside the polar caps** (table and globe): each 2 km cell's own slope is
  included, but not horizon shadowing, scattered light or IR from
  neighbouring terrain. Slopes steeper than the table's 0.4 gradient (21.8°)
  are clamped; that is about 0.1% of the surface. Slopes are 2 km averages,
  so individual crater walls are steeper than shown. "Run exact model here"
  gives the full treatment. The flat model treats the Sun as a point
  source, so its value at the exact poles (≈ 25 K) is unrealistic; the polar
  products supersede it.
- Laterally uniform regolith properties; no ice, rock, or buried structures.
- Solar constant fixed (no solar-cycle variation, ~0.1%).

## Outside the polar caps: slope-aware model

Data: USGS MESSENGER global DEM v2, 665 m/pixel (64 px/deg)
([Mercury_Messenger_USGS_DEM_Global_665m_v2.tif](https://planetarymaps.usgs.gov/mosaic/Mercury_Messenger_USGS_DEM_Global_665m_v2.tif),
531 MB, into `data/dem/`).

1. **Slope map** (`data/mercury_slopes_2km.nc`). The DEM is averaged in 3×3
   blocks (~2 km). East and north gradients come from central differences.
   Median slope is 1.2°; the 90/99/99.9th percentiles are 4.3/12.5/21.5°.
2. **Slope table** (`data/mercury_slope_table.nc`, 218 MB). Temperature at
   each table node, computed for a tilted surface with a flat horizon:
   - nodes: latitude 0–82° (2° steps) × longitude 0–178° (2° steps) ×
     9 × 9 east/north gradient nodes (0, ±0.04, ±0.1, ±0.2, ±0.4) ×
     48 local times × 13 depths
   - a tilted surface sees a fraction (1 − cos s)/2 of the level ground
     around it. Light reflected from that ground and its infrared emission
     come from the flat model.
   - southern latitudes and longitudes 180–360° use the same symmetries as
     the flat model
3. **Lookup**. `MercuryTemperature()` reads each point's slope from the map
   and interpolates the table in 6D. The polar products still override it at
   their edge latitude and above.

Accuracy of table interpolation vs direct runs, at 400 random sites with real slopes:

| Depth | Mean error | 99th percentile | Max | Worst daily-mean bias |
|---|---|---|---|---|
| surface | 0.8 K | 23 K | 56 K | 5.3 K |
| 0.1 m | 0.15 K | 1.5 K | 17 K | 4.9 K |
| 0.5 m | 0.13 K | 1.1 K | 5 K | 4.8 K |
| 1 m | 0.13 K | 1.1 K | 5 K | 4.8 K |

- Large surface errors come from time interpolation across sunrise and
  sunset, the same as in the flat table.
- The worst bias (≈ 5 K) is near 78°N at the warm-pole longitude, where
  temperature changes sharply with latitude.
- Everywhere else, daily-mean biases are below 2 K.

Example: at 40°N, an 11° slope facing the equator peaks about 29 K above
level ground, and one facing the pole about 39 K below it.

`GlobalSite` (on demand, any latitude) uses the same method as `PolarSite`:
- a 665 m near grid out to ±25 km, plus a 2 km far grid out to 250 km
- both on an oblique stereographic projection centred on the site
- 128 ray-traced horizon azimuths, a finite solar disk, single-bounce
  scattered light, and terrain IR from slope-table temperatures

## Polar regions (≥ 75° N, ≥ 70° S): terrain-aware model

At the poles the Sun stays within a few degrees of the horizon, so
topography dominates: craters hold permanently shadowed regions (PSRs) at
40–110 K, while slopes facing the Sun reach over 500 K.

**Topography**:
- North: MESSENGER MLA laser altimetry, 250 m polar stereographic
  (`hdem_75n_250m`, PDS Geosciences MLA GDR). It covers 75–90° N, so the
  terrain beyond the cap edge (70–75° N, needed for horizons) comes from the
  USGS global DEM, blended over 0.5°. The two agree to a median of 30–50 m
  where they overlap.
- South: MDIS stereo DEM, 665 m (`MSGR_DEM_USG_SP_I_V02`, PDS Imaging
  MESSDEM).
- Files go in `data/dem/`.

**Method** (`polar_terrain.py`):
- 2 km working grid: the cap (to 75° N / 70° S, `cap_lat` in `polar_terrain.py`)
  plus a 5° margin of surrounding terrain.
- Exact 3-D geometry on the sphere, so planetary curvature is handled.
- Ray-traced horizons: 64 azimuths, out to 250 km.
- Sunlight:
  - finite solar disk, so partial illumination near the horizon is captured
  - cell slope and aspect
  - sunlight scattered once from surrounding terrain
- Thermal infrared from surrounding terrain uses Lambertian view factors
  from the same rays.
- Mutual infrared heating is iterated in two passes:
  - pass 1: coarse
  - pass 2: full resolution
- The same 1D regolith column runs under every cell.
- Margin cells outside the product use the flat model.

**Products** (`data/mercury_polar_<pole>_2km.nc`: north 642 × 642 cells,
south 860 × 860, stored as int16 with scale factors; the edge latitude is
the file's `lat_min` attribute):
- `temperature(y, x, local_time, depth)`: 48 local times and 13 depths
  from 0 to 3 m.
- `permanent_shadow`
- `sun_visible_fraction`
- `direct_lit_fraction`
- `mean_direct_flux`
- `sky_view_factor`
- `horizon`
- `elevation` and `slope`
- `dem_quality`

`MercuryTemperature` automatically uses these products poleward of their `lat_min`
(pass `terrain=False` to get the flat model). `on_demand()` uses `PolarSite`
there. It runs a 25 km neighbourhood at native DEM resolution, nested in a
2 km grid out to 250 km. Infrared from distant terrain comes from the
product.

Statistics (north file 322 MB, 324,028 cells ≥ 75° N; south file 597 MB,
581,256 cells ≥ 70° S):

| | North ≥ 75° | South ≥ 70° | North ≥ 80° | South ≥ 80° |
|---|---|---|---|---|
| PSR area (2 km cells) | 15,600 km² (1.2%) | 2,900 km² (0.13%) | 13,500 km² (2.4%) | 2,100 km² (0.4%) |
| PSR maximum surface T, median | 119 K | 99 K | 109 K | 82 K |
| PSR mean T at 0.5 m, median / minimum | 81 / 31 K | 69 / 32 K | 74 / 31 K | 60 / 32 K |
| Hottest surface (Sun-facing slopes) | 599 K | 620 K | 569 K | 508 K |

**Checks**:
- Infrared iteration: the production two-pass result differs from three
  full passes by 0.01 K on average and 1.3 K at most (test region ≥ 87° N).
- Every cell converged.
- The table and on-demand runs are consistent: averaging 64 native-resolution
  runs over one 2 km cell (82° N, 30° E) gives 211.8 K at 0.5 m; the table
  gives 212.2 K.

**Resolution matters.** A table value is the average for a 2 km cell. Within
one cell, 250 m patches receive mean direct sunlight from about 100 to about
860 W m⁻². At 80–82° N single sites differ from their cell by up to ±40 K at
the surface and ±25 K at 0.5 m. Use `on_demand()` for a specific base site.

**Polar limitations**:
- South pole: within 2.5° of the pole (inside about 87.5° S) the south DEM
  has no stereo coverage, so its interpolated terrain is unreliable. These
  cells are flagged `dem_quality = 0`. This area includes most of Chao
  Meng-Fu crater, so the south PSR area is underestimated.
- The south DEM also shows faceting artifacts elsewhere.
- Scattering is single-bounce. Infrared from terrain beyond 250 km is
  neglected, and so is shadowing by features smaller than the DEM resolution.
- Zero obliquity, and the same regolith everywhere: no ice deposits
  (radar-bright PSR deposits would alter the thermal properties).

## On-demand performance (8-core Windows laptop)

| Operation | Time |
|---|---|
| Single site, on demand (one core) | 19–82 ms, median 55 ms (up to ~180 ms seen under machine load) |
| First call in a process (load compiled code) | ~0.3 s |
| Parallel batch | ~10–19 ms per site effective |
| Full 4140-site global grid | ~65 s |
| Table lookup | ~1.4 µs per point (flat), ~1.7 µs (polar), vectorised |
| Polar site, terrain-aware, first in a process | ~2.1–2.5 s (load compiled code, DEM, product) |
| Polar site, new neighbourhood (builds 2 km far grid) | ~0.9–1.2 s |
| Polar site, within cached neighbourhood (~50 km) | 43–64 ms |
| Polar product, north (324k cells) | 2.3 h on 2 cores (AMD EPYC 9R14); est. ~1.2 h on 8 |
| Polar product, south (581k cells) | 4.3 h on 2 cores; est. ~2.2 h on 8 |
| Slope-aware lookup outside the caps | ~0.7 µs per point, vectorised |
| Terrain-aware site outside the caps, new neighbourhood (`GlobalSite`) | ~0.86–0.92 s (first in a process ~1.2 s with cached compiled code) |
| Terrain-aware site outside the caps, within cached neighbourhood | 63–80 ms |
| Slope table, global (7.4M slope columns) | ~82 min on 8 cores |
