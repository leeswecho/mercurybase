"""
Download the elevation models and build every data product the web server
needs. Each step is skipped if its output already exists, and the long
steps checkpoint their progress, so the script can be re-run after an
interruption.

  downloads (~625 MB)    MESSENGER DEMs into data/dem/
  generate_grid.py       data/mercury_temperature.nc       ~1 min
  polar_terrain.py       data/mercury_polar_<pole>_2km.nc  ~33 min per pole
  global_terrain.py      data/mercury_slopes_2km.nc        ~20 s
                         data/mercury_slope_table.nc       ~80 min
(times on an 8-core laptop)

Usage:  python setup_data.py [--download-only]
"""

import argparse
import os
import subprocess
import sys
import time
import urllib.request

from global_terrain import GLOBAL_DEM, SLOPES_PATH, TABLE_PATH
from polar_terrain import DEM_DIR, DEMS, HERE

DATA = os.path.join(HERE, "data")
DOWNLOADS = [(d["file"], d["url"]) for d in DEMS.values()] + [(GLOBAL_DEM["file"], GLOBAL_DEM["url"])]
STEPS = [
    (os.path.join(DATA, "mercury_temperature.nc"), ["generate_grid.py"]),
    (os.path.join(DATA, "mercury_polar_north_2km.nc"), ["polar_terrain.py", "--pole", "north"]),
    (os.path.join(DATA, "mercury_polar_south_2km.nc"), ["polar_terrain.py", "--pole", "south"]),
    (SLOPES_PATH, ["global_terrain.py", "slopes"]),
    (TABLE_PATH, ["global_terrain.py", "table"]),
]


def download(name, url):
    path = os.path.join(DEM_DIR, name)
    if os.path.exists(path):
        print(f"have {name}")
        return
    os.makedirs(DEM_DIR, exist_ok=True)
    tmp = path + ".part"
    print(f"downloading {url}")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done, t0 = 0, time.time()
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:.0f} / {total / 1e6:.0f} MB "
                      f"({done / 1e6 / max(time.time() - t0, 1e-3):.1f} MB/s)", end="", flush=True)
    print()
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--download-only", action="store_true")
    a = ap.parse_args()
    for name, url in DOWNLOADS:
        download(name, url)
    if a.download_only:
        return
    for out, cmd in STEPS:
        if os.path.exists(out):
            print(f"have {os.path.relpath(out, HERE)}")
            continue
        print(f"\n=== {' '.join(cmd)} -> {os.path.relpath(out, HERE)}", flush=True)
        subprocess.run([sys.executable, *cmd], cwd=HERE, check=True)
    print("\nall data ready: python web_server.py")


if __name__ == "__main__":
    main()
