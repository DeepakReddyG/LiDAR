"""demo/make_demo_tile.py — cut the small road demo tile from the full survey.

Readers do not need to run this: the tile is committed as
demo/tile_road_demo.las. Kept so the tile's origin is reproducible.

    python demo/make_demo_tile.py
"""

import sys
from pathlib import Path

import laspy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import LAS_PATH
from evaluation.crop_tiles import crop_tile

# 100 x 100 US survey ft: road intersection, parking, cars, grass, trees.
# Outside the protected A/B regions and the development tile C (holdout.json).
DEMO_BOUNDS = (1390944.38, 448015.05, 1391044.38, 448115.05)
# Keep every 12th point: ~12.2 M -> ~1.0 M points (~43 MB), small enough for
# GitHub and still ~100 points per square foot.
THIN_STEP = 12

if __name__ == "__main__":
    out = Path(__file__).parent / "tile_road_demo.las"
    crop_tile(LAS_PATH, DEMO_BOUNDS, out)
    las = laspy.read(out)
    las.points = las.points[np.arange(0, len(las.points), THIN_STEP)]
    las.write(str(out))
    print(f"  thinned to {len(las.points):,} points")
