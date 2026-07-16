"""config.py — every path, prompt, threshold, and veto rule. No constants
duplicated across modules (MANUAL §4, §6.7)."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

LAS_PATH = REPO_ROOT / "data/raw/UPark_Merged_PS_NAD83_G18_USFT_las.las"

DERIVED_DIR = REPO_ROOT / "data/derived"
SLICES_DIR = REPO_ROOT / "data/slices"
MASKS_DIR = REPO_ROOT / "data/masks"
EVAL_DIR = REPO_ROOT / "data/eval"
OUTPUT_DIR = REPO_ROOT / "data/output"

GRID_META_PATH = SLICES_DIR / "grid_meta.npz"

CHUNK_SIZE = 10_000_000

# ── Class table (MANUAL §5) ─────────────────────────────────────────────────
# id -> name, SAM3 text prompt, starting confidence threshold, LAS class code.
# NOTE: pavement/sidewalk/parking all share LAS code 11 (no standard LAS code
# distinguishes them) — evaluate.py groups by LAS code, so these three are
# necessarily one evaluation bucket regardless of class id.
CLASSES = {
    0: {"name": "pavement", "prompt": "road", "threshold": 0.30, "las_code": 11},
    1: {"name": "sidewalk", "prompt": "sidewalk", "threshold": 0.30, "las_code": 11},
    2: {"name": "parking", "prompt": "parking lot", "threshold": 0.30, "las_code": 11},
    3: {"name": "grass", "prompt": "grass", "threshold": 0.35, "las_code": 3},
    4: {"name": "tree", "prompt": "tree", "threshold": 0.45, "las_code": 5},
    5: {"name": "building", "prompt": "building", "threshold": 0.50, "las_code": 6},
    6: {"name": "vehicle", "prompt": "car", "threshold": 0.50, "las_code": 64},
}
UNLABELLED_LAS_CODE = 1

# ── Eval tiles (MANUAL §6.0) ─────────────────────────────────────────────────
# PLACEHOLDER BOUNDS — these will crop the wrong (or empty) area. A human must
# pick real (x0, y0, x1, y1) in US survey feet by inspecting the site (e.g.
# data/slices/top_down.png or the raw LAS in CloudCompare) before running
# `python -m evaluation.crop_tiles`.
EVAL_TILE_BOUNDS: dict[str, tuple[float, float, float, float]] = {
    "a": (0.0, 0.0, 150.0, 150.0),  # road + parking
    "b": (0.0, 0.0, 150.0, 150.0),  # buildings
    "c": (0.0, 0.0, 150.0, 150.0),  # trees over ground
}
