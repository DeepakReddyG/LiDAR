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

# ── Ground / DTM (MANUAL §6.2) ───────────────────────────────────────────────
DECIMATE_CELL = 2.0      # ft — lowest-Z decimation cell for CSF
CLOTH_RESOLUTION = 2.0   # ft — CSF cloth resolution; raise if DTM embosses buildings
GROUND_OUTLIER_FT = 5.0  # ft — drop decimated cells this far below 5×5 median

# ── Rule-only baseline (T3) ──────────────────────────────────────────────────
# Standalone classifier thresholds — NOT the §6.5 veto table (vetoes assume
# SAM3 already claimed the class; standalone rules need tighter conjunctions).
BASELINE = dict(
    veg_exg=0.05,      # ExG above this = vegetated
    hard_exg=0.05,     # ExG at/below this = hard surface
    tree_hag=6.0,      # vegetated and taller than this = tree
    grass_hag=2.0,     # vegetated and lower than this = grass
    building_hag=8.0,  # hard surface taller than this = building
    pavement_hag=1.5,  # hard surface lower than this = pavement
)

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
# 150×150 ft boxes picked visually from data/slices/top_down.png
# (grid: x = 1390744.38 + col*0.5, y = 448710.05 - row*0.5).
EVAL_TILE_BOUNDS: dict[str, tuple[float, float, float, float]] = {
    "a": (1390954.38, 447810.05, 1391104.38, 447960.05),  # road + parking row (~15 cars) + 2 trees
    "b": (1391169.38, 447785.05, 1391319.38, 447935.05),  # two large buildings + courtyard
    "c": (1391184.38, 448035.05, 1391334.38, 448185.05),  # two tree canopies over open ground
}
