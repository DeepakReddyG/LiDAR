"""classification/fuse.py — thresholds + physics veto + priority painting
(MANUAL §6.5, T6).

SAM3 says what things look like; the LiDAR knows what they are. Per class:

    1. threshold conf_<class>.npy at its config threshold
    2. physics veto with the ortho stat grids (ExG / HAG / void_mask)
    3. paint label_grid most-specific-first, first claim wins
    4. 5×5 majority filter to kill single-pixel speckle

Outputs (data/masks/): label_grid.npy ((H, W) int32, −1 unlabelled),
conf_grid.npy (winning confidence per pixel), veto_stats.json.

The per-class veto-rejection rate is the diagnostic: high rejection = the
threshold/veto is the weak link; low rejection but low IoU = the prompt is.
"""

from __future__ import annotations

import json

import numpy as np

from config import CLASSES, FUSE_PRIORITY, MASKS_DIR, SLICES_DIR, VETO

_NAME_TO_ID = {info["name"]: cid for cid, info in CLASSES.items()}
_THRESH = {info["name"]: info["threshold"] for info in CLASSES.values()}


def veto_mask(
    name: str, exg: np.ndarray, hag: np.ndarray, void: np.ndarray
) -> np.ndarray:
    """Physics veto per MANUAL §6.5 — NaN stat cells compare False, which
    conveniently vetoes them along with the void mask."""
    v = VETO
    with np.errstate(invalid="ignore"):
        if name == "tree":
            keep = (exg > v["tree_exg"]) | (hag > v["tree_hag"])
        elif name == "grass":
            keep = (exg > v["grass_exg"]) & (hag < v["grass_hag"])
        elif name == "building":
            keep = hag > v["building_hag"]
        elif name in ("pavement", "sidewalk", "parking"):
            keep = hag < v["pavement_hag"]
        elif name == "vehicle":
            keep = (hag > v["vehicle_hag"][0]) & (hag < v["vehicle_hag"][1])
        else:
            raise ValueError(name)
    return keep & ~void


def majority_filter(grid: np.ndarray, size: int = 5) -> np.ndarray:
    """Mode filter over a small label alphabet, numpy-only."""
    H, W = grid.shape
    values = np.unique(grid)
    pad = np.pad(grid, size // 2, mode="edge")
    counts = np.zeros((len(values), H, W), dtype=np.int16)
    for i in range(size):
        for j in range(size):
            win = pad[i : i + H, j : j + W]
            for k, val in enumerate(values):
                counts[k] += win == val
    return values[counts.argmax(axis=0)].astype(grid.dtype)


def fuse(
    conf: dict[str, np.ndarray], exg: np.ndarray, hag: np.ndarray, void: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Pure core: confidence grids → (label_grid, conf_grid, veto_stats)."""
    shape = next(iter(conf.values())).shape
    label = np.full(shape, -1, dtype=np.int32)
    stats = {}

    for name in FUSE_PRIORITY:
        claimed = conf[name] >= _THRESH[name]
        kept = claimed & veto_mask(name, exg, hag, void)
        n_claim, n_kept = int(claimed.sum()), int(kept.sum())
        stats[name] = {
            "claimed_px": n_claim,
            "kept_px": n_kept,
            "veto_rejection": round(1 - n_kept / n_claim, 3) if n_claim else None,
        }
        label[(label == -1) & kept] = _NAME_TO_ID[name]

    label = majority_filter(label, size=5)

    conf_grid = np.zeros(shape, dtype=np.float32)
    for name, cid in _NAME_TO_ID.items():
        m = label == cid
        conf_grid[m] = conf[name][m]
    return label, conf_grid, stats


def run_fuse() -> None:
    exg = np.load(SLICES_DIR / "exg_grid.npy")
    hag = np.load(SLICES_DIR / "hag_grid.npy")
    void = np.load(SLICES_DIR / "void_mask.npy")
    conf = {
        info["name"]: np.load(MASKS_DIR / f"conf_{info['name']}.npy")
        for info in CLASSES.values()
    }

    label, conf_grid, stats = fuse(conf, exg, hag, void)

    np.save(MASKS_DIR / "label_grid.npy", label)
    np.save(MASKS_DIR / "conf_grid.npy", conf_grid)
    (MASKS_DIR / "veto_stats.json").write_text(json.dumps(stats, indent=2))

    n = label.size
    print("  veto rejection per class (high = threshold/veto is the weak link):")
    for name, s in stats.items():
        rej = f"{s['veto_rejection']:.0%}" if s["veto_rejection"] is not None else "n/a"
        print(
            f"    {name:<10s} claimed {s['claimed_px'] / n:6.1%}  kept {s['kept_px'] / n:6.1%}  rejected {rej}"
        )
    print(
        f"  labelled {np.mean(label >= 0):.1%} of pixels  →  label_grid.npy, conf_grid.npy"
    )


def _self_check() -> None:
    shape = (20, 20)
    z = np.zeros(shape, dtype=np.float32)
    exg = np.full(shape, 0.2, np.float32)
    exg[:, 10:] = 0.0
    hag = np.zeros(shape, np.float32)
    hag[10:, :] = 10.0
    void = np.zeros(shape, bool)
    void[0, 0] = True
    # quadrants: TL green+low=grass, TR grey+low=pavement,
    #            BL green+tall=tree,  BR grey+tall=building
    conf = {n: z.copy() for n in _NAME_TO_ID}
    conf["grass"][:10, :10] = 0.9
    conf["pavement"][:10, 10:] = 0.9
    conf["tree"][10:, :10] = 0.9
    conf["building"][10:, 10:] = 0.9
    conf["vehicle"][0, 5] = 0.9  # claimed but vetoed: hag=0 not in (1,9)
    conf["grass"][0, 0] = 0.9  # claimed but void
    conf["grass"][5, 15] = 0.9  # speckle: lone grass claim amid pavement
    #                                      (vetoed anyway: exg=0 there)
    conf["pavement"][5, 5] = 0.95  # pavement also claims one grass px —
    #                                      passes veto (hag<1.5) and pavement
    #                                      outranks grass, but majority erases it

    label, conf_grid, stats = fuse(conf, exg, hag, void)
    g = {n: _NAME_TO_ID[n] for n in _NAME_TO_ID}
    assert label[2, 2] == g["grass"] and label[2, 15] == g["pavement"]
    assert label[15, 2] == g["tree"] and label[15, 15] == g["building"]
    assert label[5, 5] == g["grass"], "majority filter failed to erase speckle"
    assert stats["vehicle"]["kept_px"] == 0, "vehicle veto failed"
    assert conf_grid[2, 2] == np.float32(0.9)
    print("self-check OK: veto, priority, void, speckle removal")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--self-check", action="store_true")
    if p.parse_args().self_check:
        _self_check()
    else:
        run_fuse()
