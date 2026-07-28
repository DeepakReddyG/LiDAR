"""segmentation/segment_sam3.py — drive mlx_sam3 across ortho tiles, stitch
per-class confidence grids (MANUAL §6.4, T5).

Per tile (data/slices/tiles/tile_r{r0}_c{c0}.png), POST /segment_batch with
all class prompts → raw detections. Per class:

    1. drop detections whose bbox covers > BBOX_FRAC_MAX of the tile
       (observed whole-image-box failure mode)
    2. union surviving instance masks into a semantic confidence map:
       per pixel, max score of any covering mask
    3. stitch tiles into site-wide grids (max over the 25 % overlaps)

Outputs: data/masks/conf_<class>.npy — (rows, cols) float32 raw scores.
Raw per-tile responses are cached in data/masks/raw/*.json so re-stitching
never re-runs inference; delete the cache to force fresh inference.
"""

from __future__ import annotations

import base64
import json
import re

import numpy as np

from config import (
    BBOX_FRAC_MAX,
    CLASSES,
    GRID_META_PATH,
    MASKS_DIR,
    SAM3_URL,
    SLICES_DIR,
)

TILES_DIR = SLICES_DIR / "tiles"
RAW_DIR = MASKS_DIR / "raw"


def rle_to_mask(rle: dict) -> np.ndarray:
    """Inverse of the backend's mask_to_rle: alternating background/foreground
    run lengths, row-major, first run is background (possibly length 0)."""
    h, w = rle["size"]
    flat = np.zeros(h * w, dtype=bool)
    pos, val = 0, False
    for run in rle["counts"]:
        if val:
            flat[pos : pos + run] = True
        pos += run
        val = not val
    return flat.reshape(h, w)


def segment_tile(png_path, prompts: list[str]) -> dict:
    """POST one tile to /segment_batch, with a JSON cache.

    Cache merges by prompt: re-running for new prompts keeps old results.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache = RAW_DIR / (png_path.stem + ".json")
    old = json.loads(cache.read_text()) if cache.exists() else {"results": {}}
    old_results = old.get("results") or {}
    if set(prompts) <= set(old_results):
        return old
    if old_results:
        print(f"  {png_path.stem}: cache missing prompts — re-running")

    import requests

    try:
        resp = requests.post(
            f"{SAM3_URL}/segment_batch",
            json={
                "image": base64.b64encode(png_path.read_bytes()).decode(),
                "prompts": prompts,
            },
            timeout=600,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.Timeout:
        raise SystemExit(
            f"SAM3 backend timed out on tile {png_path.stem} "
            f"({SAM3_URL}/segment_batch, timeout=600s). It may be overloaded "
            "or stuck — check its logs and rerun; cached tiles resume "
            "automatically."
        )
    except requests.ConnectionError:
        raise SystemExit(
            f"SAM3 backend not reachable at {SAM3_URL}.\n"
            "Start it in its own environment:\n"
            "  cd segmentation/mlx_sam3 && uv run python app/backend/main.py"
        )
    except requests.HTTPError as exc:
        raise SystemExit(
            f"SAM3 backend returned {resp.status_code} for tile "
            f"{png_path.stem} ({SAM3_URL}/segment_batch): {exc}"
        )
    except ValueError as exc:
        raise SystemExit(
            f"SAM3 backend returned a non-JSON response for tile "
            f"{png_path.stem} ({SAM3_URL}/segment_batch): {exc}\n"
            f"Response body (first 200 chars): {resp.text[:200]!r}"
        )

    merged = _merge_cache_results(old, data)
    cache.write_text(json.dumps(merged))
    return merged


def tile_confidence(
    dets: list[dict], H: int, W: int, tile_name: str = "?"
) -> np.ndarray:
    """Union instance masks into per-pixel max-score confidence. Instance
    identity is noise here — coarse civil classes need per-pixel semantics,
    and semantic union makes stitching a max instead of instance matching."""
    conf = np.zeros((H, W), dtype=np.float32)
    for det in dets:
        x0, y0, x1, y1 = det["bbox"]
        if (x1 - x0) * (y1 - y0) > BBOX_FRAC_MAX * H * W:
            continue  # whole-image box failure mode
        mask = rle_to_mask(det["mask_rle"])
        if mask.shape != (H, W):  # mask at model res — shouldn't happen
            print(
                f"  WARNING {tile_name}: dropped detection, mask shape "
                f"{mask.shape} != tile ({H}, {W})"
            )
            continue
        np.maximum(conf, np.where(mask, np.float32(det["score"]), 0.0), out=conf)
    return conf


def run_segment_sam3() -> None:
    meta = np.load(GRID_META_PATH)
    rows, cols = int(meta["rows"]), int(meta["cols"])
    prompts = [info["prompt"] for info in CLASSES.values()]

    tiles = sorted(TILES_DIR.glob("tile_r*_c*.png"))
    if not tiles:
        raise FileNotFoundError(f"no tiles in {TILES_DIR} — run ortho stage first")

    grids = {p: np.zeros((rows, cols), dtype=np.float32) for p in prompts}
    for tile_path in tiles:
        r0, c0 = map(int, re.match(r"tile_r(\d+)_c(\d+)", tile_path.stem).groups())
        data = segment_tile(tile_path, prompts)
        H, W = data["height"], data["width"]
        n_dets = sum(len(v) for v in data["results"].values())
        print(f"  {tile_path.stem}: {n_dets} detections")
        for prompt in prompts:
            conf = tile_confidence(
                data["results"].get(prompt, []), H, W, tile_name=tile_path.stem
            )
            region = grids[prompt][r0 : r0 + H, c0 : c0 + W]
            np.maximum(region, conf[: region.shape[0], : region.shape[1]], out=region)

    for info in CLASSES.values():
        out = MASKS_DIR / f"conf_{info['name']}.npy"
        np.save(out, grids[info["prompt"]])
        g = grids[info["prompt"]]
        print(f"  {out.name}: coverage>0.1 {(g > 0.1).mean():.1%}  max {g.max():.2f}")


def _merge_cache_results(old: dict, data: dict) -> dict:
    """Union per-prompt results so a partial re-run cannot wipe prior prompts."""
    merged = {**old, **data}
    merged["results"] = {**(old.get("results") or {}), **(data.get("results") or {})}
    return merged


def _self_check() -> None:
    # Cache merge: new prompt must not erase previously cached ones
    merged = _merge_cache_results(
        {"results": {"road": [{"score": 0.9}]}, "width": 10},
        {"results": {"tree": [{"score": 0.8}]}, "width": 10, "height": 10},
    )
    assert set(merged["results"]) == {"road", "tree"}
    assert merged["height"] == 10

    # RLE round-trip against the backend's encoding scheme
    rng = np.random.default_rng(3)
    mask = rng.random((17, 23)) > 0.6
    flat = mask.flatten()
    change = np.where(np.diff(flat) != 0)[0] + 1
    runs = np.diff(np.concatenate([[0], change, [flat.size]])).tolist()
    if flat[0]:
        runs = [0] + runs
    assert np.array_equal(rle_to_mask({"counts": runs, "size": [17, 23]}), mask)

    # stitching: two overlapping synthetic tiles, max wins in the overlap
    grid = np.zeros((10, 16), dtype=np.float32)
    t1 = np.full((10, 10), 0.4, dtype=np.float32)
    t2 = np.full((10, 10), 0.7, dtype=np.float32)
    for (r0, c0), t in [((0, 0), t1), ((0, 6), t2)]:
        region = grid[r0 : r0 + 10, c0 : c0 + 10]
        np.maximum(region, t[: region.shape[0], : region.shape[1]], out=region)
    assert (
        grid[5, 3] == np.float32(0.4)
        and grid[5, 8] == np.float32(0.7)
        and grid[5, 15] == np.float32(0.7)
    )

    # bbox filter: whole-image box dropped, small box kept
    dets = [
        {
            "bbox": [0, 0, 100, 100],
            "score": 0.9,
            "mask_rle": {"counts": [0, 100 * 100], "size": [100, 100]},
        },
        {
            "bbox": [10, 10, 30, 30],
            "score": 0.5,
            "mask_rle": {"counts": [1010, 20], "size": [100, 100]},
        },
    ]
    conf = tile_confidence(dets, 100, 100)
    assert conf.max() == np.float32(0.5), "whole-image box not dropped"
    print("self-check OK: cache merge, RLE round-trip, max-stitch, bbox filter")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--self-check", action="store_true")
    if p.parse_args().self_check:
        _self_check()
    else:
        run_segment_sam3()
