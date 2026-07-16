"""
segment.py — SAM2 automatic mask generation on the top-down LiDAR slice.

Pipeline position:  slice.py → [segment.py] → classify.py → map_back.py

Input:
    data/slices/top_down.png          (false-colour RGB from slice.py)

Outputs:
    data/masks/masks.npy              (M, H, W) bool array, one mask per segment
    data/masks/mask_meta.npy          list of M dicts with SAM2 mask metadata
    data/masks/overlay.png            visualisation: masks overlaid on source image
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

# ── SAM2 imports ──────────────────────────────────────────────────────────────
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


# ── Config ────────────────────────────────────────────────────────────────────
SAM2_CONFIG  = "configs/sam2.1/sam2.1_hiera_s.yaml"   # small model
SAM2_CKPT    = "models/sam2/sam2.1_hiera_small.pt"
DEVICE       = "mps"        # Apple Silicon — change to "cuda" or "cpu" if needed

SLICE_PATH   = "data/slices/top_down.png"
MASKS_DIR    = "data/masks"


# ── AMG hyperparameters tuned for civil infrastructure ────────────────────────
# Infrastructure features are large and low-frequency → fewer, higher-quality masks
AMG_KWARGS: dict[str, Any] = dict(
    points_per_side=32,          # 32×32 = 1024 prompt points across image
    points_per_batch=64,         # reduce if MPS runs out of memory
    pred_iou_thresh=0.82,        # keep masks with predicted IoU > 0.82
    stability_score_thresh=0.90, # keep geometrically stable masks
    stability_score_offset=1.0,
    box_nms_thresh=0.7,
    crop_n_layers=1,             # one extra crop pass for fine details
    crop_overlap_ratio=0.34,
    min_mask_region_area=500,    # drop masks < 500 px² (noise / tiny artefacts)
    output_mode="binary_mask",
    multimask_output=True,
)


def load_image(path: str) -> np.ndarray:
    """Load image as uint8 RGB numpy array."""
    img = Image.open(path).convert("RGB")
    return np.array(img)


def build_generator(device: str = DEVICE) -> SAM2AutomaticMaskGenerator:
    """Instantiate SAM2 and wrap in the automatic mask generator."""
    print(f"Loading SAM2 on device: {device}")
    sam2_model = build_sam2(
        config_file=SAM2_CONFIG,
        ckpt_path=SAM2_CKPT,
        device=device,
        mode="eval",
        apply_postprocessing=True,
    )
    return SAM2AutomaticMaskGenerator(sam2_model, **AMG_KWARGS)


def run_segmentation(
    image_path: str = SLICE_PATH,
    out_dir: str = MASKS_DIR,
    device: str = DEVICE,
) -> tuple[np.ndarray, list[dict]]:
    """
    Run SAM2 automatic mask generation on the 2D slice image.

    Returns
    -------
    masks_arr : np.ndarray, shape (M, H, W), dtype bool
        One boolean mask per detected segment.
    mask_meta : list[dict]
        SAM2 metadata per mask: keys include
            'segmentation'    (H,W) bool
            'area'            pixel count
            'bbox'            [x, y, w, h]
            'predicted_iou'   float
            'stability_score' float
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load image ────────────────────────────────────────────────────────────
    print(f"Loading image: {image_path}")
    image = load_image(image_path)
    H, W = image.shape[:2]
    print(f"  Image size: {H} × {W} px")

    # ── Build model & generate masks ─────────────────────────────────────────
    generator = build_generator(device=device)

    print("Running SAM2 automatic mask generation …")
    with torch.inference_mode():
        mask_meta: list[dict] = generator.generate(image)

    print(f"  Generated {len(mask_meta)} raw masks")

    # ── Sort by area (largest first) ─────────────────────────────────────────
    mask_meta.sort(key=lambda m: m["area"], reverse=True)

    # ── Stack segmentation arrays ─────────────────────────────────────────────
    masks_arr = np.stack(
        [m["segmentation"] for m in mask_meta], axis=0
    ).astype(bool)   # (M, H, W)

    print(f"  Kept {masks_arr.shape[0]} masks  |  "
          f"avg area {int(masks_arr.sum(axis=(1,2)).mean())} px²")

    # ── Save ──────────────────────────────────────────────────────────────────
    masks_path = out_dir / "masks.npy"
    np.save(masks_path, masks_arr)
    print(f"Saved masks → {masks_path}  shape={masks_arr.shape}")

    # Save metadata (strip the large 'segmentation' array to keep file small)
    meta_lite = [
        {k: v for k, v in m.items() if k != "segmentation"}
        for m in mask_meta
    ]
    meta_path = out_dir / "mask_meta.npy"
    np.save(meta_path, meta_lite, allow_pickle=True)
    print(f"Saved mask metadata → {meta_path}")

    # ── Visualise ─────────────────────────────────────────────────────────────
    overlay_path = out_dir / "overlay.png"
    _save_overlay(image, masks_arr, str(overlay_path))
    print(f"Saved overlay → {overlay_path}")

    return masks_arr, mask_meta


def _save_overlay(
    image: np.ndarray,
    masks: np.ndarray,
    out_path: str,
    max_masks: int = 200,
) -> None:
    """
    Render random-coloured mask overlays on top of the source image and save.
    Limits to `max_masks` largest masks to keep the plot readable.
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.imshow(image)

    rng = np.random.default_rng(42)
    n = min(len(masks), max_masks)

    for mask in masks[:n]:
        colour = rng.random(3)
        rgba = np.zeros((*mask.shape, 4), dtype=float)
        rgba[mask] = [*colour, 0.45]   # semi-transparent fill
        ax.imshow(rgba)

    # Minimal annotation
    ax.set_title(f"SAM2 Masks  (showing {n} of {len(masks)})", fontsize=10)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    masks_arr, mask_meta = run_segmentation(
        image_path=SLICE_PATH,
        out_dir=MASKS_DIR,
        device=DEVICE,
    )

    print("\nDone!")
    print(f"  masks.npy shape : {masks_arr.shape}")
    print(f"  mask areas (top-5): "
          + str([m["area"] for m in mask_meta[:5]]))
