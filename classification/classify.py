"""
classify.py — Per-mask class labelling using a VGG19 feature extractor.

Pipeline position:  segment.py → [classify.py] → map_back.py

Target classes (civil infrastructure):
    0  road_surface   — flat paved surface at ground level
    1  bridge_deck    — flat paved surface elevated above ground
    2  guardrail      — narrow elongated structure at deck/road edge
    3  vegetation     — irregular height variation, rough texture
    4  vehicle        — small elevated blob (parked / moving)

Modes
-----
1. Inference without trained weights  →  elevation-statistics heuristic
   Fast, zero-shot, reasonable for bridge/road separation
2. Fine-tune  →  call `train()` with a small hand-labelled mask dataset
3. Full inference with weights        →  call `predict()` after `load_weights()`

Inputs:
    data/slices/top_down.png          false-colour RGB image (from slice.py)
    data/slices/elevation_grid.npy    raw elevation values (from slice.py)
    data/masks/masks.npy              (M, H, W) bool masks (from segment.py)
    data/masks/mask_meta.npy          mask metadata list  (from segment.py)
    models/classifier/vgg19_head.pt   [optional] trained classification head

Outputs:
    data/masks/class_labels.npy       (M,) int32 class index per mask
    data/masks/label_grid.npy         (H, W) int32 class per pixel (−1 = background)
    data/masks/labelled_overlay.png   colour-coded class visualisation
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

# ── Label grid helper ─────────────────────────────────────────────────────────

def masks_to_label_grid(masks: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    Convert (M, H, W) mask stack to a (H, W) integer label grid.
    Pixel label = index of the first mask (largest area) that covers it.
    Unlabelled pixels get -1.

    Parameters
    ----------
    masks : (M, H, W) bool array, sorted largest→smallest
    H, W  : image dimensions (for consistency check)

    Returns
    -------
    label_grid : (H, W) int32, values in [-1, M-1]
    """
    assert masks.shape[1:] == (H, W), "mask/image size mismatch"
    label_grid = np.full((H, W), -1, dtype=np.int32)
    for i, mask in enumerate(masks):
        # Only assign if not yet claimed (largest mask wins)
        unlabelled = label_grid == -1
        label_grid[mask & unlabelled] = i
    return label_grid


# ── Class registry ────────────────────────────────────────────────────────────

CLASS_NAMES = ["road_surface", "bridge_deck", "guardrail", "vegetation", "vehicle"]
NUM_CLASSES = len(CLASS_NAMES)

# Colour map for visualisation (RGB, 0-1 floats)
CLASS_COLOURS = {
    0: (0.55, 0.55, 0.55),   # road_surface  — grey
    1: (0.20, 0.60, 0.90),   # bridge_deck   — steel blue
    2: (0.95, 0.65, 0.10),   # guardrail     — amber
    3: (0.20, 0.75, 0.25),   # vegetation    — green
    4: (0.90, 0.20, 0.20),   # vehicle       — red
   -1: (0.10, 0.10, 0.10),   # background    — near-black
}

# ── Paths ─────────────────────────────────────────────────────────────────────

SLICE_PATH   = "data/slices/top_down.png"
GRID_PATH    = "data/slices/elevation_grid.npy"
MASKS_PATH   = "data/masks/masks.npy"
META_PATH    = "data/masks/mask_meta.npy"
WEIGHTS_PATH = "models/classifier/vgg19_head.pt"
OUT_DIR      = "data/masks"

DEVICE       = "mps"


# ── VGG19 feature extractor + classification head ─────────────────────────────

class MaskClassifier(nn.Module):
    """
    VGG19 image encoder  +  small elevation MLP  →  5-class head.

    image_feat  : (B, 4096)  — VGG19 fc7 features from the mask crop
    elev_feat   : (B, 8)     — hand-crafted elevation statistics
    → logits    : (B, 5)
    """

    IMG_FEAT_DIM  = 4096
    ELEV_FEAT_DIM = 8

    def __init__(self, num_classes: int = NUM_CLASSES, freeze_backbone: bool = True):
        super().__init__()

        # ── VGG19 backbone (pretrained on ImageNet) ──────────────────────────
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        # Features up to classifier[3] = ReLU after fc7 (4096-d)
        self.backbone = nn.Sequential(
            vgg.features,
            vgg.avgpool,
            nn.Flatten(),
            *list(vgg.classifier.children())[:4],   # fc6, relu, dropout, fc7
        )
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # ── Elevation statistics MLP ─────────────────────────────────────────
        self.elev_mlp = nn.Sequential(
            nn.Linear(self.ELEV_FEAT_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        # ── Classification head ───────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(self.IMG_FEAT_DIM + 16, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(
        self,
        img_crop: torch.Tensor,      # (B, 3, 224, 224)
        elev_feat: torch.Tensor,     # (B, 8)
    ) -> torch.Tensor:
        img_f  = self.backbone(img_crop)            # (B, 4096)
        elev_f = self.elev_mlp(elev_feat)           # (B, 16)
        return self.head(torch.cat([img_f, elev_f], dim=1))  # (B, 5)


# ── Image pre-processing ──────────────────────────────────────────────────────

_VGG_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _crop_mask_region(
    image: np.ndarray,
    mask: np.ndarray,
    bbox: list[int],
    pad: int = 8,
) -> np.ndarray:
    """
    Crop the bounding-box region of a mask from the image.
    Pixels outside the mask are zeroed.  Returns uint8 H×W×3.
    """
    H, W = image.shape[:2]
    x, y, w, h = bbox
    x0 = max(0, x - pad);  y0 = max(0, y - pad)
    x1 = min(W, x + w + pad);  y1 = min(H, y + h + pad)

    crop = image[y0:y1, x0:x1].copy()

    # Zero-out pixels outside the mask within the crop
    mask_crop = mask[y0:y1, x0:x1]
    crop[~mask_crop] = 0

    return crop


def _elevation_features(
    elev_grid: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Extract 8 elevation statistics for one mask region.

    Features:
        0  mean elevation (normalised by global stats)
        1  std of elevation  (roughness proxy)
        2  range (max − min)
        3  fraction of mask in top-10% global elevation quantile
        4  fraction of mask in bottom-10% global elevation quantile
        5  relative elevation (mean − global mean) / global std
        6  aspect ratio of bounding box (width / height)
        7  log10(mask area in pixels)
    """
    g_mean = np.nanmean(elev_grid)
    g_std  = np.nanstd(elev_grid) + 1e-8
    q10    = np.nanpercentile(elev_grid, 10)
    q90    = np.nanpercentile(elev_grid, 90)

    z = elev_grid[mask]
    z = z[~np.isnan(z)]

    if len(z) == 0:
        return np.zeros(8, dtype=np.float32)

    # Bounding box for aspect ratio
    rows, cols = np.where(mask)
    h_bb = max(1, rows.max() - rows.min())
    w_bb = max(1, cols.max() - cols.min())

    feats = np.array([
        (z.mean() - g_mean) / g_std,          # 0  normalised mean elev
        z.std() / g_std,                       # 1  normalised roughness
        (z.max() - z.min()) / g_std,           # 2  normalised range
        (z > q90).mean(),                      # 3  high-elev fraction
        (z < q10).mean(),                      # 4  low-elev fraction
        (z.mean() - g_mean) / g_std,           # 5  relative elev (= feat 0, kept for MLP symmetry)
        w_bb / h_bb,                           # 6  aspect ratio
        np.log10(max(1, len(z))),              # 7  log mask area
    ], dtype=np.float32)

    return feats


# ── Zero-shot heuristic fallback ──────────────────────────────────────────────

def _heuristic_classify(
    elev_feat: np.ndarray,        # (8,)
    mask: np.ndarray,             # (H, W) bool
) -> int:
    """
    Rule-based class assignment from elevation features.
    Used when no trained weights are available.

    Rules (approximate for civil infrastructure):
      - Large area + flat (low std) + very high elevation  → bridge_deck (1)
      - Large area + flat (low std) + mid elevation        → road_surface (0)
      - Narrow (high aspect ratio) + edge elevation        → guardrail (2)
      - High roughness (high std)                         → vegetation (3)
      - Small area + slightly elevated blob               → vehicle (4)
    """
    norm_mean = elev_feat[0]    # relative elevation
    roughness  = elev_feat[1]   # std / global_std
    elev_range = elev_feat[2]
    high_frac  = elev_feat[3]   # fraction above 90th percentile
    low_frac   = elev_feat[4]
    aspect     = elev_feat[6]   # w/h
    log_area   = elev_feat[7]   # log10(px)

    area_px = 10 ** log_area

    # Vegetation: rough texture regardless of elevation
    if roughness > 0.8 or elev_range > 1.5:
        return 3  # vegetation

    # Guardrail: narrow elongated + at-elevation
    if (aspect > 4.0 or aspect < 0.25) and area_px < 5000:
        return 2  # guardrail

    # Vehicle: small compact blob, slightly elevated
    if area_px < 2000 and roughness < 0.5:
        return 4  # vehicle

    # Bridge deck vs road surface: elevation relative to scene mean
    if norm_mean > 0.5 or high_frac > 0.5:
        return 1  # bridge_deck (above average elevation)
    else:
        return 0  # road_surface


# ── Main API ──────────────────────────────────────────────────────────────────

def predict(
    image_path: str       = SLICE_PATH,
    grid_path:  str       = GRID_PATH,
    masks_path: str       = MASKS_PATH,
    meta_path:  str       = META_PATH,
    weights_path: str     = WEIGHTS_PATH,
    out_dir:    str       = OUT_DIR,
    device:     str       = DEVICE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Classify each SAM2 mask and produce a per-pixel label grid.

    Returns
    -------
    class_labels : (M,) int32  — class index per mask
    label_grid   : (H, W) int32 — class per pixel (−1 = background)
    """
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    # ── Load inputs ──────────────────────────────────────────────────────────
    image     = np.array(Image.open(image_path).convert("RGB"))
    elev_grid = np.load(grid_path)
    masks     = np.load(masks_path)          # (M, H, W) bool
    meta      = np.load(meta_path, allow_pickle=True).tolist()

    H, W   = image.shape[:2]
    M      = len(masks)
    print(f"Classifying {M} masks …  image={H}×{W}")

    # ── Determine mode ───────────────────────────────────────────────────────
    use_model = Path(weights_path).exists()

    if use_model:
        print(f"  Mode: VGG19 inference  (weights: {weights_path})")
        model = MaskClassifier(num_classes=NUM_CLASSES, freeze_backbone=True)
        ckpt  = torch.load(weights_path, map_location=device)
        model.load_state_dict(ckpt)
        model.to(device).eval()
    else:
        warnings.warn(
            f"No trained weights at {weights_path}. "
            "Using elevation-heuristic classifier. "
            "Run classify.train() with labelled examples to train the model.",
            UserWarning,
        )
        model = None

    # ── Build NaN mask (pixels with no LiDAR data) ───────────────────────────
    # Used to skip masks that are mostly outside the scan boundary.
    # We load the raw grid via grid_path (elevation_grid.npy saves the filled
    # grid, so reconstruct the NaN mask from where filling set values to g_min).
    # Simpler: detect originally-empty cells via the filled grid: cells that were
    # NaN become g_min after filling.  We recompute from the raw LiDAR grid if
    # it has a twin saved by slice.py; otherwise fall back to the filled grid.
    raw_grid_path = Path(grid_path).parent / "elevation_grid_raw.npy"
    if raw_grid_path.exists():
        no_data_mask = np.isnan(np.load(str(raw_grid_path)))
    else:
        # Approximate: pixels in the bottom 0.1% of the filled elevation are
        # likely NaN-fill artefacts at the scan boundary.
        filled = np.load(grid_path)
        threshold = np.nanpercentile(filled, 0.1)
        no_data_mask = filled <= threshold

    # ── Classify each mask ───────────────────────────────────────────────────
    class_labels = np.full(M, -1, dtype=np.int32)
    skipped = 0

    for i, (mask, m) in enumerate(zip(masks, meta)):
        # Skip masks where >50% of pixels fall outside the LiDAR scan boundary.
        # These are the large "green border" masks that SAM2 picks up as one
        # segment; labelling them inflates a single class to 99%+ of all points.
        no_data_frac = no_data_mask[mask].mean()
        if no_data_frac > 0.50:
            class_labels[i] = -1   # mark as background / skip
            skipped += 1
            continue

        bbox      = m["bbox"]      # [x, y, w, h]
        elev_feat = _elevation_features(elev_grid, mask)

        if model is not None:
            crop = _crop_mask_region(image, mask, bbox)
            pil_crop = Image.fromarray(crop)
            img_t  = _VGG_TRANSFORM(pil_crop).unsqueeze(0).to(device)
            elev_t = torch.tensor(elev_feat).unsqueeze(0).to(device)
            with torch.inference_mode():
                logits = model(img_t, elev_t)
            class_labels[i] = int(logits.argmax(dim=1).item())
        else:
            class_labels[i] = _heuristic_classify(elev_feat, mask)

    if skipped:
        print(f"  Skipped {skipped} out-of-scan-boundary masks")

    # ── Build per-pixel label grid ───────────────────────────────────────────
    # Paint SMALLEST masks first so specific features (buildings, trees, vehicles)
    # claim their pixels before the large background/ground mask fills the rest.
    # (masks[] is sorted largest→smallest by segment.py, so reverse for painting)
    label_grid = np.full((H, W), -1, dtype=np.int32)
    for i in range(len(masks) - 1, -1, -1):      # smallest → largest
        if class_labels[i] < 0:
            continue                               # skip background masks
        unlabelled = label_grid == -1
        label_grid[masks[i] & unlabelled] = class_labels[i]

    # ── Save ─────────────────────────────────────────────────────────────────
    np.save(out_dir_p / "class_labels.npy", class_labels)
    np.save(out_dir_p / "label_grid.npy",   label_grid)
    print(f"Saved class_labels → {out_dir_p / 'class_labels.npy'}")
    print(f"Saved label_grid   → {out_dir_p / 'label_grid.npy'}")

    # ── Distribution summary ─────────────────────────────────────────────────
    valid = class_labels[class_labels >= 0]
    print("\nClass distribution (over scan-boundary masks only):")
    for cls_id, name in enumerate(CLASS_NAMES):
        count = int((valid == cls_id).sum())
        pct   = 100 * count / len(valid) if len(valid) > 0 else 0
        print(f"  {cls_id}  {name:<14s}  {count:>4d} masks ({pct:.1f}%)")

    # Per-pixel distribution in label_grid
    total_px = H * W
    print("\nPixel distribution in label grid:")
    for cls_id, name in enumerate(CLASS_NAMES):
        px  = int((label_grid == cls_id).sum())
        pct = 100 * px / total_px
        print(f"  {cls_id}  {name:<14s}  {px:>10,} px  ({pct:.1f}%)")
    bg_px = int((label_grid == -1).sum())
    print(f" -1  background     {bg_px:>10,} px  ({100*bg_px/total_px:.1f}%)")

    # ── Visualise ─────────────────────────────────────────────────────────────
    _save_labelled_overlay(image, masks, class_labels, str(out_dir_p / "labelled_overlay.png"))
    print(f"Saved labelled_overlay → {out_dir_p / 'labelled_overlay.png'}")

    return class_labels, label_grid


def train(
    image_path:  str,
    grid_path:   str,
    masks_path:  str,
    meta_path:   str,
    labels:      np.ndarray,      # (M,) int32  — ground-truth class per mask
    weights_path: str = WEIGHTS_PATH,
    device:      str = DEVICE,
    epochs:      int = 30,
    lr:          float = 1e-3,
) -> None:
    """
    Fine-tune the VGG19 classification head on a hand-labelled set of masks.

    Parameters
    ----------
    labels : (M,) int32
        Ground-truth class index for each mask (−1 = skip / unlabelled).
    """
    Path(weights_path).parent.mkdir(parents=True, exist_ok=True)

    image     = np.array(Image.open(image_path).convert("RGB"))
    elev_grid = np.load(grid_path)
    masks     = np.load(masks_path)
    meta      = np.load(meta_path, allow_pickle=True).tolist()

    # Collect labelled samples
    img_crops, elev_feats, targets = [], [], []
    for i, (mask, m) in enumerate(zip(masks, meta)):
        if labels[i] < 0:
            continue
        crop = _crop_mask_region(image, mask, m["bbox"])
        img_crops.append(_VGG_TRANSFORM(Image.fromarray(crop)))
        elev_feats.append(torch.tensor(_elevation_features(elev_grid, mask)))
        targets.append(int(labels[i]))

    if len(targets) == 0:
        raise ValueError("No labelled masks to train on (all labels are −1).")

    imgs_t   = torch.stack(img_crops).to(device)
    elevs_t  = torch.stack(elev_feats).to(device)
    targets_t = torch.tensor(targets, dtype=torch.long).to(device)

    model = MaskClassifier(num_classes=NUM_CLASSES, freeze_backbone=True)
    model.to(device).train()

    optimiser = torch.optim.Adam(
        list(model.elev_mlp.parameters()) + list(model.head.parameters()), lr=lr
    )
    criterion = nn.CrossEntropyLoss()

    print(f"Training on {len(targets)} labelled masks for {epochs} epochs …")
    for epoch in range(1, epochs + 1):
        optimiser.zero_grad()
        logits = model(imgs_t, elevs_t)
        loss   = criterion(logits, targets_t)
        loss.backward()
        optimiser.step()
        if epoch % 5 == 0 or epoch == 1:
            acc = (logits.argmax(1) == targets_t).float().mean().item()
            print(f"  epoch {epoch:3d}/{epochs}  loss={loss.item():.4f}  acc={acc:.2%}")

    torch.save(model.state_dict(), weights_path)
    print(f"Saved model weights → {weights_path}")


# ── Visualisation ─────────────────────────────────────────────────────────────

def _save_labelled_overlay(
    image: np.ndarray,
    masks: np.ndarray,
    class_labels: np.ndarray,
    out_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(image)

    for mask, cls_id in zip(masks, class_labels):
        colour = CLASS_COLOURS.get(int(cls_id), (0.5, 0.5, 0.5))
        rgba   = np.zeros((*mask.shape, 4), dtype=float)
        rgba[mask] = [*colour, 0.45]
        ax.imshow(rgba)

    # Legend
    patches = [
        mpatches.Patch(color=CLASS_COLOURS[i], label=f"{i} {name}")
        for i, name in enumerate(CLASS_NAMES)
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=8,
              framealpha=0.8, title="Classes")
    ax.set_title("Per-mask classification (VGG19 / heuristic)", fontsize=10)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    class_labels, label_grid = predict()
    print(f"\nDone!  label_grid shape: {label_grid.shape}")
    print(f"  Labelled pixels: {(label_grid >= 0).sum():,}")
    print(f"  Background pixels: {(label_grid == -1).sum():,}")
