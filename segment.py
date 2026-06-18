"""
Step 2: Grounded SAM segmentation on the 2D orthographic projection

Uses:
  - Grounding DINO  (via HuggingFace transformers) → text prompt → bounding boxes
  - SAM             (via HuggingFace transformers) → bounding boxes → pixel masks

Inputs:
  projection_intensity.png   (preferred) or projection_depth.png

Outputs:
  segmentation_overlay.png   — color-coded mask overlaid on the projection image
  segmentation_mask.npy      — (H, W) int array, each pixel = class index (0 = background)
  segmentation_classes.txt   — maps class index → class name

Install (once):
  pip install transformers torch torchvision Pillow numpy
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from transformers import SamModel, SamProcessor

# ── Config ────────────────────────────────────────────────────────────────────

INPUT_IMAGE   = "projection_intensity.png"   # swap to projection_depth.png if needed
BOX_THRESHOLD = 0.15   # Grounding DINO confidence threshold for boxes
TEXT_THRESHOLD = 0.25  # Grounding DINO text confidence threshold

# Classes to detect — separate with " . ", trailing "." required by Grounding DINO
TEXT_PROMPT = "building . road . tree . parking lot . pavement ."

# Color per class (BGR-style but we use RGB here): index matches detected label order
CLASS_COLORS = {
    "building":    (255, 100, 100),   # red
    "road":        (100, 100, 255),   # blue
    "tree":        (100, 200, 100),   # green
    "parking lot": (255, 200,  50),   # yellow
    "pavement":    (200, 150, 255),   # purple
}
DEFAULT_COLOR = (180, 180, 180)       # grey for unrecognised labels

# ── Device ────────────────────────────────────────────────────────────────────

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple MPS (Metal)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("Using CPU (will be slow)")

# ── Load image ────────────────────────────────────────────────────────────────

def load_image(input_image):
    print(f"Loading image: {input_image}")
    from PIL import ImageOps
    image = ImageOps.autocontrast(Image.open(input_image).convert("RGB"))
    W, H = image.size
    print(f"  Image size: {W}x{H}")
    return image, W, H

# ── Step 1: Grounding DINO — text prompt → bounding boxes ─────────────────────

def run_gdino(image, box_threshold, text_threshold):
    print("\nLoading Grounding DINO...")
    gdino_model_id = "IDEA-Research/grounding-dino-tiny"
    gdino_processor = AutoProcessor.from_pretrained(gdino_model_id)
    gdino_model     = AutoModelForZeroShotObjectDetection.from_pretrained(gdino_model_id).to(device)

    print(f"Running Grounding DINO with prompt: '{TEXT_PROMPT}'")
    inputs = gdino_processor(images=image, text=TEXT_PROMPT, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = gdino_model(**inputs)

    # post_process_grounded_object_detection API varies by transformers version.
    # Try with threshold args first; fall back to manual filtering if unsupported.
    try:
        results = gdino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]
    except TypeError:
        results = gdino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            target_sizes=[image.size[::-1]],
        )[0]

    boxes  = results["boxes"].cpu().numpy()    # (N, 4) in xyxy format
    scores = results["scores"].cpu().numpy()   # (N,)
    # transformers >= 4.51 renamed "labels" (int ids) to "text_labels" (strings)
    labels = results.get("text_labels", results.get("labels", []))

    # Manual threshold filtering (handles both old and new API)
    keep   = scores >= box_threshold
    boxes, scores, labels = boxes[keep], scores[keep], [l for l, k in zip(labels, keep) if k]

    print(f"  Detected {len(boxes)} regions:")
    for label, score, box in zip(labels, scores, boxes):
        print(f"    [{score:.2f}] {label:15s}  box={box.astype(int).tolist()}")

    if len(boxes) == 0:
        print("\nNo regions detected. Try lowering BOX_THRESHOLD or changing TEXT_PROMPT.")
        exit(1)

    return boxes, scores, labels

# ── Step 2: SAM — bounding boxes → pixel masks ────────────────────────────────

def run_sam(image, boxes):
    print("\nLoading SAM...")
    sam_model_id = "facebook/sam-vit-base"
    sam_processor = SamProcessor.from_pretrained(sam_model_id)
    sam_model     = SamModel.from_pretrained(sam_model_id).to(device)

    print("Running SAM on detected boxes...")

    # SAM expects boxes as [[x1,y1,x2,y2]] per image, nested in a list
    input_boxes = [boxes.tolist()]

    sam_inputs = sam_processor(
        images=image,
        input_boxes=input_boxes,
        return_tensors="pt"
    )
    # MPS doesn't support float64 — cast all floating tensors to float32 before moving to device
    sam_inputs = {
        k: v.to(torch.float32) if isinstance(v, torch.Tensor) and v.dtype == torch.float64 else v
        for k, v in sam_inputs.items()
    }
    sam_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sam_inputs.items()}

    with torch.no_grad():
        sam_outputs = sam_model(**{k: v for k, v in sam_inputs.items()
                                   if k in ("pixel_values", "input_boxes",
                                            "input_points", "input_labels")})

    original_sizes     = sam_inputs["original_sizes"].cpu()
    reshaped_sizes     = sam_inputs["reshaped_input_sizes"].cpu()

    masks = sam_processor.post_process_masks(
        sam_outputs.pred_masks.cpu(),
        original_sizes,
        reshaped_sizes,
    )[0]  # shape: (N, 3, H, W) — SAM returns 3 mask candidates per box

    # Pick the best mask candidate (highest IoU score) for each box
    iou_scores = sam_outputs.iou_scores[0].cpu().numpy()   # (N, 3)
    best_masks = []
    for i in range(len(boxes)):
        best_idx = iou_scores[i].argmax()
        best_masks.append(masks[i][best_idx].numpy())       # (H, W) bool

    return best_masks

# ── Step 3: Build segmentation map & overlay ──────────────────────────────────

def build_segmentation_map(image, boxes, best_masks):
    print("\nBuilding segmentation map...")

    seg_map = np.zeros((image.size[1], image.size[0]), dtype=np.int32)   # 0 = background
    overlay = image.copy().convert("RGBA")
    class_index_map = {}   # label → class index

    for idx, (label, mask) in enumerate(zip(labels, best_masks), start=1):
        # Normalise label to match CLASS_COLORS keys
        label_clean = label.strip().lower()
        class_index_map[idx] = label_clean

        # Write class index into seg_map (later detections overwrite earlier ones)
        seg_map[mask] = idx

        # Draw semi-transparent colored mask on overlay
        color = CLASS_COLORS.get(label_clean, DEFAULT_COLOR) + (120,)   # RGBA with alpha
        mask_img = Image.fromarray((mask * 255).astype(np.uint8)).convert("L")
        color_layer = Image.new("RGBA", image.size, color)
        overlay.paste(color_layer, mask=mask_img)

    # Draw bounding boxes + labels on top
    draw = ImageDraw.Draw(overlay)
    for label, score, box in zip(labels, scores, boxes):
        x1, y1, x2, y2 = box.astype(int)
        color = CLASS_COLORS.get(label.strip().lower(), DEFAULT_COLOR) + (255,)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1 + 4, y1 + 4), f"{label} {score:.2f}", fill=color)

    return seg_map, overlay

# ── Step 4: Save outputs ──────────────────────────────────────────────────────

def save_outputs(seg_map, overlay):
    overlay_rgb = overlay.convert("RGB")
    overlay_rgb.save("segmentation_overlay.png")
    print("  → segmentation_overlay.png")

    np.save("segmentation_mask.npy", seg_map)
    print("  → segmentation_mask.npy")

    with open("segmentation_classes.txt", "w") as f:
        f.write("0 background\n")
        for idx, label in class_index_map.items():
            f.write(f"{idx} {label}\n")
    print("  → segmentation_classes.txt")

    print("\nDone.")
    print(f"  Classes detected: {list(class_index_map.values())}")
    print(f"  Pixels classified (non-background): {(seg_map > 0).sum():,} / {H * W:,}")
    print(f"  Coverage: {(seg_map > 0).sum() / (H * W) * 100:.1f}%")

if __name__ == "__main__":
    image, W, H = load_image(INPUT_IMAGE)
    boxes, scores, labels = run_gdino(image, BOX_THRESHOLD, TEXT_THRESHOLD)
    best_masks = run_sam(image, boxes)
    seg_map, overlay = build_segmentation_map(image, boxes, best_masks)
    save_outputs(seg_map, overlay)
