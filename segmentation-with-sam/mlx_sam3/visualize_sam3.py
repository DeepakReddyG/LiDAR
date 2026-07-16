import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

IMAGE_PATH = "IMG_6.png"
PROMPTS    = ["car", "tree", "building", "parking lot", "road"]
THRESHOLD  = 0.5   # lower threshold — expect weaker confidence on satellite imagery

def main():
    print("Loading model…")
    model     = build_sam3_image_model()
    processor = Sam3Processor(model, confidence_threshold=THRESHOLD)
    image     = Image.open(IMAGE_PATH).convert("RGB")

    n    = len(PROMPTS)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows))
    axes = axes.flatten()

    palette = [
        (1.0, 0.2, 0.2),
        (1.0, 0.85, 0.0),
        (0.2, 0.9, 0.2),
        (0.2, 0.5, 1.0),
        (0.9, 0.3, 1.0),
        (0.0, 0.9, 0.9),
    ]

    for i, prompt in enumerate(PROMPTS):
        color = palette[i % len(palette)]
        ax    = axes[i]

        state = processor.set_image(image)
        state = processor.set_text_prompt(prompt, state)

        masks  = state["masks"]
        boxes  = state["boxes"]
        scores = state["scores"]

        print(f"[{prompt:>15s}]  found={len(scores)}  "
              f"scores={[round(float(s), 3) for s in scores]}")

        ax.imshow(image)
        ax.set_title(f'"{prompt}"  ({len(scores)} obj)', fontsize=10)
        ax.axis("off")

        for j, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
            # mask overlay
            mask_np = np.squeeze(np.array(mask))
            if mask_np.ndim != 2:
                continue
            colored = np.zeros((*mask_np.shape, 4), dtype=np.float32)
            colored[mask_np > 0] = [*color, 0.5]
            ax.imshow(colored)

            # bounding box
            box_np = np.array(box).flatten()
            if box_np.shape[0] < 4:
                continue
            x0, y0, x1, y1 = float(box_np[0]), float(box_np[1]), \
                              float(box_np[2]), float(box_np[3])
            rect = patches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                linewidth=1.5, edgecolor=color, facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(
                x0, max(y0 - 4, 0),
                f"{score:.2f}",
                color="white", fontsize=8, fontweight="bold",
                bbox=dict(facecolor=color, alpha=0.8, pad=1.5, edgecolor="none")
            )

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"SAM3 · aerial intersection · threshold={THRESHOLD}", fontsize=13)
    plt.tight_layout()
    out = "sam3_Temple_Texas_Colorised_result.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out}")

if __name__ == "__main__":
    main()