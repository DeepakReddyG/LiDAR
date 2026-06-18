"""
Step 1: Load LiDAR chunk → downsample → orthographic top-down projection → save PNG

Outputs:
  projection_depth.png     — grayscale image where brightness = height (Z)
  projection_intensity.png — grayscale image where brightness = intensity (if available)
  pixel_to_point_index.npy — 2D array mapping each pixel to its point index (for backprojection)
"""

import laspy
import numpy as np
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────

LAS_FILE       = "UPark_Merged_PS_NAD83_G18_USFT_las.las"
CHUNK_SIZE     = 77_500_000   # points to load (adjust down if RAM is tight)
VOXEL_SIZE     = 0.5         # metres — downsampling resolution (larger = fewer points)
IMAGE_RESOLUTION = 2048      # output image width AND height in pixels

# ── 1. Load one chunk ─────────────────────────────────────────────────────────

def load_chunk(las_file, chunk_size):
    print("Opening LAS file...")
    with laspy.open(las_file) as f:
        print(f"Total points in file: {f.header.point_count:,}")

        for chunk in f.chunk_iterator(chunk_size):
            x = np.array(chunk.x, dtype=np.float64)
            y = np.array(chunk.y, dtype=np.float64)
            z = np.array(chunk.z, dtype=np.float64)

            # Intensity is almost always present in LiDAR data
            try:
                intensity = np.array(chunk.intensity, dtype=np.float32)
            except AttributeError:
                intensity = None
                print("No intensity channel found — skipping intensity projection.")

            break  # only first chunk

    print(f"Loaded chunk: {x.shape[0]:,} points")

    return x, y, z, intensity

# ── 2. Voxel downsample ───────────────────────────────────────────────────────
# Bin points into a voxel grid and keep one representative per voxel.

def downsample(x, y, z, intensity, voxel_size):
    print(f"Downsampling with voxel size {voxel_size}m ...")

    vx = np.floor(x / voxel_size).astype(np.int32)
    vy = np.floor(y / voxel_size).astype(np.int32)
    vz = np.floor(z / voxel_size).astype(np.int32)

    # Use a dict to keep only the first point per voxel
    voxel_keys = {}
    for i in range(len(x)):
        key = (vx[i], vy[i], vz[i])
        if key not in voxel_keys:
            voxel_keys[key] = i

    kept = np.array(list(voxel_keys.values()), dtype=np.int64)
    x, y, z = x[kept], y[kept], z[kept]
    if intensity is not None:
        intensity = intensity[kept]

    print(f"After downsampling: {len(x):,} points")

    return x, y, z, intensity

# ── 3. Orthographic top-down projection ───────────────────────────────────────
# Map (X, Y) → pixel (u, v).  Z → pixel brightness for depth image.
# Also store pixel_map[v, u] = original point index for backprojection.

def project(x, y, z, intensity, image_resolution):
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    z_min, z_max = z.min(), z.max()

    print(f"X range: {x_min:.1f} – {x_max:.1f}  ({x_max - x_min:.1f}m)")
    print(f"Y range: {y_min:.1f} – {y_max:.1f}  ({y_max - y_min:.1f}m)")
    print(f"Z range: {z_min:.1f} – {z_max:.1f}  ({z_max - z_min:.1f}m)")

    W = H = image_resolution

    # Normalise X,Y to [0, W-1] / [0, H-1]
    u = ((x - x_min) / (x_max - x_min) * (W - 1)).astype(np.int32)
    v = ((y - y_min) / (y_max - y_min) * (H - 1)).astype(np.int32)

    # Normalise Z to [0, 255]
    z_norm = ((z - z_min) / (z_max - z_min) * 255).astype(np.uint8)

    # pixel_map: stores index of the highest (max Z) point at each pixel
    # -1 means no point mapped there
    pixel_map   = np.full((H, W), -1, dtype=np.int64)
    depth_img   = np.zeros((H, W), dtype=np.uint8)

    # Sort by Z ascending so the highest point wins (last write)
    order = np.argsort(z)
    u, v, z_norm = u[order], v[order], z_norm[order]
    original_indices = kept[order]   # map back to pre-downsample indices

    for i in range(len(u)):
        depth_img[v[i], u[i]]  = z_norm[i]
        pixel_map[v[i], u[i]]  = original_indices[i]

    if intensity is not None:
        intensity_sorted = intensity[order]
        i_min, i_max = intensity_sorted.min(), intensity_sorted.max()
        if i_max > i_min:
            int_norm = ((intensity_sorted - i_min) / (i_max - i_min) * 255).astype(np.uint8)
        else:
            int_norm = np.zeros_like(intensity_sorted, dtype=np.uint8)

        int_img = np.zeros((H, W), dtype=np.uint8)
        for i in range(len(u)):
            int_img[v[i], u[i]] = int_norm[i]

    return depth_img, int_img, pixel_map

# ── 4. Save outputs ───────────────────────────────────────────────────────────

def save_depth_projection(depth_img):
    print("Saving depth projection...")
    Image.fromarray(depth_img).save("projection_depth.png")
    print("  → projection_depth.png")

def save_intensity_projection(int_img):
    if int_img is not None:
        print("Saving intensity projection...")
        Image.fromarray(int_img).save("projection_intensity.png")
        print("  → projection_intensity.png")

def save_pixel_to_point_index(pixel_map):
    np.save("pixel_to_point_index.npy", pixel_map)
    print("  → pixel_to_point_index.npy  (pixel→point index map for backprojection)")

if __name__ == "__main__":
    x, y, z, intensity = load_chunk(LAS_FILE, CHUNK_SIZE)
    depth_img, int_img, pixel_map = downsample(x, y, z, intensity, VOXEL_SIZE)
    save_depth_projection(depth_img)
    save_intensity_projection(int_img)
    save_pixel_to_point_index(pixel_map)
