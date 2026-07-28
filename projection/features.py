"""projection/features.py — per-point spectral features (MANUAL §6.1, T1).

One chunked pass over the raw LAS writing three memmaps to data/derived/,
each index-aligned with LAS point order (the contract every later stage
relies on):

    exg.npy        float32   Excess Green: (2G − R − B) / (R + G + B)
    intensity.npy  float32
    nreturns.npy   uint8     number_of_returns

ExG, not NDVI: this dataset is point format 8 (has a NIR slot) but the NIR
channel is entirely zero — verified 2026-07-16, min=max=0. NDVI is therefore
impossible; ExG is the RGB-only vegetation proxy (grass/canopy are green in
the colourisation, asphalt/roofs are not). Empirically bimodal on this data:
hard-surface spike at 0, vegetation bump ~0.1–0.3.

Also writes data/derived/exg_hist.png — HUMAN GATE: the histogram must show
that hard-surface spike + vegetation bump. If it doesn't, vegetation vetoes
(MANUAL §6.5) must be calibrated on the eval tiles.
"""

from __future__ import annotations

import argparse

import laspy
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import CHUNK_SIZE, DERIVED_DIR, LAS_PATH


def run_features(
    las_path=LAS_PATH, out_dir=DERIVED_DIR, limit_chunks: int | None = None
) -> int:
    """Stream the LAS once, write exg/intensity/nreturns memmaps.
    Returns the number of points processed."""
    out_dir.mkdir(parents=True, exist_ok=True)

    with laspy.open(las_path) as f:
        n_pts = f.header.point_count
        print(f"{n_pts:,} points → {out_dir}")

        exg = np.lib.format.open_memmap(
            out_dir / "exg.npy", mode="w+", dtype=np.float32, shape=(n_pts,)
        )
        intensity = np.lib.format.open_memmap(
            out_dir / "intensity.npy", mode="w+", dtype=np.float32, shape=(n_pts,)
        )
        nreturns = np.lib.format.open_memmap(
            out_dir / "nreturns.npy", mode="w+", dtype=np.uint8, shape=(n_pts,)
        )

        off = 0
        for i, ch in enumerate(f.chunk_iterator(CHUNK_SIZE)):
            n = len(ch)
            r = np.asarray(ch.red, dtype=np.float32)
            g = np.asarray(ch.green, dtype=np.float32)
            b = np.asarray(ch.blue, dtype=np.float32)
            exg[off : off + n] = (2 * g - r - b) / (r + g + b + 1e-6)
            intensity[off : off + n] = np.asarray(ch.intensity, dtype=np.float32)
            nreturns[off : off + n] = np.asarray(ch.number_of_returns, dtype=np.uint8)
            off += n
            print(f"  chunk {i + 1}: {off:,}/{n_pts:,}", flush=True)
            if limit_chunks is not None and i + 1 >= limit_chunks:
                break

        exg.flush()
        intensity.flush()
        nreturns.flush()

    # ExG histogram — the human gate
    hist_path = out_dir / "exg_hist.png"
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(exg[:off], bins=200, range=(-0.5, 1.0), log=True)
    ax.set_xlabel("ExG")
    ax.set_ylabel("points (log)")
    ax.set_title(
        "ExG distribution — hard-surface spike at 0 + vegetation bump ≈0.1–0.3"
    )
    fig.tight_layout()
    fig.savefig(hist_path, dpi=100)
    plt.close(fig)
    print(f"Saved {hist_path}")
    print(
        "INSPECT IT: veg bump present = ExG usable; flat = calibrate vetoes on eval tiles."
    )

    return off


def _self_check() -> None:
    """Run on the first chunk only, into a throwaway directory — never the
    real DERIVED_DIR, whose exg/intensity/nreturns memmaps are full-dataset
    outputs this must not truncate; assert output sanity."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        n = run_features(out_dir=out_dir, limit_chunks=1)
        exg = np.load(out_dir / "exg.npy", mmap_mode="r")
        with laspy.open(LAS_PATH) as f:
            assert len(exg) == f.header.point_count
        sample = exg[:n]
        assert sample.dtype == np.float32
        assert np.isfinite(sample).all(), "NaN/inf in ExG"
        # normalized chromaticity bounds: 2g−r−b over r+g+b lies in [−1, 2]
        assert (sample >= -1).all() and (sample <= 2).all(), "ExG outside [-1, 2]"
    print(f"self-check OK on {n:,} points")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit-chunks", type=int, default=None)
    p.add_argument("--self-check", action="store_true")
    args = p.parse_args()
    if args.self_check:
        _self_check()
    else:
        run_features(limit_chunks=args.limit_chunks)
