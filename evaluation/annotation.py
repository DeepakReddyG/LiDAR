"""Prepare CloudCompare-safe copies and restore exact point identities.

Use `python -m evaluation.annotation --help`. The original tile and a hashed
manifest remain outside CloudCompare. Exported large orig_index values are
never trusted; a small tile_index selects original records before GT merging.
"""

import argparse
import hashlib
import json
from pathlib import Path

import laspy
import numpy as np

from config import ANNOTATION_MAX_POINTS, ANNOTATION_XYZ_TOLERANCE


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_new(cloud, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        cloud.write(stream)


def prepare_annotation(reference_path, output_path):
    reference_path = Path(reference_path).resolve()
    output_path = Path(output_path).resolve()
    manifest_path = output_path.with_suffix(".json")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(
            "Annotation copy or manifest already exists; use a new name"
        )
    reference = laspy.read(reference_path)
    n = len(reference.points)
    if not 0 < n <= ANNOTATION_MAX_POINTS:
        raise ValueError(
            "Annotation copy must contain 1 to 2^24 points; crop a smaller region"
        )
    if "orig_index" not in reference.point_format.extra_dimension_names:
        raise ValueError("Reference tile has no orig_index")
    if "tile_index" in reference.point_format.dimension_names:
        raise ValueError("Reference already contains reserved tile_index")
    ids = np.asarray(reference.orig_index)
    if ids.dtype.kind not in "ui" or len(np.unique(ids)) != n:
        raise ValueError("Reference orig_index must contain unique integer IDs")
    manifest = {
        "version": 1,
        "reference_path": str(reference_path),
        "reference_sha256": _sha256(reference_path),
        "point_count": n,
        "identity_field": "tile_index",
    }
    # Remove the unsafe large ID so an un-restored export cannot be merged by it.
    reference.remove_extra_dim("orig_index")
    reference.add_extra_dim(laspy.ExtraBytesParams(name="tile_index", type=np.uint32))
    reference.tile_index = np.arange(n, dtype=np.uint32)
    _write_new(reference, output_path)
    with manifest_path.open("x") as stream:
        json.dump(manifest, stream, indent=2)
    print(
        f"Prepared {n:,} points: {output_path}\nKeep outside CloudCompare: {manifest_path}"
    )
    return manifest_path


def restore_annotation(export_path, manifest_path, output_path):
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("version") != 1 or manifest.get("identity_field") != "tile_index":
        raise ValueError("Unsupported annotation manifest")
    reference_path = Path(manifest["reference_path"])
    if _sha256(reference_path) != manifest["reference_sha256"]:
        raise ValueError(
            "Reference tile changed since preparation; identity mapping is invalid"
        )
    reference = laspy.read(reference_path)
    n = len(reference.points)
    if n != manifest["point_count"] or not 0 < n <= ANNOTATION_MAX_POINTS:
        raise ValueError("Reference point count does not match the manifest")
    cloud = laspy.read(export_path)
    if "tile_index" not in cloud.point_format.dimension_names:
        raise ValueError(
            "Export lost tile_index; export that extra scalar field from CloudCompare"
        )
    raw = np.asarray(cloud.tile_index)
    if len(raw) == 0:
        raise ValueError("Export contains no points")
    if not np.isfinite(raw).all() or not np.equal(raw, np.floor(raw)).all():
        raise ValueError("tile_index values must be finite integers")
    if np.any(raw < 0) or np.any(raw >= n):
        raise ValueError("tile_index is outside the reference tile range")
    idx = raw.astype(np.int64)
    if len(np.unique(idx)) != len(idx):
        raise ValueError("Export contains duplicate tile_index values")
    for dim in ("x", "y", "z"):
        expected = np.asarray(getattr(reference, dim))[idx]
        actual = np.asarray(getattr(cloud, dim))
        if not np.allclose(expected, actual, rtol=0, atol=ANNOTATION_XYZ_TOLERANCE):
            raise ValueError(
                f"Export {dim} coordinates do not match the selected reference IDs"
            )
    # Original point records retain the exact coordinates, attributes, and IDs.
    restored = laspy.LasData(reference.header.copy())
    restored.points = reference.points[idx].copy()
    restored.classification = np.asarray(cloud.classification)
    _write_new(restored, output_path)
    print(f"Restored {len(idx):,} exact orig_index values: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser(
        "prepare", help="Create a safe annotation copy and manifest"
    )
    prepare.add_argument("--reference", required=True)
    prepare.add_argument("--output", required=True)
    restore = sub.add_parser(
        "restore", help="Recover exact IDs after CloudCompare export"
    )
    restore.add_argument("--export", required=True)
    restore.add_argument("--manifest", required=True)
    restore.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_annotation(args.reference, args.output)
    else:
        restore_annotation(args.export, args.manifest, args.output)


if __name__ == "__main__":
    main()
