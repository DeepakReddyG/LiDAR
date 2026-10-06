"""Regressions for clean grid setup and scoring partially annotated LAS tiles."""

import json
import math
import subprocess
import sys
import textwrap

import laspy
import numpy as np
import pytest

from evaluation.evaluate import _make_synthetic_las, evaluate


def test_grid_initialization_preserves_matching_metadata(tmp_path):
    from projection.grid import run_grid

    las = laspy.LasData(laspy.LasHeader(point_format=8, version="1.4"))
    las.x = [10.0, 12.25]
    las.y = [20.0, 23.0]
    las.z = [1.0, 2.0]
    source = tmp_path / "source.las"
    las.write(source)
    target = tmp_path / "slices" / "grid_meta.npz"

    run_grid(source, target, resolution=0.5)
    with np.load(target) as meta:
        assert {k: meta[k].item() for k in meta.files} == {
            "x_min": 10.0,
            "y_min": 20.0,
            "x_max": 12.25,
            "y_max": 23.0,
            "resolution": 0.5,
            "rows": 6,
            "cols": 5,
        }
    saved = target.read_bytes()
    modified = target.stat().st_mtime_ns
    run_grid(source, target, resolution=0.5)
    assert target.stat().st_mtime_ns == modified
    with pytest.raises(ValueError, match="Existing grid"):
        run_grid(source, target, resolution=1.0)
    assert target.read_bytes() == saved


@pytest.mark.parametrize("resolution", [0, -0.5, float("nan"), float("inf")])
def test_grid_rejects_invalid_resolution(resolution):
    from projection.grid import grid_from_header

    header = laspy.LasHeader(point_format=8, version="1.4")
    with pytest.raises(ValueError, match="resolution"):
        grid_from_header(header, resolution)


def test_cli_bootstrap_from_empty_outputs(tmp_path):
    # A fresh process ensures all stage defaults use the temporary config.
    # Execute actual grid/features/CSF/ortho code, stopping before model inference.
    script = textwrap.dedent("""\
        import sys
        from pathlib import Path
        import laspy
        import numpy as np
        import config

        root = Path(sys.argv[1])
        for name, subdir in [("DERIVED_DIR", "derived"), ("SLICES_DIR", "slices"),
                             ("MASKS_DIR", "masks"), ("OUTPUT_DIR", "output")]:
            setattr(config, name, root / subdir)
        config.GRID_META_PATH = config.SLICES_DIR / "grid_meta.npz"
        config.LAS_PATH = root / "terrain.las"
        config.CHUNK_SIZE = 1000
        x, y = np.meshgrid(np.arange(64.0), np.arange(64.0))
        las = laspy.LasData(laspy.LasHeader(point_format=8, version="1.4"))
        las.x, las.y = x.ravel(), y.ravel()
        las.z = 800 + 0.01 * x.ravel()
        las.red = np.full(x.size, 20000, np.uint16)
        las.green = np.full(x.size, 30000, np.uint16)
        las.blue = np.full(x.size, 20000, np.uint16)
        las.write(config.LAS_PATH)

        import main
        assert main.V2_SEQUENCE[:4] == ["grid", "features", "ground", "ortho"]
        main.V2_SEQUENCE = main.V2_SEQUENCE[:4]
        sys.argv = ["main.py", "--stage", "all"]
        main.main()
        assert np.load(config.DERIVED_DIR / "hag.npy").shape == (4096,)
        assert np.isfinite(np.load(config.DERIVED_DIR / "hag.npy")).all()
        assert np.load(config.SLICES_DIR / "surface_z.npy").shape == (126, 126)
        assert (config.SLICES_DIR / "ortho_rgb.png").exists()
        assert list((config.SLICES_DIR / "tiles").glob("*.png"))
        """)
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_partial_gt_is_filtered_after_identity_alignment(tmp_path):
    gt = tmp_path / "gt.las"
    pred = tmp_path / "pred.las"
    _make_synthetic_las(gt, np.array([5, 1, 0, 3]), np.array([10, 11, 12, 13]))
    # Both ignored points predict tree, but may not inflate tree false positives.
    _make_synthetic_las(pred, np.array([5, 3, 5, 5]), np.array([12, 13, 10, 11]))
    result = evaluate(gt, pred)
    assert result["iou"]["tree"] == 1.0
    assert result["iou"]["grass"] == 1.0
    assert result["matched_points"] == 4
    assert result["scored_points"] == 2
    assert result["ignored_gt_points"] == 2
    assert np.sum(result["confusion_matrix"]) == 2
    assert math.isnan(result["iou"]["unlabelled"])


@pytest.mark.parametrize("prediction", [0, 1, 3])
def test_errors_on_annotated_points_are_not_ignored(tmp_path, prediction):
    gt = tmp_path / "gt.las"
    pred = tmp_path / "pred.las"
    _make_synthetic_las(gt, np.array([5, 5, 1]), np.arange(3))
    _make_synthetic_las(pred, np.array([5, prediction, 5]), np.arange(3))
    result = evaluate(gt, pred)
    assert result["iou"]["tree"] == 0.5
    assert np.sum(result["confusion_matrix"]) == 2


def test_completely_unannotated_gt_has_no_score(tmp_path):
    gt = tmp_path / "gt.las"
    pred = tmp_path / "pred.las"
    _make_synthetic_las(gt, np.array([0, 1]), np.arange(2))
    _make_synthetic_las(pred, np.array([5, 5]), np.arange(2))
    with pytest.raises(ValueError, match="No annotated ground-truth points"):
        evaluate(gt, pred)


def test_merged_partial_gt_scores_baseline_and_pipeline(tmp_path, monkeypatch):
    import config
    from evaluation.evaluate import run_evaluate_all
    from evaluation.merge_gt_parts import merge_gt_parts

    parts = tmp_path / "gt_parts"
    parts.mkdir()
    _make_synthetic_las(tmp_path / "tile_c.las", np.zeros(3), np.arange(3))
    _make_synthetic_las(parts / "tile_c_tree.las", np.zeros(1), np.array([1]))
    merge_gt_parts("c", eval_dir=tmp_path)
    _make_synthetic_las(
        tmp_path / "tile_c_baseline.las", np.array([3, 5, 11]), np.arange(3)
    )
    np.save(tmp_path / "labels.npy", np.array([0, 4, 3], dtype=np.int32))
    monkeypatch.setattr(config, "EVAL_DIR", tmp_path)
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(config, "EVAL_TILE_BOUNDS", {"c": (0, 0, 1, 1)})
    run_evaluate_all()
    for name in ("baseline_scores.json", "sam3_scores.json"):
        assert json.loads((tmp_path / name).read_text())["c"]["tree"] == 1.0
