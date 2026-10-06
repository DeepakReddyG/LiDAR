"""CloudCompare transport checks: trust small IDs plus the original tile only."""

import json

import laspy
import numpy as np
import pytest

from evaluation.annotation import prepare_annotation, restore_annotation
from evaluation.merge_gt_parts import merge_gt_parts


@pytest.fixture
def prepared(tmp_path):
    source = tmp_path / "tile_c.las"
    cloud = laspy.LasData(laspy.LasHeader(point_format=8, version="1.4"))
    cloud.x = np.arange(6) + 100.0
    cloud.y = np.arange(6) + 200.0
    cloud.z = np.arange(6) + 800.0
    cloud.add_extra_dim(laspy.ExtraBytesParams(name="orig_index", type=np.uint32))
    cloud.orig_index = np.array(
        [17707963, 199676099, 17709187, 17710411, 17711635, 17712859]
    )
    cloud.write(source)
    annotation = tmp_path / "annotation.las"
    manifest = prepare_annotation(source, annotation)
    return source, annotation, manifest


def test_restore_shuffled_subset_ignores_corrupted_global_ids(prepared, tmp_path):
    source, annotation, manifest = prepared
    original = laspy.read(source)
    assert "orig_index" not in laspy.read(annotation).point_format.dimension_names
    cloud = laspy.read(annotation)
    idx = np.array([4, 0, 2])
    cloud.points = cloud.points[idx].copy()
    cloud.add_extra_dim(laspy.ExtraBytesParams(name="orig_index", type=np.uint32))
    cloud.orig_index = (
        np.asarray(original.orig_index)[idx].astype(np.float32).astype(np.uint32)
    )
    cloud.classification = np.full(3, 5, dtype=np.uint8)
    exported = tmp_path / "export.las"
    cloud.write(exported)
    restored_path = tmp_path / "gt_parts/tile_c_tree.las"
    restore_annotation(exported, manifest, restored_path)
    restored = laspy.read(restored_path)
    assert np.array_equal(restored.orig_index, np.asarray(original.orig_index)[idx])
    assert np.array_equal(restored.X, np.asarray(original.X)[idx])
    assert np.all(restored.classification == 5)
    merged = laspy.read(merge_gt_parts("c", eval_dir=tmp_path))
    expected = np.ones(6, dtype=np.uint8)
    expected[idx] = 5
    assert np.array_equal(merged.classification, expected)


@pytest.mark.parametrize(
    "ids, message",
    [
        ([0, 0], "duplicate"),
        ([-1, 0], "outside"),
        ([0, 6], "outside"),
        ([0.5, 1], "finite integers"),
        ([float("nan"), 1], "finite integers"),
    ],
)
def test_invalid_transport_ids_rejected(prepared, tmp_path, ids, message):
    _, annotation, manifest = prepared
    cloud = laspy.read(annotation)
    cloud.points = cloud.points[:2].copy()
    cloud.remove_extra_dim("tile_index")
    cloud.add_extra_dim(laspy.ExtraBytesParams(name="tile_index", type=np.float64))
    cloud.tile_index = ids
    exported = tmp_path / "export.las"
    cloud.write(exported)
    output = tmp_path / "restored.las"
    with pytest.raises(ValueError, match=message):
        restore_annotation(exported, manifest, output)
    assert not output.exists()


@pytest.mark.parametrize("change", ["missing", "coordinates", "reference"])
def test_mismatched_provenance_rejected(prepared, tmp_path, change):
    source, annotation, manifest = prepared
    cloud = laspy.read(annotation)
    if change == "missing":
        cloud.remove_extra_dim("tile_index")
    elif change == "coordinates":
        cloud.X = np.asarray(cloud.X) + 100
    else:
        original = laspy.read(source)
        original.orig_index[0] = 7
        original.write(source)
    exported = tmp_path / "export.las"
    cloud.write(exported)
    with pytest.raises(ValueError):
        restore_annotation(exported, manifest, tmp_path / "restored.las")


def test_prepare_and_restore_never_overwrite(prepared, tmp_path):
    source, annotation, manifest = prepared
    before = annotation.read_bytes()
    with pytest.raises(FileExistsError):
        prepare_annotation(source, annotation)
    assert annotation.read_bytes() == before
    with pytest.raises(FileExistsError):
        restore_annotation(annotation, manifest, source)
    assert json.loads(manifest.read_text())["point_count"] == 6


def test_oversized_annotation_requires_subdivision(prepared, tmp_path, monkeypatch):
    import evaluation.annotation as module

    monkeypatch.setattr(module, "ANNOTATION_MAX_POINTS", 5)
    with pytest.raises(ValueError, match="crop a smaller region"):
        prepare_annotation(prepared[0], tmp_path / "too_large.las")


@pytest.mark.parametrize(
    "change", ["duplicate", "unknown", "coordinates", "conflict", "overwrite"]
)
def test_merge_rejects_invalid_parts(prepared, tmp_path, change):
    source, _, _ = prepared
    parts = tmp_path / "gt_parts"
    parts.mkdir()
    cloud = laspy.read(source)
    cloud.points = cloud.points[:2].copy()
    if change == "duplicate":
        cloud.orig_index[1] = cloud.orig_index[0]
    elif change == "unknown":
        cloud.orig_index[0] = 42
    elif change == "coordinates":
        cloud.X = np.asarray(cloud.X) + 100
    elif change == "conflict":
        cloud.write(parts / "tile_c_grass.las")
    elif change == "overwrite":
        (tmp_path / "tile_c_gt.las").write_bytes(b"keep existing GT")
    cloud.write(parts / "tile_c_tree.las")
    with pytest.raises((ValueError, FileExistsError)):
        merge_gt_parts("c", eval_dir=tmp_path)
    output = tmp_path / "tile_c_gt.las"
    if change == "overwrite":
        assert output.read_bytes() == b"keep existing GT"
    else:
        assert not output.exists()
