"""Pure CPU contract tests; no real image, checkpoint, or GPU is accessed."""
from argparse import Namespace
import json
import struct

import numpy as np
import pytest

from revision.sam_worker import load_jobs, mask_to_rle, safetensor_shapes, sha256


@pytest.mark.parametrize("shape", [(1, 1), (2, 3), (13, 17)])
@pytest.mark.parametrize("kind", ["zero", "one", "random"])
def test_rle_exact_round_trip(shape, kind):
    if kind == "zero":
        mask = np.zeros(shape, dtype=bool)
    elif kind == "one":
        mask = np.ones(shape, dtype=bool)
    else:
        mask = np.random.default_rng(3).random(shape) > 0.5
    encoded = mask_to_rle(mask)
    flat = np.zeros(mask.size, dtype=bool)
    offset = 0
    for i, count in enumerate(encoded["counts"]):
        flat[offset:offset + count] = bool(i % 2)
        offset += count
    assert encoded["size"] == list(shape)
    assert offset == mask.size
    np.testing.assert_array_equal(mask, flat.reshape(shape))


def test_jobs_reject_existing_output_and_mutated_image(tmp_path):
    image = tmp_path / "synthetic.png"
    image.write_bytes(b"synthetic test fixture; not loaded as an image")
    output = tmp_path / "response.json"
    document = tmp_path / "jobs.json"
    expected_hash = sha256(image)
    document.write_text(json.dumps([{"image": str(image), "output": str(output), "image_sha256": expected_hash}]))
    args = Namespace(jobs=str(document), image=None, output=None)
    jobs = load_jobs(args)
    assert jobs[0]["image_sha256"] == expected_hash
    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_jobs(args)
    output.write_text("preserve me")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        load_jobs(args)
    assert output.read_text() == "preserve me"


def test_jobs_reject_duplicate_outputs(tmp_path):
    image = tmp_path / "synthetic.png"
    image.write_bytes(b"fixture")
    output = tmp_path / "response.json"
    document = tmp_path / "jobs.json"
    item = {"image": str(image), "output": str(output)}
    document.write_text(json.dumps({"jobs": [item, item]}))
    with pytest.raises(FileExistsError, match="duplicate"):
        load_jobs(Namespace(jobs=str(document), image=None, output=None))


def test_shape_reader_reads_only_declared_tensor_header(tmp_path):
    checkpoint = tmp_path / "synthetic.safetensors"
    header = json.dumps({"__metadata__": {"source": "synthetic"}, "weight": {"shape": [2, 3], "dtype": "F32", "data_offsets": [0, 24]}}).encode()
    checkpoint.write_bytes(struct.pack("<Q", len(header)) + header + bytes(24))
    assert safetensor_shapes(checkpoint) == {"weight": (2, 3)}


def test_invalid_mask_has_clear_error():
    with pytest.raises(ValueError, match="nonempty 2D"):
        mask_to_rle(np.zeros((0, 2)))
