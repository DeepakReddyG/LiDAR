"""Synthetic regressions for M1/M-minor terrain and export corrections."""
import laspy
import numpy as np
import pytest

from projection import ground
from reprojection import map_back


@pytest.mark.parametrize("reverse", [False, True])
def test_dtm_collision_takes_lowest_ground_elevation(monkeypatch, reverse):
    monkeypatch.setattr(ground, "_grid_meta", lambda: {
        "x_min": 0., "y_max": 2., "resolution": 1., "rows": 2, "cols": 2,
    })
    # The first cell's distinct 100/110 elevations expose fmax versus fmin;
    # filled neighboring cells avoid conflating aggregation with gap-filling.
    xyz = np.array([[.2, 1.8, 110.], [.8, 1.2, 100.], [1.2, 1.8, 120.], [.2, .8, 130.], [1.2, .8, 140.]])
    if reverse:
        xyz = xyz[::-1]
    dtm = ground.build_dtm(xyz, cell=1.)
    np.testing.assert_array_equal(dtm, [[100., 120.], [130., 140.]])


def test_dtm_minimum_keeps_empty_cells_available_for_gap_fill(monkeypatch):
    monkeypatch.setattr(ground, "_grid_meta", lambda: {
        "x_min": 0., "y_max": 2., "resolution": 1., "rows": 2, "cols": 2,
    })
    dtm = ground.build_dtm(np.array([[.2, 1.8, 110.], [.8, 1.2, 100.]]), cell=1.)
    np.testing.assert_array_equal(dtm, np.full((2, 2), 100., dtype=np.float32))


def point_fixture():
    z = np.array([29., 15., .5, .5, 1., 20., 10., 10.], np.float32)
    hag = z.copy()
    exg = np.array([.3, .1, .2, 0., .2, .2, .1, .1], np.float32)
    r, c = np.zeros(8, dtype=int), np.arange(8)
    label_grid = np.full((1, 8), map_back._NAME_TO_ID["tree"], np.int32)
    label_grid[0, 6] = -1
    conf_grid = np.full((1, 8), .8, np.float32)
    conf_grid[0, 6:] = 0.
    surface = np.array([[30., 30., 30., 30., np.nan, 10., 10., 10.]], np.float32)
    return z, r, c, label_grid, conf_grid, surface, hag, exg


def test_provenance_retains_legacy_labels_and_separates_scores():
    args = point_fixture()
    legacy, flags = map_back.label_points(*args)
    labels, scores, sources = map_back.label_points_with_provenance(*args)
    np.testing.assert_array_equal(labels, legacy)
    np.testing.assert_array_equal(sources, [1, 2, 2, 2, 3, 4, 0, 1])
    assert labels.tolist() == [4, 4, 3, 0, 3, 4, -1, 4]
    assert scores.dtype == np.float32 and sources.dtype == np.uint8
    assert scores[0] == np.float32(.8)
    assert scores[7] == 0.  # Real zero when smoothing propagates an unsupported class.
    assert np.isnan(scores[1:7]).all()
    assert flags[0] == np.uint8(.8 * 255)
    assert (flags[1:6] == map_back.MAP_BACK["rule_conf"]).all()


def synthetic_las():
    header = laspy.LasHeader(point_format=8, version="1.4")
    header.scales = [.001, .002, .003]
    header.offsets = [1_391_000., 448_000., 800.]
    header.add_extra_dims([
        laspy.ExtraBytesParams(name="orig_index", type=np.uint32),
        laspy.ExtraBytesParams(name="custom_scaled", type=np.int16, scales=[.01], offsets=[-20.]),
    ])
    header.vlrs.append(laspy.VLR(user_id="test_owner", record_id=7, description="synthetic metadata", record_data=b"preserve original provider metadata"))
    header.vlrs.append(laspy.vlrs.known.WktCoordinateSystemVlr(
        wkt_string='LOCAL_CS["Synthetic feet",UNIT["US survey foot",0.3048006096012192]]'
    ))
    header.global_encoding.wkt = True
    cloud = laspy.LasData(header)
    cloud.X = np.arange(8, dtype=np.int32) * 10
    cloud.Y = np.arange(8, dtype=np.int32) * -15
    cloud.Z = np.arange(8, dtype=np.int32) * 25
    cloud.classification = np.arange(8, dtype=np.uint8)
    cloud.user_data = np.arange(8, dtype=np.uint8) + 27
    cloud.red = np.arange(8, dtype=np.uint16) * 777
    cloud.green = np.arange(8, dtype=np.uint16) * 888
    cloud.blue = np.arange(8, dtype=np.uint16) * 999
    cloud.intensity = np.arange(8, dtype=np.uint16) + 30000
    cloud.gps_time = np.arange(8, dtype=np.float64) + 12.25
    cloud.return_number = np.full(8, 2, dtype=np.uint8)
    cloud.number_of_returns = np.full(8, 3, dtype=np.uint8)
    cloud.synthetic = np.arange(8) % 2
    cloud.withheld = (np.arange(8) + 1) % 2
    cloud.orig_index = np.arange(8, dtype=np.uint32) + 400_000_000
    cloud.custom_scaled = np.arange(8) * .02 - 10.
    return cloud


def test_prediction_las_roundtrip_preserves_source_fields_and_metadata(tmp_path):
    cloud = synthetic_las()
    original = cloud.points.array.copy()
    original_dimensions = list(cloud.point_format.dimension_names)
    labels, scores, sources = map_back.label_points_with_provenance(*point_fixture())
    header = map_back.prediction_header(cloud.header)
    records = map_back.prediction_records(cloud.points, header, labels, scores, sources)
    output = tmp_path / "synthetic_predictions.las"
    with laspy.open(output, mode="w", header=header) as writer:
        writer.write_points(records)
    reopened = laspy.read(output)
    for name in original.dtype.names:
        if name != "classification":
            np.testing.assert_array_equal(reopened.points.array[name], original[name], err_msg=name)
    np.testing.assert_array_equal(reopened.classification, map_back.labels_to_las_codes(labels))
    np.testing.assert_array_equal(reopened.model_score, scores)
    np.testing.assert_array_equal(reopened.prediction_source, sources)
    np.testing.assert_array_equal(reopened.internal_class_id, labels)
    np.testing.assert_array_equal(reopened.header.scales, cloud.header.scales)
    np.testing.assert_array_equal(reopened.header.offsets, cloud.header.offsets)
    assert reopened.header.global_encoding.wkt
    for original_vlr in cloud.header.vlrs:
        if original_vlr.record_id == 4 and original_vlr.user_id == "LASF_Spec":
            continue  # ExtraBytes VLR intentionally grows with the new fields.
        retained = [v for v in reopened.header.vlrs if v.user_id == original_vlr.user_id and v.record_id == original_vlr.record_id]
        assert len(retained) == 1
        assert retained[0].record_data_bytes() == original_vlr.record_data_bytes()
    np.testing.assert_array_equal(cloud.points.array, original)
    assert list(cloud.point_format.dimension_names) == original_dimensions


@pytest.mark.parametrize("name", list(map_back.PREDICTION_DIMENSIONS))
def test_prediction_header_rejects_existing_dimension_collision(name):
    header = laspy.LasHeader(point_format=8, version="1.4")
    header.add_extra_dim(laspy.ExtraBytesParams(name=name, type=np.uint8))
    with pytest.raises(ValueError, match="collision"):
        map_back.prediction_header(header)


def test_prediction_records_reject_coordinate_reinterpretation():
    cloud = synthetic_las()
    header = map_back.prediction_header(cloud.header)
    header.scales = [.01, .02, .03]
    with pytest.raises(ValueError, match="scales/offsets"):
        map_back.prediction_records(cloud.points, header, *map_back.label_points_with_provenance(*point_fixture()))


def test_map_back_self_check_retains_legacy_api():
    map_back._self_check()
