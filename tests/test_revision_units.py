"""Unit checks use in-memory synthetic metadata only; no survey file reads."""

from types import SimpleNamespace

import laspy
import pytest

from revision.units import (
    INTERNATIONAL_FOOT_TO_METRE,
    UNKNOWN,
    US_SURVEY_FOOT_TO_METRE,
    UnitValidationError,
    validate_coordinate_units,
)


def axis(factor=US_SURVEY_FOOT_TO_METRE, name="US survey foot", direction="east"):
    return SimpleNamespace(
        unit_conversion_factor=factor,
        unit_name=name,
        direction=direction,
        name="Synthetic coordinate",
    )


def crs(axes, **kwargs):
    properties = {
        "axis_info": axes,
        "is_geographic": False,
        "is_vertical": False,
        "sub_crs_list": [],
        "datum": SimpleNamespace(name="Horizontal datum"),
        "to_string": lambda: "Synthetic CRS metadata",
        "to_wkt": lambda: "SYNTHETIC TEST ONLY",
    }
    properties.update(kwargs)
    return SimpleNamespace(**properties)


def validate(value, units="US survey foot"):
    return validate_coordinate_units(value, assumed_units=units)


def test_no_crs_is_an_explicit_assumption_not_verified():
    result = validate(None)
    assert result["crs"] == UNKNOWN
    assert result["vertical_datum"] == UNKNOWN
    assert "assumed" in result["horizontal_units_status"]
    assert result["configured_unit_to_metre"] == 1200 / 3937
    assert result["configured_unit_to_metre_exact"] == "1200/3937"
    assert result["provider_confirmation_required"]
    assert not result["coordinates_rescaled"]


def test_empty_synthetic_las_header_requires_no_pyproj_or_file():
    result = validate(laspy.LasHeader(point_format=8, version="1.4"))
    assert result["metadata_source"] == "LAS header has no declared CRS"
    assert result["crs"] == UNKNOWN


def test_us_survey_feet_axis_factors_verified_but_2d_vertical_datum_unknown():
    metadata = crs(
        [axis(), axis(direction="north")],
        datum=SimpleNamespace(name="NAVD88-looking horizontal text"),
    )
    result = validate(metadata, "US survey feet")
    assert "verified" in result["horizontal_units_status"]
    assert "assumed" in result["vertical_units_status"]
    assert result["vertical_datum"] == UNKNOWN
    assert result["axes"][0]["unit_to_metre"] == US_SURVEY_FOOT_TO_METRE


@pytest.mark.parametrize(
    "factor, text",
    [
        (1.0, "metre"),
        (INTERNATIONAL_FOOT_TO_METRE, "international foot"),
        (0.001, "conversion factor"),
    ],
)
def test_known_conflicting_axis_units_fail(factor, text):
    with pytest.raises(UnitValidationError, match=text):
        validate(crs([axis(factor), axis(factor)]))


def test_mixed_xy_units_and_vertical_metres_are_rejected():
    for axes in [[axis(), axis(1.0)], [axis(), axis(), axis(1.0, direction="up")]]:
        with pytest.raises(UnitValidationError, match="metre"):
            validate(crs(axes))


def test_international_feet_difference_is_not_hidden_by_tolerance():
    assert abs(US_SURVEY_FOOT_TO_METRE - INTERNATIONAL_FOOT_TO_METRE) < 0.000001
    with pytest.raises(UnitValidationError, match="international foot"):
        validate(crs([axis(0.3048), axis(0.3048)]))
    assert (
        "verified"
        in validate(crs([axis(0.304800609601), axis(0.304800609601)]))[
            "horizontal_units_status"
        ]
    )


def test_declared_geographic_crs_fails_even_with_misleading_unit_names():
    with pytest.raises(UnitValidationError, match="Geographic/angular"):
        validate(crs([axis(), axis()], is_geographic=True))


@pytest.mark.parametrize(
    "factor", [None, True, 0, -1, float("nan"), float("inf"), "unknown"]
)
def test_unusable_conversion_metadata_fails(factor):
    with pytest.raises(UnitValidationError):
        validate(crs([axis(factor), axis(factor)]))


@pytest.mark.parametrize(
    "units", [None, "", "feet", "ft", "metres", "international foot", 1]
)
def test_assumption_must_be_explicit_us_survey_feet(units):
    with pytest.raises(UnitValidationError, match="explicit configured assumption"):
        validate(None, units)


def test_missing_assumption_argument_is_not_silently_defaulted():
    with pytest.raises(TypeError):
        validate_coordinate_units(None)


def test_crs_string_or_name_is_not_resolved_or_guessed():
    with pytest.raises(UnitValidationError, match="not inferred"):
        validate("EPSG:2263")


def test_explicit_compound_vertical_metadata_is_recorded():
    vertical = SimpleNamespace(
        is_vertical=True,
        datum=SimpleNamespace(name="Synthetic explicit vertical datum"),
    )
    metadata = crs(
        [axis(), axis(direction="north"), axis(direction="up")], sub_crs_list=[vertical]
    )
    result = validate(metadata)
    assert result["vertical_datum"] == "Synthetic explicit vertical datum"
    assert "verified" in result["vertical_units_status"]
    assert not result["provider_confirmation_required"]


def test_3d_axes_alone_do_not_establish_vertical_datum():
    result = validate(crs([axis(), axis(direction="north"), axis(direction="up")]))
    assert result["vertical_datum"] == UNKNOWN
    assert "verified" in result["vertical_units_status"]
    assert result["provider_confirmation_required"]


def test_header_declaring_unreadable_crs_fails_closed():
    header = SimpleNamespace(
        parse_crs=lambda: None,
        vlrs=[SimpleNamespace(user_id="LASF_Projection", record_id=2112)],
    )
    with pytest.raises(UnitValidationError, match="could not be interpreted"):
        validate(header)


def test_missing_parser_dependency_is_not_mislabeled_as_absent_crs():
    def fail():
        raise ImportError("synthetic missing pyproj")

    with pytest.raises(UnitValidationError, match="dependency"):
        validate(SimpleNamespace(parse_crs=fail))


@pytest.mark.parametrize("axes", [[], [axis()], [axis()] * 4])
def test_unsupported_coordinate_dimensions_fail(axes):
    with pytest.raises(UnitValidationError, match="exactly two"):
        validate(crs(axes))
