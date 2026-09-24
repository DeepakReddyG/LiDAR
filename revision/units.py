"""Validate distance-unit assumptions from in-memory LAS/CRS metadata only.

No files are opened, no CRS is guessed from a filename/name/EPSG-like string,
and no coordinates are rescaled. This pipeline's distance thresholds are in US
survey feet, so a known conflicting coordinate unit is a hard error.
"""

from __future__ import annotations

import math
from typing import Any

UNKNOWN = "UNKNOWN - needs provider/PI"
US_SURVEY_FOOT_TO_METRE = 1200 / 3937
INTERNATIONAL_FOOT_TO_METRE = 0.3048
SUPPORTED_ASSUMPTIONS = {
    "us survey foot",
    "us survey feet",
    "us survey ft",
    "us_survey_foot",
    "us-ft",
    "ftus",
}


class UnitValidationError(ValueError):
    """The distance thresholds cannot safely be used with the declared CRS."""


def _has_crs_records(header) -> bool:
    """Inspect already-loaded CRS metadata records; do not read an input file."""
    for name in ("vlrs", "evlrs"):
        for record in getattr(header, name, None) or []:
            if getattr(record, "user_id", None) == "LASF_Projection" and getattr(
                record, "record_id", None
            ) in {2112, 34735}:
                return True
    return False


def _get_crs(header_or_crs):
    if header_or_crs is None:
        return None, "No CRS supplied"
    if hasattr(header_or_crs, "parse_crs"):
        try:
            crs = header_or_crs.parse_crs()
        except ImportError as exc:
            raise UnitValidationError(
                "Declared LAS CRS cannot be parsed because a CRS dependency "
                "(typically pyproj) is missing; do not assume its units"
            ) from exc
        except Exception as exc:
            raise UnitValidationError(
                "LAS CRS parsing failed; repair or verify the metadata before running"
            ) from exc
        if crs is None and _has_crs_records(header_or_crs):
            raise UnitValidationError(
                "LAS declares CRS metadata but it could not be interpreted; "
                "do not silently treat it as absent"
            )
        return (
            crs,
            "LAS header has no declared CRS"
            if crs is None
            else "LAS header CRS parser",
        )
    if not hasattr(header_or_crs, "axis_info"):
        raise UnitValidationError(
            "Pass an already-parsed CRS object or a LAS header; CRS strings/names are not inferred"
        )
    return header_or_crs, "Caller supplied parsed CRS object"


def _axis_record(axis, index: int) -> dict[str, Any]:
    raw_factor = getattr(axis, "unit_conversion_factor", None)
    try:
        if isinstance(raw_factor, bool) or raw_factor is None:
            raise ValueError
        factor = float(raw_factor)
    except (TypeError, ValueError, OverflowError) as exc:
        raise UnitValidationError(
            f"CRS axis {index + 1} lacks a numeric SI unit conversion factor"
        ) from exc
    if not math.isfinite(factor) or factor <= 0:
        raise UnitValidationError(
            f"CRS axis {index + 1} unit conversion factor must be finite and positive"
        )
    if not math.isclose(factor, US_SURVEY_FOOT_TO_METRE, rel_tol=1e-11, abs_tol=1e-12):
        if math.isclose(
            factor, INTERNATIONAL_FOOT_TO_METRE, rel_tol=1e-11, abs_tol=1e-12
        ):
            actual = "international foot (0.3048 metre)"
        elif math.isclose(factor, 1.0, rel_tol=1e-11, abs_tol=1e-12):
            actual = "metre (or another incompatible SI unit)"
        else:
            actual = f"unit with SI conversion factor {factor!r}"
        raise UnitValidationError(
            f"CRS axis {index + 1} uses {actual}; configured distance thresholds "
            "require US survey foot (1200/3937 metre). No automatic conversion was performed."
        )
    return {
        "axis_index": index,
        "name": str(getattr(axis, "name", "") or UNKNOWN),
        "direction": str(getattr(axis, "direction", "") or UNKNOWN),
        "declared_unit_name": str(getattr(axis, "unit_name", "") or UNKNOWN),
        "unit_to_metre": factor,
        "validated_unit": "US survey foot",
    }


def _vertical_datum(crs, axis_count: int) -> tuple[str, str]:
    if axis_count == 2:
        return UNKNOWN, "A 2D horizontal CRS does not establish the Z datum"
    # A horizontal geodetic datum is not a vertical datum. Only an explicitly
    # vertical sub-CRS contributes a recorded vertical datum name here.
    candidates = [crs] if getattr(crs, "is_vertical", False) else []
    candidates.extend(
        child
        for child in (getattr(crs, "sub_crs_list", None) or [])
        if getattr(child, "is_vertical", False)
    )
    names = []
    for candidate in candidates:
        datum = getattr(candidate, "datum", None)
        name = getattr(datum, "name", None)
        if isinstance(name, str) and name.strip().lower() not in {
            "",
            "unknown",
            "unnamed",
        }:
            names.append(name.strip())
    if len(set(names)) == 1:
        return names[0], "Explicit vertical sub-CRS datum metadata"
    if len(set(names)) > 1:
        raise UnitValidationError("Conflicting explicit vertical datum metadata")
    return (
        UNKNOWN,
        "Vertical axis units are declared, but no explicit vertical datum was established",
    )


def validate_coordinate_units(
    header_or_crs=None, *, assumed_units: str
) -> dict[str, Any]:
    """Return auditable metadata or reject a known unit conflict.

    `assumed_units` is required so absent metadata can never masquerade as an
    implicit choice. A 2D CRS verifies XY units only; the configured Z unit
    remains an assumption, and the Z datum is UNKNOWN. Importing pyproj is not
    necessary when a parsed CRS object or a LAS header without CRS is supplied.
    """
    if (
        not isinstance(assumed_units, str)
        or assumed_units.strip().lower() not in SUPPORTED_ASSUMPTIONS
    ):
        raise UnitValidationError(
            "This implementation requires the explicit configured assumption 'US survey foot'; "
            "other units require a separately implemented and tested conversion"
        )
    crs, source = _get_crs(header_or_crs)
    record = {
        "schema_version": 1,
        "configured_coordinate_units": "US survey foot",
        "configured_unit_to_metre": US_SURVEY_FOOT_TO_METRE,
        "configured_unit_to_metre_exact": "1200/3937",
        "input_assumption": assumed_units,
        "metadata_source": source,
        "coordinates_rescaled": False,
        "crs": UNKNOWN,
        "crs_wkt": None,
        "axes": [],
        "horizontal_units_status": "assumed; CRS absent",
        "vertical_units_status": "assumed; CRS absent",
        "vertical_datum": UNKNOWN,
        "vertical_datum_basis": "No CRS supplied or declared",
        "provider_confirmation_required": True,
    }
    if crs is None:
        return record
    if getattr(crs, "is_geographic", False):
        raise UnitValidationError(
            "Geographic/angular coordinates are incompatible with planar US survey foot thresholds"
        )
    axes = list(getattr(crs, "axis_info", None) or [])
    if len(axes) not in {2, 3}:
        raise UnitValidationError(
            "Expected exactly two horizontal axes or three spatial CRS axes"
        )
    validated = [_axis_record(axis, index) for index, axis in enumerate(axes)]
    vertical_datum, datum_basis = _vertical_datum(crs, len(axes))
    to_string = getattr(crs, "to_string", None)
    to_wkt = getattr(crs, "to_wkt", None)
    record.update(
        {
            "crs": str(to_string())
            if callable(to_string)
            else "Parsed CRS object; identifier unavailable",
            "crs_wkt": str(to_wkt()) if callable(to_wkt) else None,
            "axes": validated,
            "horizontal_units_status": "verified from declared CRS numeric axis factors",
            "vertical_units_status": (
                "verified from declared CRS numeric axis factor"
                if len(axes) == 3
                else "assumed; 2D CRS does not establish Z units"
            ),
            "vertical_datum": vertical_datum,
            "vertical_datum_basis": datum_basis,
            "provider_confirmation_required": len(axes) == 2
            or vertical_datum == UNKNOWN,
        }
    )
    return record
