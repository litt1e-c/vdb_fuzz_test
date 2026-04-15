#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


FIELD_ORDER = [
    "operator_id",
    "family",
    "syntax_examples",
    "doc_status",
    "doc_source_urls",
    "doc_supported_types",
    "doc_semantics",
    "doc_notes",
    "verify_status",
    "observed_summary",
    "oracle_support",
    "case_refs",
    "repro_path",
    "oracle_mapping",
]

REQUIRED_FIELDS = {
    "operator_id",
    "family",
    "syntax_examples",
    "doc_status",
    "doc_source_urls",
    "doc_supported_types",
    "doc_semantics",
    "verify_status",
    "observed_summary",
    "oracle_mapping",
}

VALID_FAMILY = {
    "comparison",
    "logical",
    "text_search",
    "array_membership",
    "nullability",
    "geo",
    "metadata_filter",
    "reference_filter",
    "other",
}

VALID_DOC_STATUS = {
    "operator_documented",
    "family_documented",
    "unspecified",
    "derived",
}

VALID_VERIFY_STATUS = {
    "not_tested",
    "sampled_match",
    "edge_validated",
    "mismatch_observed",
    "unsupported",
    "out_of_scope",
}

VALID_DOC_SUPPORTED_TYPES = {
    "scalar_numeric",
    "scalar_text",
    "scalar_bool",
    "scalar_datetime",
    "scalar_uuid",
    "scalar_geo",
    "array_field",
    "object_field",
    "reference_field",
    "metadata_field",
    "unknown",
}

VALID_ORACLE_SUPPORT = {
    "pandas",
    "pqs",
}

FAMILY_MAP = {
    "aggregate": "other",
    "aggregate_composition": "other",
    "membership": "array_membership",
    "null_state": "nullability",
    "text_pattern": "text_search",
    "rest_filter": "other",
    "geo_spatial": "geo",
}

DOC_STATUS_MAP = {
    "aggregate_filtering_documented": "derived",
    "metadata_entry_documented": "operator_documented",
    "rest_and_filtering_documented": "derived",
}

TYPE_MAP = {
    "int": ["scalar_numeric"],
    "number": ["scalar_numeric"],
    "text": ["scalar_text"],
    "boolean": ["scalar_bool"],
    "date": ["scalar_datetime"],
    "id": ["scalar_uuid"],
    "geo_coordinates": ["scalar_geo"],
    "int_array": ["array_field"],
    "number_array": ["array_field"],
    "bool_array": ["array_field"],
    "boolean_array": ["array_field"],
    "text_array": ["array_field"],
    "date_array": ["array_field"],
    "metadata_id": ["metadata_field", "scalar_uuid"],
    "metadata_timestamp": ["metadata_field"],
    "property_length": ["metadata_field"],
    "aggregate_over_all": ["unknown"],
    "array_filters": ["unknown"],
    "batch_delete_dry_run": ["unknown"],
    "filter_operand": ["unknown"],
    "filter_operands": ["unknown"],
    "null_filters": ["unknown"],
    "property_filter_predicates": ["unknown"],
    "rest_where_json": ["unknown"],
    "scalar_filters": ["unknown"],
}

ORACLE_MAP = {
    "aggregate": ["pandas"],
    "grpc": ["pandas"],
    "rest": ["pandas"],
}

TYPE_PRIORITY = [
    "scalar_numeric",
    "scalar_text",
    "scalar_bool",
    "scalar_datetime",
    "scalar_uuid",
    "scalar_geo",
    "array_field",
    "object_field",
    "reference_field",
    "metadata_field",
    "unknown",
]


def _ordered_unique(values: list[str], allowed: set[str], fallback: str) -> list[str]:
    seen = set()
    ordered = []
    for key in TYPE_PRIORITY if allowed is VALID_DOC_SUPPORTED_TYPES else sorted(allowed):
        if key in values and key not in seen:
            seen.add(key)
            ordered.append(key)
    for value in values:
        if value in allowed and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered or [fallback]


def normalize_types(raw_types: Any) -> list[str]:
    values = raw_types if isinstance(raw_types, list) else []
    normalized: list[str] = []
    for value in values:
        mapped = TYPE_MAP.get(value)
        if mapped:
            normalized.extend(mapped)
        elif value in VALID_DOC_SUPPORTED_TYPES:
            normalized.append(value)
        else:
            normalized.append("unknown")
    return _ordered_unique(normalized, VALID_DOC_SUPPORTED_TYPES, "unknown")


def normalize_oracle_support(raw_support: Any) -> list[str]:
    values = raw_support if isinstance(raw_support, list) else []
    normalized: list[str] = []
    for value in values:
        mapped = ORACLE_MAP.get(value)
        if mapped:
            normalized.extend(mapped)
        elif value in VALID_ORACLE_SUPPORT:
            normalized.append(value)
    return _ordered_unique(normalized, VALID_ORACLE_SUPPORT, "pandas") if normalized else []


def normalize_record(data: dict[str, Any]) -> dict[str, Any]:
    record = dict(data)

    family = record.get("family")
    if family in FAMILY_MAP:
        record["family"] = FAMILY_MAP[family]

    doc_status = record.get("doc_status")
    if doc_status in DOC_STATUS_MAP:
        record["doc_status"] = DOC_STATUS_MAP[doc_status]

    record["doc_supported_types"] = normalize_types(record.get("doc_supported_types"))
    record["oracle_support"] = normalize_oracle_support(record.get("oracle_support"))

    syntax_examples = record.get("syntax_examples")
    if isinstance(syntax_examples, list) and len(syntax_examples) > 3:
        record["syntax_examples"] = syntax_examples[:3]

    if "doc_notes" not in record or record["doc_notes"] is None:
        record["doc_notes"] = []
    if "oracle_support" not in record or record["oracle_support"] is None:
        record["oracle_support"] = []
    if "case_refs" not in record or record["case_refs"] is None:
        record["case_refs"] = []
    if "repro_path" not in record:
        record["repro_path"] = None

    ordered: dict[str, Any] = {}
    for field in FIELD_ORDER:
        if field in record:
            ordered[field] = record[field]
    for key, value in record.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def validate_record(path: Path, data: Any) -> list[str]:
    issues: list[str] = []
    if not isinstance(data, dict):
        return [f"{path}: top-level YAML value must be a mapping"]

    missing = [field for field in REQUIRED_FIELDS if field not in data]
    if missing:
        issues.append(f"{path}: missing required fields {missing}")

    family = data.get("family")
    if family not in VALID_FAMILY:
        issues.append(f"{path}: nonstandard family `{family}`")

    syntax_examples = data.get("syntax_examples")
    if not isinstance(syntax_examples, list) or not all(isinstance(x, str) for x in syntax_examples):
        issues.append(f"{path}: syntax_examples must be a list of strings")
    elif not (1 <= len(syntax_examples) <= 3):
        issues.append(f"{path}: syntax_examples count {len(syntax_examples)} is outside 1..3")

    doc_status = data.get("doc_status")
    if doc_status not in VALID_DOC_STATUS:
        issues.append(f"{path}: nonstandard doc_status `{doc_status}`")

    doc_urls = data.get("doc_source_urls")
    if not isinstance(doc_urls, list) or not all(isinstance(x, str) for x in doc_urls):
        issues.append(f"{path}: doc_source_urls must be a list of strings")

    supported_types = data.get("doc_supported_types")
    if not isinstance(supported_types, list) or not all(isinstance(x, str) for x in supported_types):
        issues.append(f"{path}: doc_supported_types must be a list of strings")
    else:
        bad = [value for value in supported_types if value not in VALID_DOC_SUPPORTED_TYPES]
        if bad:
            issues.append(f"{path}: nonstandard doc_supported_types {bad}")

    verify_status = data.get("verify_status")
    if verify_status not in VALID_VERIFY_STATUS:
        issues.append(f"{path}: nonstandard verify_status `{verify_status}`")

    oracle_support = data.get("oracle_support", [])
    if oracle_support is not None:
        if not isinstance(oracle_support, list) or not all(isinstance(x, str) for x in oracle_support):
            issues.append(f"{path}: oracle_support must be a list of strings")
        else:
            bad = [value for value in oracle_support if value not in VALID_ORACLE_SUPPORT]
            if bad:
                issues.append(f"{path}: nonstandard oracle_support {bad}")

    return issues


def load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    text = yaml.safe_dump(
        data,
        sort_keys=False,
        allow_unicode=True,
        width=1000,
    )
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate or normalize Weaviate YAML oracle files.")
    parser.add_argument("--dir", default="yaml_file/weaviate", help="Directory containing YAML files.")
    parser.add_argument("--normalize", action="store_true", help="Normalize legacy enum values in place.")
    args = parser.parse_args()

    root = Path(args.dir)
    paths = sorted(root.glob("*.yaml"))
    if not paths:
        print(f"No YAML files found under {root}")
        return 1

    normalized_count = 0
    issues: list[str] = []

    for path in paths:
        data = load_yaml(path)
        if args.normalize and isinstance(data, dict):
            normalized = normalize_record(data)
            if normalized != data:
                dump_yaml(path, normalized)
                data = normalized
                normalized_count += 1
        issues.extend(validate_record(path, data))

    print(f"checked_files={len(paths)}")
    print(f"normalized_files={normalized_count}")
    print(f"issue_count={len(issues)}")
    for issue in issues:
        print(issue)

    return 0 if not issues else 2


if __name__ == "__main__":
    raise SystemExit(main())
