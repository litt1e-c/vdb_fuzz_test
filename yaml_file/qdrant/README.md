# Qdrant YAML Oracle Entries

This directory stores the paper-facing operator oracle records for the Qdrant
scalar, filter, and deterministic profile subset. Each `.yaml` entry is
intended to be:

- reproducible: local evidence is tied to a retained operator test or probe;
- explainable: documented semantics and observed behavior are separated;
- auditable: schema validation and YAML-to-test mapping are machine-checkable.

## Schema

- Template: `yaml_file/qdrant/TEMPLATE.yaml.example`
- Validator: `python tools/validate_qdrant_yaml_schema.py --dir yaml_file/qdrant`
- Optional in-place normalization:
  `python tools/validate_qdrant_yaml_schema.py --dir yaml_file/qdrant --normalize`

The validator enforces field order, required fields, enum values, the `1..3`
limit on `syntax_examples`, and basic type checks for optional fields.

## YAML/Test Mapping

Primary mapping can be checked with:

`python tools/check_qdrant_yaml_operator_mapping.py`

Current intended structure:

- one-to-one mappings: most `yaml_file/qdrant/<name>.yaml` entries map to
  `operator_test/qdrant/<name>_operator.py`;
- formula mappings: arithmetic YAML entries such as `abs.yaml`, `pow.yaml`,
  and `sum.yaml` map to `operator_test/qdrant/formula/<name>_operator.py`;
- deterministic profile entries such as `text_profile.yaml` and
  `datetime_profile.yaml` remain first-class YAML entries because they capture
  documented Qdrant index-profile behavior with explicit, reproducible
  fixtures;
- historical probes in `history_find_bug/qdrant/` may still be cited from
  `case_refs` or `doc_notes`; if a YAML entry references such a probe, that
  probe should be retained.

Shared `case_refs` are allowed when one retained repro informs more than one
operator entry, but duplicate `operator_id` values or orphaned operator tests
should be treated as cleanup issues.

Standalone temporary probes, caches, and one-off local scratch files should not
remain in this directory tree.
