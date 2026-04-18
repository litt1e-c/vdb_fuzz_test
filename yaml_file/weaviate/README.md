# Weaviate YAML Oracle Entries

This directory stores the paper-facing operator oracle records for the Weaviate
scalar and filter subset. Each `.yaml` entry is intended to be:

- reproducible: local evidence is tied to a retained operator test or probe;
- explainable: documented semantics and observed behavior are separated;
- auditable: schema validation and YAML-to-test mapping are machine-checkable.

## Schema

- Template: `yaml_file/weaviate/TEMPLATE.yaml.example`
- Validator: `python tools/validate_weaviate_yaml_schema.py --dir yaml_file/weaviate`
- Optional in-place normalization:
  `python tools/validate_weaviate_yaml_schema.py --dir yaml_file/weaviate --normalize`

The validator enforces field order and enum values, and it normalizes
repo-local absolute `repro_path` values into workspace-relative paths.

## YAML/Test Mapping

Primary mapping can be checked with:

`python tools/check_weaviate_yaml_operator_mapping.py`

Current intended structure:

- one-to-one mappings: most `yaml_file/weaviate/<name>.yaml` entries map to
  `operator_test/weaviate/<name>_operator.py`;
- one-to-many aggregate mapping: all `aggregate_*` YAML entries share
  `operator_test/weaviate/aggregate_operator.py`;
- supporting edge probe: `within_geo_range.yaml` additionally relies on
  `operator_test/weaviate/within_geo_range_zero_update_operator.py` for the
  zero-distance caution captured in the YAML notes.

Standalone historical probes that are not part of the retained oracle basis
should not remain in this directory tree.
