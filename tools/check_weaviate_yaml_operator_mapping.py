#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
YAML_DIR = ROOT / "yaml_file" / "weaviate"
TEST_DIR = ROOT / "operator_test" / "weaviate"
SUPPORTING_ONLY_TESTS = {
    "within_geo_range_zero_update",
}


def collect_yaml_names() -> list[str]:
    return sorted(path.stem for path in YAML_DIR.glob("*.yaml"))


def collect_test_names() -> list[str]:
    names: list[str] = []
    for path in TEST_DIR.glob("*.py"):
        if path.stem.endswith("_operator"):
            names.append(path.stem[:-9])
    return sorted(names)


def main() -> int:
    yaml_names = collect_yaml_names()
    test_names = collect_test_names()

    aggregate_yaml = sorted(name for name in yaml_names if name.startswith("aggregate_"))
    mapped_yaml = (set(yaml_names) & set(test_names)) | set(aggregate_yaml)

    missing_yaml = sorted(set(yaml_names) - mapped_yaml)
    unexpected_test_only = sorted(
        set(test_names)
        - (set(yaml_names) & set(test_names))
        - {"aggregate"}
        - SUPPORTING_ONLY_TESTS
    )

    print(f"yaml_count={len(yaml_names)}")
    print(f"primary_test_count={len(test_names)}")
    print(f"one_to_one_count={len((set(yaml_names) & set(test_names)) - set(aggregate_yaml))}")
    print(f"aggregate_yaml_count={len(aggregate_yaml)}")
    print(f"supporting_only_tests={sorted(SUPPORTING_ONLY_TESTS)}")
    print(f"missing_yaml={missing_yaml}")
    print(f"unexpected_test_only={unexpected_test_only}")
    if aggregate_yaml:
        print(f"aggregate_operator_maps={aggregate_yaml}")

    return 0 if not missing_yaml and not unexpected_test_only else 2


if __name__ == "__main__":
    raise SystemExit(main())
