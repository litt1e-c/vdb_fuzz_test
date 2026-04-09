import time
from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema, DataType, utility
)

HOST = "127.0.0.1"
PORT = "19531"
COLLECTION_NAME = "neg_zero_index_bug"

def main():
    connections.connect(host=HOST, port=PORT)
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)

    # Define collection schema
    fields = [
        FieldSchema("id", DataType.INT64, is_primary=True),
        FieldSchema("c10", DataType.BOOL, nullable=True),
        FieldSchema("c12", DataType.FLOAT, nullable=True),
        FieldSchema("vec", DataType.FLOAT_VECTOR, dim=2),
    ]
    schema = CollectionSchema(fields)
    col = Collection(COLLECTION_NAME, schema)

    # Construct test data: contains many records with c12=0.0 and c10=False
    rows = []
    for i in range(1, 5001):
        # Let about 10% of records have c12=0.0, 10% have c12=-0.0, the rest random positive numbers or None
        if i % 10 == 0:
            c12 = 0.0
        elif i % 10 == 1:
            c12 = -0.0
        elif i % 10 == 2:
            c12 = None
        else:
            c12 = float(i % 1000)   # positive number, will not affect condition
        c10 = False if i % 4 != 0 else True   # 75% are False
        rows.append({"id": i, "c10": c10, "c12": c12, "vec": [0.0, 0.0]})

    col.insert(rows)
    col.flush()

    # Create indexes: c12 uses INVERTED, c10 also creates index (optional)
    col.create_index("vec", {"metric_type": "L2", "index_type": "FLAT"}, index_name="idx_vec")
    col.create_index("c12", {"index_type": "INVERTED"}, index_name="idx_c12")
    col.create_index("c10", {"index_type": "INVERTED"}, index_name="idx_c10")
    col.load()
    time.sleep(2)   # Ensure index loading completes

    # Warm-up query: force using c12 index to make subsequent query path prefer index scan
    warmup = col.query("c12 <= -0.0", output_fields=["id"], limit=1)
    print(f"Warmup returned {len(warmup)} rows")

    # Target expression
    expr = "(not ((c12 is not null and (c12 <= -0.0))) and (c10 is not null and (c10 == false)))"

    # Execute query
    res = col.query(expr, output_fields=["id", "c10", "c12"], consistency_level="Strong")
    actual_ids = {int(r["id"]) for r in res}

    # Calculate Python expected results (following IEEE 754 semantics)
    def py_eval(c10, c12):
        cond_inner = (c12 is not None) and (c12 <= -0.0)
        return (not cond_inner) and (c10 is False)

    expected_ids = {r["id"] for r in rows if py_eval(r["c10"], r["c12"])}

    # Compare results
    if expected_ids == actual_ids:
        print("✅ MATCH")
        return 0
    else:
        extra = actual_ids - expected_ids
        print(f"❌ MISMATCH: Milvus returned {len(actual_ids)} rows, Python expected {len(expected_ids)}")
        print(f"Extra IDs (Milvus has but should not): {sorted(extra)[:20]}")
        for rid in sorted(extra)[:5]:
            row = next(r for r in rows if r["id"] == rid)
            print(f"  ID {rid}: c10={row['c10']}, c12={row['c12']}")
        return 2

if __name__ == "__main__":
    import sys
    sys.exit(main())


#for further test,can use the script below
import argparse
import time
from typing import Dict, List, Optional, Set, Tuple

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusException,
    connections,
    utility,
)


TARGET_EXPR = "(not ((c12 is not null and (c12 <= -0.0))) and (c10 is not null and (c10 == false)))"
WARMUP_EXPR = "c12 <= -0.0"


def python_eval(c10: Optional[bool], c12: Optional[float]) -> bool:
    cond_inner = (c12 is not None) and (c12 <= -0.0)
    return (not cond_inner) and (c10 is False)


def build_rows(n: int) -> List[Dict]:
    rows: List[Dict] = []
    for i in range(1, n + 1):
        if i % 10 == 0:
            c12 = 0.0
        elif i % 10 == 1:
            c12 = -0.0
        elif i % 10 == 2:
            c12 = None
        else:
            c12 = float(i % 1000)
        c10 = False if i % 4 != 0 else True
        rows.append({"id": i, "c10": c10, "c12": c12, "vec": [0.0, 0.0]})
    return rows


def setup_collection(name: str) -> Collection:
    if utility.has_collection(name):
        utility.drop_collection(name)

    schema = CollectionSchema(
        [
            FieldSchema("id", DataType.INT64, is_primary=True),
            FieldSchema("c10", DataType.BOOL, nullable=True),
            FieldSchema("c12", DataType.FLOAT, nullable=True),
            FieldSchema("vec", DataType.FLOAT_VECTOR, dim=2),
        ]
    )
    return Collection(name, schema)


def create_indexes(col: Collection, create_c10_index: bool, create_c12_index: bool) -> None:
    col.create_index("vec", {"metric_type": "L2", "index_type": "FLAT"}, index_name="idx_vec")
    if create_c10_index:
        col.create_index("c10", {"index_type": "INVERTED"}, index_name="idx_c10")
    if create_c12_index:
        col.create_index("c12", {"index_type": "INVERTED"}, index_name="idx_c12")


def compute_expected(rows: List[Dict]) -> Set[int]:
    return {int(r["id"]) for r in rows if python_eval(r["c10"], r["c12"])}


def query_actual(col: Collection) -> Tuple[Set[int], Dict[int, Dict]]:
    res = col.query(
        TARGET_EXPR,
        output_fields=["id", "c10", "c12"],
        consistency_level="Strong",
    )
    actual_ids: Set[int] = set()
    actual_rows: Dict[int, Dict] = {}
    for row in res:
        rid = int(row["id"])
        actual_ids.add(rid)
        actual_rows[rid] = {"id": rid, "c10": row.get("c10"), "c12": row.get("c12")}
    return actual_ids, actual_rows


def print_case_result(
    name: str,
    create_c10_index: bool,
    create_c12_index: bool,
    do_warmup: bool,
    expected_ids: Set[int],
    actual_ids: Set[int],
    source_rows: Dict[int, Dict],
    actual_rows: Dict[int, Dict],
) -> bool:
    status = expected_ids == actual_ids
    print(f"\n=== {name} ===")
    print(
        "config:"
        f" c10_index={'on' if create_c10_index else 'off'},"
        f" c12_index={'on' if create_c12_index else 'off'},"
        f" warmup={'on' if do_warmup else 'off'}"
    )

    if status:
        print(f"MATCH | rows={len(actual_ids)}")
        return True

    extra = sorted(actual_ids - expected_ids)
    missing = sorted(expected_ids - actual_ids)
    print(f"MISMATCH | actual={len(actual_ids)} expected={len(expected_ids)}")
    print(f"extra={extra[:20]}")
    print(f"missing={missing[:20]}")

    if extra:
        print("extra sample:")
        for rid in extra[:5]:
            print(f"  ID {rid}: actual_row={actual_rows.get(rid)} source_row={source_rows.get(rid)}")

    if missing:
        print("missing sample:")
        for rid in missing[:5]:
            print(f"  ID {rid}: source_row={source_rows.get(rid)}")

    return False


def run_case(
    case_name: str,
    rows: List[Dict],
    create_c10_index: bool,
    create_c12_index: bool,
    do_warmup: bool,
    load_wait_s: float,
) -> bool:
    col: Optional[Collection] = None
    collection_name = f"neg_zero_compare_{case_name}"

    try:
        col = setup_collection(collection_name)
        col.insert(rows)
        col.flush()

        create_indexes(col, create_c10_index=create_c10_index, create_c12_index=create_c12_index)
        col.load()

        if load_wait_s > 0:
            time.sleep(load_wait_s)

        if do_warmup:
            warmup = col.query(WARMUP_EXPR, output_fields=["id"], limit=1, consistency_level="Strong")
            print(f"\n[{case_name}] warmup returned {len(warmup)} rows")

        expected_ids = compute_expected(rows)
        source_rows = {int(r["id"]): r for r in rows}
        actual_ids, actual_rows = query_actual(col)

        return print_case_result(
            name=case_name,
            create_c10_index=create_c10_index,
            create_c12_index=create_c12_index,
            do_warmup=do_warmup,
            expected_ids=expected_ids,
            actual_ids=actual_ids,
            source_rows=source_rows,
            actual_rows=actual_rows,
        )
    finally:
        if col is not None:
            try:
                utility.drop_collection(collection_name)
            except Exception:
                pass


def run_compare(host: str, port: str, row_count: int, load_wait_s: float) -> int:
    rows = build_rows(row_count)
    cases = [
        ("both_indexes_no_warmup", True, True, False),
        ("both_indexes_with_warmup", True, True, True),
        ("only_c10_index", True, False, False),
        ("no_scalar_indexes", False, False, False),
    ]

    try:
        connections.connect(alias="default", host=host, port=port, timeout=30)
        try:
            version = utility.get_server_version()
        except Exception:
            version = "unknown"

        print(f"Milvus version: {version}")
        print(f"rows={row_count}")
        print(f"target_expr={TARGET_EXPR}")

        all_ok = True
        for case_name, create_c10_index, create_c12_index, do_warmup in cases:
            ok = run_case(
                case_name=case_name,
                rows=rows,
                create_c10_index=create_c10_index,
                create_c12_index=create_c12_index,
                do_warmup=do_warmup,
                load_wait_s=load_wait_s,
            )
            all_ok = all_ok and ok

        print("\n=== Summary ===")
        if all_ok:
            print("All cases MATCH.")
            return 0

        print("At least one case MISMATCHED.")
        print("If only the indexed cases mismatch, that strongly suggests an index-path bug.")
        return 2
    except MilvusException as exc:
        print(f"Milvus error: {exc}")
        return 1
    finally:
        try:
            connections.disconnect("default")
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare warmup and index-path behavior for the signed-zero float filter bug."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default="19531")
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--load-wait", type=float, default=2.0)
    args = parser.parse_args()

    raise SystemExit(run_compare(args.host, args.port, args.rows, args.load_wait))


if __name__ == "__main__":
    main()