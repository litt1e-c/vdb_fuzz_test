import datetime
import random
import time
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection

HOST = "127.0.0.1"
PORT = "19531"
COLLECTION_NAME = "test_constant_expressions"
DIM = 4


def run_constant_expression_tests(host=HOST, port=PORT):
    print(f"🔍 Testing constant expressions against Milvus at {host}:{port}")
    print("=" * 60)

    # connect
    connections.connect("default", host=host, port=port)

    # cleanup
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)

    # schema: primary id + vector + json + bool
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=DIM),
        FieldSchema(name="json_field", dtype=DataType.JSON, nullable=True),
        FieldSchema(name="c_bool", dtype=DataType.BOOL, nullable=True),
    ]

    schema = CollectionSchema(fields)
    col = Collection(COLLECTION_NAME, schema)

    # insert sample data
    data = [
        {"id": 1, "vector": [random.random() for _ in range(DIM)], "json_field": {"k": "v"}, "c_bool": True},
        {"id": 2, "vector": [random.random() for _ in range(DIM)], "json_field": {"k": "x"}, "c_bool": False},
        {"id": 3, "vector": [random.random() for _ in range(DIM)], "json_field": None, "c_bool": None},
    ]

    col.insert(data)
    col.flush()
    col.create_index("vector", {"metric_type": "L2", "index_type": "FLAT"})
    col.load()

    print("Inserted records: ids=1,2,3 (id=3 has json_field=None)")

    # test cases: expression, expected ids, description
    test_cases = [
        ("1 < 2", [1, 2, 3], "constant true should return all rows"),
        ("1 > 2", [], "constant false should return no rows"),
        ("1 + 1 == 2", [1, 2, 3], "constant arithmetic true returns all"),
        ("not (1 < 2)", [], "negated constant false returns no rows"),
        ("1 < 2 and id > 1", [2, 3], "constant true combined with predicate"),
        ("1 > 2 or id == 1", [1], "constant false OR predicate"),
        ("id == 1 or 2 == 3", [1], "mixed constant comparison with predicate"),
        ("json_field is null and 1 < 2", [3], "NULL JSON combined with true constant"),
        ("1 < 2 and json_field[\"k\"] is null", [3, 4] if False else [3], "json key null check (missing or null)"),
        ("id < 2", [1], "normal predicate sanity check"),
    ]

    results = []

    for expr, expected, desc in test_cases:
        try:
            print(f"\n🧪 Test: {desc}\n  expr: {expr}\n  expected: {expected}")
            res = col.query(expr, output_fields=["id"])  # may raise if parser rejects
            actual = sorted([r["id"] for r in res])
            ok = actual == expected
            print(f"  actual: {actual} -> {'PASS' if ok else 'FAIL'}")
            results.append({"expr": expr, "expected": expected, "actual": actual, "ok": ok})
        except Exception as e:
            print(f"  ERROR executing expr: {expr}\n    {e}")
            results.append({"expr": expr, "expected": expected, "actual": None, "ok": False, "error": str(e)[:200]})

    # summary
    total = len(results)
    passed = sum(1 for r in results if r.get("ok"))
    failed = total - passed

    print("\n" + "=" * 60)
    print(f"SUMMARY: total={total}, passed={passed}, failed={failed}")

    # save report
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"const_expr_test_report_{ts}.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write("Constant Expression Test Report\n")
        f.write(f"Time: {ts}\n\n")
        for r in results:
            f.write(f"Expr: {r['expr']}\n")
            f.write(f"  Expected: {r['expected']}\n")
            f.write(f"  Actual: {r['actual']}\n")
            if r.get("error"):
                f.write(f"  ERROR: {r['error']}\n")
            f.write("\n")

    print(f"Report saved to {fname}")

    # cleanup
    utility.drop_collection(COLLECTION_NAME)


if __name__ == '__main__':
    run_constant_expression_tests()