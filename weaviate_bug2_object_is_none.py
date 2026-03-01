#!/usr/bin/env python3
"""
Weaviate Bug: OBJECT is_none(False) Returns Empty Results
==========================================================

Symptom:
  When using Filter.by_property("obj_field").is_none(False) to find objects
  where a nested OBJECT field is NOT null, Weaviate returns 0 results even
  though the majority of objects have non-null values for that field.
  is_none(True) works correctly.

Environment:
  - weaviate-client: 4.20.1
  - Weaviate server: tested on localhost:8080
  - Python: 3.11

Steps to Reproduce:
  1. Create a collection with a nested OBJECT property
  2. Insert objects — some with OBJECT value, some with null
  3. Query is_none(True) — returns correct null count
  4. Query is_none(False) — returns 0 instead of non-null count

Expected vs Actual:
  - is_none(True): Expected N null rows, Got N null rows (CORRECT)
  - is_none(False): Expected M non-null rows, Got 0 rows (WRONG)

To run:
  python weaviate_bug2_object_is_none.py
"""

import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import Filter
from weaviate.classes.data import DataObject
import random
import uuid
import time

HOST = "127.0.0.1"
PORT = 8080
COLLECTION = "BugReproObjectIsNone"

def main():
    random.seed(42)
    print("=" * 70)
    print("Weaviate Bug: OBJECT is_none(False) Returns Empty Results")
    print("=" * 70)

    client = weaviate.connect_to_local(host=HOST, port=PORT)
    try:
        # --- Step 1: Create collection with nested OBJECT ---
        print("\n[Step 1] Creating collection with nested OBJECT property...")
        if client.collections.exists(COLLECTION):
            client.collections.delete(COLLECTION)

        client.collections.create(
            name=COLLECTION,
            properties=[
                Property(name="name", data_type=DataType.TEXT),
                Property(
                    name="metadata",
                    data_type=DataType.OBJECT,
                    index_filterable=True,
                    index_null_state=True,
                    nested_properties=[
                        Property(name="price", data_type=DataType.INT),
                        Property(name="color", data_type=DataType.TEXT),
                        Property(name="active", data_type=DataType.BOOL),
                    ],
                ),
            ],
            vector_config=Configure.Vectors.self_provided(
                vector_index_config=Configure.VectorIndex.hnsw()
            ),
            inverted_index_config=Configure.inverted_index(
                index_null_state=True, index_property_length=True
            ),
        )
        col = client.collections.get(COLLECTION)
        print("  OK")

        # --- Step 2: Insert objects ---
        total = 50
        null_count = 15
        non_null_count = total - null_count
        print(f"\n[Step 2] Inserting {total} objects ({non_null_count} with OBJECT, {null_count} with null)...")

        objs = []
        for i in range(total):
            props = {"name": f"item_{i}"}
            if i >= non_null_count:
                # null metadata — don't include the key
                pass
            else:
                props["metadata"] = {
                    "price": random.randint(1, 1000),
                    "color": random.choice(["red", "blue", "green"]),
                    "active": random.choice([True, False]),
                }
            vec = [random.random() for _ in range(128)]
            objs.append(DataObject(
                uuid=str(uuid.uuid4()),
                properties=props,
                vector=vec,
            ))
        col.data.insert_many(objs)
        time.sleep(0.5)
        print(f"  Inserted {total} objects")

        # --- Step 3: Test is_none(True) ---
        print("\n[Step 3] Testing is_none(True) — should return null objects...")
        f_null = Filter.by_property("metadata").is_none(True)
        res = col.query.fetch_objects(filters=f_null, limit=100)
        got_null = len(res.objects)
        ok_null = got_null == null_count
        print(f"  is_none(True): got={got_null} expected={null_count} {'PASS' if ok_null else 'FAIL'}")

        # --- Step 4: Test is_none(False) — THE BUG ---
        print("\n[Step 4] Testing is_none(False) — should return non-null objects...")
        f_not_null = Filter.by_property("metadata").is_none(False)
        res = col.query.fetch_objects(filters=f_not_null, limit=100)
        got_not_null = len(res.objects)
        ok_not_null = got_not_null == non_null_count
        print(f"  is_none(False): got={got_not_null} expected={non_null_count} {'PASS' if ok_not_null else '*** FAIL ***'}")
        if not ok_not_null:
            print(f"    BUG: is_none(False) returned {got_not_null} instead of {non_null_count}")
            if got_not_null == 0:
                print(f"    This is likely Weaviate's known OBJECT is_none(False) bug")

        # --- Step 5: Verify with NOT(is_none(True)) as workaround ---
        print("\n[Step 5] Testing NOT(is_none(True)) as workaround...")
        f_workaround = Filter.not_(Filter.by_property("metadata").is_none(True))
        res = col.query.fetch_objects(filters=f_workaround, limit=100)
        got_workaround = len(res.objects)
        ok_workaround = got_workaround == non_null_count
        print(f"  NOT(is_none(True)): got={got_workaround} expected={non_null_count} {'PASS' if ok_workaround else 'FAIL'}")

        # --- Step 6: Also test on other data types for comparison ---
        print("\n[Step 6] Comparison: is_none(False) on TEXT field (should work)...")
        # "name" is always non-null for all 50 objects
        f_text_not_null = Filter.by_property("name").is_none(False)
        res = col.query.fetch_objects(filters=f_text_not_null, limit=100)
        got_text = len(res.objects)
        ok_text = got_text == total
        print(f"  TEXT is_none(False): got={got_text} expected={total} {'PASS' if ok_text else 'FAIL'}")

        # --- Summary ---
        print("\n" + "=" * 70)
        if ok_null and ok_not_null and ok_workaround:
            print("RESULT: ALL PASS — Bug not reproduced in this environment")
        else:
            print("RESULT: *** BUG REPRODUCED ***")
            if not ok_not_null:
                print("  OBJECT is_none(False) returns 0 results when non-null objects exist.")
                print("  This only affects OBJECT (DataType.OBJECT) fields.")
                print(f"  Workaround: NOT(is_none(True)) {'works' if ok_workaround else 'also fails'}")
        print("=" * 70)

    finally:
        client.collections.delete(COLLECTION)
        client.close()

if __name__ == "__main__":
    main()
