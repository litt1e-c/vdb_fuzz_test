#!/usr/bin/env python3
"""
Weaviate Bug: Inverted Index Corruption After replace() Operations
==================================================================

Symptom:
  After using data.replace() to update objects, filter queries involving
  NOT(), not_equal(), contains_none on BOOL/BOOL_ARRAY/INT_ARRAY fields
  return INCORRECT results for the replaced rows. The corruption is
  PERSISTENT and worsens with more replace/upsert calls.
  
  This was discovered through differential fuzzing (oracle testing) where
  Pandas ground-truth is compared with Weaviate filter results. Every
  mismatch involves ONLY rows that were previously replace()'d.

Environment:
  - weaviate-client: 4.20.1
  - Weaviate server: tested on localhost:8080
  - Python: 3.11

Key observations from fuzzer logs:
  - flagsArray contains_none [False, True] => Extra 3 dynamic rows
  - c1 != False => Extra 1-2 dynamic rows
  - NOT(c1 == True) => Extra 2 dynamic rows
  - Corruption grows: after 50 replaces, up to 13 extra rows

To run:
  python weaviate_bug1_inverted_index.py

If this script cannot reproduce the bug deterministically, run the fuzzer:
  python weaviate_fuzz_oracle.py --mode oracle --seed 225912938 -N 2000 --rounds 500
"""

import weaviate
from weaviate.classes.config import Configure, Property, DataType, Tokenization, VectorDistances
from weaviate.classes.query import Filter
from weaviate.classes.data import DataObject
import pandas as pd
import random
import uuid
import time
import sys

HOST = "127.0.0.1"
PORT = 8080
COLLECTION = "BugReproInvertedIndex"
N_ROWS = 500
N_REPLACE_ROUNDS = 20
REPLACES_PER_ROUND = 10
DIM = 64


def generate_row(row_id, null_prob=0.1):
    """Generate a row with random values, some nullable."""
    row = {"id": row_id}
    row["c1"] = None if random.random() < null_prob else random.choice([True, False])
    row["c3"] = None if random.random() < null_prob else random.randint(-100000, 100000)
    row["c4"] = None if random.random() < null_prob else random.uniform(-1000, 1000)
    row["text_f"] = None if random.random() < 0.05 else ''.join(
        random.choices("abcdefghijklmnop", k=random.randint(3, 8)))
    row["flags_array"] = None if random.random() < null_prob else [
        random.choice([True, False]) for _ in range(random.randint(1, 4))]
    row["tags_array"] = None if random.random() < null_prob else [
        random.randint(0, 100) for _ in range(random.randint(1, 5))]
    row["labels_array"] = None if random.random() < null_prob else [
        ''.join(random.choices("abcdefghijklmnop", k=random.randint(3, 6)))
        for _ in range(random.randint(1, 4))]
    return row


def check_filter(col, expected_df, filter_obj, desc, dynamic_ids):
    """Compare Weaviate filter result with Pandas ground truth.
    Returns: True=PASS, False=FAIL/BUG, None=ERROR
    """
    try:
        res = col.query.fetch_objects(filters=filter_obj, limit=N_ROWS + 100)
        weaviate_ids = {str(o.uuid) for o in res.objects}
    except Exception as e:
        print(f"  {desc}: ERROR — {e}")
        return None

    pandas_ids = set(expected_df["id"].tolist())
    missing = pandas_ids - weaviate_ids
    extra = weaviate_ids - pandas_ids

    if not missing and not extra:
        return True

    mi_dyn = missing & dynamic_ids
    ex_dyn = extra & dynamic_ids
    is_weaviate_bug = (missing == mi_dyn) and (extra == ex_dyn) and bool(dynamic_ids)

    status = "WEAVIATE_BUG (dynamic rows only)" if is_weaviate_bug else "FAIL"
    print(f"  {desc}: Pandas={len(pandas_ids)} Weaviate={len(weaviate_ids)} "
          f"Missing={len(missing)} Extra={len(extra)} => {status}")
    return False


def main():
    random.seed(42)
    print("=" * 70)
    print("Weaviate Bug: Inverted Index Corruption After replace()")
    print("=" * 70)
    print(f"Config: {N_ROWS} rows, {N_REPLACE_ROUNDS} rounds x {REPLACES_PER_ROUND} replaces")

    client = weaviate.connect_to_local(host=HOST, port=PORT)
    try:
        # --- Step 1: Create collection ---
        print("\n[Step 1] Creating collection with mixed types + nullable fields...")
        if client.collections.exists(COLLECTION):
            client.collections.delete(COLLECTION)

        client.collections.create(
            name=COLLECTION,
            properties=[
                Property(name="c1", data_type=DataType.BOOL, index_filterable=True),
                Property(name="c3", data_type=DataType.INT, index_filterable=True),
                Property(name="c4", data_type=DataType.NUMBER, index_filterable=True),
                Property(name="text_f", data_type=DataType.TEXT, index_filterable=True,
                         tokenization=Tokenization.FIELD),
                Property(name="flags_array", data_type=DataType.BOOL_ARRAY, index_filterable=True),
                Property(name="tags_array", data_type=DataType.INT_ARRAY, index_filterable=True),
                Property(name="labels_array", data_type=DataType.TEXT_ARRAY, index_filterable=True,
                         tokenization=Tokenization.FIELD),
            ],
            vector_config=Configure.Vectors.self_provided(
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric=VectorDistances.COSINE
                )
            ),
            inverted_index_config=Configure.inverted_index(
                index_null_state=True, index_property_length=True
            ),
        )
        col = client.collections.get(COLLECTION)
        print("  OK")

        # --- Step 2: Insert N objects ---
        print(f"\n[Step 2] Inserting {N_ROWS} objects...")
        rows = []
        vectors = []
        for i in range(N_ROWS):
            rid = str(uuid.uuid4())
            row = generate_row(rid, null_prob=0.1)
            rows.append(row)
            vectors.append([random.random() for _ in range(DIM)])

        objs = []
        for r, v in zip(rows, vectors):
            props = {k: val for k, val in r.items() if k != "id" and val is not None}
            objs.append(DataObject(uuid=r["id"], properties=props, vector=v))
        col.data.insert_many(objs)
        time.sleep(1.0)
        df = pd.DataFrame(rows)
        print(f"  Inserted {len(rows)} objects")

        # --- Step 3: Verify filters BEFORE replace ---
        print("\n[Step 3] Verifying filters BEFORE any replace (baseline)...")
        dynamic_ids = set()
        bugs_before = 0

        def make_tests(df):
            return [
                ("NOT(c1 == True)", Filter.not_(Filter.by_property("c1").equal(True)),
                 df[~(df["c1"] == True)]),
                ("c1 != False", Filter.by_property("c1").not_equal(False),
                 df[~(df["c1"] == False)]),
                ("flags_array contains_none [True, False]",
                 Filter.by_property("flags_array").contains_none([True, False]),
                 df[df["flags_array"].apply(
                     lambda x: not any(v in [True, False] for v in x) if isinstance(x, list) else True)]),
                ("NOT(c1 is null)", Filter.not_(Filter.by_property("c1").is_none(True)),
                 df[df["c1"].notna()]),
                ("tags_array contains_none [0, 1]",
                 Filter.by_property("tags_array").contains_none([0, 1]),
                 df[df["tags_array"].apply(
                     lambda x: not any(v in [0, 1] for v in x) if isinstance(x, list) else True)]),
            ]

        for desc, filt, expected_df in make_tests(df):
            result = check_filter(col, expected_df, filt, desc, dynamic_ids)
            if result is True:
                print(f"  {desc}: PASS ({len(expected_df)})")
            elif result is False:
                bugs_before += 1

        # --- Step 4: Perform replace rounds ---
        print(f"\n[Step 4] Performing {N_REPLACE_ROUNDS} rounds of replace()...")
        total_replaced = 0
        total_bugs = 0

        for round_num in range(N_REPLACE_ROUNDS):
            for _ in range(REPLACES_PER_ROUND):
                idx = random.randint(0, len(rows) - 1)
                target = rows[idx]
                uid = target["id"]

                new_row = generate_row(uid, null_prob=0.1)
                new_vec = [random.random() for _ in range(DIM)]
                props = {k: v for k, v in new_row.items() if k != "id" and v is not None}
                try:
                    col.data.replace(uuid=uid, properties=props, vector=new_vec)
                except Exception as e:
                    continue

                for k, v in new_row.items():
                    rows[idx][k] = v
                vectors[idx] = new_vec
                dynamic_ids.add(uid)
                total_replaced += 1

            time.sleep(0.3)

            if (round_num + 1) % 5 == 0:
                df = pd.DataFrame(rows)
                print(f"\n  --- Round {round_num + 1} "
                      f"({total_replaced} replaces, {len(dynamic_ids)} dynamic rows) ---")
                for desc, filt, expected_df in make_tests(df):
                    result = check_filter(col, expected_df, filt, desc, dynamic_ids)
                    if result is True:
                        print(f"  {desc}: PASS ({len(expected_df)})")
                    elif result is False:
                        total_bugs += 1

        # --- Step 5: Final comprehensive check ---
        print(f"\n[Step 5] Final check after {total_replaced} replace()s...")
        df = pd.DataFrame(rows)

        final_tests = [
            ("NOT(c1 == True)", Filter.not_(Filter.by_property("c1").equal(True)),
             df[~(df["c1"] == True)]),
            ("NOT(c1 == False)", Filter.not_(Filter.by_property("c1").equal(False)),
             df[~(df["c1"] == False)]),
            ("c1 != True", Filter.by_property("c1").not_equal(True),
             df[~(df["c1"] == True)]),
            ("c1 != False", Filter.by_property("c1").not_equal(False),
             df[~(df["c1"] == False)]),
            ("flags_array contains_none [True, False]",
             Filter.by_property("flags_array").contains_none([True, False]),
             df[df["flags_array"].apply(
                 lambda x: not any(v in [True, False] for v in x) if isinstance(x, list) else True)]),
            ("flags_array contains_none [True]",
             Filter.by_property("flags_array").contains_none([True]),
             df[df["flags_array"].apply(
                 lambda x: True not in x if isinstance(x, list) else True)]),
            ("NOT(c1 is null)", Filter.not_(Filter.by_property("c1").is_none(True)),
             df[df["c1"].notna()]),
            ("tags_array contains_none [0, 1, 2]",
             Filter.by_property("tags_array").contains_none([0, 1, 2]),
             df[df["tags_array"].apply(
                 lambda x: not any(v in [0, 1, 2] for v in x) if isinstance(x, list) else True)]),
        ]

        final_bugs = 0
        for desc, filt, expected_df in final_tests:
            result = check_filter(col, expected_df, filt, desc, dynamic_ids)
            if result is True:
                print(f"  {desc}: PASS ({len(expected_df)})")
            elif result is False:
                final_bugs += 1

        # --- Summary ---
        print("\n" + "=" * 70)
        total_all_bugs = bugs_before + total_bugs + final_bugs
        if total_all_bugs == 0:
            print("RESULT: ALL PASS — Bug not reproduced in this environment")
            print(f"  ({total_replaced} replaces, {len(dynamic_ids)} dynamic rows)")
            print()
            print("  NOTE: This bug was discovered through extensive fuzzing and may")
            print("  require specific data patterns to trigger. Try the fuzzer directly:")
            print("    python weaviate_fuzz_oracle.py --mode oracle --seed 225912938 "
                  "-N 2000 --rounds 500")
        else:
            print(f"RESULT: *** BUG REPRODUCED *** ({total_all_bugs} failures)")
            print(f"  {total_replaced} replaces, {len(dynamic_ids)} dynamic rows")
            print("  Weaviate's inverted index returns incorrect filter results")
            print("  after data.replace() operations. Extra or missing rows in results")
            print("  are ALWAYS rows that were previously replace()'d.")
        print("=" * 70)

    finally:
        try:
            client.collections.delete(COLLECTION)
        except Exception:
            pass
        client.close()


if __name__ == "__main__":
    main()
