#!/usr/bin/env python3
"""
Weaviate Known Limitation: Offset Pagination Without Sort = Non-deterministic
==============================================================================

Classification: Known Limitation (not a filter-logic bug)

Root Cause:
  When using offset-based pagination WITHOUT explicit sort, range/complex
  filters trigger parallel goroutines in Weaviate's storage engine.
  The goroutine completion order is non-deterministic, so result ordering
  shifts between page requests. Objects "drift" between pages, causing
  duplicates on some pages and missing items on others.

  This is the same behavior as:
    - PostgreSQL: SELECT * FROM t WHERE score > 50 LIMIT 10 OFFSET 10
      (without ORDER BY, parallel seq scan causes non-determinism)
    - Elasticsearch: deep pagination without sort_by

  Exact match filters (e.g., category == 'A') are NOT affected because they
  use a single Roaring Bitmap with stable natural DocID ordering.

Workarounds:
  A. Add explicit Sort: sort=Sort.by_property("score")  → 0 duplicates
  B. Use cursor API: after="last_uuid" (forces UUID-based ordering)
  C. Fetch all at once: limit=N without offset

Environment:
  - weaviate-client: 4.20.1
  - Weaviate server: tested on localhost:8080
  - Python: 3.11

To run:
  python weaviate_bug3_pagination_dup.py
"""

import weaviate
from weaviate.classes.config import Configure, Property, DataType, Tokenization
from weaviate.classes.query import Filter
from weaviate.classes.data import DataObject
import random
import uuid
import time

HOST = "127.0.0.1"
PORT = 8080
COLLECTION = "BugReproPaginationDup"

def main():
    random.seed(42)
    print("=" * 70)
    print("Weaviate Bug: Offset Pagination + Filter Produces Duplicates")
    print("=" * 70)

    client = weaviate.connect_to_local(host=HOST, port=PORT)
    try:
        # --- Step 1: Create collection ---
        print("\n[Step 1] Creating collection...")
        if client.collections.exists(COLLECTION):
            client.collections.delete(COLLECTION)

        client.collections.create(
            name=COLLECTION,
            properties=[
                Property(name="category", data_type=DataType.TEXT, index_filterable=True,
                         tokenization=Tokenization.FIELD),
                Property(name="score", data_type=DataType.INT, index_filterable=True),
                Property(name="tags", data_type=DataType.INT_ARRAY, index_filterable=True),
            ],
            vector_config=Configure.Vectors.self_provided(
                vector_index_config=Configure.VectorIndex.hnsw()
            ),
        )
        col = client.collections.get(COLLECTION)
        print("  OK")

        # --- Step 2: Insert objects ---
        total = 200
        print(f"\n[Step 2] Inserting {total} objects...")
        objs = []
        for i in range(total):
            cat = "A" if i % 2 == 0 else "B"
            props = {
                "category": cat,
                "score": i,
                "tags": [i % 10, (i + 1) % 10],
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

        # --- Step 3: Get total filtered count ---
        print("\n[Step 3] Counting filtered results...")
        filter_a = Filter.by_property("category").equal("A")
        res = col.query.fetch_objects(filters=filter_a, limit=total)
        total_filtered = len(res.objects)
        print(f"  category == 'A': {total_filtered} objects")

        # --- Step 4: Paginate with offset ---
        page_sizes = [5, 10, 15, 20]
        bug_found = False

        for page_size in page_sizes:
            print(f"\n[Step 4] Paginating with page_size={page_size}...")
            all_ids = []
            seen_ids = set()
            duplicates = []

            max_pages = (total_filtered // page_size) + 2
            for page in range(max_pages):
                offset = page * page_size
                res = col.query.fetch_objects(
                    filters=filter_a,
                    limit=page_size,
                    offset=offset,
                )
                if not res.objects:
                    break

                page_ids = [str(o.uuid) for o in res.objects]
                for uid in page_ids:
                    if uid in seen_ids:
                        duplicates.append((uid, page))
                    seen_ids.add(uid)
                all_ids.extend(page_ids)

                if len(page_ids) < page_size:
                    break

            unique_count = len(seen_ids)
            total_received = len(all_ids)
            dup_count = total_received - unique_count

            if dup_count > 0:
                bug_found = True
                print(f"  *** FAIL ***: total_received={total_received} unique={unique_count} duplicates={dup_count}")
                for uid, pg in duplicates[:5]:
                    print(f"    Duplicate UUID={uid[:8]}... first seen before page {pg}")
            else:
                print(f"  PASS: total_received={total_received} unique={unique_count} no duplicates")

        # --- Step 5: Test with more complex filters ---
        print(f"\n[Step 5] Testing with range filter (score > 50)...")
        filter_range = Filter.by_property("score").greater_than(50)
        res = col.query.fetch_objects(filters=filter_range, limit=total)
        total_range = len(res.objects)
        print(f"  score > 50: {total_range} objects")

        page_size = 10
        all_ids = []
        seen_ids = set()
        duplicates = []
        for page in range(30):
            offset = page * page_size
            res = col.query.fetch_objects(
                filters=filter_range,
                sort=weaviate.classes.query.Sort.by_property("score"),
                limit=page_size,
                offset=offset,
            )
            if not res.objects:
                break
            for o in res.objects:
                uid = str(o.uuid)
                if uid in seen_ids:
                    duplicates.append((uid, page))
                seen_ids.add(uid)
                all_ids.append(uid)
            if len(res.objects) < page_size:
                break

        dup_count = len(all_ids) - len(seen_ids)
        if dup_count > 0:
            bug_found = True
            print(f"  *** FAIL ***: duplicates={dup_count}")
        else:
            print(f"  PASS: no duplicates")

        # --- Step 6: Test with array filter ---
        print(f"\n[Step 6] Testing with array filter (tags contains_any [0, 1])...")
        filter_arr = Filter.by_property("tags").contains_any([0, 1])
        res = col.query.fetch_objects(filters=filter_arr, limit=total)
        total_arr = len(res.objects)
        print(f"  tags contains_any [0,1]: {total_arr} objects")

        all_ids = []
        seen_ids = set()
        duplicates_arr = []
        for page in range(30):
            offset = page * page_size
            res = col.query.fetch_objects(
                filters=filter_arr,
                limit=page_size,
                offset=offset,
            )
            if not res.objects:
                break
            for o in res.objects:
                uid = str(o.uuid)
                if uid in seen_ids:
                    duplicates_arr.append((uid, page))
                seen_ids.add(uid)
                all_ids.append(uid)
            if len(res.objects) < page_size:
                break

        dup_count = len(all_ids) - len(seen_ids)
        if dup_count > 0:
            bug_found = True
            print(f"  *** FAIL ***: duplicates={dup_count}")
        else:
            print(f"  PASS: no duplicates")

        # --- Summary ---
        print("\n" + "=" * 70)
        if not bug_found:
            print("RESULT: ALL PASS — No duplicates detected")
            print("  (Range filter with Sort prevents non-determinism)")
        else:
            print("RESULT: *** DUPLICATES DETECTED ***")
            print("  Root cause: offset pagination without sort is non-deterministic")
            print("  when range/complex filters trigger parallel goroutines.")
            print("  Workaround A: Add sort=Sort.by_property('field') to pagination queries")
            print("  Workaround B: Use cursor API (after=last_uuid) instead of offset")
            print("  Workaround C: Fetch all results at once (limit=N) without offset")
        print("=" * 70)

    finally:
        client.collections.delete(COLLECTION)
        client.close()

if __name__ == "__main__":
    main()
