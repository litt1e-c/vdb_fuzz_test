import time
import random
import numpy as np
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from milvus_fuzz_oracle import DataManager, DIM

# Repro script for Case4679 without touching the existing collection.
# It builds an isolated temp collection, inserts only the target row, runs the query,
# reports whether the unexpected row is returned, and then drops the temp collection.

SEED = 999
HOST = "127.0.0.1"
PORT = "19531"
TARGET_ID = 4526
COLLECTION_PREFIX = "logic_bug_repro"
DROP_AFTER = True  # set to False if you want to inspect the temp collection

# Query that should exclude rows where c0 is NULL but currently returns the row.
EXPR = """(((meta_json[\"active\"] == true and meta_json[\"color\"] == \"Blue\") or ((c16 >= 2538.641880850278 or ((meta_json[\"active\"] == true and meta_json[\"color\"] == \"Blue\") and (meta_json[\"price\"] > 107 and meta_json[\"price\"] < 261))) and ((c0 > 2146.802199098024 and c2 >= 36295) and (c12 == false and (meta_json[\"active\"] == true and meta_json[\"color\"] == \"Blue\"))))) and (c17 == false or (((c10 < \"IHSVh\" and c5 == false) or (c0 <= 4.787315444874056 or (meta_json[\"price\"] > 293 and meta_json[\"price\"] < 453))) and ((meta_json[\"config\"][\"version\"] == 3 or c17 is null) and (c11 > -51460 and c9 != 105382.9375)))))"""


def build_temp_collection(dm, name):
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=DIM),
    ]

    for fc in dm.schema_config:
        if fc["type"] == DataType.ARRAY:
            fields.append(FieldSchema(
                name=fc["name"],
                dtype=DataType.ARRAY,
                element_type=fc["element_type"],
                max_capacity=fc["max_capacity"],
                nullable=True,
            ))
        else:
            fields.append(FieldSchema(
                name=fc["name"],
                dtype=fc["type"],
                nullable=True,
                max_length=512,
            ))

    schema = CollectionSchema(fields, enable_dynamic_field=True)

    if utility.has_collection(name):
        utility.drop_collection(name)

    return Collection(name, schema)


def main():
    print("\n=== Minimal Repro for Case4679 (isolated collection) ===")
    random.seed(SEED)
    np.random.seed(SEED)

    dm = DataManager()
    dm.generate_schema()
    dm.generate_data()

    if TARGET_ID >= len(dm.df):
        raise ValueError(f"TARGET_ID {TARGET_ID} not in generated data")

    row = dm.df[dm.df["id"] == TARGET_ID].iloc[0].to_dict()
    vector = dm.vectors[TARGET_ID].tolist()

    record = {**row, "vector": vector}
    for k, v in record.items():
        if hasattr(v, "item"):
            record[k] = v.item()

    connections.connect("default", host=HOST, port=PORT)

    temp_name = f"{COLLECTION_PREFIX}_{int(time.time())}"
    col = build_temp_collection(dm, temp_name)
    col.insert([record])
    col.flush()
    col.load()

    res = col.query(EXPR, output_fields=["id", "c0"], limit=128, consistency_level="Strong")
    returned_ids = {r["id"] for r in res}

    print(f"Temp collection: {temp_name}")
    print(f"Row c0 value (ground truth from DataFrame): {row['c0']}")
    print(f"Query should exclude NULL/None c0. Returned count: {len(res)}")

    if TARGET_ID in returned_ids:
        print("-> BUG: target row with c0=None was returned")
    else:
        print("-> OK: target row was not returned (bug not reproduced)")

    if DROP_AFTER:
        utility.drop_collection(temp_name)
        print("Temp collection dropped (DROP_AFTER=True)")
    else:
        print("Temp collection kept for inspection (DROP_AFTER=False)")


if __name__ == "__main__":
    main()
