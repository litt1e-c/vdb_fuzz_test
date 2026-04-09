import random
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

def reproduce_null_occupancy_bug():
    connections.connect(host="localhost", port="19531")
    collection_name = "null_bug_repro"
    
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
    
    # Enable dynamic schema
    schema = CollectionSchema([
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128)
    ], enable_dynamic_field=True)
    collection = Collection(collection_name, schema)
    collection.create_index(
        "embedding",
        {
            "index_type": "HNSW",
            "metric_type": "L2",
            "params": {"M": 8, "efConstruction": 200}
        }
    )
    collection.load()

    print("\n" + "="*60)
    print("🧪 Test: Invalid group (None) occupies slots of valid groups")
    print("="*60)

    # 1. Insert interfering data:
    #    Very close to the query center (distance ~0.01),
    #    but missing the 'category' field.
    #    These records will rank at the top during search.
    garbage_data = [
        {"id": i, "embedding": [0.01] * 128}  # Intentionally omit the 'category' field
        for i in range(50)
    ]
    
    # 2. Insert real business data:
    #    Slightly farther from the center (distance ~0.5),
    #    with valid 'category' values.
    real_data = [
        {"id": 100, "embedding": [0.5] * 128, "category": "Electronics"},
        {"id": 101, "embedding": [0.5] * 128, "category": "Books"}
    ]
    
    collection.insert(garbage_data + real_data)
    collection.flush()

    # 3. Perform search: limit=1, group_size=1
    #
    # Expected behavior:
    #   The user expects to get the highest-ranked valid category (e.g., "Electronics").
    #
    # Actual behavior:
    #   Because the garbage records rank first and are grouped as `None`,
    #   the group quota is consumed by the `None` group,
    #   causing valid groups to be excluded.
    res = collection.search(
        data=[[0.0] * 128],
        anns_field="embedding",
        param={"metric_type": "L2"},
        limit=1,
        group_by_field="category",
        output_fields=["category"]
    )

    print(f"Number of returned groups: {len(res[0])}")
    for hit in res[0]:
        cat = hit.entity.get("category")
        print(f"Hit ID: {hit.id}, group key (category): {cat}")
        if cat is None:
            print("🔴 Bug confirmed: valid groups are displaced by None. "
                  "The system incorrectly aggregates null values.")
        else:
            print("✅ Successfully retrieved a valid group.")

if __name__ == "__main__":
    reproduce_null_occupancy_bug()