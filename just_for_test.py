import random
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility
)

# 1. 连接 Milvus
print("Connecting to Milvus on port 19531...")
connections.connect(alias="default", host="localhost", port="19531")

collection_name = "test_offset_mode_validity_bug"
dim = 8

# 2. 定义 Schema
# 必须包含 Nullable 的 Array
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
    FieldSchema(name="tags", dtype=DataType.ARRAY, element_type=DataType.INT64, max_capacity=10, nullable=True)
]

schema = CollectionSchema(fields, description="Reproduce Offset Mode Validity Bug")

if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)

collection = Collection(name=collection_name, schema=schema)

# 3. 准备数据
# 2000 行
# Row 0: tags = None (Null)
# Row 1-1999: tags = [1, 2, 3]
count = 2000
ids = list(range(count))
vectors = [[0.1] * dim for _ in range(count)]
tags = []
tags.append(None) # Row 0 is Null
for i in range(1, count):
    tags.append([1, 2, 3])

data = [ids, vectors, tags]

print(f"Inserting {count} rows (Row 0 is Null)...")
collection.insert(data)
collection.flush()

print("Creating Index...")
index_params = {
    "metric_type": "L2",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 32}
}
collection.create_index("vector", index_params)
collection.load()

# 4. 执行触发 Bug 的查询
# 关键组合：
# 1. id >= 0: 选中所有行，迫使 Miss Ratio = 0，触发 Offset Mode。
# 2. array_length(tags) == 0: 
#    Offset Mode 会忽略 Row 0 的 Null 标记，读取到底层默认的空数组，导致 length==0 成立。
expr = "id >= 0 and array_length(tags) == 0"

print(f"\nExecuting Query: {expr}")
results = collection.query(
    expr=expr,
    output_fields=["id", "tags"]
)

# 5. 验证结果
print("-" * 50)
print(f"Result count: {len(results)}")

bug_reproduced = False
for res in results:
    if res['id'] == 0:
        bug_reproduced = True
        print(f"❌ BUG REPRODUCED: Found Row 0! Tags: {res['tags']}")
        print("Reason: Offset Mode ignored the 'validity' bitmap. It treated the Null array as an empty array (length=0).")
        break

if not bug_reproduced:
    print("✅ BEHAVIOR CORRECT: Row 0 was filtered out.")
    if len(results) > 0:
        print(f"Note: Found other rows: {[r['id'] for r in results]}")
else:
    print("-" * 50)
    print("Code Proof:")
    print("In 'ElementFilterBitsNode.cpp', the 'offset_mode' block calls 'RowBitsetToElementOffsets'.")
    print("It uses 'doc_bitset' (data) but completely ignores 'doc_bitset_valid' (validity).")
    print("So Null rows (which have valid=0) are processed as if they are valid.")

# collection.drop()