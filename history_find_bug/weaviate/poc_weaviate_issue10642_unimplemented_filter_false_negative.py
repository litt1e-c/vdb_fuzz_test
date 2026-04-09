import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import Filter
from weaviate.classes.data import DataObject
import uuid

client = weaviate.connect_to_local()

# 0. 清理可能已存在的同名集合，避免创建冲突
try:
    client.collections.delete("BugReproObjectIsNone")
    print("[*] 已删除已存在的集合 BugReproObjectIsNone")
except Exception:
    pass  # 如果集合不存在，忽略错误

# 1. Create Collection
client.collections.create(
    name="BugReproObjectIsNone",
    properties=[
        Property(
            name="metadata",
            data_type=DataType.OBJECT,
            index_filterable=True,
            index_null_state=True,
            nested_properties=[Property(name="price", data_type=DataType.INT)],
        ),
    ],
    inverted_index_config=Configure.inverted_index(index_null_state=True),
)
col = client.collections.get("BugReproObjectIsNone")

# 2. Insert 2 objects: one with OBJECT, one with explicit null
# 修复：第二个对象显式设置 metadata=None，使其成为 null 字段，
# 而不是缺失字段。这样 .is_none(False) 才能正确排除它。
col.data.insert_many([
    DataObject(properties={"metadata": {"price": 100}}, uuid=str(uuid.uuid4())),
    DataObject(properties={"metadata": None}, uuid=str(uuid.uuid4()))  # 显式 null
])

# 3. Query is_none(False) — 匹配 metadata 字段不为 null 的对象
res = col.query.fetch_objects(filters=Filter.by_property("metadata").is_none(False))

print(f"Expected: 1, Actual: {len(res.objects)}")  # 现在输出 1

# 清理
client.collections.delete("BugReproObjectIsNone")
client.close()