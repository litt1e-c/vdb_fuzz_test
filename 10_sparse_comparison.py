import time
import random
import numpy as np
from abc import ABC, abstractmethod

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from qdrant_client import QdrantClient, models
import weaviate


# ================================
# 🔥 全局配置
# ================================
HOST = '127.0.0.1'

NUM_ENTITIES = 20000       # 必须放大! 小数据量不会触发 HNSW 随机性
TEST_REPEAT = 5            # 每个数据库测 5 次，增强触发概率
SPARSE_DIM = 2048

# 生成一个固定的稀疏向量
SPARSE_DICT = {128: 1.0, 512: 0.8, 1024: 0.5}
MAX_DIM = SPARSE_DIM

# 输出颜色
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'


# ================================
# 🔥 抽象基类
# ================================
class Adapter(ABC):
    name = "Unknown"

    @abstractmethod
    def setup(self): pass

    @abstractmethod
    def insert_clones(self): pass

    @abstractmethod
    def search_pagination(self, limit, offset): pass


# ================================
# 🔥 Milvus Adapter
# ================================
class MilvusAdapter(Adapter):
    def __init__(self):
        self.name = "Milvus"
        self.col_name = "tie_test_milvus"

    def setup(self):
        connections.connect("default", host=HOST, port="19530")
        if utility.has_collection(self.col_name):
            utility.drop_collection(self.col_name)

        schema = CollectionSchema([
            FieldSchema("id", DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema("vec", DataType.SPARSE_FLOAT_VECTOR),
        ])

        self.col = Collection(self.col_name, schema)
        print("Creating Milvus index...")
        self.col.create_index("vec",
            {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP", "params": {}}
        )

    def insert_clones(self):
        ids = list(range(NUM_ENTITIES))
        random.shuffle(ids)  # 🔥 打乱插入顺序，增强随机性

        vecs = [SPARSE_DICT] * NUM_ENTITIES
        self.col.insert([ids, vecs])
        self.col.flush()
        self.col.load()

    def search_pagination(self, limit, offset):
        res = self.col.search(
            data=[SPARSE_DICT], anns_field="vec",
            param={"metric_type": "IP"}, limit=limit, offset=offset,
            output_fields=["id"]
        )
        return [hit.id for hit in res[0]]


# ================================
# 🔥 Qdrant Adapter
# ================================
class QdrantAdapter(Adapter):
    def __init__(self):
        self.name = "Qdrant"
        self.col_name = "tie_test_qdrant"
        self.client = QdrantClient(url=f"http://{HOST}:6333")

    def setup(self):
        if self.client.collection_exists(self.col_name):
            self.client.delete_collection(self.col_name)

        print("Creating Qdrant collection...")
        self.client.create_collection(
            self.col_name,
            vectors_config={},
            sparse_vectors_config={
                "vec": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False)
                )
            }
        )

    def insert_clones(self):
        idxs = list(range(NUM_ENTITIES))
        random.shuffle(idxs)  # 🔥 HNSW 强依赖顺序，打乱非常关键

        indices = list(SPARSE_DICT.keys())
        values = list(SPARSE_DICT.values())

        batch = []
        for i in idxs:
            batch.append(models.PointStruct(
                id=i,
                vector={"vec": models.SparseVector(indices=indices, values=values)}
            ))

            if len(batch) == 500:
                self.client.upsert(self.col_name, batch)
                batch = []

        if batch:
            self.client.upsert(self.col_name, batch)

    def search_pagination(self, limit, offset):
        indices = list(SPARSE_DICT.keys())
        values = list(SPARSE_DICT.values())

        res = self.client.search(
            self.col_name,
            query_vector=models.NamedSparseVector(
                name="vec",
                vector=models.SparseVector(indices=indices, values=values)
            ),
            limit=limit,
            offset=offset
        )
        return [r.id for r in res]


# ================================
# 🔥 Weaviate Adapter
# ================================
class WeaviateAdapter(Adapter):
    def __init__(self):
        self.name = "Weaviate"
        self.col_name = "TieTestWeaviate"
        self.client = weaviate.Client(f"http://{HOST}:8080")

        # 转成稠密
        self.vec = [0.0] * MAX_DIM
        for k, v in SPARSE_DICT.items():
            self.vec[k] = v

    def setup(self):
        self.client.schema.delete_all()
        time.sleep(0.5)

        self.client.schema.create_class({
            "class": self.col_name, "vectorizer": "none",
            "properties": [{"name": "idx", "dataType": ["int"]}]
        })

    def insert_clones(self):
        order = list(range(NUM_ENTITIES))
        random.shuffle(order)

        with self.client.batch as batch:
            for i in order:
                batch.add_data_object({"idx": i}, self.col_name, vector=self.vec)

    def search_pagination(self, limit, offset):
        response = (self.client.query
            .get(self.col_name, ["idx"])
            .with_near_vector({"vector": self.vec})
            .with_limit(limit)
            .with_offset(offset)
            .do()
        )
        return [x["idx"] for x in response["data"]["Get"][self.col_name]]


# ================================
# 🔥 测试函数
# ================================
def test_adapter(adapter):
    print(f"\n\n==================== TESTING {adapter.name} ====================")

    for rep in range(TEST_REPEAT):
        print(f"\n--- RUN {rep+1}/{TEST_REPEAT} ---")

        adapter.setup()
        adapter.insert_clones()

        p1 = adapter.search_pagination(limit=20, offset=0)
        p2 = adapter.search_pagination(limit=20, offset=10)

        print(f"Page1: {p1}")
        print(f"Page2 (offset 10): {p2}")

        expected = p1[10] if len(p1) > 10 else None
        actual = p2[0] if len(p2) > 0 else None

        print(f"Expected first item of Page2 = Page1[10] = {expected}")
        print(f"Actual   first item of Page2 = {actual}")

        if expected == actual:
            print(GREEN + "PASS: Pagination SEEMS stable" + RESET)
        else:
            print(RED + "FAIL: Pagination NOT stable !!" + RESET)


# ================================
# 🔥 主入口
# ================================
if __name__ == "__main__":
    test_adapter(MilvusAdapter())
    test_adapter(QdrantAdapter())
    test_adapter(WeaviateAdapter())