import time
import random
import string
import numpy as np
from abc import ABC, abstractmethod

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from qdrant_client import QdrantClient, models
import weaviate

# ==============================================================================
# 🔥 全局高压配置
# ==============================================================================
HOST = '127.0.0.1'
NUM_ENTITIES = 20000      # 数据量
BATCH_SIZE = 2000         # 较小的 Batch，强制产生多个 Segment/Part
SPARSE_DICT = {128: 1.0, 512: 0.8, 1024: 0.5} # 固定向量，确保 Score 100% Tie
MAX_DIM = 2000

# 颜色
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
RESET = '\033[0m'

def generate_ids(n):
    # 生成随机打乱的 ID，确保物理顺序和 ID 顺序无关
    ids = list(range(n))
    random.shuffle(ids)
    return ids

# ==============================================================================
# 🧱 适配器层
# ==============================================================================
class Adapter(ABC):
    def __init__(self): self.name = "Unknown"
    @abstractmethod
    def setup(self): pass
    @abstractmethod
    def insert_data(self, ids): pass
    @abstractmethod
    def search_pages(self, limit, offset): pass

# --- 1. Milvus ---
class MilvusAdapter(Adapter):
    def __init__(self):
        self.name = "Milvus"
        self.col_name = "tie_stress_milvus"
    
    def setup(self):
        try: connections.connect("default", host=HOST, port="19530")
        except: pass
        if utility.has_collection(self.col_name): utility.drop_collection(self.col_name)
        
        # 显式指定 shards_num=2，强制触发 Proxy 层的 Reduce，增加不稳定性风险
        schema = CollectionSchema([
            FieldSchema("id", DataType.INT64, is_primary=True),
            FieldSchema("vec", DataType.SPARSE_FLOAT_VECTOR),
        ])
        self.col = Collection(self.col_name, schema, shards_num=2, consistency_level="Strong")

    def insert_data(self, ids):
        # 分批插入，模拟多 Segment
        for i in range(0, len(ids), BATCH_SIZE):
            batch_ids = ids[i : i+BATCH_SIZE]
            batch_vecs = [SPARSE_DICT] * len(batch_ids)
            self.col.insert([batch_ids, batch_vecs])
            # 甚至可以每插一批 flush 一次，制造碎片
            if i % (BATCH_SIZE*2) == 0: self.col.flush()
            
        self.col.flush()
        self.col.create_index("vec", {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP", "params": {"drop_ratio_build": 0.0}})
        self.col.load()

    def search_pages(self, limit, offset):
        res = self.col.search(
            data=[SPARSE_DICT], anns_field="vec", 
            param={"metric_type": "IP", "params": {"drop_ratio_search": 0.0}}, 
            limit=limit, offset=offset, output_fields=["id"]
        )
        return [h.id for h in res[0]]

# --- 2. Qdrant ---
class QdrantAdapter(Adapter):
    def __init__(self):
        self.name = "Qdrant"
        self.col_name = "tie_stress_qdrant"
        self.client = QdrantClient(url=f"http://{HOST}:6333", timeout=60)

    def setup(self):
        if self.client.collection_exists(self.col_name):
            self.client.delete_collection(self.col_name)
        
        self.client.create_collection(
            self.col_name,
            vectors_config={},
            sparse_vectors_config={
                "vec": models.SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
            },
            # 设置分片数，强制多路归并
            shard_number=2 
        )

    def insert_data(self, ids):
        indices = list(SPARSE_DICT.keys())
        values = list(SPARSE_DICT.values())
        
        for i in range(0, len(ids), BATCH_SIZE):
            batch_ids = ids[i : i+BATCH_SIZE]
            points = [
                models.PointStruct(id=uid, vector={"vec": models.SparseVector(indices=indices, values=values)})
                for uid in batch_ids
            ]
            self.client.upsert(self.col_name, points)

    def search_pages(self, limit, offset):
        indices = list(SPARSE_DICT.keys())
        values = list(SPARSE_DICT.values())
        res = self.client.search(
            self.col_name,
            query_vector=models.NamedSparseVector(name="vec", vector=models.SparseVector(indices=indices, values=values)),
            limit=limit, offset=offset, with_vectors=False
        )
        return [r.id for r in res]

# --- 3. Weaviate ---
class WeaviateAdapter(Adapter):
    def __init__(self):
        self.name = "Weaviate"
        self.col_name = "TieStress"
        self.client = weaviate.Client(f"http://{HOST}:8080")
        self.dense_vec = [0.0] * MAX_DIM
        for k,v in SPARSE_DICT.items(): self.dense_vec[k] = v

    def setup(self):
        self.client.schema.delete_all()
        class_obj = {
            "class": self.col_name, 
            "vectorizer": "none",
            "properties": [{"name": "idx", "dataType": ["int"]}]
        }
        self.client.schema.create_class(class_obj)

    def insert_data(self, ids):
        # Weaviate 不方便手动控制分片，只能依赖其内部逻辑
        # 我们通过小 Batch 插入模拟碎片
        for i in range(0, len(ids), BATCH_SIZE):
            batch_ids = ids[i : i+BATCH_SIZE]
            with self.client.batch as batch:
                batch.batch_size = len(batch_ids)
                for uid in batch_ids:
                    batch.add_data_object({"idx": uid}, self.col_name, vector=self.dense_vec)

    def search_pages(self, limit, offset):
        resp = (self.client.query.get(self.col_name, ["idx"])
                .with_near_vector({"vector": self.dense_vec})
                .with_limit(limit).with_offset(offset)
                .do())
        if 'errors' in resp: return []
        return [x['idx'] for x in resp['data']['Get'][self.col_name]]

# ==============================================================================
# 🧪 核心测试逻辑
# ==============================================================================
def run_stress_test(adapter):
    print(f"\n\n{CYAN}{'='*20} TESTING {adapter.name} {'='*20}{RESET}")
    
    try:
        # 1. 初始化与插入
        print(f"   📥 Inserting {NUM_ENTITIES} items (Shuffled & Fragmented)...")
        shuffled_ids = generate_ids(NUM_ENTITIES)
        
        start_t = time.time()
        adapter.setup()
        adapter.insert_data(shuffled_ids)
        print(f"   ⏱️  Setup Time: {time.time()-start_t:.2f}s")

        # 2. 验证逻辑 (重复 3 次以捕捉随机性)
        for round in range(1, 4):
            print(f"\n   🔄 Round {round}: Checking Stability...")
            
            # Query A: Page 1 (Top 10)
            page1 = adapter.search_pages(limit=10, offset=0)
            
            # Query B: Page 2 (Offset 5)
            # 预期：Page 2 的开头，应该是 Page 1 的第 6 个元素 (Index 5)
            page2 = adapter.search_pages(limit=5, offset=5)
            
            if len(page1) < 10 or len(page2) < 5:
                print(f"      {RED}❌ Failed: Not enough results returned!{RESET}")
                continue

            expected_start = page1[5]
            actual_start = page2[0]
            
            print(f"      Page 1 (0-10): {page1}")
            print(f"      Page 2 (5-10): {page2}")
            print(f"      Target ID: {expected_start} vs Actual: {actual_start}")
            
            if expected_start == actual_start:
                # 进阶检查：是否排序？
                is_sorted = (page1 == sorted(page1))
                tag = "Sorted" if is_sorted else "Unsorted"
                print(f"      {GREEN}✅ PASS ({tag}){RESET}")
            else:
                print(f"      {RED}❌ FAIL: Inconsistent! Offset logic broken.{RESET}")
                
                # 检查是不是随机抖动：检查 Page2[0] 是否在 Page1 里出现过
                if actual_start in page1[:5]:
                    print(f"         ➡️ Reason: {YELLOW}Re-ordering detected!{RESET} Items jumped across page boundary.")
                else:
                    print(f"         ➡️ Reason: Unknown drift.")

    except Exception as e:
        print(f"   {RED}❌ CRASHED: {e}{RESET}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 1. Milvus (修复了 Bug 的版本)
    run_stress_test(MilvusAdapter())
    
    # 2. Qdrant (内置 Tie-breaker)
    run_stress_test(QdrantAdapter())
    
    # 3. Weaviate (潜在的不稳定风险)
    run_stress_test(WeaviateAdapter())