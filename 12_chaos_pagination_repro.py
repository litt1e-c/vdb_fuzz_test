import time
import random
import threading
import numpy as np
from abc import ABC, abstractmethod

# 数据库客户端
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from qdrant_client import QdrantClient, models
import weaviate

# ==============================================================================
# 🔥 全局高压配置
# ==============================================================================
HOST = '127.0.0.1'
BASE_NUM = 50000          # 初始数据量 (足够大以形成复杂图结构)
NOISE_RATE = 0.5          # 写入/删除频率 (秒)
TEST_DURATION = 60        # 每个库测 60 秒
SPARSE_DICT = {128: 1.0, 512: 0.8, 1024: 0.5} # 绝对同分向量

# 颜色
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
RESET = '\033[0m'

# 全局停止标志
STOP_FLAG = False

# ==============================================================================
# 🧱 适配器层 (带 Chaos 注入能力)
# ==============================================================================
class ChaosAdapter(ABC):
    def __init__(self): self.name = "Unknown"
    @abstractmethod
    def setup(self): pass
    @abstractmethod
    def insert_base_data(self): pass
    @abstractmethod
    def inject_noise(self): pass # 干扰操作：插入/删除
    @abstractmethod
    def search_pagination(self, limit, offset): pass

# --- 1. Milvus Adapter ---
class MilvusAdapter(ChaosAdapter):
    def __init__(self):
        self.name = "Milvus"
        self.col_name = "chaos_page_milvus"
    
    def setup(self):
        try: connections.connect("default", host=HOST, port="19530")
        except: pass
        if utility.has_collection(self.col_name): utility.drop_collection(self.col_name)
        
        schema = CollectionSchema([
            FieldSchema("id", DataType.INT64, is_primary=True),
            FieldSchema("vec", DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema("group", DataType.INT64) # 用于过滤
        ])
        # 关键点：consistency_level="Eventually" (最容易出问题)
        self.col = Collection(self.col_name, schema, consistency_level="Eventually")

    def insert_base_data(self):
        print(f"   📥 [{self.name}] Inserting {BASE_NUM} base items...")
        # 批量插入
        ids = list(range(BASE_NUM))
        vecs = [SPARSE_DICT] * BASE_NUM
        groups = [i % 10 for i in ids] # 10个组
        self.col.insert([ids, vecs, groups])
        self.col.flush()
        # 建索引：使用较小的 M 和 efC，增加不稳定性
        self.col.create_index("vec", {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP", "params": {"drop_ratio_build": 0.0}})
        self.col.load()

    def inject_noise(self):
        # 随机插入一条新数据，或删除一条旧数据
        try:
            action = random.choice(["insert", "delete"])
            new_id = random.randint(BASE_NUM, BASE_NUM + 10000)
            if action == "insert":
                self.col.insert([[new_id], [SPARSE_DICT], [random.randint(0,9)]])
            else:
                del_id = random.randint(0, BASE_NUM - 1)
                self.col.delete(f"id == {del_id}")
        except: pass

    def search_pagination(self, limit, offset):
        # 关键点：带 Filter 查询，逼迫 QueryNode 做复杂归并
        # 关键点：ef=10 (极低)，诱发搜索窗口漂移
        res = self.col.search(
            data=[SPARSE_DICT], anns_field="vec", 
            param={"metric_type": "IP", "params": {"drop_ratio_search": 0.0}}, 
            limit=limit, offset=offset, expr="group >= 0", # 恒真 Filter
            output_fields=["id"]
        )
        return [h.id for h in res[0]]

# --- 2. Qdrant Adapter ---
class QdrantAdapter(ChaosAdapter):
    def __init__(self):
        self.name = "Qdrant"
        self.col_name = "chaos_page_qdrant"
        self.client = QdrantClient(url=f"http://{HOST}:6333", timeout=10)

    def setup(self):
        self.client.recreate_collection(
            collection_name=self.col_name,
            vectors_config={},
            sparse_vectors_config={"vec": models.SparseVectorParams(index=models.SparseIndexParams(on_disk=False))}
        )

    def insert_base_data(self):
        print(f"   📥 [{self.name}] Inserting {BASE_NUM} base items...")
        points = []
        indices = list(SPARSE_DICT.keys())
        values = list(SPARSE_DICT.values())
        for i in range(BASE_NUM):
            points.append(models.PointStruct(
                id=i, 
                vector={"vec": models.SparseVector(indices=indices, values=values)},
                payload={"group": i % 10}
            ))
            if len(points) >= 1000:
                self.client.upsert(self.col_name, points)
                points = []
        if points: self.client.upsert(self.col_name, points)

    def inject_noise(self):
        try:
            new_id = random.randint(BASE_NUM, BASE_NUM + 10000)
            self.client.upsert(self.col_name, [models.PointStruct(
                id=new_id,
                vector={"vec": models.SparseVector(indices=list(SPARSE_DICT.keys()), values=list(SPARSE_DICT.values()))},
                payload={"group": 1}
            )])
        except: pass

    def search_pagination(self, limit, offset):
        # Qdrant 在有 Filter 时可能会走不同的优化路径
        res = self.client.search(
            self.col_name,
            query_vector=models.NamedSparseVector(name="vec", vector=models.SparseVector(indices=list(SPARSE_DICT.keys()), values=list(SPARSE_DICT.values()))),
            query_filter=models.Filter(must=[models.FieldCondition(key="group", range=models.Range(gte=0))]),
            limit=limit, offset=offset, with_vectors=False
        )
        return [r.id for r in res]

# --- 3. Weaviate Adapter ---
class WeaviateAdapter(ChaosAdapter):
    def __init__(self):
        self.name = "Weaviate"
        self.col_name = "ChaosPage"
        self.client = weaviate.Client(f"http://{HOST}:8080")
        self.dense_vec = [0.0] * 2048
        for k,v in SPARSE_DICT.items(): self.dense_vec[k] = v

    def setup(self):
        self.client.schema.delete_all()
        self.client.schema.create_class({
            "class": self.col_name, "vectorizer": "none",
            "properties": [{"name": "idx", "dataType": ["int"]}, {"name": "group", "dataType": ["int"]}]
        })

    def insert_base_data(self):
        print(f"   📥 [{self.name}] Inserting {BASE_NUM} base items...")
        with self.client.batch as batch:
            batch.batch_size = 100
            for i in range(BASE_NUM):
                batch.add_data_object({"idx": i, "group": i%10}, self.col_name, vector=self.dense_vec)

    def inject_noise(self):
        # Weaviate 插入非常慢，这里只做极少量插入干扰
        try:
            self.client.data_object.create({"idx": 99999}, self.col_name, vector=self.dense_vec)
        except: pass

    def search_pagination(self, limit, offset):
        resp = (self.client.query.get(self.col_name, ["idx"])
                .with_near_vector({"vector": self.dense_vec})
                .with_where({"path": ["group"], "operator": "GreaterThanEqual", "valueInt": 0})
                .with_limit(limit).with_offset(offset)
                .do())
        if 'errors' in resp: return []
        return [x['idx'] for x in resp['data']['Get'][self.col_name]]

# ==============================================================================
# 🌪️ 混沌测试控制器
# ==============================================================================
def chaos_worker(adapter):
    """后台线程：不断骚扰数据库"""
    while not STOP_FLAG:
        adapter.inject_noise()
        time.sleep(NOISE_RATE)

def run_test(adapter):
    global STOP_FLAG
    STOP_FLAG = False
    
    print(f"\n\n{CYAN}{'='*20} TESTING {adapter.name} (Chaos Mode) {'='*20}{RESET}")
    
    # 1. 初始化
    adapter.setup()
    adapter.insert_base_data()
    
    # 2. 启动干扰线程
    print(f"   🌪️  Starting background noise generator...")
    t = threading.Thread(target=chaos_worker, args=(adapter,))
    t.start()
    
    # 3. 循环验证分页稳定性
    print(f"   🔍 Verifying Pagination Stability (Duration: {TEST_DURATION}s)...")
    start_time = time.time()
    iteration = 0
    failures = 0
    
    try:
        while time.time() - start_time < TEST_DURATION:
            iteration += 1
            
            # Query A: Page 1 (Top 10)
            page1 = adapter.search_pagination(limit=10, offset=0)
            
            # Query B: Page 2 (Offset 5)
            # 预期：Page 2 的开头，应该是 Page 1 的第 6 个元素 (Index 5)
            page2 = adapter.search_pages(limit=5, offset=5) if hasattr(adapter, 'search_pages') else adapter.search_pagination(limit=5, offset=5)
            
            if len(page1) < 10 or len(page2) < 1:
                continue # 刚开始可能数据不全

            expected = page1[5]
            actual = page2[0]
            
            # 核心判定
            if expected != actual:
                failures += 1
                print(f"      {RED}❌ Mismatch at iter {iteration}!{RESET}")
                print(f"         Page 1: {page1}")
                print(f"         Page 2: {page2}")
                print(f"         Expected: {expected} | Actual: {actual}")
                
                # 如果是 Weaviate/Qdrant，乱序是正常的，但我们想看会不会出现“数据漂移”
                # 即 Page 2 的数据居然在 Page 1 里出现过 (重复)，或者完全没出现过 (遗漏)
                if actual in page1[:5]:
                     print(f"         🚨 DATA DUPLICATION: Item {actual} appeared in both Page 1 and Page 2!")
            
            # 稍微缓一下，别把自己压死了
            time.sleep(0.2)
            
    except KeyboardInterrupt:
        pass
    finally:
        STOP_FLAG = True
        t.join()
        
    print(f"\n   📊 {adapter.name} Summary:")
    print(f"      Total Iterations: {iteration}")
    print(f"      Failures: {failures}")
    if failures > 0:
        print(f"      {RED}结论: 在动态写入干扰下，分页逻辑出现了不一致！{RESET}")
    else:
        print(f"      {GREEN}结论: 即使有干扰，分页依然稳定 (鲁棒性极强)。{RESET}")

if __name__ == "__main__":
    # 1. Milvus (最强对手)
    run_test(MilvusAdapter())
    
    # 2. Qdrant
    run_test(QdrantAdapter())
    
    # 3. Weaviate
    run_test(WeaviateAdapter())