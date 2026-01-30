import numpy as np
from abc import ABC, abstractmethod
import time
import uuid
import re

# --- 抽象基类 ---
class VectorDBAdapter(ABC):
    def __init__(self, dim):
        self.dim = dim
        
    @abstractmethod
    def connect(self): pass
    @abstractmethod
    def recreate_collection(self, name): pass
    @abstractmethod
    def insert_batch(self, ids, vectors, payloads): pass
    @abstractmethod
    def flush_and_index(self): pass
    
    @abstractmethod
    def search(self, query_vec, limit, expr=None, search_params=None, consistency="Strong"): 
        pass
    
    @abstractmethod
    def delete(self, id_list): pass
    @abstractmethod
    def query_by_id(self, id): pass

# ---------------------------------------------------------
# 1. Milvus Adapter
# ---------------------------------------------------------
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
class MilvusAdapter(VectorDBAdapter):
    def __init__(self, dim, host='127.0.0.1', port='19530'):
        super().__init__(dim)
        self.uri = f"{host}:{port}"
        self.col = None
        
    def connect(self):
        try: connections.connect("default", host=self.uri.split(':')[0], port=self.uri.split(':')[1])
        except: pass
        
    def recreate_collection(self, name):
        if utility.has_collection(name): utility.drop_collection(name)
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="price", dtype=DataType.INT64), 
        ]
        schema = CollectionSchema(fields)
        self.col = Collection(name, schema) 
        
    def insert_batch(self, ids, vectors, payloads):
        self.col.insert([ids, vectors, payloads['price']])
        
    def flush_and_index(self):
        self.col.flush()
        index_params = {"metric_type": "L2", "index_type": "HNSW", "params": {"M": 16, "efConstruction": 128}}
        self.col.create_index("vector", index_params)
        self.col.load()
        
    def search(self, query_vec, limit, expr=None, search_params=None, consistency="Strong"):
        # 1. 确定基础 ef
        current_ef = 128  # 默认值
        if search_params and "ef" in search_params:
            current_ef = search_params["ef"]
            
        # 2. ⚡️ 全局卫语句：强制 ef >= limit
        # 无论用户怎么设，或者默认值是多少，必须保证 ef 够大
        if current_ef < limit:
            print(f"      ⚠️ [Auto-Fix] Boosting ef from {current_ef} to {limit} (Limit constraint)")
            current_ef = limit

        # 3. 组装参数
        sp = {
            "metric_type": "L2", 
            "params": {"ef": current_ef}
        }
            
        try:
            res = self.col.search(
                data=[query_vec], anns_field="vector", 
                param=sp, limit=limit, expr=expr, 
                output_fields=["id", "price"],
                consistency_level=consistency
            )
        except Exception as e:
            # 捕获其他可能的 RPC 错误
            print(f"      ❌ Milvus Search Error: {e}")
            raise e

        if not res: return []
        return [(hit.id, hit.distance, hit.entity.get("price")) for hit in res[0]]

    def delete(self, id_list):
        expr = f"id in {id_list}"
        self.col.delete(expr)
        
    def query_by_id(self, id):
        res = self.col.query(f"id == {id}", output_fields=["id"], consistency_level="Strong")
        return len(res) > 0

# ---------------------------------------------------------
# 2. Qdrant Adapter (修正连接方式)
# ---------------------------------------------------------
from qdrant_client import QdrantClient
from qdrant_client.http import models as Qmodels

class QdrantAdapter(VectorDBAdapter):
    def __init__(self, dim, host='127.0.0.1', port=6333):
        super().__init__(dim)
        self.col_name = None
        
        # ⚡️ 修复：不再手动拼接 url 字符串，而是使用 host/port 参数
        # 这样 QdrantClient 会自动处理 HTTP 和 gRPC 的连接逻辑
        try:
            print(f"      🔌 Connecting Qdrant via gRPC (Port 6334)...")
            self.client = QdrantClient(
                host=host,
                port=port,         # HTTP 端口 (6333)
                grpc_port=6334,    # gRPC 端口 (6334)
                prefer_grpc=True,  # 强制优先使用 gRPC
                timeout=None       # 无限等待
            )
        except Exception as e:
            print(f"      ⚠️ gRPC Init Failed ({e}), falling back to HTTP")
            self.client = QdrantClient(
                host=host, 
                port=port, 
                timeout=None
            )
        
    def connect(self): pass
    
    def recreate_collection(self, name):
        self.col_name = name
        # 检查是否存在，存在则删除
        if self.client.collection_exists(name):
            self.client.delete_collection(name)
            
        self.client.create_collection(
            collection_name=name,
            vectors_config=Qmodels.VectorParams(
                size=self.dim,
                distance=Qmodels.Distance.EUCLID,
                hnsw_config=Qmodels.HnswConfigDiff(m=8, ef_construct=32)
            ),
            # 写入时禁用索引，加速导入
            optimizers_config=Qmodels.OptimizersConfigDiff(indexing_threshold=0)
        )
        
    def insert_batch(self, ids, vectors, payloads):
        points = []
        for i in range(len(ids)):
            points.append(Qmodels.PointStruct(
                id=int(ids[i]), 
                vector=vectors[i].tolist(), 
                payload={"price": int(payloads['price'][i])}
            ))
        
        # 异步写入
        self.client.upsert(collection_name=self.col_name, points=points, wait=False)
        
    def flush_and_index(self):
        print("      ⚙️  [Qdrant] Triggering Indexing...", end="", flush=True)
        # 恢复索引阈值
        self.client.update_collection(
            collection_name=self.col_name,
            optimizer_config=Qmodels.OptimizersConfigDiff(indexing_threshold=20000)
        )
        # 创建 Payload 索引
        self.client.create_payload_index(self.col_name, "price", Qmodels.PayloadSchemaType.INTEGER)
        
        # 轮询等待
        for _ in range(120): # 给足时间 (2分钟)
            info = self.client.get_collection(self.col_name)
            if info.status == Qmodels.CollectionStatus.GREEN:
                print(" Done.")
                return
            time.sleep(1)
            print(".", end="", flush=True)
        print(" (Timeout warning, but proceeding)")

    def search(self, query_vec, limit, expr=None, search_params=None, consistency="Strong"):
        q_filter = None
        if expr:
            conditions = []
            gt = re.search(r"price > (\d+)", expr)
            if gt: conditions.append(Qmodels.FieldCondition(key="price", range=Qmodels.Range(gt=int(gt.group(1)))))
            lt = re.search(r"price < (\d+)", expr)
            if lt: conditions.append(Qmodels.FieldCondition(key="price", range=Qmodels.Range(lt=int(lt.group(1)))))
            if conditions: q_filter = Qmodels.Filter(must=conditions)

        search_p = None
        if search_params and "ef" in search_params:
            search_p = Qmodels.SearchParams(hnsw_ef=search_params["ef"])
            
        res = self.client.search(
            collection_name=self.col_name,
            query_vector=query_vec.tolist(), 
            query_filter=q_filter,
            search_params=search_p,
            limit=limit
        )
        return [(h.id, h.score, h.payload['price']) for h in res]

    def delete(self, id_list):
        self.client.delete(
            collection_name=self.col_name,
            points_selector=Qmodels.PointIdsList(points=id_list),
            wait=True 
        )
        
    def query_by_id(self, id):
        try:
            res = self.client.retrieve(self.col_name, [int(id)])
            return len(res) > 0
        except: return False
# ---------------------------------------------------------
# 3. Weaviate Adapter
# ---------------------------------------------------------
import weaviate
class WeaviateAdapter(VectorDBAdapter):
    def __init__(self, dim, host='127.0.0.1', port=8080):
        super().__init__(dim)
        self.client = weaviate.Client(f"http://{host}:{port}")
        self.col_name = "AdvancedTest"
        
    def connect(self): pass
    
    def recreate_collection(self, name):
        self.client.schema.delete_all()
        class_obj = {
            "class": self.col_name,
            "vectorizer": "none",
            "properties": [{"name": "idx", "dataType": ["int"]}, {"name": "price", "dataType": ["int"]}]
        }
        self.client.schema.create_class(class_obj)
        
    def insert_batch(self, ids, vectors, payloads):
        with self.client.batch as batch:
            batch.batch_size = len(ids)
            for i in range(len(ids)):
                props = {"idx": int(ids[i]), "price": int(payloads['price'][i])}
                batch.add_data_object(props, self.col_name, vector=vectors[i].tolist())
                
    def flush_and_index(self): pass 
    
    def search(self, query_vec, limit, expr=None, search_params=None, consistency="Strong"):
        where_filter = {}
        if expr:
            operands = []
            gt = re.search(r"price > (\d+)", expr)
            if gt: operands.append({"path": ["price"], "operator": "GreaterThan", "valueInt": int(gt.group(1))})
            lt = re.search(r"price < (\d+)", expr)
            if lt: operands.append({"path": ["price"], "operator": "LessThan", "valueInt": int(lt.group(1))})
            
            if len(operands) == 1: where_filter = operands[0]
            elif len(operands) > 1: where_filter = {"operator": "And", "operands": operands}

        query = self.client.query.get(self.col_name, ["idx", "price", "_additional { distance }"])
        if where_filter: query = query.with_where(where_filter)
        
        response = query.with_near_vector({"vector": query_vec.tolist()}).with_limit(limit).do()
        
        if 'errors' in response: return []
        return [(item['idx'], item['_additional']['distance'], item['price']) for item in response['data']['Get'][self.col_name]]

    def delete(self, id_list):
        for uid in id_list:
            self.client.batch.delete_objects(
                class_name=self.col_name,
                where={"path": ["idx"], "operator": "Equal", "valueInt": uid}
            )
            
    def query_by_id(self, id):
        res = self.client.query.get(self.col_name, ["idx"]).with_where({"path": ["idx"], "operator": "Equal", "valueInt": id}).do()
        if 'errors' in res or 'data' not in res: return False
        return len(res['data']['Get'][self.col_name]) > 0

# ---------------------------------------------------------
# 4. Chroma Adapter
# ---------------------------------------------------------
import chromadb
class ChromaAdapter(VectorDBAdapter):
    def __init__(self, dim, host='127.0.0.1', port=8000):
        super().__init__(dim)
        self.client = chromadb.HttpClient(host=host, port=port)
        self.col = None
    def connect(self): pass
    def recreate_collection(self, name):
        try: self.client.delete_collection(name)
        except: pass
        self.col = self.client.get_or_create_collection(name, metadata={"hnsw:space": "l2"})
    def insert_batch(self, ids, vectors, payloads):
        metadatas = [{"price": int(p)} for p in payloads['price']]
        self.col.add(ids=[str(i) for i in ids], embeddings=vectors.tolist(), metadatas=metadatas)
    def flush_and_index(self): pass
    
    def search(self, query_vec, limit, expr=None, search_params=None, consistency="Strong"):
        where = {}
        if expr:
            gt = re.search(r"price > (\d+)", expr)
            lt = re.search(r"price < (\d+)", expr)
            if gt and lt: where = {"$and": [{"price": {"$gt": int(gt.group(1))}}, {"price": {"$lt": int(lt.group(1))}}]}
            elif gt: where = {"price": {"$gt": int(gt.group(1))}}
            
        try:
            res = self.col.query(query_embeddings=[query_vec.tolist()], n_results=limit, where=where)
            if not res['ids']: return []
            return [(int(id), dist, meta['price']) for id, dist, meta in zip(res['ids'][0], res['distances'][0], res['metadatas'][0])]
        except: return []

    def delete(self, id_list):
        self.col.delete(ids=[str(i) for i in id_list])
    def query_by_id(self, id):
        res = self.col.get(ids=[str(id)])
        return len(res['ids']) > 0