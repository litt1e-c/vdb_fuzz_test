import numpy as np
from abc import ABC, abstractmethod
import time

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
    def search(self, query_vec, limit, expr_type=None): pass

# --- 1. Milvus Adapter (保持不变) ---
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
            FieldSchema(name="age", dtype=DataType.INT64),
            FieldSchema(name="price", dtype=DataType.FLOAT),
            FieldSchema(name="tag", dtype=DataType.INT64)
        ]
        schema = CollectionSchema(fields)
        self.col = Collection(name, schema, consistency_level="Strong")
    def insert_batch(self, ids, vectors, payloads):
        self.col.insert([ids, vectors, payloads['age'], payloads['price'], payloads['tag']])
    def flush_and_index(self):
        self.col.flush()
        # 使用 FLAT 索引实现精确搜索，消除 HNSW 近似性
        # 原 HNSW 参数: {"index_type": "HNSW", "params": {"M": 64, "efConstruction": 1024}}
        index_params = {"metric_type": "L2", "index_type": "FLAT", "params": {}}
        self.col.create_index("vector", index_params)
        self.col.load()
    def search(self, query_vec, limit, expr_type=None):
        expr = self._translate_expr(expr_type)
        # FLAT 索引使用精确搜索，无需 ef 参数
        # 原 HNSW 搜索参数: ef_val = max(4096, int(limit) * 2)
        search_params = {"metric_type": "L2", "params": {}}
        try:
            res = self.col.search([query_vec], "vector", search_params, limit=limit, expr=expr, output_fields=["id"])
            return set([h.id for h in res[0]])
        except Exception as e:
            print(f"RPC error: [search] {e}")
            return set()

    # 新增：返回有序的 (id, distance) 列表用于 Top-K 与距离单调性
    def search_ordered(self, query_vec, limit, expr_type=None):
        expr = self._translate_expr(expr_type)
        # FLAT 索引精确搜索
        # 原 HNSW 参数: ef_val = max(4096, int(limit) * 2)
        search_params = {"metric_type": "L2", "params": {}}
        try:
            res = self.col.search([query_vec], "vector", search_params, limit=limit, expr=expr, output_fields=["id"])
            return [(hit.id, hit.distance) for hit in res[0]]
        except Exception as e:
            print(f"RPC error: [search_ordered] {e}")
            return []

    # 统一表达式翻译，支持简单标识与 OR/AND 组合
    def _translate_expr(self, expr_type):
        if expr_type is None:
            return ""
        # 组合表达式：("or", a, b) 或 ("and", a, b)
        if isinstance(expr_type, tuple) and len(expr_type) == 3:
            op = expr_type[0]
            if op == "or":
                a = self._translate_expr(expr_type[1])
                b = self._translate_expr(expr_type[2])
                a = f"({a})" if a else ""
                b = f"({b})" if b else ""
                return f"{a} || {b}".strip()
            elif op == "and":
                a = self._translate_expr(expr_type[1])
                b = self._translate_expr(expr_type[2])
                a = f"({a})" if a else ""
                b = f"({b})" if b else ""
                return f"{a} && {b}".strip()
        # 基本映射
        mapping = {
            "range_standard": "age >= 20 && age <= 30 && price > 500",
            "range_split": "((age >= 20 && age < 25) || (age >= 25 && age <= 30)) && price > 500",
            "cat_in": "tag in [1, 2, 3]",
            "cat_or": "tag == 1 || tag == 2 || tag == 3",
            "massive_or": " || ".join([f"id == {i}" for i in range(50)]),
            # 新增：测试属性所需的简写
            "age_20_30": "age >= 20 && age <= 30",
            "tag_0": "tag == 0",
            "tag_1": "tag == 1",
            "tag_1_tautology": "tag == 1 && age >= 0 && age <= 100 && price >= 0 && price <= 1000",  # 恒真条件冗余
            "tag_2": "tag == 2",
            "not_tag_1": "tag != 1",
            "tag_1_or_2": "tag == 1 || tag == 2",
            "tag_not_1_or_2": "tag == 0 || tag == 3",
            "age_lt_30": "age < 30",
            "age_plus_3_lt_33": "(age + 3) < 33",
            "age_any": "age >= 0"  # 放宽条件用于距离单调性
        }
        return mapping.get(expr_type, "")

# --- 2. Qdrant Adapter (保持 timeout 修复) ---
from qdrant_client import QdrantClient
from qdrant_client.http import models as Qmodels
class QdrantAdapter(VectorDBAdapter):
    def __init__(self, dim, host='127.0.0.1', port=6333):
        super().__init__(dim)
        # timeout 设置为 None (无限等待)
        self.client = QdrantClient(url=f"http://{host}:{port}", timeout=None)
        self.col_name = None
    def connect(self): pass
    def recreate_collection(self, name):
        self.col_name = name
        # 使用现代 API 避免 deprecation 警告
        try:
            self.client.delete_collection(collection_name=name)
        except Exception:
            pass
        self.client.create_collection(
            collection_name=name,
            vectors_config=Qmodels.VectorParams(size=self.dim, distance=Qmodels.Distance.EUCLID)
        )
    def insert_batch(self, ids, vectors, payloads):
        points = []
        for i in range(len(ids)):
            pt_payload = {
                "age": int(payloads['age'][i]), 
                "price": float(payloads['price'][i]), 
                "tag": int(payloads['tag'][i])
            }
            points.append(Qmodels.PointStruct(id=int(ids[i]), vector=vectors[i].tolist(), payload=pt_payload))
        self.client.upsert(collection_name=self.col_name, points=points)
    def flush_and_index(self):
        # 使用精确搜索（exact=True），无需调整 HNSW 参数
        # 原 HNSW 配置: hnsw_config=Qmodels.HnswConfigDiff(m=64, ef_construct=1024)
        # 注：Qdrant 会自动使用默认 HNSW，但 exact=True 会强制精确搜索
        self.client.create_payload_index(self.col_name, "age", Qmodels.PayloadSchemaType.INTEGER)
        self.client.create_payload_index(self.col_name, "price", Qmodels.PayloadSchemaType.FLOAT)
        self.client.create_payload_index(self.col_name, "tag", Qmodels.PayloadSchemaType.INTEGER)
    
    def search(self, query_vec, limit, expr_type=None):
        q_filter = self._translate_expr(expr_type)
        try:
            res = self.client.search(
                collection_name=self.col_name,
                query_vector=query_vec.tolist(), 
                query_filter=q_filter,
                limit=limit,
                # 使用精确搜索，消除 HNSW 近似性
                # 原 HNSW 参数: hnsw_ef=max(4096, limit * 2)
                search_params=Qmodels.SearchParams(exact=True)
            )
            return set([h.id for h in res])
        except Exception as e:
            print(f"RPC error: [Qdrant search] {e}")
            return set()
    
    def search_ordered(self, query_vec, limit, expr_type=None):
        q_filter = self._translate_expr(expr_type)
        try:
            res = self.client.search(
                collection_name=self.col_name,
                query_vector=query_vec.tolist(), 
                query_filter=q_filter,
                limit=limit,
                # 精确搜索
                # 原 HNSW 参数: hnsw_ef=max(4096, limit * 2)
                search_params=Qmodels.SearchParams(exact=True)
            )
            return [(h.id, h.score) for h in res]
        except Exception as e:
            print(f"RPC error: [Qdrant search_ordered] {e}")
            return []
    
    def _translate_expr(self, expr_type):
        if expr_type is None:
            return None
        if isinstance(expr_type, tuple) and len(expr_type) == 3:
            op = expr_type[0]
            a = self._translate_expr(expr_type[1])
            b = self._translate_expr(expr_type[2])
            if op == "or":
                parts = [p for p in [a, b] if p is not None]
                if not parts:
                    return None
                return Qmodels.Filter(should=parts)
            elif op == "and":
                parts = [p for p in [a, b] if p is not None]
                if not parts:
                    return None
                # Flatten must conditions if both are filters
                must_conditions = []
                for p in parts:
                    if isinstance(p, Qmodels.Filter) and p.must:
                        must_conditions.extend(p.must)
                    else:
                        must_conditions.append(p)
                return Qmodels.Filter(must=must_conditions)
        mapping = {
            "age_20_30": Qmodels.Filter(must=[Qmodels.FieldCondition(key="age", range=Qmodels.Range(gte=20, lte=30))]),
            "tag_0": Qmodels.Filter(must=[Qmodels.FieldCondition(key="tag", match=Qmodels.MatchValue(value=0))]),
            "tag_1": Qmodels.Filter(must=[Qmodels.FieldCondition(key="tag", match=Qmodels.MatchValue(value=1))]),
            "tag_1_tautology": Qmodels.Filter(must=[
                Qmodels.FieldCondition(key="tag", match=Qmodels.MatchValue(value=1)),
                Qmodels.FieldCondition(key="age", range=Qmodels.Range(gte=0, lte=100)),
                Qmodels.FieldCondition(key="price", range=Qmodels.Range(gte=0, lte=1000))
            ]),
            "tag_2": Qmodels.Filter(must=[Qmodels.FieldCondition(key="tag", match=Qmodels.MatchValue(value=2))]),
            "not_tag_1": Qmodels.Filter(must_not=[Qmodels.FieldCondition(key="tag", match=Qmodels.MatchValue(value=1))]),
            "tag_1_or_2": Qmodels.Filter(must=[Qmodels.FieldCondition(key="tag", match=Qmodels.MatchAny(any=[1,2]))]),
            "tag_not_1_or_2": Qmodels.Filter(must=[Qmodels.FieldCondition(key="tag", match=Qmodels.MatchAny(any=[0,3]))]),
            "age_lt_30": Qmodels.Filter(must=[Qmodels.FieldCondition(key="age", range=Qmodels.Range(lt=30))]),
            "age_plus_3_lt_33": Qmodels.Filter(must=[Qmodels.FieldCondition(key="age", range=Qmodels.Range(lt=30))]),
            "age_any": None,
            "range_standard": Qmodels.Filter(must=[Qmodels.FieldCondition(key="age", range=Qmodels.Range(gte=20, lte=30)), Qmodels.FieldCondition(key="price", range=Qmodels.Range(gt=500))]),
            "range_split": Qmodels.Filter(must=[Qmodels.FieldCondition(key="price", range=Qmodels.Range(gt=500))], should=[Qmodels.FieldCondition(key="age", range=Qmodels.Range(gte=20, lt=25)), Qmodels.FieldCondition(key="age", range=Qmodels.Range(gte=25, lte=30))]),
            "cat_in": Qmodels.Filter(must=[Qmodels.FieldCondition(key="tag", match=Qmodels.MatchAny(any=[1,2,3]))]),
            "cat_or": Qmodels.Filter(should=[Qmodels.FieldCondition(key="tag", match=Qmodels.MatchValue(value=1)), Qmodels.FieldCondition(key="tag", match=Qmodels.MatchValue(value=2)), Qmodels.FieldCondition(key="tag", match=Qmodels.MatchValue(value=3))])
        }
        return mapping.get(expr_type, None)

# --- 3. Chroma Adapter (尝试更标准的写法) ---
import chromadb
from chromadb.config import Settings
# 本地模式不需要 Settings
class ChromaAdapter(VectorDBAdapter):
    def __init__(self, dim, host='127.0.0.1', port=8000):
        super().__init__(dim)
        # 关闭匿名遥测，避免 capture() 报错噪音
        self.client = chromadb.PersistentClient(
            path="/home/caihao/compare_test/tmp/chroma",
            settings=Settings(anonymized_telemetry=False)
        )
        self.col = None
    def connect(self): pass
    def insert_batch(self, ids, vectors, payloads):
        metadatas = []
        str_ids = [str(i) for i in ids]
        for i in range(len(ids)):
            metadatas.append({
                "age": int(payloads['age'][i]), 
                "price": float(payloads['price'][i]), 
                "tag": int(payloads['tag'][i])
            })
        self.col.add(ids=str_ids, embeddings=vectors.tolist(), metadatas=metadatas)
    def flush_and_index(self):
        # Chroma 自动使用 HNSW，通过 metadata 设置高质量参数
        pass
    
    def recreate_collection(self, name):
        try: self.client.delete_collection(name)
        except: pass
        # Chroma 仅支持 HNSW 索引，无法使用 FLAT
        # 为尽量接近精确搜索，使用高质量 HNSW 参数
        # 原参数: construction_ef=800, M=64
        self.col = self.client.get_or_create_collection(
            name,
            metadata={
                "hnsw:space": "l2",
                "hnsw:construction_ef": 2000,  # 提高构建质量
                "hnsw:M": 128  # 增加连接数以提高召回
            }
        )
    
    def search(self, query_vec, limit, expr_type=None):
        where = self._translate_expr(expr_type)
        n_res = min(max(limit * 2, 200), 2000)
        try:
            res = self.col.query(query_embeddings=[query_vec.tolist()], n_results=n_res, where=where)
            if not res['ids']: return set()
            ids = [int(id) for id in res['ids'][0][:limit]]
            return set(ids)
        except Exception as e:
            # 针对 contiguous 2D array 报错，降低 n_results 重试一次
            if "contigious 2D array" in str(e) or "contiguous 2D array" in str(e):
                try:
                    fallback_n = max(limit, 200)
                    res = self.col.query(query_embeddings=[query_vec.tolist()], n_results=fallback_n, where=where)
                    if not res['ids']: return set()
                    ids = [int(id) for id in res['ids'][0][:limit]]
                    return set(ids)
                except Exception as e2:
                    print(f"   ⚠️ Chroma Query Error [{expr_type}] fallback: {e2}")
                    return set()
            print(f"   ⚠️ Chroma Query Error [{expr_type}]: {e}")
            return set()
    
    def search_ordered(self, query_vec, limit, expr_type=None):
        where = self._translate_expr(expr_type)
        n_res = min(max(limit * 2, 200), 2000)
        try:
            res = self.col.query(query_embeddings=[query_vec.tolist()], n_results=n_res, where=where, include=["distances"])
            if not res['ids']: return []
            ids = [int(id) for id in res['ids'][0][:limit]]
            dists = res['distances'][0][:limit]
            return list(zip(ids, dists))
        except Exception as e:
            if "contigious 2D array" in str(e) or "contiguous 2D array" in str(e):
                try:
                    fallback_n = max(limit, 200)
                    res = self.col.query(query_embeddings=[query_vec.tolist()], n_results=fallback_n, where=where, include=["distances"])
                    if not res['ids']: return []
                    ids = [int(id) for id in res['ids'][0][:limit]]
                    dists = res['distances'][0][:limit]
                    return list(zip(ids, dists))
                except Exception as e2:
                    print(f"   ⚠️ Chroma Query Error [{expr_type}] fallback: {e2}")
                    return []
            print(f"   ⚠️ Chroma Query Error [{expr_type}]: {e}")
            return []
    
    def _translate_expr(self, expr_type):
        if expr_type is None:
            return {}
        if isinstance(expr_type, tuple) and len(expr_type) == 3:
            op = expr_type[0]
            a = self._translate_expr(expr_type[1])
            b = self._translate_expr(expr_type[2])
            if op == "or":
                if a and b:
                    return {"$or": [a, b]}
                return a or b or {}
            elif op == "and":
                if a and b:
                    return {"$and": [a, b]}
                return a or b or {}
        mapping = {
            "age_20_30": {"$and": [{"age": {"$gte": 20}}, {"age": {"$lte": 30}}]},
            "tag_0": {"tag": {"$eq": 0}},
            "tag_1": {"tag": {"$eq": 1}},
            "tag_1_tautology": {"$and": [{"tag": {"$eq": 1}}, {"age": {"$gte": 0}}, {"age": {"$lte": 100}}, {"price": {"$gte": 0}}, {"price": {"$lte": 1000}}]},
            "tag_2": {"tag": {"$eq": 2}},
            "not_tag_1": {"tag": {"$ne": 1}},
            "tag_1_or_2": {"tag": {"$in": [1, 2]}},
            "tag_not_1_or_2": {"tag": {"$in": [0, 3]}},
            "age_lt_30": {"age": {"$lt": 30}},
            "age_plus_3_lt_33": {"age": {"$lt": 30}},
            "age_any": {},
            "range_standard": {"$and": [{"age": {"$gte": 20}}, {"age": {"$lte": 30}}, {"price": {"$gt": 500}}]},
            "range_split": {"$or": [{"$and": [{"age": {"$gte": 20}}, {"age": {"$lt": 25}}, {"price": {"$gt": 500}}]}, {"$and": [{"age": {"$gte": 25}}, {"age": {"$lte": 30}}, {"price": {"$gt": 500}}]}]},
            "cat_in": {"tag": {"$in": [1, 2, 3]}},
            "cat_or": {"$or": [{"tag": 1}, {"tag": 2}, {"tag": 3}]},
            "massive_or": None
        }
        result = mapping.get(expr_type, {})
        return result if result is not None else {}

# --- 4. Weaviate Adapter (深度修复) ---
import weaviate
class WeaviateAdapter(VectorDBAdapter):
    def __init__(self, dim, host='127.0.0.1', port=8080):
        super().__init__(dim)
        self.client = weaviate.Client(f"http://{host}:{port}")
        self.col_name = "UniversalHydra"
    def connect(self): pass
    def recreate_collection(self, name):
        # 修复：仅删除指定类，避免清空整个数据库
        try:
            self.client.schema.delete_class(name)
        except Exception:
            pass

        # Weaviate 要求类名首字母大写
        if name:
            name = name[0].upper() + name[1:]
        self.col_name = name

        class_obj = {
            "class": name,
            "vectorizer": "none",
            "properties": [
                {"name": "idx", "dataType": ["int"]}, 
                {"name": "age", "dataType": ["int"]},
                {"name": "price", "dataType": ["number"]},
                {"name": "tag", "dataType": ["int"]}
            ]
        }
        # Weaviate 仅支持 HNSW 索引，无法完全禁用
        # 使用高质量参数以接近精确搜索
        # 原参数: ef=1024, efConstruction=512, maxConnections=64
        class_obj["vectorIndexConfig"] = {
            "ef": 4096,  # 大幅提高搜索质量
            "efConstruction": 2000,  # 提高构建质量
            "maxConnections": 128,  # 增加连接数
            "distance": "l2-squared"  # 使用 L2 距离
        }
        self.client.schema.create_class(class_obj)
    def insert_batch(self, ids, vectors, payloads):
        with self.client.batch as batch:
            batch.batch_size = 100
            for i in range(len(ids)):
                props = {
                    "idx": int(ids[i]), 
                    "age": int(payloads['age'][i]), 
                    "price": float(payloads['price'][i]), 
                    "tag": int(payloads['tag'][i])
                }
                batch.add_data_object(props, self.col_name, vector=vectors[i].tolist())
        # Weaviate v3 的 batch 会在退出 context 时自动提交，无需额外等待
    def flush_and_index(self):
        # Weaviate 批处理后强制等待索引完成
        time.sleep(3)  # 给异步索引充足时间
        # 验证数据是否真正写入
        try:
            agg = self.client.query.aggregate(self.col_name).with_meta_count().do()
            count = agg.get('data', {}).get('Aggregate', {}).get(self.col_name, [{}])[0].get('meta', {}).get('count', 0)
            if count == 0:
                print(f"   ⚠️ Weaviate: 索引后集合 {self.col_name} 为空！数据可能未成功写入")
        except Exception as e:
            print(f"   ⚠️ Weaviate count check failed: {e}")
    
    def search(self, query_vec, limit, expr_type=None):
        where_filter = self._translate_expr(expr_type)
        
        try:
            if where_filter:
                response = (
                    self.client.query
                    .get(self.col_name, ["idx", "_additional { distance }"])
                    .with_near_vector({"vector": query_vec.tolist()}) 
                    .with_where(where_filter)
                    .with_limit(limit)
                    .do()
                )
            else:
                response = (
                    self.client.query
                    .get(self.col_name, ["idx", "_additional { distance }"])
                    .with_near_vector({"vector": query_vec.tolist()})
                    .with_limit(limit)
                    .do()
                )
            
            if 'errors' in response:
                print(f"   ⚠️ Weaviate Query Error [{expr_type}]: {response['errors']}")
                return set()
            
            items = response.get('data', {}).get('Get', {}).get(self.col_name, [])
            if not items:
                # 调试：查一次无过滤的，确认数据是否真的在
                if expr_type is not None:
                    test_res = self.client.query.get(self.col_name, ["idx"]).with_limit(5).do()
                    test_items = test_res.get('data', {}).get('Get', {}).get(self.col_name, [])
                    if not test_items:
                        print(f"   ⚠️ Weaviate: 集合 {self.col_name} 完全为空，数据未成功写入")
                return set()
            
            return set([item['idx'] for item in items])
        except Exception as e:
            print(f"   ⚠️ Weaviate search exception [{expr_type}]: {e}")
            return set()
    
    def search_ordered(self, query_vec, limit, expr_type=None):
        where_filter = self._translate_expr(expr_type)
        try:
            response = (
                self.client.query
                .get(self.col_name, ["idx", "_additional { distance }"])
                .with_near_vector({"vector": query_vec.tolist()})
                .with_where(where_filter) if where_filter else
                self.client.query
                .get(self.col_name, ["idx", "_additional { distance }"])
                .with_near_vector({"vector": query_vec.tolist()})
            )
            if where_filter:
                response = response.with_limit(limit).do()
            else:
                response = response.with_limit(limit).do()
            
            if 'errors' in response:
                print(f"   ⚠️ Weaviate Error: {response['errors']}")
                return []
            
            return [(item['idx'], item['_additional']['distance']) for item in response['data']['Get'][self.col_name]]
        except Exception as e:
            print(f"   ⚠️ Weaviate search_ordered Error: {e}")
            return []
    
    def _translate_expr(self, expr_type):
        if expr_type is None:
            return {}
        if isinstance(expr_type, tuple) and len(expr_type) == 3:
            op = expr_type[0]
            a = self._translate_expr(expr_type[1])
            b = self._translate_expr(expr_type[2])
            if op == "or":
                if a and b:
                    return {"operator": "Or", "operands": [a, b]}
                return a or b or {}
            elif op == "and":
                if a and b:
                    return {"operator": "And", "operands": [a, b]}
                return a or b or {}
        mapping = {
            "age_20_30": {"operator": "And", "operands": [{"path": ["age"], "operator": "GreaterThanEqual", "valueInt": 20}, {"path": ["age"], "operator": "LessThanEqual", "valueInt": 30}]},
            "tag_0": {"path": ["tag"], "operator": "Equal", "valueInt": 0},
            "tag_1": {"path": ["tag"], "operator": "Equal", "valueInt": 1},
            "tag_1_tautology": {"operator": "And", "operands": [{"path": ["tag"], "operator": "Equal", "valueInt": 1}, {"path": ["age"], "operator": "GreaterThanEqual", "valueInt": 0}, {"path": ["age"], "operator": "LessThanEqual", "valueInt": 100}, {"path": ["price"], "operator": "GreaterThanEqual", "valueNumber": 0}, {"path": ["price"], "operator": "LessThanEqual", "valueNumber": 1000}]},
            "tag_2": {"path": ["tag"], "operator": "Equal", "valueInt": 2},
            "not_tag_1": {"path": ["tag"], "operator": "NotEqual", "valueInt": 1},
            "tag_1_or_2": {"operator": "Or", "operands": [{"path": ["tag"], "operator": "Equal", "valueInt": 1}, {"path": ["tag"], "operator": "Equal", "valueInt": 2}]},
            "tag_not_1_or_2": {"operator": "Or", "operands": [{"path": ["tag"], "operator": "Equal", "valueInt": 0}, {"path": ["tag"], "operator": "Equal", "valueInt": 3}]},
            "age_lt_30": {"path": ["age"], "operator": "LessThan", "valueInt": 30},
            "age_plus_3_lt_33": {"path": ["age"], "operator": "LessThan", "valueInt": 30},
            "age_any": {},
            "range_standard": {"operator": "And", "operands": [{"path": ["age"], "operator": "GreaterThanEqual", "valueInt": 20}, {"path": ["age"], "operator": "LessThanEqual", "valueInt": 30}, {"path": ["price"], "operator": "GreaterThan", "valueNumber": 500.0}]},
            "cat_in": {"operator": "Or", "operands": [{"path": ["tag"], "operator": "Equal", "valueInt": 1}, {"path": ["tag"], "operator": "Equal", "valueInt": 2}, {"path": ["tag"], "operator": "Equal", "valueInt": 3}]},
            "cat_or": {"operator": "Or", "operands": [{"path": ["tag"], "operator": "Equal", "valueInt": 1}, {"path": ["tag"], "operator": "Equal", "valueInt": 2}, {"path": ["tag"], "operator": "Equal", "valueInt": 3}]}
        }
        result = mapping.get(expr_type, {})
        return result if result else {}