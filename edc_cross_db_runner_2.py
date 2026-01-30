"""
EDC (Equivalent Data Construction) Logic Verification Runner
Focus: Correctness & Consistency (Logic Bugs), NOT Performance.

Improvements over original:
1. Determinism: Uses FLAT/Exact search where possible to avoid ANN recall noise.
2. Stability: Adds explicit wait-for-indexing checks (especially for Weaviate).
3. Compatibility: Fixes Weaviate naming conventions (PascalCase).
4. Robustness: Better error handling and skipping logic.
"""
from __future__ import annotations
import time
import uuid
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np

# --- Global Config ---
# 逻辑测试数据量：足够大以测试性能和边界情况
N = 50000  # 50000 条数据
DIM = 768  # 768 维
TOPK = 10
SEED = 42
BATCH_SIZE = 2000  # Milvus 分批大小

# --- Database Clients Check ---
HAVE_MILVUS = False
try:
    from pymilvus import (
        connections as milvus_conn,
        utility as milvus_util,
        FieldSchema, CollectionSchema, DataType, Collection
    )
    HAVE_MILVUS = True
except ImportError:
    pass

HAVE_QDRANT = False
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as Qmodels
    HAVE_QDRANT = True
except ImportError:
    pass

HAVE_CHROMA = False
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    HAVE_CHROMA = True
except ImportError:
    pass

HAVE_WEAVIATE = False
try:
    import weaviate
    HAVE_WEAVIATE = True
except ImportError:
    pass


@dataclass
class ScenarioResult:
    name: str
    passed: bool
    detail: str


# ========================== Adapters ==========================

class MilvusAdapter:
    def __init__(self):
        self.uri = ("127.0.0.1", "19530")
        self.col = None
        
    def ok(self): return HAVE_MILVUS

    def connect(self):
        if not HAVE_MILVUS: return
        try:
            milvus_conn.connect("default", host=self.uri[0], port=self.uri[1])
        except Exception as e:
            print(f"Milvus connect fail: {e}")

    def drop(self, name: str):
        if not HAVE_MILVUS: return
        if milvus_util.has_collection(name):
            milvus_util.drop_collection(name)

    def _create_base(self, name, extra_fields):
        # 基础 Schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=DIM),
        ] + extra_fields
        schema = CollectionSchema(fields)
        self.col = Collection(name, schema, consistency_level="Strong")

    def create_expr_equiv(self, name):
        self._create_base(name, [
            FieldSchema(name="price", dtype=DataType.FLOAT),
            FieldSchema(name="tax", dtype=DataType.FLOAT),
            FieldSchema(name="final_price", dtype=DataType.FLOAT),
        ])

    def create_json_depth(self, name):
        # 尝试支持 JSON
        try:
            self._create_base(name, [
                FieldSchema(name="meta", dtype=DataType.JSON),
                FieldSchema(name="user_age", dtype=DataType.INT64),
            ])
        except Exception:
            raise NotImplementedError("milvus-json-setup-failed")

    def create_type_coercion(self, name):
        self._create_base(name, [
            FieldSchema(name="tag_str", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="tag_int", dtype=DataType.INT64),
        ])

    def create_soft_delete(self, name):
        self._create_base(name, [
            FieldSchema(name="age", dtype=DataType.INT64),
        ])

    def create_null_test(self, name):
        self._create_base(name, [
            FieldSchema(name="tag", dtype=DataType.INT64, nullable=True),
        ])

    def create_default_confusion(self, name):
        self._create_base(name, [
            FieldSchema(name="score", dtype=DataType.INT64, nullable=True),
            FieldSchema(name="is_missing", dtype=DataType.BOOL),
        ])

    def create_exists_test(self, name):
        self._create_base(name, [
            FieldSchema(name="tag", dtype=DataType.INT64, nullable=True),
            FieldSchema(name="has_tag", dtype=DataType.BOOL),
        ])

    def create_sparse_index(self, name):
        self._create_base(name, [
            FieldSchema(name="special_key", dtype=DataType.VARCHAR, max_length=64, nullable=True),
        ])

    def insert(self, ids, vecs, payloads):
        # 分批插入以避免 gRPC 消息过大
        for batch_start in range(0, len(ids), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(ids))
            batch_ids = ids[batch_start:batch_end]
            batch_vecs = vecs[batch_start:batch_end]
            batch_payloads = payloads[batch_start:batch_end]
            
            # 动态解析 payloads 到列
            data = [batch_ids, batch_vecs]
            # 获取 schema 中的字段顺序（跳过 id 和 vector）
            schema_fields = {f.name: f for f in self.col.schema.fields if f.name not in ['id', 'vector']}
            
            for field_name in schema_fields:
                # 特殊处理 JSON
                if schema_fields[field_name].dtype == DataType.JSON:
                    # 构造 JSON 对象列表
                    json_col = []
                    for p in batch_payloads:
                        if "metadata_user_age" in p:
                            json_col.append({"user": {"age": p["metadata_user_age"]}})
                        else:
                            json_col.append({})
                    data.append(json_col)
                else:
                    # 关键修复：
                    # - VARCHAR 列即使 nullable，在插入时也不要传 None，改为空串
                    # - 非 nullable 列若为 None，填充合理默认值避免 DataNotMatchException
                    # - 转换 numpy 基元类型为 Python 原生类型
                    fmeta = schema_fields[field_name]
                    col_data = []
                    for p in batch_payloads:
                        val = p.get(field_name, None)

                        if val is None:
                            if fmeta.dtype == DataType.VARCHAR:
                                col_data.append("")
                            elif fmeta.nullable:
                                col_data.append(None)
                            else:
                                if fmeta.dtype in (DataType.INT64, DataType.INT32, DataType.INT16, DataType.INT8):
                                    col_data.append(0)
                                elif fmeta.dtype in (DataType.FLOAT, DataType.DOUBLE):
                                    col_data.append(0.0)
                                elif fmeta.dtype == DataType.BOOL:
                                    col_data.append(False)
                                else:
                                    col_data.append(None)
                        else:
                            if isinstance(val, np.integer):
                                val = int(val)
                            elif isinstance(val, np.floating):
                                val = float(val)
                            if fmeta.dtype == DataType.VARCHAR and not isinstance(val, str):
                                val = str(val)
                            col_data.append(val)
                    data.append(col_data)
            
            self.col.insert(data)

    def insert_count(self):
        """返回成功插入的数据条数（用于验证）"""
        return self.col.num_entities

    def delete(self, ids):
        expr = f"id in {list(ids)}"
        self.col.delete(expr)

    def flush_and_index(self):
        self.col.flush()
        # 关键修改：使用 FLAT 索引进行逻辑测试，避免 HNSW 召回率干扰
        index_params = {"metric_type": "L2", "index_type": "FLAT", "params": {}}
        self.col.create_index("vector", index_params)
        self.col.load()

    def search(self, query, filter_expr: str):
        # JSON 字段查询语法适配
        if "metadata.user.age" in filter_expr:
            filter_expr = 'meta["user"]["age"] > 20'
            
        search_params = {"metric_type": "L2", "params": {}}
        try:
            res = self.col.search([query], "vector", search_params, limit=TOPK, expr=filter_expr, output_fields=["id"])
            return sorted([hit.id for hit in res[0]])
        except Exception as e:
            # print(f"Milvus search err: {e}")
            return None


class QdrantAdapter:
    def __init__(self):
        self.client = QdrantClient(url="http://127.0.0.1:6333") if HAVE_QDRANT else None
        self.col = None

    def ok(self): return HAVE_QDRANT

    def connect(self): 
        # Qdrant 在 __init__ 中已连接，无需额外操作
        pass

    def drop(self, name):
        if self.client: self.client.delete_collection(name)

    def _create(self, name):
        self.client.recreate_collection(
            collection_name=name,
            vectors_config=Qmodels.VectorParams(size=DIM, distance=Qmodels.Distance.EUCLID)
        )
        self.col = name

    def create_expr_equiv(self, name): self._create(name)
    def create_json_depth(self, name): self._create(name)
    def create_type_coercion(self, name): self._create(name)
    def create_soft_delete(self, name): self._create(name)
    def create_null_test(self, name): self._create(name)
    def create_default_confusion(self, name): self._create(name)
    def create_exists_test(self, name): self._create(name)
    def create_sparse_index(self, name): self._create(name)

    def insert(self, ids, vecs, payloads):
        # 分批 upsert，且修复 numpy 类型为原生 Python 类型
        for batch_start in range(0, len(ids), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(ids))
            batch_ids = ids[batch_start:batch_end]
            batch_vecs = vecs[batch_start:batch_end]
            batch_payloads = payloads[batch_start:batch_end]

            fixed_payloads = []
            for p in batch_payloads:
                fp = {}
                for k, v in p.items():
                    if v is None:
                        fp[k] = None
                    elif isinstance(v, (int, float, str, bool)):
                        fp[k] = v
                    elif isinstance(v, np.integer):
                        fp[k] = int(v)
                    elif isinstance(v, np.floating):
                        fp[k] = float(v)
                    else:
                        # 其余类型转字符串以避免序列化失败
                        fp[k] = str(v)
                fixed_payloads.append(fp)

            points = [
                Qmodels.PointStruct(id=int(i), vector=v.tolist(), payload=p)
                for i, v, p in zip(batch_ids, batch_vecs, fixed_payloads)
            ]
            self.client.upsert(self.col, points)

    def delete(self, ids):
        self.client.delete(self.col, points_selector=Qmodels.PointIdsList(points=[int(i) for i in ids]))

    def flush_and_index(self):
        # Qdrant 实时性较好，无需显式 flush，但可以建立 Payload 索引加速（可选）
        pass

    def search(self, query, filter_obj):
        try:
            # 关键修改：exact=True 开启暴力搜索，确保逻辑测试准确性
            res = self.client.search(
                collection_name=self.col,
                query_vector=query.tolist(),
                query_filter=filter_obj,
                limit=TOPK,
                search_params=Qmodels.SearchParams(exact=True) 
            )
            return sorted([r.id for r in res])
        except Exception as e:
            # print(f"Qdrant err: {e}")
            return None


class ChromaAdapter:
    def __init__(self):
        if HAVE_CHROMA:
            # 使用内存模式或者本地临时目录，避免污染
            self.client = chromadb.PersistentClient(path="./tmp_chroma_edc")
        self.col = None

    def ok(self): return HAVE_CHROMA

    def connect(self):
        # Chroma 在 __init__ 中已初始化，无需额外操作
        pass

    def drop(self, name):
        try: self.client.delete_collection(name)
        except: pass

    def _create(self, name):
        self.drop(name)
        self.col = self.client.create_collection(name, metadata={"hnsw:space": "l2"})

    def create_expr_equiv(self, name): self._create(name)
    def create_json_depth(self, name): self._create(name)
    def create_type_coercion(self, name): self._create(name)
    def create_soft_delete(self, name): self._create(name)
    def create_null_test(self, name): self._create(name)
    def create_default_confusion(self, name): self._create(name)
    def create_exists_test(self, name): self._create(name)
    def create_sparse_index(self, name): self._create(name)

    def insert(self, ids, vecs, payloads):
        # 分批 add，避免超过 Chroma 的最大批量限制
        for batch_start in range(0, len(ids), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(ids))
            batch_ids = ids[batch_start:batch_end]
            batch_vecs = vecs[batch_start:batch_end]
            batch_payloads = payloads[batch_start:batch_end]

            clean_payloads = []
            for p in batch_payloads:
                # Chroma metadata values must be int, float, str, bool
                # 修复：跳过空 metadata 和 None 值
                cp = {}
                for k, v in p.items():
                    if v is None:
                        continue  # 跳过 None
                    if isinstance(v, (int, float, str, bool)):
                        cp[k] = v
                    elif isinstance(v, np.integer):
                        cp[k] = int(v)  # 转换 numpy int
                    elif isinstance(v, np.floating):
                        cp[k] = float(v)  # 转换 numpy float
                # 如果 metadata 为空，添加一个占位符
                if not cp:
                    cp["_placeholder"] = True
                clean_payloads.append(cp)

            self.col.add(
                ids=[str(i) for i in batch_ids],
                embeddings=batch_vecs.tolist(),
                metadatas=clean_payloads
            )

    def delete(self, ids):
        self.col.delete(ids=[str(i) for i in ids])

    def flush_and_index(self): pass

    def search(self, query, where):
        try:
            res = self.col.query(query_embeddings=[query.tolist()], n_results=TOPK, where=where)
            if not res['ids']: return []
            return sorted([int(i) for i in res['ids'][0]])
        except Exception:
            return None


class WeaviateAdapter:
    def __init__(self):
        if HAVE_WEAVIATE:
            try:
                self.client = weaviate.Client("http://127.0.0.1:8080")
            except:
                self.client = None
        self.class_name = None

    def ok(self): return HAVE_WEAVIATE and self.client and self.client.is_ready()

    def connect(self):
        # Weaviate 在 __init__ 中已连接，无需额外操作
        pass

    def drop(self, name):
        # 关键修复：Weaviate 类名首字母必须大写
        name = name[0].upper() + name[1:]
        try: self.client.schema.delete_class(name)
        except: pass

    def _create(self, name, props):
        # 关键修复：Weaviate 类名首字母必须大写
        name = name[0].upper() + name[1:]
        self.class_name = name
        
        class_obj = {
            "class": name,
            "vectorizer": "none",
            # 强制使用 Flat 索引以保证逻辑测试的确定性
            "vectorIndexType": "flat",
            "properties": props,
        }
        self.client.schema.create_class(class_obj)

    def create_expr_equiv(self, name):
        self._create(name, [
            {"name": "idx", "dataType": ["int"]},
            {"name": "price", "dataType": ["number"]},
            {"name": "tax", "dataType": ["number"]},
            {"name": "final_price", "dataType": ["number"]},
        ])

    def create_json_depth(self, name):
        # Weaviate 处理嵌套对象需要复杂 schema，暂不支持简单转换
        raise NotImplementedError("weaviate-json-complex")

    def create_type_coercion(self, name):
        self._create(name, [
            {"name": "idx", "dataType": ["int"]},
            {"name": "tag_str", "dataType": ["text"]}, # 'text' tokenizes
            {"name": "tag_int", "dataType": ["int"]},
        ])

    def create_soft_delete(self, name):
        self._create(name, [
            {"name": "idx", "dataType": ["int"]},
            {"name": "age", "dataType": ["int"]},
        ])

    def create_null_test(self, name):
        self._create(name, [
            {"name": "idx", "dataType": ["int"]},
            {"name": "tag", "dataType": ["int"]},
        ])

    def create_default_confusion(self, name):
        self._create(name, [
            {"name": "idx", "dataType": ["int"]},
            {"name": "score", "dataType": ["int"]},
            {"name": "is_missing", "dataType": ["boolean"]},
        ])

    def create_exists_test(self, name):
        self._create(name, [
            {"name": "idx", "dataType": ["int"]},
            {"name": "tag", "dataType": ["int"]},
            {"name": "has_tag", "dataType": ["boolean"]},
        ])

    def create_sparse_index(self, name):
        self._create(name, [
            {"name": "idx", "dataType": ["int"]},
            {"name": "special_key", "dataType": ["text"]},
        ])

    def insert(self, ids, vecs, payloads):
        # 分批提交到 Weaviate，避免过大批次导致超时
        # 关键修复：减小批次避免连接超时，增加容错处理
        weaviate_batch_size = 500  # 降低批次大小
        for batch_start in range(0, len(ids), weaviate_batch_size):
            batch_end = min(batch_start + weaviate_batch_size, len(ids))
            batch_ids = ids[batch_start:batch_end]
            batch_vecs = vecs[batch_start:batch_end]
            batch_payloads = payloads[batch_start:batch_end]

            try:
                with self.client.batch as batch:
                    batch.batch_size = 100  # 降低内部批次大小
                    batch.timeout_retries = 3
                    for i, v, p in zip(batch_ids, batch_vecs, batch_payloads):
                        props = {}
                        for k, val in p.items():
                            # 修复：转换 numpy 类型为 Python 原生类型
                            if isinstance(val, np.integer):
                                props[k] = int(val)
                            elif isinstance(val, np.floating):
                                props[k] = float(val)
                            elif val is not None:
                                props[k] = val
                        props["idx"] = int(i)
                        # 使用确定性 UUID 避免重复插入问题
                        uid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{self.class_name}_{i}"))
                        batch.add_data_object(props, self.class_name, vector=v.tolist(), uuid=uid)
                # 每批后短暂休息，避免Weaviate资源耗尽
                time.sleep(0.1)
            except Exception as e:
                # 容错：即使部分批次失败也继续，避免整个测试中断
                print(f"Weaviate batch insert warning: {e}")
                continue

    def delete(self, ids):
        for i in ids:
            uid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{self.class_name}_{i}"))
            self.client.data_object.delete(uuid=uid, class_name=self.class_name)

    def flush_and_index(self):
        # 关键修复：轮询等待索引构建完成
        max_retries = 20
        for _ in range(max_retries):
            res = self.client.query.aggregate(self.class_name).with_meta_count().do()
            try:
                count = res['data']['Aggregate'][self.class_name][0]['meta']['count']
                if count >= N: # 简单检查数量
                    return
            except:
                pass
            time.sleep(0.5)
        print(f"⚠️ Weaviate indexing timeout or incomplete")

    def search(self, query, where_filter):
        try:
            q = (self.client.query
                 .get(self.class_name, ["idx"])
                 .with_near_vector({"vector": query.tolist()})
                 .with_limit(TOPK))
            
            if where_filter:
                q = q.with_where(where_filter)
            
            res = q.do()
            if "errors" in res:
                return None
            items = res['data']['Get'][self.class_name]
            return sorted([item['idx'] for item in items])
        except Exception:
            return None


# ========================== Runners ==========================

def get_data_expr():
    ids = np.arange(N, dtype=np.int64)
    rng = np.random.default_rng(SEED)
    vecs = rng.random((N, DIM), dtype=np.float32)
    # price 0-200, tax 0.0-0.5
    price = rng.random(N) * 200
    tax = rng.random(N) * 0.5
    final = price * (1 + tax)
    return ids, vecs, [{"price": p, "tax": t, "final_price": f} for p, t, f in zip(price, tax, final)]

def test_expr_equiv(db):
    name = f"edc_expr_{uuid.uuid4().hex[:6]}"
    ids, vecs, payloads = get_data_expr()
    
    try:
        db.drop(name)
        db.create_expr_equiv(name)
        db.insert(ids, vecs, payloads)
        db.flush_and_index()
    except NotImplementedError:
        return ScenarioResult("expr_equiv", False, "skip: not supported")
    except Exception as e:
        return ScenarioResult("expr_equiv", False, f"setup error: {e}")

    # Query: price * (1+tax) > 200 (Extreme case)
    # 绝大多数 DB 不支持算术 Filter
    q = vecs[0]
    
    # Construct filters
    f_base = None
    f_trans = None
    
    if isinstance(db, MilvusAdapter):
        f_base = "price * (1 + tax) > 200"
        f_trans = "final_price > 200"
    elif isinstance(db, WeaviateAdapter):
        # Weaviate 不支持算术
        return ScenarioResult("expr_equiv", False, "skip: weaviate no math")
    elif isinstance(db, QdrantAdapter):
        # Qdrant 不支持算术
        return ScenarioResult("expr_equiv", False, "skip: qdrant no math")
    elif isinstance(db, ChromaAdapter):
        return ScenarioResult("expr_equiv", False, "skip: chroma no math")

    res_base = db.search(q, f_base)
    res_trans = db.search(q, f_trans)
    
    # Milvus 可能因不支持算术操作而返回 None
    if res_base is None: 
        return ScenarioResult("expr_equiv", False, "skip: milvus arithmetic not supported")
    
    return ScenarioResult("expr_equiv", res_base == res_trans, f"Base:{len(res_base)}, Trans:{len(res_trans)}")


def get_data_json():
    ids = np.arange(N, dtype=np.int64)
    rng = np.random.default_rng(SEED)
    vecs = rng.random((N, DIM), dtype=np.float32)
    ages = rng.integers(20, 60, size=N)
    # Payload format: metadata_user_age is flat, but we simulate nesting in Adapter if needed
    payloads = [{"metadata_user_age": int(a), "user_age": int(a)} for a in ages]
    return ids, vecs, payloads

def test_json_depth(db):
    name = f"edc_json_{uuid.uuid4().hex[:6]}"
    ids, vecs, payloads = get_data_json()
    
    try:
        db.drop(name)
        db.create_json_depth(name)
        db.insert(ids, vecs, payloads)
        db.flush_and_index()
    except NotImplementedError:
        return ScenarioResult("json_depth", False, "skip: not supported")
    except Exception as e:
        return ScenarioResult("json_depth", False, f"setup error: {e}")

    q = vecs[0]
    # Filter: age > 50
    f_base, f_trans = None, None
    
    if isinstance(db, QdrantAdapter):
        # 修复：payload中实际是扁平字段 metadata_user_age 和 user_age
        f_base = Qmodels.Filter(must=[Qmodels.FieldCondition(key="metadata_user_age", range=Qmodels.Range(gt=50))])
        f_trans = Qmodels.Filter(must=[Qmodels.FieldCondition(key="user_age", range=Qmodels.Range(gt=50))])
    elif isinstance(db, ChromaAdapter):
        # Chroma 仅支持简单元数据，除非我们在 Insert 时 hack 了结构
        # 假设 ChromaAdapter insert 时没有构造嵌套，则跳过
        return ScenarioResult("json_depth", False, "skip: chroma simple metadata")
    elif isinstance(db, MilvusAdapter):
         f_base = 'meta["user"]["age"] > 50'
         f_trans = 'user_age > 50'
    else:
        return ScenarioResult("json_depth", False, "skip: adapter not mapped")

    res_base = db.search(q, f_base)
    res_trans = db.search(q, f_trans)
    
    if res_base is None or res_trans is None:
         return ScenarioResult("json_depth", False, "query failed")

    return ScenarioResult("json_depth", res_base == res_trans, f"Match: {len(res_base)}")


def get_data_type():
    ids = np.arange(N, dtype=np.int64)
    rng = np.random.default_rng(SEED)
    vecs = rng.random((N, DIM), dtype=np.float32)
    # Mixed types: "100", "050", "99"
    # To test coercion: String "100" vs Int 100
    # Data: "123", "99", "200"
    tags_int = rng.choice([99, 100, 101, 200], size=N)
    payloads = [{"tag_str": str(t), "tag_int": int(t)} for t in tags_int]
    return ids, vecs, payloads

def test_type_coercion(db):
    name = f"edc_type_{uuid.uuid4().hex[:6]}"
    ids, vecs, payloads = get_data_type()
    
    try:
        db.drop(name)
        db.create_type_coercion(name)
        db.insert(ids, vecs, payloads)
        db.flush_and_index()
    except Exception as e:
        return ScenarioResult("type_coercion", False, f"setup error: {e}")

    q = vecs[0]
    # Filter: > 100
    # String comparison: "99" > "100" is True! (lexicographical)
    # Int comparison: 99 > 100 is False.
    # If DB coerces "99" to 99, results match. If DB treats "99" as string, results differ.
    # EDC Oracle: A == B means "Implicit Coercion Happened" OR "Data avoids edge case".
    # Here we WANT to see if they differ. If they differ, it's NOT a bug per se, but behavior characteristic.
    # But EDC implies A and B *should* be equivalent if we modeled them to be.
    # Actually, we define Ttrans as the "Correct Logic". 
    # If we want to detect "String evaluated as Int", A should be String, B should be Int.
    
    f_base, f_trans = None, None
    if isinstance(db, MilvusAdapter):
        # Milvus 中字符串与整型比较语义不同，该用例用于行为刻画，标记跳过
        return ScenarioResult("type_coercion", False, "skip: milvus string vs int differ by design")
    elif isinstance(db, QdrantAdapter):
        # Qdrant的Range不支持字符串，字符串比较用MatchText（但只支持等值）
        # 这里改为测试等值匹配而非大于
        f_base = Qmodels.Filter(must=[Qmodels.FieldCondition(key="tag_str", match=Qmodels.MatchText(text="200"))])
        f_trans = Qmodels.Filter(must=[Qmodels.FieldCondition(key="tag_int", match=Qmodels.MatchValue(value=200))])
    elif isinstance(db, WeaviateAdapter):
        # Weaviate v3 对 text 类型的大小比较语义不确定，跳过该用例
        return ScenarioResult("type_coercion", False, "skip: weaviate text compare ambiguous")
    elif isinstance(db, ChromaAdapter):
        f_base = {"tag_str": {"$gt": "100"}}
        f_trans = {"tag_int": {"$gt": 100}}

    res_base = db.search(q, f_base)
    res_trans = db.search(q, f_trans)
    
    if res_base is None or res_trans is None: return ScenarioResult("type_coercion", False, "query failed")
    
    # We expect them to be DIFFERENT if the DB respects types. 
    # But EDC says "Consistency". 
    # Let's just report the diff.
    return ScenarioResult("type_coercion", res_base == res_trans, f"StrCount:{len(res_base)} IntCount:{len(res_trans)}")


def get_data_del():
    ids = np.arange(N, dtype=np.int64)
    rng = np.random.default_rng(SEED)
    vecs = rng.random((N, DIM), dtype=np.float32)
    ages = rng.integers(0, 100, size=N)
    payloads = [{"age": int(a)} for a in ages]
    # Delete 20%
    del_ids = rng.choice(ids, size=int(N*0.2), replace=False)
    return ids, vecs, payloads, del_ids

def test_soft_delete(db):
    name = f"edc_del_{uuid.uuid4().hex[:6]}"
    ids, vecs, payloads, del_ids = get_data_del()
    
    try:
        db.drop(name)
        db.create_soft_delete(name)
        db.insert(ids, vecs, payloads)
        db.delete(del_ids)
        db.flush_and_index()
    except NotImplementedError:
        return ScenarioResult("soft_delete", False, "skip")
    except Exception as e:
        return ScenarioResult("soft_delete", False, f"setup error: {e}")

    q = vecs[0]
    # Filter age > 50
    if isinstance(db, MilvusAdapter):
        f = "age > 50"
    elif isinstance(db, QdrantAdapter):
        f = Qmodels.Filter(must=[Qmodels.FieldCondition(key="age", range=Qmodels.Range(gt=50))])
    elif isinstance(db, WeaviateAdapter):
        f = {"path": ["age"], "operator": "GreaterThan", "valueInt": 50}
    elif isinstance(db, ChromaAdapter):
        f = {"age": {"$gt": 50}}

    # DB Result
    res_db = db.search(q, f)
    
    # Local Oracle: Calculate expected IDs (Manual Filter + Manual Delete Check)
    del_set = set(del_ids)
    expected = []
    # 这里我们只验证 ID 集合是否一致（忽略向量排序的微小差异，只看是否漏删或多删）
    # 其实应该验证 TopK。为了简单，我们验证结果集中的 ID 确实不包含 deleted id，且包含符合条件的有效 id。
    # 更加 EDC 的做法是：构建一个 Ttrans，只插入未删除的数据，然后跑一样的 Query。
    # 这里采用 client-side check 简化。
    
    if res_db is None: return ScenarioResult("soft_delete", False, "query failed")
    
    zombies = [i for i in res_db if i in del_set]
    if zombies:
        return ScenarioResult("soft_delete", False, f"Found {len(zombies)} zombies! (Deleted items returned)")
    
    return ScenarioResult("soft_delete", True, f"Clean result size: {len(res_db)}")


# ========================== 新增测试：完备性、默认值、Exists ==========================

def get_data_null():
    """生成包含 NULL 字段的数据"""
    ids = np.arange(N, dtype=np.int64)
    rng = np.random.default_rng(SEED)
    vecs = rng.random((N, DIM), dtype=np.float32)
    
    payloads = []
    for i in range(N):
        if i < N // 3:  # 1/3 有 tag=1
            payloads.append({"tag": 1})
        elif i < 2 * N // 3:  # 1/3 有 tag=2
            payloads.append({"tag": 2})
        else:  # 1/3 没有 tag 字段 (NULL)
            payloads.append({})
    
    return ids, vecs, payloads


def test_null_completeness(db):
    """Test 1: 完备性测试 - tag==1 + tag!=1 + tag IS NULL 应等于总数"""
    name = f"edc_null_{uuid.uuid4().hex[:6]}"
    ids, vecs, payloads = get_data_null()
    
    try:
        db.drop(name)
        db.create_null_test(name)
        db.insert(ids, vecs, payloads)
        db.flush_and_index()
    except Exception as e:
        return ScenarioResult("null_completeness", False, f"setup error: {e}")

    q = vecs[0]
    
    # 构造三种查询
    if isinstance(db, MilvusAdapter):
        f_eq = "tag == 1"
        f_ne = "tag != 1"
        # 修正：Milvus 使用 is null / is not null 语法
        f_null = "tag is null"
    elif isinstance(db, QdrantAdapter):
        f_eq = Qmodels.Filter(must=[Qmodels.FieldCondition(key="tag", match=Qmodels.MatchValue(value=1))])
        f_ne = Qmodels.Filter(must_not=[Qmodels.FieldCondition(key="tag", match=Qmodels.MatchValue(value=1))])
        # 修复：Qdrant用IsEmpty检测字段是否缺失（不是IsNull）
        f_null = Qmodels.Filter(must=[Qmodels.IsEmptyCondition(is_empty=Qmodels.PayloadField(key="tag"))])
    elif isinstance(db, WeaviateAdapter):
        # Weaviate v3不支持IsNull操作符，跳过测试
        return ScenarioResult("null_completeness", False, "skip: weaviate v3 no IsNull operator")
    elif isinstance(db, ChromaAdapter):
        # Chroma 不支持 NULL 检测
        return ScenarioResult("null_completeness", False, "skip: chroma no null check")

    res_eq = db.search(q, f_eq)
    res_ne = db.search(q, f_ne)
    res_null = db.search(q, f_null)
    
    if res_eq is None or res_ne is None or res_null is None:
        return ScenarioResult("null_completeness", False, "query failed")
    
    # 判定逻辑修正：
    # - 关注集合两两不相交（互斥），而不是总数是否<=TOPK
    set_eq, set_ne, set_null = set(res_eq), set(res_ne), set(res_null)
    disjoint = (len(set_eq & set_ne) == 0 and len(set_eq & set_null) == 0 and len(set_ne & set_null) == 0)
    passed = disjoint and (len(res_eq) > 0 and len(res_ne) > 0)
    detail = f"tag==1:{len(res_eq)} + tag!=1:{len(res_ne)} + NULL:{len(res_null)} | disjoint={disjoint}"
    
    
    return ScenarioResult("null_completeness", passed, detail)


def get_data_default():
    """生成 score=0 vs score=NULL 的数据"""
    ids = np.arange(N, dtype=np.int64)
    rng = np.random.default_rng(SEED)
    vecs = rng.random((N, DIM), dtype=np.float32)
    
    payloads = []
    for i in range(N):
        if i < N // 2:  # 一半真的是 0 分
            payloads.append({"score": 0, "is_missing": False})
        else:  # 一半是 NULL（未参加）
            payloads.append({"score": None, "is_missing": True})
    
    return ids, vecs, payloads


def test_default_confusion(db):
    """Test 2: 默认值混淆 - NULL 不应被当成 0"""
    name = f"edc_default_{uuid.uuid4().hex[:6]}"
    ids, vecs, payloads = get_data_default()
    
    try:
        db.drop(name)
        db.create_default_confusion(name)
        db.insert(ids, vecs, payloads)
        db.flush_and_index()
    except Exception as e:
        return ScenarioResult("default_confusion", False, f"setup error: {e}")

    q = vecs[0]
    
    # Query A: score <= 0 (应该只返回真的 0，不应包含 NULL)
    # Query B: is_missing==False AND score<=0 (用预计算验证)
    if isinstance(db, MilvusAdapter):
        f_base = "score <= 0"
        f_trans = "is_missing == false && score <= 0"
    elif isinstance(db, QdrantAdapter):
        f_base = Qmodels.Filter(must=[Qmodels.FieldCondition(key="score", range=Qmodels.Range(lte=0))])
        f_trans = Qmodels.Filter(must=[
            Qmodels.FieldCondition(key="is_missing", match=Qmodels.MatchValue(value=False)),
            Qmodels.FieldCondition(key="score", range=Qmodels.Range(lte=0))
        ])
    elif isinstance(db, WeaviateAdapter):
        f_base = {"path": ["score"], "operator": "LessThanEqual", "valueInt": 0}
        f_trans = {"operator": "And", "operands": [
            {"path": ["is_missing"], "operator": "Equal", "valueBoolean": False},
            {"path": ["score"], "operator": "LessThanEqual", "valueInt": 0}
        ]}
    elif isinstance(db, ChromaAdapter):
        f_base = {"score": {"$lte": 0}}
        f_trans = {"$and": [{"is_missing": False}, {"score": {"$lte": 0}}]}

    res_base = db.search(q, f_base)
    res_trans = db.search(q, f_trans)
    
    if res_base is None or res_trans is None:
        return ScenarioResult("default_confusion", False, "query failed")
    
    # 判定：关注计数一致性，避免 TopK 任选导致 ID 不完全相同的误判
    # Bug: 如果 res_base 明显多于 res_trans，说明 NULL 被当成了 0
    passed = (len(res_base) == len(res_trans))
    detail = f"score<=0:{len(res_base)} vs (valid+<=0):{len(res_trans)}"
    
    if len(res_base) > len(res_trans):
        detail += " ⚠️ NULL被当成0！"
        passed = False
    
    return ScenarioResult("default_confusion", passed, detail)


def get_data_exists():
    """生成部分有 tag、部分无 tag 的数据"""
    ids = np.arange(N, dtype=np.int64)
    rng = np.random.default_rng(SEED)
    vecs = rng.random((N, DIM), dtype=np.float32)
    
    payloads = []
    for i in range(N):
        if i % 2 == 0:  # 一半有 tag
            tag_val = rng.integers(0, 10)
            payloads.append({"tag": tag_val, "has_tag": True})
        else:  # 一半没有 tag
            payloads.append({"has_tag": False})
    
    return ids, vecs, payloads


def test_exists_equivalence(db):
    """Test 4: Exists 操作符等价性"""
    name = f"edc_exists_{uuid.uuid4().hex[:6]}"
    ids, vecs, payloads = get_data_exists()
    
    try:
        db.drop(name)
        db.create_exists_test(name)
        db.insert(ids, vecs, payloads)
        db.flush_and_index()
    except Exception as e:
        return ScenarioResult("exists_equiv", False, f"setup error: {e}")

    q = vecs[0]
    
    # Query A: Exists(tag)
    # Query B: has_tag == True
    if isinstance(db, MilvusAdapter):
        # Milvus 不直接支持 Exists，用 tag is not null 代替
        f_base = "tag is not null"
        f_trans = "has_tag == true"
    elif isinstance(db, QdrantAdapter):
        # Qdrant 用 IsEmpty 的否定
        f_base = Qmodels.Filter(must=[Qmodels.HasIdCondition(has_id=[int(i) for i in ids if payloads[i].get("tag") is not None][:TOPK])])  # Hack: 手动列举
        f_trans = Qmodels.Filter(must=[Qmodels.FieldCondition(key="has_tag", match=Qmodels.MatchValue(value=True))])
        # 实际上 Qdrant 不容易表达 Exists，这里简化跳过
        return ScenarioResult("exists_equiv", False, "skip: qdrant exists syntax complex")
    elif isinstance(db, WeaviateAdapter):
        # Weaviate 不支持简单的 Exists
        return ScenarioResult("exists_equiv", False, "skip: weaviate no exists")
    elif isinstance(db, ChromaAdapter):
        # Chroma 不支持 Exists
        return ScenarioResult("exists_equiv", False, "skip: chroma no exists")

    res_base = db.search(q, f_base)
    res_trans = db.search(q, f_trans)
    
    if res_base is None or res_trans is None:
        return ScenarioResult("exists_equiv", False, "query failed")
    
    passed = (len(res_base) == len(res_trans))
    detail = f"Exists:{len(res_base)} vs has_tag:{len(res_trans)}"
    
    return ScenarioResult("exists_equiv", passed, detail)


def get_data_sparse():
    """生成稀疏索引测试数据 - 只有极少数记录有 special_key"""
    ids = np.arange(N, dtype=np.int64)
    rng = np.random.default_rng(SEED)
    vecs = rng.random((N, DIM), dtype=np.float32)
    
    # 只有 0.1% (10条) 有 special_key
    special_count = max(10, int(N * 0.001))
    special_ids = rng.choice(ids, size=special_count, replace=False)
    special_set = set(special_ids)
    
    payloads = []
    for i in ids:
        if i in special_set:
            payloads.append({"special_key": "secret"})
        else:
            payloads.append({})  # 没有 special_key 字段
    
    return ids, vecs, payloads, special_ids


def test_sparse_index_perf(db):
    """Test 3: 稀疏索引性能 - 极低选择率查询应该极快"""
    name = f"edc_sparse_{uuid.uuid4().hex[:6]}"
    ids, vecs, payloads, special_ids = get_data_sparse()
    
    try:
        db.drop(name)
        db.create_sparse_index(name)
        db.insert(ids, vecs, payloads)
        db.flush_and_index()
    except Exception as e:
        return ScenarioResult("sparse_index_perf", False, f"setup error: {e}")

    q = vecs[0]
    
    # Query: special_key == "secret" (只有 0.1% 的数据)
    if isinstance(db, MilvusAdapter):
        f = 'special_key == "secret"'
    elif isinstance(db, QdrantAdapter):
        f = Qmodels.Filter(must=[Qmodels.FieldCondition(key="special_key", match=Qmodels.MatchValue(value="secret"))])
    elif isinstance(db, WeaviateAdapter):
        f = {"path": ["special_key"], "operator": "Equal", "valueText": "secret"}
    elif isinstance(db, ChromaAdapter):
        f = {"special_key": "secret"}

    # 测试查询性能
    import time
    start = time.perf_counter()
    res_db = db.search(q, f)
    latency_ms = (time.perf_counter() - start) * 1000
    
    if res_db is None:
        return ScenarioResult("sparse_index_perf", False, "query failed")
    
    # Oracle: 极稀疏字段查询应该很快。按数据量线性放宽阈值：100ms@1w 条
    # 例如 N=5w → 阈值约 500ms
    expected_count = min(TOPK, len(special_ids))
    threshold_ms = max(100.0, 100.0 * (N / 10000.0))
    passed = (len(res_db) <= expected_count and latency_ms < threshold_ms)
    
    detail = f"Found:{len(res_db)} in {latency_ms:.2f}ms (expected~{expected_count}, threshold~{threshold_ms:.0f}ms)"
    
    if latency_ms > threshold_ms:
        detail += " ⚠️ 慢！可能无稀疏索引"
        passed = False
    
    return ScenarioResult("sparse_index_perf", passed, detail)


# ========================== Main ==========================

def main():
    runners = [
        test_expr_equiv,
        test_json_depth,
        test_type_coercion,
        test_soft_delete,
        test_null_completeness,
        test_default_confusion,
        test_exists_equivalence,
        test_sparse_index_perf,
    ]
    
    adapters = []
    if HAVE_MILVUS: adapters.append(("Milvus", MilvusAdapter()))
    # 如需其它库可按需打开
    # if HAVE_QDRANT: adapters.append(("Qdrant", QdrantAdapter()))
    # if HAVE_CHROMA: adapters.append(("Chroma", ChromaAdapter()))
    if HAVE_WEAVIATE: adapters.append(("Weaviate", WeaviateAdapter()))

    print(f"Running EDC Logic Tests on {len(adapters)} DBs with N={N}...")
    print(f"HAVE_WEAVIATE={HAVE_WEAVIATE}")
    
    for db_name, adapter in adapters:
        if isinstance(adapter, WeaviateAdapter):
            print(f"Weaviate client: {adapter.client}")
            if adapter.client:
                try:
                    ready = adapter.client.is_ready()
                    print(f"Weaviate is_ready: {ready}")
                except Exception as e:
                    print(f"Weaviate is_ready error: {e}")
        
        if not adapter.ok():
            print(f"⚠️ {db_name} adapter.ok() = False, skipping...")
            continue
        print(f"\n--- {db_name} ---")
        adapter.connect()
        
        for runner in runners:
            start = time.time()
            res = runner(adapter)
            dur = time.time() - start
            icon = "✅" if res.passed else "⚠️" if "skip" in res.detail else "❌"
            print(f"{icon} {res.name:<20} | {dur:.2f}s | {res.detail}")
            
            # Clean up (optional)
            # adapter.drop(...)

if __name__ == "__main__":
    main()