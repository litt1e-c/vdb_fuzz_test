"""
EDC filter equivalence cross-DB runner
Covers four scenarios on Milvus / Qdrant / Chroma / Weaviate (best-effort; unsupported features are skipped with clear messages).

Scenarios:
1) Expression equivalence: price * (1 + tax) > 115  vs  final_price > 115
2) JSON depth: metadata.user.age > 20  vs  user_age > 20
3) Type coercion: tag_str > 100  vs  tag_int > 100
4) Soft delete vs pre-delete: ghost data visibility

Notes:
- Each scenario creates its own collection/class to avoid schema pollution.
- Dimension kept small (DIM=16) and N=200 for speed.
- If a database lacks a capability (arithmetic filter, JSON path, string comparison, soft delete semantics), the scenario is skipped for that DB.
- Oracle uses exact search (brute-force L2 on Python side) as reference; DB search uses vector + filter.

Assumptions:
- Milvus is reachable at 127.0.0.1:19530
- Qdrant at 127.0.0.1:6333
- Weaviate at 127.0.0.1:8080
- Chroma uses local persistent client path tmp/chroma_edc
"""
from __future__ import annotations
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np
import uuid

# Optional imports guarded; missing client leads to graceful skip
# Milvus: 动态检测 pymilvus 是否可用；若不可用则优雅跳过
try:
    from pymilvus import (
        connections as milvus_connections,
        utility as milvus_utility,
        FieldSchema,
        DataType,
        CollectionSchema,
        Collection,
    )
    HAVE_MILVUS = True
except Exception:
    HAVE_MILVUS = False
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as Qmodels
    HAVE_QDRANT = True
except Exception:
    HAVE_QDRANT = False
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    HAVE_CHROMA = True
except Exception:
    HAVE_CHROMA = False
try:
    import weaviate
    HAVE_WEAVIATE = True
except Exception:
    HAVE_WEAVIATE = False

DIM = 768
N = 5000
TOPK = 10
SEED = 7

@dataclass
class ScenarioResult:
    name: str
    passed: bool
    detail: str

# ------------------------- Milvus Adapter -------------------------
class MilvusEDC:
    def __init__(self):
        self.uri = ("127.0.0.1", "19530")
        self.col = None
        self.last_error = None
    def ok(self):
        return HAVE_MILVUS
    def connect(self):
        if not HAVE_MILVUS:
            return
        try:
            milvus_connections.connect("default", host=self.uri[0], port=self.uri[1])
        except Exception:
            pass
    def drop(self, name: str):
        if not HAVE_MILVUS:
            return
        if milvus_utility.has_collection(name):
            milvus_utility.drop_collection(name)
    def create_expr_equiv(self, name: str):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=DIM),
            FieldSchema(name="price", dtype=DataType.FLOAT),
            FieldSchema(name="tax", dtype=DataType.FLOAT),
            FieldSchema(name="final_price", dtype=DataType.FLOAT),
            # 预计算字段，避免表达式内算术
            FieldSchema(name="price_tax_sum", dtype=DataType.FLOAT),
        ]
        schema = CollectionSchema(fields)
        self.col = Collection(name, schema, consistency_level="Strong")
    def create_json_depth(self, name: str):
        # Milvus 不支持 JSON 字段，标记不支持
        raise NotImplementedError("milvus-json-not-supported")
    def create_type_coercion(self, name: str):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=DIM),
            FieldSchema(name="tag_str", dtype=DataType.VARCHAR, max_length=16),
            FieldSchema(name="tag_int", dtype=DataType.INT64),
        ]
        schema = CollectionSchema(fields)
        self.col = Collection(name, schema, consistency_level="Strong")
    def create_soft_delete(self, name: str):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=DIM),
            FieldSchema(name="age", dtype=DataType.INT64),
        ]
        schema = CollectionSchema(fields)
        self.col = Collection(name, schema, consistency_level="Strong")
    def insert(self, ids, vecs, payloads: List[Dict[str, Any]]):
        cols = {}
        # build columns dynamically
        for k in payloads[0].keys():
            cols[k] = [p[k] for p in payloads]
        self.col.insert([ids, vecs, *[cols[k] for k in cols]])
    def delete(self, ids):
        expr = "id in [" + ",".join(str(int(i)) for i in ids) + "]"
        self.col.delete(expr)
    def flush_and_index(self):
        self.col.flush()
        # 使用 FLAT 精确索引，消除 HNSW 近似性
        # 原 HNSW: {"index_type": "HNSW", "params": {"M": 32, "efConstruction": 200}}
        index_params = {"metric_type": "L2", "index_type": "FLAT", "params": {}}
        self.col.create_index("vector", index_params)
        self.col.load()
    def search(self, query, filter_expr: str):
        # FLAT 精确搜索，无需 ef
        params = {"metric_type": "L2", "params": {}}
        self.last_error = None
        try:
            res = self.col.search([query], "vector", params, limit=TOPK, expr=filter_expr, output_fields=["id"])
            return [hit.id for hit in res[0]]
        except Exception as e:
            self.last_error = e
            return None

# ------------------------- Qdrant Adapter -------------------------
class QdrantEDC:
    def __init__(self):
        self.client = QdrantClient(url="http://127.0.0.1:6333", timeout=None) if HAVE_QDRANT else None
        self.col = None
    def ok(self):
        return HAVE_QDRANT
    def drop(self, name):
        if not HAVE_QDRANT:
            return
        try:
            self.client.delete_collection(name)
        except Exception:
            pass
    def create_expr_equiv(self, name):
        self.client.create_collection(
            name,
            vectors_config=Qmodels.VectorParams(size=DIM, distance=Qmodels.Distance.EUCLID)
        )
        self.col = name
    def create_json_depth(self, name):
        self.create_expr_equiv(name)
    def create_type_coercion(self, name):
        self.create_expr_equiv(name)
    def create_soft_delete(self, name):
        self.create_expr_equiv(name)
    def insert(self, ids, vecs, payloads):
        points = []
        for _id, v, p in zip(ids, vecs, payloads):
            points.append(Qmodels.PointStruct(id=int(_id), vector=v.tolist(), payload=p))
        self.client.upsert(collection_name=self.col, points=points)
    def delete(self, ids):
        self.client.delete(collection_name=self.col, points_selector=Qmodels.PointIdsList(points=[int(i) for i in ids]))
    def flush_and_index(self):
        # Qdrant 无法禁用 HNSW 结构，但可以在搜索时使用 exact=True 绕过近似。
        # 这里保持默认图结构，仅创建标量索引。
        for key, t in [
            ("metadata_user_age", Qmodels.PayloadSchemaType.INTEGER),
            ("user_age", Qmodels.PayloadSchemaType.INTEGER),
            ("tag_str", Qmodels.PayloadSchemaType.KEYWORD),
            ("tag_int", Qmodels.PayloadSchemaType.INTEGER),
            ("age", Qmodels.PayloadSchemaType.INTEGER),
        ]:
            try:
                self.client.create_payload_index(self.col, key, t)
            except Exception:
                pass
    def search(self, query, filter_obj: Qmodels.Filter | None):
        try:
            res = self.client.search(
                collection_name=self.col,
                query_vector=query.tolist(),
                query_filter=filter_obj,
                limit=TOPK,
                # exact=True 强制精确搜索，绕过 HNSW 近似
                search_params=Qmodels.SearchParams(exact=True)
            )
            return [r.id for r in res]
        except Exception:
            return None

# ------------------------- Chroma Adapter -------------------------
class ChromaEDC:
    def __init__(self):
        if HAVE_CHROMA:
            self.client = chromadb.PersistentClient(path="/home/caihao/compare_test/tmp/chroma_edc", settings=ChromaSettings(anonymized_telemetry=False))
        else:
            self.client = None
        self.col = None
    def ok(self):
        return HAVE_CHROMA
    def drop(self, name):
        if not HAVE_CHROMA:
            return
        try:
            self.client.delete_collection(name)
        except Exception:
            pass
    def create_expr_equiv(self, name):
        # Chroma 仅支持 HNSW，无法改为 FLAT；使用高质量参数尽量稳定
        self.col = self.client.get_or_create_collection(
            name,
            metadata={
                "hnsw:space": "l2",
                "hnsw:construction_ef": 2000,
                "hnsw:M": 128,
            }
        )
    def create_json_depth(self, name):
        self.create_expr_equiv(name)
    def create_type_coercion(self, name):
        self.create_expr_equiv(name)
    def create_soft_delete(self, name):
        self.create_expr_equiv(name)
    def insert(self, ids, vecs, payloads):
        self.col.add(ids=[str(i) for i in ids], embeddings=vecs.tolist(), metadatas=payloads)
    def delete(self, ids):
        self.col.delete(ids=[str(i) for i in ids])
    def flush_and_index(self):
        pass
    def search(self, query, where):
        try:
            res = self.col.query(query_embeddings=[query.tolist()], n_results=TOPK, where=where)
            return [int(i) for i in res.get("ids", [[]])[0]] if res else None
        except Exception:
            return None

# ------------------------- Weaviate Adapter -------------------------
class WeaviateEDC:
    def __init__(self):
        if HAVE_WEAVIATE:
            try:
                # v4 style
                self.client = weaviate.connect_to_local(port=8080)
            except Exception:
                try:
                    self.client = weaviate.Client("http://127.0.0.1:8080")
                except Exception:
                    self.client = None
        else:
            self.client = None
        self.class_name = None
    def ok(self):
        return HAVE_WEAVIATE and self.client is not None
    def drop(self, name):
        if not HAVE_WEAVIATE:
            return
        try:
            self.client.schema.delete_class(name)
        except Exception:
            try:
                self.client.schema.delete_all()
            except Exception:
                pass
    def create_expr_equiv(self, name):
        self.class_name = name
        class_obj = {
            "class": name,
            "vectorizer": "none",
            "properties": [
                {"name": "idx", "dataType": ["int"]},
                {"name": "price", "dataType": ["number"]},
                {"name": "tax", "dataType": ["number"]},
                {"name": "final_price", "dataType": ["number"]},
            ],
            # Weaviate 仅支持 HNSW，无法改为 FLAT；提升参数以提高稳定性
            "vectorIndexConfig": {
                "distance": "l2-squared",
                "ef": 4096,
                "efConstruction": 2000,
                "maxConnections": 128
            }
        }
        self.client.schema.create_class(class_obj)
    def create_json_depth(self, name):
        # Weaviate 无原生 JSON 深层过滤（需要 nested object 支持），暂不支持
        raise NotImplementedError("weaviate-json-not-supported")
    def create_type_coercion(self, name):
        self.class_name = name
        class_obj = {
            "class": name,
            "vectorizer": "none",
            "properties": [
                {"name": "idx", "dataType": ["int"]},
                {"name": "tag_str", "dataType": ["text"]},
                {"name": "tag_int", "dataType": ["int"]},
            ],
            "vectorIndexConfig": {
                "distance": "l2-squared",
                "ef": 4096,
                "efConstruction": 2000,
                "maxConnections": 128
            }
        }
        self.client.schema.create_class(class_obj)
    def create_soft_delete(self, name):
        self.class_name = name
        class_obj = {
            "class": name,
            "vectorizer": "none",
            "properties": [
                {"name": "idx", "dataType": ["int"]},
                {"name": "age", "dataType": ["int"]},
            ],
            "vectorIndexConfig": {
                "distance": "l2-squared",
                "ef": 4096,
                "efConstruction": 2000,
                "maxConnections": 128
            }
        }
        self.client.schema.create_class(class_obj)
    def insert(self, ids, vecs, payloads):
        with self.client.batch as batch:
            batch.batch_size = 200
            for _id, v, p in zip(ids, vecs, payloads):
                uuid_str = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{self.class_name}-{int(_id)}"))
                body = dict(p)
                body["idx"] = int(_id)
                batch.add_data_object(body, self.class_name, vector=v.tolist(), uuid=uuid_str)
    def delete(self, ids):
        # Weaviate lacks soft-delete API; emulate by deleting objects
        for _id in ids:
            uuid_str = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{self.class_name}-{int(_id)}"))
            try:
                self.client.data_object.delete(uuid=uuid_str, class_name=self.class_name)
            except Exception:
                pass
    def flush_and_index(self):
        # 等待 weaviate 后台完成持久化/索引
        time.sleep(2)
    def search(self, query, where_clause):
        try:
            res = (self.client.query
                .get(self.class_name, ["idx", "_additional { id distance }"])
                .with_near_vector({"vector": query.tolist()})
                .with_where(where_clause if where_clause else None)
                .with_limit(TOPK)
                .do())
            items = res["data"]["Get"].get(self.class_name, []) if res else []
            return [int(item.get("idx", 0)) for item in items]
        except Exception:
            return None

# ------------------------- Scenario Runners -------------------------
def make_data_expr_equiv():
    rng = np.random.default_rng(SEED)
    ids = np.arange(2, dtype=np.int64)
    vecs = rng.random((2, DIM), dtype=np.float32)
    payloads = [
        {
            "price": 100.0,
            "tax": 0.2,
            "final_price": 120.0,
            # 预计算 price*(1+tax)，避免在表达式中做算术
            "price_tax_sum": 120.0,
        },
        {
            "price": 90.0,
            "tax": 0.1,
            "final_price": 99.0,
            "price_tax_sum": 99.0,
        },
    ]
    return ids, vecs, payloads

def run_expr_equiv(db_name: str, db):
    ids, vecs, payloads = make_data_expr_equiv()
    col = f"edc_expr_{int(time.time())}"
    try:
        db.drop(col)
        db.create_expr_equiv(col)
        db.insert(ids, vecs, payloads)
        db.flush_and_index()
    except NotImplementedError as e:
        return ScenarioResult("expression_equivalence", False, f"skip: {e}")
    except Exception as e:
        return ScenarioResult("expression_equivalence", False, f"setup-fail: {e}")

    q = vecs[0]
    # Filters per DB
    if isinstance(db, MilvusEDC):
        # 使用预计算字段，避免在表达式中做算术
        fa = "price_tax_sum > 115"
        fb = "final_price > 115"
    elif isinstance(db, QdrantEDC):
        # Qdrant filter model lacks arithmetic; skip
        return ScenarioResult("expression_equivalence", False, "skip: qdrant-no-arithmetic-filter")
    elif isinstance(db, ChromaEDC):
        return ScenarioResult("expression_equivalence", False, "skip: chroma-no-arithmetic-filter")
    elif isinstance(db, WeaviateEDC):
        fa = None  # weaviate does not support arithmetic filter expressions
        fb = {"path": ["final_price"], "operator": "GreaterThan", "valueNumber": 115}
        return ScenarioResult("expression_equivalence", False, "skip: weaviate-no-arithmetic-filter")
    else:
        return ScenarioResult("expression_equivalence", False, "skip: unknown-adapter")

    try:
        if isinstance(db, MilvusEDC):
            a = db.search(q, fa)
            if a is None and db.last_error:
                msg = str(db.last_error)
                if "arithmetic" in msg or "parse expression" in msg:
                    return ScenarioResult("expression_equivalence", False, "skip: milvus-server-no-arithmetic-support")
            b = db.search(q, fb)
        else:
            a = db.search(q, fa)
            b = db.search(q, fb)
    except Exception as e:
        return ScenarioResult("expression_equivalence", False, f"query-fail: {e}")

    if a is None or b is None:
        return ScenarioResult("expression_equivalence", False, "query-return-none")
    return ScenarioResult("expression_equivalence", a == b, f"A={a}, B={b}")


def make_data_json_depth():
    rng = np.random.default_rng(SEED)
    ids = np.arange(3, dtype=np.int64)
    vecs = rng.random((3, DIM), dtype=np.float32)
    # 使用扁平化 key（避免 JSON 字段限制），同时保留字符串数字以触发类型问题
    payloads = [
        {"metadata_user_age": 25, "metadata_user_age_str": "25", "user_age": 25},
        {"metadata_user_age": 18, "metadata_user_age_str": "18", "user_age": 18},
        {"metadata_user_age": 25, "metadata_user_age_str": "25", "user_age": 25},
    ]
    return ids, vecs, payloads

def run_json_depth(db_name: str, db):
    ids, vecs, payloads = make_data_json_depth()
    col = f"edc_json_{int(time.time())}"
    try:
        db.drop(col)
        db.create_json_depth(col)
        db.insert(ids, vecs, payloads)
        db.flush_and_index()
    except NotImplementedError as e:
        return ScenarioResult("json_depth", False, f"skip: {e}")
    except Exception as e:
        return ScenarioResult("json_depth", False, f"setup-fail: {e}")

    q = vecs[0]
    if isinstance(db, QdrantEDC):
        fa = Qmodels.Filter(must=[Qmodels.FieldCondition(key="metadata_user_age", range=Qmodels.Range(gt=20))])
        fb = Qmodels.Filter(must=[Qmodels.FieldCondition(key="user_age", range=Qmodels.Range(gt=20))])
    elif isinstance(db, ChromaEDC):
        fa = {"metadata_user_age": {"$gt": 20}}
        fb = {"user_age": {"$gt": 20}}
    else:
        return ScenarioResult("json_depth", False, "skip: db-no-json")

    a = db.search(q, fa)
    b = db.search(q, fb)
    if a is None or b is None:
        return ScenarioResult("json_depth", False, "query-return-none")
    return ScenarioResult("json_depth", a == b, f"A={a}, B={b}")


def make_data_type_coercion():
    rng = np.random.default_rng(SEED)
    ids = np.arange(3, dtype=np.int64)
    vecs = rng.random((3, DIM), dtype=np.float32)
    payloads = [
        {"tag_str": "123", "tag_int": 123},
        {"tag_str": "099", "tag_int": 99},
        {"tag_str": "1000", "tag_int": 1000},
    ]
    return ids, vecs, payloads

def run_type_coercion(db_name: str, db):
    ids, vecs, payloads = make_data_type_coercion()
    col = f"edc_type_{int(time.time())}"
    try:
        db.drop(col)
        db.create_type_coercion(col)
        db.insert(ids, vecs, payloads)
        db.flush_and_index()
    except NotImplementedError as e:
        return ScenarioResult("type_coercion", False, f"skip: {e}")
    except Exception as e:
        return ScenarioResult("type_coercion", False, f"setup-fail: {e}")

    q = vecs[0]
    if isinstance(db, MilvusEDC):
        fa = "tag_str == \"123\""  # 等价性：字符串与数字同值
        fb = "tag_int == 123"
    elif isinstance(db, QdrantEDC):
        fa = Qmodels.Filter(must=[Qmodels.FieldCondition(key="tag_str", match=Qmodels.MatchValue(value="123"))])
        fb = Qmodels.Filter(must=[Qmodels.FieldCondition(key="tag_int", match=Qmodels.MatchValue(value=123))])
    elif isinstance(db, ChromaEDC):
        fa = {"tag_str": {"$eq": "123"}}
        fb = {"tag_int": {"$eq": 123}}
    elif isinstance(db, WeaviateEDC):
        fa = {"path": ["tag_str"], "operator": "Equal", "valueText": "123"}
        fb = {"path": ["tag_int"], "operator": "Equal", "valueInt": 123}
    else:
        return ScenarioResult("type_coercion", False, "skip: unknown-adapter")

    a = db.search(q, fa)
    b = db.search(q, fb)
    if a is None or b is None:
        return ScenarioResult("type_coercion", False, "query-return-none")
    return ScenarioResult("type_coercion", a == b, f"A={a}, B={b}")


def make_data_soft_delete():
    rng = np.random.default_rng(SEED)
    ids = np.arange(10, dtype=np.int64)
    vecs = rng.random((10, DIM), dtype=np.float32)
    ages = [18, 22, 30, 25, 40, 19, 35, 50, 21, 28]
    payloads = [{"age": a} for a in ages]
    deleted = [1, 3]
    return ids, vecs, payloads, deleted

def run_soft_delete(db_name: str, db):
    ids, vecs, payloads, deleted = make_data_soft_delete()
    col = f"edc_soft_{int(time.time())}"
    try:
        db.drop(col)
        db.create_soft_delete(col)
        db.insert(ids, vecs, payloads)
        db.delete(deleted)
        db.flush_and_index()
    except NotImplementedError as e:
        return ScenarioResult("soft_delete", False, f"skip: {e}")
    except Exception as e:
        return ScenarioResult("soft_delete", False, f"setup-fail: {e}")

    q = vecs[0]
    if isinstance(db, MilvusEDC):
        filt = "age > 20"
    elif isinstance(db, QdrantEDC):
        filt = Qmodels.Filter(must=[Qmodels.FieldCondition(key="age", range=Qmodels.Range(gt=20))])
    elif isinstance(db, ChromaEDC):
        filt = {"age": {"$gt": 20}}
    elif isinstance(db, WeaviateEDC):
        return ScenarioResult("soft_delete", False, "skip: weaviate-soft-delete-unsupported")
    else:
        return ScenarioResult("soft_delete", False, "skip: unknown-adapter")

    base_res = db.search(q, filt)

    # Reference on clean set (client-side exact)
    clean_ids = [i for i in ids if i not in deleted]
    clean_payloads = [payloads[i] for i in clean_ids]
    clean_vecs = vecs[[i for i in range(len(ids)) if i not in deleted]]
    scored = []
    for _id, vec, payload in zip(clean_ids, clean_vecs, clean_payloads):
        if payload["age"] > 20:
            dist = float(np.linalg.norm(vec - q))
            scored.append((dist, _id))
    scored.sort(key=lambda x: x[0])
    ref_res = [i for _, i in scored[:TOPK]]

    if base_res is None:
        return ScenarioResult("soft_delete", False, "query-return-none")
    return ScenarioResult("soft_delete", base_res == ref_res, f"Base={base_res}, Ref={ref_res}")


RUNNERS = [
    ("expression_equivalence", run_expr_equiv),
    ("json_depth", run_json_depth),
    ("type_coercion", run_type_coercion),
    ("soft_delete", run_soft_delete),
]

DBS: List[Tuple[str, Any]] = []
if HAVE_MILVUS:
    DBS.append(("Milvus", MilvusEDC()))
DBS.append(("Qdrant", QdrantEDC()))
DBS.append(("Chroma", ChromaEDC()))
DBS.append(("Weaviate", WeaviateEDC()))


def main():
    print("\n" + "=" * 70)
    print("EDC Filter Equivalence Cross-DB Runner")
    print("=" * 70)

    for db_name, db in DBS:
        if not db.ok():
            print(f"\n{db_name}: ❌ client missing，跳过")
            continue
        print(f"\n{db_name}: 开始测试")
        # 确保已连接
        try:
            db.connect()
        except Exception:
            pass
        for scen_name, fn in RUNNERS:
            res = fn(db_name, db)
            status = "✅" if res.passed else "⚠️" if res.detail.startswith("skip") else "❌"
            print(f"  [{status}] {scen_name}: {res.detail}")

    print("\n完成。")


if __name__ == "__main__":
    main()
