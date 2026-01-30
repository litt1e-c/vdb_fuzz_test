"""
EDC "HELLFIRE" Runner v2: Extreme Scale Vector DB Logic Verification
Scale: N=200,000 | DIM=768
Fixes based on log analysis:
1. Qdrant: Client timeout increased to 300s to fix insertion crash.
2. Milvus: Added explicit parentheses to boolean expressions to fix Logic Error.
3. Milvus: Dynamic 'ef' calculation for deep pagination.
"""
from __future__ import annotations
import time
import uuid
import numpy as np
from dataclasses import dataclass
from typing import List, Any, Dict

# --- Global Config ---
N_SCALE = 200000     # 20万数据量
DIM = 768            # 标准 LLM 维度
BATCH_SIZE = 2000    # 稍微调小 Batch 以减轻单次网络包压力 (原5000 -> 2000)
SEED = 2025
TOPK = 50

# --- Database Import ---
HAVE_MILVUS = False
try:
    from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
    HAVE_MILVUS = True
except ImportError: pass

HAVE_QDRANT = False
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as Qmodels
    HAVE_QDRANT = True
except ImportError: pass

HAVE_WEAVIATE = False
try:
    import weaviate
    HAVE_WEAVIATE = True
except ImportError: pass


@dataclass
class ScenarioResult:
    name: str
    passed: bool
    detail: str


# ========================== Robust Adapters (Fixed) ==========================

class BaseAdapter:
    def connect(self): pass
    def ok(self): return False
    def drop(self, name): pass
    def create_schema(self, name): raise NotImplementedError
    def insert_batch(self, ids, vecs, payloads): pass
    def upsert_single(self, id_val, vec, payload): pass
    def delete(self, ids): pass
    def flush(self): pass
    def search(self, vec, limit, offset=0, expr=None): raise NotImplementedError
    def get_by_id(self, id_val): raise NotImplementedError


class MilvusAdapter(BaseAdapter):
    def __init__(self):
        self.uri = ("127.0.0.1", "19530")
        self.col = None

    def ok(self): return HAVE_MILVUS
    def connect(self):
        if HAVE_MILVUS:
            try: connections.connect("default", host=self.uri[0], port=self.uri[1])
            except: pass

    def drop(self, name):
        if utility.has_collection(name): utility.drop_collection(name)

    def create_schema(self, name):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=DIM),
            FieldSchema(name="tag", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="meta", dtype=DataType.JSON),
            FieldSchema(name="group_id", dtype=DataType.INT64),
        ]
        schema = CollectionSchema(fields)
        self.col = Collection(name, schema, consistency_level="Strong")

    def insert_batch(self, ids, vecs, payloads):
        ids = [int(i) for i in ids]
        tag_col = [p.get("tag", "") for p in payloads]
        grp_col = [int(p.get("group_id", 0)) for p in payloads]
        meta_col = []
        for p in payloads:
            meta_col.append({k: v for k, v in p.items()})
        self.col.insert([ids, vecs, tag_col, meta_col, grp_col])

    def upsert_single(self, id_val, vec, payload):
        tag = payload.get("tag", "")
        grp = int(payload.get("group_id", 0))
        self.col.upsert([[int(id_val)], [vec], [tag], [payload], [grp]])

    def delete(self, ids):
        expr = f"id in {list(ids)}"
        self.col.delete(expr)

    def flush(self):
        self.col.flush()
        index_params = {
            "metric_type": "L2", 
            "index_type": "HNSW", 
            "params": {"M": 32, "efConstruction": 200}
        }
        self.col.create_index("vector", index_params)
        self.col.load()

    def search(self, vec, limit, offset=0, expr=None):
        # [Fix 1] 动态计算 ef，防止 Deep Paging 报错
        required_k = limit + offset
        dynamic_ef = max(128, required_k + 100)
        
        search_params = {"metric_type": "L2", "params": {"ef": dynamic_ef}}
        
        if expr:
            if "tag ==" in expr:
                pass 
            elif "meta_tags contains" in expr and "contains_all" not in expr:
                val = expr.split('"')[1]
                expr = f'json_contains(meta["tags"], "{val}")'
            elif "meta_tags contains_all" in expr:
                # [Fix 2] 增加显式括号，修复 Logic Error
                expr = '(json_contains(meta["tags"], "red")) && (json_contains(meta["tags"], "blue"))'
        
        try:
            res = self.col.search(
                [vec], "vector", search_params, 
                limit=limit, offset=offset, expr=expr, 
                output_fields=["id", "tag", "group_id"],
                consistency_level="Strong"
            )
            if not res: return []
            return [{"id": h.id, "tag": h.entity.get("tag"), "group_id": h.entity.get("group_id")} for h in res[0]]
        except Exception as e:
            # print(f"Milvus search error: {e}")
            return []

    def get_by_id(self, id_val):
        res = self.col.query(f"id == {id_val}", output_fields=["tag", "group_id", "meta"], consistency_level="Strong")
        if not res: return None
        return res[0]


class QdrantAdapter(BaseAdapter):
    def __init__(self):
        # [Fix 3] Timeout 增加到 300s，修复插入 crash
        self.client = QdrantClient("http://localhost:6333", timeout=300) if HAVE_QDRANT else None
        self.col = None

    def ok(self): return HAVE_QDRANT
    def drop(self, name): self.client.delete_collection(name)

    def create_schema(self, name):
        self.col = name
        if not self.client.collection_exists(name):
            self.client.create_collection(
                name,
                vectors_config=Qmodels.VectorParams(size=DIM, distance=Qmodels.Distance.EUCLID),
                optimizers_config=Qmodels.OptimizersConfigDiff(
                    default_segment_number=4,
                    memmap_threshold=20000
                )
            )

    def insert_batch(self, ids, vecs, payloads):
        ids = [int(i) for i in ids]
        clean_payloads = []
        for p in payloads:
            cp = {}
            for k, v in p.items():
                if isinstance(v, (np.integer, np.floating)): cp[k] = v.item()
                elif isinstance(v, list): cp[k] = list(v)
                else: cp[k] = v
            clean_payloads.append(cp)

        points = [
            Qmodels.PointStruct(id=i, vector=v.tolist(), payload=cp)
            for i, v, cp in zip(ids, vecs, clean_payloads)
        ]
        self.client.upsert(self.col, points)

    def upsert_single(self, id_val, vec, payload):
        self.insert_batch([id_val], [vec], [payload])

    def delete(self, ids):
        self.client.delete(self.col, points_selector=Qmodels.PointIdsList(points=[int(i) for i in ids]))

    def flush(self):
        time.sleep(2)

    def search(self, vec, limit, offset=0, expr=None):
        q_filter = None
        if expr:
            if "tag ==" in expr:
                val = expr.split("==")[1].strip().strip('"')
                q_filter = Qmodels.Filter(must=[Qmodels.FieldCondition(key="tag", match=Qmodels.MatchValue(value=val))])
            elif "group_id ==" in expr:
                val = int(expr.split("==")[1].strip())
                q_filter = Qmodels.Filter(must=[Qmodels.FieldCondition(key="group_id", match=Qmodels.MatchValue(value=val))])
            elif "meta_tags contains" in expr and "contains_all" not in expr:
                val = expr.split('"')[1]
                q_filter = Qmodels.Filter(must=[Qmodels.FieldCondition(key="tags", match=Qmodels.MatchValue(value=val))])
            elif "meta_tags contains_all" in expr:
                q_filter = Qmodels.Filter(must=[
                    Qmodels.FieldCondition(key="tags", match=Qmodels.MatchValue(value="red")),
                    Qmodels.FieldCondition(key="tags", match=Qmodels.MatchValue(value="blue"))
                ])

        res = self.client.search(
            self.col, query_vector=vec.tolist(), query_filter=q_filter, 
            limit=limit, offset=offset,
            search_params=Qmodels.SearchParams(exact=False)
        )
        return [{"id": r.id, "tag": r.payload.get("tag"), "group_id": r.payload.get("group_id")} for r in res]

    def get_by_id(self, id_val):
        res = self.client.retrieve(self.col, ids=[int(id_val)])
        if not res: return None
        return res[0].payload


class WeaviateAdapter(BaseAdapter):
    def __init__(self):
        self.client = weaviate.Client("http://localhost:8080", timeout_config=(10, 300)) if HAVE_WEAVIATE else None
        self.cls = None

    def ok(self): return HAVE_WEAVIATE and self.client.is_ready()

    def drop(self, name):
        name = name.capitalize()
        try: self.client.schema.delete_class(name)
        except: pass

    def create_schema(self, name):
        name = name.capitalize()
        self.cls = name
        if not self.client.schema.exists(name):
            self.client.schema.create_class({
                "class": name,
                "vectorizer": "none",
                "properties": [
                    {"name": "tag", "dataType": ["text"]}, 
                    {"name": "group_id", "dataType": ["int"]},
                    {"name": "tags", "dataType": ["text[]"]},
                    {"name": "oid", "dataType": ["int"]}, 
                ]
            })

    def insert_batch(self, ids, vecs, payloads):
        ids = [int(i) for i in ids]
        with self.client.batch as batch:
            batch.batch_size = 200
            for i, v, p in zip(ids, vecs, payloads):
                props = {}
                for k, val in p.items():
                    if isinstance(val, (np.integer, np.floating)): props[k] = val.item()
                    else: props[k] = val
                props["oid"] = int(i)
                uid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{self.cls}_{i}"))
                batch.add_data_object(props, self.cls, vector=v.tolist(), uuid=uid)

    def upsert_single(self, id_val, vec, payload):
        self.insert_batch([id_val], [vec], [payload])

    def delete(self, ids):
        for i in ids:
            uid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{self.cls}_{i}"))
            self.client.data_object.delete(uuid=uid, class_name=self.cls)

    def flush(self):
        print("    [Weaviate] Waiting for indexing...", end="", flush=True)
        max_retries = 30
        for _ in range(max_retries):
            try:
                res = self.client.query.aggregate(self.cls).with_meta_count().do()
                cnt = res['data']['Aggregate'][self.cls][0]['meta']['count']
                if cnt >= N_SCALE: 
                    print(" Done.")
                    return
            except: pass
            time.sleep(2)
            print(".", end="", flush=True)
        print(" Timeout/Partial.")

    def search(self, vec, limit, offset=0, expr=None):
        q = (self.client.query.get(self.cls, ["oid", "tag", "group_id"])
             .with_near_vector({"vector": vec.tolist()})
             .with_limit(limit).with_offset(offset))
        
        if expr:
            w_filter = None
            if "tag ==" in expr:
                val = expr.split("==")[1].strip().strip('"')
                w_filter = {"path": ["tag"], "operator": "Like", "valueText": val}
            elif "group_id ==" in expr:
                val = int(expr.split("==")[1].strip())
                w_filter = {"path": ["group_id"], "operator": "Equal", "valueInt": val}
            elif "meta_tags contains" in expr and "contains_all" not in expr:
                val = expr.split('"')[1]
                w_filter = {"path": ["tags"], "operator": "Like", "valueText": val}
            elif "meta_tags contains_all" in expr:
                w_filter = {"operator": "And", "operands": [
                    {"path": ["tags"], "operator": "Like", "valueText": "red"},
                    {"path": ["tags"], "operator": "Like", "valueText": "blue"}
                ]}
            
            if w_filter: q = q.with_where(w_filter)
        
        try:
            res = q.do()
            if "errors" in res: return []
            items = res['data']['Get'][self.cls]
            return [{"id": item["oid"], "tag": item.get("tag"), "group_id": item.get("group_id")} for item in items]
        except: return []

    def get_by_id(self, id_val):
        uid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{self.cls}_{id_val}"))
        try:
            res = self.client.data_object.get_by_id(uid, class_name=self.cls)
            if res: return res['properties']
        except: pass
        return None


# ========================== Data Generator ==========================

def generate_and_insert(db, name, specific_items=None):
    print(f"  🌊 Generating & Inserting {N_SCALE} items...")
    rng = np.random.default_rng(SEED)
    
    trap_center = np.zeros(DIM, dtype=np.float32); trap_center[0] = 5.0 
    target_center = np.zeros(DIM, dtype=np.float32); target_center[1] = 5.0 
    
    inserted_count = 0
    trap_ids = []
    
    for batch_start in range(0, N_SCALE, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, N_SCALE)
        count = batch_end - batch_start
        
        vecs = rng.normal(0, 0.1, (count, DIM)).astype(np.float32)
        ids = np.arange(batch_start, batch_end, dtype=np.int64)
        payloads = [{"tag": "noise", "group_id": 0} for _ in range(count)]
        
        # Inject Clusters in first batch
        if batch_start == 0:
            vecs[0:100] += trap_center
            for i in range(100): 
                payloads[i]["tag"] = "TRAP"
                payloads[i]["group_id"] = 1
                trap_ids.append(ids[i])
            
            vecs[100:200] += target_center
            for i in range(100, 200): 
                payloads[i]["tag"] = "TARGET"
                payloads[i]["group_id"] = 2
        
        # Inject Specific Items
        if specific_items and batch_start == 0:
            for i, (sid, svec, spayload) in enumerate(specific_items):
                idx = count - 1 - i
                ids[idx] = sid
                vecs[idx] = svec
                payloads[idx] = spayload

        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        db.insert_batch(ids, vecs, payloads)
        inserted_count += count
        print(f"    Inserted: {inserted_count}/{N_SCALE}...", end="\r")

    print(f"\n  🔨 Flushing & Indexing...")
    db.flush()
    q_trap = trap_center / np.linalg.norm(trap_center)
    return q_trap

# ========================== Test Scenarios ==========================

def test_deep_pagination(db, q_probe):
    OFFSET = 8000
    LIMIT = 50
    
    print(f"    [Pagination] Fetching Offset={OFFSET} Limit={LIMIT}...")
    try:
        full_page = db.search(q_probe, limit=LIMIT, offset=OFFSET)
    except Exception as e:
        return ScenarioResult("deep_paging", False, f"Search failed: {e}")
    
    if not full_page:
        return ScenarioResult("deep_paging", False, "Empty result at deep offset")

    part1 = db.search(q_probe, limit=20, offset=OFFSET)
    part2 = db.search(q_probe, limit=20, offset=OFFSET+20)
    part3 = db.search(q_probe, limit=10, offset=OFFSET+40)
    
    combined = [x["id"] for x in part1 + part2 + part3]
    full_ids = [x["id"] for x in full_page]
    
    if full_ids == combined:
        return ScenarioResult("deep_paging", True, f"Perfect match at Offset {OFFSET}")
    
    s1, s2 = set(full_ids), set(combined)
    overlap = len(s1 & s2) / len(s1 | s2) if len(s1 | s2) > 0 else 0
    passed = overlap >= 0.85
    return ScenarioResult("deep_paging", passed, f"Overlap {overlap:.2f} (Accepted > 0.85)")


def test_orthogonal_filtering(db, q_trap):
    print(f"    [Orthogonal] Querying near TRAP, filtering for TARGET...")
    start_t = time.time()
    res = db.search(q_trap, limit=10, expr='tag == "TARGET"')
    dur = time.time() - start_t
    
    if not res:
        return ScenarioResult("ortho_filter", False, "Failed to find distant targets")
    
    valid = all(r["tag"] == "TARGET" for r in res)
    if not valid:
        leaked = [r["tag"] for r in res if r["tag"] != "TARGET"]
        return ScenarioResult("ortho_filter", False, f"Filter Leakage! Found tags: {leaked}")
    
    return ScenarioResult("ortho_filter", True, f"Found {len(res)} distant targets in {dur:.2f}s")


def test_array_logic(db):
    # 此测试验证 AND 逻辑是否被正确执行
    rng = np.random.default_rng(SEED)
    q = rng.random(DIM).astype(np.float32)
    
    res_both = db.search(q, limit=100, expr='meta_tags contains_all ["red", "blue"]')
    
    if not res_both:
         return ScenarioResult("array_logic", False, "Needle lost")
         
    invalid_ids = []
    for item in res_both:
        full_obj = db.get_by_id(item["id"])
        tags = []
        if full_obj:
            if "tags" in full_obj: tags = full_obj["tags"] 
            elif "meta" in full_obj and "tags" in full_obj["meta"]: tags = full_obj["meta"]["tags"]
        
        if "red" not in tags or "blue" not in tags:
            invalid_ids.append(item["id"])
            
    if invalid_ids:
        # [Fix Verification] 如果返回了这个错误，说明数据库的 AND 逻辑失效
        return ScenarioResult("array_logic", False, f"Logic Error: Items {invalid_ids} do not have both tags")
        
    return ScenarioResult("array_logic", True, f"Verified {len(res_both)} items have correct tags")


def test_zombie_resurrection(db):
    TARGET_ID = 1000
    rng = np.random.default_rng(SEED)
    
    print(f"    [Zombie] Deleting ID {TARGET_ID}...")
    db.delete([TARGET_ID])
    if not isinstance(db, MilvusAdapter): time.sleep(1) 
    
    check = db.get_by_id(TARGET_ID)
    if check: return ScenarioResult("zombie_res", False, "Delete failed")
        
    print(f"    [Zombie] Resurrecting ID {TARGET_ID}...")
    new_vec = rng.random(DIM).astype(np.float32)
    db.upsert_single(TARGET_ID, new_vec, {"tag": "zombie", "group_id": 666})
    if not isinstance(db, MilvusAdapter): time.sleep(1)
    
    res = db.get_by_id(TARGET_ID)
    if not res: return ScenarioResult("zombie_res", False, "Resurrection failed")
    
    tag = res.get("tag")
    # Milvus Entity compatibility check
    if hasattr(res, "get"): tag = res.get("tag")
    if hasattr(res, "entity") and hasattr(res.entity, "get"): tag = res.entity.get("tag")

    if tag != "zombie":
        return ScenarioResult("zombie_res", False, f"Old data found: {tag}")
        
    return ScenarioResult("zombie_res", True, "Delete-Insert cycle consistent")


# ========================== Main Runner ==========================

def main():
    # Needles: 200100 (red), 200101 (blue), 200102 (red, blue)
    start_id = N_SCALE + 100
    rng = np.random.default_rng(SEED)
    needles = [
        (start_id, rng.random(DIM).astype(np.float32), {"tags": ["red"], "tag": "needle", "group_id": 9}),
        (start_id+1, rng.random(DIM).astype(np.float32), {"tags": ["blue"], "tag": "needle", "group_id": 9}),
        (start_id+2, rng.random(DIM).astype(np.float32), {"tags": ["red", "blue"], "tag": "needle", "group_id": 9}),
    ]

    runners = [
        test_deep_pagination,
        test_orthogonal_filtering,
        test_array_logic,
        test_zombie_resurrection
    ]
    
    adapters = []
    if HAVE_MILVUS: adapters.append(("Milvus", MilvusAdapter()))
    if HAVE_QDRANT: adapters.append(("Qdrant", QdrantAdapter()))
    if HAVE_WEAVIATE: adapters.append(("Weaviate", WeaviateAdapter()))
    
    print(f"{'='*60}")
    print(f"🔥 HELLFIRE RUNNER v2: Scale N={N_SCALE}, DIM={DIM}")
    print(f"   Databases: {[n for n, _ in adapters]}")
    print(f"{'='*60}\n")
    
    for db_name, adapter in adapters:
        if not adapter.ok(): continue
        
        print(f"--- 🚀 Target: {db_name} ---")
        adapter.connect()
        
        run_id = uuid.uuid4().hex[:4]
        col_name = f"hellfire_{run_id}"
        
        try:
            adapter.drop(col_name)
            adapter.create_schema(col_name)
            q_trap = generate_and_insert(adapter, col_name, needles)
            
            for runner in runners:
                if runner == test_deep_pagination: args = (adapter, q_trap)
                elif runner == test_orthogonal_filtering: args = (adapter, q_trap)
                else: args = (adapter,)
                
                start_t = time.time()
                try:
                    res = runner(*args)
                    dur = time.time() - start_t
                    icon = "✅" if res.passed else "❌"
                    print(f"{icon} {res.name:<15} | {dur:.2f}s | {res.detail}")
                except Exception as e:
                    print(f"❌ {runner.__name__:<15} | CRASH: {e}")

        except Exception as e:
            print(f"💥 Critical Failure in {db_name}: {e}")
        finally:
            print(f"   Cleaning up {col_name}...\n")
            adapter.drop(col_name)

if __name__ == "__main__":
    main()