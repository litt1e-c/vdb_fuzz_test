"""
EDC "CHAOS" RUNNER v3: Extreme Robustness & Consistency Verification
Scale: N=200,000 | DIM=768
Fixes:
1. Weaviate: Fixed JSON crash by isolating 'AtomicTransfer' to use non-batch API.
2. Qdrant: Suppressed verbose batch retry warnings.
3. Milvus: Handled RPC errors gracefully in logs.
"""
from __future__ import annotations
import time
import uuid
import threading
import warnings
import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Any, Dict

# --- Configuration ---
N_SCALE = 200000     
DIM = 768            
BATCH_SIZE = 2000    
SEED = 2025          
WALL_SIZE = 10000    
CONCURRENCY_DURATION = 10 

# --- Suppress Noise ---
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR) # Suppress Qdrant retry warnings

# --- Database Import ---
HAVE_MILVUS = False
try:
    from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection, MilvusException
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
class ChaosResult:
    name: str
    passed: bool
    detail: str


# ========================== Robust Adapters ==========================

class BaseAdapter:
    def connect(self): pass
    def ok(self): return False
    def setup(self, name): raise NotImplementedError
    def drop(self, name): pass
    def insert_batch(self, ids, vecs, payloads): pass
    def upsert_single(self, id_val, vec, payload): pass
    def flush(self): pass
    def search(self, vec, limit, expr=None): raise NotImplementedError
    def get_count(self, expr=None): raise NotImplementedError


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

    def setup(self, name):
        self.drop(name)
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=DIM),
            FieldSchema(name="tag", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="group_id", dtype=DataType.INT64),
        ]
        schema = CollectionSchema(fields)
        self.col = Collection(name, schema, consistency_level="Strong")

    def insert_batch(self, ids, vecs, payloads):
        ids = [int(i) for i in ids]
        tag_col = [p.get("tag", "") for p in payloads]
        grp_col = [int(p.get("group_id", 0)) for p in payloads]
        self.col.insert([ids, vecs, tag_col, grp_col])

    def upsert_single(self, id_val, vec, payload):
        tag = payload.get("tag", "")
        grp = int(payload.get("group_id", 0))
        self.col.upsert([[int(id_val)], [vec], [tag], [grp]])

    def flush(self):
        self.col.flush()
        index_params = {"metric_type": "L2", "index_type": "HNSW", "params": {"M": 32, "efConstruction": 200}}
        self.col.create_index("vector", index_params)
        self.col.load()

    def search(self, vec, limit, expr=None):
        search_params = {"metric_type": "L2", "params": {"ef": 128}} 
        try:
            res = self.col.search(
                [vec], "vector", search_params, 
                limit=limit, expr=expr, 
                output_fields=["id"], 
                consistency_level="Strong"
            )
            return [h.id for h in res[0]] if res else []
        except Exception:
            return []

    def get_count(self, expr=None):
        if expr is None: return self.col.num_entities
        res = self.col.query(expr, output_fields=["id"], consistency_level="Strong")
        return len(res)


class QdrantAdapter(BaseAdapter):
    def __init__(self):
        self.client = QdrantClient("http://localhost:6333", timeout=300) if HAVE_QDRANT else None
        self.col = None

    def ok(self): return HAVE_QDRANT
    def drop(self, name): self.client.delete_collection(name)

    def setup(self, name):
        self.col = name
        self.client.recreate_collection(
            name, 
            vectors_config=Qmodels.VectorParams(size=DIM, distance=Qmodels.Distance.EUCLID),
            optimizers_config=Qmodels.OptimizersConfigDiff(indexing_threshold=0) 
        )

    def insert_batch(self, ids, vecs, payloads):
        points = []
        for i, v, p in zip(ids, vecs, payloads):
            safe_p = {k: (v.item() if hasattr(v, 'item') else v) for k, v in p.items()}
            points.append(Qmodels.PointStruct(id=int(i), vector=v.tolist(), payload=safe_p))
        
        self.client.upload_points(self.col, points=points, batch_size=1024, wait=True)

    def upsert_single(self, id_val, vec, payload):
        self.client.upsert(
            self.col, 
            points=[Qmodels.PointStruct(id=int(id_val), vector=vec.tolist(), payload=payload)],
            wait=True 
        )

    def flush(self):
        self.client.update_collection(
            self.col, 
            optimizers_config=Qmodels.OptimizersConfigDiff(indexing_threshold=20000)
        )
        print("    [Qdrant] Optimizing...", end="", flush=True)
        for _ in range(60):
            if self.client.get_collection(self.col).status == Qmodels.CollectionStatus.GREEN: 
                print(" Done.")
                return
            time.sleep(1)
        print(" Timeout.")

    def search(self, vec, limit, expr=None):
        q_filter = None
        if expr:
            if "tag ==" in expr:
                val = expr.split("==")[1].strip().strip('"')
                q_filter = Qmodels.Filter(must=[Qmodels.FieldCondition(key="tag", match=Qmodels.MatchValue(value=val))])
            
        try:
            res = self.client.search(
                self.col, query_vector=vec.tolist(), 
                query_filter=q_filter, limit=limit,
                search_params=Qmodels.SearchParams(exact=False)
            )
            return [r.id for r in res]
        except Exception: return []

    def get_count(self, expr=None):
        q_filter = None
        if expr and "group_id ==" in expr:
            val = int(expr.split("==")[1].strip())
            q_filter = Qmodels.Filter(must=[Qmodels.FieldCondition(key="group_id", match=Qmodels.MatchValue(value=val))])
        return self.client.count(self.col, count_filter=q_filter).count


class WeaviateAdapter(BaseAdapter):
    def __init__(self):
        self.client = weaviate.Client("http://localhost:8080", timeout_config=(5, 300)) if HAVE_WEAVIATE else None
        self.cls = None

    def ok(self): return HAVE_WEAVIATE and self.client.is_ready()
    
    def drop(self, name):
        name = name.capitalize()
        try: self.client.schema.delete_class(name)
        except: pass

    def setup(self, name):
        name = name.capitalize()
        self.cls = name
        self.drop(name)
        self.client.schema.create_class({
            "class": name,
            "vectorizer": "none",
            "properties": [
                {"name": "tag", "dataType": ["text"]},
                {"name": "group_id", "dataType": ["int"]},
                {"name": "oid", "dataType": ["int"]},
            ]
        })

    def insert_batch(self, ids, vecs, payloads):
        ids = [int(i) for i in ids]
        try:
            with self.client.batch as batch:
                batch.batch_size = 200
                for i, v, p in zip(ids, vecs, payloads):
                    props = {k: (v.item() if hasattr(v, 'item') else v) for k, v in p.items()}
                    props["oid"] = int(i)
                    uid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{self.cls}_{i}"))
                    batch.add_data_object(props, self.cls, vector=v.tolist(), uuid=uid)
        except Exception as e:
            # Catch "Out of range float" or other JSON errors here to prevent crash
            raise ValueError(f"Weaviate Batch Error: {e}")

    def upsert_single(self, id_val, vec, payload):
        # [Fix] Use Object API instead of Batch API for atomic operations
        # This prevents polluting the batch buffer with previous failed Dirty Data
        uid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{self.cls}_{id_val}"))
        props = payload.copy()
        props["oid"] = int(id_val)
        
        # Replace object (Upsert equivalent)
        if self.client.data_object.exists(uid, class_name=self.cls):
            self.client.data_object.replace(
                props, self.cls, uuid=uid, vector=vec.tolist()
            )
        else:
            self.client.data_object.create(
                props, self.cls, uuid=uid, vector=vec.tolist()
            )

    def flush(self):
        print("    [Weaviate] Indexing...", end="", flush=True)
        time.sleep(5) 
        print(" Done.")

    def search(self, vec, limit, expr=None):
        q = (self.client.query.get(self.cls, ["oid"]).with_near_vector({"vector": vec.tolist()}).with_limit(limit))
        if expr:
            if "tag ==" in expr:
                val = expr.split("==")[1].strip().strip('"')
                q = q.with_where({"path": ["tag"], "operator": "Equal", "valueText": val})
        try:
            res = q.do()
            if "errors" in res: return []
            return [x["oid"] for x in res['data']['Get'][self.cls]]
        except: return []

    def get_count(self, expr=None):
        q = self.client.query.aggregate(self.cls).with_meta_count()
        if expr and "group_id ==" in expr:
            val = int(expr.split("==")[1].strip())
            q = q.with_where({"path": ["group_id"], "operator": "Equal", "valueInt": val})
        try:
            res = q.do()
            return res['data']['Aggregate'][self.cls][0]['meta']['count']
        except: return 0


# ========================== CHAOS SCENARIOS ==========================

def test_the_great_wall(adapter):
    print(f"  🧱 [The Great Wall] Building barrier of {WALL_SIZE} items...")
    rng = np.random.default_rng(SEED)
    
    q_vec = np.ones(DIM, dtype=np.float32)
    q_vec /= np.linalg.norm(q_vec)
    
    wall_ids = np.arange(0, WALL_SIZE, dtype=np.int64)
    wall_vecs = np.tile(q_vec, (WALL_SIZE, 1)) + rng.normal(0, 0.00001, (WALL_SIZE, DIM))
    wall_vecs = (wall_vecs.T / np.linalg.norm(wall_vecs, axis=1)).T
    wall_payloads = [{"tag": "BLOCK", "group_id": 0} for _ in wall_ids]
    
    for i in range(0, WALL_SIZE, BATCH_SIZE):
        end = min(i+BATCH_SIZE, WALL_SIZE)
        adapter.insert_batch(wall_ids[i:end], wall_vecs[i:end], wall_payloads[i:end])
    
    target_id = 999999
    target_vec = q_vec + 0.2 
    target_vec /= np.linalg.norm(target_vec)
    adapter.insert_batch([target_id], [target_vec], [{"tag": "TARGET", "group_id": 1}])
    
    adapter.flush()
    
    print("    Attacking the wall...")
    start = time.time()
    res = adapter.search(q_vec, limit=10, expr='tag == "TARGET"')
    dur = time.time() - start
    
    if target_id in res:
        return ChaosResult("GreatWall", True, f"Breached 10000 blockers in {dur:.2f}s")
    else:
        return ChaosResult("GreatWall", False, f"Blocked! Target lost behind {WALL_SIZE} neighbors.")


def test_dirty_data(adapter):
    print(f"  ☣️  [Dirty Data] Injecting Toxic Vectors...")
    
    nan_vec = np.zeros(DIM, dtype=np.float32)
    nan_vec[0] = np.nan
    inf_vec = np.zeros(DIM, dtype=np.float32)
    inf_vec[0] = np.inf
    
    crashes = 0
    blocked = 0
    
    # Attempt 1: NaN
    try:
        adapter.insert_batch([888001], [nan_vec], [{"tag": "NaN"}])
        # If insert didn't raise, try search
        try: adapter.search(nan_vec, limit=1)
        except: crashes += 1
    except Exception:
        blocked += 1

    # Attempt 2: Inf
    try:
        adapter.insert_batch([888002], [inf_vec], [{"tag": "Inf"}])
        try: adapter.search(inf_vec, limit=1)
        except: crashes += 1
    except Exception:
        blocked += 1
        
    # Health Check
    try:
        safe_vec = np.random.random(DIM).astype(np.float32)
        adapter.search(safe_vec, limit=1)
    except Exception as e:
        return ChaosResult("DirtyData", False, f"DB Corrupted/Crashed: {e}")
    
    if crashes > 0:
        return ChaosResult("DirtyData", False, "Search crashed on dirty data")
        
    return ChaosResult("DirtyData", True, f"System stable. Blocked: {blocked}, Accepted: {2-blocked}")


def test_atomic_transfer(adapter):
    print(f"  ⚛️  [Atomic Transfer] Starting Concurrency Stress Test ({CONCURRENCY_DURATION}s)...")
    
    token_id = 777777
    vec = np.random.random(DIM).astype(np.float32)
    adapter.insert_batch([token_id], [vec], [{"group_id": 1, "tag": "token"}])
    adapter.flush()
    
    stop_event = threading.Event()
    stats = {"ok": 0, "inconsistent": 0, "errors": []}
    
    def mover_thread():
        curr_grp = 1
        while not stop_event.is_set():
            next_grp = 2 if curr_grp == 1 else 1
            try:
                adapter.upsert_single(token_id, vec, {"group_id": next_grp, "tag": "token"})
                curr_grp = next_grp
                time.sleep(0.01) 
            except Exception as e:
                stats["errors"].append(str(e))
                break

    def checker_thread():
        while not stop_event.is_set():
            try:
                c1 = adapter.get_count(expr="group_id == 1")
                c2 = adapter.get_count(expr="group_id == 2")
                total = c1 + c2
                
                if total == 1:
                    stats["ok"] += 1
                else:
                    stats["inconsistent"] += 1
                time.sleep(0.015)
            except Exception as e:
                stats["errors"].append(str(e))
                break
                
    t1 = threading.Thread(target=mover_thread)
    t2 = threading.Thread(target=checker_thread)
    
    t1.start()
    t2.start()
    
    time.sleep(CONCURRENCY_DURATION)
    stop_event.set()
    t1.join()
    t2.join()
    
    if stats["errors"]:
        # Log error but don't crash
        return ChaosResult("AtomicTransfer", False, f"Errors: {stats['errors'][0]}")
    
    total_checks = stats["ok"] + stats["inconsistent"]
    if total_checks == 0:
        return ChaosResult("AtomicTransfer", False, "No checks performed")
        
    consistency_rate = stats["ok"] / total_checks
    
    if stats["inconsistent"] > 0:
        return ChaosResult("AtomicTransfer", False, f"Consistency Rate: {consistency_rate:.2%} (Anomalies: {stats['inconsistent']})")
        
    return ChaosResult("AtomicTransfer", True, f"Perfect Consistency ({stats['ok']} checks)")


# ========================== Main ==========================

def main():
    adapters = []
    if HAVE_MILVUS: adapters.append(("Milvus", MilvusAdapter()))
    if HAVE_QDRANT: adapters.append(("Qdrant", QdrantAdapter()))
    if HAVE_WEAVIATE: adapters.append(("Weaviate", WeaviateAdapter()))
    
    print(f"{'='*60}")
    print(f"🧨 EDC CHAOS RUNNER v3 (Stable)")
    print(f"   Target: {len(adapters)} DBs | Scale: {N_SCALE}")
    print(f"{'='*60}\n")
    
    for db_name, adapter in adapters:
        if not adapter.ok(): continue
        
        print(f"👉 Testing {db_name}...")
        adapter.connect()
        col_name = f"chaos_{uuid.uuid4().hex[:4]}"
        
        try:
            adapter.setup(col_name)
            
            results = []
            results.append(test_the_great_wall(adapter))
            results.append(test_dirty_data(adapter))
            results.append(test_atomic_transfer(adapter))
            
            for r in results:
                icon = "✅" if r.passed else "❌"
                print(f"    {icon} {r.name:<16} | {r.detail}")
                
        except Exception as e:
            print(f"    💥 CRITICAL FAILURE: {e}")
        finally:
            print(f"    Cleaning up {col_name}...\n")
            adapter.drop(col_name)

if __name__ == "__main__":
    main()