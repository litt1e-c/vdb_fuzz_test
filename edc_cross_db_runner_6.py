"""
EDC "TITAN" Runner: Production-Grade Vector DB A/B Verification
Scale: Configurable (Default N=50,000 for mechanism check, scalable to 1M+)
Methodology:
  1. RAW Mode (Brute Force/Flat): Establishes the "Ground Truth".
  2. OPT Mode (HNSW/Optimizers): Represents "Production Behavior".
  3. Comparison:
     - Passed RAW + Failed OPT = Algorithm/Index Bug (CRITICAL).
     - Failed RAW + Failed OPT = Logic/API Bug.
"""
from __future__ import annotations
import time
import uuid
import warnings
import numpy as np
from dataclasses import dataclass
from typing import List, Any, Dict, Optional

# --- Configuration ---
N_SCALE = 200000      # 数据量：5w足够触发Segment合并，如需压测可改为 200000
DIM = 768            # 维度
BATCH_SIZE = 2000    # 批处理大小
SEED = 2025          # 固定种子确保数据一致性
TOPK = 50

# --- Suppress Noise ---
warnings.filterwarnings("ignore")

# --- Database Drivers ---
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
class TestResult:
    scenario: str
    mode: str
    passed: bool
    latency: float
    detail: str


# ========================== Abstract Adapter ==========================

class BaseAdapter:
    def connect(self): pass
    def ok(self): return False
    def drop(self, name): pass
    # Setup collection with specific mode (Raw vs Optimized)
    def setup(self, name, mode: str): raise NotImplementedError
    def insert_batch(self, ids, vecs, payloads): pass
    def upsert_single(self, id_val, vec, payload): pass
    def delete(self, ids): pass
    def flush(self): pass
    def search(self, vec, limit, expr=None): raise NotImplementedError
    def get_by_id(self, id_val): raise NotImplementedError


# ========================== Milvus Implementation ==========================

class MilvusAdapter(BaseAdapter):
    def __init__(self):
        self.uri = ("127.0.0.1", "19530")
        self.col = None
        self.mode = "RAW"

    def ok(self): return HAVE_MILVUS
    def connect(self):
        if HAVE_MILVUS:
            try: connections.connect("default", host=self.uri[0], port=self.uri[1])
            except: pass

    def drop(self, name):
        if utility.has_collection(name): utility.drop_collection(name)

    def setup(self, name, mode: str):
        self.mode = mode
        self.drop(name)
        
        # Schema Definition
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=DIM),
            FieldSchema(name="tag", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="val", dtype=DataType.INT64),
            FieldSchema(name="group_id", dtype=DataType.INT64),
        ]
        schema = CollectionSchema(fields, description="EDC Test")
        # Critical: Strong consistency to ensure Read-Your-Write
        self.col = Collection(name, schema, consistency_level="Strong")

    def insert_batch(self, ids, vecs, payloads):
        ids = [int(i) for i in ids]
        tag_col = [p.get("tag", "") for p in payloads]
        val_col = [int(p.get("val", 0)) for p in payloads]
        grp_col = [int(p.get("group_id", 0)) for p in payloads]
        self.col.insert([ids, vecs, tag_col, val_col, grp_col])

    def upsert_single(self, id_val, vec, payload):
        tag = payload.get("tag", "")
        val = int(payload.get("val", 0))
        grp = int(payload.get("group_id", 0))
        self.col.upsert([[int(id_val)], [vec], [tag], [val], [grp]])

    def delete(self, ids):
        expr = f"id in {list(ids)}"
        self.col.delete(expr)

    def flush(self):
        self.col.flush()
        
        # Mode-Switching Logic for Index
        if self.mode == "OPT":
            # Optimized: HNSW
            index_params = {
                "metric_type": "L2", 
                "index_type": "HNSW", 
                "params": {"M": 32, "efConstruction": 200}
            }
            self.col.create_index("vector", index_params)
        else:
            # Raw: FLAT (Brute Force)
            index_params = {
                "metric_type": "L2", 
                "index_type": "FLAT", 
                "params": {}
            }
            self.col.create_index("vector", index_params)
            
        self.col.load()

    def search(self, vec, limit, expr=None):
        # Config params based on mode
        if self.mode == "OPT":
            search_params = {"metric_type": "L2", "params": {"ef": 256}}
        else:
            search_params = {"metric_type": "L2", "params": {}} # Flat needs no ef

        try:
            res = self.col.search(
                [vec], "vector", search_params, 
                limit=limit, expr=expr, 
                output_fields=["id", "tag", "val"],
                consistency_level="Strong"
            )
            if not res: return []
            return [{"id": h.id, "tag": h.entity.get("tag"), "val": h.entity.get("val")} for h in res[0]]
        except Exception as e:
            # print(f"Milvus error: {e}")
            return []

    def get_by_id(self, id_val):
        res = self.col.query(f"id == {id_val}", output_fields=["tag", "val"], consistency_level="Strong")
        return res[0] if res else None


# ========================== Qdrant Implementation ==========================

class QdrantAdapter(BaseAdapter):
    def __init__(self):
        # Increased timeout for large batches
        self.client = QdrantClient("http://localhost:6333", timeout=300) if HAVE_QDRANT else None
        self.col = None
        self.mode = "RAW"

    def ok(self): return HAVE_QDRANT
    def drop(self, name): self.client.delete_collection(name)

    def setup(self, name, mode: str):
        self.mode = mode
        self.col = name
        
        # Config Optimizers
        if mode == "OPT":
            # HNSW Enabled (Default behavior)
            # We explicitly set threshold to ensure it triggers after ingest
            opt_config = Qmodels.OptimizersConfigDiff(indexing_threshold=20000)
        else:
            # HNSW Disabled (Raw Mode)
            # indexing_threshold=0 prevents HNSW graph creation
            opt_config = Qmodels.OptimizersConfigDiff(indexing_threshold=0)

        if not self.client.collection_exists(name):
            self.client.create_collection(
                name,
                vectors_config=Qmodels.VectorParams(size=DIM, distance=Qmodels.Distance.EUCLID),
                optimizers_config=opt_config
            )

    def insert_batch(self, ids, vecs, payloads):
        points = []
        for i, v, p in zip(ids, vecs, payloads):
            # Clean numpy types
            clean = {}
            for k, val in p.items():
                if isinstance(val, (np.integer, np.floating)): clean[k] = val.item()
                else: clean[k] = val
            points.append(Qmodels.PointStruct(id=int(i), vector=v.tolist(), payload=clean))
        
        # Use Bulk Upload for reliability
        self.client.upload_points(
            collection_name=self.col, points=points, batch_size=1024, parallel=4, wait=True
        )

    def upsert_single(self, id_val, vec, payload):
        self.client.upsert(
            self.col, points=[Qmodels.PointStruct(id=int(id_val), vector=vec.tolist(), payload=payload)]
        )

    def delete(self, ids):
        self.client.delete(self.col, points_selector=Qmodels.PointIdsList(points=[int(i) for i in ids]))

    def flush(self):
        # Wait for persistence
        time.sleep(1) 
        if self.mode == "OPT":
            # Force Optimization for HNSW
            self.client.update_collection(self.col, optimizers_config=Qmodels.OptimizersConfigDiff(indexing_threshold=10000))
            # Wait for Green
            for _ in range(30):
                if self.client.get_collection(self.col).status == Qmodels.CollectionStatus.GREEN: break
                time.sleep(1)

    def search(self, vec, limit, expr=None):
        q_filter = None
        if expr:
            # Simple expression parser for Qdrant
            if "tag ==" in expr:
                val = expr.split("==")[1].strip().strip('"')
                q_filter = Qmodels.Filter(must=[Qmodels.FieldCondition(key="tag", match=Qmodels.MatchValue(value=val))])
            elif "val >" in expr:
                v = int(expr.split(">")[1].strip())
                q_filter = Qmodels.Filter(must=[Qmodels.FieldCondition(key="val", range=Qmodels.Range(gt=v))])

        # exact=True forces Brute Force in Qdrant API
        # exact=False allows HNSW usage
        force_exact = (self.mode == "RAW")
        
        res = self.client.search(
            self.col, query_vector=vec.tolist(), query_filter=q_filter, 
            limit=limit, search_params=Qmodels.SearchParams(exact=force_exact)
        )
        return [{"id": r.id, "tag": r.payload.get("tag"), "val": r.payload.get("val")} for r in res]

    def get_by_id(self, id_val):
        res = self.client.retrieve(self.col, ids=[int(id_val)])
        return res[0].payload if res else None


# ========================== Weaviate Implementation ==========================

class WeaviateAdapter(BaseAdapter):
    def __init__(self):
        self.client = weaviate.Client("http://localhost:8080", timeout_config=(10, 300)) if HAVE_WEAVIATE else None
        self.cls = None
        self.mode = "RAW"

    def ok(self): return HAVE_WEAVIATE and self.client.is_ready()

    def drop(self, name):
        name = name.capitalize()
        try: self.client.schema.delete_class(name)
        except: pass

    def setup(self, name, mode: str):
        self.mode = mode
        name = name.capitalize()
        self.cls = name
        self.drop(name)
        
        # Config Index Type
        idx_type = "hnsw" if mode == "OPT" else "flat"
        
        self.client.schema.create_class({
            "class": name,
            "vectorizer": "none",
            "vectorIndexType": idx_type, # Key for A/B testing
            "properties": [
                {"name": "tag", "dataType": ["text"]}, 
                {"name": "val", "dataType": ["int"]},
                {"name": "group_id", "dataType": ["int"]},
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
        # Weaviate polling
        print("    [Weaviate] Indexing...", end="", flush=True)
        for _ in range(60):
            try:
                res = self.client.query.aggregate(self.cls).with_meta_count().do()
                cnt = res['data']['Aggregate'][self.cls][0]['meta']['count']
                if cnt >= N_SCALE: 
                    print(" Done.")
                    return
            except: pass
            time.sleep(2)
        print(" Timeout.")

    def search(self, vec, limit, expr=None):
        q = (self.client.query.get(self.cls, ["oid", "tag", "val"])
             .with_near_vector({"vector": vec.tolist()})
             .with_limit(limit))
        
        if expr:
            w_filter = None
            if "tag ==" in expr:
                val = expr.split("==")[1].strip().strip('"')
                w_filter = {"path": ["tag"], "operator": "Equal", "valueText": val}
            elif "val >" in expr:
                v = int(expr.split(">")[1].strip())
                w_filter = {"path": ["val"], "operator": "GreaterThan", "valueInt": v}
            if w_filter: q = q.with_where(w_filter)
        
        try:
            res = q.do()
            if "errors" in res: return []
            items = res['data']['Get'][self.cls]
            return [{"id": item["oid"], "tag": item.get("tag"), "val": item.get("val")} for item in items]
        except: return []

    def get_by_id(self, id_val):
        uid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{self.cls}_{id_val}"))
        try:
            res = self.client.data_object.get_by_id(uid, class_name=self.cls)
            if res: return res['properties']
        except: pass
        return None


# ========================== Test Scenarios ==========================

def get_data_gen():
    """Generates data stream to ensure consistency across modes."""
    rng = np.random.default_rng(SEED) # Fixed seed
    return rng

def populate_db(adapter, rng):
    print(f"  🌊 Populating {N_SCALE} items...")
    inserted = 0
    # Special Needles for Logic Tests
    # ID 100: tag="A", val=50
    # ID 101: tag="B", val=150
    needles_ids = [100, 101]
    
    for batch_start in range(0, N_SCALE, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, N_SCALE)
        count = batch_end - batch_start
        ids = np.arange(batch_start, batch_end, dtype=np.int64)
        vecs = rng.normal(0, 0.1, (count, DIM)).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        
        payloads = []
        for i in range(count):
            abs_id = batch_start + i
            if abs_id == 100: p = {"tag": "A", "val": 50, "group_id": 1}
            elif abs_id == 101: p = {"tag": "B", "val": 150, "group_id": 1}
            else: p = {"tag": "common", "val": 10, "group_id": 9}
            payloads.append(p)
            
        adapter.insert_batch(ids, vecs, payloads)
        inserted += count
        print(f"    Inserted: {inserted}...", end="\r")
    
    print("\n  🔨 Flushing...")
    adapter.flush()
    return vecs[0] # Return a query vector

# --- Test 1: Update Visibility (The "Stale Read" Check) ---
def test_update_visibility(adapter, q_vec, mode):
    target_id = 100
    # 1. Update ID 100 tag A -> UPDATED
    rng = np.random.default_rng(SEED)
    new_vec = rng.random(DIM).astype(np.float32)
    new_vec /= np.linalg.norm(new_vec)
    
    adapter.upsert_single(target_id, new_vec, {"tag": "UPDATED", "val": 999, "group_id": 1})
    adapter.flush() # Ensure it lands
    
    # 2. Search for Old Tag "A"
    t0 = time.time()
    res_old = adapter.search(new_vec, limit=20, expr='tag == "A"')
    found_old = any(r["id"] == target_id for r in res_old)
    
    # 3. Search for New Tag "UPDATED"
    res_new = adapter.search(new_vec, limit=20, expr='tag == "UPDATED"')
    found_new = any(r["id"] == target_id for r in res_new)
    lat = (time.time() - t0) * 1000
    
    if found_old:
        return TestResult("UpdateVis", mode, False, lat, "Stale: Found old tag")
    if not found_new:
        return TestResult("UpdateVis", mode, False, lat, "Lost: New tag not found")
        
    return TestResult("UpdateVis", mode, True, lat, "Consistent")

# --- Test 2: Zombie Resurrection ---
def test_zombie(adapter, q_vec, mode):
    target_id = 101
    
    # 1. Delete
    adapter.delete([target_id])
    time.sleep(1) # Allow propagation
    
    # 2. Check
    check = adapter.get_by_id(target_id)
    if check: return TestResult("Zombie", mode, False, 0, "Delete failed")
    
    # 3. Re-insert
    rng = np.random.default_rng(SEED+1)
    vec = rng.random(DIM).astype(np.float32)
    adapter.upsert_single(target_id, vec, {"tag": "ZOMBIE", "val": 0})
    adapter.flush()
    
    # 4. Search
    res = adapter.search(vec, limit=10)
    found = any(r["id"] == target_id for r in res)
    
    if not found: return TestResult("Zombie", mode, False, 0, "Resurrection failed")
    return TestResult("Zombie", mode, True, 0, "Cycle OK")

# --- Test 3: The Filtering Curse (Selectivity) ---
def test_filtering_curse(adapter, q_vec, mode):
    # Common tag "common" (most items) vs Rare tag "UPDATED" (1 item)
    # We use "common" which exists in N-2 items.
    
    t0 = time.perf_counter()
    res_common = adapter.search(q_vec, limit=50, expr='tag == "common"')
    lat_common = (time.perf_counter() - t0) * 1000
    
    # Rare query (e.g. val > 1000, likely none or few) -> Empty result scan
    t0 = time.perf_counter()
    res_rare = adapter.search(q_vec, limit=50, expr='val > 9000') # Should be 0
    lat_rare = (time.perf_counter() - t0) * 1000
    
    detail = f"Common: {lat_common:.2f}ms, Rare: {lat_rare:.2f}ms"
    
    # In OPT mode, Rare should NOT be 10x slower than Common
    if lat_rare > lat_common * 10 and lat_rare > 100: # threshold 100ms
        return TestResult("FilterPerf", mode, False, lat_rare, f"Cliff detected: {detail}")
        
    return TestResult("FilterPerf", mode, True, lat_rare, f"Stable: {detail}")


# ========================== Controller ==========================

def run_suite(adapter, mode):
    rng = get_data_gen()
    q_vec = populate_db(adapter, rng)
    
    results = []
    print(f"    Running tests in [{mode}] mode...")
    
    results.append(test_update_visibility(adapter, q_vec, mode))
    results.append(test_zombie(adapter, q_vec, mode))
    results.append(test_filtering_curse(adapter, q_vec, mode))
    
    return results

def main():
    adapters = []
    if HAVE_MILVUS: adapters.append(("Milvus", MilvusAdapter()))
    if HAVE_QDRANT: adapters.append(("Qdrant", QdrantAdapter()))
    if HAVE_WEAVIATE: adapters.append(("Weaviate", WeaviateAdapter()))
    
    print(f"{'='*60}")
    print(f"🛡️  EDC TITAN: A/B Mechanism Verification")
    print(f"   Scale: {N_SCALE} | Modes: RAW (Flat) vs OPT (HNSW)")
    print(f"{'='*60}\n")
    
    for db_name, adapter in adapters:
        if not adapter.ok(): continue
        
        print(f"👉 Target: {db_name}")
        adapter.connect()
        
        # --- A/B Execution ---
        final_report = {}
        
        for mode in ["RAW", "OPT"]:
            col_name = f"titan_{mode.lower()}_{uuid.uuid4().hex[:4]}"
            print(f"\n  [Phase: {mode}] Setup & Ingest...")
            
            try:
                adapter.setup(col_name, mode)
                run_res = run_suite(adapter, mode)
                
                # Print phase results
                for r in run_res:
                    icon = "✅" if r.passed else "❌"
                    print(f"    {icon} {r.scenario:<15} | {r.detail}")
                    
                final_report[mode] = {r.scenario: r for r in run_res}
                
            except Exception as e:
                print(f"    💥 CRASH in {mode}: {e}")
                # import traceback; traceback.print_exc()
            finally:
                adapter.drop(col_name)
        
        # --- Comparative Analysis ---
        print(f"\n  📊 Comparative Analysis for {db_name}:")
        if "RAW" in final_report and "OPT" in final_report:
            raw_res = final_report["RAW"]
            opt_res = final_report["OPT"]
            
            for sc in raw_res:
                r_raw = raw_res.get(sc)
                r_opt = opt_res.get(sc)
                
                if not r_opt: continue
                
                if r_raw.passed and not r_opt.passed:
                    print(f"    🚨 ALGO BUG: '{sc}' passed in RAW but failed in OPT.")
                    print(f"       -> Likely an Indexing/Optimization artifact.")
                elif not r_raw.passed and not r_opt.passed:
                    print(f"    ⚠️  LOGIC BUG: '{sc}' failed in BOTH modes.")
                    print(f"       -> Likely a Storage/API logic issue.")
                elif r_raw.passed and r_opt.passed:
                    print(f"    ✅ '{sc}': Robust across modes.")
        else:
            print("    ⚠️  Could not complete comparison (One mode crashed).")
            
        print("-" * 60)

if __name__ == "__main__":
    main()