"""
EDC "HELLFIRE" Runner v5: The Final Verification
Scale: N=200,000 | DIM=768
Fixes:
1. Data Gen: Use IDs > N_SCALE for needles to avoid overwrite/collision within batch.
2. Weaviate: Fix API syntax for retrieving distance (_additional).
3. Weaviate: Increase index build timeout.
4. Milvus: Pass expression correctly in search_with_score.
"""
from __future__ import annotations
import time
import uuid
import warnings
import numpy as np
from dataclasses import dataclass
from typing import List, Any, Dict

# --- Suppress Warnings ---
warnings.filterwarnings("ignore")

# --- Global Config ---
N_SCALE = 200000     # 20万数据量
DIM = 768            
BATCH_SIZE = 2000    
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


# ========================== Robust Adapters ==========================

class BaseAdapter:
    def connect(self): pass
    def ok(self): return False
    def drop(self, name): pass
    def create_schema(self, name): raise NotImplementedError
    def insert_batch(self, ids, vecs, payloads): pass
    def flush(self): pass
    def search_with_score(self, vec, limit, expr=None): raise NotImplementedError
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
            FieldSchema(name="val", dtype=DataType.INT64), 
            FieldSchema(name="group", dtype=DataType.INT64), 
        ]
        schema = CollectionSchema(fields)
        self.col = Collection(name, schema, consistency_level="Strong")

    def insert_batch(self, ids, vecs, payloads):
        ids = [int(i) for i in ids]
        tag_col = [p.get("tag", "") for p in payloads]
        val_col = [int(p.get("val", 0)) for p in payloads]
        grp_col = [int(p.get("group", 0)) for p in payloads]
        self.col.insert([ids, vecs, tag_col, val_col, grp_col])

    def flush(self):
        self.col.flush()
        index_params = {
            "metric_type": "L2", 
            "index_type": "HNSW", 
            "params": {"M": 32, "efConstruction": 200}
        }
        self.col.create_index("vector", index_params)
        self.col.load()

    def search_with_score(self, vec, limit, expr=None):
        search_params = {"metric_type": "L2", "params": {"ef": 256}}
        
        try:
            res = self.col.search(
                [vec], "vector", search_params, 
                limit=limit, expr=expr, 
                output_fields=["id", "tag", "val", "group"],
                consistency_level="Strong"
            )
            if not res: return []
            return [{
                "id": h.id, 
                "score": h.distance,  
                "tag": h.entity.get("tag"),
                "val": h.entity.get("val"),
                "group": h.entity.get("group")
            } for h in res[0]]
        except Exception as e:
            print(f"Milvus error: {e}")
            return []

    def get_by_id(self, id_val):
        res = self.col.query(f"id == {id_val}", output_fields=["val"], consistency_level="Strong")
        return res[0] if res else None


class QdrantAdapter(BaseAdapter):
    def __init__(self):
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
                optimizers_config=Qmodels.OptimizersConfigDiff(indexing_threshold=0)
            )

    def insert_batch(self, ids, vecs, payloads):
        points = []
        for i, v, p in zip(ids, vecs, payloads):
            points.append(Qmodels.PointStruct(id=int(i), vector=v.tolist(), payload=p))
        
        self.client.upload_points(
            collection_name=self.col,
            points=points,
            batch_size=1024,
            parallel=4,
            wait=True
        )

    def flush(self):
        print("    [Qdrant] Optimizing...", end="", flush=True)
        self.client.update_collection(
            collection_name=self.col,
            optimizers_config=Qmodels.OptimizersConfigDiff(indexing_threshold=20000)
        )
        # Wait for green
        for _ in range(60):
            info = self.client.get_collection(self.col)
            if info.status == Qmodels.CollectionStatus.GREEN:
                print(" Done.")
                return
            time.sleep(2)
        print(" Timeout.")

    def search_with_score(self, vec, limit, expr=None):
        q_filter = None
        if expr:
            musts = []
            if "val >" in expr:
                v = int(expr.split(">")[1].strip())
                musts.append(Qmodels.FieldCondition(key="val", range=Qmodels.Range(gt=v)))
            
            if "group ==" in expr and "||" in expr:
                # (Group=1 OR Group=2)
                g_filter = Qmodels.Filter(should=[
                    Qmodels.FieldCondition(key="group", match=Qmodels.MatchValue(value=1)),
                    Qmodels.FieldCondition(key="group", match=Qmodels.MatchValue(value=2))
                ])
                # (Tag=A OR Tag=B)
                t_filter = Qmodels.Filter(should=[
                    Qmodels.FieldCondition(key="tag", match=Qmodels.MatchValue(value="A")),
                    Qmodels.FieldCondition(key="tag", match=Qmodels.MatchValue(value="B"))
                ])
                musts.append(Qmodels.Filter(must=[g_filter, t_filter]))
            
            if musts:
                q_filter = Qmodels.Filter(must=musts)

        res = self.client.search(
            self.col, query_vector=vec.tolist(), query_filter=q_filter, 
            limit=limit,
            search_params=Qmodels.SearchParams(exact=False)
        )
        return [{
            "id": r.id, 
            "score": r.score, 
            "tag": r.payload.get("tag"),
            "val": r.payload.get("val"),
            "group": r.payload.get("group")
        } for r in res]

    def get_by_id(self, id_val):
        res = self.client.retrieve(self.col, ids=[int(id_val)])
        return res[0].payload if res else None


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
                    {"name": "val", "dataType": ["int"]},
                    {"name": "group", "dataType": ["int"]},
                    {"name": "oid", "dataType": ["int"]}, 
                ]
            })

    def insert_batch(self, ids, vecs, payloads):
        ids = [int(i) for i in ids]
        with self.client.batch as batch:
            batch.batch_size = 200
            for i, v, p in zip(ids, vecs, payloads):
                props = p.copy()
                props["oid"] = int(i)
                uid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{self.cls}_{i}"))
                batch.add_data_object(props, self.cls, vector=v.tolist(), uuid=uid)

    def flush(self):
        print("    [Weaviate] Waiting for indexing...", end="", flush=True)
        # Increased wait time for 200k items
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

    def search_with_score(self, vec, limit, expr=None):
        # [Fix] Correct syntax for retrieving distance in Weaviate v3
        q = (self.client.query.get(self.cls, ["oid", "tag", "val", "group"])
             .with_near_vector({"vector": vec.tolist()}) 
             .with_additional(["distance"])  # <--- FIXED
             .with_limit(limit))
        
        if expr:
            w_filter = None
            if "val >" in expr:
                v = int(expr.split(">")[1].strip())
                w_filter = {"path": ["val"], "operator": "GreaterThan", "valueInt": v}
            elif "group ==" in expr and "||" in expr:
                # (Group=1 OR Group=2) AND (Tag=A OR Tag=B)
                g_filter = {"operator": "Or", "operands": [
                    {"path": ["group"], "operator": "Equal", "valueInt": 1},
                    {"path": ["group"], "operator": "Equal", "valueInt": 2}
                ]}
                t_filter = {"operator": "Or", "operands": [
                    {"path": ["tag"], "operator": "Like", "valueText": "A"},
                    {"path": ["tag"], "operator": "Like", "valueText": "B"}
                ]}
                w_filter = {"operator": "And", "operands": [g_filter, t_filter]}
            
            if w_filter: q = q.with_where(w_filter)
        
        try:
            res = q.do()
            if "errors" in res: return []
            items = res['data']['Get'][self.cls]
            return [{
                "id": item["oid"], 
                "score": item.get("_additional", {}).get("distance", 0.0),
                "tag": item.get("tag"),
                "val": item.get("val"),
                "group": item.get("group")
            } for item in items]
        except Exception as e:
            print(f"Weaviate search err: {e}")
            return []

    def get_by_id(self, id_val):
        return None 


# ========================== Generator & Setup ==========================

def generate_and_insert_logic_data(db, name):
    print(f"  🌊 Generating {N_SCALE} items for Logic Tests...")
    rng = np.random.default_rng(SEED)
    
    # Define Special IDs OUTSIDE the noise range (200000+)
    # This prevents noise data (0-199999) from overwriting our logic needles
    ID_TARGET_1 = N_SCALE + 100
    ID_TARGET_2 = N_SCALE + 101
    ID_DISTRACT_1 = N_SCALE + 102
    ID_DISTRACT_2 = N_SCALE + 103
    
    special_ids = [ID_TARGET_1, ID_TARGET_2]
    q_vec = None

    # Insert Noise
    for batch_start in range(0, N_SCALE, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, N_SCALE)
        count = batch_end - batch_start
        
        ids = np.arange(batch_start, batch_end, dtype=np.int64)
        vecs = rng.normal(0, 0.1, (count, DIM)).astype(np.float32)
        
        groups = rng.integers(0, 10, size=count)
        vals = rng.integers(0, 101, size=count)
        tags_pool = np.array(["A", "B", "C", "D"])
        tags = rng.choice(tags_pool, size=count)
        
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        
        # Capture a vector for query (use the first noise vector as query base)
        if batch_start == 0:
            q_vec = vecs[0].copy()
            
        payloads = [
            {"tag": str(t), "val": int(v), "group": int(g)}
            for t, v, g in zip(tags, vals, groups)
        ]
        
        db.insert_batch(ids, vecs, payloads)
        print(f"    Inserted Noise: {batch_end}...", end="\r")

    # Insert Logic Needles (Separate Batch)
    print(f"\n    Inserting Logic Needles...")
    # All needles use q_vec (distance 0) to ensure they are Top candidates
    needle_vecs = np.tile(q_vec, (4, 1)) 
    
    needle_ids = [ID_TARGET_1, ID_TARGET_2, ID_DISTRACT_1, ID_DISTRACT_2]
    needle_payloads = [
        {"group": 1, "tag": "A", "val": 90}, # Target
        {"group": 2, "tag": "B", "val": 90}, # Target
        {"group": 3, "tag": "A", "val": 90}, # Wrong Group
        {"group": 1, "tag": "C", "val": 90}, # Wrong Tag
    ]
    
    db.insert_batch(needle_ids, needle_vecs, needle_payloads)
    
    print(f"  🔨 Flushing...")
    db.flush()
    return q_vec, special_ids


# ========================== Tests ==========================

def test_score_monotonicity(db, q_vec):
    print("    [Ranking] Checking score monotonicity...")
    res = db.search_with_score(q_vec, limit=50)
    
    if len(res) < 2:
        return ScenarioResult("score_mono", False, "Not enough results")
    
    scores = [r["score"] for r in res]
    
    is_ascending = all(scores[i] <= scores[i+1] for i in range(len(scores)-1))
    is_descending = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
    
    if is_ascending or is_descending:
        direction = "Asc" if is_ascending else "Desc"
        return ScenarioResult("score_mono", True, f"Strictly monotonic: {direction}")
    
    fails = []
    for i in range(len(scores)-1):
        if not (scores[i] <= scores[i+1] or scores[i] >= scores[i+1]): 
             pass # Logic error in check above, effectively checking sort
        # Simple check: L2 should be Asc, Cosine Desc. 
        # Just fail if it wiggles.
        if scores[i] < scores[i+1] and scores[i] > scores[i+1]: # Impossible
            pass 
            
    return ScenarioResult("score_mono", False, f"Ranking violation (Not sorted)")


def test_nested_boolean(db, q_vec):
    print("    [Logic] Checking nested boolean: (G1|G2) & (TagA|TagB)...")
    
    # We want to find the needles we inserted
    # ID_TARGET_1 (100+N) and ID_TARGET_2 (101+N)
    
    expr = "(group == 1 || group == 2) && (tag == 'A' || tag == 'B')"
    
    res = db.search_with_score(q_vec, limit=20, expr=expr)
    found_ids = {r["id"] for r in res}
    
    targets = [N_SCALE + 100, N_SCALE + 101]
    distractors = [N_SCALE + 102, N_SCALE + 103]
    
    # Recall Check
    if not all(t in found_ids for t in targets):
        return ScenarioResult("nested_bool", False, f"Recall fail: Expected {targets}. Found: {list(found_ids)}")
        
    # Precision Check
    if any(d in found_ids for d in distractors):
         return ScenarioResult("nested_bool", False, f"Filter Leakage: Found distractors {distractors}")
        
    return ScenarioResult("nested_bool", True, f"Perfect Recall & Precision on Nested Logic")


def test_range_boundary(db, q_vec):
    print("    [Logic] Checking Range Boundary (Val > 90)...")
    
    # Needles have val=90. Filter is > 90.
    # They should NOT appear.
    
    res = db.search_with_score(q_vec, limit=50, expr="val > 90")
    
    needles = [N_SCALE + 100, N_SCALE + 101, N_SCALE + 102, N_SCALE + 103]
    leaked = [r["id"] for r in res if r["id"] in needles]
    
    if leaked:
        return ScenarioResult("range_bound", False, f"Boundary Error: Found ID {leaked} (Val=90) in '> 90' filter")
    
    return ScenarioResult("range_bound", True, f"Boundary respected.")


# ========================== Main ==========================

def main():
    runners = [
        test_score_monotonicity,
        test_nested_boolean,
        test_range_boundary
    ]
    
    adapters = []
    if HAVE_MILVUS: adapters.append(("Milvus", MilvusAdapter()))
    if HAVE_QDRANT: adapters.append(("Qdrant", QdrantAdapter()))
    if HAVE_WEAVIATE: adapters.append(("Weaviate", WeaviateAdapter()))
    
    print(f"{'='*60}")
    print(f"🔥 HELLFIRE RUNNER v5: The Final Verification")
    print(f"   Scale: {N_SCALE} | Databases: {[n for n, _ in adapters]}")
    print(f"{'='*60}\n")
    
    for db_name, adapter in adapters:
        if not adapter.ok(): continue
        
        print(f"--- 🚀 Target: {db_name} ---")
        adapter.connect()
        
        run_id = uuid.uuid4().hex[:4]
        col_name = f"hf_v5_{run_id}"
        
        try:
            adapter.drop(col_name)
            adapter.create_schema(col_name)
            
            q_vec, _ = generate_and_insert_logic_data(adapter, col_name)
            
            for runner in runners:
                start_t = time.time()
                try:
                    res = runner(adapter, q_vec)
                    dur = time.time() - start_t
                    icon = "✅" if res.passed else "❌"
                    print(f"{icon} {res.name:<15} | {dur:.2f}s | {res.detail}")
                except Exception as e:
                    print(f"❌ {runner.__name__:<15} | CRASH: {e}")

        except Exception as e:
            print(f"💥 Critical Failure: {e}")
        finally:
            print(f"   Cleaning up {col_name}...\n")
            adapter.drop(col_name)

if __name__ == "__main__":
    main()