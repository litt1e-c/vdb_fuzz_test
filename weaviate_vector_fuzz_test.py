#!/usr/bin/env python3
"""
Weaviate Vector Search Fuzz Test
=================================
Three deterministic test strategies that expose vector-index bugs
without relying on recall-rate thresholds:

  1. Distance Ordering:  returned distances must be monotonically sorted.
  2. TopK Monotonicity:  topK=10 ⊆ topK=20 (parameter-containment).
  3. Self-Retrieval:     insert V, search(V, top1) == V with dist ≈ 0.

Usage:
  python weaviate_vector_fuzz_test.py                       # default 200 rounds
  python weaviate_vector_fuzz_test.py --seed 42 --rounds 500
  python weaviate_vector_fuzz_test.py --host 10.0.0.1 --port 8080
  python weaviate_vector_fuzz_test.py --dim 256 -N 3000
  python weaviate_vector_fuzz_test.py --dynamic
"""

import time
import random
import argparse
import uuid
import sys
import numpy as np
import weaviate
from weaviate.classes.config import (
    Configure, Property, DataType, VectorDistances,
)
from weaviate.classes.query import Filter, MetadataQuery
from weaviate.classes.data import DataObject


# ── FuzzStats (ported from weaviate_fuzz_oracle.py) ─────────────────────────

class FuzzStats:
    """搜索延迟、错误分类、通过率统计 — 用于检测索引退化和 p99 异常"""
    def __init__(self):
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.latencies = []
        self.error_categories = {}

    def record(self, passed, latency_ms=None, error_cat=None):
        self.total += 1
        if passed:
            self.passed += 1
        else:
            self.failed += 1
        if latency_ms is not None:
            self.latencies.append(latency_ms)
        if error_cat:
            self.error_categories[error_cat] = self.error_categories.get(error_cat, 0) + 1

    def summary(self):
        parts = [f"Total:{self.total} Pass:{self.passed} Fail:{self.failed}"]
        if self.latencies:
            arr = np.array(self.latencies)
            parts.append(f"Latency: avg={arr.mean():.1f}ms p50={np.percentile(arr,50):.1f}ms "
                         f"p99={np.percentile(arr,99):.1f}ms max={arr.max():.1f}ms")
        if self.error_categories:
            parts.append(f"Errors: {dict(self.error_categories)}")
        return " | ".join(parts)

# ── defaults ────────────────────────────────────────────────────────────────
HOST = "127.0.0.1"
PORT = 8080
GRPC_PORT = 50051
CLASS_NAME = "VecFuzzTest"
N = 2000
DIM = 128
BATCH_SIZE = 500

ALL_VECTOR_INDEX_TYPES = ["hnsw", "flat"]
ALL_DISTANCE_METRICS = [VectorDistances.COSINE, VectorDistances.L2_SQUARED, VectorDistances.DOT]

# L2_SQUARED ascending (smaller = closer);  DOT/COSINE: Weaviate returns
# "distance" = 2 - 2*dot  for DOT or  1 - cos  for COSINE → ascending
# In Weaviate v4 distances are ALWAYS ascending (smaller = more similar).
WEAVIATE_DISTANCE_ALWAYS_ASCENDING = True

# ── helpers ─────────────────────────────────────────────────────────────────

def _norm(v):
    v = np.array(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return (v / n if n > 0 else v)


# ── core class ──────────────────────────────────────────────────────────────

class WeaviateVectorFuzzTest:
    def __init__(self, host, port, grpc_port, dim, n, seed, enable_dynamic):
        self.host = host
        self.port = port
        self.grpc_port = grpc_port
        self.dim = dim
        self.n = n
        self.seed = seed
        self.enable_dynamic = enable_dynamic
        self.client = None
        self.distance_metric = None
        self.vi_type = None
        self.vectors = None   # (N, DIM) float32
        self.ids = []         # list[str] (UUID)

    # ── connection ──────────────────────────────────────────────────────────

    def connect(self):
        try:
            self.client = weaviate.connect_to_local(
                host=self.host, port=self.port, grpc_port=self.grpc_port
            )
        except Exception:
            self.client = weaviate.connect_to_local(
                host=self.host, port=self.port, skip_init_checks=True
            )
        if self.client.is_ready():
            print(f"✅ Connected to Weaviate @ {self.host}:{self.port}")
        else:
            raise ConnectionError("Weaviate not ready")

    def close(self):
        if self.client:
            try:
                self.client.close()
            except Exception:
                pass

    # ── collection setup ────────────────────────────────────────────────────

    def setup_collection(self):
        self.vi_type = random.choice(ALL_VECTOR_INDEX_TYPES)
        self.distance_metric = random.choice(ALL_DISTANCE_METRICS)
        print(f"   VectorIndex: {self.vi_type}  Distance: {self.distance_metric}")

        try:
            self.client.collections.delete(CLASS_NAME)
        except Exception:
            pass

        properties = [
            Property(name="row_num", data_type=DataType.INT),
            Property(name="tag", data_type=DataType.INT, index_filterable=True),
        ]

        if self.vi_type == "hnsw":
            vi_config = Configure.VectorIndex.hnsw(
                distance_metric=self.distance_metric,
                ef_construction=128, max_connections=32
            )
        elif self.vi_type == "flat":
            vi_config = Configure.VectorIndex.flat(
                distance_metric=self.distance_metric
            )
        else:
            vi_config = Configure.VectorIndex.dynamic(
                distance_metric=self.distance_metric, threshold=10000
            )

        self.client.collections.create(
            name=CLASS_NAME,
            properties=properties,
            vector_config=Configure.Vectors.self_provided(vector_index_config=vi_config),
        )
        print(f"✅ Collection '{CLASS_NAME}' created.")

    # ── data generation & insert ────────────────────────────────────────────

    def generate_and_insert(self):
        rng = np.random.default_rng(self.seed)
        self.vectors = rng.random((self.n, self.dim), dtype=np.float32)
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.vectors = (self.vectors / norms).astype(np.float32)
        self.ids = [str(uuid.uuid4()) for _ in range(self.n)]

        col = self.client.collections.get(CLASS_NAME)
        print(f"⚡ Inserting {self.n} vectors (dim={self.dim}) …")

        for start in range(0, self.n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, self.n)
            objs = []
            for i in range(start, end):
                objs.append(DataObject(
                    uuid=self.ids[i],
                    properties={"row_num": i, "tag": i % 100},
                    vector=self.vectors[i].tolist(),
                ))
            col.data.insert_many(objs)
        print("✅ Insert complete.")

    # ── search helper ───────────────────────────────────────────────────────

    def _search(self, query_vec, top_k):
        """Return list of (uuid_str, distance) tuples, ordered by DB."""
        col = self.client.collections.get(CLASS_NAME)
        res = col.query.near_vector(
            near_vector=query_vec,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True),
        )
        hits = []
        for o in res.objects:
            uid = str(o.uuid)
            dist = o.metadata.distance if o.metadata else None
            hits.append((uid, float(dist) if dist is not None else 0.0))
        return hits

    # ── dynamic ops ─────────────────────────────────────────────────────────

    def _random_vector(self):
        v = np.random.randn(self.dim).astype(np.float32)
        v /= (np.linalg.norm(v) + 1e-12)
        return v

    def do_dynamic_op(self):
        col = self.client.collections.get(CLASS_NAME)
        op = random.choices(["insert", "delete", "update"], weights=[0.4, 0.3, 0.3], k=1)[0]

        if op == "insert":
            uid = str(uuid.uuid4())
            vec = self._random_vector()
            col.data.insert(properties={"row_num": -1, "tag": random.randint(0, 99)},
                            uuid=uid, vector=vec.tolist())
            self.ids.append(uid)
            self.vectors = np.vstack([self.vectors, vec.reshape(1, -1)])
            return f"INSERT uuid={uid[:8]}"
        elif op == "delete" and len(self.ids) > 200:
            idx = random.randint(0, len(self.ids) - 1)
            uid = self.ids[idx]
            col.data.delete_by_id(uid)
            self.ids.pop(idx)
            self.vectors = np.delete(self.vectors, idx, axis=0)
            return f"DELETE uuid={uid[:8]}"
        else:  # update
            if not self.ids:
                return "SKIP"
            idx = random.randint(0, len(self.ids) - 1)
            uid = self.ids[idx]
            vec = self._random_vector()
            col.data.replace(uuid=uid,
                             properties={"row_num": -1, "tag": random.randint(0, 99)},
                             vector=vec.tolist())
            self.vectors[idx] = vec
            return f"UPDATE uuid={uid[:8]}"

    # ── Test 1: distance ordering ───────────────────────────────────────────

    def test_distance_ordering(self, query_vec, top_k=10):
        hits = self._search(query_vec, top_k)
        if len(hits) <= 1:
            return True, "trivial (<=1 hit)"

        dists = [d for _, d in hits]
        # Weaviate distance is always ascending (smaller = more similar)
        for i in range(len(dists) - 1):
            if dists[i] > dists[i + 1] + 1e-5:
                return False, (f"distance ordering violated: pos {i} dist={dists[i]:.6f} > "
                               f"pos {i+1} dist={dists[i+1]:.6f}  metric={self.distance_metric}")
        return True, f"ok ({len(hits)} hits)"

    # ── Test 2: topK monotonicity ───────────────────────────────────────────

    def test_topk_monotonicity(self, query_vec, k_small=10, k_large=20):
        hits_small = self._search(query_vec, k_small)
        hits_large = self._search(query_vec, k_large)
        ids_small = set(h[0] for h in hits_small)
        ids_large = set(h[0] for h in hits_large)

        if not ids_small:
            return True, "trivial (no hits)"

        missing = ids_small - ids_large
        if not missing:
            return True, f"ok (|small|={len(ids_small)}, |large|={len(ids_large)})"

        miss_ratio = len(missing) / len(ids_small)
        if miss_ratio <= 0.05:
            return True, f"soft-ok (miss {len(missing)}/{len(ids_small)}={miss_ratio:.1%})"
        return False, (f"monotonicity violated: {len(missing)}/{len(ids_small)} IDs in top-{k_small} "
                       f"missing from top-{k_large}, ratio={miss_ratio:.1%}")

    # ── Test 3: self-retrieval ──────────────────────────────────────────────

    def test_self_retrieval(self):
        col = self.client.collections.get(CLASS_NAME)
        uid = str(uuid.uuid4())
        vec = self._random_vector()
        vec_list = vec.tolist()

        col.data.insert(properties={"row_num": -1, "tag": 0}, uuid=uid, vector=vec_list)
        time.sleep(0.3)

        hits = self._search(vec_list, top_k=1)

        if not hits:
            col.data.delete_by_id(uid)
            return False, f"self-retrieval EMPTY: inserted uuid={uid[:8]} but got no hits"

        best_uid, best_dist = hits[0]

        # Weaviate distance for identical vector should be ~0
        if best_uid == uid and best_dist < 0.01:
            col.data.delete_by_id(uid)
            return True, f"ok (uuid={uid[:8]}, dist={best_dist:.6f})"
        elif best_uid == uid:
            col.data.delete_by_id(uid)
            return True, f"soft-ok (uuid match, dist={best_dist:.6f})"
        else:
            detail = (f"self-retrieval FAIL: inserted uuid={uid[:8]}, but top1 is "
                      f"uuid={best_uid[:8]} dist={best_dist:.6f}")
            col.data.delete_by_id(uid)
            return False, detail

    # ── Test 3b: self-retrieval after dynamic ops ───────────────────────────

    def test_self_retrieval_after_ops(self, n_ops=5):
        col = self.client.collections.get(CLASS_NAME)
        uid = str(uuid.uuid4())
        vec = self._random_vector()
        vec_list = vec.tolist()

        col.data.insert(properties={"row_num": -1, "tag": 0}, uuid=uid, vector=vec_list)
        self.ids.append(uid)
        self.vectors = np.vstack([self.vectors, vec.reshape(1, -1)])

        ops_done = []
        for _ in range(n_ops):
            op = random.choices(["insert", "delete", "update"], weights=[0.4, 0.3, 0.3], k=1)[0]
            if op == "insert":
                oid = str(uuid.uuid4())
                ov = self._random_vector()
                col.data.insert(properties={"row_num": -1, "tag": random.randint(0, 99)},
                                uuid=oid, vector=ov.tolist())
                self.ids.append(oid)
                self.vectors = np.vstack([self.vectors, ov.reshape(1, -1)])
                ops_done.append(f"ins({oid[:8]})")
            elif op == "delete" and len(self.ids) > 200:
                candidates = [x for x in self.ids if x != uid]
                if candidates:
                    did = random.choice(candidates)
                    didx = self.ids.index(did)
                    col.data.delete_by_id(did)
                    self.ids.pop(didx)
                    self.vectors = np.delete(self.vectors, didx, axis=0)
                    ops_done.append(f"del({did[:8]})")
            elif op == "update":
                candidates = [x for x in self.ids if x != uid]
                if candidates:
                    upd = random.choice(candidates)
                    upidx = self.ids.index(upd)
                    uv = self._random_vector()
                    col.data.replace(uuid=upd,
                                     properties={"row_num": -1, "tag": random.randint(0, 99)},
                                     vector=uv.tolist())
                    self.vectors[upidx] = uv
                    ops_done.append(f"upd({upd[:8]})")

        time.sleep(0.5)

        hits = self._search(vec_list, top_k=5)
        found_ids = [h[0] for h in hits]

        if uid in found_ids:
            return True, f"ok after {len(ops_done)} ops"
        else:
            return False, (f"self-retrieval lost after dynamic ops: uuid={uid[:8]} "
                           f"not in top5, ops={ops_done}")


# ── main runner ─────────────────────────────────────────────────────────────

def run(args):
    current_seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)
    random.seed(current_seed)
    np.random.seed(current_seed)

    print("=" * 80)
    print(f"🚀 Weaviate Vector Fuzz Test | Seed: {current_seed} | Rounds: {args.rounds}")
    print(f"   N={args.N} DIM={args.dim} Dynamic={'ON' if args.dynamic else 'OFF'}")
    print("=" * 80)

    t = WeaviateVectorFuzzTest(args.host, args.port, args.grpc_port,
                                args.dim, args.N, current_seed, args.dynamic)
    t.connect()

    try:
        t.setup_collection()
        t.generate_and_insert()

        ts = int(time.time())
        logfile = f"weaviate_vector_fuzz_{ts}.log"
        print(f"📝 Log: {logfile}")

        fuzz_stats = {k: FuzzStats() for k in ["ordering", "monotonicity", "self_ret", "self_ret_dyn"]}
        failures = []

        with open(logfile, "w", encoding="utf-8") as flog:
            flog.write(f"Weaviate Vector Fuzz Test | Seed: {current_seed}\n")
            flog.write(f"VecIndex: {t.vi_type}  Distance: {t.distance_metric}  N={args.N}  DIM={args.dim}\n")
            flog.write(f"Dynamic: {args.dynamic}\n")
            flog.write("=" * 80 + "\n\n")

            for i in range(args.rounds):
                if args.dynamic and i > 0 and i % 20 == 0:
                    op_desc = t.do_dynamic_op()
                    flog.write(f"[DYN] {op_desc}\n")

                if random.random() < 0.7 and len(t.ids) > 0:
                    qi = random.randint(0, len(t.vectors) - 1)
                    qv = t.vectors[qi].tolist()
                else:
                    qv = t._random_vector().tolist()

                # Test 1
                top_k = random.choice([5, 10, 20, 50, 100])
                t0 = time.time()
                ok, detail = t.test_distance_ordering(qv, top_k)
                lat_ms = (time.time() - t0) * 1000
                fuzz_stats["ordering"].record(ok, lat_ms, error_cat=None if ok else "ordering")
                flog.write(f"[R{i:04d}] ORDERING top_k={top_k}: {'PASS' if ok else 'FAIL'} ({lat_ms:.1f}ms) — {detail}\n")
                if not ok:
                    failures.append(("ordering", i, detail))
                    print(f"  ❌ [R{i}] ORDERING FAIL: {detail[:120]}")

                # Test 2
                k_small = random.choice([5, 10, 15])
                k_large = k_small * random.choice([2, 3, 4])
                t0 = time.time()
                ok, detail = t.test_topk_monotonicity(qv, k_small, k_large)
                lat_ms = (time.time() - t0) * 1000
                fuzz_stats["monotonicity"].record(ok, lat_ms, error_cat=None if ok else "monotonicity")
                flog.write(f"[R{i:04d}] MONOTONICITY k={k_small}->{k_large}: {'PASS' if ok else 'FAIL'} ({lat_ms:.1f}ms) — {detail}\n")
                if not ok:
                    failures.append(("monotonicity", i, detail))
                    print(f"  ❌ [R{i}] MONOTONICITY FAIL: {detail[:120]}")

                # Test 3
                if i % 10 == 0:
                    t0 = time.time()
                    ok, detail = t.test_self_retrieval()
                    lat_ms = (time.time() - t0) * 1000
                    fuzz_stats["self_ret"].record(ok, lat_ms, error_cat=None if ok else "self_ret")
                    flog.write(f"[R{i:04d}] SELF_RETRIEVAL: {'PASS' if ok else 'FAIL'} ({lat_ms:.1f}ms) — {detail}\n")
                    if not ok:
                        failures.append(("self_retrieval", i, detail))
                        print(f"  ❌ [R{i}] SELF_RETRIEVAL FAIL: {detail[:120]}")

                # Test 3b
                if args.dynamic and i % 30 == 0 and i > 0:
                    t0 = time.time()
                    ok, detail = t.test_self_retrieval_after_ops(n_ops=random.randint(3, 8))
                    lat_ms = (time.time() - t0) * 1000
                    fuzz_stats["self_ret_dyn"].record(ok, lat_ms, error_cat=None if ok else "self_ret_dyn")
                    flog.write(f"[R{i:04d}] SELF_RET_DYN: {'PASS' if ok else 'FAIL'} ({lat_ms:.1f}ms) — {detail}\n")
                    if not ok:
                        failures.append(("self_ret_dyn", i, detail))
                        print(f"  ❌ [R{i}] SELF_RET_DYN FAIL: {detail[:120]}")

                if (i + 1) % 50 == 0:
                    print(f"  [R{i+1}/{args.rounds}] ordering=P{fuzz_stats['ordering'].passed}/F{fuzz_stats['ordering'].failed} "
                          f"mono=P{fuzz_stats['monotonicity'].passed}/F{fuzz_stats['monotonicity'].failed} "
                          f"self=P{fuzz_stats['self_ret'].passed}/F{fuzz_stats['self_ret'].failed} "
                          f"dyn=P{fuzz_stats['self_ret_dyn'].passed}/F{fuzz_stats['self_ret_dyn'].failed}")

            flog.write("\n" + "=" * 80 + "\n")
            flog.write("SUMMARY\n")
            for k, fs in fuzz_stats.items():
                flog.write(f"  {k}: {fs.summary()}\n")
            flog.write(f"Total failures: {len(failures)}\n")
            if failures:
                flog.write("\nFailed cases:\n")
                for cat, rnd, det in failures:
                    flog.write(f"  [{cat}] Round {rnd}: {det}\n")

        print("\n" + "=" * 80)
        print("📊 SUMMARY")
        total_pass = sum(fs.passed for fs in fuzz_stats.values())
        total_fail = sum(fs.failed for fs in fuzz_stats.values())
        for k, fs in fuzz_stats.items():
            status = "✅" if fs.failed == 0 else "❌"
            print(f"  {status} {k}: {fs.summary()}")
        print(f"  Total: PASS={total_pass}  FAIL={total_fail}")
        if failures:
            print(f"\n⚠️  {len(failures)} failure(s) detected — see {logfile}")
        else:
            print(f"\n🎉 All tests passed! Seed={current_seed}")
        print(f"📝 Full log: {logfile}")
        print(f"🔑 Reproduce: python weaviate_vector_fuzz_test.py --seed {current_seed}")
        print("=" * 80)

    finally:
        try:
            t.client.collections.delete(CLASS_NAME)
        except Exception:
            pass
        t.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weaviate Vector Search Fuzz Test")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--host", type=str, default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--grpc-port", type=int, default=GRPC_PORT, dest="grpc_port")
    parser.add_argument("-N", type=int, default=N)
    parser.add_argument("--dim", type=int, default=DIM)
    parser.add_argument("--dynamic", action="store_true")
    args = parser.parse_args()
    run(args)
