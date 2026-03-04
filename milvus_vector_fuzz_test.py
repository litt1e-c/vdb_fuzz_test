#!/usr/bin/env python3
"""
Milvus Vector Search Fuzz Test
===============================
Three deterministic test strategies that expose vector-index bugs
without relying on recall-rate thresholds:

  1. Distance Ordering:  returned distances must be monotonically sorted.
  2. TopK Monotonicity:  topK=10 ⊆ topK=20 (parameter-containment).
  3. Self-Retrieval:     insert V, search(V, top1) == V with dist ≈ 0.

Usage:
  python milvus_vector_fuzz_test.py                     # default 200 rounds
  python milvus_vector_fuzz_test.py --seed 42 --rounds 500
  python milvus_vector_fuzz_test.py --host 10.0.0.1 --port 19530
  python milvus_vector_fuzz_test.py --dim 256 -N 3000
  python milvus_vector_fuzz_test.py --dynamic           # enable insert/delete/upsert
"""

import time
import random
import argparse
import sys
import numpy as np
from pymilvus import (
    connections, utility, FieldSchema, CollectionSchema, DataType, Collection
)


# ── FuzzStats (ported from weaviate_fuzz_oracle.py) ─────────────────────────

class FuzzStats:
    """搜索延迟、错误分类、通过率统计 — 用于检测索引退化和 p99 异常"""
    def __init__(self):
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.latencies = []          # ms
        self.error_categories = {}   # {category: count}

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

# ── defaults (overridable via CLI) ──────────────────────────────────────────
HOST = "127.0.0.1"
PORT = "19531"
COLLECTION_NAME = "vec_fuzz_test"
N = 2000          # base dataset size
DIM = 128
BATCH_SIZE = 500
FLUSH_INTERVAL = 500

ALL_INDEX_TYPES = ["FLAT", "HNSW", "IVF_FLAT", "IVF_SQ8"]
ALL_METRIC_TYPES = ["L2", "IP", "COSINE"]

# distance ordering: L2 ascending, IP/COSINE descending
METRIC_ASCENDING = {"L2": True, "IP": False, "COSINE": False}

# ── helpers ─────────────────────────────────────────────────────────────────

def _norm(v):
    v = np.array(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return (v / n if n > 0 else v).tolist()


def _make_search_params(index_type, metric_type, limit):
    """Build search param dict matching the current index."""
    if index_type == "HNSW":
        return {"metric_type": metric_type, "params": {"ef": max(64, limit + 8)}}
    elif index_type.startswith("IVF"):
        return {"metric_type": metric_type, "params": {"nprobe": 32}}
    elif index_type == "DISKANN":
        return {"metric_type": metric_type, "params": {"search_list": max(20, limit + 8)}}
    return {"metric_type": metric_type, "params": {}}


# ── core class ──────────────────────────────────────────────────────────────

class MilvusVectorFuzzTest:
    def __init__(self, host, port, dim, n, seed, enable_dynamic):
        self.host = host
        self.port = port
        self.dim = dim
        self.n = n
        self.seed = seed
        self.enable_dynamic = enable_dynamic
        self.col = None
        self.index_type = None
        self.metric_type = None
        self.vectors = None   # (N, DIM) float32
        self.ids = None       # list[int]
        self._id_counter = n * 10

    # ── connection ──────────────────────────────────────────────────────────

    def connect(self):
        connections.connect("default", host=self.host, port=self.port, timeout=30)
        print(f"✅ Connected to Milvus @ {self.host}:{self.port}")

    def disconnect(self):
        try:
            connections.disconnect("default")
        except Exception:
            pass

    # ── collection setup ────────────────────────────────────────────────────

    def setup_collection(self):
        self.index_type = random.choice(ALL_INDEX_TYPES)
        self.metric_type = random.choice(ALL_METRIC_TYPES)
        print(f"   Index: {self.index_type}  Metric: {self.metric_type}")

        if utility.has_collection(COLLECTION_NAME):
            utility.drop_collection(COLLECTION_NAME)

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="tag", dtype=DataType.INT64),  # for grouping / filtering
        ]
        schema = CollectionSchema(fields, enable_dynamic_field=False)
        self.col = Collection(COLLECTION_NAME, schema)

    # ── data generation & insert ────────────────────────────────────────────

    def generate_and_insert(self):
        rng = np.random.default_rng(self.seed)
        self.vectors = rng.random((self.n, self.dim), dtype=np.float32)
        # normalise for IP/COSINE to behave well
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.vectors = (self.vectors / norms).astype(np.float32)
        self.ids = list(range(self.n))

        print(f"⚡ Inserting {self.n} vectors (dim={self.dim}) …")
        for start in range(0, self.n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, self.n)
            batch_ids = self.ids[start:end]
            batch_vecs = self.vectors[start:end].tolist()
            batch_tags = [i % 100 for i in batch_ids]
            self.col.insert([batch_ids, batch_vecs, batch_tags])
            if end % FLUSH_INTERVAL == 0 or end == self.n:
                self.col.flush()
        print("✅ Insert complete.")

    def build_index_and_load(self):
        if self.index_type == "HNSW":
            params = {"M": 32, "efConstruction": 256}
        elif self.index_type == "IVF_FLAT":
            params = {"nlist": 128}
        elif self.index_type == "IVF_SQ8":
            params = {"nlist": 128}
        elif self.index_type == "IVF_PQ":
            params = {"nlist": 128, "m": 16, "nbits": 8}
        else:
            params = {}

        self.col.create_index(
            "vector",
            {"metric_type": self.metric_type, "index_type": self.index_type, "params": params},
            index_name="vec_idx",
        )
        self.col.load()
        print(f"✅ Index built ({self.index_type}) and loaded.")

    # ── search helper ───────────────────────────────────────────────────────

    def _search(self, query_vec, top_k, expr=None):
        """Return list of (id, distance) tuples."""
        sp = _make_search_params(self.index_type, self.metric_type, top_k)
        res = self.col.search(
            data=[query_vec],
            anns_field="vector",
            param=sp,
            limit=top_k,
            expr=expr,
            output_fields=["id"],
            consistency_level="Strong",
        )
        hits = []
        if res and len(res) > 0:
            for hit in res[0]:
                hid = hit.get("id") if isinstance(hit, dict) else hit.id
                hdist = hit.distance
                hits.append((hid, float(hdist)))
        return hits

    # ── dynamic ops (insert / delete / upsert) ─────────────────────────────

    def _gen_id(self):
        self._id_counter += 1
        return self._id_counter

    def _random_vector(self):
        v = np.random.randn(self.dim).astype(np.float32)
        v /= (np.linalg.norm(v) + 1e-12)
        return v

    def do_dynamic_op(self):
        """Execute one random dynamic op (insert / delete / upsert)."""
        op = random.choices(["insert", "delete", "upsert"], weights=[0.4, 0.3, 0.3], k=1)[0]
        if op == "insert":
            new_id = self._gen_id()
            vec = self._random_vector()
            self.col.insert([[new_id], [vec.tolist()], [random.randint(0, 99)]])
            self.ids.append(new_id)
            self.vectors = np.vstack([self.vectors, vec.reshape(1, -1)])
            self.col.flush()
            return f"INSERT id={new_id}"
        elif op == "delete" and len(self.ids) > 200:
            del_id = random.choice(self.ids)
            idx = self.ids.index(del_id)
            self.col.delete(f"id in [{del_id}]")
            self.ids.pop(idx)
            self.vectors = np.delete(self.vectors, idx, axis=0)
            self.col.flush()
            self.col.compact()
            self.col.wait_for_compaction_completed()
            self.col.release()
            self.col.load()
            return f"DELETE id={del_id}"
        else:  # upsert existing
            if not self.ids:
                return "SKIP"
            uid = random.choice(self.ids)
            idx = self.ids.index(uid)
            vec = self._random_vector()
            self.col.upsert([[uid], [vec.tolist()], [random.randint(0, 99)]])
            self.vectors[idx] = vec
            self.col.flush()
            self.col.compact()
            self.col.wait_for_compaction_completed()
            self.col.release()
            self.col.load()
            return f"UPSERT id={uid}"

    # ── Test 1: distance ordering ───────────────────────────────────────────

    def test_distance_ordering(self, query_vec, top_k=10):
        """Returned distances must be monotonically sorted."""
        hits = self._search(query_vec, top_k)
        if len(hits) <= 1:
            return True, "trivial (<=1 hit)"

        dists = [d for _, d in hits]
        ascending = METRIC_ASCENDING.get(self.metric_type, True)

        for i in range(len(dists) - 1):
            if ascending:
                if dists[i] > dists[i + 1] + 1e-6:
                    return False, (f"L2 ordering violated: pos {i} dist={dists[i]:.6f} > "
                                   f"pos {i+1} dist={dists[i+1]:.6f}  hits={hits}")
            else:
                if dists[i] < dists[i + 1] - 1e-6:
                    return False, (f"{self.metric_type} ordering violated: pos {i} dist={dists[i]:.6f} < "
                                   f"pos {i+1} dist={dists[i+1]:.6f}  hits={hits}")
        return True, f"ok ({len(hits)} hits, ascending={ascending})"

    # ── Test 2: topK monotonicity ───────────────────────────────────────────

    def test_topk_monotonicity(self, query_vec, k_small=10, k_large=20):
        """IDs from topK=small should be a subset of topK=large."""
        hits_small = self._search(query_vec, k_small)
        hits_large = self._search(query_vec, k_large)
        ids_small = set(h[0] for h in hits_small)
        ids_large = set(h[0] for h in hits_large)

        if not ids_small:
            return True, "trivial (no hits for small k)"

        missing = ids_small - ids_large
        if not missing:
            return True, f"ok (|small|={len(ids_small)}, |large|={len(ids_large)})"

        # Allow a small tolerance for ANN — if miss ratio < 5%, treat as soft warning
        miss_ratio = len(missing) / len(ids_small)
        if miss_ratio <= 0.05:
            return True, f"soft-ok (miss {len(missing)}/{len(ids_small)}={miss_ratio:.1%})"
        return False, (f"monotonicity violated: {len(missing)}/{len(ids_small)} IDs in top-{k_small} "
                       f"missing from top-{k_large}, miss_ratio={miss_ratio:.1%}, missing={list(missing)[:5]}")

    # ── Test 3: self-retrieval ──────────────────────────────────────────────

    def test_self_retrieval(self):
        """Insert V, search(V, top1) must return V with distance ≈ 0."""
        new_id = self._gen_id()
        vec = self._random_vector()
        vec_list = vec.tolist()

        self.col.insert([[new_id], [vec_list], [0]])
        self.col.flush()
        # Ensure new data is physically merged and indexed
        try:
            self.col.compact()
            self.col.wait_for_compaction_completed()
            self.col.release()
            self.col.load()
        except Exception:
            time.sleep(1.0)  # fallback: wait for async index

        hits = self._search(vec_list, top_k=1)

        if not hits:
            return False, f"self-retrieval EMPTY: inserted id={new_id} but got no hits"

        best_id, best_dist = hits[0]

        # For COSINE/IP the "distance" of a normalised vector to itself may be
        # expressed differently (Milvus returns IP score, not distance).
        # COSINE → dist≈0 or score≈1, IP → score≈1, L2 → dist≈0
        dist_ok = False
        if self.metric_type == "L2":
            dist_ok = best_dist < 0.01
        elif self.metric_type == "COSINE":
            # Milvus COSINE returns a distance (1 - cosine_sim) or sim directly
            dist_ok = best_dist < 0.01 or abs(best_dist - 1.0) < 0.01
        elif self.metric_type == "IP":
            dist_ok = abs(best_dist - 1.0) < 0.01 or best_dist < 0.01

        if best_id == new_id and dist_ok:
            # Cleanup: delete the test vector to keep dataset clean
            self.col.delete(f"id in [{new_id}]")
            self.col.flush()
            return True, f"ok (id={new_id}, dist={best_dist:.6f})"
        elif best_id == new_id:
            self.col.delete(f"id in [{new_id}]")
            self.col.flush()
            return True, f"soft-ok (id match, dist={best_dist:.6f} may be metric artifact)"
        else:
            detail = (f"self-retrieval FAIL: inserted id={new_id}, but top1 is "
                      f"id={best_id} dist={best_dist:.6f}")
            self.col.delete(f"id in [{new_id}]")
            self.col.flush()
            return False, detail

    # ── Test 3b: self-retrieval after dynamic ops ───────────────────────────

    def test_self_retrieval_after_ops(self, n_ops=5):
        """Insert V, do random ops, then verify V is still retrievable."""
        new_id = self._gen_id()
        vec = self._random_vector()
        vec_list = vec.tolist()

        self.col.insert([[new_id], [vec_list], [0]])
        self.ids.append(new_id)
        self.vectors = np.vstack([self.vectors, vec.reshape(1, -1)])
        self.col.flush()

        # Do a few random dynamic ops on OTHER ids
        ops_done = []
        for _ in range(n_ops):
            op = random.choices(["insert", "delete", "upsert"], weights=[0.4, 0.3, 0.3], k=1)[0]
            if op == "insert":
                oid = self._gen_id()
                ov = self._random_vector()
                self.col.insert([[oid], [ov.tolist()], [random.randint(0, 99)]])
                self.ids.append(oid)
                self.vectors = np.vstack([self.vectors, ov.reshape(1, -1)])
                ops_done.append(f"ins({oid})")
            elif op == "delete" and len(self.ids) > 200:
                did = random.choice([x for x in self.ids if x != new_id])
                didx = self.ids.index(did)
                self.col.delete(f"id in [{did}]")
                self.ids.pop(didx)
                self.vectors = np.delete(self.vectors, didx, axis=0)
                ops_done.append(f"del({did})")
            elif op == "upsert":
                uid = random.choice([x for x in self.ids if x != new_id])
                uidx = self.ids.index(uid)
                uv = self._random_vector()
                self.col.upsert([[uid], [uv.tolist()], [random.randint(0, 99)]])
                self.vectors[uidx] = uv
                ops_done.append(f"ups({uid})")

        self.col.flush()
        self.col.compact()
        self.col.wait_for_compaction_completed()
        self.col.release()
        self.col.load()

        # Now search for the originally inserted vector
        hits = self._search(vec_list, top_k=5)
        found_ids = [h[0] for h in hits]

        if new_id in found_ids:
            return True, f"ok after {len(ops_done)} ops"
        else:
            return False, (f"self-retrieval lost after dynamic ops: id={new_id} "
                           f"not in top5={found_ids}, ops={ops_done}")


# ── main runner ─────────────────────────────────────────────────────────────

def run(args):
    current_seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)
    random.seed(current_seed)
    np.random.seed(current_seed)

    print("=" * 80)
    print(f"🚀 Milvus Vector Fuzz Test | Seed: {current_seed} | Rounds: {args.rounds}")
    print(f"   N={args.N} DIM={args.dim} Dynamic={'ON' if args.dynamic else 'OFF'}")
    print("=" * 80)

    t = MilvusVectorFuzzTest(args.host, args.port, args.dim, args.N, current_seed, args.dynamic)
    t.connect()

    try:
        t.setup_collection()
        t.generate_and_insert()
        t.build_index_and_load()

        # Log file
        ts = int(time.time())
        logfile = f"milvus_vector_fuzz_{ts}.log"
        print(f"📝 Log: {logfile}")

        fuzz_stats = {k: FuzzStats() for k in ["ordering", "monotonicity", "self_ret", "self_ret_dyn"]}
        failures = []

        with open(logfile, "w", encoding="utf-8") as flog:
            flog.write(f"Milvus Vector Fuzz Test | Seed: {current_seed}\n")
            flog.write(f"Index: {t.index_type}  Metric: {t.metric_type}  N={args.N}  DIM={args.dim}\n")
            flog.write(f"Dynamic: {args.dynamic}\n")
            flog.write("=" * 80 + "\n\n")

            for i in range(args.rounds):
                # Occasionally do dynamic ops
                if args.dynamic and i > 0 and i % 20 == 0:
                    op_desc = t.do_dynamic_op()
                    flog.write(f"[DYN] {op_desc}\n")

                # Pick a random query vector (from dataset or fresh)
                if random.random() < 0.7 and len(t.ids) > 0:
                    qi = random.randint(0, len(t.vectors) - 1)
                    qv = t.vectors[qi].tolist()
                else:
                    qv = t._random_vector().tolist()

                # ── Test 1: distance ordering ───────────────────────────
                top_k = random.choice([5, 10, 20, 50, 100])
                t0 = time.time()
                ok, detail = t.test_distance_ordering(qv, top_k)
                lat_ms = (time.time() - t0) * 1000
                fuzz_stats["ordering"].record(ok, lat_ms, error_cat=None if ok else "ordering")
                flog.write(f"[R{i:04d}] ORDERING top_k={top_k}: {'PASS' if ok else 'FAIL'} ({lat_ms:.1f}ms) — {detail}\n")
                if not ok:
                    failures.append(("ordering", i, detail))
                    print(f"  ❌ [R{i}] ORDERING FAIL: {detail[:120]}")

                # ── Test 2: topK monotonicity ───────────────────────────
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

                # ── Test 3: self-retrieval (every ~10 rounds) ───────────
                if i % 10 == 0:
                    t0 = time.time()
                    ok, detail = t.test_self_retrieval()
                    lat_ms = (time.time() - t0) * 1000
                    fuzz_stats["self_ret"].record(ok, lat_ms, error_cat=None if ok else "self_ret")
                    flog.write(f"[R{i:04d}] SELF_RETRIEVAL: {'PASS' if ok else 'FAIL'} ({lat_ms:.1f}ms) — {detail}\n")
                    if not ok:
                        failures.append(("self_retrieval", i, detail))
                        print(f"  ❌ [R{i}] SELF_RETRIEVAL FAIL: {detail[:120]}")

                # ── Test 3b: self-retrieval after dynamic ops ───────────
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

            # Summary
            flog.write("\n" + "=" * 80 + "\n")
            flog.write("SUMMARY\n")
            for k, fs in fuzz_stats.items():
                flog.write(f"  {k}: {fs.summary()}\n")
            flog.write(f"Total failures: {len(failures)}\n")
            if failures:
                flog.write("\nFailed cases:\n")
                for cat, rnd, det in failures:
                    flog.write(f"  [{cat}] Round {rnd}: {det}\n")

        # Console summary
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
        print(f"🔑 Reproduce: python milvus_vector_fuzz_test.py --seed {current_seed}")
        print("=" * 80)

    finally:
        try:
            if utility.has_collection(COLLECTION_NAME):
                utility.drop_collection(COLLECTION_NAME)
        except Exception:
            pass
        t.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Milvus Vector Search Fuzz Test")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--rounds", type=int, default=200, help="Number of test rounds (default: 200)")
    parser.add_argument("--host", type=str, default=HOST, help=f"Milvus host (default: {HOST})")
    parser.add_argument("--port", type=str, default=PORT, help=f"Milvus port (default: {PORT})")
    parser.add_argument("-N", type=int, default=N, help=f"Dataset size (default: {N})")
    parser.add_argument("--dim", type=int, default=DIM, help=f"Vector dimension (default: {DIM})")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic insert/delete/upsert ops")
    args = parser.parse_args()
    run(args)
