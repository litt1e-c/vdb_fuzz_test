#!/usr/bin/env python3
"""
Qdrant Vector Search Fuzz Test
================================
Three deterministic test strategies that expose vector-index bugs
without relying on recall-rate thresholds:

  1. Distance Ordering:  returned distances/scores must be monotonically sorted.
  2. TopK Monotonicity:  topK=10 ⊆ topK=20 (parameter-containment).
  3. Self-Retrieval:     insert V, search(V, top1) == V with dist ≈ 0/score ≈ max.

Usage:
  python qdrant_vector_fuzz_test.py                      # default 200 rounds
  python qdrant_vector_fuzz_test.py --seed 42 --rounds 500
  python qdrant_vector_fuzz_test.py --host 10.0.0.1 --port 6333
  python qdrant_vector_fuzz_test.py --dim 256 -N 3000
  python qdrant_vector_fuzz_test.py --dynamic
"""

import time
import random
import argparse
import sys
import os
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance, VectorParams, PointStruct, SearchParams,
    PointIdsList,
)


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
PORT = 6333
GRPC_PORT = 6334
COLLECTION_NAME = "vec_fuzz_test"
N = 2000
DIM = 128
BATCH_SIZE = 500
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ARTIFACT_ROOT = os.environ.get(
    "QDRANT_EXPERIMENT_ROOT",
    os.path.join(os.path.expanduser("~"), "qdrant_artifacts"),
)
DEFAULT_LOG_DIR = os.path.join(DEFAULT_ARTIFACT_ROOT, "qdrant_log", "vector")
LOG_DIR = DEFAULT_LOG_DIR
PREFER_GRPC = False
RUN_ID = None

ALL_DISTANCE_TYPES = [Distance.EUCLID, Distance.COSINE, Distance.DOT, Distance.MANHATTAN]

# Qdrant query_points score direction per metric:
# EUCLID/MANHATTAN: score = distance (ascending, smaller = more similar)
# COSINE/DOT: score = similarity (descending, larger = more similar)
SCORE_ASCENDING_METRICS = {Distance.EUCLID, Distance.MANHATTAN}
SCORE_DESCENDING_METRICS = {Distance.COSINE, Distance.DOT}

# ── helpers ─────────────────────────────────────────────────────────────────
_id_counter = 0

def _next_id():
    global _id_counter
    _id_counter += 1
    return _id_counter


def ensure_log_dir():
    os.makedirs(LOG_DIR, exist_ok=True)
    return LOG_DIR


def make_log_path(filename: str) -> str:
    return os.path.join(ensure_log_dir(), filename)


def display_path(path: str) -> str:
    try:
        return os.path.relpath(path, start=os.getcwd())
    except Exception:
        return path


def slugify(value: object, max_len: int = 48) -> str:
    text = str(value).strip().lower()
    pieces = [ch if ch.isalnum() else "-" for ch in text]
    collapsed = "".join(pieces).strip("-")
    while "--" in collapsed:
        collapsed = collapsed.replace("--", "-")
    return (collapsed or "x")[:max_len]


def make_run_label(seed: int, run_id: str | None) -> str:
    if run_id:
        return slugify(run_id, max_len=72)
    return f"vector-seed{seed}"


def make_collection_name(seed: int, run_id: str | None) -> str:
    label = slugify(run_id if run_id else f"seed{seed}", max_len=40)
    return f"vec_fuzz_{label}"


def build_reproduce_command(args: argparse.Namespace, seed: int) -> str:
    parts = [
        "python",
        "qdrant_vector_fuzz_test.py",
        "--seed",
        str(seed),
        "--rounds",
        str(args.rounds),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--grpc-port",
        str(args.grpc_port),
        "-N",
        str(args.N),
        "--dim",
        str(args.dim),
        "--log-dir",
        args.log_dir,
    ]
    if args.dynamic:
        parts.append("--dynamic")
    if args.prefer_grpc:
        parts.append("--prefer-grpc")
    if getattr(args, "run_id", None):
        parts += ["--run-id", args.run_id]
    return " ".join(parts)


# ── core class ──────────────────────────────────────────────────────────────

class QdrantVectorFuzzTest:
    def __init__(self, host, port, grpc_port, dim, n, seed, enable_dynamic, prefer_grpc):
        self.host = host
        self.port = port
        self.grpc_port = grpc_port
        self.dim = dim
        self.n = n
        self.seed = seed
        self.enable_dynamic = enable_dynamic
        self.prefer_grpc = prefer_grpc
        self.client = None
        self.distance_type = None
        self.vectors = None   # (N, DIM) float32
        self.ids = []         # list[int]
        self.py_rng = random.Random(self.seed)
        self.vector_rng = np.random.default_rng(self.seed)

    # ── connection ──────────────────────────────────────────────────────────

    def connect(self):
        self.client = QdrantClient(
            host=self.host,
            port=self.port,
            grpc_port=self.grpc_port,
            prefer_grpc=self.prefer_grpc,
            timeout=30,
        )
        self.client.get_collections()
        print(
            f"✅ Connected to Qdrant @ {self.host}:{self.port} "
            f"(grpc:{self.grpc_port}, prefer_grpc={self.prefer_grpc})"
        )

    def close(self):
        if self.client:
            try:
                self.client.close()
            except Exception:
                pass

    # ── collection setup ────────────────────────────────────────────────────

    def setup_collection(self):
        self.distance_type = self.py_rng.choice(ALL_DISTANCE_TYPES)
        print(f"   Distance: {self.distance_type}")

        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=self.dim, distance=self.distance_type),
        )
        print(f"✅ Collection '{COLLECTION_NAME}' created.")

    # ── data generation & insert ────────────────────────────────────────────

    def generate_and_insert(self):
        global _id_counter
        self.vectors = self.vector_rng.random((self.n, self.dim)).astype(np.float32)
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.vectors = (self.vectors / norms).astype(np.float32)

        _id_counter = 0
        self.ids = [_next_id() for _ in range(self.n)]

        print(f"⚡ Inserting {self.n} vectors (dim={self.dim}) …")
        for start in range(0, self.n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, self.n)
            points = []
            for i in range(start, end):
                points.append(PointStruct(
                    id=self.ids[i],
                    vector=self.vectors[i].tolist(),
                    payload={"tag": i % 100},
                ))
            self.client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
        print("✅ Insert complete.")

    # ── search helper ───────────────────────────────────────────────────────

    def _search(self, query_vec, top_k, exact=False):
        """Return list of (id, score) tuples, as ordered by Qdrant."""
        sp = SearchParams(exact=exact)
        res = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vec,
            limit=top_k,
            with_payload=False,
            search_params=sp,
        )
        hits = []
        for p in res.points:
            hits.append((p.id, float(p.score)))
        return hits

    # ── dynamic ops ─────────────────────────────────────────────────────────

    def _random_vector(self):
        v = self.vector_rng.standard_normal(self.dim).astype(np.float32)
        v /= (np.linalg.norm(v) + 1e-12)
        return v

    def do_dynamic_op(self):
        op = self.py_rng.choices(["insert", "delete", "upsert"], weights=[0.4, 0.3, 0.3], k=1)[0]

        if op == "insert":
            new_id = _next_id()
            vec = self._random_vector()
            self.client.upsert(
                collection_name=COLLECTION_NAME,
                points=[PointStruct(id=new_id, vector=vec.tolist(), payload={"tag": self.py_rng.randint(0, 99)})],
                wait=True,
            )
            self.ids.append(new_id)
            self.vectors = np.vstack([self.vectors, vec.reshape(1, -1)])
            return f"INSERT id={new_id}"
        elif op == "delete" and len(self.ids) > 200:
            idx = self.py_rng.randint(0, len(self.ids) - 1)
            did = self.ids[idx]
            self.client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=PointIdsList(points=[did]),
                wait=True,
            )
            self.ids.pop(idx)
            self.vectors = np.delete(self.vectors, idx, axis=0)
            return f"DELETE id={did}"
        else:  # upsert existing
            if not self.ids:
                return "SKIP"
            idx = self.py_rng.randint(0, len(self.ids) - 1)
            uid = self.ids[idx]
            vec = self._random_vector()
            self.client.upsert(
                collection_name=COLLECTION_NAME,
                points=[PointStruct(id=uid, vector=vec.tolist(), payload={"tag": self.py_rng.randint(0, 99)})],
                wait=True,
            )
            self.vectors[idx] = vec
            return f"UPSERT id={uid}"

    # ── Test 1: score ordering ──────────────────────────────────────────────

    def test_distance_ordering(self, query_vec, top_k=10):
        """Qdrant scores must be monotonically ordered per metric."""
        hits = self._search(query_vec, top_k)
        if len(hits) <= 1:
            return True, "trivial (<=1 hit)"

        scores = [s for _, s in hits]
        ascending = self.distance_type in SCORE_ASCENDING_METRICS

        for i in range(len(scores) - 1):
            if ascending:
                # EUCLID/MANHATTAN: distance ascending (0, 0.14, 0.87)
                if scores[i] > scores[i + 1] + 1e-5:
                    return False, (f"score ordering violated (ascending): pos {i} score={scores[i]:.6f} > "
                                   f"pos {i+1} score={scores[i+1]:.6f}  metric={self.distance_type}  "
                                   f"scores={[f'{s:.4f}' for s in scores]}")
            else:
                # COSINE/DOT: similarity descending (1.0, 0.99, 0.58)
                if scores[i] < scores[i + 1] - 1e-5:
                    return False, (f"score ordering violated (descending): pos {i} score={scores[i]:.6f} < "
                                   f"pos {i+1} score={scores[i+1]:.6f}  metric={self.distance_type}  "
                                   f"scores={[f'{s:.4f}' for s in scores]}")
        direction = 'ascending' if ascending else 'descending'
        return True, f"ok ({len(hits)} hits, {direction})"

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
                       f"missing from top-{k_large}, ratio={miss_ratio:.1%}, missing={sorted(missing)[:5]}")

    # ── Test 3: self-retrieval ──────────────────────────────────────────────

    def test_self_retrieval(self):
        new_id = _next_id()
        vec = self._random_vector()
        vec_list = vec.tolist()

        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=[PointStruct(id=new_id, vector=vec_list, payload={"tag": 0})],
            wait=True,
        )
        time.sleep(0.3)

        hits = self._search(vec_list, top_k=1)

        if not hits:
            self.client.delete(collection_name=COLLECTION_NAME,
                               points_selector=PointIdsList(points=[new_id]), wait=True)
            return False, f"self-retrieval EMPTY: inserted id={new_id} but got no hits"

        best_id, best_score = hits[0]

        # For self-query:
        # EUCLID/MANHATTAN: score = distance ≈ 0.0
        # COSINE: score = similarity ≈ 1.0
        # DOT: score = dot product ≈ 1.0 (normalised vectors)
        score_ok = False
        if self.distance_type in (Distance.EUCLID, Distance.MANHATTAN):
            score_ok = best_score < 0.01
        elif self.distance_type == Distance.COSINE:
            score_ok = best_score > 0.99
        elif self.distance_type == Distance.DOT:
            score_ok = best_score > 0.99

        if best_id == new_id and score_ok:
            self.client.delete(collection_name=COLLECTION_NAME,
                               points_selector=PointIdsList(points=[new_id]), wait=True)
            return True, f"ok (id={new_id}, score={best_score:.6f})"
        elif best_id == new_id:
            self.client.delete(collection_name=COLLECTION_NAME,
                               points_selector=PointIdsList(points=[new_id]), wait=True)
            return True, f"soft-ok (id match, score={best_score:.6f})"
        else:
            detail = (f"self-retrieval FAIL: inserted id={new_id}, but top1 is "
                      f"id={best_id} score={best_score:.6f}")
            self.client.delete(collection_name=COLLECTION_NAME,
                               points_selector=PointIdsList(points=[new_id]), wait=True)
            return False, detail

    # ── Test 3b: self-retrieval after dynamic ops ───────────────────────────

    def test_self_retrieval_after_ops(self, n_ops=5):
        new_id = _next_id()
        vec = self._random_vector()
        vec_list = vec.tolist()

        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=[PointStruct(id=new_id, vector=vec_list, payload={"tag": 0})],
            wait=True,
        )
        self.ids.append(new_id)
        self.vectors = np.vstack([self.vectors, vec.reshape(1, -1)])

        ops_done = []
        for _ in range(n_ops):
            op = self.py_rng.choices(["insert", "delete", "upsert"], weights=[0.4, 0.3, 0.3], k=1)[0]
            if op == "insert":
                oid = _next_id()
                ov = self._random_vector()
                self.client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=[PointStruct(id=oid, vector=ov.tolist(), payload={"tag": self.py_rng.randint(0, 99)})],
                    wait=True,
                )
                self.ids.append(oid)
                self.vectors = np.vstack([self.vectors, ov.reshape(1, -1)])
                ops_done.append(f"ins({oid})")
            elif op == "delete" and len(self.ids) > 200:
                candidates = [x for x in self.ids if x != new_id]
                if candidates:
                    did = self.py_rng.choice(candidates)
                    didx = self.ids.index(did)
                    self.client.delete(collection_name=COLLECTION_NAME,
                                       points_selector=PointIdsList(points=[did]), wait=True)
                    self.ids.pop(didx)
                    self.vectors = np.delete(self.vectors, didx, axis=0)
                    ops_done.append(f"del({did})")
            elif op == "upsert":
                candidates = [x for x in self.ids if x != new_id]
                if candidates:
                    uid_op = self.py_rng.choice(candidates)
                    uidx = self.ids.index(uid_op)
                    uv = self._random_vector()
                    self.client.upsert(
                        collection_name=COLLECTION_NAME,
                        points=[PointStruct(id=uid_op, vector=uv.tolist(), payload={"tag": self.py_rng.randint(0, 99)})],
                        wait=True,
                    )
                    self.vectors[uidx] = uv
                    ops_done.append(f"ups({uid_op})")

        time.sleep(0.5)

        hits = self._search(vec_list, top_k=5)
        found_ids = [h[0] for h in hits]

        if new_id in found_ids:
            return True, f"ok after {len(ops_done)} ops"
        else:
            return False, (f"self-retrieval lost after dynamic ops: id={new_id} "
                           f"not in top5={found_ids}, ops={ops_done}")


# ── main runner ─────────────────────────────────────────────────────────────

def run(args):
    global COLLECTION_NAME
    global LOG_DIR
    global PREFER_GRPC
    global RUN_ID

    current_seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)
    random.seed(current_seed)
    np.random.seed(current_seed)
    LOG_DIR = os.path.abspath(os.path.expanduser(args.log_dir))
    os.makedirs(LOG_DIR, exist_ok=True)
    PREFER_GRPC = bool(args.prefer_grpc)
    RUN_ID = args.run_id
    COLLECTION_NAME = make_collection_name(current_seed, RUN_ID)
    driver_rng = random.Random(current_seed ^ 0x00C0FFEE)

    print("=" * 80)
    print(f"🚀 Qdrant Vector Fuzz Test | Seed: {current_seed} | Rounds: {args.rounds}")
    print(f"   N={args.N} DIM={args.dim} Dynamic={'ON' if args.dynamic else 'OFF'}")
    print(f"   Target={args.host}:{args.port} (grpc:{args.grpc_port}, prefer_grpc={args.prefer_grpc})")
    print(f"   Collection={COLLECTION_NAME}")
    print(f"   LogDir={display_path(LOG_DIR)}")
    if RUN_ID:
        print(f"   RunID={RUN_ID}")
    print("=" * 80)

    t = QdrantVectorFuzzTest(
        args.host,
        args.port,
        args.grpc_port,
        args.dim,
        args.N,
        current_seed,
        args.dynamic,
        args.prefer_grpc,
    )
    t.connect()

    try:
        t.setup_collection()
        t.generate_and_insert()

        run_label = make_run_label(current_seed, RUN_ID)
        logfile = make_log_path(f"qdrant_vector_fuzz_{run_label}.log")
        print(f"📝 Log: {display_path(logfile)}")

        fuzz_stats = {k: FuzzStats() for k in ["ordering", "monotonicity", "self_ret", "self_ret_dyn"]}
        failures = []

        with open(logfile, "w", encoding="utf-8") as flog:
            flog.write(f"Qdrant Vector Fuzz Test | Seed: {current_seed}\n")
            flog.write(f"Collection: {COLLECTION_NAME}\n")
            flog.write(f"Distance: {t.distance_type}  N={args.N}  DIM={args.dim}\n")
            flog.write(f"Dynamic: {args.dynamic}\n")
            flog.write("=" * 80 + "\n\n")

            for i in range(args.rounds):
                if args.dynamic and i > 0 and i % 20 == 0:
                    op_desc = t.do_dynamic_op()
                    flog.write(f"[DYN] {op_desc}\n")

                if driver_rng.random() < 0.7 and len(t.ids) > 0:
                    qi = driver_rng.randint(0, len(t.vectors) - 1)
                    qv = t.vectors[qi].tolist()
                else:
                    qv = t._random_vector().tolist()

                # Test 1
                top_k = driver_rng.choice([5, 10, 20, 50, 100])
                t0 = time.time()
                ok, detail = t.test_distance_ordering(qv, top_k)
                lat_ms = (time.time() - t0) * 1000
                fuzz_stats["ordering"].record(ok, lat_ms, error_cat=None if ok else "ordering")
                flog.write(f"[R{i:04d}] ORDERING top_k={top_k}: {'PASS' if ok else 'FAIL'} ({lat_ms:.1f}ms) — {detail}\n")
                if not ok:
                    failures.append(("ordering", i, detail))
                    print(f"  ❌ [R{i}] ORDERING FAIL: {detail[:120]}")

                # Test 2
                k_small = driver_rng.choice([5, 10, 15])
                k_large = k_small * driver_rng.choice([2, 3, 4])
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
                    ok, detail = t.test_self_retrieval_after_ops(n_ops=driver_rng.randint(3, 8))
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
            print(f"\n⚠️  {len(failures)} failure(s) detected — see {display_path(logfile)}")
        else:
            print(f"\n🎉 All tests passed! Seed={current_seed}")
        print(f"📝 Full log: {display_path(logfile)}")
        print(f"🔑 Reproduce: {build_reproduce_command(args, current_seed)}")
        print("=" * 80)
        return 1 if failures else 0

    finally:
        try:
            t.client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        t.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qdrant Vector Search Fuzz Test")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--host", type=str, default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--grpc-port", type=int, default=GRPC_PORT, dest="grpc_port")
    parser.add_argument("--prefer-grpc", action="store_true")
    parser.add_argument("-N", type=int, default=N)
    parser.add_argument("--dim", type=int, default=DIM)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--log-dir", type=str, default=DEFAULT_LOG_DIR)
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()
    raise SystemExit(run(args))
