#!/usr/bin/env python3
"""
Probe a suspected Weaviate HFRESH stale-doc vector-search bug.

Observed symptom:
  after inserting a temporary vector, waiting until it is searchable, and then
  deleting it, later `near_vector` queries can fail with:

    no object found for doc id <N>: no object for doc id, it could have been deleted

This script turns that pattern into a small standalone reproducer with a control
index option (`hnsw`, `flat`, `dynamic`).
"""

import argparse
import os
import random
import sys
import time
import uuid

import numpy as np
import weaviate
from weaviate.classes.config import Configure, Property, DataType, VectorDistances
from weaviate.classes.data import DataObject
from weaviate.classes.query import MetadataQuery


HOST = "127.0.0.1"
PORT = 8080
GRPC_PORT = 50051
DIM = 64
BASE_N = 600
BATCH_SIZE = 200
LOG_DIR = "weaviate_log"


def ensure_log_dir():
    os.makedirs(LOG_DIR, exist_ok=True)
    return LOG_DIR


def make_log_path(prefix):
    return os.path.join(ensure_log_dir(), prefix)


def connect(host, port, grpc_port):
    try:
        return weaviate.connect_to_local(host=host, port=port, grpc_port=grpc_port)
    except Exception:
        return weaviate.connect_to_local(host=host, port=port, skip_init_checks=True)


def build_vector_index_config(index_name):
    if index_name == "hfresh":
        return Configure.VectorIndex.hfresh(
            distance_metric=VectorDistances.COSINE,
            max_posting_size_kb=256,
            search_probe=8,
        )
    if index_name == "hnsw":
        return Configure.VectorIndex.hnsw(
            distance_metric=VectorDistances.COSINE,
            ef_construction=128,
            max_connections=32,
        )
    if index_name == "flat":
        return Configure.VectorIndex.flat(
            distance_metric=VectorDistances.COSINE,
            vector_cache_max_objects=100000,
        )
    if index_name == "dynamic":
        return Configure.VectorIndex.dynamic(
            distance_metric=VectorDistances.COSINE,
            threshold=10000,
        )
    raise ValueError(f"Unsupported index: {index_name}")


def unit_random_vectors(rng, n, dim):
    vectors = rng.random((n, dim), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return (vectors / norms).astype(np.float32)


def random_unit_vector(rng, dim):
    vec = rng.standard_normal(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    return (vec / (norm if norm else 1)).astype(np.float32)


def search(col, query_vec, limit):
    result = col.query.near_vector(
        near_vector=query_vec,
        limit=limit,
        return_metadata=MetadataQuery(distance=True),
    )
    return [
        (str(obj.uuid), float(obj.metadata.distance if obj.metadata else 0.0))
        for obj in result.objects
    ]


def wait_until_visible(col, query_vec, target_id, timeout_sec):
    deadline = time.time() + timeout_sec
    polls = 0
    last_hits = []
    while time.time() < deadline:
        polls += 1
        last_hits = search(col, query_vec, 5)
        if any(hit_id == target_id for hit_id, _ in last_hits):
            return True, polls, last_hits
        time.sleep(0.2)
    return False, polls, last_hits


def main():
    parser = argparse.ArgumentParser(description="Probe suspected HFRESH stale-doc search crash")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--grpc-port", type=int, default=GRPC_PORT, dest="grpc_port")
    parser.add_argument("--index", choices=["hfresh", "hnsw", "flat", "dynamic"], default="hfresh")
    parser.add_argument("--seed", type=int, default=17003)
    parser.add_argument("--dim", type=int, default=DIM)
    parser.add_argument("-N", type=int, default=BASE_N, dest="base_n")
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--searches-per-cycle", type=int, default=40)
    parser.add_argument("--search-limit", type=int, default=50)
    parser.add_argument("--visible-timeout-sec", type=float, default=5.0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    collection_name = f"HfreshStaleProbe{int(time.time())}{uuid.uuid4().hex[:8]}"
    log_path = make_log_path(f"weaviate_hfresh_stale_doc_probe_{int(time.time())}.log")

    print("=" * 80)
    print(
        f"HFRESH Stale-Doc Probe | Index={args.index} | Seed={args.seed} | "
        f"N={args.base_n} | Dim={args.dim}"
    )
    print(f"Log: {log_path}")
    print("=" * 80)

    client = connect(args.host, args.port, args.grpc_port)
    if not client.is_ready():
        raise RuntimeError("Weaviate is not ready")

    crash_detail = None
    setup_issue = None
    crash_count = 0

    with open(log_path, "w", encoding="utf-8") as logf:
        def log(msg):
            print(msg)
            logf.write(msg + "\n")
            logf.flush()

        try:
            client.collections.create(
                name=collection_name,
                properties=[
                    Property(name="row_num", data_type=DataType.INT),
                    Property(name="tag", data_type=DataType.INT),
                ],
                vector_config=Configure.Vectors.self_provided(
                    vector_index_config=build_vector_index_config(args.index)
                ),
                inverted_index_config=Configure.inverted_index(
                    index_null_state=True,
                    index_property_length=True,
                    index_timestamps=True,
                ),
            )
            col = client.collections.get(collection_name)
            log(f"Collection created: {collection_name}")

            base_vectors = unit_random_vectors(rng, args.base_n, args.dim)
            base_ids = [str(uuid.uuid4()) for _ in range(args.base_n)]
            for start in range(0, args.base_n, BATCH_SIZE):
                end = min(start + BATCH_SIZE, args.base_n)
                objs = [
                    DataObject(
                        uuid=base_ids[i],
                        properties={"row_num": i, "tag": i % 100},
                        vector=base_vectors[i].tolist(),
                    )
                    for i in range(start, end)
                ]
                col.data.insert_many(objs)
            log(f"Inserted base objects: {args.base_n}")

            for cycle in range(args.cycles):
                temp_id = str(uuid.uuid4())
                temp_vec = random_unit_vector(rng, args.dim)
                col.data.insert(
                    properties={"row_num": -1, "tag": cycle},
                    uuid=temp_id,
                    vector=temp_vec.tolist(),
                )
                visible, polls, last_hits = wait_until_visible(
                    col, temp_vec.tolist(), temp_id, args.visible_timeout_sec
                )
                log(
                    f"[cycle {cycle}] inserted temp={temp_id[:8]} visible={visible} "
                    f"polls={polls} last_hits={[(hid[:8], round(dist, 6)) for hid, dist in last_hits[:3]]}"
                )
                if not visible:
                    setup_issue = (
                        f"temp object {temp_id} never became visible within "
                        f"{args.visible_timeout_sec:.1f}s"
                    )
                    log(f"SETUP-ISSUE: {setup_issue}")
                    break

                col.data.delete_by_id(temp_id)
                log(f"[cycle {cycle}] deleted temp={temp_id[:8]}")
                time.sleep(0.1)

                for attempt in range(args.searches_per_cycle):
                    query_vec = temp_vec if attempt % 2 == 0 else base_vectors[(cycle * args.searches_per_cycle + attempt) % args.base_n]
                    try:
                        hits = search(col, query_vec.tolist(), args.search_limit)
                        top1 = hits[0][0][:8] if hits else "-"
                        log(
                            f"[cycle {cycle}][search {attempt}] PASS hits={len(hits)} top1={top1}"
                        )
                    except Exception as exc:
                        crash_count += 1
                        crash_detail = str(exc)
                        log(f"[cycle {cycle}][search {attempt}] CRASH: {crash_detail}")
                        if "no object found for doc id" in crash_detail and "could have been deleted" in crash_detail:
                            log("CLASSIFICATION: suspected stale deleted doc id returned by vector index")
                        break
                if crash_count:
                    break

        finally:
            try:
                if client.collections.exists(collection_name):
                    client.collections.delete(collection_name)
                    log(f"Cleaned up collection: {collection_name}")
            finally:
                client.close()

    print("-" * 80)
    if crash_count:
        print("RESULT: REPRODUCED CRASH")
        print(crash_detail)
        return 1
    if setup_issue:
        print("RESULT: SETUP ISSUE")
        print(setup_issue)
        return 2
    print("RESULT: NO CRASH OBSERVED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
