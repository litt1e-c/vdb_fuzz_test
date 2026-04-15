#!/usr/bin/env python3
"""
Minimal reproducer for a suspected Weaviate HFRESH stale-deleted-doc bug.

Observed behavior:
  1. insert a temporary object with a self-provided vector
  2. wait until a near_vector search can retrieve it
  3. delete that object
  4. immediately search again with the same vector

Expected:
  the deleted object should disappear from results and the query should still succeed.

Observed on HFRESH:
  near_vector can fail with:
    no object found for doc id <N>: no object for doc id, it could have been deleted

Controls:
  the same script can be run with --index hnsw / flat / dynamic; these should not crash.
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid

import numpy as np
import weaviate
from weaviate.classes.config import Configure, DataType, Property, VectorDistances
from weaviate.classes.data import DataObject
from weaviate.classes.query import MetadataQuery


HOST = "127.0.0.1"
PORT = 8080
GRPC_PORT = 50051
DIM = 64
BASE_N = 600
BATCH_SIZE = 200


def connect(host: str, port: int, grpc_port: int):
    try:
        return weaviate.connect_to_local(host=host, port=port, grpc_port=grpc_port)
    except Exception:
        return weaviate.connect_to_local(host=host, port=port, skip_init_checks=True)


def vector_index_config(index_name: str):
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
    raise ValueError(f"unsupported index: {index_name}")


def unit_vectors(seed: int, n: int, dim: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vectors = rng.random((n, dim), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)


def one_unit_vector(seed: int, dim: int) -> np.ndarray:
    rng = np.random.default_rng(seed ^ 0xBAD5EED)
    vec = rng.standard_normal(dim).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    return (vec / (norm if norm else 1.0)).astype(np.float32)


def object_uuid(seed: int, label: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"weaviate-hfresh-stale-doc::{seed}::{label}"))


def search_ids(collection, vector: list[float], limit: int) -> list[tuple[str, float]]:
    result = collection.query.near_vector(
        near_vector=vector,
        limit=limit,
        return_metadata=MetadataQuery(distance=True),
    )
    return [
        (str(obj.uuid), float(obj.metadata.distance if obj.metadata else 0.0))
        for obj in result.objects
    ]


def wait_until_visible(collection, vector: list[float], target_uuid: str, timeout_sec: float) -> tuple[bool, int, list[tuple[str, float]]]:
    deadline = time.time() + timeout_sec
    polls = 0
    hits: list[tuple[str, float]] = []
    while time.time() < deadline:
        polls += 1
        hits = search_ids(collection, vector, 10)
        if any(hit_uuid == target_uuid for hit_uuid, _ in hits):
            return True, polls, hits
        time.sleep(0.2)
    return False, polls, hits


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal HFRESH stale deleted doc reproducer")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--grpc-port", type=int, default=GRPC_PORT, dest="grpc_port")
    parser.add_argument("--index", choices=["hfresh", "hnsw", "flat", "dynamic"], default="hfresh")
    parser.add_argument("--seed", type=int, default=9813)
    parser.add_argument("--dim", type=int, default=DIM)
    parser.add_argument("-N", type=int, default=BASE_N, dest="base_n")
    parser.add_argument("--search-limit", type=int, default=50)
    parser.add_argument("--visible-timeout-sec", type=float, default=5.0)
    args = parser.parse_args()

    collection_name = f"HFreshDeletedDocPOC{int(time.time())}{uuid.uuid4().hex[:8]}"
    base_vectors = unit_vectors(args.seed, args.base_n, args.dim)
    temp_vector = one_unit_vector(args.seed, args.dim)
    temp_uuid = object_uuid(args.seed, "temp")

    print("=" * 80)
    print(
        f"HFRESH deleted-doc POC | index={args.index} | seed={args.seed} | "
        f"N={args.base_n} | dim={args.dim}"
    )
    print("=" * 80)

    client = connect(args.host, args.port, args.grpc_port)
    if not client.is_ready():
        raise RuntimeError("weaviate is not ready")

    try:
        client.collections.create(
            name=collection_name,
            properties=[
                Property(name="row_num", data_type=DataType.INT),
                Property(name="tag", data_type=DataType.TEXT),
            ],
            vector_config=Configure.Vectors.self_provided(
                vector_index_config=vector_index_config(args.index)
            ),
            inverted_index_config=Configure.inverted_index(index_null_state=True),
        )
        collection = client.collections.get(collection_name)

        for start in range(0, args.base_n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, args.base_n)
            objects = [
                DataObject(
                    uuid=object_uuid(args.seed, f"base-{i}"),
                    properties={"row_num": i, "tag": f"base_{i}"},
                    vector=base_vectors[i].tolist(),
                )
                for i in range(start, end)
            ]
            collection.data.insert_many(objects)
        print(f"Inserted base objects: {args.base_n}")

        collection.data.insert(
            uuid=temp_uuid,
            properties={"row_num": -1, "tag": "temp"},
            vector=temp_vector.tolist(),
        )
        visible, polls, hits = wait_until_visible(
            collection,
            temp_vector.tolist(),
            temp_uuid,
            args.visible_timeout_sec,
        )
        print(
            f"Temp visible: {visible} polls={polls} "
            f"top3={[(hit_uuid[:8], round(dist, 6)) for hit_uuid, dist in hits[:3]]}"
        )
        if not visible:
            print("SETUP ISSUE: temp object never became searchable")
            return 2

        collection.data.delete_by_id(temp_uuid)
        print(f"Deleted temp object: {temp_uuid}")

        try:
            post_delete_hits = search_ids(collection, temp_vector.tolist(), args.search_limit)
            top1 = post_delete_hits[0][0] if post_delete_hits else "-"
            print(f"POST-DELETE SEARCH OK: hits={len(post_delete_hits)} top1={top1}")
            if args.index == "hfresh":
                print("NO BUG OBSERVED: HFRESH did not reproduce this time")
                return 1
            print("CONTROL OK: non-HFRESH index survived post-delete search")
            return 0
        except Exception as exc:
            message = str(exc)
            print(f"POST-DELETE SEARCH CRASH: {message}")
            if args.index == "hfresh" and "no object found for doc id" in message:
                print("BUG CONFIRMED: deleted internal doc id leaks into HFRESH search results")
                return 2
            print("UNEXPECTED CRASH")
            return 3
    finally:
        try:
            if client.collections.exists(collection_name):
                client.collections.delete(collection_name)
        finally:
            client.close()


if __name__ == "__main__":
    sys.exit(main())
