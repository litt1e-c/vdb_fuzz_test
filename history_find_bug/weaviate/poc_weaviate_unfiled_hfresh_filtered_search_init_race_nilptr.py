#!/usr/bin/env python3
"""
Minimal reproducer for a suspected Weaviate HFRESH filtered-search init race.

Requirements / assumptions:
  - server uses ASYNC_INDEXING=true
  - vector index is HFRESH
  - the query uses a filter, so HFRESH enters the small-allowlist flatSearch path

Observed behavior:
  right after insert_many returns, an immediate filtered near_vector query can fail with:
    panic occurred: runtime error: invalid memory address or nil pointer dereference

Control:
  if we wait briefly before issuing the same filtered query, it succeeds.

This matches the source-level hypothesis:
  SearchByVector() enters flatSearch() before checking whether quantizer/distancer
  have been initialized, and distToNode() dereferences h.distancer.
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
from weaviate.classes.query import Filter


HOST = "127.0.0.1"
PORT = 8080
GRPC_PORT = 50051
N = 1800
DIM = 96
SEED = 9813


def connect(host: str, port: int, grpc_port: int):
    try:
        return weaviate.connect_to_local(host=host, port=port, grpc_port=grpc_port)
    except Exception:
        return weaviate.connect_to_local(host=host, port=port, skip_init_checks=True)


def unit_vectors(seed: int, n: int, dim: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vecs / norms).astype(np.float32)


def stable_uuid(seed: int, i: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"hfresh-init-race::{seed}::{i}"))


def create_collection(client, name: str):
    client.collections.create(
        name=name,
        properties=[
            Property(name="bucket", data_type=DataType.INT, index_filterable=True),
            Property(name="flag", data_type=DataType.BOOL, index_filterable=True),
        ],
        vector_config=Configure.Vectors.self_provided(
            vector_index_config=Configure.VectorIndex.hfresh(
                distance_metric=VectorDistances.L2_SQUARED,
                max_posting_size_kb=256,
                search_probe=8,
            )
        ),
        inverted_index_config=Configure.inverted_index(index_null_state=True),
    )


def load_objects(collection, vectors: np.ndarray, seed: int):
    objects = [
        DataObject(
            uuid=stable_uuid(seed, i),
            properties={"bucket": i % 8, "flag": bool(i % 2)},
            vector=vectors[i].tolist(),
        )
        for i in range(len(vectors))
    ]
    result = collection.data.insert_many(objects)
    errors = getattr(result, "errors", None)
    if errors:
        raise RuntimeError(f"insert_many errors: {errors}")


def issue_filtered_query(collection, query_vec: list[float]):
    flt = Filter.by_property("bucket").equal(2) & Filter.by_property("flag").equal(False)
    return collection.query.near_vector(near_vector=query_vec, limit=10, filters=flt)


def main() -> int:
    parser = argparse.ArgumentParser(description="HFRESH filtered-search init-race nil-ptr reproducer")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--grpc-port", type=int, default=GRPC_PORT, dest="grpc_port")
    parser.add_argument("-N", type=int, default=N, dest="rows")
    parser.add_argument("--dim", type=int, default=DIM)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--control-wait-sec", type=float, default=1.5)
    args = parser.parse_args()

    vectors = unit_vectors(args.seed, args.rows, args.dim)
    query_vec = vectors[0].tolist()
    client = connect(args.host, args.port, args.grpc_port)
    immediate_name = f"HFreshInitRaceNow{int(time.time())}{uuid.uuid4().hex[:6]}"
    control_name = f"HFreshInitRaceWait{int(time.time())}{uuid.uuid4().hex[:6]}"

    print("=" * 80)
    print(
        f"HFRESH init-race nil-ptr POC | seed={args.seed} | N={args.rows} | dim={args.dim}"
    )
    print("=" * 80)

    try:
        create_collection(client, immediate_name)
        immediate = client.collections.get(immediate_name)
        load_objects(immediate, vectors, args.seed)
        try:
            issue_filtered_query(immediate, query_vec)
            print("IMMEDIATE QUERY: unexpectedly succeeded")
            immediate_bug = False
        except Exception as exc:
            message = str(exc)
            print(f"IMMEDIATE QUERY CRASH: {message}")
            immediate_bug = "nil pointer dereference" in message

        create_collection(client, control_name)
        control = client.collections.get(control_name)
        load_objects(control, vectors, args.seed ^ 0x1111)
        time.sleep(args.control_wait_sec)
        try:
            result = issue_filtered_query(control, query_vec)
            print(f"WAITED CONTROL OK: hits={len(result.objects)}")
            control_ok = True
        except Exception as exc:
            print(f"WAITED CONTROL FAILED: {exc}")
            control_ok = False

        if immediate_bug and control_ok:
            print("BUG CONFIRMED: immediate filtered flatSearch can dereference nil distancer before HFRESH init completes")
            return 2
        if not immediate_bug:
            print("NO BUG OBSERVED: immediate query did not trigger the nil-ptr window")
            return 1
        print("PARTIAL RESULT: immediate crash observed but control did not pass")
        return 3
    finally:
        try:
            if client.collections.exists(immediate_name):
                client.collections.delete(immediate_name)
            if client.collections.exists(control_name):
                client.collections.delete(control_name)
        finally:
            client.close()


if __name__ == "__main__":
    sys.exit(main())
