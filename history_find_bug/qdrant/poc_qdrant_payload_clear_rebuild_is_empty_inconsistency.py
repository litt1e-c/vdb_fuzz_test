"""
Minimal reproduction candidate for a Qdrant payload/index inconsistency.

Observed in local Qdrant v1.17.0 coverage runs:
- A point whose payload field starts as explicit `null`
- then goes through `clear_payload`
- and then the payload index is deleted/recreated
may stop matching `is_empty` on one run, even though ordinary missing-field rows
still match.

This script keeps the scenario intentionally small:
1. Create a collection with a keyword payload index on `status`
2. Insert three points:
   - id=1: `status=None`
   - id=2: status missing
   - id=3: `status="live"`
3. Verify `status is_empty` returns `[1, 2]`
4. `clear_payload` on point 1, turning explicit null into missing
5. Verify `status is_empty` still returns `[1, 2]`
6. Delete and recreate the payload index
7. Re-run `is_empty(status)` through `scroll`, exact `count`, and exact
   `query_points`

Expected final result:
    scroll_ids == [1, 2]
    count == 2
    query_ids == [1, 2]

If point 1 disappears after the rebuild while point 2 still matches, that is a
strong signal that "explicit-null -> clear_payload -> missing" is interacting
incorrectly with payload-index rebuild.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    Filter,
    IsEmptyCondition,
    KeywordIndexParams,
    KeywordIndexType,
    PayloadField,
    PointStruct,
    SearchParams,
    VectorParams,
)


QUERY_VECTOR = [1.0, 0.0]


def build_client(args: argparse.Namespace, prefer_grpc: bool) -> QdrantClient:
    return QdrantClient(
        host=args.host,
        port=args.port,
        grpc_port=args.grpc_port,
        prefer_grpc=prefer_grpc,
        timeout=30,
    )


def collection_name(run_id: str, prefer_grpc: bool, trial: int) -> str:
    transport = "grpc" if prefer_grpc else "rest"
    return f"repro_payload_clear_rebuild_{run_id}_{transport}_{trial:02d}"


def ensure_artifact_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def empty_filter() -> Filter:
    return Filter(must=[IsEmptyCondition(is_empty=PayloadField(key="status"))])


def scroll_ids(client: QdrantClient, name: str) -> list[int]:
    points, _ = client.scroll(
        collection_name=name,
        scroll_filter=empty_filter(),
        limit=16,
        with_payload=False,
        with_vectors=False,
    )
    return sorted(int(point.id) for point in points)


def count_hits(client: QdrantClient, name: str) -> int:
    return int(
        client.count(
            collection_name=name,
            count_filter=empty_filter(),
            exact=True,
        ).count
    )


def query_ids_exact(client: QdrantClient, name: str) -> list[int]:
    response = client.query_points(
        collection_name=name,
        query=QUERY_VECTOR,
        query_filter=empty_filter(),
        limit=16,
        with_payload=False,
        with_vectors=False,
        search_params=SearchParams(exact=True),
    )
    return sorted(int(point.id) for point in response.points)


def capture(client: QdrantClient, name: str) -> dict[str, Any]:
    return {
        "scroll_ids": scroll_ids(client, name),
        "count": count_hits(client, name),
        "query_ids": query_ids_exact(client, name),
    }


def capture_matrix(clients: dict[str, QdrantClient], name: str) -> dict[str, dict[str, Any]]:
    return {transport: capture(client, name) for transport, client in clients.items()}


def retrieve_payloads(client: QdrantClient, name: str) -> dict[int, dict[str, Any]]:
    records = client.retrieve(
        collection_name=name,
        ids=[1, 2, 3],
        with_payload=True,
        with_vectors=False,
    )
    out: dict[int, dict[str, Any]] = {}
    for record in records:
        out[int(record.id)] = dict(record.payload or {})
    return out


def create_index(client: QdrantClient, name: str) -> None:
    client.create_payload_index(
        collection_name=name,
        field_name="status",
        field_schema=KeywordIndexParams(type=KeywordIndexType.KEYWORD, on_disk=True),
        wait=True,
    )


def rebuild_index(client: QdrantClient, name: str) -> None:
    client.delete_payload_index(
        collection_name=name,
        field_name="status",
        wait=True,
    )
    create_index(client, name)


def prepare_collection(client: QdrantClient, name: str) -> None:
    try:
        client.delete_collection(name)
    except Exception:
        pass

    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=2, distance=Distance.DOT),
    )
    create_index(client, name)
    client.upsert(
        collection_name=name,
        points=[
            PointStruct(id=1, vector=[1.0, 0.0], payload={"status": None}),
            PointStruct(id=2, vector=[0.0, 1.0], payload={}),
            PointStruct(id=3, vector=[1.0, 1.0], payload={"status": "live"}),
        ],
        wait=True,
    )


def write_trial_artifact(args: argparse.Namespace, trial_data: dict[str, Any]) -> None:
    artifact_root = ensure_artifact_dir(os.path.expanduser(args.artifact_dir))
    filename = (
        f"payload_clear_rebuild_{trial_data['update_transport']}_trial"
        f"{trial_data['trial']:02d}.json"
    )
    path = os.path.join(artifact_root, filename)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(trial_data, handle, indent=2, sort_keys=True)
    print(f"artifact={path}")


def run_trial(args: argparse.Namespace, prefer_grpc: bool, trial: int) -> bool:
    transport = "gRPC" if prefer_grpc else "REST"
    update_transport = "grpc" if prefer_grpc else "rest"
    client = build_client(args, prefer_grpc)
    read_clients = {
        "rest": build_client(args, prefer_grpc=False),
        "grpc": build_client(args, prefer_grpc=True),
    }
    name = collection_name(args.run_id, prefer_grpc, trial)
    expected = {"scroll_ids": [1, 2], "count": 2, "query_ids": [1, 2]}
    trial_data: dict[str, Any] = {
        "collection": name,
        "expected": expected,
        "trial": trial,
        "update_transport": update_transport,
    }

    try:
        prepare_collection(client, name)

        baseline = capture_matrix(read_clients, name)
        client.clear_payload(
            collection_name=name,
            points_selector=[1],
            wait=True,
        )
        after_clear = capture_matrix(read_clients, name)
        payloads_after_clear = retrieve_payloads(client, name)

        rebuild_index(client, name)
        after_rebuild = capture_matrix(read_clients, name)
        payloads_after_rebuild = retrieve_payloads(client, name)

        ok = (
            all(snapshot == expected for snapshot in baseline.values())
            and all(snapshot == expected for snapshot in after_clear.values())
            and all(snapshot == expected for snapshot in after_rebuild.values())
        )

        probe_snapshots: list[dict[str, Any]] = []
        if not ok:
            for delay in args.probe_delays:
                if delay > 0:
                    time.sleep(delay)
                probe_snapshots.append(
                    {
                        "delay_seconds": delay,
                        "captures": capture_matrix(read_clients, name),
                    }
                )

        trial_data.update(
            {
                "baseline": baseline,
                "after_clear": after_clear,
                "after_rebuild": after_rebuild,
                "payloads_after_clear": payloads_after_clear,
                "payloads_after_rebuild": payloads_after_rebuild,
                "post_failure_probes": probe_snapshots,
                "ok": ok,
            }
        )
        write_trial_artifact(args, trial_data)

        print(f"\n--- payload clear/rebuild repro ({transport}, trial={trial}) ---")
        print(f"collection={name}")
        print(f"baseline      = {json.dumps(baseline, sort_keys=True)}")
        print(f"after_clear   = {json.dumps(after_clear, sort_keys=True)}")
        print(f"after_rebuild = {json.dumps(after_rebuild, sort_keys=True)}")
        print(f"payloads_after_clear   = {json.dumps(payloads_after_clear, sort_keys=True)}")
        print(f"payloads_after_rebuild = {json.dumps(payloads_after_rebuild, sort_keys=True)}")
        if probe_snapshots:
            print(f"post_failure_probes = {json.dumps(probe_snapshots, sort_keys=True)}")
        print(f"expected      = {json.dumps(expected, sort_keys=True)}")
        print(f"result        = {'PASS' if ok else 'FAIL'}")
        return ok
    finally:
        if args.keep_failed_collections and not trial_data.get("ok", False):
            print(f"keeping_failed_collection={name}")
        else:
            try:
                client.delete_collection(name)
            except Exception:
                pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce payload clear + index rebuild is_empty inconsistency")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6333)
    parser.add_argument("--grpc-port", type=int, default=6334, dest="grpc_port")
    parser.add_argument("--trials", type=int, default=3, help="Trials per transport")
    parser.add_argument("--run-id", default="manual")
    parser.add_argument(
        "--artifact-dir",
        default="~/qdrant_artifacts/qdrant_log/repro_payload_clear_rebuild",
        help="Directory for per-trial JSON artifacts",
    )
    parser.add_argument(
        "--probe-delays",
        type=float,
        nargs="*",
        default=[0.0, 0.1, 0.5, 1.0],
        help="Extra readback delays after a failure to detect transient post-rebuild recovery",
    )
    parser.add_argument(
        "--keep-failed-collections",
        action="store_true",
        help="Keep failed collections for manual inspection",
    )
    args = parser.parse_args()

    rest_failures = 0
    grpc_failures = 0

    for trial in range(1, args.trials + 1):
        if not run_trial(args, prefer_grpc=False, trial=trial):
            rest_failures += 1

    for trial in range(1, args.trials + 1):
        if not run_trial(args, prefer_grpc=True, trial=trial):
            grpc_failures += 1

    print("\n=== Summary ===")
    print(f"REST failures: {rest_failures}/{args.trials}")
    print(f"gRPC failures: {grpc_failures}/{args.trials}")

    return 1 if (rest_failures or grpc_failures) else 0


if __name__ == "__main__":
    raise SystemExit(main())
