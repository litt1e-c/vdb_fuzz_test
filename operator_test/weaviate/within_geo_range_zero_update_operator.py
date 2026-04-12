import math
import random
import time
import uuid

import weaviate
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import Filter, GeoCoordinate


HOST = "127.0.0.1"
PORT = 8080
GRPC_PORT = 50051
COLLECTION_PREFIX = "WithinGeoZeroUpdate"
VECTOR = [0.0, 0.0, 0.0, 0.0]
EARTH_RADIUS_M = 6_371_000.0


BASE_COORDS = [
    (0.0, 0.0),
    (-0.0, -0.0),
    (0.0, 179.9999),
    (0.0, -179.9999),
    (89.9999, 0.0),
    (-89.9999, 0.0),
    (45.0, 45.0),
    (-45.0, -45.0),
    (0.123456789, -0.987654321),
]


MOVE_COORDS = [
    (0.01, 0.0),
    (0.0, 0.01),
    (0.0, -179.9998),
    (0.0, 179.9998),
    (89.9998, 0.5),
    (-89.9998, -0.5),
    (45.0001, 45.0),
    (-45.0001, -45.0),
    (0.123556789, -0.987654321),
]


def haversine_m(a, b):
    lat1, lon1 = a
    lat2, lon2 = b
    lat1 = math.radians(float(lat1))
    lat2 = math.radians(float(lat2))
    dlat = lat2 - lat1
    dlon = math.radians(float(lon2) - float(lon1))
    h = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(h)))


def geo_dict(coord):
    return {"latitude": float(coord[0]), "longitude": float(coord[1])}


def create_collection(client, name):
    return client.collections.create(
        name=name,
        properties=[
            Property(name="tag", data_type=DataType.TEXT, index_filterable=True, index_searchable=True),
            Property(name="location", data_type=DataType.GEO_COORDINATES, index_filterable=True),
        ],
        vector_config=Configure.Vectors.self_provided(),
        inverted_index_config=Configure.inverted_index(index_null_state=True),
    )


def query_tags(collection, coord):
    response = collection.query.fetch_objects(
        filters=Filter.by_property("location").within_geo_range(
            coordinate=GeoCoordinate(latitude=float(coord[0]), longitude=float(coord[1])),
            distance=0.0,
        ),
        limit=100,
    )
    return sorted(obj.properties["tag"] for obj in response.objects)


def expected_tags(state, coord):
    return sorted(
        tag
        for tag, stored_coord in state.items()
        if stored_coord is not None and haversine_m(stored_coord, coord) <= 1e-6
    )


def check(collection, state, coord, label, verbose_failures=False):
    expected = expected_tags(state, coord)
    observed = query_tags(collection, coord)
    ok = expected == observed
    print(f"{label}: expected={expected} observed={observed} result={'PASS' if ok else 'FAIL'}")
    if verbose_failures and not ok:
        print(f"  query_coord={coord}")
        for tag in sorted(set(expected) | set(observed)):
            stored_coord = state.get(tag)
            print(f"  {tag}: coord={stored_coord} distance_m={haversine_m(stored_coord, coord):.12f}")
    return ok


def cleanup(client, name):
    try:
        client.collections.delete(name)
        print(f"Cleaned up collection: {name}")
    except Exception as exc:
        print(f"Cleanup warning for {name}: {exc}")


def main():
    collection_name = f"{COLLECTION_PREFIX}{int(time.time())}{uuid.uuid4().hex[:8]}"
    client = weaviate.connect_to_local(host=HOST, port=PORT, grpc_port=GRPC_PORT)
    try:
        collection = create_collection(client, collection_name)
        state = {}
        ids = {}
        for idx, coord in enumerate(BASE_COORDS):
            tag = f"row_{idx}"
            object_id = str(uuid.UUID(int=idx + 1))
            ids[tag] = object_id
            state[tag] = coord
            collection.data.insert(
                uuid=object_id,
                properties={"tag": tag, "location": geo_dict(coord)},
                vector=VECTOR,
            )

        checks = []
        for idx, coord in enumerate(BASE_COORDS):
            checks.append(check(collection, state, coord, f"initial_zero_{idx}"))

        for idx, (old_coord, new_coord) in enumerate(zip(BASE_COORDS, MOVE_COORDS)):
            tag = f"row_{idx}"
            collection.data.replace(
                uuid=ids[tag],
                properties={"tag": tag, "location": geo_dict(new_coord)},
                vector=VECTOR,
            )
            state[tag] = new_coord
            checks.append(check(collection, state, old_coord, f"after_move_old_zero_{idx}"))
            checks.append(check(collection, state, new_coord, f"after_move_new_zero_{idx}"))

        rng = random.Random(20260412)
        all_coords = BASE_COORDS + MOVE_COORDS
        transient_failures = 0
        persistent_failures = 0
        for step in range(80):
            idx = rng.randrange(len(BASE_COORDS))
            tag = f"row_{idx}"
            old_coord = state[tag]
            base = rng.choice(all_coords)
            jitter = (rng.uniform(-1e-7, 1e-7), rng.uniform(-1e-7, 1e-7))
            new_coord = (
                max(-90.0, min(90.0, base[0] + jitter[0])),
                ((base[1] + jitter[1] + 180.0) % 360.0) - 180.0,
            )
            collection.data.replace(
                uuid=ids[tag],
                properties={"tag": tag, "location": geo_dict(new_coord)},
                vector=VECTOR,
            )
            state[tag] = new_coord
            immediate_ok = check(collection, state, old_coord, f"stress_{step}_old_immediate", verbose_failures=True)
            immediate_ok = check(collection, state, new_coord, f"stress_{step}_new_immediate", verbose_failures=True) and immediate_ok
            if not immediate_ok:
                transient_failures += 1
                time.sleep(0.25)
                delayed_ok = check(collection, state, old_coord, f"stress_{step}_old_delayed", verbose_failures=True)
                delayed_ok = check(collection, state, new_coord, f"stress_{step}_new_delayed", verbose_failures=True) and delayed_ok
                if not delayed_ok:
                    persistent_failures += 1

        print(f"Summary: transient_failures={transient_failures} persistent_failures={persistent_failures}")
        if not all(checks) or persistent_failures:
            raise SystemExit(1)
    finally:
        cleanup(client, collection_name)
        client.close()


if __name__ == "__main__":
    main()
