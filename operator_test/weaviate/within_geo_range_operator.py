import math
import sys
import time
import uuid

import weaviate
from weaviate.classes.config import Configure, DataType, Property, Tokenization
from weaviate.classes.data import DataObject, GeoCoordinate
from weaviate.classes.query import Filter


HOST = "127.0.0.1"
PORT = 8080
GRPC_PORT = 50051
CORE_COLLECTION_PREFIX = "WithinGeoRangeOperatorValidation"
CROWDED_COLLECTION_PREFIX = "WithinGeoRangeCrowdedValidation"
CROWD_COUNT = 805

EARTH_RADIUS_M = 6_371_000.0


def deterministic_uuid(label: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"weaviate-within-geo-range-{label}"))


def lat_offset_deg(distance_m: float) -> float:
    return math.degrees(distance_m / EARTH_RADIUS_M)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


CORE_ROWS = [
    DataObject(
        uuid=deterministic_uuid("center"),
        properties={
            "tag": "center",
            "location": GeoCoordinate(latitude=0.0, longitude=0.0),
        },
        vector=[1.0, 0.0, 0.0],
    ),
    DataObject(
        uuid=deterministic_uuid("inside"),
        properties={
            "tag": "inside",
            "location": GeoCoordinate(latitude=lat_offset_deg(500.0), longitude=0.0),
        },
        vector=[0.0, 1.0, 0.0],
    ),
    DataObject(
        uuid=deterministic_uuid("boundary"),
        properties={
            "tag": "boundary",
            "location": GeoCoordinate(latitude=lat_offset_deg(1000.0), longitude=0.0),
        },
        vector=[0.0, 0.0, 1.0],
    ),
    DataObject(
        uuid=deterministic_uuid("outside"),
        properties={
            "tag": "outside",
            "location": GeoCoordinate(latitude=lat_offset_deg(1015.0), longitude=0.0),
        },
        vector=[0.5, 0.5, 0.0],
    ),
    DataObject(
        uuid=deterministic_uuid("far"),
        properties={
            "tag": "far",
            "location": GeoCoordinate(latitude=0.2, longitude=0.2),
        },
        vector=[0.5, 0.0, 0.5],
    ),
    DataObject(
        uuid=deterministic_uuid("null"),
        properties={
            "tag": "null",
            "location": None,
        },
        vector=[0.2, 0.2, 0.2],
    ),
    DataObject(
        uuid=deterministic_uuid("missing"),
        properties={
            "tag": "missing",
        },
        vector=[0.1, 0.1, 0.1],
    ),
    DataObject(
        uuid=deterministic_uuid("dateline_east"),
        properties={
            "tag": "dateline_east",
            "location": GeoCoordinate(latitude=0.0, longitude=179.9990),
        },
        vector=[0.3, 0.1, 0.2],
    ),
    DataObject(
        uuid=deterministic_uuid("dateline_west"),
        properties={
            "tag": "dateline_west",
            "location": GeoCoordinate(latitude=0.0, longitude=-179.9990),
        },
        vector=[0.4, 0.1, 0.2],
    ),
]


def create_core_collection(client, name: str):
    return client.collections.create(
        name=name,
        properties=[
            Property(
                name="tag",
                data_type=DataType.TEXT,
                tokenization=Tokenization.FIELD,
                index_filterable=True,
            ),
            Property(
                name="location",
                data_type=DataType.GEO_COORDINATES,
                index_filterable=True,
            ),
        ],
        vector_config=Configure.Vectors.self_provided(),
        inverted_index_config=Configure.inverted_index(index_null_state=True),
    )


def create_crowded_collection(client, name: str):
    return client.collections.create(
        name=name,
        properties=[
            Property(
                name="tag",
                data_type=DataType.TEXT,
                tokenization=Tokenization.FIELD,
                index_filterable=True,
            ),
            Property(
                name="band",
                data_type=DataType.TEXT,
                tokenization=Tokenization.FIELD,
                index_filterable=True,
            ),
            Property(
                name="location",
                data_type=DataType.GEO_COORDINATES,
                index_filterable=True,
            ),
        ],
        vector_config=Configure.Vectors.self_provided(),
        inverted_index_config=Configure.inverted_index(index_null_state=True),
    )


def make_crowded_rows():
    rows = []
    for meter in range(1, CROWD_COUNT + 1):
        rows.append(
            DataObject(
                uuid=deterministic_uuid(f"crowd-{meter}"),
                properties={
                    "tag": f"crowd_{meter:03d}",
                    "band": "beyond_800" if meter > 800 else "head_800",
                    "location": GeoCoordinate(latitude=lat_offset_deg(float(meter)), longitude=0.0),
                },
                vector=[1.0, 0.0, 0.0],
            )
        )
    return rows


def insert_rows(collection, rows):
    result = collection.data.insert_many(rows)
    errors = getattr(result, "errors", None)
    if errors:
        raise RuntimeError(f"insert_many errors: {errors}")


def fetch_values(collection, flt, prop: str = "tag", limit: int = 50):
    response = collection.query.fetch_objects(filters=flt, limit=limit)
    return sorted(obj.properties[prop] for obj in response.objects)


def print_check(name, doc_expectation, oracle_expectation, expected, actual):
    print(name)
    print(f"  docs:    {doc_expectation}")
    print(f"  oracle:  {oracle_expectation}")
    print(f"  expected:{expected}")
    print(f"  observed:{actual}")
    ok = actual == expected
    print(f"  result:  {'PASS' if ok else 'FAIL'}")
    return ok


def print_observation(name, note, actual):
    print(name)
    print(f"  note:    {note}")
    print(f"  observed:{actual}")
    print("  result:  OBSERVED")


def make_filter(lat: float, lon: float, distance: float):
    return Filter.by_property("location").within_geo_range(
        coordinate=GeoCoordinate(latitude=lat, longitude=lon),
        distance=distance,
    )


def cleanup_collection(client, name: str):
    try:
        if client.collections.exists(name):
            client.collections.delete(name)
            print(f"Cleaned up collection: {name}")
    except Exception as exc:
        print(f"Cleanup warning for {name}: {exc}")


def main():
    core_collection_name = f"{CORE_COLLECTION_PREFIX}{int(time.time())}{uuid.uuid4().hex[:8]}"
    crowded_collection_name = f"{CROWDED_COLLECTION_PREFIX}{int(time.time())}{uuid.uuid4().hex[:8]}"
    client = weaviate.connect_to_local(host=HOST, port=PORT, grpc_port=GRPC_PORT)
    created_collections = []
    try:
        core_collection = create_core_collection(client, core_collection_name)
        created_collections.append(core_collection_name)
        insert_rows(core_collection, CORE_ROWS)

        boundary_distance = haversine_m(0.0, 0.0, lat_offset_deg(1000.0), 0.0)
        outside_distance = haversine_m(0.0, 0.0, lat_offset_deg(1015.0), 0.0)

        checks = []
        checks.append(
            print_check(
                "within_1000m_core_subset",
                "WithinGeoRange matches geo-coordinate properties within the requested distance in meters.",
                "A conservative oracle uses great-circle distance and treats null or missing geo values as non-matches.",
                ["boundary", "center", "inside"],
                fetch_values(core_collection, make_filter(0.0, 0.0, 1000.0)),
            )
        )
        print_observation(
            "within_zero_distance_observation",
            "Docs describe radius filtering but do not state a separate zero-distance precision rule.",
            fetch_values(core_collection, make_filter(0.0, 0.0, 0.0)),
        )
        checks.append(
            print_check(
                "within_600m_excludes_null_missing_and_farther_points",
                "Docs describe radius filtering; they do not provide a null or missing truth table.",
                "Current oracle treats null or missing geo values as non-matches for positive within_geo_range.",
                ["center", "inside"],
                fetch_values(core_collection, make_filter(0.0, 0.0, 600.0)),
            )
        )

        print_observation(
            "boundary_distance_observation",
            f"The exact inserted boundary point is approximately {boundary_distance:.6f} meters from the query center using the local haversine oracle helper.",
            fetch_values(core_collection, make_filter(0.0, 0.0, boundary_distance)),
        )
        print_observation(
            "outside_distance_observation",
            f"The slightly-outside point is approximately {outside_distance:.6f} meters from the query center using the local haversine oracle helper.",
            fetch_values(core_collection, make_filter(0.0, 0.0, outside_distance)),
        )

        dateline_center_lon = 179.9995
        dateline_east_distance = haversine_m(0.0, dateline_center_lon, 0.0, 179.9990)
        dateline_west_distance = haversine_m(0.0, dateline_center_lon, 0.0, -179.9990)
        dateline_distance = max(dateline_east_distance, dateline_west_distance) + 5.0
        checks.append(
            print_check(
                "within_geo_range_dateline_wraparound",
                "Docs describe radius filtering on geo coordinates but do not spell out anti-meridian wraparound details.",
                "A conservative geo oracle uses spherical distance modulo longitude wraparound, so both sides of the date line should match here.",
                ["dateline_east", "dateline_west"],
                fetch_values(core_collection, make_filter(0.0, dateline_center_lon, dateline_distance)),
            )
        )

        crowded_collection = create_crowded_collection(client, crowded_collection_name)
        created_collections.append(crowded_collection_name)
        insert_rows(crowded_collection, make_crowded_rows())

        crowded_filter = make_filter(0.0, 0.0, 5000.0)
        expected_head_800 = [f"crowd_{meter:03d}" for meter in range(1, 801)]
        checks.append(
            print_check(
                "within_geo_range_nearest_800_cap",
                "The checked docs say geo-coordinate filtering is limited to the nearest 800 results from the source location.",
                "The main fuzzer must not assume full mathematical-radius semantics when more than 800 rows fall inside the radius.",
                expected_head_800,
                fetch_values(crowded_collection, crowded_filter, limit=1000),
            )
        )
        checks.append(
            print_check(
                "within_geo_range_cap_applies_before_additional_filter",
                "The checked docs say the nearest-800 geo candidate reduction happens before any additional filters are applied.",
                "The conservative oracle therefore avoids generated geo predicates whose raw geo candidate set exceeds 800 rows.",
                [],
                fetch_values(
                    crowded_collection,
                    crowded_filter & Filter.by_property("band").equal("beyond_800"),
                    limit=1000,
                ),
            )
        )

        if all(checks):
            print("Summary: all validated within_geo_range checks passed on the local Weaviate service.")
            return 0

        print("Summary: one or more validated within_geo_range checks failed.")
        return 1
    finally:
        for name in reversed(created_collections):
            cleanup_collection(client, name)
        client.close()


if __name__ == "__main__":
    sys.exit(main())
