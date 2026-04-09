from qdrant_client import QdrantClient
from qdrant_client.models import (
PointStruct, Filter, FieldCondition, GeoBoundingBox, GeoPoint, VectorParams, Distance
)

def run_test(client_type, client):
    collection = "geo_boundary_test"

    if client.collection_exists(collection):
        client.delete_collection(collection)
    
    client.create_collection(
        collection_name=collection, 
        vectors_config=VectorParams(size=2, distance=Distance.DOT)
    )

    client.upsert(
        collection_name=collection,
        points=[
            PointStruct(id=1, vector=[0.1, 0.1], payload={"location_geo": {"lat": 90.0, "lon": 180.0}})
        ]
    )

    bbox = GeoBoundingBox(
        top_left=GeoPoint(lon=158.75, lat=90.00),
        bottom_right=GeoPoint(lon=180.00, lat=69.40)
    )

    geo_filter = Filter(must=[FieldCondition(key="location_geo", geo_bounding_box=bbox)])
    results = client.scroll(collection_name=collection, scroll_filter=geo_filter)[0]

    print(f"[{client_type}] Actual IDs returned: {[p.id for p in results]}")
if __name__ == "main":
    print("--- Qdrant GeoBoundingBox Consistency Test ---")

    client_memory = QdrantClient(":memory:")
    run_test("Memory Emulator", client_memory)

    try:
        client_server = QdrantClient(host="localhost", port=6333)
        run_test("Rust Server (localhost)", client_server)
    except Exception as e:
        print(f"[Rust Server (localhost)] Connection failed. Please ensure Qdrant is running on port 6333.")