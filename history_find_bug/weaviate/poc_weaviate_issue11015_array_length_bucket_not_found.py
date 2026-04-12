import weaviate
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import Filter

# Connect to local Weaviate instance
client = weaviate.connect_to_local()

CLASS_NAME = "BugReportLengthIndex"

try:
    if client.collections.exists(CLASS_NAME):
        client.collections.delete(CLASS_NAME)

    # 1. Create collection with GLOBAL index_property_length=True
    col = client.collections.create(
        name=CLASS_NAME,
        inverted_index_config=Configure.inverted_index(
            index_property_length=True  # We explicitly request length indexing
        ),
        properties=[
            # Control Group: index_filterable=True
            Property(
                name="ints_filterable",
                data_type=DataType.INT_ARRAY,
                index_filterable=True,
            ),
            # Bug Group: index_filterable=False
            Property(
                name="ints_non_filterable",
                data_type=DataType.INT_ARRAY,
                index_filterable=False,  # This disables the value index
            ),
        ],
    )

    # 2. Insert test data
    col.data.insert({
        "ints_filterable": [1, 2],
        "ints_non_filterable": [1, 2]
    })

    # 3. Test 1: Query by length on the filterable array (Works as expected)
    print("--- Test 1: Querying length on 'ints_filterable' ---")
    try:
        col.query.fetch_objects(
            filters=Filter.by_property("ints_filterable", length=True).equal(2)
        )
        print("Result: SUCCESS\n")
    except Exception as e:
        print(f"Result: FAILED - {e}\n")

    # 4. Test 2: Query by length on the non-filterable array (Triggers the Bug)
    # Expected behavior: 
    #   Either successful length filtering (since index_property_length=True globally)
    #   OR a graceful 400 Bad Request validation error.
    # Actual behavior: 
    #   Unhandled gRPC panic / underlying bucket not found error.
    print("--- Test 2: Querying length on 'ints_non_filterable' ---")
    try:
        col.query.fetch_objects(
            filters=Filter.by_property("ints_non_filterable", length=True).equal(2)
        )
        print("Result: SUCCESS\n")
    except Exception as e:
        print("Result: BUG REPRODUCED!")
        print(f"Exception Type: {type(e).__name__}")
        print(f"Error Message : {e}")

finally:
    # Cleanup
    if client.collections.exists(CLASS_NAME):
        client.collections.delete(CLASS_NAME)
    client.close()