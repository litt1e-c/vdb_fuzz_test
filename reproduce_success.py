import time
from pymilvus import (
    connections, utility, FieldSchema, CollectionSchema, DataType, Collection
)

# --- Configuration ---
HOST = "127.0.0.1"
PORT = "19531"
COLLECTION_NAME = "milvus_logic_bug_proof"

# --- Test Data Row ---
DATA_ROW = {
    'id': 1051, 
    'vector': [0.1] * 128,
    'c0': None, 'c1': True, 'c2': 55943, 'c3': 2379.7519128726576, 'c4': None, 'c5': True, 
    'c6': -57942, 'c7': False, 'c8': 'Vaxr5xzbWPHJazy9loD', 'c9': 1682.7331496087677, 
    'c10': None, 'c11': -62195, 'c12': False, 'c13': 71210, 'c14': None, 
    'c15': 2884.9004720171306, 'c16': 4866.809897346821, 'c17': True, 
    'meta_json': {'price': 561, 'color': 'Green', 'active': False, 'config': {'version': 7}, 'history': [7, 20, 88], 'random_payload': 12168}, 
    'tags_array': [73]
}

# --- Block A: Left Expression ---
EXPR_LEFT = """
(((meta_json["config"]["version"] == 9 or (meta_json["history"][0] > 49 or (meta_json["history"][0] > 47 and (meta_json["price"] > 294 and meta_json["price"] < 379)))) and (((meta_json["config"]["version"] == 7 or meta_json["history"][0] > 27) or (tags_array is not null or null is null)) or c17 == false)) or ((((c14 <= 863.28694295187 or (c10 != "kaKmPWPbAaEFnHzX" and c10 != "ZmBmv")) and ((c4 > 1113.2711377477458 or c13 >= -76839) or (meta_json["config"]["version"] == 3 or (meta_json["price"] > 150 and meta_json["price"] < 237)))) and c7 == false) and ((((c1 is not null and null is null) or (c12 == false and meta_json["config"]["version"] == 4)) or (((meta_json["active"] == true and meta_json["color"] == "Blue") and c12 == true) and (c6 is not null or c13 == 180192))) or (c4 < 105376.953125 or ((c15 <= 3484.876420871185 or c1 == false) or (meta_json["config"]["version"] == 9 and c10 != "OjcbvwxZ5LfV2PZNPgS7"))))))
"""

# --- Block B: Right Expression ---
EXPR_RIGHT = """
((((((c17 == false or exists(meta_json["non_exist"])) or c14 <= 5363.7872388701035) or meta_json["history"][0] > 72) or (((c4 is not null and c17 == true) and ((c2 < 60835 or c1 is null) or (c5 == false or meta_json["history"][0] > 64))) and ((c6 >= -56376 or c16 is not null) or ((c4 >= 812.9318963654309 and c16 >= 2907.7772038042867) and (meta_json["price"] > 449 and meta_json["price"] < 624))))) or (((((null is not null and c13 < 180192) or (c13 < -71746 or c17 == true)) and ((meta_json["price"] > 407 and meta_json["price"] < 521) and (c8 like "w%" or c14 > 3021.9496428003613))) and (((c12 == false or (meta_json["price"] > 412 and meta_json["price"] < 571)) or (c9 < 1264.220988053753 or c6 != 2063)) or c5 == false)) and (meta_json["active"] == true and meta_json["color"] == "Red"))) or ((((c13 != 54759 and (c7 is null and (c17 == true or c9 <= 1067.5165787120216))) and (((c11 != 36936 or meta_json["history"][0] > 59) or ((meta_json["price"] > 354 and meta_json["price"] < 528) and c9 >= 135.84002581554114)) and ((c0 > 105381.9375 or exists(meta_json["non_exist"])) or (c12 == true or meta_json is null)))) or ((((c14 >= 2547.2285277180335 and c1 == false) and (meta_json["history"][0] > 37 or json_contains(meta_json["k_11"], "o"))) and ((c9 < 2869.0647504243566 and c4 > 449.2640493406897) or (meta_json["config"]["version"] == 9 and c16 < 3626.0660282789318))) and (((c11 <= -57792 or (meta_json["price"] > 320 and meta_json["price"] < 488)) and (meta_json["active"] == false and c3 >= 105383.3671875)) or ((c13 >= -32496 and c5 == true) and (meta_json["active"] == true and meta_json["color"] == "Blue"))))) and c13 <= 180192))
"""

# Combine expressions with outer parentheses
EXPR_COMBINED = f"({EXPR_LEFT}) OR ({EXPR_RIGHT})"


def run_proof():
    print("Starting Milvus logic bug reproduction script...")

    try:
        connections.connect("default", host=HOST, port=PORT)
    except Exception as e:
        print(f"Connection failed: {e}")
        return
    
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)

    # --- Define collection schema ---
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128),
        FieldSchema(name="c0", dtype=DataType.DOUBLE, nullable=True),
        FieldSchema(name="c1", dtype=DataType.BOOL, nullable=True),
        FieldSchema(name="c2", dtype=DataType.INT64, nullable=True),
        FieldSchema(name="c3", dtype=DataType.DOUBLE, nullable=True),
        FieldSchema(name="c4", dtype=DataType.DOUBLE, nullable=True),
        FieldSchema(name="c5", dtype=DataType.BOOL, nullable=True),
        FieldSchema(name="c6", dtype=DataType.INT64, nullable=True),
        FieldSchema(name="c7", dtype=DataType.BOOL, nullable=True),
        FieldSchema(name="c8", dtype=DataType.VARCHAR, max_length=512, nullable=True),
        FieldSchema(name="c9", dtype=DataType.DOUBLE, nullable=True),
        FieldSchema(name="c10", dtype=DataType.VARCHAR, max_length=512, nullable=True),
        FieldSchema(name="c11", dtype=DataType.INT64, nullable=True),
        FieldSchema(name="c12", dtype=DataType.BOOL, nullable=True),
        FieldSchema(name="c13", dtype=DataType.INT64, nullable=True),
        FieldSchema(name="c14", dtype=DataType.DOUBLE, nullable=True),
        FieldSchema(name="c15", dtype=DataType.DOUBLE, nullable=True),
        FieldSchema(name="c16", dtype=DataType.DOUBLE, nullable=True),
        FieldSchema(name="c17", dtype=DataType.BOOL, nullable=True),
        FieldSchema(name="meta_json", dtype=DataType.JSON, nullable=True),
        FieldSchema(name="tags_array", dtype=DataType.ARRAY, element_type=DataType.INT64, max_capacity=50, nullable=True)
    ]
    
    schema = CollectionSchema(fields, enable_dynamic_field=True)
    col = Collection(COLLECTION_NAME, schema)
    
    print("Inserting test data...")
    col.insert([DATA_ROW])
    col.flush()
    
    print("Building vector index (any index type also reproduces the issue)...")
    index_params = {
        "metric_type": "L2",
        "index_type": "HNSW",
        "params": {"M": 32, "efConstruction": 256}
    }
    col.create_index("vector", index_params)
    col.load()

    print("\nRunning boolean logic verification (Expected: False OR False = False)...")
    
    # Test left expression
    res_left = col.query(EXPR_LEFT, output_fields=["id"])
    is_left_true = len(res_left) > 0
    print(f"[Expr Left ] Result: {is_left_true} (Expected: False)")
    
    # Test right expression
    res_right = col.query(EXPR_RIGHT, output_fields=["id"])
    is_right_true = len(res_right) > 0
    print(f"[Expr Right] Result: {is_right_true} (Expected: False)")
    
    # Test combined expression
    res_combined = col.query(EXPR_COMBINED, output_fields=["id"])
    is_combined_true = len(res_combined) > 0
    print(f"[Combined ] Result: {is_combined_true}")

    print("\nFinal Verdict:")
    if not is_left_true and not is_right_true and is_combined_true:
        print("BUG CONFIRMED: Milvus evaluates (False OR False) as True.")
        print("Likely cause: incorrect bitset merge in filter executor under vector query path.")
    elif is_combined_true:
        print("Combined is True but subexpressions are also True — test case invalid.")
    else:
        print("No bug reproduced: (False OR False = False).")


if __name__ == "__main__":
    run_proof()
