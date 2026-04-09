import time
import numpy as np
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection

HOST = "127.0.0.1"
PORT = "19531" 

def main():
    connections.connect("default", host=HOST, port=PORT)
    col_name = "issue_float_downcast_mre"
    if utility.has_collection(col_name):
        utility.drop_collection(col_name)

    # 1. Setup Collection with a FLOAT (float32) scalar field
    schema = CollectionSchema([
        FieldSchema("id", DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema("vec", DataType.FLOAT_VECTOR, dim=2),
        FieldSchema("val_f", DataType.FLOAT),
    ])
    col = Collection(col_name, schema)

    # 2. Insert data: val_f = 1000.0
    # Note: 1000.0 * 26 = 26000.0 (exact representation in both float32 and float64)
    col.insert([[1], [np.random.randn(2).astype(np.float32).tolist()], [1000.0]])
    col.flush()
    col.create_index("vec", {"metric_type": "L2", "index_type": "FLAT", "params": {}})
    col.load()
    time.sleep(1)

    print("=== Milvus Scalar Expression Precision Leakage ===")
    
    # 3. Case 1: The Unexpected False (False Negative)
    # Math logic: 26000.0 > 25999.9999 is strictly TRUE.
    # Milvus logic: "25999.9999" is cast to float32 (26000.0), making it 26000.0 > 26000.0 -> FALSE.
    expr_false = "val_f * 26 > 25999.9999"
    res_false = col.query(expr_false, output_fields=["val_f"])
    
    print(f"\n[Test 1] Expression: {expr_false}")
    print(f"Expected: Return entity id=1 (since 26000.0 > 25999.9999)")
    print(f"Actual:   {res_false}  <-- Evaluated as False!")

    # 4. Case 2: The Unexpected True (False Positive)
    # Math logic: 26000.0 == 25999.9999 is strictly FALSE.
    # Milvus logic: "25999.9999" is cast to float32 (26000.0), making it 26000.0 == 26000.0 -> TRUE.
    expr_true = "val_f * 26 == 25999.9999"
    res_true = col.query(expr_true, output_fields=["val_f"])
    
    print(f"\n[Test 2] Expression: {expr_true}")
    print(f"Expected: Return empty [] (since 26000.0 != 25999.9999)")
    print(f"Actual:   {res_true}  <-- Evaluated as True!")

    print("\nRoot Cause: The expression parser implicitly downcasts the float64 string literal "
          "('25999.9999') to the column's type (float32) before evaluation. "
          "This causes precision loss that swallows the tolerance bounds.")

    utility.drop_collection(col_name)

if __name__ == "__main__":
    main()