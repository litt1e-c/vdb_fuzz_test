import time
from pymilvus import (
    connections, utility, FieldSchema, CollectionSchema, DataType, Collection
)

# --- 配置 ---
HOST = "127.0.0.1"
PORT = "19531"
COLLECTION_NAME = "milvus_logic_probe_fixed"

def run_truth_table():
    print(f"🔌 Connecting to Milvus...")
    connections.connect("default", host=HOST, port=PORT)

    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)

    # 1. 只有一行数据：{id: 1, c_null: None, c_true: True}
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128),
        FieldSchema(name="c_null", dtype=DataType.BOOL, nullable=True), 
        FieldSchema(name="c_true", dtype=DataType.BOOL),                
    ]
    
    schema = CollectionSchema(fields, enable_dynamic_field=True)
    col = Collection(COLLECTION_NAME, schema)
    
    print("⚡ Inserting 1 row: {'c_null': None, 'c_true': True}...")
    col.insert([{
        "id": 1,
        "vector": [0.1]*128,
        "c_null": None,
        "c_true": True
    }])
    col.flush()
    col.create_index("vector", {"metric_type": "L2", "index_type": "FLAT", "params": {}})
    col.load()

    print("\n🕵️‍♂️ Probing Milvus Logic System (Fixed)...")
    print("=" * 70)
    print(f"{'Expression':<45} | {'Hits':<5} | {'Meaning'}")
    print("-" * 70)

    def probe(expr):
        try:
            res = col.query(expr, output_fields=["id"])
            hits = len(res)
            return hits
        except Exception as e:
            print(f"{expr:<45} | ERR   | {e}")
            return -1

    # --- 1. 基础定义 ---
    # c_true 肯定是 True，c_null 是 Unknown
    probe("c_true == true") # 验证数据存在 (Expect 1)

    # --- 2. 验证 Null 的真值 (Atom Logic) ---
    # 2.1: Null == False
    # 如果 hits=0，说明 Null != False (可能是 False，也可能是 Unknown)
    h_atom = probe("c_null == false") 
    print(f"{'c_null == false':<45} | {h_atom:<5} | Null is not False")

    # 2.2: NOT (Null == False)
    # 如果 hits=0，说明 NOT(Unknown) = Unknown (被过滤)。这是 SQL 标准行为。
    # 如果 hits=1，说明 NOT(False) = True。这是 Python 行为。
    h_not_atom = probe("not (c_null == false)")
    logic_type = "3VL (SQL-like)" if h_not_atom == 0 else "2VL (Python-like)"
    print(f"{'not (c_null == false)':<45} | {h_not_atom:<5} | Logic Type: {logic_type}")

    # --- 3. 验证复合逻辑 (Complex Logic) ---
    # 我们用 c_true == false 来代表 False 常量
    
    # 3.1: OR 逻辑 -> Unknown OR True
    # SQL: Unknown OR True = True
    expr_or = "(c_null == false) or (c_true == true)"
    h_or = probe(expr_or)
    print(f"{expr_or:<45} | {h_or:<5} | Unknown OR True = {h_or}")

    # 3.2: AND 逻辑 -> Unknown AND False
    # SQL: Unknown AND False = False
    expr_and = "(c_null == false) and (c_true == false)"
    h_and = probe(expr_and)
    print(f"{expr_and:<45} | {h_and:<5} | Unknown AND False = {h_and}")

    # --- 4. 关键验证: 复合取反 (The Inconsistency Check) ---
    # NOT ( Unknown AND False )
    # 如果是 SQL 3VL: NOT(False) -> True (应该返回 1)
    # 如果是 Milvus Bug: 可能返回 0
    
    print("-" * 70)
    expr_complex = "not ((c_null == false) and (c_true == false))"
    h_complex = probe(expr_complex)
    print(f"{expr_complex:<45} | {h_complex:<5} | NOT(Unknown AND False)")
    
    print("=" * 70)
    
    # --- 最终诊断 ---
    if h_not_atom == 0 and h_complex == 1:
        print("✅ MILVUS IS STANDARD SQL 3-VALUED LOGIC.")
        print("   - Level 3 (0 hits) is CORRECT: NOT(Unknown) is Unknown.")
        print("   - Level 4 (1 hit)  is CORRECT: NOT(False) is True.")
        print("   👉 CONCLUSION: This is NOT a bug. Your fuzzer needs to adapt to SQL logic.")
    
    elif h_not_atom == 0 and h_complex == 0:
        print("🚨 BUG CONFIRMED: LOGIC BREAKDOWN.")
        print("   - Milvus follows 3VL for atoms (NOT Unknown = Unknown).")
        print("   - BUT fails for compound logic (NOT False should be True).")
        print("   👉 CONCLUSION: Please submit the issue.")
        
    elif h_not_atom == 1:
        print("ℹ️  MILVUS IS 2-VALUED LOGIC.")
        print("   - Null is treated strictly as False.")
        print("   - If previous tests failed, it's specific to complex query optimization.")

if __name__ == "__main__":
    run_truth_table()