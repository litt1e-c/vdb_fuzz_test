import os
import sys
import traceback
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

COLLECTION_NAME = "ef_array_probe_fixed"
HOST = os.environ.get("MILVUS_HOST", "127.0.0.1")
PORT = os.environ.get("MILVUS_PORT", "19531")

# 测试列表：包含标准函数（对照组）和实验性函数（element_filter）
exprs = [
    # 对照组：这是 Milvus 官方支持的标准语法，应该成功
    ("control_std", "ARRAY_CONTAINS(int_array, 1)"),
    
    # 实验组：测试 element_filter 是否存在及除零逻辑
    ("div_int", "element_filter(int_array, 1 / x == 1)"),
    ("eq_test", "element_filter(int_array, x == 0)")
]

def ensure_collection():
    if utility.has_collection(COLLECTION_NAME):
        print(f">> Dropping existing collection `{COLLECTION_NAME}`")
        utility.drop_collection(COLLECTION_NAME)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="int_array", dtype=DataType.ARRAY, element_type=DataType.INT64, max_capacity=20),
        # 必须有向量字段
        FieldSchema(name="dummy_vector", dtype=DataType.FLOAT_VECTOR, dim=8),
    ]
    schema = CollectionSchema(fields, description="element_filter probe collection")
    col = Collection(COLLECTION_NAME, schema)
    print(f">> Created collection `{COLLECTION_NAME}`")
    return col

def insert_test_rows(col: Collection):
    print(">> Inserting test rows: id=1 [1], id=2 [0]")
    col.insert([
        [1, 2],  
        [[1], [0]], 
        [[0.1] * 8, [0.0] * 8], 
    ])
    col.flush()
    print(">> Inserted and flushed")

def run_probe():
    try:
        print(f">> Connecting to Milvus at {HOST}:{PORT}")
        connections.connect(host=HOST, port=PORT)
        col = ensure_collection()
        insert_test_rows(col)

        # ==========================================
        # 【关键修复】Load 之前必须建向量索引
        # ==========================================
        print(">> Building Index on dummy_vector...")
        col.create_index("dummy_vector", {"index_type": "FLAT", "metric_type": "L2", "params": {}})
        
        print(">> Loading collection...")
        col.load()
        print(">> Collection loaded successfully.")

    except Exception:
        print("!! Setup failed:")
        traceback.print_exc()
        sys.exit(2)

    print('\n' + '=' * 60)
    print(">> Starting Probes...")

    for name, expr in exprs:
        print(f"\n>> Testing `{expr}` ({name})")
        try:
            # 执行查询
            res = col.query(expr=expr, output_fields=["id", "int_array"])
            print(f"--> [ACCEPTED] Returned {len(res)} rows")
            for r in res:
                print(f"    Hit: {r}")
                
        except Exception as e:
            # 捕获不支持的语法错误，不让脚本崩溃
            error_msg = str(e).split('\n')[0] # 只打印第一行错误
            print(f"--> [REJECTED] Server said: {error_msg}")
            
            # 分析错误类型
            if "field x not exist" in str(e):
                print("    [Analysis]: Server does not support lambda variable 'x'. Feature missing.")
            elif "function" in str(e) and "not found" in str(e):
                print("    [Analysis]: Function not implemented in this version.")

    print('\n' + '=' * 60)
    print(">> Conclusion:")
    print("1. If 'control_std' passed, the environment is healthy.")
    print("2. If 'div_int' failed with 'field x not exist', it means element_filter is NOT supported.")
    print("3. No crash occurred (Milvus is stable).")

if __name__ == '__main__':
    run_probe()