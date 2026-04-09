import time
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

HOST = "127.0.0.1"
PORT = "19531"  # 替换为你实际的 Docker 端口
COLLECTION = "bug_null_array_index"

def run_array_index_bug_repro():
    print(f"🔌 连接 Milvus {HOST}:{PORT}...")
    connections.connect(host=HOST, port=PORT)
    
    if utility.has_collection(COLLECTION):
        utility.drop_collection(COLLECTION)

    schema = CollectionSchema([
        FieldSchema("id", DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema("vec", DataType.FLOAT_VECTOR, dim=2),
        FieldSchema("meta", DataType.JSON, nullable=True),
    ])
    col = Collection(COLLECTION, schema)

    # 我们只插入两条数据
    # ID 1: 正常数据，包含普通键和数组键
    # ID 2: 危险数据，完全为空的 JSON {}
    ids = [1, 2]
    vecs = [[0.1, 0.2], [0.3, 0.4]]
    metas = [
        {"val": 1, "arr": [1]},  # ID 1
        {}                       # ID 2 (Missing keys)
    ]
    
    col.insert([ids, vecs, metas])
    col.flush()
    col.create_index("vec", {"metric_type": "L2", "index_type": "FLAT", "params": {}})
    col.load()
    time.sleep(1)

    print("\n=== 预备知识：基于当前的 3VL 逻辑 Bug ===")
    print("在当前的 Milvus 中，not(缺失键 == 999) 会被错误地评估为 True。")
    print("因此，ID 2 应该被返回。")

    # --- 测试 1：对象键访问 (证明你之前的结论) ---
    print("\n" + "=" * 50)
    print("测试 1: 对象键访问 -> not (meta[\"val\"] == 999)")
    expr_key = 'not (meta["val"] == 999)'
    res_key = col.query(expr_key, output_fields=["id"])
    ids_key = sorted([r["id"] for r in res_key])
    print(f"实际返回 IDs: {ids_key}")
    if 2 in ids_key:
        print("💡 表现：如你的 Issue #47164 所述，ID 2 因 3VL 逻辑 Bug 被返回（多出数据）。")

    # --- 测试 2：数组下标访问 (证明新发现的物理崩溃 Bug) ---
    print("\n" + "=" * 50)
    print("测试 2: 数组下标访问 -> not (meta[\"arr\"][0] == 999)")
    expr_arr = 'not (meta["arr"][0] == 999)'
    res_arr = col.query(expr_arr, output_fields=["id"])
    ids_arr = sorted([r["id"] for r in res_arr])
    print(f"实际返回 IDs: {ids_arr}")
    if 2 not in ids_arr:
        print("🚨 致命 Bug 触发：ID 2 消失了！")
        print("原因：底层在计算 meta[\"arr\"][0] 时，由于 meta[\"arr\"] 是 null，")
        print("强行执行 [0] 引发了 C++ 异常。表达式静默崩溃，整行被判为 False，外层的 not 失效！")

    print("\n" + "=" * 50)
    print("结论对比：")
    print(f"同是缺失键，对象访问返回 {ids_key}，数组访问返回 {ids_arr}。")
    print("这证明数组下标操作缺乏 Null Type Guard（类型安全检查）。")
    
    utility.drop_collection(COLLECTION)

if __name__ == "__main__":
    run_array_index_bug_repro()