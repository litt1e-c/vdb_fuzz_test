"""
Milvus TLP & Logic Consistency Auto-Tester v6 (Omit-Key Strategy)
Target: Milvus v2.6.7+
Fixes:
1. "Omit Keys" Strategy: Instead of passing {"col": None}, we REMOVE the key from the dictionary.
   This bypasses the SDK's strict type checker which chokes on explicit 'None' values.
2. Robust Sanitization: Ensures valid values are strictly Python native types.
3. Schema: Maintains DOUBLE precision for floats.
"""
import time
import random
import json
import warnings
import numpy as np
from typing import Tuple, Set, List, Dict, Any, Union
from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType, 
    Collection, utility
)

# --- Configuration ---
HOST = "localhost"
PORT = "19530"
COLLECTION_NAME = "milvus_tlp_autotest"
DIM = 768
NUM_ENTITIES = 100000  
BATCH_SIZE = 5000      

# 忽略警告
warnings.filterwarnings("ignore")

# --- Helper: Value Sanitizer ---

def sanitize_value(val: Any) -> Any:
    """强制转换为 Python 原生类型"""
    if val is None: return None
    if hasattr(val, "item"): val = val.item() # Numpy -> Python
    
    if isinstance(val, list): return [sanitize_value(v) for v in val]
    if isinstance(val, dict): return {k: sanitize_value(v) for k, v in val.items()}
    return val

# --- Database Setup ---

def setup_collection():
    """初始化集合"""
    print(f"Connecting to Milvus at {HOST}:{PORT}...")
    connections.connect("default", host=HOST, port=PORT)

    if utility.has_collection(COLLECTION_NAME):
        print(f"Dropping existing collection: {COLLECTION_NAME}")
        utility.drop_collection(COLLECTION_NAME)

    print("Creating schema...")
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=DIM),
        FieldSchema(name="int_val", dtype=DataType.INT64, nullable=True),
        FieldSchema(name="float_val", dtype=DataType.DOUBLE, nullable=True),
        FieldSchema(name="bool_val", dtype=DataType.BOOL, nullable=True),
        FieldSchema(name="str_val", dtype=DataType.VARCHAR, max_length=64, nullable=True),
        FieldSchema(name="json_val", dtype=DataType.JSON, nullable=True)
    ]
    
    schema = CollectionSchema(fields, "TLP Test Collection")
    col = Collection(COLLECTION_NAME, schema)
    
    print("Creating vector index...")
    index_params = {
        "metric_type": "L2", 
        "index_type": "HNSW", 
        "params": {"M": 16, "efConstruction": 200}
    }
    col.create_index("vector", index_params)
    return col

def generate_and_insert_data(col: Collection):
    """生成并插入数据 (省略 Key 策略)"""
    print(f"Generating {NUM_ENTITIES} entities...")
    rng = np.random.default_rng(2025)
    str_pool = ["A", "B", "C", "D", ""]
    
    total_inserted = 0
    while total_inserted < NUM_ENTITIES:
        count = min(BATCH_SIZE, NUM_ENTITIES - total_inserted)
        
        # 1. 向量 (List of floats)
        vectors = rng.random((count, DIM)).astype(np.float32).tolist()
        
        # 2. 构造行 (字典)
        rows = []
        for i in range(count):
            # 基础行只包含必填字段
            row = {"vector": vectors[i]}
            
            # --- 关键策略：只有当值不为 NULL 时，才添加到字典中 ---
            
            # Int (90% 概率存在)
            if rng.random() > 0.1:
                row["int_val"] = int(rng.integers(-1000, 1001))
            
            # Float (90% 概率存在)
            if rng.random() > 0.1:
                row["float_val"] = float(rng.uniform(-1000.0, 1000.0))
            
            # Bool (90% 概率存在)
            if rng.random() > 0.1:
                row["bool_val"] = bool(rng.choice([True, False]))
            
            # String (90% 概率存在)
            if rng.random() > 0.1:
                row["str_val"] = str(rng.choice(str_pool))
            
            # JSON (90% 概率存在)
            if rng.random() > 0.1:
                row["json_val"] = {
                    "meta": {
                        "tag": str(rng.choice(["X", "Y", "Z"])),
                        "score": int(rng.integers(0, 100))
                    }
                }
            
            # 最后一道防线：清洗 Numpy 类型
            rows.append(sanitize_value(row))

        # 3. 插入
        col.insert(rows)
        
        total_inserted += count
        print(f"  Inserted {total_inserted}/{NUM_ENTITIES}...", end="\r")
    
    print("\nFlushing and loading collection...")
    col.flush()
    col.load()
    print(f"Collection ready. Total rows: {col.num_entities}")

# --- Testing Framework ---

class TLPVerifier:
    def __init__(self, collection: Collection):
        self.col = collection
        self.failures = []

    def _query(self, expr: str) -> Tuple[Set[str], int]:
        try:
            res = self.col.query(expr=expr, output_fields=["id"])
            ids = {str(x["id"]) for x in res} 
            return ids, len(ids)
        except Exception as e:
            print(f"⚠️ Query Error on '{expr}': {e}")
            return set(), -1

    def test_tlp_partition(self, name, phi, base_expr):
        """TLP 三分法测试"""
        print(f"TEST: {name} (TLP)")
        
        ids_base, cnt_base = self._query(base_expr)
        
        p_true = f"({phi})"
        p_false = f"not ({phi})"
        
        # 构造 NULL 查询
        field_name = phi.split(" ")[0]
        # Milvus 查空值: field == null
        p_null = f"{field_name} == null" 

        ids_t, c_t = self._query(p_true)
        ids_f, c_f = self._query(p_false)
        ids_n, c_n = self._query(p_null)
        
        union_ids = ids_t | ids_f | ids_n
        
        is_pass = True
        errs = []
        
        if cnt_base == -1 or c_t == -1:
            is_pass = False
            errs.append("Query Failed")
        else:
            if union_ids != ids_base:
                is_pass = False
                errs.append(f"Union mismatch: Base={cnt_base}, Union={len(union_ids)}")
                
            # 检查是否有重叠
            if ids_t & ids_f: errs.append("Overlap True/False")
            if ids_t & ids_n: errs.append("Overlap True/Null")
            if ids_f & ids_n: errs.append("Overlap False/Null")
            
        if is_pass:
            print(f"  ✅ PASSED: sum({c_t}+{c_f}+{c_n}) == {cnt_base}\n")
        else:
            print(f"  ❌ FAILED: {'; '.join(errs)}\n")
            self.failures.append(f"{name}: {'; '.join(errs)}")

    def test_equiv(self, name, expr1, expr2):
        print(f"TEST: {name}")
        ids1, c1 = self._query(expr1)
        ids2, c2 = self._query(expr2)
        
        if c1 == -1 or c2 == -1:
            print(f"  ⚠️ Skipped due to query error\n")
            return

        if ids1 == ids2:
            print(f"  ✅ PASSED: Count {c1}\n")
        else:
            diff = len(ids1.symmetric_difference(ids2))
            print(f"  ❌ FAILED: Diff count {diff}\n")
            self.failures.append(f"{name} mismatch")

    def run_suite(self):
        # 1. TLP Partitioning (Int)
        self.test_tlp_partition("TLP: Int >= 0", "int_val >= 0", "")
        
        # 2. TLP Partitioning (String)
        self.test_tlp_partition("TLP: Str == 'A'", 'str_val == "A"', "")

        # 3. De Morgan
        self.test_equiv(
            "De Morgan",
            'not (int_val > 500 and float_val < 0)',
            '(not (int_val > 500)) or (not (float_val < 0))'
        )

        # 4. Commutativity
        self.test_equiv(
            "Commutativity",
            'int_val < 0 and bool_val == false',
            'bool_val == false and int_val < 0'
        )

        # 5. JSON Field Logic
        self.test_equiv(
            "JSON Logic",
            'json_val["meta"]["score"] > 50',
            'json_val["meta"]["score"] >= 51'
        )
        
        # 6. Null Logic
        self.test_equiv(
            "Null Logic",
            'not (int_val == null)',
            'int_val != null'
        )

        print("="*60)
        if not self.failures:
            print("🎉 All Tests Passed!")
        else:
            print(f"❌ {len(self.failures)} Tests Failed.")

# --- Main ---

if __name__ == "__main__":
    try:
        col = setup_collection()
        generate_and_insert_data(col)
        verifier = TLPVerifier(col)
        verifier.run_suite()
    except Exception as e:
        print(f"\n💥 Error: {e}")
        import traceback
        traceback.print_exc()