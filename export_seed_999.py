import random
import numpy as np
import pandas as pd
import json
from milvus_fuzz_oracle import DataManager

# ================= 配置 =================
SEED = 999
OUTPUT_CSV = "fuzz_data_999.csv"
OUTPUT_JSON = "fuzz_data_999.json"

def save_dataset():
    print(f"🔒 Setting seed to {SEED} to match the Fuzzer run...")
    random.seed(SEED)
    np.random.seed(SEED)

    # 1. 调用 Fuzzer 的逻辑生成数据
    print("🎲 Initializing DataManager...")
    dm = DataManager()
    
    print("📋 Generating Schema...")
    dm.generate_schema()
    
    print("🌊 Regenerating 5000 rows in memory (Exact Replica)...")
    dm.generate_data()
    
    # 获取 DataFrame
    df = dm.df.copy()

    # 2. 处理向量数据
    # DataManager 生成的 vectors 通常存储在 dm.vectors (numpy array)，而不是 df 中
    # 我们把它合并进 df 以便保存
    if hasattr(dm, 'vectors'):
        print("   -> Merging vectors into dataset...")
        # 将 numpy array 转换为 list，方便保存为 JSON/CSV
        df['vector'] = dm.vectors.tolist()

    # 3. 保存为 CSV (适合 Excel 查看，但 JSON 字段会变成字符串)
    print(f"💾 Saving to {OUTPUT_CSV}...")
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    # 4. 保存为 JSON (保留 meta_json 和 array 的结构，适合程序读取)
    print(f"💾 Saving to {OUTPUT_JSON}...")
    # default_handler 用于处理 numpy 类型
    df.to_json(OUTPUT_JSON, orient="records", indent=2, force_ascii=False)

    print("\n✅ Export Complete!")
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {list(df.columns)}")
    print("\n你可以用 Excel 打开 .csv 文件，或者用文本编辑器查看 .json 文件。")

if __name__ == "__main__":
    save_dataset()