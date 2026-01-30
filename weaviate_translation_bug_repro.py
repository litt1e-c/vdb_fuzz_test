"""
Weaviate 平移不变性 Bug 复现脚本
=====================================
Bug 描述：L2 距离对平移应完全不变，但 Weaviate 返回不同的 Top-K 结果

理论：d(x+t, y+t) = d(x, y) 对任意平移向量 t
预期：平移前后 Top-K ID 应完全一致，距离误差 < 1e-5
实际：Weaviate 返回不同的结果集（19/20 不匹配）

数据规模：768 维 × 50000 条
"""

import numpy as np
import weaviate
import time
from typing import List, Tuple

# ==================== 配置 ====================
DIM = 768
N = 50000
TOPK = 20
SEED = 123
BATCH_SIZE = 2000

# 平移向量：所有维度 +10.0
TRANSLATION = np.ones(DIM, dtype=np.float32) * 10.0

# 颜色输出
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'

# ==================== 数据准备 ====================
print(f"\n{'='*70}")
print(f"🔬 Weaviate 平移不变性 Bug 复现")
print(f"{'='*70}")
print(f"数据规模: {N} 条, 维度: {DIM}, Top-K: {TOPK}")
print(f"平移向量: 所有维度 +{TRANSLATION[0]:.1f}")

rng = np.random.default_rng(SEED)
ids = np.arange(N, dtype=np.int64)
vecs = rng.random((N, DIM), dtype=np.float32)
probe = vecs[0].copy()  # 使用第一个向量作为查询

# 构造平移后的数据
vecs_translated = vecs + TRANSLATION
probe_translated = probe + TRANSLATION

print(f"\n✅ 数据准备完成")
print(f"   原始查询向量范围: [{probe.min():.3f}, {probe.max():.3f}]")
print(f"   平移查询向量范围: [{probe_translated.min():.3f}, {probe_translated.max():.3f}]")

# ==================== Weaviate 初始化 ====================
print(f"\n🔗 连接 Weaviate...")
client = weaviate.Client("http://127.0.0.1:8080")

# 清理旧数据
try:
    client.schema.delete_all()
    print(f"   已清理旧 Schema")
except:
    pass

# 创建集合定义
class_obj = {
    "class": "TranslationTest",
    "vectorizer": "none",
    "properties": [
        {"name": "idx", "dataType": ["int"]},
        {"name": "group", "dataType": ["string"]}  # "original" or "translated"
    ],
    "vectorIndexConfig": {
        "ef": 5000,
        "efConstruction": 1024,
        "maxConnections": 128
    }
}

client.schema.create_class(class_obj)
print(f"   已创建集合 'TranslationTest'")

# ==================== 插入原始数据 ====================
print(f"\n📥 插入原始数据（group=original）...")
start = time.time()
with client.batch as batch:
    batch.batch_size = BATCH_SIZE
    for i in range(N):
        props = {"idx": int(ids[i]), "group": "original"}
        batch.add_data_object(props, "TranslationTest", vector=vecs[i].tolist())
        if (i + 1) % 10000 == 0:
            print(f"   已插入 {i+1}/{N}...")

elapsed = time.time() - start
print(f"✅ 原始数据插入完成，耗时: {elapsed:.2f}s")

# 等待索引构建（关键！）
print(f"⏳ 等待索引构建...")
time.sleep(5)

# ==================== 查询 1: 原始数据 + 原始查询 ====================
print(f"\n🔍 [查询 1] 原始数据 + 原始查询向量")
response_original = (
    client.query
    .get("TranslationTest", ["idx", "_additional { distance }"])
    .with_near_vector({"vector": probe.tolist()})
    .with_where({"path": ["group"], "operator": "Equal", "valueString": "original"})
    .with_limit(TOPK)
    .do()
)

if 'errors' in response_original:
    print(f"{RED}❌ 查询失败: {response_original['errors']}{RESET}")
    exit(1)

results_original = response_original['data']['Get']['TranslationTest']
ids_original = [item['idx'] for item in results_original]
dists_original = [item['_additional']['distance'] for item in results_original]

print(f"   Top-5 ID: {ids_original[:5]}")
print(f"   Top-5 距离: {[f'{d:.6f}' for d in dists_original[:5]]}")

# ==================== 插入平移数据 ====================
print(f"\n📥 插入平移数据（group=translated）...")
start = time.time()
with client.batch as batch:
    batch.batch_size = BATCH_SIZE
    for i in range(N):
        props = {"idx": int(ids[i]), "group": "translated"}
        batch.add_data_object(props, "TranslationTest", vector=vecs_translated[i].tolist())
        if (i + 1) % 10000 == 0:
            print(f"   已插入 {i+1}/{N}...")

elapsed = time.time() - start
print(f"✅ 平移数据插入完成，耗时: {elapsed:.2f}s")

# 等待索引更新
print(f"⏳ 等待索引更新...")
time.sleep(5)

# ==================== 查询 2: 平移数据 + 平移查询 ====================
print(f"\n🔍 [查询 2] 平移数据 + 平移查询向量")
response_translated = (
    client.query
    .get("TranslationTest", ["idx", "_additional { distance }"])
    .with_near_vector({"vector": probe_translated.tolist()})
    .with_where({"path": ["group"], "operator": "Equal", "valueString": "translated"})
    .with_limit(TOPK)
    .do()
)

if 'errors' in response_translated:
    print(f"{RED}❌ 查询失败: {response_translated['errors']}{RESET}")
    exit(1)

results_translated = response_translated['data']['Get']['TranslationTest']
ids_translated = [item['idx'] for item in results_translated]
dists_translated = [item['_additional']['distance'] for item in results_translated]

print(f"   Top-5 ID: {ids_translated[:5]}")
print(f"   Top-5 距离: {[f'{d:.6f}' for d in dists_translated[:5]]}")

# ==================== 结果对比 ====================
print(f"\n{'='*70}")
print(f"📊 结果对比分析")
print(f"{'='*70}")

# ID 匹配率
id_matches = sum(1 for a, b in zip(ids_original, ids_translated) if a == b)
id_match_rate = id_matches / TOPK

# 距离差异
dists_original_arr = np.array(dists_original)
dists_translated_arr = np.array(dists_translated)
dist_diff = np.abs(dists_original_arr - dists_translated_arr)
avg_dist_diff = np.mean(dist_diff)
max_dist_diff = np.max(dist_diff)

print(f"\n1️⃣  ID 一致性:")
print(f"   匹配数量: {id_matches}/{TOPK}")
print(f"   匹配率: {id_match_rate*100:.1f}%")
if id_match_rate == 1.0:
    print(f"   状态: {GREEN}✅ 完全一致{RESET}")
else:
    print(f"   状态: {RED}❌ 不一致（发现 Bug！）{RESET}")
    
    # 显示不匹配的位置
    mismatches = [(i, ids_original[i], ids_translated[i]) 
                  for i in range(TOPK) if ids_original[i] != ids_translated[i]]
    print(f"\n   不匹配详情（前10个）:")
    for rank, orig_id, trans_id in mismatches[:10]:
        print(f"      Rank {rank+1}: {orig_id} → {trans_id}")

print(f"\n2️⃣  距离一致性:")
print(f"   平均差异: {avg_dist_diff:.8f}")
print(f"   最大差异: {max_dist_diff:.8f}")
print(f"   理论期望: < 1e-5 (浮点精度)")

if avg_dist_diff < 1e-5:
    print(f"   状态: {GREEN}✅ 精度正常{RESET}")
else:
    print(f"   状态: {RED}❌ 精度异常（发现 Bug！）{RESET}")

# ==================== 验证理论 ====================
print(f"\n3️⃣  理论验证:")
print(f"   L2 距离对平移的不变性: d(x+t, y+t) = d(x, y)")

# 手动计算前5个向量的距离
print(f"\n   手动验证（前5个ID）:")
for i in range(min(5, len(ids_original))):
    id_orig = ids_original[i]
    id_trans = ids_translated[i]
    
    # 原始距离
    d_orig = np.linalg.norm(probe - vecs[id_orig])
    # 平移距离
    d_trans = np.linalg.norm(probe_translated - vecs_translated[id_trans])
    
    match_symbol = "✅" if id_orig == id_trans else "❌"
    print(f"      ID {id_orig:6d} vs {id_trans:6d} {match_symbol}  |  "
          f"距离: {d_orig:.6f} vs {d_trans:.6f}  (diff: {abs(d_orig-d_trans):.8f})")

# ==================== 最终结论 ====================
print(f"\n{'='*70}")
print(f"🎯 最终结论")
print(f"{'='*70}")

if id_match_rate == 1.0 and avg_dist_diff < 1e-5:
    print(f"{GREEN}✅ Weaviate 平移不变性测试通过{RESET}")
    print(f"   未复现 Bug，可能已修复或环境差异")
else:
    print(f"{RED}🚨 Weaviate 平移不变性 Bug 已复现！{RESET}")
    print(f"\nBug 症状:")
    print(f"  ❌ Top-K ID 不一致: {TOPK - id_matches} 个不匹配")
    print(f"  ❌ 距离差异异常: 平均 {avg_dist_diff:.6f} (期望 < 1e-5)")
    print(f"\n影响范围:")
    print(f"  - L2 距离搜索结果不稳定")
    print(f"  - 数据分布变化会影响召回结果")
    print(f"  - 不适合需要几何正确性的应用场景")
    print(f"\n建议:")
    print(f"  1. 上报至 Weaviate GitHub Issues")
    print(f"  2. 生产环境切换至 Milvus/Qdrant")
    print(f"  3. 等待官方修复后再使用")

print(f"\n{'='*70}\n")

# 清理
print(f"🧹 清理数据...")
try:
    client.schema.delete_class("TranslationTest")
    print(f"✅ 已删除测试集合")
except:
    pass
