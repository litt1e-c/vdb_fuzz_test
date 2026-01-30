"""
Weaviate 独立集合测试
======================
对比两种方式：
1. 同一个 class + filter（之前的方式）
2. 两个独立 class（与 Milvus/Qdrant 一致）
"""

import numpy as np
import weaviate
import time

DIM = 768
N = 50000
TOPK = 20
SEED = 123
BATCH_SIZE = 2000
TRANSLATION = np.ones(DIM, dtype=np.float32) * 10.0

GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'

print(f"\n{'='*70}")
print(f"🔬 Weaviate 独立集合测试")
print(f"{'='*70}")

rng = np.random.default_rng(SEED)
ids = np.arange(N, dtype=np.int64)
vecs = rng.random((N, DIM), dtype=np.float32)
probe = vecs[0].copy()

vecs_translated = vecs + TRANSLATION
probe_translated = probe + TRANSLATION

client = weaviate.Client("http://127.0.0.1:8080")
client.schema.delete_all()

# ==================== 方法 1: 两个独立 Class ====================
print(f"\n{'='*70}")
print(f"📦 方法 1: 两个独立 Class（与 Milvus/Qdrant 一致）")
print(f"{'='*70}")

# 创建 Class A: Original
class_original = {
    "class": "TranslationOriginal",
    "vectorizer": "none",
    "vectorIndexConfig": {
    "distance": "l2-squared",
    "ef": 10000,
    "efConstruction": 2048,
    "maxConnections": 512
}
}
client.schema.create_class(class_original)

print(f"📥 插入原始数据到 TranslationOriginal...")
with client.batch as batch:
    batch.batch_size = BATCH_SIZE
    for i in range(N):
        batch.add_data_object({"idx": int(ids[i])}, "TranslationOriginal", vector=vecs[i].tolist())
        if (i + 1) % 10000 == 0:
            print(f"   {i+1}/{N}...")

time.sleep(3)

# 创建 Class B: Translated
class_translated = {
    "class": "TranslationTranslated",
    "vectorizer": "none",
    "properties": [{"name": "idx", "dataType": ["int"]}],
    "vectorIndexConfig": {"distance": "l2-squared","ef": 10000, "efConstruction": 2048, "maxConnections": 512}
}
client.schema.create_class(class_translated)

print(f"📥 插入平移数据到 TranslationTranslated...")
with client.batch as batch:
    batch.batch_size = BATCH_SIZE
    for i in range(N):
        batch.add_data_object({"idx": int(ids[i])}, "TranslationTranslated", vector=vecs_translated[i].tolist())
        if (i + 1) % 10000 == 0:
            print(f"   {i+1}/{N}...")

time.sleep(3)

# 查询原始
print(f"\n🔍 查询 TranslationOriginal...")
res_orig = (client.query
    .get("TranslationOriginal", ["idx", "_additional { distance }"])
    .with_near_vector({"vector": probe.tolist()})
    .with_limit(TOPK).do())
ids_orig = [item['idx'] for item in res_orig['data']['Get']['TranslationOriginal']]
dists_orig = [item['_additional']['distance'] for item in res_orig['data']['Get']['TranslationOriginal']]

# 查询平移
print(f"🔍 查询 TranslationTranslated...")
res_trans = (client.query
    .get("TranslationTranslated", ["idx", "_additional { distance }"])
    .with_near_vector({"vector": probe_translated.tolist()})
    .with_limit(TOPK).do())
ids_trans = [item['idx'] for item in res_trans['data']['Get']['TranslationTranslated']]
dists_trans = [item['_additional']['distance'] for item in res_trans['data']['Get']['TranslationTranslated']]

print(f"\n结果对比（方法 1）:")
print(f"   原始 Top-10 ID   : {ids_orig[:10]}")
print(f"   平移 Top-10 ID   : {ids_trans[:10]}")
print(f"   原始 Top-10 距离 : {[f'{d:.6f}' for d in dists_orig[:10]]}")
print(f"   平移 Top-10 距离 : {[f'{d:.6f}' for d in dists_trans[:10]]}")

id_matches_method1 = sum(1 for a, b in zip(ids_orig, ids_trans) if a == b)
match_rate_method1 = id_matches_method1 / TOPK

if match_rate_method1 == 1.0:
    print(f"   匹配率: {GREEN}{match_rate_method1*100:.0f}% ({id_matches_method1}/{TOPK}) ✅{RESET}")
else:
    print(f"   匹配率: {YELLOW}{match_rate_method1*100:.0f}% ({id_matches_method1}/{TOPK}){RESET}")

# ==================== 方法 2: 同一个 Class + Filter ====================
print(f"\n{'='*70}")
print(f"📦 方法 2: 同一个 Class + Filter（之前的方式）")
print(f"{'='*70}")

client.schema.delete_all()

class_mixed = {
    "class": "TranslationMixed",
    "vectorizer": "none",
    "properties": [
        {"name": "idx", "dataType": ["int"]},
        {"name": "group", "dataType": ["string"]}
    ],
    "vectorIndexConfig": {
        "distance": "l2-squared",
        "ef": 10000, 
        "efConstruction": 2048, 
        "maxConnections": 512
    }
}
client.schema.create_class(class_mixed)

print(f"📥 插入原始数据 (group=original)...")
with client.batch as batch:
    batch.batch_size = BATCH_SIZE
    for i in range(N):
        batch.add_data_object({"idx": int(ids[i]), "group": "original"}, 
                            "TranslationMixed", vector=vecs[i].tolist())
        if (i + 1) % 10000 == 0:
            print(f"   {i+1}/{N}...")

time.sleep(3)

print(f"📥 插入平移数据 (group=translated)...")
with client.batch as batch:
    batch.batch_size = BATCH_SIZE
    for i in range(N):
        batch.add_data_object({"idx": int(ids[i]), "group": "translated"}, 
                            "TranslationMixed", vector=vecs_translated[i].tolist())
        if (i + 1) % 10000 == 0:
            print(f"   {i+1}/{N}...")

time.sleep(3)

# 查询原始
print(f"\n🔍 查询 original...")
res_orig2 = (client.query
    .get("TranslationMixed", ["idx", "_additional { distance }"])
    .with_near_vector({"vector": probe.tolist()})
    .with_where({"path": ["group"], "operator": "Equal", "valueString": "original"})
    .with_limit(TOPK).do())
ids_orig2 = [item['idx'] for item in res_orig2['data']['Get']['TranslationMixed']]
dists_orig2 = [item['_additional']['distance'] for item in res_orig2['data']['Get']['TranslationMixed']]

# 查询平移
print(f"🔍 查询 translated...")
res_trans2 = (client.query
    .get("TranslationMixed", ["idx", "_additional { distance }"])
    .with_near_vector({"vector": probe_translated.tolist()})
    .with_where({"path": ["group"], "operator": "Equal", "valueString": "translated"})
    .with_limit(TOPK).do())
ids_trans2 = [item['idx'] for item in res_trans2['data']['Get']['TranslationMixed']]
dists_trans2 = [item['_additional']['distance'] for item in res_trans2['data']['Get']['TranslationMixed']]

print(f"\n结果对比（方法 2）:")
print(f"   原始 Top-10 ID   : {ids_orig2[:10]}")
print(f"   平移 Top-10 ID   : {ids_trans2[:10]}")
print(f"   原始 Top-10 距离 : {[f'{d:.6f}' for d in dists_orig2[:10]]}")
print(f"   平移 Top-10 距离 : {[f'{d:.6f}' for d in dists_trans2[:10]]}")

id_matches_method2 = sum(1 for a, b in zip(ids_orig2, ids_trans2) if a == b)
match_rate_method2 = id_matches_method2 / TOPK

if match_rate_method2 == 1.0:
    print(f"   匹配率: {GREEN}{match_rate_method2*100:.0f}% ({id_matches_method2}/{TOPK}) ✅{RESET}")
else:
    print(f"   匹配率: {YELLOW}{match_rate_method2*100:.0f}% ({id_matches_method2}/{TOPK}){RESET}")

# ==================== 结论 ====================
print(f"\n{'='*70}")
print(f"📊 结论")
print(f"{'='*70}")

print(f"\n方法对比:")
print(f"  1️⃣  独立 Class: {match_rate_method1*100:.0f}% 匹配")
print(f"  2️⃣  混合 Class:  {match_rate_method2*100:.0f}% 匹配")

if match_rate_method1 > match_rate_method2:
    print(f"\n{GREEN}✅ 验证成功！{RESET}")
    print(f"   独立集合方式匹配率更高")
    print(f"   混合插入会污染 HNSW 图结构")
elif match_rate_method1 == match_rate_method2:
    print(f"\n⚠️  两种方式匹配率相同")
    if match_rate_method1 == 1.0:
        print(f"   可能参数足够高达到近似精确搜索")
    else:
        print(f"   可能 Weaviate 的 filter 实现与其他数据库不同")
else:
    print(f"\n🤔 意外：混合方式匹配率反而更高")

print(f"\n💡 为什么 Milvus/Qdrant/Chroma 都通过了？")
print(f"   1. 使用独立集合（与方法 1 一致）")
print(f"   2. 相同插入顺序 → HNSW 图同构")
print(f"   3. 高 ef 参数 → 召回率接近 100%")
print(f"   4. 随机均匀数据 → 图结构确定性高")

print(f"\n{'='*70}\n")

client.schema.delete_all()
