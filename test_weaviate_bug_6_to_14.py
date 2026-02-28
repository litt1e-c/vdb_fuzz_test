#!/usr/bin/env python3
"""
Weaviate Bug #6-#14 综合探针测试
=================================
验证 fuzzer 开发中发现的其他问题:
  Bug #6:  contains_none 对 NULL/空数组返回 True
  Bug #7:  TEXT 字段比较大小写不敏感
  Bug #8:  停用词导致 TEXT_ARRAY 查询失败
  Bug #9:  Dynamic 向量索引类型创建失败
  Bug #10: L2 距离 HNSW 平移不变性失效
  Bug #11: 未索引字段查询报错 (index_filterable=False)
  Bug #12: 不支持 IN / NOT IN 查询
  Bug #13: inverted_index_config 创建可能失败
  Bug #14: 默认不索引 NULL (需手动 index_null_state)

用法: python test_weaviate_bug_6_to_14.py
"""
import time
import numpy as np
import weaviate
from weaviate.classes.config import Configure, Property, DataType, VectorDistances
from weaviate.classes.query import Filter
from weaviate.classes.data import DataObject

# ─── 工具 ──────────────────────────────────────────────────────
def connect():
    client = weaviate.connect_to_local(host="localhost", port=8080, grpc_port=50051)
    assert client.is_ready(), "Weaviate 未就绪"
    return client

def ids_from(result):
    return {o.properties.get("row_id") for o in result.objects}

def sep(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def result_line(label, expected, actual):
    match = expected == actual
    status = "✅ PASS" if match else "❌ DIFFER"
    print(f"  {label}")
    print(f"    预期: {sorted(expected) if isinstance(expected, set) else expected}")
    print(f"    实际: {sorted(actual) if isinstance(actual, set) else actual}")
    print(f"    -> {status}")
    return match

def safe_delete(client, name):
    try: client.collections.delete(name)
    except: pass


# ═══════════════════════════════════════════════════════════════
#  Bug #6: contains_none 对 NULL/空数组返回 True
# ═══════════════════════════════════════════════════════════════
def test_bug6_contains_none_null(client):
    sep("Bug #6: contains_none 对 NULL/空数组的行为")
    CLASS = "BugTest6"
    safe_delete(client, CLASS)

    client.collections.create(
        name=CLASS,
        properties=[
            Property(name="row_id", data_type=DataType.TEXT),
            Property(name="tags", data_type=DataType.INT_ARRAY),
        ],
        vectorizer_config=Configure.Vectorizer.none(),
        inverted_index_config=Configure.inverted_index(index_null_state=True),
    )
    col = client.collections.get(CLASS)

    # A: [1,2,3]  B: [2,4,5]  C: NULL  D: [6,7]  E: []
    print("\n  📋 测试数据:")
    print("    A: tags=[1,2,3], B: tags=[2,4,5], C: tags=NULL, D: tags=[6,7], E: tags=[]\n")
    col.data.insert_many([
        DataObject(properties={"row_id": "A", "tags": [1, 2, 3]}),
        DataObject(properties={"row_id": "B", "tags": [2, 4, 5]}),
        DataObject(properties={"row_id": "C"}),
        DataObject(properties={"row_id": "D", "tags": [6, 7]}),
        DataObject(properties={"row_id": "E", "tags": []}),
    ])
    time.sleep(0.5)

    results = {}

    # 6a: contains_any([2]) — 包含 2 的行
    r = col.query.fetch_objects(filters=Filter.by_property("tags").contains_any([2]), limit=100)
    results["contains_any_2"] = ids_from(r)

    # 6b: contains_all([2,3]) — 同时包含 2 和 3
    r = col.query.fetch_objects(filters=Filter.by_property("tags").contains_all([2, 3]), limit=100)
    results["contains_all_23"] = ids_from(r)

    # 6c: NOT(contains_any([2])) — 不包含 2 的行
    # SQL 标准: {D} (C=NULL 排除, E=[] 排除或看做空)
    # 如果 Weaviate 二值逻辑: NULL→比较失败→False→NOT→True → 包含 C 和 E
    r = col.query.fetch_objects(filters=Filter.not_(Filter.by_property("tags").contains_any([2])), limit=100)
    results["not_contains_any_2"] = ids_from(r)

    # 6d: 尝试 contains_none (如果 API 支持)
    try:
        # Weaviate Python v4 没有原生 contains_none, 但可能有
        # 等价表达: NOT(contains_any(targets))
        has_contains_none = hasattr(Filter.by_property("tags"), "contains_none")
        print(f"  ℹ️  contains_none API 是否存在: {has_contains_none}")
    except:
        has_contains_none = False

    ok = True
    ok &= result_line("contains_any([2])", {"A", "B"}, results["contains_any_2"])
    ok &= result_line("contains_all([2,3])", {"A"}, results["contains_all_23"])
    ok &= result_line("NOT(contains_any([2])) SQL标准={D}", {"D"}, results["not_contains_any_2"])

    if not ok:
        print(f"\n  📊 分析:")
        extra = results["not_contains_any_2"] - {"D"}
        if extra:
            print(f"    NOT(contains_any([2])) 多出: {sorted(extra)}")
            if "C" in extra or "E" in extra:
                print(f"    ⚠️ NULL/空数组行在 NOT(contains) 中被返回")
                print(f"    → 二值逻辑: NULL→contains失败→False→NOT→True")

    safe_delete(client, CLASS)
    return ok


# ═══════════════════════════════════════════════════════════════
#  Bug #7: TEXT 字段比较大小写不敏感
# ═══════════════════════════════════════════════════════════════
def test_bug7_text_case_insensitive(client):
    sep("Bug #7: TEXT 字段比较是否大小写敏感")
    CLASS = "BugTest7"
    safe_delete(client, CLASS)

    client.collections.create(
        name=CLASS,
        properties=[
            Property(name="row_id", data_type=DataType.TEXT),
            Property(name="name", data_type=DataType.TEXT),
        ],
        vectorizer_config=Configure.Vectorizer.none(),
    )
    col = client.collections.get(CLASS)

    print("\n  📋 测试数据:")
    print('    A: name="Hello", B: name="hello", C: name="HELLO", D: name="World", E: name="hElLo"\n')
    col.data.insert_many([
        DataObject(properties={"row_id": "A", "name": "Hello"}),
        DataObject(properties={"row_id": "B", "name": "hello"}),
        DataObject(properties={"row_id": "C", "name": "HELLO"}),
        DataObject(properties={"row_id": "D", "name": "World"}),
        DataObject(properties={"row_id": "E", "name": "hElLo"}),
    ])
    time.sleep(0.5)

    results = {}

    # 7a: equal("Hello") — 如果 case-sensitive → {A}; 如果 insensitive → {A,B,C,E}
    r = col.query.fetch_objects(filters=Filter.by_property("name").equal("Hello"), limit=100)
    results["eq_Hello"] = ids_from(r)

    # 7b: equal("hello")
    r = col.query.fetch_objects(filters=Filter.by_property("name").equal("hello"), limit=100)
    results["eq_hello"] = ids_from(r)

    # 7c: not_equal("Hello") — case-sensitive → {B,C,D,E}; insensitive → {D}
    r = col.query.fetch_objects(filters=Filter.by_property("name").not_equal("Hello"), limit=100)
    results["neq_Hello"] = ids_from(r)

    # 7d: like("hel*") — Weaviate like 通配符
    r = col.query.fetch_objects(filters=Filter.by_property("name").like("hel*"), limit=100)
    results["like_hel"] = ids_from(r)

    # 7e: like("HEL*")
    r = col.query.fetch_objects(filters=Filter.by_property("name").like("HEL*"), limit=100)
    results["like_HEL"] = ids_from(r)

    # 7f: greater_than 字符串比较
    r = col.query.fetch_objects(filters=Filter.by_property("name").greater_than("Hello"), limit=100)
    results["gt_Hello"] = ids_from(r)

    ok = True
    ok &= result_line(
        'equal("Hello") case-sensitive预期={A}',
        {"A"}, results["eq_Hello"]
    )
    ok &= result_line(
        'equal("hello") case-sensitive预期={B}',
        {"B"}, results["eq_hello"]
    )
    ok &= result_line(
        'not_equal("Hello") case-sensitive预期={B,C,D,E}',
        {"B", "C", "D", "E"}, results["neq_Hello"]
    )

    print(f"\n  📊 LIKE 操作:")
    print(f"    like('hel*'):  {sorted(results['like_hel'])}")
    print(f"    like('HEL*'):  {sorted(results['like_HEL'])}")
    print(f"    → LIKE 是否大小写无关: {'是' if results['like_hel'] == results['like_HEL'] else '否'}")

    print(f"\n  📊 字符串比较 (greater_than):")
    print(f"    name > 'Hello': {sorted(results['gt_Hello'])}")

    if results["eq_Hello"] == {"A", "B", "C", "E"}:
        print(f"\n  📊 结论:")
        print(f"    ⚠️ equal('Hello') 匹配了所有大小写变体 → TEXT 比较完全大小写不敏感")
        print(f"    → 这与大多数数据库 (MySQL utf8_bin, PostgreSQL) 默认行为不同")
        print(f"    → 但与 MySQL utf8_general_ci / SQLite 默认行为一致")
    elif results["eq_Hello"] == {"A"}:
        print(f"\n  📊 结论: TEXT 比较是大小写敏感的 (与 SQL 默认一致)")

    safe_delete(client, CLASS)
    return ok


# ═══════════════════════════════════════════════════════════════
#  Bug #8: 停用词导致 TEXT_ARRAY 查询失败
# ═══════════════════════════════════════════════════════════════
def test_bug8_stopwords(client):
    sep("Bug #8: 停用词 (Stopwords) 对 TEXT/TEXT_ARRAY 查询的影响")
    CLASS = "BugTest8"
    safe_delete(client, CLASS)

    client.collections.create(
        name=CLASS,
        properties=[
            Property(name="row_id", data_type=DataType.TEXT),
            Property(name="words", data_type=DataType.TEXT_ARRAY),
            Property(name="title", data_type=DataType.TEXT),
        ],
        vectorizer_config=Configure.Vectorizer.none(),
    )
    col = client.collections.get(CLASS)

    print("\n  📋 测试数据:")
    print('    A: words=["the","cat","sat"], title="the big cat"')
    print('    B: words=["a","dog","ran"],   title="a small dog"')
    print('    C: words=["hello","world"],   title="hello world"\n')
    col.data.insert_many([
        DataObject(properties={"row_id": "A", "words": ["the", "cat", "sat"], "title": "the big cat"}),
        DataObject(properties={"row_id": "B", "words": ["a", "dog", "ran"], "title": "a small dog"}),
        DataObject(properties={"row_id": "C", "words": ["hello", "world"], "title": "hello world"}),
    ])
    time.sleep(0.5)

    # 测试常见停用词
    stopwords_to_test = ["the", "a", "is", "and", "of", "in", "to"]
    normal_words = ["cat", "dog", "hello", "world"]

    print(f"  📋 TEXT_ARRAY contains_any 测试:")
    print(f"  {'词语':<12} {'类型':<10} {'结果':<20} {'状态':<8}")
    print(f"  {'─'*12} {'─'*10} {'─'*20} {'─'*8}")

    stopword_failures = 0
    for word in stopwords_to_test:
        try:
            r = col.query.fetch_objects(
                filters=Filter.by_property("words").contains_any([word]), limit=100)
            ids = ids_from(r)
            print(f"  {word:<12} {'停用词':<10} {str(sorted(ids)):<20} {'⚠️ 空' if not ids else '✅'}")
            if not ids:
                stopword_failures += 1
        except Exception as e:
            err = str(e)[:60]
            print(f"  {word:<12} {'停用词':<10} {'ERROR':<20} ❌ {err}")
            stopword_failures += 1

    for word in normal_words:
        try:
            r = col.query.fetch_objects(
                filters=Filter.by_property("words").contains_any([word]), limit=100)
            ids = ids_from(r)
            print(f"  {word:<12} {'普通词':<10} {str(sorted(ids)):<20} {'✅'}")
        except Exception as e:
            print(f"  {word:<12} {'普通词':<10} {'ERROR':<20} ❌")

    # TEXT like 停用词测试
    print(f"\n  📋 TEXT like 停用词测试:")
    like_tests = [
        ("the*", "停用词前缀"),
        ("a*", "停用词"),
        ("cat*", "普通词前缀"),
    ]
    like_failures = 0
    for pat, desc in like_tests:
        try:
            r = col.query.fetch_objects(
                filters=Filter.by_property("title").like(pat), limit=100)
            ids = ids_from(r)
            print(f"    like('{pat}'): {sorted(ids)} ({desc})")
        except Exception as e:
            err_msg = str(e)
            if "stopword" in err_msg.lower():
                print(f"    like('{pat}'): ❌ 停用词报错 ({desc})")
                like_failures += 1
            else:
                print(f"    like('{pat}'): ❌ {str(e)[:60]} ({desc})")
                like_failures += 1

    # 纯停用词 like 测试 (容易触发 "only stopwords provided" 错误)
    print(f"\n  📋 纯停用词搜索:")
    pure_stopword_tests = ["the", "is", "and"]
    for sw in pure_stopword_tests:
        try:
            r = col.query.fetch_objects(
                filters=Filter.by_property("title").like(sw), limit=100)
            print(f"    like('{sw}'): {sorted(ids_from(r))}")
        except Exception as e:
            err_msg = str(e)
            if "stopword" in err_msg.lower():
                print(f"    like('{sw}'): ❌ 报错: 'only stopwords provided'")
            else:
                print(f"    like('{sw}'): ❌ {err_msg[:80]}")

    ok = stopword_failures == 0 and like_failures == 0
    print(f"\n  📊 结论:")
    if stopword_failures:
        print(f"    ⚠️ {stopword_failures}/{len(stopwords_to_test)} 个停用词在 TEXT_ARRAY 查询中失败/返回空")
        print(f"    → Weaviate 内置英文停用词过滤, 这些词不被索引")
        print(f"    → 影响: 包含常见英文词的数组字段查询不可靠")
    else:
        print(f"    停用词在 contains_any 中正常工作")

    safe_delete(client, CLASS)
    return ok


# ═══════════════════════════════════════════════════════════════
#  Bug #9: Dynamic 向量索引类型创建失败
# ═══════════════════════════════════════════════════════════════
def test_bug9_dynamic_index(client):
    sep("Bug #9: Dynamic 向量索引类型是否可用")
    CLASS = "BugTest9"

    index_types = {
        "HNSW": lambda: Configure.VectorIndex.hnsw(
            distance_metric=VectorDistances.COSINE),
        "FLAT": lambda: Configure.VectorIndex.flat(
            distance_metric=VectorDistances.COSINE),
        "DYNAMIC": lambda: Configure.VectorIndex.dynamic(
            distance_metric=VectorDistances.COSINE, threshold=10000),
    }

    results = {}
    for name, make_config in index_types.items():
        safe_delete(client, CLASS)
        try:
            client.collections.create(
                name=CLASS,
                properties=[Property(name="val", data_type=DataType.INT)],
                vectorizer_config=Configure.Vectorizer.none(),
                vector_index_config=make_config(),
            )
            # 尝试插入数据验证可用性
            col = client.collections.get(CLASS)
            col.data.insert(properties={"val": 1}, vector=[0.1] * 128)
            r = col.query.fetch_objects(limit=1)
            results[name] = "✅ 可用" if r.objects else "⚠️ 创建成功但查询异常"
        except Exception as e:
            err = str(e)[:100]
            results[name] = f"❌ 失败: {err}"
        finally:
            safe_delete(client, CLASS)

    print("\n  📋 向量索引类型测试:")
    ok = True
    for name, status in results.items():
        print(f"    {name:<10} {status}")
        if "❌" in status:
            ok = False

    if not ok:
        print(f"\n  📊 分析:")
        print(f"    ⚠️ Dynamic 索引可能需要特定版本或配置")
        print(f"    → 生产中建议使用 HNSW (最稳定)")

    return ok


# ═══════════════════════════════════════════════════════════════
#  Bug #10: L2 距离 HNSW 平移不变性
# ═══════════════════════════════════════════════════════════════
def test_bug10_l2_translation_invariance(client):
    sep("Bug #10: L2 距离 HNSW 搜索平移不变性")
    CLASS = "BugTest10"
    safe_delete(client, CLASS)

    DIM = 128
    N = 500  # 数据量 (小规模便于快速测试)
    TOPK = 20

    client.collections.create(
        name=CLASS,
        properties=[Property(name="idx", data_type=DataType.INT)],
        vectorizer_config=Configure.Vectorizer.none(),
        vector_index_config=Configure.VectorIndex.hnsw(
            distance_metric=VectorDistances.L2_SQUARED,
            ef_construction=256,
            max_connections=64,
        ),
    )
    col = client.collections.get(CLASS)

    # 生成随机数据
    np.random.seed(42)
    vectors = np.random.randn(N, DIM).astype(np.float32)
    query = np.random.randn(DIM).astype(np.float32)
    offset = np.random.randn(DIM).astype(np.float32) * 100  # 大偏移

    print(f"\n  📋 设置: {N} 个向量, dim={DIM}, topk={TOPK}")
    print(f"    距离: L2_SQUARED, 偏移量: ||offset||={np.linalg.norm(offset):.1f}")

    # --- 第一轮: 原始数据搜索 ---
    objs = [DataObject(properties={"idx": i}, vector=vectors[i].tolist()) for i in range(N)]
    col.data.insert_many(objs)
    time.sleep(0.5)

    r1 = col.query.near_vector(near_vector=query.tolist(), limit=TOPK, return_properties=["idx"])
    ids_original = [o.properties["idx"] for o in r1.objects]

    # --- 第二轮: 平移后数据搜索 ---
    safe_delete(client, CLASS)
    client.collections.create(
        name=CLASS,
        properties=[Property(name="idx", data_type=DataType.INT)],
        vectorizer_config=Configure.Vectorizer.none(),
        vector_index_config=Configure.VectorIndex.hnsw(
            distance_metric=VectorDistances.L2_SQUARED,
            ef_construction=256,
            max_connections=64,
        ),
    )
    col = client.collections.get(CLASS)

    shifted_vectors = vectors + offset
    shifted_query = query + offset
    objs2 = [DataObject(properties={"idx": i}, vector=shifted_vectors[i].tolist()) for i in range(N)]
    col.data.insert_many(objs2)
    time.sleep(0.5)

    r2 = col.query.near_vector(near_vector=shifted_query.tolist(), limit=TOPK, return_properties=["idx"])
    ids_shifted = [o.properties["idx"] for o in r2.objects]

    # --- 第三轮: NumPy 暴力计算 (ground truth) ---
    dists = np.sum((vectors - query) ** 2, axis=1)
    gt_ids = np.argsort(dists)[:TOPK].tolist()

    # 对比
    set_orig = set(ids_original)
    set_shifted = set(ids_shifted)
    set_gt = set(gt_ids)

    overlap_orig_shifted = len(set_orig & set_shifted)
    overlap_orig_gt = len(set_orig & set_gt)
    overlap_shifted_gt = len(set_shifted & set_gt)

    print(f"\n  📊 Top-{TOPK} 结果对比:")
    print(f"    原始 vs 平移后:  {overlap_orig_shifted}/{TOPK} 重合 ({overlap_orig_shifted/TOPK*100:.0f}%)")
    print(f"    原始 vs Ground Truth: {overlap_orig_gt}/{TOPK} ({overlap_orig_gt/TOPK*100:.0f}%)")
    print(f"    平移后 vs Ground Truth: {overlap_shifted_gt}/{TOPK} ({overlap_shifted_gt/TOPK*100:.0f}%)")

    # L2 理论: d(x+t, y+t) = d(x,y), 所以 top-k 应该完全一致
    ok = overlap_orig_shifted == TOPK
    if not ok:
        print(f"\n    ⚠️ 平移前后 Top-{TOPK} 不完全一致!")
        print(f"    → L2距离理论上平移不变: d(x+t, y+t) = d(x, y)")
        print(f"    → 但 HNSW 图结构因数据分布变化导致近似搜索结果不同")
        diff = set_orig.symmetric_difference(set_shifted)
        print(f"    → 差异 ID: {sorted(diff)[:10]}{'...' if len(diff) > 10 else ''}")
    else:
        print(f"\n    ✅ 平移前后 Top-{TOPK} 完全一致 (在此数据规模下)")

    safe_delete(client, CLASS)
    return ok


# ═══════════════════════════════════════════════════════════════
#  Bug #11: 未索引字段查询报错
# ═══════════════════════════════════════════════════════════════
def test_bug11_unindexed_field_query(client):
    sep("Bug #11: 未索引字段 (index_filterable=False) 查询行为")
    CLASS = "BugTest11"
    safe_delete(client, CLASS)

    client.collections.create(
        name=CLASS,
        properties=[
            Property(name="row_id", data_type=DataType.TEXT, index_filterable=True),
            Property(name="indexed_val", data_type=DataType.INT, index_filterable=True),
            Property(name="unindexed_val", data_type=DataType.INT, index_filterable=False),
            Property(name="no_search_text", data_type=DataType.TEXT,
                     index_filterable=True, index_searchable=False),
        ],
        vectorizer_config=Configure.Vectorizer.none(),
        inverted_index_config=Configure.inverted_index(index_null_state=True),
    )
    col = client.collections.get(CLASS)

    print("\n  📋 测试数据:")
    print("    A: indexed=10, unindexed=100")
    print("    B: indexed=20, unindexed=200")
    print("    字段配置: indexed_val(filterable=True), unindexed_val(filterable=False)")
    print("              no_search_text(filterable=True, searchable=False)\n")
    col.data.insert_many([
        DataObject(properties={"row_id": "A", "indexed_val": 10, "unindexed_val": 100,
                               "no_search_text": "hello world"}),
        DataObject(properties={"row_id": "B", "indexed_val": 20, "unindexed_val": 200,
                               "no_search_text": "foo bar"}),
    ])
    time.sleep(0.5)

    tests = [
        ("indexed_val == 10 (有索引)", lambda: Filter.by_property("indexed_val").equal(10)),
        ("unindexed_val == 100 (无索引)", lambda: Filter.by_property("unindexed_val").equal(100)),
        ("unindexed_val > 50 (无索引 范围)", lambda: Filter.by_property("unindexed_val").greater_than(50)),
        ("unindexed_val is null (无索引 null)", lambda: Filter.by_property("unindexed_val").is_none(True)),
        ("no_search_text like 'hel*' (无搜索索引)", lambda: Filter.by_property("no_search_text").like("hel*")),
    ]

    print(f"  📋 查询测试:")
    print(f"  {'查询':<40} {'结果':<15} {'状态'}")
    print(f"  {'─'*40} {'─'*15} {'─'*10}")

    any_error = False
    for desc, make_filter in tests:
        try:
            r = col.query.fetch_objects(filters=make_filter(), limit=100)
            ids = ids_from(r)
            print(f"  {desc:<40} {str(sorted(ids)):<15} ✅")
        except Exception as e:
            err = str(e)
            any_error = True
            if "inverted index" in err.lower() or "indexed" in err.lower() or "not found" in err.lower():
                print(f"  {desc:<40} {'索引报错':<15} ❌ 需要索引")
            else:
                print(f"  {desc:<40} {'ERROR':<15} ❌ {err[:50]}")

    ok = not any_error
    if any_error:
        print(f"\n  📊 分析:")
        print(f"    ⚠️ 未索引字段查询直接报错, 而非返回空结果或全扫描")
        print(f"    → 生产中必须提前规划哪些字段需要过滤")
        print(f"    → 对比: Milvus/Qdrant 在查询时自动处理无索引字段 (性能差但不报错)")

    safe_delete(client, CLASS)
    return ok


# ═══════════════════════════════════════════════════════════════
#  Bug #12: 不支持 IN / NOT IN 查询
# ═══════════════════════════════════════════════════════════════
def test_bug12_in_not_in(client):
    sep("Bug #12: IN / NOT IN 查询支持")
    CLASS = "BugTest12"
    safe_delete(client, CLASS)

    client.collections.create(
        name=CLASS,
        properties=[
            Property(name="row_id", data_type=DataType.TEXT),
            Property(name="status", data_type=DataType.INT),
        ],
        vectorizer_config=Configure.Vectorizer.none(),
    )
    col = client.collections.get(CLASS)

    print("\n  📋 测试数据:")
    print("    A: status=1, B: status=2, C: status=3, D: status=4, E: status=5\n")
    col.data.insert_many([
        DataObject(properties={"row_id": "A", "status": 1}),
        DataObject(properties={"row_id": "B", "status": 2}),
        DataObject(properties={"row_id": "C", "status": 3}),
        DataObject(properties={"row_id": "D", "status": 4}),
        DataObject(properties={"row_id": "E", "status": 5}),
    ])
    time.sleep(0.5)

    print(f"  📋 IN 查询测试:")

    # 方法 1: 检查 API 是否有原生 IN
    has_contains_any = hasattr(Filter.by_property("status"), "contains_any")

    # 方法 2: 用 contains_any 模拟 (这是数组操作, 对标量可能不行)
    approaches = []

    # 2a: contains_any 尝试
    try:
        r = col.query.fetch_objects(
            filters=Filter.by_property("status").contains_any([1, 3, 5]), limit=100)
        ids = ids_from(r)
        approaches.append(("contains_any([1,3,5])", ids, None))
        print(f"    contains_any([1,3,5]): {sorted(ids)} {'✅' if ids == {'A','C','E'} else '⚠️'}")
    except Exception as e:
        approaches.append(("contains_any", set(), str(e)))
        print(f"    contains_any([1,3,5]): ❌ {str(e)[:80]}")

    # 2b: 手动 OR 链
    try:
        f = (Filter.by_property("status").equal(1) |
             Filter.by_property("status").equal(3) |
             Filter.by_property("status").equal(5))
        r = col.query.fetch_objects(filters=f, limit=100)
        ids = ids_from(r)
        approaches.append(("OR链 (eq(1)|eq(3)|eq(5))", ids, None))
        print(f"    OR链 (eq(1)|eq(3)|eq(5)): {sorted(ids)} {'✅' if ids == {'A','C','E'} else '⚠️'}")
    except Exception as e:
        approaches.append(("OR链", set(), str(e)))
        print(f"    OR链: ❌ {str(e)[:80]}")

    # 2c: NOT IN — NOT(OR链)
    try:
        f = Filter.not_(
            Filter.by_property("status").equal(1) |
            Filter.by_property("status").equal(3) |
            Filter.by_property("status").equal(5))
        r = col.query.fetch_objects(filters=f, limit=100)
        ids = ids_from(r)
        approaches.append(("NOT(OR链) = NOT IN", ids, None))
        print(f"    NOT(OR链) = NOT IN: {sorted(ids)} {'✅' if ids == {'B','D'} else '⚠️'}")
    except Exception as e:
        approaches.append(("NOT IN", set(), str(e)))
        print(f"    NOT(OR链): ❌ {str(e)[:80]}")

    # 检查是否有原生 "in_" 或 "within" 方法
    filter_methods = [m for m in dir(Filter.by_property("status"))
                      if not m.startswith("_") and callable(getattr(Filter.by_property("status"), m, None))]
    print(f"\n  ℹ️  Filter 可用方法: {filter_methods}")

    has_native_in = any("in" in m.lower() and m not in ["contains_any", "contains_all"]
                        for m in filter_methods)

    print(f"\n  📊 结论:")
    if has_native_in:
        print(f"    ✅ 存在原生 IN 操作符")
    else:
        native_in_missing = True
        contains_any_works = any(a[0].startswith("contains_any") and a[1] == {"A", "C", "E"} for a in approaches)
        or_works = any("OR链" in a[0] and a[1] == {"A", "C", "E"} for a in approaches)
        print(f"    ⚠️ 无原生 IN / NOT IN 操作符")
        print(f"    → contains_any 对标量: {'可用 ✅' if contains_any_works else '不可用 ❌'}")
        print(f"    → OR 链模拟 IN:        {'可用 ✅' if or_works else '不可用 ❌'}")
        if or_works:
            print(f"    → 变通: 可用 OR 链模拟, 但 N 大时表达式膨胀")

    ok = any(a[1] == {"A", "C", "E"} for a in approaches)
    safe_delete(client, CLASS)
    return ok


# ═══════════════════════════════════════════════════════════════
#  Bug #13: inverted_index_config 创建可能失败
# ═══════════════════════════════════════════════════════════════
def test_bug13_inverted_index_config(client):
    sep("Bug #13: inverted_index_config 各种配置组合")
    CLASS = "BugTest13"

    configs = [
        ("无 inverted_index_config", None),
        ("index_null_state=True", Configure.inverted_index(index_null_state=True)),
        ("index_null_state=True + index_property_length=True",
         Configure.inverted_index(index_null_state=True, index_property_length=True)),
        ("index_null_state=False",
         Configure.inverted_index(index_null_state=False)),
        ("index_timestamps=True",
         Configure.inverted_index(index_timestamps=True)),
    ]

    print("\n  📋 inverted_index_config 组合测试:")
    all_ok = True
    for desc, config in configs:
        safe_delete(client, CLASS)
        try:
            kwargs = dict(
                name=CLASS,
                properties=[
                    Property(name="val", data_type=DataType.INT),
                    Property(name="txt", data_type=DataType.TEXT),
                ],
                vectorizer_config=Configure.Vectorizer.none(),
            )
            if config is not None:
                kwargs["inverted_index_config"] = config
            client.collections.create(**kwargs)

            # 验证 null 查询是否可用
            col = client.collections.get(CLASS)
            col.data.insert(properties={"val": 42, "txt": "hello"})
            col.data.insert(properties={"txt": "world"})  # val=NULL
            time.sleep(0.3)

            null_works = True
            try:
                r = col.query.fetch_objects(
                    filters=Filter.by_property("val").is_none(True), limit=10)
                null_count = len(r.objects)
            except Exception:
                null_works = False
                null_count = "ERROR"

            print(f"    {desc:<52} ✅ 创建成功 | is_none: {'可用('+str(null_count)+'行)' if null_works else '❌不可用'}")
            if not null_works and config is not None and "null_state=True" in desc:
                all_ok = False
        except Exception as e:
            print(f"    {desc:<52} ❌ 失败: {str(e)[:60]}")
            all_ok = False
        finally:
            safe_delete(client, CLASS)

    print(f"\n  📊 结论:")
    if all_ok:
        print(f"    所有配置组合均可创建")
    else:
        print(f"    ⚠️ 部分配置组合创建失败")
    print(f"    → 注意: 不设 index_null_state=True 时 is_none 查询不可用")

    return all_ok


# ═══════════════════════════════════════════════════════════════
#  Bug #14: 默认不索引 NULL
# ═══════════════════════════════════════════════════════════════
def test_bug14_default_null_index(client):
    sep("Bug #14: 默认配置下 NULL 查询是否可用")
    CLASS = "BugTest14"
    safe_delete(client, CLASS)

    # 不设 inverted_index_config (默认配置)
    client.collections.create(
        name=CLASS,
        properties=[
            Property(name="row_id", data_type=DataType.TEXT),
            Property(name="score", data_type=DataType.INT),
        ],
        vectorizer_config=Configure.Vectorizer.none(),
    )
    col = client.collections.get(CLASS)

    print("\n  📋 测试: 默认配置 (无 index_null_state) 下 is_none 是否可用")
    print("    数据: A=10, B=NULL\n")
    col.data.insert_many([
        DataObject(properties={"row_id": "A", "score": 10}),
        DataObject(properties={"row_id": "B"}),  # NULL
    ])
    time.sleep(0.5)

    results = {}

    # 14a: 普通查询 (应该正常)
    try:
        r = col.query.fetch_objects(filters=Filter.by_property("score").equal(10), limit=100)
        results["equal_10"] = ("OK", ids_from(r))
        print(f"    score==10: {sorted(ids_from(r))} ✅")
    except Exception as e:
        results["equal_10"] = ("ERROR", str(e))
        print(f"    score==10: ❌ {str(e)[:60]}")

    # 14b: is_none(True) (可能不可用)
    try:
        r = col.query.fetch_objects(filters=Filter.by_property("score").is_none(True), limit=100)
        results["is_none_true"] = ("OK", ids_from(r))
        print(f"    is_none(True): {sorted(ids_from(r))} ✅")
    except Exception as e:
        results["is_none_true"] = ("ERROR", str(e)[:80])
        print(f"    is_none(True): ❌ {str(e)[:80]}")

    # 14c: is_none(False)
    try:
        r = col.query.fetch_objects(filters=Filter.by_property("score").is_none(False), limit=100)
        results["is_none_false"] = ("OK", ids_from(r))
        print(f"    is_none(False): {sorted(ids_from(r))} ✅")
    except Exception as e:
        results["is_none_false"] = ("ERROR", str(e)[:80])
        print(f"    is_none(False): ❌ {str(e)[:80]}")

    ok = all(v[0] == "OK" for v in results.values())
    print(f"\n  📊 结论:")
    if ok:
        print(f"    ✅ 默认配置下 is_none 可用 (可能版本已修复)")
    else:
        null_error = results.get("is_none_true", ("", ""))[0] == "ERROR"
        if null_error:
            print(f"    ⚠️ 默认配置下 is_none 不可用!")
            print(f"    → 必须设置 index_null_state=True 才能使用 null 过滤")
            print(f"    → 这是 Weaviate 的设计: null 索引默认关闭以节省存储")
        else:
            print(f"    部分 null 查询失败, 请查看详细输出")

    safe_delete(client, CLASS)
    return ok


# ═══════════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════════
def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  Weaviate Bug #6-#14 综合探针测试                               ║")
    print("║  验证 fuzzer 开发中发现的更多问题                                ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    client = connect()
    print(f"✅ 已连接 Weaviate")

    try:
        summary = {}
        summary["#6  contains与NULL"]       = test_bug6_contains_none_null(client)
        summary["#7  TEXT大小写"]            = test_bug7_text_case_insensitive(client)
        summary["#8  停用词"]               = test_bug8_stopwords(client)
        summary["#9  Dynamic索引"]          = test_bug9_dynamic_index(client)
        summary["#10 L2平移不变性"]          = test_bug10_l2_translation_invariance(client)
        summary["#11 未索引字段查询"]         = test_bug11_unindexed_field_query(client)
        summary["#12 IN/NOT IN"]            = test_bug12_in_not_in(client)
        summary["#13 inverted_index配置"]    = test_bug13_inverted_index_config(client)
        summary["#14 默认NULL索引"]          = test_bug14_default_null_index(client)

        sep("最终汇总")
        for name, passed in summary.items():
            status = "✅ 符合预期/可用" if passed else "⚠️ 存在问题/与预期不同"
            print(f"  {name:<24} {status}")

        issues = sum(1 for v in summary.values() if not v)
        print(f"\n  📊 {issues}/{len(summary)} 项存在问题或与预期不同")
        print(f"  → 结合 Bug #1-#5 共计 {issues + 5} 项发现")

    finally:
        client.close()
        print("\n🔌 连接已关闭")


if __name__ == "__main__":
    main()
