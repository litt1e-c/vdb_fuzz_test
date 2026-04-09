"""
Weaviate Null 语义探针测试
目标: 精确验证以下假设是否成立:
  1. not_equal 是否包含 null 行?
  2. NOT(equal) 是否包含 null 行?
  3. NOT(greater_than) 是否包含 null 行?
  4. 空数组 [] 是否被视为 null?
  5. contains_none 是否包含 null/空数组行?

方法: 插入已知数据, 逐条断言, 输出明确结论.
"""
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import Filter
from weaviate.classes.data import DataObject
import sys
import json

HOST = "127.0.0.1"
PORT = 8080
CLASS = "NullSemanticsProbe"

# ── 测试数据 (6行, 每行语义明确) ──
# id | int_val | text_val | int_arr        | description
# A  | 10      | "hello"  | [1,2,3]        | normal row
# B  | 20      | "world"  | [4,5]          | normal row
# C  | None    | None     | None           | all null
# D  | 10      | "hello"  | []             | empty array
# E  | None    | "test"   | [1]            | int null, text present
# F  | 30      | None     | [2,3]          | text null

ROWS = [
    {"_id": "00000000-0000-0000-0000-00000000000a", "tag": "A", "int_val": 10, "text_val": "hello", "int_arr": [1, 2, 3]},
    {"_id": "00000000-0000-0000-0000-00000000000b", "tag": "B", "int_val": 20, "text_val": "world", "int_arr": [4, 5]},
    {"_id": "00000000-0000-0000-0000-00000000000c", "tag": "C", "int_val": None, "text_val": None, "int_arr": None},
    {"_id": "00000000-0000-0000-0000-00000000000d", "tag": "D", "int_val": 10, "text_val": "hello", "int_arr": []},
    {"_id": "00000000-0000-0000-0000-00000000000e", "tag": "E", "int_val": None, "text_val": "test", "int_arr": [1]},
    {"_id": "00000000-0000-0000-0000-00000000000f", "tag": "F", "int_val": 30, "text_val": None, "int_arr": [2, 3]},
]

def tags(objects):
    """从结果中提取 tag 集合"""
    return set(o.properties.get("tag", "?") for o in objects)

def run_test(name, filter_obj, expected_tags, col):
    """执行单个测试并返回 pass/fail"""
    res = col.query.fetch_objects(filters=filter_obj, limit=100)
    actual = tags(res.objects)
    ok = actual == expected_tags
    status = "✅ PASS" if ok else "❌ FAIL"
    print(f"  {status} | {name}")
    print(f"         Expected: {sorted(expected_tags)}")
    print(f"         Actual:   {sorted(actual)}")
    if not ok:
        missing = expected_tags - actual
        extra = actual - expected_tags
        if missing: print(f"         Missing: {sorted(missing)}")
        if extra: print(f"         Extra:   {sorted(extra)}")
    return ok


def main():
    print("=" * 70)
    print("Weaviate Null 语义探针测试")
    print("=" * 70)

    client = weaviate.connect_to_local(host=HOST, port=PORT, grpc_port=50051)
    try:
        # ── Setup ──
        try:
            client.collections.delete(CLASS)
        except:
            pass

        col = client.collections.create(
            name=CLASS,
            properties=[
                Property(name="tag", data_type=DataType.TEXT),
                Property(name="int_val", data_type=DataType.INT),
                Property(name="text_val", data_type=DataType.TEXT),
                Property(name="int_arr", data_type=DataType.INT_ARRAY),
            ],
            vectorizer_config=Configure.Vectorizer.none(),
            inverted_index_config=Configure.inverted_index(index_null_state=True),
        )

        objs = []
        for r in ROWS:
            props = {k: v for k, v in r.items() if k != "_id" and v is not None}
            objs.append(DataObject(uuid=r["_id"], properties=props))
        col.data.insert_many(objs)
        print(f"\n📊 Inserted {len(ROWS)} rows: A(10,hello,[1,2,3]) B(20,world,[4,5]) C(null,null,null) D(10,hello,[]) E(null,test,[1]) F(30,null,[2,3])")

        results = []

        # ════════════════════════════════════════
        # 测试组 1: not_equal 与 null
        # ════════════════════════════════════════
        print("\n── 测试组 1: not_equal 与 null ──")

        # 1a: int_val != 10 → SQL逻辑: {B,F} (null排除) vs Weaviate可能: {B,C,E,F} (null包含)
        results.append(("1a_ne_int_excl_null",
            run_test("int_val != 10 (测试null是否被包含)",
                Filter.by_property("int_val").not_equal(10),
                {"B", "F"},  # 先假设SQL语义
                col)))

        # 如果上面 FAIL, 重测看是否包含null
        res_1a = col.query.fetch_objects(filters=Filter.by_property("int_val").not_equal(10), limit=100)
        actual_1a = tags(res_1a.objects)
        if "C" in actual_1a or "E" in actual_1a:
            print(f"  ⚠️  not_equal(int) 实际返回了 null 行: {sorted(actual_1a)}")
            results.append(("1a_ne_int_incl_null",
                run_test("int_val != 10 (验证: null行被包含)",
                    Filter.by_property("int_val").not_equal(10),
                    {"B", "C", "E", "F"},  # null包含语义
                    col)))

        # 1b: text_val != "hello" → SQL: {B,E} vs 可能: {B,C,E,F}
        results.append(("1b_ne_text_excl_null",
            run_test("text_val != 'hello' (测试null是否被包含)",
                Filter.by_property("text_val").not_equal("hello"),
                {"B", "E"},  # SQL语义
                col)))

        res_1b = col.query.fetch_objects(filters=Filter.by_property("text_val").not_equal("hello"), limit=100)
        actual_1b = tags(res_1b.objects)
        if "C" in actual_1b or "F" in actual_1b:
            print(f"  ⚠️  not_equal(text) 实际返回了 null 行: {sorted(actual_1b)}")
            results.append(("1b_ne_text_incl_null",
                run_test("text_val != 'hello' (验证: null行被包含)",
                    Filter.by_property("text_val").not_equal("hello"),
                    {"B", "C", "E", "F"},  # null包含语义
                    col)))

        # ════════════════════════════════════════
        # 测试组 2: NOT(equal) 与 null
        # ════════════════════════════════════════
        print("\n── 测试组 2: NOT(equal) 与 null ──")

        # 2a: NOT(int_val == 10) → SQL: {B,F} vs 可能: {B,C,E,F}
        results.append(("2a_not_eq_int_excl",
            run_test("NOT(int_val == 10) (测试null是否被包含)",
                Filter.not_(Filter.by_property("int_val").equal(10)),
                {"B", "F"},  # SQL语义
                col)))

        res_2a = col.query.fetch_objects(filters=Filter.not_(Filter.by_property("int_val").equal(10)), limit=100)
        actual_2a = tags(res_2a.objects)
        if "C" in actual_2a or "E" in actual_2a:
            print(f"  ⚠️  NOT(equal(int)) 实际返回了 null 行: {sorted(actual_2a)}")
            results.append(("2a_not_eq_int_incl",
                run_test("NOT(int_val == 10) (验证: null行被包含)",
                    Filter.not_(Filter.by_property("int_val").equal(10)),
                    {"B", "C", "E", "F"},
                    col)))

        # 2b: NOT(text_val == "hello")
        results.append(("2b_not_eq_text_excl",
            run_test("NOT(text_val == 'hello') (测试null是否被包含)",
                Filter.not_(Filter.by_property("text_val").equal("hello")),
                {"B", "E"},  # SQL语义
                col)))

        res_2b = col.query.fetch_objects(filters=Filter.not_(Filter.by_property("text_val").equal("hello")), limit=100)
        actual_2b = tags(res_2b.objects)
        if "C" in actual_2b or "F" in actual_2b:
            print(f"  ⚠️  NOT(equal(text)) 实际返回了 null 行: {sorted(actual_2b)}")
            results.append(("2b_not_eq_text_incl",
                run_test("NOT(text_val == 'hello') (验证: null行被包含)",
                    Filter.not_(Filter.by_property("text_val").equal("hello")),
                    {"B", "C", "E", "F"},
                    col)))

        # ════════════════════════════════════════
        # 测试组 3: NOT(比较) 与 null
        # ════════════════════════════════════════
        print("\n── 测试组 3: NOT(比较运算符) 与 null ──")

        # 3a: NOT(int_val > 15) → SQL: {A,D} vs 可能: {A,C,D,E}
        results.append(("3a_not_gt_excl",
            run_test("NOT(int_val > 15) (测试null是否被包含)",
                Filter.not_(Filter.by_property("int_val").greater_than(15)),
                {"A", "D"},  # SQL: <=15 的非null行
                col)))

        res_3a = col.query.fetch_objects(filters=Filter.not_(Filter.by_property("int_val").greater_than(15)), limit=100)
        actual_3a = tags(res_3a.objects)
        if "C" in actual_3a or "E" in actual_3a:
            print(f"  ⚠️  NOT(greater_than) 实际返回了 null 行: {sorted(actual_3a)}")
            results.append(("3a_not_gt_incl",
                run_test("NOT(int_val > 15) (验证: null行被包含)",
                    Filter.not_(Filter.by_property("int_val").greater_than(15)),
                    {"A", "C", "D", "E"},
                    col)))

        # 3b: NOT(int_val < 15) → SQL: {B,F} vs 可能: {B,C,E,F}
        results.append(("3b_not_lt_excl",
            run_test("NOT(int_val < 15) (测试null是否被包含)",
                Filter.not_(Filter.by_property("int_val").less_than(15)),
                {"B", "F"},
                col)))

        res_3b = col.query.fetch_objects(filters=Filter.not_(Filter.by_property("int_val").less_than(15)), limit=100)
        actual_3b = tags(res_3b.objects)
        if "C" in actual_3b or "E" in actual_3b:
            print(f"  ⚠️  NOT(less_than) 实际返回了 null 行: {sorted(actual_3b)}")
            results.append(("3b_not_lt_incl",
                run_test("NOT(int_val < 15) (验证: null行被包含)",
                    Filter.not_(Filter.by_property("int_val").less_than(15)),
                    {"B", "C", "E", "F"},
                    col)))

        # 3c: NOT(int_val >= 20) → SQL: {A,D} vs 可能: {A,C,D,E}
        results.append(("3c_not_gte_excl",
            run_test("NOT(int_val >= 20) (测试null是否被包含)",
                Filter.not_(Filter.by_property("int_val").greater_or_equal(20)),
                {"A", "D"},
                col)))

        res_3c = col.query.fetch_objects(filters=Filter.not_(Filter.by_property("int_val").greater_or_equal(20)), limit=100)
        actual_3c = tags(res_3c.objects)
        if "C" in actual_3c or "E" in actual_3c:
            results.append(("3c_not_gte_incl",
                run_test("NOT(int_val >= 20) (验证: null行被包含)",
                    Filter.not_(Filter.by_property("int_val").greater_or_equal(20)),
                    {"A", "C", "D", "E"},
                    col)))

        # ════════════════════════════════════════
        # 测试组 4: 空数组 [] vs null
        # ════════════════════════════════════════
        print("\n── 测试组 4: 空数组 [] vs null ──")

        # 4a: int_arr is_none(True) → 应该只有C (null), 但D (空数组) 是否也算?
        results.append(("4a_arr_null_strict",
            run_test("int_arr is null (测试空数组是否为null)",
                Filter.by_property("int_arr").is_none(True),
                {"C"},  # 严格: 只有显式 null
                col)))

        res_4a = col.query.fetch_objects(filters=Filter.by_property("int_arr").is_none(True), limit=100)
        actual_4a = tags(res_4a.objects)
        if "D" in actual_4a:
            print(f"  ⚠️  is_none(True) 将空数组视为 null: {sorted(actual_4a)}")
            results.append(("4a_arr_null_incl_empty",
                run_test("int_arr is null (验证: 空数组=null)",
                    Filter.by_property("int_arr").is_none(True),
                    {"C", "D"},
                    col)))

        # 4b: int_arr is_none(False) → 应该 {A,B,D,E,F}, 但D可能被排除
        results.append(("4b_arr_notnull_strict",
            run_test("int_arr is not null (测试空数组算非null?)",
                Filter.by_property("int_arr").is_none(False),
                {"A", "B", "D", "E", "F"},  # 严格: 空数组不是null
                col)))

        res_4b = col.query.fetch_objects(filters=Filter.by_property("int_arr").is_none(False), limit=100)
        actual_4b = tags(res_4b.objects)
        if "D" not in actual_4b:
            print(f"  ⚠️  is_none(False) 排除了空数组: {sorted(actual_4b)}")
            results.append(("4b_arr_notnull_excl_empty",
                run_test("int_arr is not null (验证: 空数组被排除)",
                    Filter.by_property("int_arr").is_none(False),
                    {"A", "B", "E", "F"},
                    col)))

        # 4c: contains_any on empty array → D不应匹配
        results.append(("4c_contains_any_1",
            run_test("int_arr contains_any [1] (A和E有1, D空数组不该匹配)",
                Filter.by_property("int_arr").contains_any([1]),
                {"A", "E"},
                col)))

        # ════════════════════════════════════════
        # 测试组 5: contains_none 与 null/空数组
        # ════════════════════════════════════════
        print("\n── 测试组 5: contains_none 与 null/空数组 ──")

        # 5a: int_arr contains_none [1] → 不含1的行: B(4,5), F(2,3)
        #     C(null)和D(空)是否包含?
        results.append(("5a_contains_none_strict",
            run_test("int_arr contains_none [1] (测试null/空数组是否被包含)",
                Filter.by_property("int_arr").contains_none([1]),
                {"B", "F"},  # 严格: 只有有内容且不含1的
                col)))

        res_5a = col.query.fetch_objects(filters=Filter.by_property("int_arr").contains_none([1]), limit=100)
        actual_5a = tags(res_5a.objects)
        null_in = "C" in actual_5a
        empty_in = "D" in actual_5a
        if null_in or empty_in:
            expected = {"B", "F"}
            if null_in: expected.add("C")
            if empty_in: expected.add("D")
            print(f"  ⚠️  contains_none 包含: null={null_in}, 空数组={empty_in}, 实际: {sorted(actual_5a)}")
            results.append(("5a_contains_none_incl",
                run_test("int_arr contains_none [1] (验证实际语义)",
                    Filter.by_property("int_arr").contains_none([1]),
                    expected,
                    col)))

        # 5b: contains_none [1,4] → 不含1也不含4: F(2,3)
        results.append(("5b_contains_none_multi",
            run_test("int_arr contains_none [1,4] (严格: 只有F)",
                Filter.by_property("int_arr").contains_none([1, 4]),
                {"F"},
                col)))

        res_5b = col.query.fetch_objects(filters=Filter.by_property("int_arr").contains_none([1, 4]), limit=100)
        actual_5b = tags(res_5b.objects)
        if actual_5b != {"F"}:
            expected_5b = {"F"}
            if "C" in actual_5b: expected_5b.add("C")
            if "D" in actual_5b: expected_5b.add("D")
            results.append(("5b_contains_none_multi_incl",
                run_test("int_arr contains_none [1,4] (验证实际语义)",
                    Filter.by_property("int_arr").contains_none([1, 4]),
                    expected_5b,
                    col)))

        # ════════════════════════════════════════
        # 测试组 6: NOT(is_none) 一致性
        # ════════════════════════════════════════
        print("\n── 测试组 6: NOT(is_none) 一致性 ──")

        # 6a: NOT(int_val is null) 应等于 int_val is not null
        results.append(("6a_not_isnull",
            run_test("NOT(int_val is null) (应等于 is_none(False))",
                Filter.not_(Filter.by_property("int_val").is_none(True)),
                {"A", "B", "D", "F"},  # int_val 非null的行
                col)))

        # 6b: NOT(int_val is not null) 应等于 int_val is null
        results.append(("6b_not_isnotnull",
            run_test("NOT(int_val is not null) (应等于 is_none(True))",
                Filter.not_(Filter.by_property("int_val").is_none(False)),
                {"C", "E"},  # int_val 为null的行
                col)))

        # ════════════════════════════════════════
        # 测试组 7: not_equal vs NOT(equal) 一致性
        # ════════════════════════════════════════
        print("\n── 测试组 7: not_equal vs NOT(equal) 一致性 ──")

        res_ne = col.query.fetch_objects(filters=Filter.by_property("int_val").not_equal(10), limit=100)
        res_not_eq = col.query.fetch_objects(filters=Filter.not_(Filter.by_property("int_val").equal(10)), limit=100)
        ne_tags = tags(res_ne.objects)
        not_eq_tags = tags(res_not_eq.objects)
        consistent = ne_tags == not_eq_tags
        print(f"  {'✅' if consistent else '❌'} not_equal(10) vs NOT(equal(10)): {'一致' if consistent else '不一致'}")
        print(f"         not_equal:  {sorted(ne_tags)}")
        print(f"         NOT(equal): {sorted(not_eq_tags)}")
        results.append(("7_consistency", consistent))

        # ════════════════════════════════════════
        # 测试组 8: 交叉验证 — equal + not_equal 应覆盖全部行?
        # ════════════════════════════════════════
        print("\n── 测试组 8: equal ∪ not_equal 覆盖测试 ──")

        res_eq = col.query.fetch_objects(filters=Filter.by_property("int_val").equal(10), limit=100)
        eq_tags_set = tags(res_eq.objects)
        ne_tags_set = ne_tags
        union = eq_tags_set | ne_tags_set
        all_tags = {"A", "B", "C", "D", "E", "F"}
        covers_all = union == all_tags
        print(f"  equal(10):     {sorted(eq_tags_set)}")
        print(f"  not_equal(10): {sorted(ne_tags_set)}")
        print(f"  Union:         {sorted(union)}")
        print(f"  All rows:      {sorted(all_tags)}")
        print(f"  {'✅' if covers_all else '⚠️ '} equal ∪ not_equal {'= 全集' if covers_all else '≠ 全集 (gap: ' + str(sorted(all_tags - union)) + ')'}")
        results.append(("8_coverage", covers_all))

        # ════════════════════════════════════════
        # 汇总
        # ════════════════════════════════════════
        print("\n" + "=" * 70)
        print("📋 结论汇总:")
        print("=" * 70)

        # 判断 not_equal 是否包含 null
        ne_includes_null = "C" in ne_tags or "E" in ne_tags
        print(f"\n  1. not_equal 包含 null 行?  → {'是 ✓' if ne_includes_null else '否 ✗'}")
        print(f"     int_val != 10 返回: {sorted(ne_tags)}")

        # 判断 NOT(equal) 是否包含 null
        not_eq_includes_null = "C" in not_eq_tags or "E" in not_eq_tags
        print(f"\n  2. NOT(equal) 包含 null 行? → {'是 ✓' if not_eq_includes_null else '否 ✗'}")
        print(f"     NOT(int_val == 10) 返回: {sorted(not_eq_tags)}")

        # 判断空数组是否视为 null
        print(f"\n  3. 空数组 [] 被视为 null?   → {'是 ✓' if 'D' in actual_4a else '否 ✗'}")
        print(f"     int_arr is_none(True) 返回: {sorted(actual_4a)}")

        # contains_none null 行为
        print(f"\n  4. contains_none 包含 null 行? → {'是 ✓' if null_in else '否 ✗'}")
        print(f"     contains_none 包含空数组?   → {'是 ✓' if empty_in else '否 ✗'}")

        # not_equal vs NOT(equal) 一致性
        print(f"\n  5. not_equal ≡ NOT(equal)?  → {'是 ✓ (一致)' if consistent else '否 ✗ (不一致!)'}")

        # 覆盖性
        print(f"\n  6. equal ∪ not_equal = 全集? → {'是 ✓' if covers_all else '否 ✗ (缺少: ' + str(sorted(all_tags - union)) + ')'}")

        # 设计理念判断
        print("\n" + "-" * 70)
        if ne_includes_null and not_eq_includes_null and consistent:
            print("  🔍 结论: Weaviate 采用**布尔二值逻辑** (非 SQL 三值逻辑)")
            print("     - NOT/not_equal 对 null 行返回 True (null 不参与内层比较 → inner=False → NOT=True)")
            print("     - 这是一致的设计, 非 bug. Fuzzer oracle 中应适配此语义.")
        elif not ne_includes_null and not not_eq_includes_null:
            print("  🔍 结论: Weaviate 采用 **SQL 三值逻辑**")
            print("     - NOT/not_equal 对 null 行返回 UNKNOWN (被排除)")
            print("     - Fuzzer oracle 中 null → False 是正确的.")
        else:
            print("  ⚠️  结论: 行为不一致, 可能存在 bug!")
            print(f"     not_equal null={ne_includes_null}, NOT(equal) null={not_eq_includes_null}")

        if "D" in actual_4a:
            print("\n  🔍 空数组结论: Weaviate 不区分 [] 和 null, 均视为 null.")
            print("     Fuzzer 应在 oracle DataFrame 中将 [] 标准化为 None.")
        else:
            print("\n  🔍 空数组结论: Weaviate 区分 [] 和 null.")

        print("-" * 70)

        # Cleanup
        client.collections.delete(CLASS)

    finally:
        client.close()


if __name__ == "__main__":
    main()
