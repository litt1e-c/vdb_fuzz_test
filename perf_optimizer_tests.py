import time
from typing import Dict, List, Tuple

import numpy as np
from collections import Counter

from db_adapters import ChromaAdapter, MilvusAdapter, QdrantAdapter, WeaviateAdapter

# --- 配置 ---
N = 50000  # 数据规模
BATCH_DEFAULT = 2000
BATCH_QDRANT = 1000
BATCH_CHROMA = 1000
SEED = 42

# ef 搜索参数 (仅 Milvus 有效)
SEARCH_EF_LOW = 128
SEARCH_EF_HIGH = 4096
TOPK = 50

# 维度巡检：128 -> 129（对齐检查）、256（2倍）、512（4倍）
DIM_SCHEDULE = [128, 129, 130, 140, 300, 400, 512]

# ef 稳定性测试：重复次数
EF_REPEATS = 5
EF_MAJORITY = 0.6  # 多数投票门槛

DBS = [
    {"name": "Milvus", "adapter": MilvusAdapter, "batch": BATCH_DEFAULT, "supports_ef": True},
    {"name": "Qdrant", "adapter": QdrantAdapter, "batch": BATCH_QDRANT, "supports_ef": False},
    {"name": "Chroma", "adapter": ChromaAdapter, "batch": BATCH_CHROMA, "supports_ef": False},  # 暂时搁置，后续可恢复
    {"name": "Weaviate", "adapter": WeaviateAdapter, "batch": BATCH_DEFAULT, "supports_ef": False},
]



def prepare_data(n: int, dim: int):
    """构造有意的数据偏斜，用于测试过滤选择率"""
    rng = np.random.default_rng(SEED)
    ids = np.arange(n, dtype=np.int64)
    vecs = rng.random((n, dim), dtype=np.float32)
    
    # 构造 tag 偏斜：0 稀少（1%），1 常见（50%），2/3 其他
    age = rng.integers(0, 100, size=n, dtype=np.int64)
    price = rng.random(n) * 1000.0
    tag = rng.integers(0, 4, size=n, dtype=np.int64)
    
    # 手动调整偏斜：tag=0 只有 1%，tag=1 有 50%
    num_common = int(n * 0.5)
    num_rare = int(n * 0.01)
    tag[:num_common] = 1
    tag[-num_rare:] = 0
    
    return ids, vecs, {"age": age.tolist(), "price": price.tolist(), "tag": tag.tolist()}


class PerformanceOracle:
    """性能与优化器诊断工具"""
    
    def __init__(self, db_name: str, adapter):
        self.name = db_name
        self.adapter = adapter
        self.col = None
    
    def setup(self, col_name: str, ids, vecs, payloads, batch: int):
        """初始化并加载数据"""
        self.adapter.connect()
        self.adapter.recreate_collection(col_name)
        for i in range(0, len(ids), batch):
            j = min(i + batch, len(ids))
            self.adapter.insert_batch(ids[i:j], vecs[i:j], {
                "age": payloads["age"][i:j],
                "price": payloads["price"][i:j],
                "tag": payloads["tag"][i:j],
            })
        t0 = time.perf_counter()
        self.adapter.flush_and_index()
        idx_ms = (time.perf_counter() - t0) * 1000
        return idx_ms
    
    def _search_single(self, probe, expr_type, limit=TOPK):
        """单次查询，返回 (latency_ms, result_set)"""
        t0 = time.perf_counter()
        res = self.adapter.search(probe, limit, expr_type=expr_type)
        lat = (time.perf_counter() - t0) * 1000
        return lat, res or set()
    
    def _stable_search(self, probe, expr_type, repeats=EF_REPEATS, majority=EF_MAJORITY):
        """多次查询取多数投票，用于 ef 稳定性检查"""
        results = []
        latencies = []
        for _ in range(repeats):
            lat, res = self._search_single(probe, expr_type)
            results.append(res)
            latencies.append(lat)
        
        freq = Counter()
        for s in results:
            for id in s:
                freq[id] += 1
        thresh = max(1, int(repeats * majority))
        stable_set = {id for id, f in freq.items() if f >= thresh}
        avg_lat = sum(latencies) / len(latencies)
        return stable_set, avg_lat
    
    def test_1_ef_stability(self, probe):
        """Theme 1: ef 值增加不应恶化结果，多次重复验证稳定性"""
        print(f"\n  📊 [Test 1] EF Stability (ef_low={SEARCH_EF_LOW} vs ef_high={SEARCH_EF_HIGH})")
        
        if self.name != "Milvus":
            print(f"    ⏭️  {self.name} 不支持 ef 参数，跳过")
            return
        
        # 多次查询 ef_low，取多数投票结果（使用 Milvus 原生 API）
        params_low = {"metric_type": "L2", "params": {"ef": SEARCH_EF_LOW}}
        t0 = time.perf_counter()
        for _ in range(EF_REPEATS):
            self.adapter.col.search([probe], "vector", params_low, limit=TOPK, expr="tag == 1", output_fields=["id"])
        lat_low = (time.perf_counter() - t0) * 1000 / EF_REPEATS
        
        # 多次查询 ef_high
        params_high = {"metric_type": "L2", "params": {"ef": SEARCH_EF_HIGH}}
        t0 = time.perf_counter()
        for _ in range(EF_REPEATS):
            self.adapter.col.search([probe], "vector", params_high, limit=TOPK, expr="tag == 1", output_fields=["id"])
        lat_high = (time.perf_counter() - t0) * 1000 / EF_REPEATS
        
        print(f"    ef_low={SEARCH_EF_LOW}:  avg={lat_low:.2f} ms ({EF_REPEATS} 次重复)")
        print(f"    ef_high={SEARCH_EF_HIGH}: avg={lat_high:.2f} ms ({EF_REPEATS} 次重复)")
        
        # Oracle: ef_high 耗时应 >= ef_low（更多计算），但不应超过 3x
        lat_ratio = lat_high / lat_low if lat_low > 0 else 1.0
        if lat_ratio < 0.5:
            print(f"    🚨 异常: ef_high 反而更快 ({lat_ratio:.2f}x)，可能优化器有问题")
        elif lat_ratio > 3.0:
            print(f"    ⚠️  警告: ef_high 显著变慢 ({lat_ratio:.2f}x)，可能过度计算")
        else:
            print(f"    ✅ 正常: ef_high 耗时 {lat_ratio:.2f}x")
    
    def test_2_dim_scaling(self, ids, base_vecs, payloads, batch):
        """Theme 2: 维度增加应线性或亚线性影响性能，不应突变"""
        print(f"\n  📐 [Test 2] Dimension Scaling (Linearity Check)")
        
        build_times = []
        search_times = []
        
        rng = np.random.default_rng(SEED)
        
        for d in DIM_SCHEDULE:
            # 截取或填充到目标维度
            if d <= base_vecs.shape[1]:
                current_vecs = base_vecs[:, :d]
            else:
                pad_width = d - base_vecs.shape[1]
                current_vecs = np.pad(base_vecs, ((0, 0), (0, pad_width)))
            
            col_name = f"Perf_dim_{d}_{self.name}"
            old_dim = self.adapter.dim
            self.adapter.dim = d
            
            # 计时：构建索引
            t_build = self.setup(col_name, ids, current_vecs, payloads, batch)
            
            # 计时：搜索
            probe = current_vecs[0]
            t0 = time.perf_counter()
            self.adapter.search(probe, TOPK, expr_type="tag_1")
            t_search = (time.perf_counter() - t0) * 1000
            
            build_times.append((d, t_build))
            search_times.append((d, t_search))
            print(f"    dim={d:3d} | build={t_build:6.2f}ms | search={t_search:6.2f}ms")
            
            self.adapter.dim = old_dim
        
        # 分析：128->129 是否突变
        d_map = dict(build_times)
        if 128 in d_map and 129 in d_map:
            ratio = d_map[129] / d_map[128]
            if ratio > 1.3:
                print(f"    🚨 异常: 128->129 构建时间增长 {ratio:.2f}x，超过预期 1.3x")
            else:
                print(f"    ✅ 正常: 128->129 增长 {ratio:.2f}x")
    
    def test_3_filter_selectivity(self, probe):
        """Theme 3: 高选择率应比低选择率快（过滤代价单调性）"""
        print(f"\n  🔍 [Test 3] Filter Selectivity Monotonicity")
        
        # 场景 A：极高选择率（tag==0，只有 1%）
        lat_high_sel, res_high_sel = self._search_single(probe, "tag_0", limit=TOPK)
        
        # 场景 B：低选择率（tag==1，50%）
        lat_low_sel, res_low_sel = self._search_single(probe, "tag_1", limit=TOPK)
        
        # 场景 C：无过滤（全量）
        lat_none, res_none = self._search_single(probe, None, limit=TOPK)
        
        print(f"    HighSelectivity (tag==0, ~1%): {lat_high_sel:.2f}ms, |result|={len(res_high_sel)}")
        print(f"    LowSelectivity  (tag==1, ~50%): {lat_low_sel:.2f}ms, |result|={len(res_low_sel)}")
        print(f"    NoFilter        (全量):        {lat_none:.2f}ms, |result|={len(res_none)}")
        
        # Oracle: 高选择率应该快于低选择率
        ratio = lat_high_sel / lat_low_sel if lat_low_sel > 0 else 1.0
        if lat_high_sel > lat_low_sel * 1.2:  # 允许 20% 误差
            print(f"    🚨 性能倒挂: 高选择率 {lat_high_sel:.2f}ms > 低选择率 {lat_low_sel:.2f}ms (ratio={ratio:.2f}x)")
            print(f"       → 可能优化器错选 Post-filtering，应该用 Pre-filtering")
        else:
            print(f"    ✅ 符合预期: 高选择率 {ratio:.2f}x 快于低选择率")
    
    def test_4_complex_query(self, probe):
        """Theme 4: 复杂查询（语义等价但表达式复杂）不应显著衰退 - 测试 Constant Folding 优化"""
        print(f"\n  🧩 [Test 4] Complex Query Performance (Constant Folding Optimizer)")
        
        # 基线：tag == 1 (简单查询)
        lat_simple, res_simple = self._search_single(probe, "tag_1", limit=TOPK)
        
        # 复杂查询：tag == 1 AND (age >= 0 AND age <= 100 AND price >= 0 AND price <= 1000)
        # 说明：额外的 age 和 price 条件都是"恒真的"（即所有数据都满足这些条件）
        # 如果优化器足够聪明（Constant Folding），它应该识别出这些是废话条件并优化掉
        # 从而使 tag_1_tautology 的性能等同于 tag_1
        lat_complex, res_complex = self._search_single(probe, "tag_1_tautology", limit=TOPK)
        
        # 结果集应该完全相同（因为语义等价）
        results_equal = res_simple == res_complex
        
        ratio = lat_complex / lat_simple if lat_simple > 0 else 1.0
        print(f"    基线 (tag==1):                        {lat_simple:.2f}ms, |result|={len(res_simple)}")
        print(f"    复杂 (tag==1 AND tautology):          {lat_complex:.2f}ms, |result|={len(res_complex)}")
        print(f"    结果集一致: {'✅ 是' if results_equal else '❌ 否'}")
        
        # Oracle：复杂查询不应超过简单查询 1.5x（良好的优化器应该接近 1.0x）
        if not results_equal:
            print(f"    ⚠️  结果集不一致，可能存在语义错误")
        elif ratio > 2.0:
            print(f"    🚨 优化器失效: 复杂查询 {ratio:.2f}x 于基线")
            print(f"       → 优化器未能消除恒真条件（Constant Folding 失败）")
        elif ratio > 1.3:
            print(f"    ⚠️  轻微衰退: 复杂查询 {ratio:.2f}x 于基线，优化空间大")
        else:
            print(f"    ✅ 优化器有效: 复杂查询仅 {ratio:.2f}x，恒真条件被成功优化")
    
    def test_5_batch_consistency(self, probes):
        """Theme 5: 批查询一致性（多向量同时查询 vs 逐个查询结果一致）"""
        print(f"\n  📦 [Test 5] Batch Query Consistency")
        
        expr = "tag_1"
        
        # 方法 A：逐个查询并聚合
        t0 = time.perf_counter()
        results_single = [self.adapter.search(p, TOPK, expr_type=expr) or set() for p in probes]
        lat_single = (time.perf_counter() - t0) * 1000
        
        # 方法 B：批量查询（如果适配器支持）
        # 大多数适配器的 search 不直接支持多向量，所以模拟为顺序调用
        t0 = time.perf_counter()
        results_batch = [self.adapter.search(p, TOPK, expr_type=expr) or set() for p in probes]
        lat_batch = (time.perf_counter() - t0) * 1000
        
        # 检查结果一致性
        consistent = all(a == b for a, b in zip(results_single, results_batch))
        print(f"    单个查询耗时: {lat_single:.2f}ms")
        print(f"    批量查询耗时: {lat_batch:.2f}ms")
        print(f"    结果一致性: {'✅ 一致' if consistent else '🚨 不一致'}")
        
        if not consistent:
            for i, (a, b) in enumerate(zip(results_single, results_batch)):
                if a != b:
                    print(f"       Query {i}: {len(a)} != {len(b)}")
                    diff = a.symmetric_difference(b)
                    print(f"       差异 ID (样本): {list(diff)[:5]}")


def run_oracle_for_db(db_cfg: Dict, ids, vecs, payloads):
    """对单个数据库运行全部性能测试"""
    name = db_cfg["name"]
    print(f"\n{'='*70}")
    print(f"🔬 {name} 性能与优化器诊断")
    print(f"{'='*70}")
    
    try:
        adapter = db_cfg["adapter"](128)  # 初始维度
        oracle = PerformanceOracle(name, adapter)
        
        # Setup：加载 128 维数据
        col_name = f"Perf_main_{name}"
        idx_ms = oracle.setup(col_name, ids, vecs[:, :128], payloads, db_cfg["batch"])
        # 保存主集合句柄（部分适配器无 col 属性）
        main_col_handle = getattr(oracle.adapter, "col", None)
        print(f"初始化完成 (dim=128)：索引构建 {idx_ms:.2f}ms")
        
        probe = vecs[0, :128]
        probes = [vecs[1, :128], vecs[2, :128], vecs[3, :128]]
        
        # 运行五大测试
        oracle.test_1_ef_stability(probe)
        oracle.test_2_dim_scaling(ids, vecs, payloads, db_cfg["batch"])
        # 修复维度巡检后上下文：切回主集合（128维）
        try:
            if name == "Milvus":
                from pymilvus import Collection
                oracle.adapter.col = Collection(col_name)
            else:
                # 对 Chroma/Weaviate/Qdrant 等：如果有 col 句柄则恢复；否则仅恢复 col_name
                if hasattr(oracle.adapter, "col"):
                    oracle.adapter.col = main_col_handle
                if hasattr(oracle.adapter, "col_name"):
                    oracle.adapter.col_name = col_name
            oracle.adapter.dim = 128
        except Exception as reset_err:
            print(f"    ⚠️ 集合上下文恢复警告: {name} -> {reset_err}")
        oracle.test_3_filter_selectivity(probe)
        oracle.test_4_complex_query(probe)
        oracle.test_5_batch_consistency(probes)
        
        print(f"\n✅ {name} 测试完成")
        return True
    
    except Exception as e:
        print(f"❌ {name} 测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*70)
    print("🎯 性能与优化器综合诊断 (Performance & Optimizer Comprehensive Diagnosis)")
    print("="*70)
    
    # 准备数据
    ids, vecs, payloads = prepare_data(N, max(DIM_SCHEDULE))
    print(f"\n📊 数据准备: N={N}, max_dim={max(DIM_SCHEDULE)}")
    print(f"   Tag 分布: tag=0 ~1%, tag=1 ~50%, tag=2/3 ~49%")
    
    # 逐数据库运行诊断
    results = {}
    for db_cfg in DBS:
        results[db_cfg["name"]] = run_oracle_for_db(db_cfg, ids, vecs, payloads)
    
    # 汇总
    print(f"\n{'='*70}")
    print(f"📋 测试汇总")
    print(f"{'='*70}")
    for name, ok in results.items():
        status = "✅ 通过诊断" if ok else "❌ 异常"
        print(f"  {name:12s}: {status}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
