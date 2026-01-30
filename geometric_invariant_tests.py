import time
import numpy as np
from typing import List, Tuple, Set
from db_adapters import MilvusAdapter, QdrantAdapter, ChromaAdapter, WeaviateAdapter

# --- 配置 ---
N = 50000  # 数据规模：平衡精度验证与测试速度
DIM = 768  # 常见文本嵌入维度（BERT/OpenAI 等）
BATCH_DEFAULT = 2000
BATCH_QDRANT = 1000
BATCH_CHROMA = 1000
SEED = 123
TOPK = 20

GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
RESET = '\033[0m'

DBS = [
    # 使用精确搜索的 FLAT / exact 配置
    {"name": "Milvus", "adapter": MilvusAdapter, "batch": BATCH_DEFAULT},   # FLAT 索引
    {"name": "Qdrant", "adapter": QdrantAdapter, "batch": BATCH_QDRANT},   # exact=True 精确搜索
    # 如需测试近似库，可再启用：
    # {"name": "Chroma", "adapter": ChromaAdapter, "batch": BATCH_CHROMA},
    # {"name": "Weaviate", "adapter": WeaviateAdapter, "batch": BATCH_DEFAULT},
]


def generate_random_rotation_matrix(dim: int, seed: int) -> np.ndarray:
    """生成随机正交旋转矩阵（QR 分解法）"""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(dim, dim)).astype(np.float32)
    Q, _ = np.linalg.qr(A)
    return Q.astype(np.float32)


def prepare_data(n: int, dim: int):
    """准备测试数据"""
    rng = np.random.default_rng(SEED)
    ids = np.arange(n, dtype=np.int64)
    vecs = rng.random((n, dim), dtype=np.float32)
    
    # 标量字段（用于测试标量无关性）
    age = rng.integers(20, 80, size=n, dtype=np.int64)
    price = rng.random(n) * 1000.0
    tag = rng.integers(0, 4, size=n, dtype=np.int64)
    
    return ids, vecs, {"age": age.tolist(), "price": price.tolist(), "tag": tag.tolist()}


def compare_topk_results(res1: List[Tuple[int, float]], res2: List[Tuple[int, float]], 
                         tolerance: float = 1e-4) -> Tuple[bool, float, int]:
    """
    比较两个 Top-K 结果
    返回: (ID完全一致, 平均距离差异, ID不匹配数量)
    """
    ids1 = [id for id, _ in res1]
    ids2 = [id for id, _ in res2]
    
    id_match = ids1 == ids2
    id_mismatch_count = sum(1 for a, b in zip(ids1, ids2) if a != b)
    
    # 计算距离差异
    if len(res1) > 0 and len(res2) > 0:
        dists1 = np.array([d for _, d in res1])
        dists2 = np.array([d for _, d in res2])
        avg_dist_diff = np.mean(np.abs(dists1 - dists2))
    else:
        avg_dist_diff = 0.0
    
    return id_match, avg_dist_diff, id_mismatch_count


class GeometricInvariantTester:
    """几何不变性测试器"""
    
    def __init__(self, db_name: str, adapter, batch_size: int):
        self.name = db_name
        self.adapter = adapter
        self.batch_size = batch_size
    
    def setup_collection(self, col_name: str, ids, vecs, payloads):
        """初始化集合并加载数据"""
        self.adapter.connect()
        self.adapter.recreate_collection(col_name)
        
        for i in range(0, len(ids), self.batch_size):
            j = min(i + self.batch_size, len(ids))
            self.adapter.insert_batch(ids[i:j], vecs[i:j], {
                "age": payloads["age"][i:j],
                "price": payloads["price"][i:j],
                "tag": payloads["tag"][i:j],
            })
        
        self.adapter.flush_and_index()
    
    def test_1_translation_invariance(self, ids, vecs, payloads, probe):
        """
        Test 1: 平移不变性 (L2 距离)
        Oracle: 对所有向量和查询向量施加相同平移，Top-K ID 必须不变
        """
        print(f"\n  🔄 [Test 1] Translation Invariance (L2)")
        self.setup_collection("geom_original", ids, vecs, payloads)
        res_original = self.adapter.search_ordered(probe, TOPK, expr_type=None)
        
        # 平移变换：所有向量 + 常数向量
        translation = np.ones(DIM, dtype=np.float32) * 10.0
        vecs_translated = vecs + translation
        probe_translated = probe + translation
        
        self.setup_collection("geom_translated", ids, vecs_translated, payloads)
        res_translated = self.adapter.search_ordered(probe_translated, TOPK, expr_type=None)
        
        # 比较结果
        id_match, dist_diff, mismatch_count = compare_topk_results(res_original, res_translated)
        
        # L2 距离应完全不变（平移不影响欧氏距离）
        dist_ok = dist_diff < 1e-3
        
        print(f"    原始 Top-5 ID: {[id for id, _ in res_original[:5]]}")
        print(f"    平移 Top-5 ID: {[id for id, _ in res_translated[:5]]}")
        print(f"    ID 完全一致: {'✅' if id_match else f'❌ ({mismatch_count} 不匹配)'}")
        print(f"    平均距离差异: {dist_diff:.6f} -> {'✅' if dist_ok else '❌'}")
        
        if not id_match:
            raise AssertionError(f"Translation invariance violated: {mismatch_count} ID mismatches")
        if not dist_ok:
            print(f"    ⚠️ 距离差异超出容忍范围，可能精度问题")
    
    def test_2_rotation_invariance(self, ids, vecs, payloads, probe):
        """
        Test 2: 旋转不变性 (L2 距离)
        Oracle: 正交旋转不改变 L2 距离，Top-K ID 必须不变
        """
        print(f"\n  🔁 [Test 2] Rotation Invariance (L2)")
        
        # 原始查询
        self.setup_collection("geom_original_rot", ids, vecs, payloads)
        res_original = self.adapter.search_ordered(probe, TOPK, expr_type=None)
        
        # 旋转变换：正交矩阵
        rotation_matrix = generate_random_rotation_matrix(DIM, SEED + 1)
        vecs_rotated = vecs @ rotation_matrix.T
        probe_rotated = probe @ rotation_matrix.T
        
        self.setup_collection("geom_rotated", ids, vecs_rotated, payloads)
        res_rotated = self.adapter.search_ordered(probe_rotated, TOPK, expr_type=None)
        
        # 比较结果
        id_match, dist_diff, mismatch_count = compare_topk_results(res_original, res_rotated)
        dist_ok = dist_diff < 1e-3
        
        print(f"    原始 Top-5 ID: {[id for id, _ in res_original[:5]]}")
        print(f"    旋转 Top-5 ID: {[id for id, _ in res_rotated[:5]]}")
        print(f"    ID 完全一致: {'✅' if id_match else f'❌ ({mismatch_count} 不匹配)'}")
        print(f"    平均距离差异: {dist_diff:.6f} -> {'✅' if dist_ok else '❌'}")
        
        if not id_match:
            raise AssertionError(f"Rotation invariance violated: {mismatch_count} ID mismatches")
        if not dist_ok:
            print(f"    ⚠️ 距离差异超出容忍范围，可能精度问题")
    
    def test_3_dimension_expansion(self, ids, vecs, payloads, probe):
        """
        Test 3: 维度扩充不变性
        Oracle: 在尾部增加全 0 维度，L2 距离和 Top-K 结果完全不变
        Bug 模式: SIMD 优化在非 8 倍数维度时的边界问题
        """
        print(f"\n  📏 [Test 3] Dimension Expansion Invariance")
        
        # 原始查询（768维）
        self.setup_collection("geom_dim768", ids, vecs, payloads)
        res_original = self.adapter.search_ordered(probe, TOPK, expr_type=None)
        
        # 扩充维度：768 -> 777 (非8倍数，测试SIMD边界)
        pad_width = 9
        vecs_expanded = np.pad(vecs, ((0, 0), (0, pad_width)), mode='constant')
        probe_expanded = np.pad(probe, (0, pad_width), mode='constant')
        
        # 切换到新维度
        old_dim = self.adapter.dim
        self.adapter.dim = DIM + pad_width
        self.setup_collection("geom_dim777", ids, vecs_expanded, payloads)
        res_expanded = self.adapter.search_ordered(probe_expanded, TOPK, expr_type=None)
        self.adapter.dim = old_dim
        
        # 比较结果
        id_match, dist_diff, mismatch_count = compare_topk_results(res_original, res_expanded)
        dist_ok = dist_diff < 1e-3
        
        print(f"    原始 (dim={DIM}) Top-5 ID: {[id for id, _ in res_original[:5]]}")
        print(f"    扩充 (dim={DIM+pad_width}) Top-5 ID: {[id for id, _ in res_expanded[:5]]}")
        print(f"    ID 完全一致: {'✅' if id_match else f'❌ ({mismatch_count} 不匹配)'}")
        print(f"    平均距离差异: {dist_diff:.6f} -> {'✅' if dist_ok else '❌'}")
        
        if not id_match:
            raise AssertionError(f"Dimension expansion invariance violated: {mismatch_count} ID mismatches")
        if not dist_ok:
            print(f"    ⚠️ 距离差异超出容忍范围，可能 SIMD 边界 Bug")
    
    def test_4_scalar_independence(self, ids, vecs, payloads, probe):
        """
        Test 4: 标量无关性
        Oracle: 仅修改标量字段，纯向量搜索（无过滤）的结果必须完全不变
        Bug 模式: 混合索引更新时错误触发向量索引重建
        """
        print(f"\n  🔢 [Test 4] Scalar Independence")
        
        # 原始查询（无过滤）
        self.setup_collection("geom_scalar_orig", ids, vecs, payloads)
        res_before = self.adapter.search_ordered(probe, TOPK, expr_type=None)
        
        # 修改标量字段（模拟 Update）
        # 注意: 大多数向量数据库不支持原地更新，这里通过重建集合+修改 payload 模拟
        modified_payloads = {
            "age": [age + 10 for age in payloads["age"]],  # 所有年龄 +10
            "price": [price * 1.5 for price in payloads["price"]],  # 所有价格 x1.5
            "tag": payloads["tag"]  # tag 不变
        }
        
        self.setup_collection("geom_scalar_modified", ids, vecs, modified_payloads)
        res_after = self.adapter.search_ordered(probe, TOPK, expr_type=None)
        
        # 比较结果（纯向量搜索，标量变化不应影响）
        id_match, dist_diff, mismatch_count = compare_topk_results(res_before, res_after)
        dist_ok = dist_diff < 1e-5
        
        print(f"    修改前 Top-5 ID: {[id for id, _ in res_before[:5]]}")
        print(f"    修改后 Top-5 ID: {[id for id, _ in res_after[:5]]}")
        print(f"    ID 完全一致: {'✅' if id_match else f'❌ ({mismatch_count} 不匹配)'}")
        print(f"    平均距离差异: {dist_diff:.6f} -> {'✅' if dist_ok else '❌'}")
        
        if not id_match:
            raise AssertionError(f"Scalar independence violated: {mismatch_count} ID mismatches")
        if not dist_ok:
            print(f"    ⚠️ 距离发生变化，可能向量索引被意外重建")


def run_tests_for_db(db_cfg: dict, ids, vecs, payloads, probe):
    """对单个数据库运行所有几何不变性测试"""
    name = db_cfg["name"]
    print(f"\n{'='*70}")
    print(f"{GREEN}🔬 {name} 几何不变性测试{RESET}")
    print(f"{'='*70}")
    
    try:
        adapter = db_cfg["adapter"](DIM)
        tester = GeometricInvariantTester(name, adapter, db_cfg["batch"])
        
        failures = []
        tests = [
            ("translation", lambda: tester.test_1_translation_invariance(ids, vecs, payloads, probe)),
            ("rotation", lambda: tester.test_2_rotation_invariance(ids, vecs, payloads, probe)),
            ("dimension", lambda: tester.test_3_dimension_expansion(ids, vecs, payloads, probe)),
            ("scalar", lambda: tester.test_4_scalar_independence(ids, vecs, payloads, probe)),
        ]
        
        for test_name, test_fn in tests:
            try:
                test_fn()
            except Exception as e:
                failures.append((test_name, str(e)))
                print(f"   ⚠️ 测试 {test_name} 失败: {e}")
        
        if failures:
            print(f"\n{RED}❌ {name} 存在失败测试{RESET}")
            for test_name, msg in failures:
                print(f"   ↪ {test_name}: {msg}")
            return False
        else:
            print(f"\n{GREEN}✅ 全部 {name} 几何不变性测试通过{RESET}")
            return True
    
    except Exception as e:
        print(f"{RED}❌ {name} 初始化失败: {e}{RESET}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*70)
    print("🎯 几何与算法不变性测试 (Geometric & Algorithmic Invariants)")
    print("="*70)
    
    # 准备数据
    ids, vecs, payloads = prepare_data(N, DIM)
    probe = vecs[0]  # 使用第一个向量作为查询
    
    print(f"\n📊 数据准备: N={N}, DIM={DIM}")
    
    # 逐数据库运行测试
    results = {}
    for db_cfg in DBS:
        results[db_cfg["name"]] = run_tests_for_db(db_cfg, ids, vecs, payloads, probe)
    
    # 汇总
    print(f"\n{'='*70}")
    print(f"{GREEN}📋 测试结果汇总{RESET}")
    print(f"{'='*70}")
    for name, ok in results.items():
        status = f"{GREEN}✅ 通过{RESET}" if ok else f"{RED}❌ 失败{RESET}"
        print(f"  {name:12s} : {status}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
