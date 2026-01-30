# 🐛 Milvus NULL 比较 Bug 报告

## 📋 执行概要

在对 Milvus v2.6.7 进行模糊测试时，发现了一个关于 NULL 值处理的严重 bug：**当使用主键精确匹配（`id == X`）结合标量字段数值比较时，NULL 值被错误地判定为满足比较条件。**

## 🔍 Bug 详情

### 问题描述
当执行类似 `(id == X) AND (field <= value)` 的查询时，如果该行的 `field` 字段为 NULL，Milvus 会错误地返回该行。

### 根本原因
违反了 SQL-92 标准的 NULL 三值逻辑：
- **预期行为**: `NULL <= any_value` 应该返回 UNKNOWN（在布尔上下文中视为 False）
- **实际行为**: Milvus 在某些情况下将其视为 True

## 📊 复现步骤

### 环境
- Milvus 版本: v2.6.7
- 部署方式: Docker
- Collection: fuzz_stable_v3
- 数据规模: 5000 行
- Schema: 包含 20 个动态标量字段 + JSON + ARRAY + 128维向量

### 最小化复现代码

```python
from pymilvus import connections, Collection

# 连接
connections.connect("default", host="127.0.0.1", port="19531")
col = Collection("fuzz_stable_v3")
col.load()

# 查询 ID=3588，该行的 c4 字段为 NULL
expr = "(id == 3588) and (c4 <= 7.3154750146117635)"
result = col.query(
    expr,
    output_fields=["id", "c4"],
    limit=1,
    consistency_level="Strong"
)

# 结果: 返回了 [3588]，但 c4=NULL 不应该满足 <= 条件
print(f"返回: {[r['id'] for r in result]}")  # 输出: [3588]
print(f"c4值: {[r['c4'] for r in result]}")   # 输出: [None]
```

### 运行复现脚本
```bash
cd /home/caihao/compare_test
python demo_for_senior.py
```

**预期结果**: 不返回任何行（因为 `NULL <= 7.31` 应该为 False）  
**实际结果**: 返回了 ID=3588（c4=NULL）

## 🎯 关键发现

### 1. 触发条件
- ✅ 使用主键精确匹配 `(id == X)`
- ✅ 结合标量字段的数值比较 `(field <= value)` 或 `(field >= value)` 等
- ✅ 该标量字段值为 NULL

### 2. 不触发的情况
- ❌ 单独的数值比较查询（如 `c4 <= 7.31`，不使用 id==）
  - 这种情况下 Milvus 正确处理了 NULL，不会返回 NULL 行

### 3. 发现过程
通过模糊测试（Fuzzing）发现：
1. 随机生成复杂查询表达式
2. 使用 Pandas 作为 oracle 计算正确结果
3. 对比 Milvus 返回结果
4. 发现 Milvus **多返回**了一些 ID（Extra IDs）
5. 逐个分析这些 Extra IDs，定位到 NULL 比较问题

## 💥 影响范围

### 严重性: **High**

1. **数据正确性**: 查询返回了不应该返回的数据（假阳性）
2. **标准合规性**: 违反 SQL-92 标准的 NULL 语义
3. **业务影响**: 可能导致下游业务逻辑错误
4. **静默失败**: 用户可能不会意识到返回了错误数据

### 受影响的场景
- 任何包含主键过滤 + 标量字段数值比较的查询
- 数据集中存在 NULL 值的场景
- 特别是使用 `<=`, `>=`, `<`, `>` 等数值比较运算符

## 🔧 建议修复

### 短期方案
在查询引擎中正确实现 NULL 的三值逻辑：
```
NULL == any_value  → UNKNOWN (treated as False)
NULL != any_value  → UNKNOWN (treated as False)
NULL <  any_value  → UNKNOWN (treated as False)
NULL <= any_value  → UNKNOWN (treated as False)
NULL >  any_value  → UNKNOWN (treated as False)
NULL >= any_value  → UNKNOWN (treated as False)
```

### 排查方向
根据测试结果，bug 很可能位于：
- **主键索引查询优化器**: 使用 `id ==` 时走了不同的代码路径
- **标量过滤器的 NULL 处理**: 在与主键过滤组合时未正确处理 NULL
- **查询计划生成**: 主键过滤 + 标量条件的组合查询可能跳过了 NULL 检查

## 📁 相关文件

### 测试脚本
- `demo_for_senior.py` - 给学姐的演示脚本（最小化复现）
- `test_id_3588.py` - 详细测试 ID=3588 的各种查询形式
- `critical_finding.py` - 对比全表扫描 vs 主键定位的 NULL 处理差异
- `exact_bug_repro.py` - 使用实际表数据的 bug 复现

### 原始发现
- `milvus_fuzz_oracle.py` - 模糊测试框架（发现此 bug 的工具）
- `deep_verify_bug.py` - 逐条件分析脚本
- `check_mismatch.py` - 初始 mismatch 验证

## 📈 测试数据

### Bug 复现的 ID 列表
从模糊测试中发现的受影响的行（c4=NULL）：
- **ID=3588** ✅ 稳定复现
- ID=2053
- ID=4466
- ID=4953
- ID=4501
- ID=4238
- ID=4463

### 数据生成
使用 seed=999 可以完全复现测试数据：
```python
import random, numpy as np
random.seed(999)
np.random.seed(999)
# 然后运行 milvus_fuzz_oracle.py 中的 DataManager
```

## 🎓 技术洞察

### 发现方法：Differential Testing (差分测试)
1. **Oracle**: 使用 Pandas 作为参考实现（Ground Truth）
2. **Fuzzing**: 随机生成查询表达式和数据
3. **Comparison**: 对比 Milvus 和 Pandas 的结果
4. **Analysis**: 分析差异，定位 root cause

### 验证技术：AND Validation
使用 `(id == X) AND (condition)` 来验证 Milvus 是否认为某行满足条件：
- 如果返回该行 → Milvus 认为 condition 为 True
- 如果不返回 → Milvus 认为 condition 为 False

这种技术可以绕过复杂表达式，直接测试单个条件的求值结果。

## ✅ 验证清单

- [x] Bug 已确认并可稳定复现
- [x] 创建了最小化复现脚本
- [x] 分析了触发条件和根本原因
- [x] 评估了影响范围
- [x] 提供了修复建议
- [x] 生成了详细的测试数据

## 📞 联系信息

**发现者**: [Your Name]  
**发现日期**: 2026年1月  
**测试框架**: Milvus Fuzz Oracle (自研)  
**验证人**: 学姐（待确认）

---

## 🚀 下一步

1. ✅ 向学姐展示 `demo_for_senior.py`
2. ⏳ 获得确认后，提交到 Milvus GitHub Issues
3. ⏳ 准备详细的 Issue 报告（可基于此文档）
4. ⏳ 跟踪修复进度

---

**附注**: 这个 bug 是通过系统化的模糊测试发现的，不是偶然遇到的。这证明了 **oracle-based differential testing** 在数据库系统验证中的有效性。
