# 实验环境版本信息

## 📦 核心依赖版本

| 组件 | 版本 |
|------|------|
| **Python** | 3.11.5 |
| **PyMilvus** | 2.6.5 |
| **Pandas** | 1.5.3 |
| **NumPy** | 1.26.4 |
| **Milvus Server** | v2.6.7 |
| **Milvus Lite** | 2.4.12 |

## 🖥️ 系统信息

- **操作系统**: Linux
- **Python 编译器**: GCC 11.2.0
- **部署方式**: Docker (Milvus Server)

## 🔧 关键配置

### Milvus 连接
- Host: 127.0.0.1
- Port: 19531
- Consistency Level: Strong

### 测试数据
- Collection: fuzz_stable_v3
- 数据量: 5000 rows
- 向量维度: 128-dim FLOAT_VECTOR
- 随机种子: 999 (确保可复现)

### Schema
- 主键: id (INT64)
- 标量字段: 20 个动态字段
  - DOUBLE (nullable, 可能为 NULL)
  - BOOL (nullable)
  - INT64 (nullable)
  - VARCHAR
- JSON: meta_json (嵌套深度 1-5)
- ARRAY: tags_array (ARRAY<INT64>)
- 向量: 128-dim FLOAT_VECTOR

### 索引配置
随机选择以下之一：
- FLAT (暴力搜索)
- HNSW (M=32, efConstruction=256)
- IVF_FLAT (nlist=128)

## 📋 完整依赖列表

获取完整依赖：
```bash
pip list > requirements_full.txt
```

或仅核心依赖：
```bash
pip freeze | grep -E "(pymilvus|pandas|numpy)" > requirements.txt
```

## ✅ 版本验证

运行以下命令验证环境：
```python
import sys
import pymilvus
import pandas as pd
import numpy as np

print(f"Python: {sys.version.split()[0]}")
print(f"PyMilvus: {pymilvus.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"NumPy: {np.__version__}")
```

预期输出：
```
Python: 3.11.5
PyMilvus: 2.6.5
Pandas: 1.5.3
NumPy: 1.26.4
```

## 🔄 复现环境搭建

如果其他人想复现此 bug，需要：

1. **安装 Python 3.11.5** (或兼容版本)
   
2. **安装依赖包**:
   ```bash
   pip install pymilvus==2.6.5 pandas==1.5.3 numpy==1.26.4
   ```

3. **启动 Milvus v2.6.7**:
   ```bash
   docker run -d --name milvus \
     -p 19530:19530 -p 19531:19531 \
     milvusdb/milvus:v2.6.7
   ```

4. **运行复现脚本**:
   ```bash
   python issue_repro.py
   ```

## 📝 注意事项

- ⚠️ PyMilvus 版本 (2.6.5) 与 Milvus Server 版本 (2.6.7) 不完全一致，但兼容
- ✅ 使用 seed=999 确保数据生成的可复现性
- ✅ 所有测试都使用 consistency_level="Strong" 确保数据一致性
- ✅ Bug 在此版本组合下可稳定复现

## 🔗 相关链接

- [Milvus v2.6.7 Release Notes](https://github.com/milvus-io/milvus/releases/tag/v2.6.7)
- [PyMilvus Documentation](https://milvus.io/docs/release_notes.md)
- [SQL-92 NULL Semantics](https://en.wikipedia.org/wiki/Null_(SQL))
