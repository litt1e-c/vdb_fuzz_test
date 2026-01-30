# 查询能力支持度统计

| 数据库 | 支持 | 不支持 | 错误 | 支持率 |
|--------|------|--------|------|--------|
| Milvus | 18 | 1 | 2 | 85.7% |
| Qdrant | 18 | 2 | 1 | 85.7% |
| Chroma | 14 | 7 | 0 | 66.7% |
| Weaviate | 13 | 8 | 0 | 61.9% |

# Vector Database Query Capability Matrix

**生成时间**: 2025-12-17 09:49:21

**测试数据库**:
- ✅ Milvus
- ✅ Qdrant
- ✅ Chroma
- ✅ Weaviate

---

## 图例说明

| 符号 | 含义 |
|------|------|
| ✅ | 完全支持 |
| ❌ | 不支持或出错 |
| ⊘ | 数据库未安装/连接失败 |
| ⚠️ | 部分支持 |

## ARITHMETIC

| 查询类型 | Milvus | Qdrant | Chroma | Weaviate |
|----------|--------|--------|--------|----------|
| **算术表达式** | ❌ 不支持 | ❌ 不支持 | ❌ 不支持 | ❌ 不支持 |

## ARRAY

| 查询类型 | Milvus | Qdrant | Chroma | Weaviate |
|----------|--------|--------|--------|----------|
| **数组包含** | ✅ 支持<br/>(10 hits, 340.7ms) | ✅ 支持<br/>(10 hits, 3.7ms) | ❌ 不支持 | ❌ 不支持 |

## EQUALITY

| 查询类型 | Milvus | Qdrant | Chroma | Weaviate |
|----------|--------|--------|--------|----------|
| **整数等值** | ✅ 支持<br/>(10 hits, 159.0ms) | ✅ 支持<br/>(10 hits, 7.3ms) | ✅ 支持<br/>(10 hits, 3.5ms) | ✅ 支持<br/>(10 hits, 6.8ms) |
| **字符串等值** | ✅ 支持<br/>(10 hits, 358.0ms) | ✅ 支持<br/>(10 hits, 4.8ms) | ✅ 支持<br/>(10 hits, 2.8ms) | ✅ 支持<br/>(10 hits, 6.3ms) |
| **布尔等值** | ✅ 支持<br/>(10 hits, 239.0ms) | ✅ 支持<br/>(10 hits, 4.6ms) | ✅ 支持<br/>(10 hits, 3.6ms) | ✅ 支持<br/>(10 hits, 6.8ms) |

## JSON

| 查询类型 | Milvus | Qdrant | Chroma | Weaviate |
|----------|--------|--------|--------|----------|
| **JSON字段访问** | ✅ 支持<br/>(10 hits, 270.4ms) | ✅ 支持<br/>(10 hits, 3.4ms) | ❌ 不支持 | ❌ 不支持 |
| **JSON多级嵌套** | ✅ 支持<br/>(10 hits, 297.7ms) | ✅ 支持<br/>(10 hits, 4.2ms) | ❌ 不支持 | ❌ 不支持 |

## LOGIC

| 查询类型 | Milvus | Qdrant | Chroma | Weaviate |
|----------|--------|--------|--------|----------|
| **AND逻辑** | ✅ 支持<br/>(10 hits, 363.1ms) | ✅ 支持<br/>(10 hits, 4.8ms) | ✅ 支持<br/>(10 hits, 3.7ms) | ✅ 支持<br/>(10 hits, 6.7ms) |
| **OR逻辑** | ✅ 支持<br/>(10 hits, 248.3ms) | ✅ 支持<br/>(10 hits, 3.9ms) | ✅ 支持<br/>(10 hits, 3.3ms) | ✅ 支持<br/>(10 hits, 6.0ms) |
| **NOT逻辑** | ✅ 支持<br/>(10 hits, 224.6ms) | ✅ 支持<br/>(10 hits, 5.3ms) | ✅ 支持<br/>(10 hits, 4.0ms) | ✅ 支持<br/>(10 hits, 8.6ms) |
| **复杂组合** | ✅ 支持<br/>(10 hits, 234.4ms) | ❌ 错误 | ✅ 支持<br/>(10 hits, 4.8ms) | ✅ 支持<br/>(10 hits, 9.0ms) |

## NULL HANDLING

| 查询类型 | Milvus | Qdrant | Chroma | Weaviate |
|----------|--------|--------|--------|----------|
| **IS NULL** | ❌ 错误 | ✅ 支持<br/>(10 hits, 6.0ms) | ❌ 不支持 | ❌ 不支持 |
| **IS NOT NULL** | ❌ 错误 | ✅ 支持<br/>(10 hits, 6.2ms) | ❌ 不支持 | ❌ 不支持 |

## RANGE

| 查询类型 | Milvus | Qdrant | Chroma | Weaviate |
|----------|--------|--------|--------|----------|
| **大于** | ✅ 支持<br/>(10 hits, 221.2ms) | ✅ 支持<br/>(10 hits, 9.2ms) | ✅ 支持<br/>(10 hits, 4.3ms) | ✅ 支持<br/>(10 hits, 7.2ms) |
| **大于等于** | ✅ 支持<br/>(10 hits, 253.4ms) | ✅ 支持<br/>(10 hits, 4.6ms) | ✅ 支持<br/>(10 hits, 4.6ms) | ✅ 支持<br/>(10 hits, 8.1ms) |
| **小于** | ✅ 支持<br/>(10 hits, 223.2ms) | ✅ 支持<br/>(10 hits, 4.6ms) | ✅ 支持<br/>(10 hits, 3.8ms) | ✅ 支持<br/>(10 hits, 6.1ms) |
| **范围查询** | ✅ 支持<br/>(10 hits, 305.9ms) | ✅ 支持<br/>(10 hits, 9.3ms) | ✅ 支持<br/>(10 hits, 4.0ms) | ✅ 支持<br/>(10 hits, 8.5ms) |
| **浮点数范围** | ✅ 支持<br/>(10 hits, 341.2ms) | ✅ 支持<br/>(10 hits, 5.9ms) | ✅ 支持<br/>(10 hits, 3.4ms) | ✅ 支持<br/>(10 hits, 6.6ms) |

## SET OPS

| 查询类型 | Milvus | Qdrant | Chroma | Weaviate |
|----------|--------|--------|--------|----------|
| **IN查询** | ✅ 支持<br/>(10 hits, 334.3ms) | ✅ 支持<br/>(10 hits, 4.4ms) | ✅ 支持<br/>(10 hits, 3.8ms) | ❌ 不支持 |
| **NOT IN查询** | ✅ 支持<br/>(10 hits, 610.1ms) | ✅ 支持<br/>(10 hits, 5.7ms) | ✅ 支持<br/>(10 hits, 3.3ms) | ❌ 不支持 |

## STRING

| 查询类型 | Milvus | Qdrant | Chroma | Weaviate |
|----------|--------|--------|--------|----------|
| **字符串前缀** | ✅ 支持<br/>(10 hits, 306.9ms) | ❌ 不支持 | ❌ 不支持 | ✅ 支持<br/>(10 hits, 9.6ms) |

## 详细错误信息

### null_handling/IS NULL

- **Milvus**: DataNotMatchException: <DataNotMatchException: (code=1, message=The Input data type is inconsistent with defined schema, pl

### null_handling/IS NOT NULL

- **Milvus**: DataNotMatchException: <DataNotMatchException: (code=1, message=The Input data type is inconsistent with defined schema, pl

### logic/复杂组合

- **Qdrant**: ValidationError: 1 validation error for Filter
min_should
  Input should be a valid dictionary or instance of MinShou

