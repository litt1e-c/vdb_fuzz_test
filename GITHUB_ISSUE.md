# NULL values incorrectly satisfy comparison operators when combined with primary key filters

## Bug Description

When executing a query that combines a primary key exact match (`id == X`) with a scalar field comparison (`field <= value`), Milvus incorrectly returns rows where the scalar field is NULL. According to SQL-92 standard, NULL compared with any value should return UNKNOWN (treated as False in boolean context).

## Environment

- **Milvus version**: v2.6.7
- **Deployment**: Docker
- **SDK**: PyMilvus
- **OS**: Linux

## Steps to Reproduce

### 1. Setup
Collection with nullable DOUBLE field and 5000 rows generated with seed=999.

### 2. Test Data
Row id=3588 has field `c4 = NULL`:
```python
id=3588, c4=None
```

### 3. Execute Query
```python
from pymilvus import connections, Collection

connections.connect("default", host="127.0.0.1", port="19531")
col = Collection("fuzz_stable_v3")
col.load()

# Query: Get row 3588 where c4 <= 7.31
result = col.query(
    "(id == 3588) and (c4 <= 7.3154750146117635)",
    output_fields=["id", "c4"],
    consistency_level="Strong"
)

print(f"Returned: {len(result)} rows")
if result:
    print(f"id={result[0]['id']}, c4={result[0]['c4']}")
```

## Expected Behavior

**Should return 0 rows**

Logic:
- `(id == 3588)` → True
- `(c4 <= 7.31)` where `c4=NULL` → UNKNOWN (False)
- `True AND False` → False

According to SQL-92, NULL in comparison operations returns UNKNOWN, which is treated as False.

## Actual Behavior

**Returns 1 row: id=3588, c4=None** ❌

Milvus incorrectly treats NULL as satisfying the `<= 7.31` condition.

## Key Finding

Interestingly, the bug only occurs when using primary key filters:

```python
# With primary key filter - INCORRECT
query = "(id == 3588) and (c4 <= 7.31)"
result = col.query(query, ...)
# Returns id=3588 (c4=NULL) ❌

# Without primary key filter - CORRECT  
query = "c4 <= 7.31"
result = col.query(query, limit=100, ...)
# Does NOT return id=3588 ✓
```

This suggests the bug is in the **query optimizer** when combining primary key filters with scalar field comparisons.

## Impact

- **Severity**: High
- **Data Correctness**: Queries return incorrect data (false positives)
- **Standard Compliance**: Violates SQL-92 NULL semantics
- **Business Impact**: May cause downstream logic errors

## Reproduction Script

Complete reproduction script available: [issue_repro.py](https://github.com/user/repo/blob/main/issue_repro.py)

```bash
python issue_repro.py
```

## Screenshots/Logs

<details>
<summary>Click to expand full output</summary>

```
================================================================================
Milvus NULL Comparison Bug - Reproduction
================================================================================

[1] Generating test data (seed=999)...
    Test row (id=3588):
      c4 = None  <-- This is NULL

[2] Connecting to Milvus...
    Connected. Collection has 5000 rows.

[3] Executing query...
    Query: (id == 3588) and (c4 <= 7.3154750146117635)

    Expected: 0 rows (NULL should not satisfy <=)
    Logic: True AND (NULL <= 7.31) = True AND False = False

[4] Result:
    Returned: 1 row(s)
      id=3588, c4=None

================================================================================
❌ BUG CONFIRMED
================================================================================

Milvus returned a row where c4=NULL, but NULL should not
satisfy the condition (c4 <= 7.31).

This violates SQL-92 NULL semantics.

--------------------------------------------------------------------------------
[BONUS] Testing without primary key filter...
    Query: c4 <= 7.3154750146117635
    Returned IDs: [1277, 1483, 2948, 3785, 4457]
    Contains id=3588? False

💡 Key Finding:
    • With (id == X): NULL handled incorrectly ❌
    • Without (id == X): NULL handled correctly ✓

    Bug is in the query optimizer when combining
    primary key filters with scalar comparisons.
```

</details>

## Suggested Fix

Ensure NULL handling follows SQL-92 three-valued logic in all query paths:

```
NULL == any_value  → UNKNOWN (False)
NULL != any_value  → UNKNOWN (False)
NULL <  any_value  → UNKNOWN (False)
NULL <= any_value  → UNKNOWN (False)
NULL >  any_value  → UNKNOWN (False)
NULL >= any_value  → UNKNOWN (False)
```

The bug likely resides in the query optimizer code path that handles combined primary key and scalar field filters.

## Additional Context

Discovered through differential fuzzing using Pandas as oracle. Multiple affected rows found in testing (id=3588, 2053, 4466, 4953, etc.).

## Related Issues

(Search for related NULL handling issues)

---

**Note**: Full reproduction code and test data available upon request.
