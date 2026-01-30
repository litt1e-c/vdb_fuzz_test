"""
Milvus NULL Comparison Bug - Minimal Reproduction

Bug: NULL values incorrectly satisfy comparison operators when using primary key filters

Environment:
- Milvus: v2.6.7
- PyMilvus: 2.6.5
- Python: 3.11.5
- Pandas: 1.5.3
- NumPy: 1.26.4
"""
from pymilvus import connections, Collection
import pymilvus
import pandas as pd
import random
import numpy as np
from milvus_fuzz_oracle import DataManager
import sys

# Configuration
HOST = "127.0.0.1"
PORT = "19531"
COLLECTION = "fuzz_stable_v3"
SEED = 999
TEST_ID = 3588

print("=" * 80)
print("Milvus NULL Comparison Bug - Reproduction")
print("=" * 80)
print(f"\nEnvironment:")
print(f"  • Python: {sys.version.split()[0]}")
print(f"  • PyMilvus: {pymilvus.__version__}")
print(f"  • Pandas: {pd.__version__}")
print(f"  • NumPy: {np.__version__}")
print(f"  • Milvus Server: v2.6.7")
print("=" * 80)

# Step 1: Generate test data
print("\n[1] Generating test data (seed=999)...")
random.seed(SEED)
np.random.seed(SEED)
dm = DataManager()
dm.generate_schema()
dm.generate_data()

# Show the test row
test_row = dm.df[dm.df['id'] == TEST_ID].iloc[0]
print(f"\n    Test row (id={TEST_ID}):")
print(f"      c4 = {test_row['c4']}  <-- This is NULL\n")

# Step 2: Connect to Milvus
print("[2] Connecting to Milvus...")
connections.connect("default", host=HOST, port=PORT)
col = Collection(COLLECTION)
col.load()
print(f"    Connected. Collection has {col.num_entities} rows.\n")

# Step 3: Execute query
print("[3] Executing query...")
query = "(id == 3588) and (c4 <= 7.3154750146117635)"
print(f"    Query: {query}")
print(f"\n    Expected: 0 rows (NULL should not satisfy <=)")
print(f"    Logic: True AND (NULL <= 7.31) = True AND False = False\n")

result = col.query(
    query,
    output_fields=["id", "c4"],
    consistency_level="Strong"
)

# Step 4: Show result
print("[4] Result:")
print(f"    Returned: {len(result)} row(s)")

if len(result) > 0:
    for r in result:
        print(f"      id={r['id']}, c4={r['c4']}")
    
    print("\n" + "=" * 80)
    print("❌ BUG CONFIRMED")
    print("=" * 80)
    print("\nMilvus returned a row where c4=NULL, but NULL should not")
    print("satisfy the condition (c4 <= 7.31).")
    print("\nThis violates SQL-92 NULL semantics.")
else:
    print("\n✓ No rows returned (correct behavior)")

# Bonus: Comparison test
print("\n" + "-" * 80)
print("[BONUS] Testing without primary key filter...")
query2 = "c4 <= 7.3154750146117635"
result2 = col.query(query2, output_fields=["id"], limit=10, consistency_level="Strong")
ids = [r['id'] for r in result2]
print(f"    Query: {query2}")
print(f"    Returned IDs: {ids}")
print(f"    Contains id=3588? {3588 in ids}")

if 3588 not in ids:
    print("\n💡 Key Finding:")
    print("    • With (id == X): NULL handled incorrectly ❌")
    print("    • Without (id == X): NULL handled correctly ✓")
    print("\n    Bug is in the query optimizer when combining")
    print("    primary key filters with scalar comparisons.")

print("\n" + "=" * 80)
