import sys
import os
import traceback
import time

# Add current directory to path
sys.path.append(os.getcwd())

try:
    import milvus_fuzz_oracle as oracle
except ImportError:
    # If execute from parent dir
    sys.path.append(os.path.join(os.getcwd(), 'compare_test'))
    import milvus_fuzz_oracle as oracle

SEEDS = [1001, 2024, 5555,666,999,1234,777,222,333]
ROUNDS = 100  # Run short rounds to save time, but sufficient to trigger logic

def run_safe(test_name, func, seed):
    print(f"\n{'#'*60}")
    print(f"🚀 Running {test_name} with Seed {seed}")
    print(f"{'#'*60}")
    try:
        func(rounds=ROUNDS, seed=seed)
        print(f"✅ {test_name} [Seed {seed}] PASSED")
        return True
    except Exception as e:
        print(f"❌ {test_name} [Seed {seed}] FAILED")
        traceback.print_exc()
        return False

def main():
    results = {}
    
    # Cleaning up first? MilvusManager in oracle does drop_collection inside reset_collection
    
    for seed in SEEDS:
        print(f"\n⏩ Starting Batch for Seed {seed}...")
        
        # 1. Main Fuzz
        # Main fuzz uses oracle.run
        if not run_safe("Main Fuzz", oracle.run, seed):
            results[f"Main_{seed}"] = "FAIL"
        else:
            results[f"Main_{seed}"] = "PASS"
            
        time.sleep(1)
            
        # 2. PQS
        if not run_safe("PQS Mode", oracle.run_pqs_mode, seed):
            results[f"PQS_{seed}"] = "FAIL"
        else:
            results[f"PQS_{seed}"] = "PASS"

        time.sleep(1)

        # 3. GroupBy
        # Note: Oracle might not have run_groupby_test exposed if it was inside an if block, 
        # but I saw it defined at top level in previous turns.
        if hasattr(oracle, 'run_groupby_test'):
            if not run_safe("GroupBy Mode", oracle.run_groupby_test, seed):
                results[f"GroupBy_{seed}"] = "FAIL"
            else:
                results[f"GroupBy_{seed}"] = "PASS"
        else:
            print("⚠️ GroupBy Mode function not found in module.")
            
        time.sleep(1)
            
        # 4. Equivalence
        if hasattr(oracle, 'run_equivalence_mode'):
            if not run_safe("Equivalence Mode", oracle.run_equivalence_mode, seed):
                 results[f"Equivalence_{seed}"] = "FAIL"
            else:
                 results[f"Equivalence_{seed}"] = "PASS"
        else:
             print("⚠️ Equivalence Mode function not found in module.")
             
    print("\n" + "="*60)
    print("📊 Comprehensive Test Summary")
    print("="*60)
    
    pass_count = 0
    total_count = 0
    
    for k, v in results.items():
        total_count += 1
        if v == "PASS":
            pass_count += 1
        print(f"{k:<20} : {v}")
        
    print("-" * 60)
    print(f"Total: {total_count}, Passed: {pass_count}, Failed: {total_count - pass_count}")

if __name__ == "__main__":
    main()
