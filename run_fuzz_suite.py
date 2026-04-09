
import subprocess
import time
import os
import sys

# Color codes for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ORACLE_PATH = os.path.join(SCRIPT_DIR, "milvus_fuzz_oracle.py")

TEST_CONFIG = [
    # (Mode Args, Rounds, Seed List, Description)
    ("", 15, [101, 102, 103, 104], "Standard Fuzzing"),
    ("--no-dynamic-ops", 12, [105, 106], "Standard NoDyn"),
    ("--equiv", 12, [201, 202, 203], "Equivalence Mode"),
    ("--pqs", 80, [301, 302], "PQS Mode"),  # PQS still needs more rounds
    ("--groupby-test", 12, [401, 402], "GroupBy Mode"),
    # Explicit Metric Test
    ("--metric COSINE", 12, [501, 502], "Explicit Metric: COSINE"),
    ("--metric IP", 12, [503, 504], "Explicit Metric: IP"),
]

def run_command(cmd, log_file):
    """Run a shell command and redirect output to a log file."""
    print(f"   Running: {cmd} > {log_file}")
    with open(log_file, "w") as f:
        # Check if python or python3 is hitting the correct venv is up to the user environment,
        # but we assume 'python' is the same interpreter as this script.
        # Construct the full command properly
        process = subprocess.Popen(cmd, shell=True, cwd=SCRIPT_DIR, stdout=f, stderr=subprocess.STDOUT)
        return process.wait()

def analyze_log(log_file, description):
    """Check the log file for success indicators."""
    try:
        with open(log_file, "r") as f:
            lines = f.readlines()
            content = "".join(lines)
    except FileNotFoundError:
        return False, "Log file not found"

    # Common failure indicators
    if "Traceback (most recent call last)" in content or "ERROR" in content:
        # However, some tests might print "ERROR" as part of negative testing?
        # Our fuzz oracle usually prints "CRITICAL ERROR" or python traceback on actual failure.
        # Let's look for known success messages first.
        pass

    success = False
    details = ""

    # Check for Exit Code 0 (implicit in run_command return, but double check content)
    # The oracle prints specific success messages at the end.
    
    # Generic success message for standard fuzzing output
    if "✅ 所有" in content and "轮测试全部通过" in content:
        return True, "Success Message Found"

    # Prioritize explicit success messages
    if "Standard" in description:
        if "所有" in content and "通过" in content: return True, "Success Message Found"
        if "Exit code: 0" in content: return True, "Exit Code 0"

    elif "Equivalence Mode" in description:
        if "所有等价性测试通过" in content: return True, "Success Message Found"

    elif "PQS Mode" in description:
        if "PQS 测试完成" in content: return True, "Success Message Found"
    
    elif "GroupBy Mode" in description:
        if "GroupBy 测试完成" in content: return True, "Success Message Found"

    elif "Explicit Metric" in description:
         if "Exit code: 0" in content: return True, "Exit Code 0"

    # Only check for failure markers if no success message found
    if "CRITICAL ERROR" in content or "Traceback" in content:
        return False, "Traceback/Error detected"
    
    if "MISMATCH" in content:
        return False, "Data Mismatch detected"

    return False, "Unknown Result"

def main():
    print(f"{GREEN}🚀 Starting Milvus Fuzz Oracle Integration Suite{RESET}")
    print("=" * 60)
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []

    # Ensure timestamp for unique logs
    timestamp = int(time.time())
    
    for mode_arg, rounds, seeds, desc in TEST_CONFIG:
        print(f"\n📂 Group: {desc}")
        
        for seed in seeds:
            total_tests += 1
            log_filename = f"suite_test_{timestamp}_{seed}_{desc.split()[0]}.log"
            log_path = os.path.join(SCRIPT_DIR, log_filename)
            cmd = f"{sys.executable} {ORACLE_PATH} --seed {seed} --rounds {rounds} {mode_arg} --consistency Strong"
            
            # 1. Run
            exit_code = run_command(cmd, log_path)
            
            # 2. Analyze
            is_success, details = analyze_log(log_path, desc)
            
            if exit_code == 0 and is_success:
                print(f"   ✅ Seed {seed}: PASS")
                passed_tests += 1
            else:
                print(f"   ❌ Seed {seed}: FAIL (Exit Code: {exit_code}, Details: {details})")
                print(f"      Log: {log_path}")
                failed_tests.append((desc, seed, log_path))

    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print(f"Total: {total_tests}")
    print(f"Passed: {GREEN}{passed_tests}{RESET}")
    print(f"Failed: {RED}{len(failed_tests)}{RESET}")
    
    if failed_tests:
        print("\nFailures:")
        for desc, seed, log in failed_tests:
            print(f"- {desc} (Seed {seed}): See {log}")
        sys.exit(1)
    else:
        print(f"\n{GREEN}🎉 All system tests passed!{RESET}")
        sys.exit(0)

if __name__ == "__main__":
    main()
