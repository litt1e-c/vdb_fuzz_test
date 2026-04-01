#!/usr/bin/env python3
"""
Qdrant Fuzz Oracle — 集成测试套件
=================================
自动组合不同 模式 × 种子 × 轮次 × 功能开关 运行 qdrant_fuzz_oracle.py，
收集日志并汇总结果。

用法:
    python run_qdrant_fuzz_suite.py            # 运行全部预设用例
    python run_qdrant_fuzz_suite.py --quick     # 快速冒烟（轮次减半）
"""

import subprocess
import time
import os
import sys
import argparse

# ── 终端颜色 ──────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ORACLE_PATH  = os.path.join(SCRIPT_DIR, "qdrant_fuzz_oracle.py")
LOG_ROOT_DIR = os.path.join(SCRIPT_DIR, "qdrant_log")

# ── 测试矩阵 ─────────────────────────────────────────────
# 格式: (extra_args, rounds, seed_list, description)
#   extra_args  : 追加到命令行的参数字符串
#   rounds      : --rounds / --pqs-rounds 的值
#   seed_list   : 要跑的种子列表
#   description : 显示在报告中的名称
TEST_MATRIX = [
    # ━━━━━ Oracle 模式 ━━━━━
    ("--oracle",                          50,  [42, 1337, 9999],       "Oracle-Basic"),
    ("--oracle --dynamic",                80,  [100, 200],             "Oracle-Dynamic"),
    ("--oracle --dynamic --chaos",        80,  [300, 400],             "Oracle-Dynamic+Chaos10%"),
    ("--oracle --dynamic --chaos-rate 0.3", 60, [500],                 "Oracle-Dynamic+Chaos30%"),

    # ━━━━━ Equivalence 模式 ━━━━━
    ("--equiv",                           50,  [2001, 2002],           "Equiv-Basic"),
    ("--equiv --dynamic",                 80,  [2003, 2004],           "Equiv-Dynamic"),

    # ━━━━━ PQS 模式 ━━━━━
    ("--pqs",                            100,  [3001, 3002],           "PQS-Basic"),
    ("--pqs --dynamic",                  120,  [3003],                 "PQS-Dynamic"),

    # ━━━━━ GroupBy 模式 ━━━━━
    ("--group",                           50,  [4001, 4002],           "GroupBy-Basic"),
    ("--group --dynamic",                 80,  [4003],                 "GroupBy-Dynamic"),

    # ━━━━━ 无种子测试（验证自动种子生成和不可预测输入）━━━━━
    ("--oracle --dynamic",                40,  [None],                 "Oracle-RandomSeed"),
    ("--pqs --dynamic",                   60,  [None],                 "PQS-RandomSeed"),
    ("--equiv --dynamic",                 40,  [None],                 "Equiv-RandomSeed"),
    ("--group --dynamic",                 40,  [None],                 "GroupBy-RandomSeed"),

    # ━━━━━ 压力测试（高轮次 + 全开）━━━━━
    ("--oracle --dynamic --chaos-rate 0.2", 300, [31337],              "Oracle-Stress"),
]


# ── 工具函数 ──────────────────────────────────────────────

def build_cmd(extra_args: str, rounds: int, seed) -> str:
    """拼接完整命令行。"""
    parts = [sys.executable, ORACLE_PATH]
    parts += extra_args.split()

    # PQS 用 --pqs-rounds，其余用 --rounds
    if "--pqs" in extra_args:
        parts += ["--pqs-rounds", str(rounds)]
    else:
        parts += ["--rounds", str(rounds)]

    if seed is not None:
        parts += ["--seed", str(seed)]

    return " ".join(parts)


def run_one(cmd: str, log_path: str) -> int:
    """执行命令，stdout/stderr 重定向到 log_path, 返回 exit code。"""
    with open(log_path, "w") as fout:
        proc = subprocess.Popen(
            cmd, shell=True, cwd=SCRIPT_DIR,
            stdout=fout, stderr=subprocess.STDOUT,
        )
        return proc.wait()


def analyze_log(log_path: str, description: str):
    """
    解析日志判定结果。
    返回 (is_pass: bool, detail: str)
    """
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except FileNotFoundError:
        return False, "日志文件不存在"

    # ── 显式成功消息 ──
    success_markers = [
        "所有",                       # "所有 N 轮测试全部通过"
        "所有等价性测试通过",
        "PQS 测试完成",
        "GroupBy 测试完成",
    ]
    found_success = any(m in content for m in success_markers)

    # ── 失败标记 ──
    has_traceback  = "Traceback (most recent call last)" in content
    has_mismatch   = "MISMATCH" in content

    # file_log 里的 ERROR 行（排除 VectorCheck ERROR 0 这种统计行）
    error_lines = [
        ln for ln in content.splitlines()
        if "ERROR" in ln
        and "VectorCheck.*ERROR" not in ln   # 排除统计汇总
        and "ERROR: 0" not in ln             # 排除 count=0
    ]
    has_real_error = len(error_lines) > 0

    # ── 判定 ──
    if has_traceback:
        return False, "Python Traceback 异常"
    if has_mismatch:
        return False, f"MISMATCH 检测到 (共 {content.count('MISMATCH')} 处)"
    if found_success:
        return True, "成功"
    if has_real_error:
        return False, f"日志含 ERROR 行 ({len(error_lines)} 条)"

    # 没有明确成功消息也没有失败标记 → 归为未知
    return False, "未检测到成功消息（可能超时/崩溃）"


def extract_seed_from_log(log_path: str):
    """尝试从日志中提取实际使用的种子。"""
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if "Seed:" in line or "seed:" in line or "种子" in line:
                    # 尝试提取数字
                    import re
                    m = re.search(r"[Ss]eed[:\s]+(\d+)", line)
                    if m:
                        return m.group(1)
                    m = re.search(r"种子[:\s]*(\d+)", line)
                    if m:
                        return m.group(1)
    except Exception:
        pass
    return None


def fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s}s"


# ── 主流程 ────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Qdrant Fuzz Oracle 集成测试套件")
    ap.add_argument("--quick", action="store_true",
                    help="快速冒烟模式：所有轮次减半")
    ap.add_argument("--stress-only", action="store_true",
                    help="只跑压力测试用例")
    ap.add_argument("--no-stress", action="store_true",
                    help="跳过压力测试用例")
    args = ap.parse_args()

    # 过滤测试矩阵
    matrix = TEST_MATRIX[:]
    if args.stress_only:
        matrix = [t for t in matrix if "Stress" in t[3]]
    elif args.no_stress:
        matrix = [t for t in matrix if "Stress" not in t[3]]

    # quick 模式轮次减半
    if args.quick:
        matrix = [(a, max(r // 2, 5), s, d) for a, r, s, d in matrix]

    timestamp   = int(time.time())
    log_dir     = os.path.join(LOG_ROOT_DIR, f"qdrant_suite_logs_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    total_tests  = sum(len(seeds) for _, _, seeds, _ in matrix)
    passed_tests = 0
    failed_list  = []    # (desc, seed, log_path, detail)
    results      = []    # for final table

    print(f"\n{BOLD}{CYAN}{'='*68}{RESET}")
    print(f"{BOLD}{CYAN}  🚀 Qdrant Fuzz Oracle 集成测试套件{RESET}")
    print(f"{BOLD}{CYAN}{'='*68}{RESET}")
    print(f"  脚本:     {ORACLE_PATH}")
    print(f"  日志目录: {log_dir}")
    print(f"  用例总数: {total_tests}")
    print(f"  模式:     {'快速冒烟' if args.quick else '完整'}")
    print(f"{CYAN}{'='*68}{RESET}\n")

    suite_start = time.time()
    test_idx = 0

    for extra_args, rounds, seeds, desc in matrix:
        print(f"{BOLD}📂 [{desc}]{RESET}  rounds={rounds}  seeds={seeds}")

        for seed in seeds:
            test_idx += 1
            seed_label = str(seed) if seed is not None else "random"
            safe_desc  = desc.replace(" ", "_").replace("%", "pct").replace("+", "_")
            log_name   = f"{safe_desc}_seed{seed_label}.log"
            log_path   = os.path.join(log_dir, log_name)

            cmd = build_cmd(extra_args, rounds, seed)
            print(f"   [{test_idx}/{total_tests}] seed={seed_label} ", end="", flush=True)

            t0 = time.time()
            exit_code = run_one(cmd, log_path)
            elapsed = time.time() - t0

            is_pass, detail = analyze_log(log_path, desc)
            actual_seed = extract_seed_from_log(log_path) or seed_label

            if exit_code == 0 and is_pass:
                passed_tests += 1
                print(f"{GREEN}✅ PASS{RESET}  ({fmt_duration(elapsed)}, seed={actual_seed})")
                results.append((desc, seed_label, actual_seed, "PASS", detail, elapsed))
            else:
                detail_full = f"exit={exit_code}, {detail}"
                print(f"{RED}❌ FAIL{RESET}  ({fmt_duration(elapsed)}, {detail_full})")
                failed_list.append((desc, seed_label, log_path, detail_full))
                results.append((desc, seed_label, actual_seed, "FAIL", detail_full, elapsed))

        print()  # group separator

    suite_elapsed = time.time() - suite_start

    # ── 汇总报告 ──────────────────────────────────────────
    print(f"\n{BOLD}{'='*68}{RESET}")
    print(f"{BOLD}📊  测试结果汇总{RESET}")
    print(f"{'='*68}")

    # 表格
    hdr = f"{'Group':<28} {'Seed':<8} {'ActualSeed':<12} {'Result':<6} {'Time':<8} Detail"
    print(hdr)
    print("-" * 90)
    for desc, seed_label, actual_seed, result, detail, elapsed in results:
        color = GREEN if result == "PASS" else RED
        print(f"{desc:<28} {seed_label:<8} {actual_seed:<12} "
              f"{color}{result:<6}{RESET} {fmt_duration(elapsed):<8} {detail}")

    print(f"\n{'─'*68}")
    print(f"  总用例:  {total_tests}")
    print(f"  通过:    {GREEN}{passed_tests}{RESET}")
    print(f"  失败:    {RED}{len(failed_list)}{RESET}")
    print(f"  总耗时:  {fmt_duration(suite_elapsed)}")
    print(f"  日志目录: {log_dir}")

    if failed_list:
        print(f"\n{RED}{BOLD}失败详情:{RESET}")
        for desc, seed_label, log_path, detail in failed_list:
            print(f"  • {desc} (seed={seed_label}): {detail}")
            print(f"    日志: {log_path}")
        sys.exit(1)
    else:
        print(f"\n{GREEN}{BOLD}🎉 全部 {total_tests} 个测试用例通过！{RESET}")
        sys.exit(0)


if __name__ == "__main__":
    main()
