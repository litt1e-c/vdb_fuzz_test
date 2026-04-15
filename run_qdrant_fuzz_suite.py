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
import shlex
import csv
import json

# ── 终端颜色 ──────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ORACLE_PATH  = os.path.join(SCRIPT_DIR, "qdrant_fuzz_oracle.py")
DEFAULT_ARTIFACT_ROOT = os.environ.get(
    "QDRANT_EXPERIMENT_ROOT",
    os.path.join(os.path.expanduser("~"), "qdrant_artifacts"),
)
LOG_ROOT_DIR = os.path.join(DEFAULT_ARTIFACT_ROOT, "qdrant_log")

# ── 测试矩阵 ─────────────────────────────────────────────
# 格式: (extra_args, rounds, seed_list, description)
#   extra_args  : 追加到命令行的参数字符串
#   rounds      : --rounds / --pqs-rounds 的值
#   seed_list   : 要跑的种子列表
#   description : 显示在报告中的名称
TEST_MATRIX = [
    # ━━━━━ Oracle 模式 ━━━━━
    ("--oracle",                          50,  [42, 1337, 9999],       "Oracle-Basic"),
    ("--oracle --scroll-mode paged",      35,  [271, 272],             "Oracle-PagedScroll"),
    ("--oracle --dynamic",                80,  [100, 200],             "Oracle-Dynamic"),
    ("--oracle --dynamic --payload-mutations", 70, [610, 611],         "Oracle-Dynamic+PayloadMutation"),
    ("--oracle --dynamic --evo-null-sync", 70, [260, 261],             "Oracle-Dynamic+EvoNull"),
    ("--oracle --dynamic --chaos",        80,  [300, 400],             "Oracle-Dynamic+Chaos10%"),
    ("--oracle --dynamic --chaos-rate 0.3", 60, [500],                 "Oracle-Dynamic+Chaos30%"),
    ("--oracle --include-known-int64-boundaries", 40, [1701],          "Oracle-Int64Boundary"),

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

def build_cmd(extra_args: str, rounds: int, seed, args: argparse.Namespace) -> list[str]:
    """拼接完整命令行。"""
    parts = [sys.executable, ORACLE_PATH]
    parts += shlex.split(extra_args)

    # PQS 用 --pqs-rounds，其余用 --rounds
    if "--pqs" in extra_args:
        parts += ["--pqs-rounds", str(rounds)]
    else:
        parts += ["--rounds", str(rounds)]

    if seed is not None:
        parts += ["--seed", str(seed)]

    parts += ["--host", args.host, "--port", str(args.port), "--grpc-port", str(args.grpc_port)]

    if args.prefer_grpc:
        parts.append("--prefer-grpc")
    if args.read_consistency != "all":
        parts += ["--read-consistency", args.read_consistency]
    if args.write_ordering != "strong":
        parts += ["--write-ordering", args.write_ordering]
    if args.rows is not None:
        parts += ["-N", str(args.rows)]
    if args.dim is not None:
        parts += ["--dim", str(args.dim)]
    if args.batch_size is not None:
        parts += ["--batch-size", str(args.batch_size)]
    if args.sleep_interval is not None:
        parts += ["--sleep-interval", str(args.sleep_interval)]
    if args.distance is not None:
        parts += ["--distance", args.distance]
    if args.oracle_log_dir:
        parts += ["--log-dir", args.oracle_log_dir]
    if args.run_id:
        seed_label = str(seed) if seed is not None else "random"
        safe_mode = "_".join(shlex.split(extra_args)).replace("-", "").replace("_", "-")
        parts += ["--run-id", f"{args.run_id}-{safe_mode}-seed{seed_label}"]
    if args.oracle_args:
        parts += shlex.split(args.oracle_args)

    return parts


def run_one(cmd: list[str], log_path: str) -> int:
    """执行命令，stdout/stderr 重定向到 log_path, 返回 exit code。"""
    with open(log_path, "w") as fout:
        proc = subprocess.Popen(
            cmd, cwd=SCRIPT_DIR,
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
    if found_success:
        return True, "成功"
    if has_mismatch:
        return False, f"MISMATCH 检测到 (共 {content.count('MISMATCH')} 处)"
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


def slugify(value: object, max_len: int = 72) -> str:
    text = str(value).strip().lower()
    pieces = [ch if ch.isalnum() else "-" for ch in text]
    collapsed = "".join(pieces).strip("-")
    while "--" in collapsed:
        collapsed = collapsed.replace("--", "-")
    return (collapsed or "x")[:max_len]


def write_case_results_csv(path: str, rows: list[dict[str, object]]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = [
        "group",
        "seed",
        "actual_seed",
        "result",
        "elapsed_seconds",
        "detail",
        "log_path",
        "command",
        "reproduce_command",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_summary_json(path: str, payload: dict[str, object]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, indent=2, ensure_ascii=False)
        fout.write("\n")


# ── 主流程 ────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Qdrant Fuzz Oracle 集成测试套件")
    ap.add_argument("--quick", action="store_true",
                    help="快速冒烟模式：所有轮次减半")
    ap.add_argument("--stress-only", action="store_true",
                    help="只跑压力测试用例")
    ap.add_argument("--no-stress", action="store_true",
                    help="跳过压力测试用例")
    ap.add_argument("--max-tests", type=int, default=None,
                    help="仅运行前 N 个展开后的测试用例（用于快速 smoke）")
    ap.add_argument("--host", default="127.0.0.1",
                    help="Qdrant REST 主机地址 (默认: 127.0.0.1)")
    ap.add_argument("--port", type=int, default=6333,
                    help="Qdrant REST 端口 (默认: 6333)")
    ap.add_argument("--grpc-port", type=int, default=6334, dest="grpc_port",
                    help="Qdrant gRPC 端口 (默认: 6334)")
    ap.add_argument("--prefer-grpc", action="store_true",
                    help="优先使用 gRPC 查询路径（仍保留 REST 端口）")
    ap.add_argument("--read-consistency",
                    choices=["all", "majority", "quorum", "1", "random"],
                    default="all",
                    help="Qdrant 读一致性策略透传 (默认: all)")
    ap.add_argument("--write-ordering",
                    choices=["weak", "medium", "strong", "random"],
                    default="strong",
                    help="Qdrant 写 ordering 策略透传 (默认: strong)")
    ap.add_argument("--rows", type=int, default=None,
                    help="透传到 qdrant_fuzz_oracle.py 的 -N/--rows")
    ap.add_argument("--dim", type=int, default=None,
                    help="透传到 qdrant_fuzz_oracle.py 的 --dim")
    ap.add_argument("--batch-size", type=int, default=None, dest="batch_size",
                    help="透传到 qdrant_fuzz_oracle.py 的 --batch-size")
    ap.add_argument("--sleep-interval", type=float, default=None, dest="sleep_interval",
                    help="透传到 qdrant_fuzz_oracle.py 的 --sleep-interval")
    ap.add_argument("--distance",
                    choices=["random", "euclid", "cosine", "dot", "manhattan"],
                    default=None,
                    help="透传到 qdrant_fuzz_oracle.py 的 --distance")
    ap.add_argument("--oracle-log-dir", default=None,
                    help="透传到 qdrant_fuzz_oracle.py 的 --log-dir")
    ap.add_argument("--oracle-args", default=None,
                    help="原样追加到每个 qdrant_fuzz_oracle.py 命令的额外参数字符串")
    ap.add_argument("--run-id", default=None,
                    help="可复现实验 run id；用于稳定日志目录和子命令 --run-id")
    ap.add_argument("--log-dir", default=None,
                    help="本 suite 自身日志目录；默认 qdrant_log/qdrant_suite_<run-id或timestamp>")
    ap.add_argument("--summary-json", default=None,
                    help="输出 suite_summary.json 路径")
    ap.add_argument("--case-results-csv", default=None,
                    help="输出逐 case 结果 CSV 路径")
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

    suite_run_id = slugify(args.run_id, max_len=96) if args.run_id else f"logs_{int(time.time())}"
    log_dir = os.path.abspath(os.path.expanduser(args.log_dir)) if args.log_dir else os.path.join(LOG_ROOT_DIR, f"qdrant_suite_{suite_run_id}")
    os.makedirs(log_dir, exist_ok=True)
    summary_json_path = os.path.abspath(os.path.expanduser(args.summary_json)) if args.summary_json else os.path.join(log_dir, "suite_summary.json")
    case_results_csv_path = os.path.abspath(os.path.expanduser(args.case_results_csv)) if args.case_results_csv else os.path.join(log_dir, "case_results.csv")

    total_tests  = sum(len(seeds) for _, _, seeds, _ in matrix)
    if args.max_tests is not None:
        total_tests = min(total_tests, max(0, args.max_tests))
    passed_tests = 0
    failed_list  = []    # (desc, seed, log_path, detail)
    results      = []    # for final table
    case_rows    = []    # stable CSV/JSON rows

    print(f"\n{BOLD}{CYAN}{'='*68}{RESET}")
    print(f"{BOLD}{CYAN}  🚀 Qdrant Fuzz Oracle 集成测试套件{RESET}")
    print(f"{BOLD}{CYAN}{'='*68}{RESET}")
    print(f"  脚本:     {ORACLE_PATH}")
    print(f"  日志目录: {log_dir}")
    print(f"  Run ID:   {args.run_id or '(timestamped)'}")
    print(f"  用例总数: {total_tests}")
    print(f"  模式:     {'快速冒烟' if args.quick else '完整'}")
    print(f"  目标:     {args.host}:{args.port} (grpc:{args.grpc_port})")
    print(f"  传输:     {'prefer-grpc' if args.prefer_grpc else 'rest-default'}")
    if args.rows is not None or args.dim is not None:
        print(f"  数据规模: rows={args.rows or 'default'} dim={args.dim or 'default'}")
    if args.oracle_log_dir:
        print(f"  Oracle日志: {args.oracle_log_dir}")
    print(f"{CYAN}{'='*68}{RESET}\n")

    suite_start = time.time()
    test_idx = 0

    for extra_args, rounds, seeds, desc in matrix:
        print(f"{BOLD}📂 [{desc}]{RESET}  rounds={rounds}  seeds={seeds}")

        for seed in seeds:
            if args.max_tests is not None and test_idx >= args.max_tests:
                break
            test_idx += 1
            seed_label = str(seed) if seed is not None else "random"
            safe_desc  = desc.replace(" ", "_").replace("%", "pct").replace("+", "_")
            log_name   = f"{safe_desc}_seed{seed_label}.log"
            log_path   = os.path.join(log_dir, log_name)

            cmd = build_cmd(extra_args, rounds, seed, args)
            command_str = shlex.join(cmd)
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
                case_rows.append({
                    "group": desc,
                    "seed": seed_label,
                    "actual_seed": actual_seed,
                    "result": "PASS",
                    "elapsed_seconds": f"{elapsed:.3f}",
                    "detail": detail,
                    "log_path": log_path,
                    "command": command_str,
                    "reproduce_command": command_str,
                })
            else:
                detail_full = f"exit={exit_code}, {detail}"
                print(f"{RED}❌ FAIL{RESET}  ({fmt_duration(elapsed)}, {detail_full})")
                failed_list.append((desc, seed_label, log_path, detail_full))
                results.append((desc, seed_label, actual_seed, "FAIL", detail_full, elapsed))
                case_rows.append({
                    "group": desc,
                    "seed": seed_label,
                    "actual_seed": actual_seed,
                    "result": "FAIL",
                    "elapsed_seconds": f"{elapsed:.3f}",
                    "detail": detail_full,
                    "log_path": log_path,
                    "command": command_str,
                    "reproduce_command": command_str,
                })

        print()  # group separator
        if args.max_tests is not None and test_idx >= args.max_tests:
            break

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
    print(f"  Summary: {summary_json_path}")
    print(f"  CSV:     {case_results_csv_path}")

    summary_payload = {
        "run_id": args.run_id,
        "suite_passed": len(failed_list) == 0,
        "log_dir": log_dir,
        "summary_json": summary_json_path,
        "case_results_csv": case_results_csv_path,
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "failed_tests": len(failed_list),
        "suite_elapsed_seconds": suite_elapsed,
        "quick": bool(args.quick),
        "stress_only": bool(args.stress_only),
        "no_stress": bool(args.no_stress),
        "host": args.host,
        "port": args.port,
        "grpc_port": args.grpc_port,
        "prefer_grpc": bool(args.prefer_grpc),
        "rows": args.rows,
        "dim": args.dim,
        "batch_size": args.batch_size,
        "sleep_interval": args.sleep_interval,
        "distance": args.distance,
        "cases": case_rows,
    }
    write_case_results_csv(case_results_csv_path, case_rows)
    write_summary_json(summary_json_path, summary_payload)

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
