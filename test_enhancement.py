#!/usr/bin/env python3
"""
快速测试脚本：验证 PQSQueryGenerator 的增强效果

用法:
  python test_enhancement.py         # 快速测试 (50 轮)
  python test_enhancement.py 500     # 完整测试 (500 轮)
  python test_enhancement.py 1000    # 深度测试 (1000 轮)
"""

import sys
import subprocess
import time
from datetime import datetime

def run_test(rounds=50):
    print(f"""
╔════════════════════════════════════════════════════════════════╗
║  PQSQueryGenerator 增强效果验证                                 ║
║  开始 {rounds} 轮 PQS 模式测试...                                  ║
╚════════════════════════════════════════════════════════════════╝

📊 测试参数:
   轮数: {rounds}
   时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
   
⏱️  预计耗时:
   50 轮:  ~2-3 分钟
   100 轮: ~4-5 分钟
   500 轮: ~20-25 分钟
   1000 轮: ~40-50 分钟

🔍 测试目标:
   ✓ 验证边界值生成器工作正常
   ✓ 验证深层 JSON 下钻能力
   ✓ 检测浮点精度 Bug
   ✓ 检测数组越界问题
   ✓ 检测逻辑否定 Bug
   
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

""")
    
    start_time = time.time()
    
    # 运行 PQS 模式测试
    cmd = [
        "python", "milvus_fuzz_oracle.py",
        "--pqs-rounds", str(rounds)
    ]
    
    print(f"🚀 执行命令: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd="/home/caihao/compare_test")
    
    elapsed = time.time() - start_time
    
    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⏱️  测试完成！
   总耗时: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)
   平均: {elapsed/rounds:.2f} 秒/轮

📈 结果分析:
   ✓ 如果有发现新的 Bug，说明增强成功！
   ✓ 查看输出中的 ❌ 和 ⚠️ 标记
   
📝 输出位置:
   - 控制台: 上述失败案例
   - 日志文件: fuzz_test_*.log
   
下一步:
   1. 查看 log 文件了解具体 Bug 细节
   2. 尝试 1000 轮以发现更多 Bug
   3. 生成 GitHub Issue 报告

════════════════════════════════════════════════════════════════
""")
    
    return result.returncode

if __name__ == "__main__":
    rounds = 50
    
    if len(sys.argv) > 1:
        try:
            rounds = int(sys.argv[1])
        except ValueError:
            print("❌ 用法: python test_enhancement.py [轮数]")
            print("   例如: python test_enhancement.py 500")
            sys.exit(1)
    
    exit_code = run_test(rounds)
    sys.exit(exit_code)
