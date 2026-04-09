import multiprocessing
import random
from pymilvus import connections, utility
# 假设你的主逻辑在 milvus_fuzz_oracle.py 中
import milvus_fuzz_oracle as milvus_fuzzer

def worker(proc_id, rounds, collection_name):
    # 💡 每个进程需要独立的连接
    connections.connect(alias="default", host="127.0.0.1", port="19531")
    
    try:
        print(f"🚀 进程 {proc_id} 开始测试集合: {collection_name}")
        # 设置每个进程独立的集合名
        milvus_fuzzer.COLLECTION_NAME = collection_name
        # 调用 run
        milvus_fuzzer.run(rounds=rounds)
    except Exception as e:
        print(f"❌ 进程 {proc_id} 运行出错: {e}")
    finally:
        # 🗑️ 无论测试成功还是失败，最终都会执行清理
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            print(f"✅ 进程 {proc_id} 已释放集合: {collection_name}")
        connections.disconnect("default")

if __name__ == "__main__":
    processes = []
    # 启动 4 个并行任务
    for i in range(4):
        coll_name = f"fuzz_test_{i}"
        p = multiprocessing.Process(target=worker, args=(i, 500, coll_name))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()