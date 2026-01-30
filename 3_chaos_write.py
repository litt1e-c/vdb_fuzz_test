import time
import random
import numpy as np
from pymilvus import connections, Collection

HOST = '127.0.0.1'
PORT = '19530'
COLLECTION_NAME = "crash_test_empty_v2523"

def chaos_loop():
    print("Connecting...")
    connections.connect("default", host=HOST, port=PORT)
    col = Collection(COLLECTION_NAME)
    
    print("Starting Chaos Write/Delete Loop...")
    while True:
        try:
            # 1. 插入少量数据 (干扰内存视图)
            ids = [random.randint(10000000, 99999999) for _ in range(100)]
            vectors = np.random.random((100, 128)).astype(np.float32)
            # 随机 flag，有时为1，有时为0
            flags = [random.choice([0, 1]) for _ in range(100)] 
            group_ids = [0] * 100
            json_data = [{"tag": "chaos"} for _ in range(100)]
            
            col.insert([ids, vectors, group_ids, flags, json_data])
            
            # 2. 立即 Flush (强制落盘，最耗资源)
            # 注意：频繁 Flush 会严重拖慢系统，每隔几秒做一次即可
            if random.random() < 0.1: 
                print(">>> FLUSHING...")
                col.flush()
            
            # 3. 随机删一点数据
            # col.delete(f"id in {ids[:10]}") 
            
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Chaos Error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    chaos_loop()