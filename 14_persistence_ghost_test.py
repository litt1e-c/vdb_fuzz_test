import time
import random
import docker
import threading
import numpy as np
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import weaviate
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'
# --- 🔥 高压配置 ---
HOST = "127.0.0.1"
DIM = 128
TOTAL_ROUNDS = 5000      # 每一轮操作的数据量
CHAOS_ROUNDS = 3         # 重复测试 3 次（复活 3 次）
KILL_DELAY_RANGE = (2, 5) # 在操作开始后 2-5 秒内随机杀掉进程

# Docker 控制器
docker_client = docker.from_env()

def restart_container(keyword):
    """暴力重启容器 (模拟断电)"""
    print(f"   ⚡ [CHAOS] Hunting for container '{keyword}'...")
    containers = docker_client.containers.list()
    target = None
    for c in containers:
        # 模糊匹配，找到 deploy_all 里的对应容器
        if keyword in c.name and "minio" not in c.name and "etcd" not in c.name:
            target = c
            break
    
    if target:
        print(f"   ⚡ [CHAOS] KILLING {target.name} NOW! (timeout=0)")
        target.restart(timeout=0) # 0秒超时 = 立即 Kill
        print(f"   🚑 [RECOVERY] Waiting for {target.name} to come back...")
        
        # 轮询等待端口通畅
        for i in range(60):
            if target.status == 'running':
                # 简单 sleep 等待服务初始化
                time.sleep(2)
                # 这里可以加更复杂的端口探测，暂且用 sleep 代替
                if i > 10: break 
            time.sleep(1)
        print(f"   ✅ [RECOVERY] Container {target.name} is back online.")
        time.sleep(10) # 给数据库一点喘息时间恢复 WAL
    else:
        print(f"   ❌ Container not found for keyword '{keyword}'")

# ==========================================
# 1. Weaviate 压力测试
# ==========================================
def run_weaviate_stress():
    print(f"\n{'='*20} STARTING WEAVIATE STRESS TEST {'='*20}")
    col_name = "StressGhost"
    client = weaviate.Client(f"http://{HOST}:8080")
    
    # Cleanup
    client.schema.delete_all()
    client.schema.create_class({
        "class": col_name, "vectorizer": "none",
        "properties": [{"name": "idx", "dataType": ["int"]}]
    })

    # 目标 ID (我们将删除它，然后看它会不会复活)
    target_ids = list(range(0, 1000)) 
    
    # --- 阶段 1: 铺底数据 ---
    print(f"   📥 Seeding {len(target_ids)} base objects...")
    with client.batch as batch:
        batch.batch_size = 100
        for uid in target_ids:
            batch.add_data_object({"idx": uid}, col_name, vector=[0.1]*DIM)
            
    # --- 阶段 2: 混沌循环 ---
    # 启动一个线程疯狂写入干扰数据，主线程执行删除，然后突然重启
    
    def noise_injector():
        c = weaviate.Client(f"http://{HOST}:8080")
        try:
            with c.batch as batch:
                batch.batch_size = 50
                for i in range(10000, 20000): # 插入 1万条噪音
                    batch.add_data_object({"idx": i}, col_name, vector=[0.1]*DIM)
                    time.sleep(0.001)
        except: pass # 忽略连接被切断的报错

    print("   🌪️  Starting Noise Injector (Insert Storm)...")
    t = threading.Thread(target=noise_injector)
    t.start()
    
    # 执行删除操作
    print(f"   🗑️  Deleting target IDs (0-999)...")
    try:
        # 批量删除
        for uid in target_ids:
            client.batch.delete_objects(
                class_name=col_name,
                where={"path": ["idx"], "operator": "Equal", "valueInt": uid}
            )
    except:
        print("   ⚠️  Delete interrupted (Expected)")

    # 随机时间截杀
    time.sleep(random.uniform(*KILL_DELAY_RANGE))
    restart_container("weaviate")
    
    # --- 阶段 3: 验尸 ---
    print("   🔍 Verifying Consistency...")
    client = weaviate.Client(f"http://{HOST}:8080")
    
    # 检查 ID 0 是否复活
    res = client.query.get(col_name, ["idx"]).with_where({
        "path": ["idx"], "operator": "Equal", "valueInt": 0
    }).do()
    
    # 检查 ID 500 是否复活 (防止删除只执行了一半)
    res_mid = client.query.get(col_name, ["idx"]).with_where({
        "path": ["idx"], "operator": "Equal", "valueInt": 500
    }).do()

    zombies = []
    if res['data']['Get'][col_name]: zombies.append(0)
    if res_mid['data']['Get'][col_name]: zombies.append(500)
    
    if zombies:
        print(f"   {RED}❌ CRITICAL FAIL: Zombie Data Found! IDs {zombies} resurrected.{RESET}")
        print("   Reason: Tombstones were lost during crash.")
    else:
        print(f"   {GREEN}✅ PASS: Deleted data remained deleted.{RESET}")

# ==========================================
# 2. Milvus 压力测试
# ==========================================
def run_milvus_stress():
    print(f"\n{'='*20} STARTING MILVUS STRESS TEST {'='*20}")
    connections.connect("default", host=HOST, port="19530")
    col_name = "stress_ghost_milvus"
    
    if utility.has_collection(col_name): utility.drop_collection(col_name)
    
    fields = [FieldSchema("id", DataType.INT64, is_primary=True), FieldSchema("vector", DataType.FLOAT_VECTOR, dim=DIM)]
    schema = CollectionSchema(fields)
    col = Collection(col_name, schema, consistency_level="Strong")
    
    # 铺底
    print(f"   📥 Seeding base objects...")
    ids = list(range(2000))
    vecs = [[0.1]*DIM] * 2000
    col.insert([ids, vecs])
    col.flush()
    
    print("   🏗️  Building Index (Required)...")
    col.create_index("vector", {"metric_type": "L2", "index_type": "FLAT", "params": {}})
    col.load()
    
    # 干扰线程
    def noise_injector():
        try:
            # 疯狂插入新 Partition 或新 Segment
            c = Collection(col_name)
            for i in range(10):
                c.insert([list(range(10000+i*100, 10100+i*100)), [[0.1]*DIM]*100])
        except: pass

    print("   🌪️  Starting Noise Injector...")
    t = threading.Thread(target=noise_injector)
    t.start()
    
    # 删除
    print(f"   🗑️  Deleting target IDs (0-1000)...")
    try:
        col.delete("id < 1000") # 这是一个耗时操作
    except: pass
    
    # 截杀
    time.sleep(random.uniform(1, 3)) # Milvus 动作快，杀快点
    restart_container("milvus")
    
    # 验尸
    print("   🔍 Verifying Consistency...")
    for _ in range(10):
        try:
            connections.connect("default", host=HOST, port="19530")
            col = Collection(col_name)
            col.load()
            break
        except: time.sleep(2)
        
    # 检查 ID 0
    res = col.query("id == 0", output_fields=["id"])
    if len(res) > 0:
        print(f"   {RED}❌ CRITICAL FAIL: Zombie Data Found! ID 0 resurrected.{RESET}")
    else:
        print(f"   {GREEN}✅ PASS: Milvus WAL replay works perfectly.{RESET}")

if __name__ == "__main__":
    # 依次执行
    run_weaviate_stress()
    run_milvus_stress()