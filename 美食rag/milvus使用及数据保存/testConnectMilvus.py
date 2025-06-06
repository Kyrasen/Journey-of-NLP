from pymilvus import connections

# 替换为你服务器的公网 IP
MILVUS_HOST = "8.138.27.163"
MILVUS_PORT = "19530"

try:
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    print("✅ 成功连接 Milvus！")
except Exception as e:
    print("❌ 连接失败：", e)
