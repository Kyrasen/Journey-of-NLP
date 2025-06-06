from pymilvus import Collection, connections, FieldSchema, CollectionSchema, DataType
import time
import numpy as np

# 连接 Milvus 服务器（修改为你的地址）
connections.connect("default", host="8.138.27.163", port="19530")

# 定义集合名
collection_name = "test_insert_timing"

# 如果集合已存在则删除
if Collection.exists(collection_name):
    Collection(name=collection_name).drop()

# 定义字段（假设主键+向量字段）
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128)
]

# 创建 schema 和集合
schema = CollectionSchema(fields=fields, description="Insert timing test")
collection = Collection(name=collection_name, schema=schema)

# 向量数量与维度
num_vectors = 10000
dim = 128

# 生成数据
ids = list(range(num_vectors))
embeddings = np.random.random((num_vectors, dim)).astype(np.float32).tolist()

# 插入数据并计时
start_time = time.time()
collection.insert([ids, embeddings])
end_time = time.time()

# 输出耗时信息
print(f"插入 {num_vectors} 条向量耗时：{end_time - start_time:.4f} 秒")



