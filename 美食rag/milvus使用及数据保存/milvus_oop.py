import json
from sentence_transformers import SentenceTransformer
from pymilvus import DataType, CollectionSchema, FieldSchema, connections, utility, Collection
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import time
import random

class Recipe:
    def __init__(self, data:pd.DataFrame):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        return self.data.iloc[idx]

class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.indices = np.arange(len(dataset))

    def __iter__(self):
        self.cursor = 0
        return self
    
    def __next__(self):
        if self.cursor >= len(self.dataset):
            raise StopIteration
        
        batch_data = self.dataset[self.cursor: self.cursor+self.batch_size]
        self.cursor += self.batch_size
        return batch_data

    def __len__(self):
        return int(np.ceil( len(self.dataset) / self.batch_size))

class MilvusDataset:
    def __init__(self, config_file):
        with open(config_file, "r", encoding="utf-8") as f:
            config_data = json.load(f)
        self.tags = config_data["tags"]

        self.emb = SentenceTransformer(config_data["model_path"])
        self.client_url = config_data["client_url"]
        host, port = self.client_url.split(":")
        connections.connect(host=host, port=port)
        print("✅ Milvus 已连接")
        self.collections = {}

    def create_collection(self, collection_name):
        fields = []
        for tagKey, tagValue in self.tags.items():
            if tagValue == "id":
                fields.append(FieldSchema(name=tagValue, dtype=DataType.INT64, is_primary=True))
            elif tagValue == "vector":
                fields.append(FieldSchema(name=tagValue, dtype=DataType.FLOAT_VECTOR, dim=768))
            else:
                # fields.append(FieldSchema(name=tagValue, dtype=DataType.VARCHAR, max_length=16384))
                continue
        schema = CollectionSchema(fields, description="Food collection schema")

        # 如果存在同名collection 就删除
        if utility.has_collection(collection_name):
            Collection(name=collection_name).drop()
        # 创建集合并加载
        collection = Collection(name=collection_name, schema=schema)
        print(f"✅ 创建集合：{collection}")
        collection.create_index(field_name="vector", index_params={
            "metric_type":"COSINE",
            "index_type":"IVF_FLAT",
            "params":{"nlist":128}
        })
        collection.load()
        self.collections["collection_name"] = collection
        print(f"✅ 创建集合成功：{collection_name}")


    def insert(self, collection_name, entity:dict):
        '''
        entity: 包含id, text(需要生成向量)、其他字段
        '''
        if not utility.has_collection(collection_name = collection_name):
            raise ValueError(f"集合 {collection_name} 不存在，请先创建")
        collection = Collection(name=collection_name)
        fields_data = []

        for tagKey, tagValue in self.tags.items():
            if tagValue == "id":
                fields_data.append([collection.num_entities])
            elif tagValue == "vector":
                vector = self.emb.encode(entity[self.tags["tagName"]]).tolist()
                fields_data.append([vector])
            else:
                fields_data.append([";".join(entity.get(tagValue, ""))])

        collection.insert(fields_data)
        print(f" ✅ 已插入数据到集合{collection_name}：{entity}")

        

    def clear(self, collection_name):
        if utility.has_collection(collection_name = collection_name):
            Collection(name = collection_name).drop()
            print(f"✅ {collection_name} 已成功删除")
        else:
            print(f"✅ {collection_name} 无法删除")
        
    def insert_batch(self, collection_name, df, batch_size=2000):
        dataset = Recipe(df)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        collection = Collection(name=collection_name)
        collection_nums = collection.num_entities

        for batch in tqdm(loader):
            ids = range(collection_nums+1, collection_nums+1+len(batch))
            vectors = self.emb.encode(batch["name"].tolist())
            entities = [
                {
                    self.tags["tagName"]: batch.iloc[i][self.tags['tagName']],
                    self.tags["tagDish"]: batch.iloc[i][self.tags['tagDish']],
                    self.tags["tagDesc"]: batch.iloc[i][self.tags['tagDesc']],
                    self.tags["tagReig"]: batch.iloc[i][self.tags['tagReig']],
                    self.tags["tagReis"]: batch.iloc[i][self.tags['tagReis']],
                    self.tags["tagAuth"]: batch.iloc[i][self.tags['tagAuth']],
                    self.tags["tagKeyw"]: batch.iloc[i][self.tags['tagKeyw']],
                    self.tags["tagId"]: ids[i],
                    self.tags["tagVec"]: vectors[i]

                } 
                for i in range(len(batch))
            ]

            collection.insert(entities)
            collection.flush()
            print(f"✅ 批量插入 {batch_size} 条数据，现在数据一共 {collection.num_entities}条数据")

    def insert_batch2(self, collection_name, df, batch_size=2000):
        MAX_LEN = 8000  # Milvus 对字符串字段最大长度限制
        dataset = Recipe(df)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        collection = Collection(name=collection_name)
        collection_nums = collection.num_entities

        for batch in tqdm(loader):
            ids = range(collection_nums + 1, collection_nums + 1 + len(batch))
            vectors = self.emb.encode(batch["name"].tolist())

            entities = []
            for i in range(len(batch)):
                entity = {}
                for tag in [
                    "tagName", "tagDish", "tagDesc",
                    "tagReig", "tagReis", "tagAuth", "tagKeyw"
                ]:
                    key = self.tags[tag]

                    # ✅ 改成 loc 索引，确保拿到原始值
                    try:
                        val = df.loc[batch.index[i], key]
                    except KeyError:
                        val = ""

                    # ✅ 安全转换 + 截断
                    if pd.isna(val):
                        val = ""
                    elif isinstance(val, list):
                        val = "；".join(map(str, val))
                    else:
                        val = str(val)

                    val = val[:MAX_LEN]
                    entity[key] = val

                entity[self.tags["tagId"]] = ids[i]
                entity[self.tags["tagVec"]] = vectors[i]
                for k, v in entity.items():
                    if isinstance(v, str) and len(v) > MAX_LEN:
                        print(f"⚠️ 字段过长: {k} -> 长度: {len(v)}，前100字符：{v[:100]}")
                entities.append(entity)

            collection.insert(entities)
            collection.flush()


            print(f"✅ 批量插入 {len(batch)} 条数据，现在数据一共 {collection.num_entities} 条")
    def insert_batch3(self, collection_name, df, batch_size=2000):
        MAX_LEN = 16384  # Milvus 对字符串字段最大长度限制
        dataset = Recipe(df)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        collection = Collection(name=collection_name)
        collection_nums = collection.num_entities

        for batch in tqdm(loader):
            ids = range(collection_nums + 1, collection_nums + 1 + len(batch))
            vectors = self.emb.encode(batch["name"].tolist())

            entities = []
            for i in range(len(batch)):
                entity = {}
            #     for tag in [
            #         "tagName", "tagDish", "tagDesc",
            #         "tagReig", "tagReis", "tagAuth", "tagKeyw"
            #     ]:
            #         key = self.tags[tag]
            #         try:
            #             val = df.loc[batch.index[i], key]
            #         except KeyError:
            #             val = ""

            #         # 处理 NaN 和其他类型
            #         if pd.isna(val):
            #             val = ""
            #         else:
            #             # 强制转换为字符串，处理可能的列表或其他结构
            #             if isinstance(val, list):
            #                 val = "；".join(map(str, val))
            #             else:
            #                 val = str(val)
            #         # 确保截断到最大长度
            #         val = val[:MAX_LEN]
            #         entity[key] = val

                # 处理向量和ID
                entity[self.tags["tagId"]] = ids[i]
                entity[self.tags["tagVec"]] = vectors[i].tolist()  # 确保向量是列表格式

                # 检查字段长度
                for k, v in entity.items():
                    if isinstance(v, str) and len(v) > MAX_LEN:
                        print(f"⚠️ 字段过长: {k} -> 长度: {len(v)}，前100字符：{v[:100]}")
                entities.append(entity)

            # 插入数据
            try:
                collection.insert(entities)
                collection.flush()
                collection_nums = collection.num_entities  # 更新当前数量
                print(f"✅ 批量插入 {len(batch)} 条数据，现在数据一共 {collection_nums} 条")
            except Exception as e:
                print(f"插入失败，错误信息：{e}")
                # 输出第一个出错的实体信息
                if entities:
                    first_entity = entities[0]
                    for k, v in first_entity.items():
                        if isinstance(v, str):
                            print(f"字段 {k} 长度：{len(v)}")
                raise


    def search(self, collection_name, query_text, top_k=1):
        if not utility.has_collection(collection_name):
            raise ValueError(f"集合 {collection_name} 不存在")
        collection = Collection(name=collection_name)
        collection.load()
        query_vector = self.emb.encode(query_text).tolist()

        results = collection.search(
            data = [query_vector],
            anns_field = "vector",
            param={"metric_type": "COSINE", "params":{"nprobe":10}},
            limit=top_k,
            output_fields=[tag for tag in self.tags.values() if tag not in ("id", "vector")]
        )
        print(f" 搜索结果（Top {top_k}）：")
        for hits in results:
            for hit in hits:
                print(f"score:{hit.distance: .4f}, entity: {hit.entity}")

    def count(self, collection_name):
        if not utility.has_collection(collection_name):
            raise ValueError(f"集合 {collection_name} 不存在")
        collection = Collection(name=collection_name)
        return collection.num_entities

def read_data1(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    fieldNames = list(data[0].keys())
    df = pd.DataFrame(data[10000])

    df["recipeIngredient"] = df["description"].apply(lambda x: "；".join(x))
    df["recipeInstructions"] = df["recipeInstructions"].apply(lambda x: "；".join(x))
    df["keywords"] = df["keywords"].apply(lambda x: "；".join(x))
    
    return df, fieldNames

def read_data2(file_path):
    data = []
    count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
                count += 1
            if count == 10000:
                break

            # count += 1
            # if count > 44000 and count < 48000:
            #     data.append(json.loads(line))


    df = pd.DataFrame(data)
    fieldNames = list(df.columns)

    MAX_LEN = 5000

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].apply(lambda x: "；".join(map(str, x)) if isinstance(x, list) else str(x))
            df[col] = df[col].apply(lambda x: x[:MAX_LEN])

    return df, fieldNames

if __name__ == "__main__":
    config_file = "day30RAG/src/config_data.json"
    data_file = "day29Test/叶森莹-题目1/data/recipe_corpus_full.json"

    df, fieldNames = read_data2(data_file)

    milvusDataset = MilvusDataset(config_file)

    collection_name = "Food2"
    milvusDataset.create_collection(collection_name)

    # 插入数据并计时
    start_time = time.time()
    milvusDataset.insert_batch3(collection_name = collection_name,df = df)
    end_time = time.time()
    print(f"{collection_name} 一共有 {milvusDataset.count(collection_name)}条数据, 耗时：{end_time - start_time:.4f} 秒")

    # 查询数据并计时
    # nums = milvusDataset.count(collection_name)
    # print(f"nums : {nums}")

    # search_keywords = []
    # for i in range(10000):
    #     search_keywords.append(df.iloc[random.randint(0, 9999)]["name"])

    # start_time = time.time()
    # for i in search_keywords:
    #     milvusDataset.search(collection_name, i)
    # end_time = time.time()
    # print(f"查询{len(search_keywords)}条数据，耗时：{end_time - start_time:.4f} 秒")
    # while True:
    #     input_query = input("请输入：")
    #     if input_query == "exit":
    #         break
    #     start_time = time.time()
    #     milvusDataset.search(collection_name, input_query)
    #     end_time = time.time()
    #     print(f"查询1条数据, 耗时：{end_time - start_time:.4f} 秒")




    pass

