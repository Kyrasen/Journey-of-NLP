import json
from sentence_transformers import SentenceTransformer
from pymilvus import DataType, CollectionSchema, FieldSchema, connections, utility, Collection
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from neo4j import GraphDatabase
import torch

class Recipe:
    def __init__(self, data:pd.DataFrame):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        return self.data.iloc[idx]

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
            limit=top_k
        )
        print(f" 搜索结果（Top {top_k}）：")
        for hits in results:
            for hit in hits:
                print(f"score:{hit.distance: .4f}, entity: {hit.entity}")
        return results

    def count(self, collection_name):
        if not utility.has_collection(collection_name):
            raise ValueError(f"集合 {collection_name} 不存在")
        collection = Collection(name=collection_name)
        return collection.num_entities
    

class Neo4jManager:
    def __init__(self, url = "bolt://localhost:7687", user = "neo4j", password = "neo4j"):
        self.driver = GraphDatabase.driver(url, auth = (user, password))

    def close(self):
        self.driver.close()

    # ------------ 节点相关 ---------------------
    def create_node(self, label, properties:dict):
        with self.driver.session() as session:
            session.execute_write(self._create_node, label, properties)

    @staticmethod
    def _create_node(tx, label, properties):
        query = f"MERGE (n:{label} {{"
        query += ", ".join([f"{k} : ${k}" for k in properties])
        query += "})"

        tx.run(query, **properties)

    def get_nodes(self, label, match_field=None, value=None):
        with self.driver.session() as session:
            return session.execute_write(self._get_nodes, label, match_field, value)

    @staticmethod
    def _get_nodes(tx, label, match_field, value):
        if match_field:
            result = tx.run(f"MATCH (n:{label}) WHERE n.{match_field} = $value RETURN n", value=value)
        else:
            result = tx.run(f"MATCH (n:{label}) RETURN n")
        return [record["n"] for record in result]
    
    def update_node(self, label, match_field, match_value, update_fields: dict):
        with self.driver.session() as session:
            session.execute_read(self._update_node, label, match_field, match_value, update_fields)

    @staticmethod
    def _update_node(tx, label, match_field, match_value, update_fields):
        set_clause = ", ".join([f"n.{k} = ${k}" for k in update_fields])
        query = f"""
        MATCH (n:{label}) WHERE n.{match_field} = $match_value
        SET {set_clause}
        """
        tx.run(query, match_value=match_value, **update_fields)

    def delete_node(self, label, match_field, match_value):
        with self.driver.session() as session:
            session.execute_write(self._delete_node, label, match_field, match_value)

    @staticmethod
    def _delete_node(tx, label, match_field, match_value):
        query = f"""
        MATCH (n:{label}) WHERE n.{match_field} = $match_value
        DETACH DELETE n
        """
        tx.run(query, match_value=match_value)


    # ---------- 关系相关 ----------

    def create_relationship(self, label1, props1, rel_type, label2, props2):
        with self.driver.session() as session:
            session.execute_write(
                self._create_relationship, label1, props1, rel_type, label2, props2
            )

    @staticmethod
    def _create_relationship(tx, label1, props1, rel_type, label2, props2):
        query = f"""
        MERGE (a:{label1} {{ {', '.join([f'{k}: $a_{k}' for k in props1])} }})
        MERGE (b:{label2} {{ {', '.join([f'{k}: $b_{k}' for k in props2])} }})
        MERGE (a)-[r:{rel_type}]->(b)
        """
        params = {f"a_{k}": v for k, v in props1.items()}
        params.update({f"b_{k}": v for k, v in props2.items()})
        tx.run(query, **params)

    def get_relationships(self, label1, label2, rel_type):
        with self.driver.session() as session:
            return session.execute_read(self._get_relationships, label1, label2, rel_type)

    @staticmethod
    def _get_relationships(tx, label1, label2, rel_type):
        query = f"""
        MATCH (a:{label1})-[r:{rel_type}]->(b:{label2})
        RETURN a, r, b
        """
        result = tx.run(query)
        return [(record["a"], record["r"], record["b"]) for record in result]
    
    def delete_relationship(self, label1, match1, rel_type, label2, match2):
        with self.driver.session() as session:
            session.execute_write(self._delete_relationship, label1, match1, rel_type, label2, match2)

    @staticmethod
    def _delete_relationship(tx, label1, match1, rel_type, label2, match2):
        query = f"""
        MATCH (a:{label1} {{ {', '.join([f'{k}: $a_{k}' for k in match1])} }})-[r:{rel_type}]->(b:{label2} {{ {', '.join([f'{k}: $b_{k}' for k in match2])} }})
        DELETE r
        """
        params = {f"a_{k}": v for k, v in match1.items()}
        params.update({f"b_{k}": v for k, v in match2.items()})
        tx.run(query, **params)

    def get_dishes_by_ingredients(self, ingredient_names: list):
        """
        查询包含所有指定食材的菜。
        :param ingredient_names: 食材名称列表，例如 ["西红柿", "鸡蛋"]
        :return: 包含这些食材的菜节点列表
        """

        with self.driver.session() as session:
            return session.execute_read(self._get_dishes_by_ingredients, ingredient_names)

    @staticmethod
    def _get_dishes_by_ingredients(tx, ingredient_names):
        # 构造查询子句
        match_clauses = []
        where_clauses = []
        for idx, name in enumerate(ingredient_names):
            alias = f"i{idx}"
            match_clauses.append(f"(d)-[:材料]->({alias}:食材)")
            where_clauses.append(f"{alias}.name = $name{idx}")
        
        match_clause = "MATCH (d:菜)" + "".join(["-[:材料]->(i:食材)" for _ in ingredient_names])
        match_clause = "MATCH " + ", ".join([f"(d)-[:材料]->(i{idx}:食材)" for idx in range(len(ingredient_names))])
        where_clause = " AND ".join(where_clauses)

        query = f"""
        {match_clause}
        WHERE {where_clause}
        RETURN DISTINCT d
        LIMIT 10
        """

        params = {f"name{idx}": name for idx, name in enumerate(ingredient_names)}
        result = tx.run(query, **params)
        return [record["d"] for record in result]

class MyModel(torch.nn.Module):
    def __init__(self, bge_model, labels_num, hidden_size=768):
        super().__init__()
        self.emb = bge_model
        self.cls = torch.nn.Linear(hidden_size, labels_num)

    def forward(self, input_ids):
        bge_output = self.emb(input_ids, attention_mask=(input_ids!=0)).last_hidden_state
        attention_mask = (input_ids != 0).unsqueeze(-1).float()
        masked_output = bge_output * attention_mask
        sum_output = masked_output.sum(dim=1)
        lengths = attention_mask.sum(dim=1)
        sentence_embeddings = sum_output / lengths

        logits = self.cls(sentence_embeddings)

        return logits