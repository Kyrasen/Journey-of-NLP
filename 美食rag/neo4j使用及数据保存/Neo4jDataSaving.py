import json
from neo4j import GraphDatabase
from tqdm import tqdm
import time

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


def read_data1(file):
    data = []
    count = 0
    with open(file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if line.strip():
                item = json.loads(line)
                item["id"] = idx  # 原始文件中的行号
                data.append(item)
                count += 1
            if count == 10000:
                break
    return data  # 每个元素是一个字典，包含 7 个属性

def read_data2(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data

def process_materials(materials_data_list):
    material_vocab = {}
    for item in materials_data_list:
        for material in item["materials"]:
            material_vocab.setdefault(material, len(material_vocab))

    list_material = list(material_vocab.keys())

    return list_material

if __name__ == "__main__":
    dish_data_file = "day29Test/叶森莹-题目1/data/recipe_corpus_full.json"
    dish_data_list = read_data1(dish_data_file)

    materials_data_file = "day30RAG/data/neo4j_matarials.json"
    materials_data_list = read_data2(materials_data_file)

    neo4j_manager = Neo4jManager()
    # 创建 “菜” 节点
    # properties = {}
    # start_time = time.time()
    # for item in tqdm(data_list):
    #     label = "菜"
    #     properties["dish"] = item["dish"]
    #     properties["description"] = item["description"]
    #     properties["recipeInstructions"] = item["recipeInstructions"]
    #     properties["id"] = item["id"]
    #     properties["name"] = item["name"]
    #     neo4j_manager.create_node(label, properties)
    #     # nodes = neo4j_manager.get_nodes(label)
    # end_time = time.time()
    # nodes = neo4j_manager.get_nodes("菜")
    # print(f"一共插入了{len(nodes)}个节点， 耗时：{end_time - start_time}")

    # 创建 “食材” 节点
    # unipue_materials = process_materials(materials_data_list)
    # properties = {}
    # start_time = time.time()
    # for item in tqdm(unipue_materials):
    #     label = "食材"
    #     properties["name"] = item

    #     neo4j_manager.create_node(label, properties)
    #     pass
    #     # nodes = neo4j_manager.get_nodes(label)
    # end_time = time.time()
    # nodes = neo4j_manager.get_nodes("食材")
    # print(f"一共插入了{len(nodes)}个节点， 耗时：{end_time - start_time}")


    '''
    查看菜为：小朋友超喜欢的巧克力奶香馒头（自发粉版）的所有关系：MATCH (a :菜 {name: "小朋友超喜欢的巧克力奶香馒头（自发粉版）"})-[r:材料]->(i:食材) RETURN a, r, i
    删除所有关系 “材料”：MATCH ()-[r:材料]->() DELETE r
    
    '''

    # 建立 “菜” 和 “食材” 之间的关系 ： “材料”
    rel_type = "材料"
    start_time = time.time()
    for item in tqdm(materials_data_list):
        label1 = "菜"
        props1 = {"id" : item["id"]}
        label2 = "食材"
        for props in item["materials"]:
            props2 = {"name": props}
            neo4j_manager.create_relationship(label1, props1, rel_type, label2, props2)
            
    end_time = time.time()
    print(f"建立关系完成， 耗时：{end_time-start_time}")

    pass