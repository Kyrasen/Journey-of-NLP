from flask import Flask, request
from openai import OpenAI
import ast
import torch
from transformers import AutoModel, AutoTokenizer
from dataUtils import Neo4jManager, MilvusDataset, MyModel
import ast

app = Flask(__name__)

def get_aliyun_reply(prompt):
    client = OpenAI(
        api_key="sk-92f3be7c0f3d4dabbd917bb05ea35d5f",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    try:
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        print(f"[错误] 请求失败：{e}")
        return None  # 改成 None 表示失败

def get_intent(query):
    model_intent.eval()

    seq_len = 20
    inputs = tokenizer_intent(
            query,
            truncation = True,
            padding = "max_length",
            max_length = seq_len,
            return_tensors = "pt"
        )

    input_ids = inputs["input_ids"]
    model_intent.eval()
    with torch.no_grad():
        logits = model_intent(input_ids)
        pre_class = torch.argmax(logits, dim=-1).item()

    return pre_class

def get_neo4j_answer(entity, field):

    result = neo4j_manager.get_nodes("菜", field, entity)

    if result:
        return result
    else:
        return 0
    
def get_milvus_answer(entity):
    res = milvus_manager.search("Food2", entity)

    if res[0].distances[0] > 0.7:
        return res[0].ids
    else:
        return None

def get_entity(query):
    # 编码输入文本
    input_ids = tokenizer_ner(
        query,
        truncation=True,
        padding="max_length",
        max_length=20,
        return_tensors="pt"
    )["input_ids"]  # 注意这里需要获取 input_ids 字段

    # 模型预测
    logits = model_ner(input_ids)
    pre_labels = torch.argmax(logits, dim=-1)[0].tolist()  # 取第一个样本的预测标签

    # 映射标签索引到BIO标签
    labels = [index_to_tag[i] for i in pre_labels]

    current_entity = ""
    current_type = ""
    entities = []

    for idx, tag in enumerate(labels[:len(query)]):  # 限制不超过原始文本长度
        if tag.startswith("B-"):
            if current_entity:  # 若已有实体，保存之前的
                entities.append({"entity": current_entity, "type": current_type})
            current_entity = query[idx]
            current_type = tag[2:]
        elif tag.startswith("I-"):
            if current_entity and tag[2:] == current_type:
                current_entity += query[idx]
            else:
                # 异常的 I- 开头，直接丢弃当前实体
                current_entity = ""
                current_type = ""
        elif tag == "O":
            if current_entity:
                entities.append({"entity": current_entity, "type": current_type})
                current_entity = ""
                current_type = ""

    # 收尾处理
    if current_entity:
        entities.append({"entity": current_entity, "type": current_type})

    return entities

@app.route("/", methods=["POST"])
def home():
    data = request.json # 接收到前端发来的信息，是json
    query = data["query"] # 获取问题

    # 获取意图
    intent = get_intent(query)
    # 获取实体
    entity = get_entity(query)

    if intent == 4:
        return {"res": "无法回答"}

    database_res = ""
    # 在数据库中查找相关的菜系
    # 制作方法，返回 recipeInstructions
    if intent == 0:
        results_neo4j = get_neo4j_answer(entity, "name")
        if results_neo4j:
            database_res = results_neo4j[0]._properties["recipeInstructions"] # 拿到数据库中的结果
        else:
            results_milvus = get_milvus_answer(entity[0])
            if results_milvus is not None:
                results_neo4j_1 = get_neo4j_answer(results_milvus[0]-1, "id")
                database_res = results_neo4j_1[0]._properties["recipeInstructions"]
            else:
                database_res = None


    # 原材料，返回recipeIngrediance
    if intent == 1:
        results_neo4j = get_neo4j_answer(entity, "name")
        if results_neo4j:
            database_res = results_neo4j[0]._properties["recipeIngredient"] # 拿到数据库中的结果
        else:
            results_milvus = get_milvus_answer(entity[0])
            if results_milvus is not None:
                results_neo4j_1 = get_neo4j_answer(results_milvus[0]-1, "id")
                database_res = results_neo4j_1[0]._properties["recipeIngredient"]
            else:
                database_res = None

    # 有原材料,可做的菜，返回 dish
    if intent == 2:
        processed_query = f"{query}，请你帮我分离出上句话里面的食材，并用python的List的形式回答，不要说其他话。"
        ingredients = entity

        dishes = neo4j_manager.get_dishes_by_ingredients(ingredients)

        database_res = [item._properties["name"] for item in dishes]
        return {"res": "\n".join(database_res)}
        

    # 某某某擅长做的菜， 返回 dish
    if intent == 3:
        results_neo4j = get_neo4j_answer(entity)
        if results_neo4j:
            database_res = results_neo4j # 拿到数据库中的结果
        else:
            results_milvus = get_milvus_answer(entity)
            if results_milvus is not None:
                results_neo4j_1 = get_neo4j_answer(results_milvus)
                database_res = results_neo4j_1
            else:
                database_res = None

    if database_res is not None:
        prompt = f"请你根据这个信息{database_res}，回答这个问题{query}，不要说关于我给你的信息，你就直接回答问题就行了，但是要参考我给你的信息。"
    else:
        prompt = f"请你回答这个问题：{query}"
    results = get_aliyun_reply(prompt)

    return {"res": results, "intent":intent, "entity": entity}

if __name__ == "__main__":
    
    config_file = "day30RAG/src/config_data.json"
    tokenizer_intent = AutoTokenizer.from_pretrained("models/bge-base-zh-v1.5")
    model_intent = torch.load("day30RAG/src2_VLLM/NER/model/model.pth", weights_only=False)
    tokenizer_ner = AutoTokenizer.from_pretrained("models/bert-base-chinese")
    model_ner = AutoModel.from_pretrained("day30RAG/美食rag (1)/7_项目总结/实体识别模型-checkpoint-447")
    tag_to_index = {"PAD":0, "B-food":1, "I-food":2, "B-cookbook":3, "I-cookbook":4, "O":5}
    index_to_tag = list(tag_to_index)

    neo4j_manager = Neo4jManager()
    milvus_manager = MilvusDataset(config_file)
    
    app.run(
        host = "127.0.0.1",
        port = 44444
    )

    pass

