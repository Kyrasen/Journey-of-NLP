import json
import os
import ast
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def read_data(file):
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
    return data  # 每个元素是一个字典，包含 id, name, recipeIngredient

def get_aliyun_reponse(prompt):
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
        # ast.literal_eval这个函数能够将字符串的内容转换为Python对象
        return ast.literal_eval(completion.choices[0].message.content)
    except Exception as e:
        print(f"[错误] 请求失败：{e}")
        return None  # 改成 None 表示失败

def process_single_item(item):
    name = item["name"]
    materials = item["recipeIngredient"]
    item_id = item["id"]

    prompt = f"请帮我将以下文本中的食材分离出来，去除量词，用列表保存，只返回列表，不要说其他废话。文本为：{materials}"
    new_materials = get_aliyun_reponse(prompt)
    
    if new_materials is None:
        return None, item  # 返回失败项
    return {
        "id": item_id,
        "name": name,
        "materials": new_materials
    }, None

def process_materials_multithread(data_list, max_workers=12):
    success_data = []
    failed_data = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_item, item) for item in data_list]

        for future in tqdm(as_completed(futures), total=len(futures), desc="处理中"):
            try:
                success, failed = future.result()
                if success:
                    success_data.append(success)
                elif failed:
                    failed_data.append(failed)
            except Exception as e:
                print(f"[错误]处理失败：{e}")

    # 保存成功数据
    os.makedirs("day30RAG/data", exist_ok=True)
    with open("day30RAG/data/neo4j_matarials.json", "w", encoding="utf-8") as f:
        json.dump(success_data, f, ensure_ascii=False, indent=4)

    # 保存失败数据
    with open("day30RAG/data/failed_items.json", "w", encoding="utf-8") as f:
        json.dump(failed_data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    data_file = "day29Test/叶森莹-题目1/data/recipe_corpus_full.json"
    data_list = read_data(data_file)

    process_materials_multithread(data_list, max_workers=12)
    print("✅ 数据处理完成")

    with open("day30RAG/data/neo4j_matarials.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"成功导入：{len(data)} 条")

    with open("day30RAG/data/failed_items.json", "r", encoding="utf-8") as f:
        failed = json.load(f)
    print(f"失败条数：{len(failed)} 条")
