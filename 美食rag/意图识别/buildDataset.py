import json
import random
from tqdm import tqdm

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

def process_materials(materials_data_file):
    with open(materials_data_file, "r", encoding = "utf-8") as f:
        materials_data_list = json.load(f)
        material_vocab = {}
        for item in materials_data_list:
            for material in item["materials"]:
                material_vocab.setdefault(material, len(material_vocab))

        list_material = list(material_vocab.keys())

    return list_material

def generate_random_materials(materials_data_list):
    materials = process_materials(materials_data_list)
    for i in range(5):
        raw_materials = random.choice(materials)

    return raw_materials

def generate_random_name():
    first_name = ["赵","钱","孙","李","周","吴","郑","王","冯","陈","褚","卫","蒋","沈","韩","杨","朱","秦","尤","许","何","吕","施","张","孔","曹","严","华","金","魏","陶","姜","戚","谢","邹","喻","柏","水","窦","章","云","苏","潘","葛","奚","范","彭","郎","鲁","韦","昌","马","苗","凤","花","方","俞","任","袁","柳","江","童","颜","郭","梅","盛","林","刁","钟","徐","邱","骆","高","夏","蔡","田","樊","胡","凌","霍","虞","万","支","柯","昝","管","卢","莫","经","房","裘","缪","干","解","应","宗","丁","宣","贲","邓","郁","单","杭","洪","包","诸","左","石","崔","吉","钮","龚","程","嵇","邢","滑","裴","陆","荣","翁","荀","羊","於","惠","甄","麹","家","封","芮","羿","储","靳","汲","邴","糜","松","井","段","富","巫"]

    second_name = ["国运","鸣翰","成钧","米林","尔一","际广","巧肖","德太","士旭","金亮","香思","临衷","森羽","百顷","思开","禹文","和城","哲振", "艺松","迎叶","霏乐","钰升","洋传","雨彩", "柏正","卓巧","竹咏","晶霖","祥本","毅讯"]

    name = f"{random.choice(first_name)}{random.choice(second_name)}"

    return name


def make_data(dish_data_list, outout_file, materials_data_list):
    with open(outout_file, "a+", encoding="utf-8") as f:
        for item in tqdm(dish_data_list):
            # 制作方法 0
            random_text = random.choice(item["keywords"])
            f.write(random_text +" " + "0" +"\n")

            # 原材料 1
            appending_text1 = f"{item['name']}的原材料有哪些？"
            f.write(appending_text1 + " " + "1" + "\n")

            # 可做的菜 2
            raw_materials = generate_random_materials(materials_data_list)
            appending_text2 = f"有这些东西：{raw_materials}，可以做什么菜？"
            f.write(appending_text2 + " " + "2" + "\n")

            # x 擅长做的菜 3
            random_name = generate_random_name()
            appending_text3 = f"{random_name}可以做什么菜？"
            f.write(appending_text3 + " " + "3" + "\n")

            # 其他 4
            appending_text4 = f"{random_name}是谁？"
            f.write(appending_text4 + " " + "4" + "\n")
    
    print("5个类型写入文件完成")

if __name__ == "__main__":
    '''
        0: 制作方法
        1：原材料
        2：可做的菜
        3：XXX擅长做的菜
        4：其他
        一共 5 个类别，每个类别 1 万条数据
    '''
    dish_data_file = "day29Test/叶森莹-题目1/data/recipe_corpus_full.json"
    materials_data_list = "day30RAG/data/neo4j_matarials.json"
    outout_file = "day30RAG\src2(VLLM)/NER/data/all_dataset.txt"
    dish_data_list = read_data1(dish_data_file)
    

    make_data(dish_data_list, outout_file, materials_data_list)
    

    

    pass