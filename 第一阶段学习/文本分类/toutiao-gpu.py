import re

def read_data(file):
    with open(file, "r", encoding="utf-8") as f:
        data = f.read().split("\n")

        pattern = re.compile(r"_!_(\d+)_!_.*?_!_(.*?)_!_")

        for line in data:
            matches = pattern.findall(line)
        pass



if __name__ == "__main__":
    toutiao_file = "data/toutiao_cat_data.txt"
    stopwords_file = "data/mystopWords.txt"

    corpus, labels = read_data(toutiao_file)