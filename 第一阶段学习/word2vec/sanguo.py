import re
import jieba
import numpy as np
import json

def read_data(file):
    with open(file, "r", encoding="utf-8") as f:
        data = f.read().split("\n")

    data = [line for line in data if line]

    return data

def split_chapter(raw_data):
    pattern = re.compile(r"第[零一二三四五六七八九十百]{1,5}回")
    contents = []
    chapters = []
    start = 10
    for current_idx,line in enumerate(raw_data):
        if pattern.match(line):
            chapter = line
            chapters.append(chapter)
            if current_idx > start:
                contents.append(raw_data[start+1: current_idx])
                start = current_idx
    
    contents.append(raw_data[start+1:])

    chapter_dict = dict(zip(chapters, contents))
            
    return chapter_dict

def tokenize(chapter_dict:dict):
    vocab = {"[PAD]":0, "[UNK]":1}
    tokens_perChapter = {}
    for chapter, content in chapter_dict.items():
        tokens_sentence = []
        for sentence in content:
            tokens = jieba.lcut(sentence)   # 每句话分词
            tokens = [token for token in tokens if token not in stopwords]  # 每句话分词后去掉停词
            tokens_sentence.append(tokens)  # 收集每个句子的分词
            for token in tokens:    # 为每个分词编号
                vocab.setdefault(token, len(vocab))

        tokens_perChapter[chapter] = tokens_sentence    # 收集每章的分词

        with open("data/vocabsanguo.json", "w") as f:
            json.dump(vocab, f, ensure_ascii = False, indent=4)

        with open("data/tokens_perChapter.json", "w") as f:
            json.dump(tokens_perChapter, f, ensure_ascii=False,indent=4)

        return vocab

def loadStopwords(file):

    with open(file, "r", encoding="utf-8") as f:
        stopwords = f.read().split("\n")

    return stopwords

def load_vocab(file):
    with open(file, "r") as f:
        vocab = json.load(f)
    return vocab


if __name__ == "__main__":
    sanguo_file = "data/《三国演义》.txt"
    stopwords_file = "data/mystopWords.txt"

    raw_data = read_data(sanguo_file)
    chapter_dict = split_chapter(raw_data)

    stopwords = loadStopwords(stopwords_file)

    vocab = load_vocab("data/vocabsanguo.json")
    pass