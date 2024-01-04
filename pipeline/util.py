from typing import Iterable, Dict
import gzip
import json
import os
import csv
import sys
sys.path.append("..")
from const import *

def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False):
    """
    Writes an iterable of dictionaries to jsonl
    """
    if append:
        mode = 'ab'
    else:
        mode = 'wb'
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode='wb') as gzfp:
                for x in data:
                    gzfp.write((json.dumps(x) + "\n").encode('utf-8'))
    else:
        with open(filename, mode) as fp:
            for x in data:
                fp.write((json.dumps(x) + "\n").encode('utf-8'))


def read_example_dict(filename):
    example_dicts = []
    with open(filename, encoding='utf-8') as f:
        lines =  f.readlines()
    for line in lines:
        line = line.strip()
        example_dict = json.loads(line)
        example_dicts.append(example_dict)
    return example_dicts

def convert_csv_to_json(filename):
    # 读取csv文件并将数据转换为json格式
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows



def convert_txt_to_jsonl(filename, res_filename):
    """
    方便起见，把txt数据转为jsonl并编号
    """
    with open(filename, encoding="utf-8") as f:
        lines = f.readlines()
    
    example_dicts = []
    for index, line in enumerate(lines):
        line = line.strip()
        example_dict = {
            "id": index,
            "text":line
        }
        example_dicts.append(example_dict)
    write_jsonl(res_filename, example_dicts)
    print(f"{res_filename}保存成功,共{len(example_dicts)}条数据")

def read_txt2list(filename):
    texts = []
    with open(filename, encoding="utf-8") as f:
        for line in f:
            texts.append(line.strip())
    return texts

def remove_duplicate_examples(examples, key_name="requirement"):
    print(f"去重前数量:{len(examples)}")
    has_texts = set()
    filter_examples = []
    for example in examples:
        text = example[key_name].lower()
        if text not in has_texts:
            filter_examples.append(example)
            has_texts.add(text)
    print(f"去重后数量:{len(filter_examples)}")
    return filter_examples

def extract_keywords(text):
    from rake_nltk import Rake
    # 使用Rake算法，同时传入自定义的停用词
    rake_nltk_var = Rake(stopwords=my_stopwords)
    # 提取关键词
    rake_nltk_var.extract_keywords_from_text(text)

    # 获取关键词短语排名
    keyword_ranked = rake_nltk_var.get_ranked_phrases()
    return keyword_ranked

def remove_punctuation(text):
    import re
    # 使用正则表达式去除标点符号：匹配所有非单词字符
    stripped_text = re.sub(r'[^\w\s]', '', text)
    return stripped_text