"""
准备成relgan训练需要的数据
2023.12.23
"""
import re
import os
import sys
sys.path.append("..")
sys.path.append("../..")
from pipeline.util import *
import random
from util import *

datasets = ["SRRP", "DOSSPRE", "promise", "SecReq"]
# 输出文件
train_json_file = "../../data/requirement/four/train.jsonl"
test_json_file =  "../../data/requirement/four/test.jsonl"
# 提取文件夹路径
folder_path = os.path.dirname(test_json_file)
# 检查文件夹是否存在
if not os.path.exists(folder_path):
    # 如果不存在，则创建文件夹
    os.makedirs(folder_path)

train_file = "../../methods/relgan/data/four.txt"
test_file = "../../methods/relgan/data/testdata/test_four.txt"

train_keywords_file = "../../methods/relgan/data/four_keywords.txt"
test_keywords_file = "../../methods/relgan/data/testdata/test_four_keywords.txt"

# RelGAN 需要将标点符号分隔开
def separate_punctuation_and_words(sentence):
    # 使用正则表达式将标点符号和单词分开
    # \w+ 匹配一个或多个字母数字下划线字符（单词）
    # \W 匹配任何非单词字符（标点符号和空格等）
    pattern = re.compile(r'(\w+|\W)')

    # 使用正则表达式分割句子
    result = re.findall(pattern, sentence)

    # 将列表中的元素连接成字符串，用单个空格分隔
    separated_sentence = ' '.join(result)

    # 将多个空格替换为一个空格
    separated_sentence = re.sub(r'\s+', ' ', separated_sentence)

    return separated_sentence

def get_all_examples():
    examples = []
    for dataset in datasets:
        dataset_name = f"../../data/requirement/{dataset}/{dataset}.jsonl"
        examples.extend(read_example_dict(dataset_name))
    return examples


def make_relgan_data(examples, split_ratio = 0.8):
    #shuf and 分割
    random.shuffle(examples)
    # 去重
    examples = remove_duplicate_examples(examples)
    # 计算分割索引
    split_index = int(len(examples) * split_ratio)
    # 分割列表
    train_set = examples[:split_index]
    test_set = examples[split_index:]
    write_jsonl(train_json_file, train_set)
    write_jsonl(test_json_file, test_set)
    # 打印比例
    print("训练集：")
    show_requirement_cate(train_set)
    print("测试集：")
    show_requirement_cate(test_set)

def convert2relgan():
    train_set = read_example_dict(train_json_file)
    test_set = read_example_dict(test_json_file)
    train_texts = [ separate_punctuation_and_words(item["requirement"]) for item in train_set]
    test_texts = [separate_punctuation_and_words(item["requirement"]) for item in test_set]
    write_text_file(train_file, train_texts)
    write_text_file(test_file, test_texts)


def see_duplicate():
    train_set = read_example_dict(train_json_file)
    test_set = read_example_dict(test_json_file)
    train_texts = [ separate_punctuation_and_words(item["requirement"]) for item in train_set]
    test_texts = [separate_punctuation_and_words(item["requirement"]) for item in test_set]

    train_texts = [x.lower() for x in train_texts ]
    test_texts = [x.lower()  for x in test_texts]

    # all_texts = train_texts + test_texts
    # all_set = list(set(all_texts))
    # print("去重前:", len(all_texts))
    # print("去重后:", len(all_set))
    dup_num = 0
    for x in test_texts:
        if x in train_texts:
            dup_num += 1
            continue
    print(f"测试集在训练集里的数量：{dup_num}")


def convert2relgankeywords():
    # 只保留安全需求
    train_set = read_example_dict(train_json_file)
    test_set = read_example_dict(test_json_file)
    train_texts = []
    test_texts = []
    train_keywords = []
    test_keywords = []
    for item in train_set:
        if item["cate"] == "security":
            train_texts.append(separate_punctuation_and_words(item["requirement"]))
            keywords = extract_keywords(item["requirement"])
            keywords = get_keywords(keywords)
            train_keywords.append(separate_punctuation_and_words(keywords))
    assert len(train_texts) == len(train_keywords)

    for item in test_set:
        if item["cate"] == "security":
            test_texts.append(separate_punctuation_and_words(item["requirement"]))
            keywords = extract_keywords(item["requirement"])
            keywords = get_keywords(keywords)
            test_keywords.append(separate_punctuation_and_words(keywords))
    assert len(test_texts) == len(test_keywords)
    
    write_text_file(train_file, train_texts)
    write_text_file(train_keywords_file, train_keywords)
    write_text_file(test_file, test_texts)
    write_text_file(test_keywords_file, test_keywords)


def get_keywords(keywords):
    """
    从keywords里选择一个去除标点后不为空的
    """
    find_keywords = ""
    index = 0
    while len(find_keywords)==0 and index < len(keywords):
        tmp = keywords[index]
        tmp = remove_punctuation(tmp).strip()
        find_keywords = tmp
        index += 1
    if find_keywords == "":
        return "not find"
    return find_keywords


if __name__ == "__main__":
    # 汇总&采样
    # examples = get_all_examples()
    # make_relgan_data(examples)

    convert2relgankeywords()

    # see_duplicate()





