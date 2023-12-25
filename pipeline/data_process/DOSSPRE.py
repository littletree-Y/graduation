
import pandas as pd
from util import *
import sys
sys.path.append("..")
from pipeline.util import *

fun_csv_file = "../../data/requirement/DOSSPRE/DOSSPRE 1.0Original.csv" #从这个里面拿功能需求
sec_csv_file = "../../data/requirement/DOSSPRE/DOSSPRE 2.0Binary.csv" #从这个里面拿安全需求

def get_fun_examples():
    df = pd.read_csv(fun_csv_file,  encoding='windows-1252')
    example_dicts = []
    # 遍历 DataFrame 的每一行
    for index, row in df.iterrows():
        requirement = row["Requirement"]
        # 去除单引号
        if requirement[0]=="'" and requirement[-1]=="'":
            requirement = requirement[1:len(requirement)-1]
        cate = row["Class"].lower()
        if cate == "functionality":
            tmp_dict = {
                "requirement": requirement.strip(),
                "cate": "functional"
            }
            example_dicts.append(tmp_dict)

    return example_dicts

def get_sec_examples():
    df = pd.read_csv(sec_csv_file,  encoding='windows-1252')
    example_dicts = []
    # 遍历 DataFrame 的每一行
    for index, row in df.iterrows():
        requirement = row["Requirement"]
        # 去除单引号
        if requirement[0] == "'" and requirement[-1] == "'":
            requirement = requirement[1:len(requirement) - 1]
        cate = row["Class"].lower()
        if cate == "sr":
            tmp_dict = {
                "requirement": requirement.strip(),
                "cate": "security"
            }
            example_dicts.append(tmp_dict)

    return example_dicts

def get_all_dosspre():
    example_dicts = []
    example_dicts.extend(get_fun_examples())
    example_dicts.extend(get_sec_examples())
    add_id2example_dicts(example_dicts, "DOSSPRE")
    show_requirement_cate(example_dicts)
    return example_dicts

if __name__ == "__main__":
    example_dicts = get_all_dosspre()
    output_name =  "../../data/requirement/DOSSPRE/DOSSPRE.jsonl"
    write_jsonl(output_name, example_dicts)