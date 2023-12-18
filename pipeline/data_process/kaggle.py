"""
kaggle数据集处理
"""
from util import *
import sys
sys.path.append("..")
from pipeline.util import *
raw_train_file = "../../data/requirement/kaggle/raw/nfr.txt"
raw_test_file = "../../data/requirement/kaggle/raw/test.txt"


def get_single_kaggle_data(filename):
    with open(filename, encoding='utf-8') as f:
        lines = f.readlines()
    example_dicts = []
    # 只保留功能需求和安全需求
    for line in lines:
        line = line.strip()
        if line.startswith("F:"):
            line = line.replace("F:", "", 1)
            tmp_dict = {
                "cate": "functional",
                "requirement":line.strip()
            }
            example_dicts.append(tmp_dict)
        if line.startswith("SE:"):
            line = line.replace("SE:", "", 1)
            tmp_dict = {
                "cate": "security",
                "requirement": line.strip()
            }
            example_dicts.append(tmp_dict)
    return example_dicts



def get_all_kaggle_data():
    example_dicts = []
    example_dicts.extend(get_single_kaggle_data(raw_train_file))
    example_dicts.extend(get_single_kaggle_data(raw_test_file))
    show_requirement_cate(example_dicts)
    return example_dicts

if __name__ == "__main__":
    # example_dicts = get_all_kaggle_data()
    # add_id2example_dicts(example_dicts)
    # output_name =  "../../data/requirement/kaggle/fun_se.jsonl"
    # write_jsonl(output_name, example_dicts)

    # 采样
    filename = "../../data/requirement/kaggle/fun_se.jsonl"
    sample_name = "../../data/requirement/kaggle/kaggle.jsonl"
    example_dicts = read_example_dict(filename)
    example_dicts = sample_fun_se(example_dicts, 1.0)
    show_requirement_cate(example_dicts)
    write_jsonl(sample_name, example_dicts)









