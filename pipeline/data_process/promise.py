
import pandas as pd
from util import *
import sys
sys.path.append("..")
from pipeline.util import *

csv_file = "../../data/requirement/promise/promise.csv"


def get_promise_example_dicts():
    df = pd.read_csv(csv_file, encoding='windows-1252')
    example_dicts = []
    # 遍历 DataFrame 的每一行
    for index, row in df.iterrows():
        requirement = row["RequirementText"]
        requirement = requirement.replace('\'', '').strip()
        cate = row["class"]
        if cate == "F":
            tmp_dict = {
                "requirement": requirement.strip(),
                "cate": "functional"
            }
            example_dicts.append(tmp_dict)
        if cate == "SE":
            tmp_dict = {
                "requirement": requirement.strip(),
                "cate": "security"
            }
            example_dicts.append(tmp_dict)
    return example_dicts

if __name__ == "__main__":
    # example_dicts = get_promise_example_dicts()
    # add_id2example_dicts(example_dicts)
    # show_requirement_cate(example_dicts)
    # output_name =  "../../data/requirement/promise/fun_se.jsonl"
    # write_jsonl(output_name, example_dicts)

    # 采样
    filename = "../../data/requirement/promise/fun_se.jsonl"
    sample_name =  "../../data/requirement/promise/promise.jsonl"
    example_dicts = read_example_dict(filename)
    example_dicts = sample_fun_se(example_dicts, 1.0)
    show_requirement_cate(example_dicts)
    write_jsonl(sample_name, example_dicts)