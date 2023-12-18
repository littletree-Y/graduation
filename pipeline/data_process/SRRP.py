"""

"""

import pandas as pd
from util import *
import sys
sys.path.append("..")
from pipeline.util import *

csv_file_path = "../../data/requirement/SRRP/Software Requirements Risk Predection.csv"



def get_SRRP_examples():
    # 使用 pandas 读取 CSV 文件
    df = pd.read_csv(csv_file_path)
    example_dicts = []
    # 遍历 DataFrame 的每一行
    for index, row in df.iterrows():
        # 在这里处理每一行的数据
        id = row["id"]
        requirement = row["requirement_text"]
        cate = row["requirement_category"].lower()
        if cate == "functional":
            tmp_dict = {
                "id":id,
                "requirement":requirement.strip(),
                "cate" : "functional"
            }
            example_dicts.append(tmp_dict)
        if cate == "security":
            tmp_dict = {
                "id": id,
                "requirement": requirement.strip(),
                "cate": "security"
            }
            example_dicts.append(tmp_dict)
    return example_dicts



if __name__ == "__main__":
    # example_dicts = get_SRRP_examples()
    # show_requirement_cate(example_dicts)
    # output_name =  "../../data/requirement/SRRP/fun_se.jsonl"
    # write_jsonl(output_name, example_dicts)

    # 采样
    filename = "../../data/requirement/SRRP/fun_se.jsonl"
    sample_name = "../../data/requirement/SRRP/SRRP.jsonl"
    example_dicts = read_example_dict(filename)
    example_dicts = sample_fun_se(example_dicts, 1.0)
    show_requirement_cate(example_dicts)
    write_jsonl(sample_name, example_dicts)