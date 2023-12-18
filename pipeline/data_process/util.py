import random
import sys
sys.path.append("..")
from util import *


def show_requirement_cate(filename_example_dicts):
    example_dicts = []
    if isinstance(filename_example_dicts, list):
        example_dicts = filename_example_dicts
    elif isinstance(filename_example_dicts, str):
        filename_example_dicts = read_example_dicts(filename_example_dicts)
    else:
        print(f"{filename_example_dicts}参数错误")

    num = len(example_dicts)
    fun_num = 0
    se_num = 0
    for example_dict in example_dicts:
        if example_dict["cate"]=="functional":
            fun_num += 1
        if example_dict["cate"]=="security":
            se_num += 1
    print(f"需求总数：{num}, 功能需求：{fun_num}, 安全需求：{se_num}")

def add_id2example_dicts(example_dicts):
    for i, example_dict in enumerate(example_dicts):
        assert "id" not in example_dict
        example_dict["id"] = i
    return example_dicts

def sample_fun_se(example_dicts, ratio):
    """
    根据安全需求的数量采样功能需求
    ratio 表示 功能需求:安全需求的比例
    """
    fun_examples = []
    se_examples = []
    for example_dict in example_dicts:
        if example_dict["cate"]=="functional":
            fun_examples.append(example_dict)
        elif example_dict["cate"] == "security":
            se_examples.append(example_dict)

    se_num = len(se_examples)
    fun_num = int(se_num/(1.0 * ratio))

    sample_fun_example_dicts = random.sample(fun_examples, fun_num)
    print(f"采样后功能需求： {len(sample_fun_example_dicts)}, 安全需求: {len(se_examples)}")
    all_examples = se_examples
    all_examples.extend(sample_fun_example_dicts)
    return all_examples
