from typing import Iterable, Dict
import gzip
import json
import os
import csv

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
