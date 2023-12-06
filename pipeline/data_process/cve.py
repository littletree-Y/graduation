"""
cve 的格式转换
"""

filename = "../../data/cve/allitems.csv"
output_name = "../../data/cve/cve.jsonl"
import pandas as pd
import sys
sys.path.append("..")
from util import *

# 读取 CSV 文件，并指定要读取的列
df = pd.read_csv(filename, usecols=[0, 1, 2], skiprows=9, encoding='ISO-8859-1',on_bad_lines='skip')

example_dicts = []

for index, row in df.iterrows():
    id = row[0]
    text = str(row[2])
    if not text.startswith("**"):
        example_dicts.append({
            "id" : id,
            "text" : text,
        })
write_jsonl(output_name, example_dicts)