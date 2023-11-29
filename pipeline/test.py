from util import *
from const import *
from tfidf import *
import random

if __name__ == "__main__":
    etcs_dicts = read_example_dict(etcs_file)
    cwe_dicts = read_example_dict(cwe_file)
    etcs = [tmp["text"] for tmp in etcs_dicts]
    cwes = [tmp["Name"] for tmp in cwe_dicts]
    matched_indices, scores = get_top_k(etcs, cwes, 2)
    # print(matched_indices[:1])
    # print(scores[0:1])
    random_list = random.sample(range(len(etcs)), 5)
    for i in random_list:
        print(etcs[i])
        for j in matched_indices[i]:
            print(cwes[j])
        print("="*10)