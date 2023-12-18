
import pandas as pd
from util import *
import sys
sys.path.append("..")
from pipeline.util import *

cpn_csv_file = "../../data/requirement/SecReq/CPN.csv"
epurse_csv_file = "../../data/requirement/SecReq/ePurse-selective.csv"
gps_csv_file = "../../data/requirement/SecReq/GPS.csv"


def get_single_file(filename):
    example_dicts = []
    try:
        df = pd.read_csv(filename, header=None, sep=';', encoding='windows-1252')
    except:
        df = pd.read_csv(filename, header=None, sep=';', encoding='latin1')
    for index, row in df.iterrows():
        requirement = row[0].strip()
        try:
            cate = row[1].lower().strip()
        except:
            continue
        if cate == "nonsec":
            tmp_dict = {
                "requirement": requirement.strip(),
                "cate": "functional"
            }
            example_dicts.append(tmp_dict)
        if cate == "sec":
            tmp_dict = {
                "requirement": requirement.strip(),
                "cate": "security"
            }
            example_dicts.append(tmp_dict)
    return example_dicts

def get_all():
    example_dicts = []
    example_dicts.extend(get_single_file(cpn_csv_file))
    example_dicts.extend(get_single_file(epurse_csv_file))
    example_dicts.extend(get_single_file(gps_csv_file))
    show_requirement_cate(example_dicts)
    return example_dicts

if __name__ == "__main__":
    example_dicts = get_all()
    example_dicts = add_id2example_dicts(example_dicts)
    output_name = "../../data/requirement/SecReq/secreq.jsonl"
    write_jsonl(output_name, example_dicts)
