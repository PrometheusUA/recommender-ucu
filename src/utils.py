import pandas as pd
import json


def read_json_df(file_path):
    data = []
    with open(file_path, encoding="utf8") as data_file:
        for line in data_file:
            data.append(json.loads(line))
    res_df = pd.DataFrame(data)
    return res_df
