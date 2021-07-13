import pandas as pd 
import numpy as np
import json


def read_json(fullpath):
    with open(fullpath, "r") as json_file_readed:
        json_readed = json.load(json_file_readed)
    return json_readed

def return_json(filepath):
    df = pd.read_csv(filepath)
    return df.to_json() 