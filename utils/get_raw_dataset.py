# @Time : 2024/5/10 10:31
# @Author : Cheng Yang
# @File ：get_raw_dataset.py

import pandas as pd
import pickle


def read_pickle_file(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        print("File not found.")
        return None
    except Exception as e:
        print("An error occurred:", e)
        return None


def save_to_csv(data, output_file):
    try:
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        print("Data saved to", output_file)
    except Exception as e:
        print("An error occurred while saving to CSV:", e)


file_path = "/content/MDFEND-Weibo21/data/val.pkl"
output_file = "/content/MDFEND-Weibo21/data/val.csv"

data = read_pickle_file(file_path)
if data is not None:
    print("Data:\n", data)
    save_to_csv(data, output_file)
