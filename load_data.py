import numpy as np
import pandas as pd

all_data = np.load('ZFish_dataNgenes.npz', allow_pickle=True) # Load data

def pre_process_data():
    def read_file(filename: str):
        if not filename.endswith('.csv'):
            raise Exception('Wrong file format.')
        return pd.read_csv(filename)

    file_names = all_data.files
    data_map = {}

    for file_name in file_names:
        df = pd.DataFrame(all_data[file_name])
        data_map[file_name] = df
    return data_map

processed_data_map = pre_process_data()
