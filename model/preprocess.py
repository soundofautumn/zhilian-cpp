import os
import re

import numpy as np
from sklearn.preprocessing import StandardScaler
from natsort import natsorted


def data_preprocess(data: np.ndarray) -> np.ndarray:
    return data


def preprocess(folder_path):
    labels = []
    data = []
    cnt = 0
    for filename in os.listdir(folder_path):
        cnt += 1
        if filename.endswith(".bin"):
            match = re.search(r'label_(\d+)_', filename)
            if match:
                label = int(match.group(1))
            else:
                continue
            with open(os.path.join(folder_path, filename), 'rb') as file:
                data_row_bin = file.read()
                data_row_float16 = np.frombuffer(data_row_bin, dtype=np.float16)  # 原始数据是float16，直接把二进制bin读成float16的数组
                data.append(data_preprocess(data_row_float16))
                labels.append(label)
    return data, labels


def preprocess_test(folder_path):
    data = []
    cnt = 0
    for filename in natsorted(os.listdir(folder_path)):
        cnt += 1
        if filename.endswith(".bin"):
            with open(os.path.join(folder_path, filename), 'rb') as file:
                data_row_bin = file.read()
                data_row_float16 = np.frombuffer(data_row_bin, dtype=np.float16)  # 原始数据是float16，直接把二进制bin读成float16的数组
                data.append(data_preprocess(data_row_float16))
    return data
