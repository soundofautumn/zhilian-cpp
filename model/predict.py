import pandas as pd
import torch
# from sklearn.preprocessing import StandardScaler
# from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import re

from model.preprocess import preprocess_test
from train import Model
from torch.utils.data import DataLoader, Dataset




class ComplexDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {'data': torch.tensor(self.data[idx], dtype=torch.float32)}
        return sample


def collate_fn(batch):
    features = []
    for _, item in enumerate(batch):
        features.append(item['data'].view(-1))  # 将输入数据的形状改为(30720)
    return torch.stack(features, 0)


if __name__ == '__main__':
    folder_path = "../../model/1718182751266233482/"
    data = preprocess_test(folder_path)

    # 创建测试数据集实例
    test_dataset = ComplexDataset(data)

    # 构建test_loader
    test_loader = DataLoader(test_dataset, batch_size=52, shuffle=False, collate_fn=collate_fn)

    # 加载模型

    model_path = './best_model.pth'   # 替换为你的模型路径
    model = torch.load(model_path, map_location='cpu')
    print(model)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 假设test_loader用于加载无标签的测试数据
    predictions = []

    with torch.no_grad():
        for inputs in test_loader:
            print(inputs.shape)
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)  # 获取最大概率的类别索引作为预测结果
            predictions.extend(preds.cpu().numpy())  # 将预测结果从GPU转移到CPU，并添加到列表中

    df_predictions = pd.DataFrame({'Prediction': predictions})

    # 将预测结果保存到CSV文件，提交时注意去除表头
    csv_output_path = '../result.csv'
    df_predictions.to_csv(csv_output_path, index=False, header=False)  # index=False避免将索引写入CSV文件，header=0避免写入列名

    print(f'Predictions have been saved to {csv_output_path}')
