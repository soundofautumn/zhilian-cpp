import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau  # 导入ReduceLROnPlateau


# 定义模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(30720, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(1024, 612),
            nn.BatchNorm1d(612),
            nn.LeakyReLU()
        )

        self.layer3 = nn.Linear(612, 11)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.dropout(x)

        return x


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
                labels.append(label)
                data_row_float16 = np.frombuffer(data_row_bin, dtype=np.float16)  # 原始数据是float16，直接把二进制bin读成float16的数组
                data_row_float16 = np.array(data_row_float16)
                data.append(data_row_float16)
    return data, labels


if __name__ == '__main__':
    # 开始
    print("start")

    # 加载数据
    folder_path = "../../model/train_set_remake"
    data, labels = preprocess(folder_path)

    # 加载成功
    print("load data success")

    # 划分数据集
    train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.4, random_state=42)


    class ComplexDataset(Dataset):
        def __init__(self, data, labels):
            self.data = data
            self.labels = labels

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            sample = {'data': torch.tensor(self.data[idx], dtype=torch.float32),
                      'label': torch.tensor(self.labels[idx], dtype=torch.long)}
            return sample


    train_dataset = ComplexDataset(train_data, train_labels)
    val_dataset = ComplexDataset(val_data, val_labels)


    def collate_fn(batch):
        features = []
        labels = []
        for _, item in enumerate(batch):
            features.append(item['data'])
            labels.append(item['label'])
        return torch.stack(features, 0), torch.stack(labels, 0)


    train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=48, shuffle=False, collate_fn=collate_fn)

    # 定义模型
    model = Model()
    criterion = nn.CrossEntropyLoss()
    # 调整优化器参数
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-8)

    # 初始化学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # 训练模型
    num_epochs = 1000
    patience = 10
    best_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    for epoch in range(num_epochs):
        if early_stop:
            print("Early stopping")
            break

        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # 验证集上的评估
        model.eval()
        val_loss = 0
        total_correct = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == targets).sum().item()

        val_loss /= len(val_loader)
        accuracy = total_correct / len(val_dataset)
        print(f'Epoch {epoch + 1}, Validation Accuracy: {accuracy}, Validation Loss: {val_loss}')
        # torch.save(model, f'./model/model/model_{int(accuracy * 10000)}.pth')
        # Check for early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            torch.save(model, f'best_model.pth')
        else:
            epochs_no_improve += 1
            print("!")

        if epochs_no_improve >= patience:
            early_stop = True

        # 在每个epoch结束后，更新学习率
        # scheduler.step()
        scheduler.step(val_loss)

    # Save the final model
    torch.save(model, 'final_model.pth')

    print(torch.seed())
    # save seed
    with open('seed.txt', 'w') as f:
        f.write(str(torch.seed()))
