import pandas as pd
import numpy as np
import torchvision
from torchvision import models
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch
import os
import re


class SignatureDataset(Dataset):
    def __init__(
        self,
        original_dataset: torchvision.datasets.DatasetFolder,  # (sig, correct_label)
    ):
        self.original_dataset = original_dataset

    def __getitem__(self, index):
        sig, validity = self.original_dataset.__getitem__(index)
        sig_path = self.original_dataset.samples[index][0]
        sig_filename = os.path.basename(sig_path)
        # TODO: 拡張子は必要に応じて変える
        m = re.match(r"tensor(.*).pt", sig_filename)
        sig_id = m.group(1)
        assert sig_id.isnumeric()
        # get pred_label
        # pred_label = int(self.metadata[sig_id]["pred_label"])
        # pred_label = sig_id // 300
        # correct_label_from_metadata = int(self.metadata[sig_id]["correct_label"])
        correct_label = sig_id // 300

        # format: (signature, correct_label, pred_label, validity)
        return (sig, correct_label)

    def __len__(self):
        return len(self.original_dataset)


batch_size = 32
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
original_dataset = torchvision.datasets.DatasetFolder(
    loader=torch.load, extensions=("pt"), root="./data", transform=transform
)
dataset = SignatureDataset(original_dataset=original_dataset)
num_data = len(dataset)
val_size = 0.2 * num_data
train_size = num_data - val_size
# FIXME:
print(dataset[0])
train_dataset, valid_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)
print("Dataset split: train={}, valid={}".format(train_size, val_size))
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)

model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 6)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []

nb_epoch = 20

for i in range(nb_epoch):
    train_loss = 0
    train_acc = 0
    val_loss = 0
    val_acc = 0

    # 学習
    net.train()

    for images, labels in train_loader:

        # 勾配の初期化(ループの頭でやる必要あり)
        optimizer.zero_grad()

        # 訓練データの準備
        images = images.to(device)
        labels = labels.to(device)

        # 順伝搬計算
        outputs = net(images)

        # 誤差計算
        loss = criterion(outputs, labels)
        train_loss += loss.item()

        # 学習
        loss.backward()
        optimizer.step()

        # 予測値算出
        predicted = outputs.max(1)[1]

        # 正解件数算出
        train_acc += (predicted == labels).sum()

    # 訓練データに対する損失と精度の計算
    avg_train_loss = train_loss / len(train_loader.dataset)
    avg_train_acc = train_acc / len(train_loader.dataset)

    # 評価
    net.eval()
    with torch.no_grad():

        for images, labels in valid_loader:

            # テストデータの準備
            images = images.to(device)
            labels = labels.to(device)

            # 順伝搬計算
            outputs = net(images)

            # 誤差計算
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # 予測値算出
            predicted = outputs.max(1)[1]

            # 正解件数算出
            val_acc += (predicted == labels).sum()

        # 検証データに対する損失と精度の計算
        avg_val_loss = val_loss / len(valid_loader.dataset)
        avg_val_acc = val_acc / len(valid_loader.dataset)

    print(
        f"Epoch [{(i+1)}/{nb_epoch}], loss: {avg_train_loss:.5f} acc: {avg_train_acc:.5f} val_loss: {avg_val_loss:.5f}, val_acc: {avg_val_acc:.5f}"
    )
    train_loss_list.append(avg_train_loss)
    train_acc_list.append(avg_train_acc)
    val_loss_list.append(avg_val_loss)
    val_acc_list.append(avg_val_acc)
