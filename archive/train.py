import numpy as np
import torchvision
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import re
import matplotlib.pyplot as plt


class TensorDataset(Dataset):
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
        correct_label = int(sig_id) // 600

        # format: (signature, correct_label)
        return (sig, correct_label)

    def __len__(self):
        return len(self.original_dataset)


batch_size = 128
transform = torchvision.transforms.Compose([])
original_dataset = torchvision.datasets.DatasetFolder(
    loader=torch.load, extensions=("pt"), root="./data", transform=transform
)
dataset = TensorDataset(original_dataset=original_dataset)

num_data = len(dataset)
num_classes = 3
val_size = int(0.1 * num_data)
train_size = num_data - val_size
train_dataset, valid_dataset = torch.utils.data.random_split(
    dataset=dataset, lengths=[train_size, val_size], generator=torch.Generator()
)
print("Dataset split: train={}, valid={}".format(train_size, val_size))
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(52, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.linear_relu_stack(x)
        return x


net = Net()
checkpoint = torch.load("model_weight2.pth")
state_dict = checkpoint.state_dict()
net.load_state_dict(state_dict)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []

nb_epoch = 3000

max_val_acc = 0.5141242742538452
for i in range(nb_epoch):
    train_loss = 0
    train_acc = 0
    val_loss = 0
    val_acc = 0

    # training
    net.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device)

        outputs = net(images)
        loss = criterion(outputs, labels)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

        predicted = outputs.max(1)[1]
        train_acc += (predicted == labels).sum()

    avg_train_loss = train_loss / len(train_loader.dataset)
    avg_train_acc = train_acc / len(train_loader.dataset)

    # evaluation
    net.eval()
    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            predicted = outputs.max(1)[1]
            val_acc += (predicted == labels).sum()

        avg_val_loss = val_loss / len(valid_loader.dataset)
        avg_val_acc = val_acc / len(valid_loader.dataset)

    print(
        f"Epoch [{(i+1)}/{nb_epoch}], loss: {avg_train_loss:.5f} acc: {avg_train_acc:.5f} val_loss: {avg_val_loss:.5f}, val_acc: {avg_val_acc:.5f}"
    )
    if avg_val_acc > max_val_acc:
        max_val_acc = avg_val_acc
        torch.save(net, "model_weight2.pth")
        print("weight saved to ./model_weight2.pth")

    train_loss_list.append(avg_train_loss)
    train_acc_list.append(avg_train_acc)
    val_loss_list.append(avg_val_loss)
    val_acc_list.append(avg_val_acc)

print(f"maximum val_acc: {max_val_acc}")

# # acc transition
# plt.figure(figsize=(8, 6))
# plt.plot(val_acc_list, label="val", lw=2, c="b")
# plt.plot(train_acc_list, label="train", lw=2, c="k")
# plt.title("acc transition")
# plt.xticks(size=14)
# plt.yticks(size=14)
# plt.grid(lw=2)
# plt.legend(fontsize=14)
# plt.show()
