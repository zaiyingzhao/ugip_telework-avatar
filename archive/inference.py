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
num_classes = 3
transform = torchvision.transforms.Compose([])
original_dataset = torchvision.datasets.DatasetFolder(
    loader=torch.load, extensions=("pt"), root="./data", transform=transform
)
dataset = TensorDataset(original_dataset=original_dataset)
test_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(52, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.linear_relu_stack(x)
        return x


net = Net()
checkpoint = torch.load("model_weight.pth")
state_dict = checkpoint.state_dict()
net.load_state_dict(state_dict)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
net = net.to(device).eval()

test_acc = 0.0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = net(images)

        predicted = outputs.max(1)[1]
        test_acc += (predicted == labels).sum()

    avg_test_acc = test_acc / len(test_loader.dataset)

print(f"accuracy: {avg_test_acc}")
