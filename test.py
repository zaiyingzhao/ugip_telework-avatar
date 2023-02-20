import pandas as pd
import numpy as np
import torchvision
from torchvision import models
import torch.nn as nn
import torch
import os

pd.set_option("display.max_columns", None)
# pd.set_option("display.max_rows", None)
tiredness = pd.read_csv("./res_tiredness.csv")

columns = tiredness.columns
columns = columns[175:-2]
tiredness = tiredness[columns]

for column in columns:
    max_value = tiredness[column].max()
    tiredness[column] /= max_value

tiredness_value = tiredness.values
# print(tiredness_value[0])
# print(len(tiredness_value[0]))
# print(len(tiredness_value))

y = [5 for _ in range(len(tiredness_value))]
for i in range(5):
    for j in range(300):
        y[300 * i + j] = i

# transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

# model = models.resnet50(pretrained=True)
# model.fc = nn.Linear(model.fc.in_features, 6)

# net = model.to(device)

for i in range(6):
    os.makedirs("./data/label" + str(i), exist_ok=True)

for i in range(len(tiredness_value)):
    print(i)
    image_tensor = torch.from_numpy(tiredness_value[0].astype(np.float32)).clone()
    path = "./data/label" + str(y[i])
    torch.save(image_tensor, path + f"/tensor{i}.pt")
