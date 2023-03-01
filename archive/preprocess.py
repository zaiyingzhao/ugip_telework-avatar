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

y = [2 for _ in range(len(tiredness_value))]
for i in range(2):
    for j in range(600):
        y[600 * i + j] = i

for i in range(3):
    os.makedirs("./data/label" + str(i), exist_ok=True)

for i in range(len(tiredness_value)):
    print(i)
    image_tensor = torch.from_numpy(tiredness_value[i].astype(np.float32)).clone()
    path = "./data/label" + str(y[i])
    torch.save(image_tensor, path + f"/tensor{i}.pt")
