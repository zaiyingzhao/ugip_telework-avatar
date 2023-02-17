import pandas as pd
import numpy as np
import xgboost as xgb
import pickle

df = pd.read_csv("res_tiredness.csv", index_col=0)
X = df.drop(["frame", "face_token", "tiredness"],axis=1)
y = df.loc[:,["tiredness"]]

X["attributes_gender_value_Male"] = X.apply(lambda x: 1 if x["attributes_gender_value"] == "Male" else 0, axis=1)
X["attributes_glass_value_Dark"] = X.apply(lambda x: 1 if x["attributes_glass_value"] == "Dark" else 0, axis=1)
X["attributes_glass_value_Normal"] = X.apply(lambda x: 1 if x["attributes_glass_value"] == "Normal" else 0, axis=1)
X = X.drop(["attributes_gender_value","attributes_glass_value"],axis=1)

#とりあえず。実際にデータとってから調整
reg = xgb.XGBRegressor()
reg.fit(X,y)
pickle.dump(reg, open("model.pkl", "wb"))