import pandas as pd
import numpy as np
import xgboost as xgb
import pickle

df = pd.read_csv("res_tiredness.csv", index_col=0)
X = df.drop(["frame", "face_token", "tiredness"],axis=1)
y = df.loc[:,["tiredness"]]
X = pd.get_dummies(X)

#とりあえず。実際にデータとってから調整
reg = xgb.XGBRegressor()
reg.fit(X,y)
pickle.dump(reg, open("model.pkl", "wb"))