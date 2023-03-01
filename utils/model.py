import pandas as pd
import numpy as np
import xgboost as xgb
import pickle

from utils import collect_data

def ohe(X):
    #カテゴリ変数をワンホットに
    X["attributes_gender_value_Male"] = X.apply(lambda x: 1 if x["attributes_gender_value"] == "Male" else 0, axis=1)
    X["attributes_glass_value_Dark"] = X.apply(lambda x: 1 if x["attributes_glass_value"] == "Dark" else 0, axis=1)
    X["attributes_glass_value_Normal"] = X.apply(lambda x: 1 if x["attributes_glass_value"] == "Normal" else 0, axis=1)
    X = X.drop(["attributes_gender_value","attributes_glass_value"],axis=1)

    return X

def make_model():
    #読み込み
    df = pd.read_csv(f"{collect_data.OUTPUT_FOLDER}/res_tiredness_std.csv", index_col=0)
    X = df.drop(["frame", "face_token", 
                "face_rectangle_top", "face_rectangle_left", "face_rectangle_width", "face_rectangle_height",
                "tiredness"],axis=1)
    y = df.loc[:,["tiredness"]]

    #カテゴリ変数をワンホットに
    X = ohe(X)
    
    #平均とる
    mean_X = []
    for i in range(19,len(X)):
        mean_X.append(list(X[i-19:i+1].mean().values))
    mean_X = np.array(mean_X)
    y = y[19:]
    mean_X = pd.DataFrame(mean_X, columns=X.columns)

    #モデル作成
    reg = xgb.XGBRegressor()
    reg.fit(mean_X,y)
    pickle.dump(reg, open("./model/model_onda.pkl", "wb"))