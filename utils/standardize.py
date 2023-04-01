import pandas as pd

from utils import collect_data


def standardize(df):
    df["center_x"] = df["face_rectangle_left"] + df["face_rectangle_width"] / 2
    df["center_y"] = df["face_rectangle_top"] + df["face_rectangle_height"] / 2
    # 正規化
    df.iloc[:, 6:172:2] = df.iloc[:, 6:172:2].apply(
        lambda x: (x - df["center_x"]) / df["face_rectangle_width"]
    )
    df.iloc[:, 7:172:2] = df.iloc[:, 7:172:2].apply(
        lambda x: (x - df["center_y"]) / df["face_rectangle_height"]
    )

    df = df.drop(["center_x", "center_y"], axis=1)

    return df


def standardize_data():
    # 読み込み
    df = pd.read_csv(f"{collect_data.OUTPUT_FOLDER}/res_tiredness.csv", index_col=0)
    # 正規化
    df = standardize(df)
    # 出力
    df.to_csv(f"{collect_data.OUTPUT_FOLDER}/res_tiredness_std.csv")
