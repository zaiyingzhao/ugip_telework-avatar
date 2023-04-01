import base64
import pickle
from collections import deque

import cv2
import numpy as np
import pandas as pd

from utils import client, collect_data, model, standardize


def calc_level(tiredness):
    if tiredness < 0.33:
        return 0
    elif tiredness < 0.67:
        return 1
    else:
        return 2


if __name__ == "__main__":
    # 相手のPCに繋ぐ
    c = client.connect2server()

    # 記録用ファイル
    f = open(f"{collect_data.OUTPUT_FOLDER}/tiredness.txt", "w")

    # カメラをセット
    cap = collect_data.set_cap()

    # 作成したモデルの読み込み
    reg = pickle.load(open("./model/model.pkl", "rb"))

    # バッファ
    buf_X = deque()
    buf_y = deque()

    frame = 0
    while True:
        frame += 1
        # カメラ画像読み込み
        ret, img = cap.read()

        # APIに渡す形式に変更
        result, dst_data = cv2.imencode(".jpg", img)
        img_bin = base64.b64encode(dst_data)

        # 画像を特徴量に変換
        num_face, X = collect_data.read_img(img_bin, frame)

        if not (X is None):
            # モデルに入れるように整形
            X = standardize.standardize(X)
            X = X.drop(
                [
                    "frame",
                    "face_token",
                    "face_rectangle_top",
                    "face_rectangle_left",
                    "face_rectangle_width",
                    "face_rectangle_height",
                ],
                axis=1,
            )
            X = model.ohe(X)

            # 過去20行分の平均をとって入力
            buf_X.append(X.values[0])
            if len(buf_X) > 20:
                buf_X.popleft()
            tiredness = reg.predict(
                pd.DataFrame([np.array(buf_X).mean(axis=0)], columns=X.columns)
            )[0]

            # 出力も過去10行分を平均した値にする
            buf_y.append(tiredness)
            if len(buf_y) > 10:
                buf_y.popleft()
            output = sum(buf_y) / len(buf_y)

            # ファイル出力
            print(output, file=f)

            # 3段階にレベルわけする
            tired_level = calc_level(output)

            # 相手のPCに送る
            client.send2server(c, tired_level)

            cv2.imshow("video", img)

        if cv2.waitKey(1000) & 0xFF == ord("q"):
            break

    cap.release()
    f.close()
    c.close()
