import pandas as pd
import numpy as np
import requests
import json
import cv2
import pickle
import base64
from collections import deque

import collect_data

def img_to_feature(img_bin):
    response = requests.post(collect_data.endpoint + collect_data.detect, 
                            data={
                            "api_key": collect_data.API_KEY,
                            "api_secret": collect_data.API_SECRET,
                            "image_base64":img_bin,
                            "return_landmark": 1,
                            "return_attributes": "gender,age,smiling,glass,headpose,blur,eyestatus,emotion,facequality,beauty,mouthstatus,eyegaze,skinstatus"
                            }
            )
    data = json.loads(response.text)

    if data["faces"]:
        item = collect_data.flatten_dict(data["faces"][0]).items()
        keys = ["frame"]
        keys += ["_".join(list(i[0])) for i in item]
        value = [[frame]]
        value[0] += [i[1] for i in item]
        df_tmp = pd.DataFrame(value, columns=keys)

        return df_tmp
    else:
        return None

if __name__ == "__main__":
    f = open("tiredness.txt", "w")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("cam cannot open.")
        exit()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, collect_data.WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, collect_data.HEIGHT)
    df = None
    frame=0
    reg = pickle.load(open("model.pkl", "rb"))
    buf = deque()
    while True :
        frame += 1
        ret, img = cap.read()
        cv2.imshow("video",img)
        #APIに渡す形式に変更
        result, dst_data = cv2.imencode('.jpg', img)
        img_bin = base64.b64encode(dst_data)

        X = img_to_feature(img_bin)
        if not (X is None):
            X = X.drop(["frame", "face_token"],axis=1)
            X["attributes_gender_value_Male"] = X.apply(lambda x: 1 if x["attributes_gender_value"] == "Male" else 0, axis=1)
            X["attributes_glass_value_Dark"] = X.apply(lambda x: 1 if x["attributes_glass_value"] == "Dark" else 0, axis=1)
            X["attributes_glass_value_Normal"] = X.apply(lambda x: 1 if x["attributes_glass_value"] == "Normal" else 0, axis=1)
            X = X.drop(["attributes_gender_value","attributes_glass_value"],axis=1)
            tiredness = reg.predict(X)[0]
            buf.append(tiredness)
            if len(buf)>10:
                buf.popleft()
            print(sum(buf)/len(buf), file=f) #直近10分間の平均値を1分毎にtiredness.txtに出力

        if cv2.waitKey(1000 * 60) & 0xFF == ord('q'): #1分に一回とる
            break

    cap.release()
    f.close()