import pandas as pd
import numpy as np
import requests
import json
import cv2
import pickle
import base64
from collections import deque

from setup import collect_data
from setup import standardize
from setup import model
from setup import api

def img_to_feature(img_bin):
    response = requests.post(api.endpoint + api.detect, 
                            data={
                            "api_key": api.API_KEY,
                            "api_secret": api.API_SECRET,
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
    cap = collect_data.set_cap()

    df = None
    reg = pickle.load(open("./model/model_onda.pkl", "rb"))
    #平均値を入れる
    buf_X = deque()
    buf_y = deque()
    frame = 0
    while True :
        frame += 1
        ret, img = cap.read()
        cv2.imshow("video",img)
        #APIに渡す形式に変更
        result, dst_data = cv2.imencode('.jpg', img)
        img_bin = base64.b64encode(dst_data)

        X = img_to_feature(img_bin)
        if not (X is None):

            X = standardize.standardize(X)
            X = X.drop(["frame", "face_token",
                        "face_rectangle_top", "face_rectangle_left", "face_rectangle_width", "face_rectangle_height"],axis=1)
            X = model.ohe(X)

            buf_X.append(X.values[0])
            if(len(buf_X)>20):
                buf_X.popleft()
            tiredness = reg.predict(pd.DataFrame([np.array(buf_X).mean(axis=0)], columns=X.columns))[0]
            buf_y.append(tiredness)
            if len(buf_y)>10:
                buf_y.popleft()
            print(sum(buf_y)/len(buf_y), file=f) #直近10分間の平均値を1分毎にtiredness.txtに出力

        if cv2.waitKey(1000) & 0xFF == ord('q'): #1分に一回とる
            break

    cap.release()
    f.close()