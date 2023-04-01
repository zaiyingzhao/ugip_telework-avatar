import base64
import json
import os

import cv2
import pandas as pd
import requests

from utils import api

WIDTH = 640
HEIGHT = 480

OUTPUT_FOLDER = "./data"


def flatten_dict(d, pre_lst=None, result=None):
    if result is None:
        result = {}
    if pre_lst is None:
        pre_lst = []
    for k, v in d.items():
        if isinstance(v, dict):
            flatten_dict(v, pre_lst=pre_lst + [k], result=result)
        else:
            result[tuple(pre_lst + [k])] = v
    return result


def read_img(img_bin, frame):
    response = requests.post(
        api.endpoint + api.detect,
        data={
            "api_key": api.API_KEY,
            "api_secret": api.API_SECRET,
            "image_base64": img_bin,
            "return_landmark": 1,
            "return_attributes": "gender,age,smiling,glass,headpose,blur,eyestatus,emotion,facequality,beauty,mouthstatus,eyegaze,skinstatus",
        },
    )
    data = json.loads(response.text)
    df_tmp = None
    if "faces" in data.keys():
        if data["faces"]:
            item = flatten_dict(data["faces"][0]).items()
            keys = ["frame"]
            keys += ["_".join(list(i[0])) for i in item]
            value = [[frame]]
            value[0] += [i[1] for i in item]
            df_tmp = pd.DataFrame(value, columns=keys)

    return len(data["faces"]), df_tmp


def set_img(img, df):
    row = df.iloc[-1, :]
    # 顔の範囲を描画
    left = row["face_rectangle_left"]
    top = row["face_rectangle_top"]
    right = left + row["face_rectangle_width"]
    down = top + row["face_rectangle_height"]
    cv2.rectangle(img, (left, top), (right, down), color=(0, 0, 255), thickness=10)
    # 感情を表示
    emo = [
        "attributes_emotion_anger",
        "attributes_emotion_disgust",
        "attributes_emotion_fear",
        "attributes_emotion_happiness",
        "attributes_emotion_neutral",
        "attributes_emotion_sadness",
        "attributes_emotion_surprise",
    ]
    emotion = pd.to_numeric(row[emo]).idxmax()[len("attributes_emotion_") :]
    cv2.putText(
        img,
        text=emotion,
        org=(int(WIDTH / 10), int(HEIGHT / 5)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1.0,
        color=(0, 255, 0),
        thickness=1,
        lineType=cv2.LINE_4,
    )
    # 肌の状態
    skin = [
        "attributes_skinstatus_health",
        "attributes_skinstatus_stain",
        "attributes_skinstatus_dark_circle",
        "attributes_skinstatus_acne",
    ]
    skin_s = pd.to_numeric(row[skin]).idxmax()[len("attributes_skinstatus_") :]
    cv2.putText(
        img,
        text=skin_s,
        org=(int(WIDTH / 10), int(HEIGHT * 2 / 5)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1.0,
        color=(255, 0, 0),
        thickness=1,
        lineType=cv2.LINE_4,
    )

    return img


def set_cap():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("cam cannot open.")
        exit()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    return cap


def collect_data(cap):
    frame = 0
    while True:
        frame += 1
        ret, img = cap.read()
        if ret:
            # APIに渡す形式に変更
            result, dst_data = cv2.imencode(".jpg", img)
            img_bin = base64.b64encode(dst_data)

            num_face, df_tmp = read_img(img_bin, frame)
            if num_face != 0:
                img = set_img(img, df_tmp)

                if os.path.isfile(f"{OUTPUT_FOLDER}/res.csv"):
                    df_tmp.to_csv(f"{OUTPUT_FOLDER}/res.csv", mode="a", header=False)
                else:
                    df_tmp.to_csv(f"{OUTPUT_FOLDER}/res.csv")
                print(frame)

            cv2.imshow("Video", img)
            if cv2.waitKey(1000) & 0xFF == ord("q"):
                break
