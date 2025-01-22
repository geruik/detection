# -*- encoding: utf-8 -*-
"""
Flask API, 提供人脸注册和删除接口

@Time    :   2024/04/28 14:25:39
@Author  :   liuxiao 
@Version :   1.0
"""
import io
import config
import log_config
from concurrent.futures import ThreadPoolExecutor
import base64
import json
import sys
import cv2
from flask import Flask, request
from loguru import logger
from PIL import Image
from io import BytesIO
import numpy as np
import requests
import face.facefunction
from pos.desk_service import DeskService
from pos.pos_service import PosService


app = Flask(__name__)
DATABASE_FILE = "face/face.db"
executor = ThreadPoolExecutor(max_workers=10)
pos_service = None
desk_service = None


@app.route("/detectPos", methods=["POST"])
def detect_pos():
    """对指定rtsp视频流进行姿势检测

    Returns:
        str: success
    """
    data = request.get_json()
    if data is not None:
        rtsp_url = data["rtspUrl"]
        camera_sn = data["cameraSN"]
        if rtsp_url is not None and camera_sn is not None:
            executor.submit(pos_service.detect, rtsp_url, camera_sn,config.detection["pose_detect_callback"])

    return success()

@app.route("/detectDesk", methods=["POST"])
def detect_desk():
    """对指定rtsp视频流进行姿势检测

    Returns:
        str: success
    """
    data = request.get_json()
    if data is not None:
        rtsp_url = data["rtspUrl"]
        camera_sn = data["cameraSN"]
        if rtsp_url is not None and camera_sn is not None:
            executor.submit(desk_service.run, rtsp_url, camera_sn,config.detection["desk_detect_callback"])
            logger.info("发起桌面洁净度检测")

    return success()


@app.route("/addFace", methods=["POST"])
def add_face():
    dataList = request.get_json()
    if len(dataList) > 0:
        frg = face.facefunction.getRecognizaObject(None)
        args = (frg, dataList)
        executor.submit(add_face_asyn, args)

    return success()


@app.route("/deleteFace", methods=["POST"])
def delete_face():
    data = request.get_json()
    if data is not None and "userIds" in data:
        id_str = ", ".join("'" + str(item) + "'" for item in data["userIds"])
        type = data["type"] if "type" in data else "common"
        frg = face.facefunction.getRecognizaObject(None)
        frg.delete(id_str, type)

    return success()


@app.route("/strangerSignChange", methods=["POST"])
def stranger_sign_change():
    data = request.get_json()
    if data is not None:
        original_sign = data["originalSign"]
        new_sign = data["newSign"]
        frg = face.facefunction.getRecognizaObject(None)
        frg.update_user_name(original_sign, new_sign)

    return success()


@app.route("/hello")
def hello():
    return "hello world"


def start(process_share):
    logger.info("flask执行初始化")
    recognize_object = face.facefunction.getRecognizaObject(process_share, False)
    global pos_service
    global desk_service
    pos_service = PosService(recognize_object)
    desk_service = DeskService()
    app.run(host=config.flask["flask_ip"], port=config.flask["flask_port"])


def success():
    return json.dumps({"code": 200, "message": "success"})


def fail(message):
    return json.dumps({"code": 500, "message": message})


def add_face_asyn(args):
    (
        frg,
        dataList,
    ) = args
    """
    异步新增人脸数据
    """
    for data in dataList:
        face_url = data["faceUrl"]
        if len(face_url) > 0:
            name = data["name"]
            face_type = "common"
            if "type" in data:
                face_type = data["type"]
            url_arr = face_url.split(",")
            for url in url_arr:
                # 对Base64字符串进行解码，得到字节数据
                image_bytes = base64.b64decode(url)
                # 使用BytesIO将字节数据包装成类文件对象，方便后续被Image.open使用
                image_file_like = io.BytesIO(image_bytes)
                image = Image.open(image_file_like)
                image_array = np.array(image)
                image_array = image_array[:, :, :3]
                try:
                    results = frg.detect(image_array)
                except Exception as e:
                    logger.error(f"识别用户[{name}]人脸失败，失败原因：{e}")
                if len(results) == 0:
                    logger.error(f"用户{name}的头像图片未识别到人脸，注册失败")
                elif len(results) > 1:
                    logger.error(
                        f"用户{name}的头像图片识别到多个人脸，请上传单人照片"
                    )
                else:
                    result = results[0]
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                    _, encoded_img = cv2.imencode(".jpg", image_array)
                    encoded_img_base64 = base64.b64encode(encoded_img).decode(
                        "utf-8"
                    )
                    face_image_base64 = (
                        f"data:image/jpg;base64,{encoded_img_base64}"
                    )
                    try:
                        frg.register(
                            data["userId"],
                            result["embedding"],
                            face_image_base64,
                            None,
                            None,
                            face_type,
                        )
                        logger.info(f"用户{name}注册人脸成功")
                    except Exception as e:
                        logger.error(f"用户{name}注册人脸失败，失败原因:{e}")
