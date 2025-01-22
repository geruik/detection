"""
姿势-检测-Service-层.

@Time    :   2024/11/05 10:26:06
@Author  :   creativor
@Version :   1.0
"""

import base64
import io
import json
import multiprocessing
import time
import cv2
import numpy as np
import requests
import torch
from ultralytics import YOLO
from mods.utils import capture_stream_screen
from pos.pos_processor import PosProcessor
from PIL import Image
from loguru import logger
from ultralytics.utils.plotting import save_one_box

DEFAULT_MODELS_DIR = "default_models"
"""默认模型目录"""
DEFAULT_MODEL_FILE = "yolov8m-desk.pt"
"""默认模型文件"""
_INFERENCE_ARGS = {"device": "cuda", "conf": 0.5, "verbose": False}

class DeskService:
    """姿势检测Service-层"""

    def __init__(self):
         self.model = YOLO(DEFAULT_MODELS_DIR + "/" + DEFAULT_MODEL_FILE)


    def detect(self, *args):
        (rtsp_url, camera_sn, callback_url) = args
        try:
            """执行姿势检测

            Args:
                rtsp_url (str): rtsp视频流地址
                camera_sn (str): 摄像头编号
            """
            orig_img = capture_stream_screen(rtsp_url)
            timestamp = int(time.time() * 1000)
            # 使用YOLO模型进行检测
            results = self.model(orig_img, **_INFERENCE_ARGS)
            if len(results) == 0:
                logger.info(f"桌面洁净度识别未检测到符合的结果,对应的视频流信息rtsp:{rtsp_url},camera_sn:{camera_sn}")
                return
            result = results[0]
            if len(result) > 0:
                formatted_results = []
                # 绘制检测结果
                for r in result:
                    result_json = r.tojson(normalize=True)
                    if result_json != "[]":
                        for detection in json.loads(result_json):
                            label_name = detection["name"]
                            score = round(detection["confidence"], 5)
                            box = detection["box"]
                            formatted_result = {
                                "label_name": label_name,
                                "image_data": None,
                                "score": score,
                                "x0": box["x1"],
                                "y0": box["y1"],
                                "x1": box["x2"],
                                "y1": box["y2"],
                            }
                            box = np.array(
                            [
                                box["x1"],
                                box["y1"],
                                box["x2"],
                                box["y2"],
                            ],
                            dtype=np.float32,
                            )
                            box = np.array([[box[0], box[1]], [box[2], box[3]]])
                            box[:, 0] *= orig_img.shape[1]
                            box[:, 1] *= orig_img.shape[0]
                            box = box.astype(np.int32)
                            cv2.rectangle(
                                orig_img,
                                (int(box[0][0]), int(box[0][1])),
                                (int(box[1][0]), int(box[1][1])),
                                (255, 255, 0),
                                1,
                            )
                            formatted_results.append(formatted_result)
                if not formatted_results:
                    logger.info(f"桌面洁净度识别未识别到符合的结果,对应的视频流信息rtsp:{rtsp_url},camera_sn:{camera_sn}")
                    return
                result_img = numpy_to_base64(orig_img)
                json_cls = {
                    "background": result_img,
                    "camera_sn": camera_sn,
                    "time": timestamp,
                    "result_data": formatted_results,
                    "detect_type": "messdesk",
                    "face_detect": True,
                    "face_detect_type": "检测范围内",
                }
                logger.info(f"桌面洁净度识别进程识别到结果,向{callback_url}发送检测结果")
                try:
                    response = requests.post(
                        callback_url,
                        json=json_cls,
                        timeout=3,
                    )
                    if not response or response.status_code != 200:
                        logger.error(f"错误：桌面洁净度识别进程回调失败,摄像头编号：{camera_sn},回调地址：{callback_url}。")
                    else:
                        logger.info(f"桌面洁净度识别进程发送摄像头{camera_sn}检测结果成功")
                except Exception as e:
                    logger.error(
                        f"桌面洁净度识别进程发送检测结果给回调地址{callback_url}时抛出异常,摄像头编号：{camera_sn},异常信息{e}"
                    )

            else:  # 检测失败
                logger.error("桌面洁净度识别失败，或未识别到结果")
        except Exception as e:
            logger.exception(f"桌面洁净度识别异常，异常信息：{e}")

    def run(self, rtsp_url, camera_sn, callback_url):
        args = (rtsp_url, camera_sn, callback_url)
        process = multiprocessing.Process(target=self.detect, args=args)
        process.start()

def numpy_to_base64(img_array, quality=85):
    # 将 NumPy 数组转换为 PIL 图像
    img = Image.fromarray(img_array)

    # 创建字节流对象
    byte_io = io.BytesIO()

    # 将图像保存为 JPEG 格式，写入字节流
    img.save(byte_io, format="JPEG", quality=quality)

    # 获取字节流内容
    byte_data = byte_io.getvalue()

    # 将字节流内容转换为 Base64 编码
    base64_str = base64.b64encode(byte_data).decode("utf-8")

    return base64_str
