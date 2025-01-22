"""
姿势-检测-Service-层.

@Time    :   2024/11/05 10:26:06
@Author  :   creativor
@Version :   1.0
"""

import base64
import io
import cv2
import numpy as np
import requests
import torch
from face.facefunction import FaceRecognition
from pos.pos_processor import PosProcessor
from PIL import Image
from loguru import logger
from ultralytics.utils.plotting import save_one_box


class PosService:
    """姿势检测Service-层"""

    def __init__(self,face_recognition):
         self.face_recognition = face_recognition


    def detect(self, rtsp_url, camera_sn, callback_url):
        try:
            """执行姿势检测

            Args:
                rtsp_url (str): rtsp视频流地址
                camera_sn (str): 摄像头编号
            """
            pos_processor = PosProcessor(rtsp_url)
            pos_processor.run()
            result = pos_processor.receive_result()
            # TODO: 处理检测结果并回调到业务层
            if result.is_successful():  # 检测成功            
                orig_img = result.data.orig_img
                result_img = numpy_to_base64(orig_img)
                timestamp = result.data.timestamp
                boxes = result.data.boxes
                poses = result.data.poses
                formatted_results = []
                # 绘制检测结果
                for i in range(len(boxes)):
                    bbox = torch.from_numpy(boxes[i]).to(dtype=torch.float32)
                    crop_img = save_one_box(bbox, orig_img, save=False)
                    # crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                    results = self.face_recognition.recognition_advance(crop_img)
                    if len(results) > 0:
                        face_result = results[0]
                        x1, y1, x2, y2 = boxes[i]
                        state1, state2 = poses[i]
                        formatted_result = {
                            "label_name": state1 + ',' + state2,
                            "image_data": None,
                            "score": 1,
                            # "x0": x1,
                            # "y0": y1,
                            # "x1": x2,
                            # "y1": y2,
                            "face_result": {
                                        "user_name": face_result["user_name"],
                                        "type": face_result["type"]
                                    }
                        }
                        orig_img = cv2.rectangle(
                            orig_img,
                            (x1, y1),
                            (x2, y2),
                            (255, 255, 0),
                            1,
                        )
                        formatted_results.append(formatted_result)
                if not formatted_results:
                    logger.info(f"姿态识别未识别到符合的结果,对应的视频流信息rtsp:{rtsp_url},camera_sn:{camera_sn}")
                    return
                result_img = numpy_to_base64(orig_img)
                json_cls = {
                    "background": result_img,
                    "camera_sn": camera_sn,
                    "time": timestamp,
                    "result_data": formatted_results,
                    "detect_type": formatted_results[0]["label_name"],
                    "face_detect": True,
                    "face_detect_type": "检测范围内",
                }
                logger.info(f"姿态识别进程识别到结果,向{callback_url}发送检测结果")
                try:
                    response = requests.post(
                        callback_url,
                        json=json_cls,
                        timeout=3,
                    )
                    if not response or response.status_code != 200:
                        logger.error(f"错误：姿态识别进程回调失败,摄像头编号：{camera_sn},回调地址：{callback_url}。")
                    else:
                        logger.info(f"姿态识别进程发送摄像头{camera_sn}检测结果成功")
                except Exception as e:
                    logger.error(
                        f"姿态识别进程发送检测结果给回调地址{callback_url}时抛出异常,摄像头编号：{camera_sn},异常信息{e}"
                    )

            else:  # 检测失败
                logger.error("姿态识别失败，或未识别到结果：", result.message)
        except Exception as e:
            logger.exception(f"姿态识别异常，异常信息：{e}")


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
