"""
物件检测线程管理.

@Time    :   2024/03/27 14:26:06
@Author  :   creativor,liuxiao
@Version :   2.0
"""

import base64
import ctypes
import datetime
import io
import re
import time
import json
from types import SimpleNamespace
from enum import Enum, unique
import sqlite3
from typing import Dict, List
import queue
from multiprocessing import Process, Value
import cv2
import numpy as np
import requests

from PIL import Image
from loguru import logger
from ultralytics import YOLO
from ultralytics.utils.plotting import save_one_box
from ultralytics.engine.results import Results
from paho.mqtt import client as MQTT
import threads_config
from face.facefunction import FaceRecognition
from mods.mqtt_client import MQTTClient
from mods.yolo_loaders import *
from mods.av_load_streams import *


@unique
class DetectionSource(Enum):
    """检测来源枚举,参考https://docs.ultralytics.com/modes/predict/#inference-sources"""

    STREAM = "STREAM"
    """视频流"""
    MQTT = "MQTT"
    """MQTT"""


class DetectionThread:
    """
    检测线程
    """

    thread_id: str
    """线程id"""
    que: queue.Queue = None
    """队列,只有MQTT模式会启用"""
    _running_status: ctypes.c_bool
    """运行状态,进程间共享"""
    config: dict = None
    """线程配置"""
    process: Process
    """对应的进程实例"""

    def __init__(self, thread_id: str, config: dict):
        self.thread_id = thread_id
        self._running_status = Value(ctypes.c_bool, True)
        self.config = config

    def start(self, args):
        """启动线程"""
        process_method = self.process_stream_detection
        # 如果是MQTT则:
        if DetectionSource.MQTT.value == self.config.get("SOURCE", None):
            process_method = self.process_mqtt_detection
        # AI检测是CPU密集的操作，而Python默认情况下同一时刻只有一个线程在执行 Python bytecode（参考https://docs.python.org/zh-cn/3/glossary.html#term-global-interpreter-lock），
        # 所以需要启动子进程去执行。
        process = Process(
            target=self.process_detection,
            args=args,
            kwargs={"process_method": process_method, "process_share": shared_data},
        )
        process.start()
        self.process = process

    def stop(self):
        """停止线程"""
        self._running_status.value = False
        self.process.terminate()
        logger.info(f"线程{self.thread_id}已停止")

    def is_running(self):
        """线程是否在运行中"""
        return self._running_status.value

    def process_detection(self, *args, **kwargs):
        """运行检测的总入口,执行实际的检测过程并捕获异常"""
        # 加载配置
        import log_config
        import config

        try:
            real_process_method = kwargs["process_method"]
            real_process_method(args, kwargs["process_share"])
        except BaseException as e:
            # 捕获所有异常，包括系统退出和中断
            logger.exception(f"检测线程{self.thread_id}发生异常[{e}],退出...")
        finally:
            self._running_status.value = False  # 无论如何重置运行状态

    def process_mqtt_detection(self, args, process_share):
        """运行MQTT物件检测"""
        (
            streams,
            callback_url,
            model_path,
            inference,
            plot,
            sleeptime,
            imgquality,
            thread_id,
            restrictarea,
            restrictareajson,
        ) = args
        # 初始化MQTT客户端
        mqtt_client: MQTTClient = MQTTClient(_on_mqtt_message, self)  # MQTT客户端实例
        try:
            mqtt_client.start()
        except Exception as e:
            logger.exception(f"启动MQTT客户端发生异常，异常信息{e}")
        # 只有MQTT源需要队列
        self.que = queue.Queue(maxsize=10000)
        logger.info(f"开始运行检测线程[{self.thread_id}]")
        local_model = YOLO(model_path)
        face_detect = plot["face_detect"] if "face_detect" in plot else False
        face_detect_type = (
            plot["face_detect_type"] if "face_detect_type" in plot else "检测范围内"
        )
        smoke_detect = plot["smoke_detect"] if "smoke_detect" in plot else False
        face_recognition = FaceRecognition(process_share) if face_detect else None
        while self._running_status.value:
            try:
                msg = self.que.get(block=True, timeout=60)
            except queue.Empty as e:
                continue
            # msg = deserialize_mqtt_message(msg)
            # logger.info(f"检测线程{self.thread_id}处理MQTT消息[{msg.payload}]")
            try:
                payload = json.loads(msg.payload)
                img_base64 = payload["values"]["image"]
                dev_mac = payload["values"]["devMac"].replace(":", "")
            except Exception as e:
                logger.exception(f"解析MQTT消息失败:{e}")
                continue
            if not img_base64:
                logger.error(f"检测线程{self.thread_id}处理MQTT消息时获取不到图片信息")
                continue
            # base64格式转换成numpy格式
            base64_data = re.sub("^data:image/.+;base64,", "", img_base64)
            image = base64_to_numpy(base64_data)
            # image = Image.open(io.BytesIO(image_data))
            # 检测图片
            results = local_model(image, **inference)
            if smoke_detect:
                _handle_smoke_result(
                    results[0],
                    callback_url,
                    dev_mac,
                    thread_id,
                    image,
                    restrictarea,
                    restrictareajson,
                    face_detect,
                    face_detect_type,
                    face_recognition,
                    imgquality,
                )
            else:
                _handle_result(
                    results[0],
                    callback_url,
                    dev_mac,
                    thread_id,
                    image,
                    restrictarea,
                    restrictareajson,
                    face_detect,
                    face_detect_type,
                    face_recognition,
                    imgquality,
                )

        logger.info(f"MQTT任务线程{thread_id}退出")

    def process_stream_detection(self, args, process_share):
        """运行视频流物件检测"""
        (
            streams,
            callback_url,
            model_path,
            inference,
            plot,
            sleeptime,
            imgquality,
            thread_id,
            restrictarea,
            restrictareajson,
        ) = args
        # 建立 RTSP_URL 与 CAMERA_SN 的映射关系字典
        rtsp_url_sn_map = build_rtsp_sn_map(streams)
        rtsp_urls = [x["RTSP_URL"] for x in streams]
        # # Run inference on the source, 参数请参考:https://docs.ultralytics.com/modes/predict/#inference-arguments
        local_model = YOLO(model_path)
        face_detect = plot["face_detect"] if "face_detect" in plot else False
        face_detect_type = (
            plot["face_detect_type"] if "face_detect_type" in plot else "检测范围内"
        )
        smoke_detect = plot["smoke_detect"] if "smoke_detect" in plot else False
        face_recognition = FaceRecognition(process_share) if face_detect else None

        my_load_streams = AvLoadStreams(
            sources=make_streams_temp_file(rtsp_urls), grab_interval=sleeptime
        )
        # results: List[Results] = local_model(
        #     source=MyLoadStreams(
        #         sources=make_streams_temp_file(rtsp_urls), grab_interval=sleeptime
        #     ),
        #     stream=True,
        #     **inference,
        # )
        # print("self:" + self.thread_id)
        try:
            for sources_result, images_result, bss_result in my_load_streams:
                if not self._running_status.value:
                    break
                for img_idx, image_result in enumerate(images_result):
                    path = sources_result[img_idx]
                    results = local_model(image_result, **inference)
                    for result in results:
                        camera_sn = rtsp_url_sn_map.get(
                            path
                        )  # 通过path找到对应摄像头编号
                        if smoke_detect:
                            _handle_smoke_result(
                                result,
                                callback_url,
                                camera_sn,
                                thread_id,
                                result.orig_img,
                                restrictarea,
                                restrictareajson,
                                face_detect,
                                face_detect_type,
                                face_recognition,
                                imgquality,
                            )
                        else:
                            _handle_result(
                                result,
                                callback_url,
                                camera_sn,
                                thread_id,
                                result.orig_img,
                                restrictarea,
                                restrictareajson,
                                face_detect,
                                face_detect_type,
                                face_recognition,
                                imgquality,
                            )
                time.sleep(sleeptime)
        except Exception as e:
            logger.exception(f"线程[ {thread_id} ]运行时出现异常，异常信息：{e}")
        finally:
            # Important: 强制关闭数据源，否则会有内存泄露
            # local_model.predictor.dataset.close()
            pass
        logger.info(f"线程[ {thread_id} ]运行过程遇到异常或被标记为停止，线程退出...")


# executor = None
# """线程池"""
DATABASE_FILE = "detection_data.db"
__initialized = False
"""这是一个模块级别的变量"""
threads: Dict[str, DetectionThread] = {}
"""进程共享数据"""
shared_data = None


def _create_database():
    with sqlite3.connect(DATABASE_FILE) as conn:
        c = conn.cursor()
        c.execute(
            """CREATE TABLE IF NOT EXISTS detection_data
                    (thread_id TEXT, detection_time TEXT, box_count INTEGER, max_confidence REAL, min_confidence REAL)"""
        )
        conn.commit()


def base64_to_numpy(base64_string):
    # 将 base64 编码的字符串解码为字节数据
    image_data = base64.b64decode(base64_string)
    # 将字节数据转换为图像数据
    nparr = np.frombuffer(image_data, np.uint8)
    # 从图像数据中读取图像
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    return image


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


# def _insert_detection_data(
#     thread_id, detection_time, box_count, max_confidence, min_confidence
# ):
#     with sqlite3.connect(DATABASE_FILE) as conn:
#         c = conn.cursor()
#         c.execute(
#             "INSERT INTO detection_data (thread_id, detection_time, box_count, max_confidence, min_confidence) VALUES (?, ?, ?, ?, ?)",
#             (thread_id, detection_time, box_count, max_confidence, min_confidence),
#         )
#         conn.commit()


def is_thread_running(thread_id: str) -> bool:
    """
    指定线程是否正在运行

    ---
    Args:
        thread_id: 线程ID
    ---
    Returs:
        True 代表正在运行, 否则为 False
    """
    if thread_id not in threads:
        return False
    thread = threads[thread_id]
    return thread.is_running()


def start_thread(thread_name):
    config = threads_config.load_config()
    if thread_name not in config:
        return responsify({"error": "线程名称不存在"}, 400)

    thread_id = config[thread_name]["THREAD_ID"]
    if thread_id in threads and threads[thread_id].is_running():
        return responsify({"error": f"线程 {thread_name} 已经在运行中"}, 400)

    thread_config = config[thread_name]
    inference = thread_config["INFERENCE_ARGS"]
    inference["verbose"] = False  # 禁止啰嗦模式，以免控制台失控
    if "classes" in inference:
        classes = list(map(int, inference["classes"]))
        if len(classes) > 0:
            inference["classes"] = classes
        else:
            del inference["classes"]
    args = (
        thread_config.get("STREAMS"),
        thread_config["CALLBACK_URL"],
        thread_config["MODEL_PATH"],
        inference,
        thread_config["PLOT_ARGS"],
        thread_config["SLEEP_TIME"],
        thread_config["IMGQUALITY"],
        thread_id,
        thread_config["RESTRICTAREA"],
        thread_config["RESTRICTAREAJSON"],
    )

    detection_thread = DetectionThread(thread_id=thread_id, config=config[thread_name])
    detection_thread.start(args)
    threads[thread_id] = detection_thread

    return responsify({"message": f"线程 {thread_name} 已启动"}, 200)


def stop_thread(thread_name):
    config = threads_config.load_config()

    if thread_name not in config:
        return responsify({"error": "线程名称不存在"}, 400)

    thread_id = config[thread_name]["THREAD_ID"]
    if thread_id not in threads or not threads[thread_id]._running_status:
        return responsify({"error": f"线程 {thread_name} 已经停止或从未启动"}, 400)
    threads[thread_id].stop()
    del threads[thread_id]
    return responsify({"message": f"线程 {thread_name} 已停止"}, 200)


def _handle_result(
    result: Results,
    callback_url,
    camera_sn,
    thread_id,
    frame,
    restrictarea,
    restrictareajson,
    face_detect,
    face_detect_type,
    face_recognition: FaceRecognition,
    quality,
):
    """解析检测结果参数"""
    if len(result) > 0:
        box_count = 0
        formatted_results = []
        timestamp = int(time.time() * 1000)
        # 将识别到的图片转换成base64
        source_im_array = result.plot()
        img_str = cv2.imencode(".jpg", source_im_array)[1].tostring()
        compressed_im_b64 = base64.b64encode(img_str).decode("utf-8")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for r in result:
            result_json = r.tojson(normalize=True)
            if result_json != "[]":
                for detection in json.loads(result_json):
                    label_name = detection["name"]
                    score = round(detection["confidence"], 5)
                    box = detection["box"]
                    box_count += 1
                    # 裁剪检测区域,并转换成base64
                    crop_img = save_one_box(r.boxes.xyxy, result.orig_img, save=False)
                    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                    img_str = cv2.imencode(".jpg", crop_img)[1].tostring()
                    crop_img_compressed_im_b64 = base64.b64encode(img_str).decode(
                        "utf-8"
                    )
                    formatted_result = {
                        "label_name": label_name,
                        "image_data": crop_img_compressed_im_b64,
                        "score": score,
                        "x0": box["x1"],
                        "y0": box["y1"],
                        "x1": box["x2"],
                        "y1": box["y2"],
                    }
                    # 检测范围内的人脸识别
                    if face_detect and face_detect_type == "检测范围内":
                        face_results = face_recognition.recognition(crop_img)
                        if len(face_results) > 0:
                            face_result = face_results[0]
                            dist_str = (
                                str(face_result["dist"].item())
                                if isinstance(face_result["dist"], np.ndarray)
                                else str(face_result["dist"])
                            )
                            user_name = face_result["user_name"]
                            if user_name != "unknown":
                                formatted_result["face_result"] = {
                                    "dist": dist_str,
                                    "user_name": user_name,
                                }
                    formatted_results.append(formatted_result)
                if restrictarea:
                    restrict_area_list = eval(restrictareajson)
                    restrict_area_coords = [
                        tuple(np.float32(x) for x in coord)
                        for coord in restrict_area_list
                    ]
                    restrict_area_coords = np.array(
                        restrict_area_coords, dtype=np.float32
                    )  # 数组转换
                    restrict_area_coords[:, 0] *= frame.shape[1]
                    restrict_area_coords[:, 1] *= frame.shape[0]
                    restrict_area_coords = restrict_area_coords.astype(np.int32)
                    cv2.polylines(
                        frame,
                        [restrict_area_coords],
                        isClosed=True,
                        color=(0, 255, 0),
                        thickness=2,
                    )  # 画多边形框

                    filtered_results = []
                    for detection in formatted_results:
                        box = np.array(
                            [
                                detection["x0"],
                                detection["y0"],
                                detection["x1"],
                                detection["y1"],
                            ],
                            dtype=np.float32,
                        )
                        box = np.array([[box[0], box[1]], [box[2], box[3]]])
                        box[:, 0] *= frame.shape[1]
                        box[:, 1] *= frame.shape[0]
                        box = box.astype(np.int32)
                        center_x = int(box[0][0] + (box[1][0] - box[0][0]) / 2)
                        center_y = int(box[0][1] + (box[1][1] - box[0][1]) / 2)
                        flag = cv2.pointPolygonTest(
                            restrict_area_coords, (center_x, center_y), False
                        )
                        if (
                            flag >= 0
                        ):  # 限制区域内，则添加到过滤结果中 大于0在区域内，小于0在区域外，就不写入
                            filtered_results.append(detection)
                            cv2.rectangle(
                                frame,
                                (int(box[0][0]), int(box[0][1])),
                                (int(box[1][0]), int(box[1][1])),
                                (0, 255, 0),
                                2,
                            )
                            # cv2.putText(
                            #     frame,
                            #     f"{detection['label_name']} {detection['score']:.2f}",
                            #     (int(box[1][0]), int(box[1][1]) - 5),
                            #     cv2.FONT_HERSHEY_SIMPLEX,
                            #     0.5,
                            #     (0, 255, 0),
                            #     2,
                            # )

                    # 没有检测到，就录入数据直接进入下一个循环
                    if not filtered_results:
                        logger.info(
                            f"线程 {thread_id} 在限制的检测框内没有检测到结果,不绘制"
                        )
                        return
                    else:
                        formatted_results = filtered_results
                        compressed_im_b64 = numpy_to_base64(frame, quality)
                else:
                    for detection in formatted_results:
                        box = np.array(
                            [
                                detection["x0"],
                                detection["y0"],
                                detection["x1"],
                                detection["y1"],
                            ],
                            dtype=np.float32,
                        )
                        box = np.array([[box[0], box[1]], [box[2], box[3]]])
                        box[:, 0] *= frame.shape[1]
                        box[:, 1] *= frame.shape[0]
                        box = box.astype(np.int32)
                        cv2.rectangle(
                            frame,
                            (int(box[0][0]), int(box[0][1])),
                            (int(box[1][0]), int(box[1][1])),
                            (0, 255, 0),
                            2,
                        )
                    compressed_im_b64 = numpy_to_base64(frame, quality)
        if not formatted_results:
            logger.info(f"线程 {thread_id} 未检测到符合的结果")
            return
        json_cls = {
            "background": compressed_im_b64,
            "camera_sn": camera_sn,
            "time": timestamp,
            "result_data": formatted_results,
            "detect_type": formatted_results[0]["label_name"],
            "face_detect": face_detect,
            "face_detect_type": face_detect_type,
        }
        # 全图的人脸识别
        if face_detect and face_detect_type == "全图":
            face_results = face_recognition.recognition(source_im_array)
            if len(face_results) > 0:
                face_results_json = []
                for face_result in face_results:
                    dist_str = (
                        str(face_result["dist"].item())
                        if isinstance(face_result["dist"], np.ndarray)
                        else str(face_result["dist"])
                    )
                    user_name = face_result["user_name"]
                    if user_name != "unknown":
                        face_results_json.append(
                            {"dist": dist_str, "user_name": user_name}
                        )
                json_cls["face_results"] = face_results_json
        # 所有的 label_name
        label_names = [item["label_name"] for item in formatted_results]
        logger.info(f"线程{thread_id}监测到{label_names},向{callback_url}发送检测结果")
        # logger.info(json_cls)
        try:
            response = requests.post(
                callback_url,
                json=json_cls,
                timeout=3,
            )
            if not response or response.status_code != 200:
                logger.error(f"错误：线程 {thread_id} 回调失败。")
            else:
                logger.info(f"线程{thread_id}发送检测结果成功")
        except Exception as e:
            logger.error(
                f"线程{thread_id}发送检测结果给回调地址{callback_url}时抛出异常，异常信息{e}"
            )


def _handle_smoke_result(
    result: Results,
    callback_url,
    camera_sn,
    thread_id,
    frame,
    restrictarea,
    restrictareajson,
    face_detect,
    face_detect_type,
    face_recognition: FaceRecognition,
    quality,
):
    frame_copy = np.copy(frame)
    result_json = result.tojson(normalize=True)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    is_detect = False
    if result_json != "[]":
        detections = json.loads(result_json)
        for detection in detections:
            if detection["name"] == "cig":
                for other_detection in detections:
                    if other_detection["name"] in ["head", "hand"]:
                        cig_box = detection["box"]
                        other_box = other_detection["box"]
                        if check_intersection(cig_box, other_box):
                            # 调整检测框并保存图片
                            x1 = int(cig_box["x1"] * frame.shape[1])
                            y1 = int(cig_box["y1"] * frame.shape[0])
                            x2 = int(cig_box["x2"] * frame.shape[1])
                            y2 = int(cig_box["y2"] * frame.shape[0])
                            # 扩大框的范围

                            x1, y1, x2, y2 = x1 - 20, y1 - 20, x2 + 20, y2 + 20
                            frame = cv2.rectangle(
                                frame, (x1, y1), (x2, y2), (255, 255, 0), 5
                            )
                            is_detect = True

        if is_detect:
            logger.info(f"线程{thread_id}检测到抽烟")
            # 组装数据
            formatted_results = []
            formatted_result = {
                "label_name": "smoke",
                "image_data": None,
                "score": 1,
                "x0": None,
                "y0": None,
                "x1": None,
                "y1": None,
            }
            formatted_results.append(formatted_result)
            compressed_im_b64 = numpy_to_base64(frame, quality)
            json_cls = {
                "background": compressed_im_b64,
                "camera_sn": camera_sn,
                "time": int(time.time() * 1000),
                "result_data": formatted_results,
                "detect_type": "smoke",
                "face_detect": face_detect,
                "face_detect_type": "全图",
            }
            if face_detect:
                face_results = face_recognition.recognition(frame_copy)
                if len(face_results) > 0:
                    face_results_json = []
                    for face_result in face_results:
                        dist_str = (
                            str(face_result["dist"].item())
                            if isinstance(face_result["dist"], np.ndarray)
                            else str(face_result["dist"])
                        )
                        user_name = face_result["user_name"]
                        if user_name != "unknown":
                            face_results_json.append(
                                {"dist": dist_str, "user_name": user_name}
                            )
                    json_cls["face_results"] = face_results_json
            logger.info(
                f"线程{thread_id}监测到[smoke],向{callback_url}发送检测结果"
            )
            # logger.info(json_cls)
            try:
                response = requests.post(
                    callback_url,
                    json=json_cls,
                    timeout=3,
                )
                if not response or response.status_code != 200:
                    logger.error(f"错误：线程 {thread_id} 回调失败。")
                else:
                    logger.info(f"线程{thread_id}发送检测结果成功")
            except Exception as e:
                logger.error(
                    f"线程{thread_id}发送检测结果给回调地址{callback_url}时抛出异常，异常信息{e}"
                )


def check_intersection(boxA, boxB):
    xA = max(boxA["x1"], boxB["x1"])
    yA = max(boxA["y1"], boxB["y1"])
    xB = min(boxA["x2"], boxB["x2"])
    yB = min(boxA["y2"], boxB["y2"])
    # 计算交集的面积
    interArea = max(0, xB - xA) * max(0, yB - yA)
    return interArea > 0


def responsify(txt_obj, code):
    """构造http模拟响应"""
    return SimpleNamespace(
        text=json.dumps(txt_obj, ensure_ascii=False), status_code=code
    )


def _on_mqtt_message(
    client: MQTT.Client, target_thread: DetectionThread, msg: MQTT.MQTTMessage
):
    """
    MQTT 消息到达.
    """
    # try:
    #     payload = json.loads(msg.payload)
    #     dev_mac = payload["values"]["devMac"].replace(":", "")
    # except Exception as e:
    #     logger.exception(f"解析MQTT消息失败:{e}")
    #     raise e
    # if not dev_mac:
    #     logger.error(f"MQTT{payload}消息找不到设备Mac地址")
    #     return
    # 直接转发给线程队列
    target_thread.que.put(msg)


# def serialize_mqtt_message(msg):
#     # 将MQTTMessage对象转换为可序列化的字典
#     return {
#         "topic": msg.topic,
#         "payload": msg.payload,
#     }


# def deserialize_mqtt_message(data):
#     # 从字典中恢复MQTTMessage对象
#     msg = MQTT.MQTTMessage()
#     msg.topic = data["topic"]
#     msg.payload = data["payload"]
#     # 根据需要添加其他字段
#     return msg


def __do_init__(data):
    logger.info(f"模块 {__name__} 执行__do_init__...")
    global config
    config = threads_config.load_config()
    global shared_data
    shared_data = data
    _create_database()
    # 测试初始化MQTT客户端，移到MQTT子进程中启动
    # mqtt_client = MQTTClient(_on_mqtt_message)
    # try:
    #     mqtt_client.start()
    # except Exception as e:
    #     logger.error(f"启动MQTT客户端发生异常，异常信息{e}")


def initialize(data):
    """初始化本模块"""
    global __initialized  # 声明 __initialized 为全局变量
    if __initialized is False:
        # 执行初始化操作
        __do_init__(data)
    __initialized = True
