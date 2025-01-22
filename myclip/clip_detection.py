"""
相似识别线程管理.

@Time    :   2024/09/12 14:26:06
@Author  :   creativor
@Version :   1.0
"""

import base64
import ctypes
import time
import json
from types import SimpleNamespace
from enum import Enum, unique
from typing import Dict, List
from multiprocessing import Process, Value
import cv2
import numpy as np
import requests

from loguru import logger
from ultralytics.utils.plotting import save_one_box
from face.facefunction import FaceRecognition
import threads_config
from myclip.clipfunction import ClipRecognition, ClipClass
from mods.yolo_loaders import *
from mods.av_load_streams import *


class ClipDetectionThread:
    """
    检测线程
    """

    thread_id: str
    """线程id"""
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

    def is_running(self):
        """线程是否在运行中"""
        return self._running_status.value

    def start(self, args):
        """启动线程"""
        process_method = self.process_clip_detection
        # 启动子进程去执行
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

    def process_clip_detection(self, args, process_share):
        (streams, callback_url, classes, sleep_time, thread_id) = args
        # 创建相似识别类
        clip_classes = [ClipClass(class_config) for class_config in classes]
        clip_recognition = ClipRecognition(process_share)
        needFaceDetection = False
        for clip_class in clip_classes:
            if clip_class.needFaceDetection():
                needFaceDetection = True
                break
        # 创建人脸识别类
        face_recognition = None
        # 为节约资源，如果不需要人脸检测，则不创建人脸识别类
        if needFaceDetection:
            face_recognition = FaceRecognition(process_share)

        rtsp_url_sn_map = build_rtsp_sn_map(streams)
        rtsp_urls = [x["RTSP_URL"] for x in streams]
        dataset = AvLoadStreams(
            sources=make_streams_temp_file(rtsp_urls), grab_interval=sleep_time
        )
        for batch in dataset:
            paths, im0s, s = batch
            n = len(im0s)
            timestamp = int(time.time() * 1000)
            for i in range(n):
                im0 = im0s[i]
                path = paths[i]
                if path in rtsp_url_sn_map:
                    camera_sn = rtsp_url_sn_map[path]
                else:
                    continue
                encoded_img_base64 = None
                img_visual_output = clip_recognition.get_visual_output(im0)
                for clip_class in clip_classes:  # 遍历所有检测类别
                    detect_type = clip_class.class_name
                    face_detect = clip_class.needFaceDetection()
                    match = clip_class.detect(img_visual_output)
                    if not match:
                        continue
                    if encoded_img_base64 is None:
                        _, encoded_img = cv2.imencode(".jpg", im0)
                        encoded_img_base64 = base64.b64encode(encoded_img).decode(
                            "utf-8"
                        )
                    # 组装数据
                    formatted_results = []
                    formatted_result = {
                        "label_name": detect_type,
                        "image_data": None,
                        "score": 1,
                        "x0": None,
                        "y0": None,
                        "x1": None,
                        "y1": None,
                    }
                    formatted_results.append(formatted_result)
                    json_cls = {
                        "background": encoded_img_base64,
                        "camera_sn": camera_sn,
                        "time": timestamp,
                        "result_data": formatted_results,
                        "detect_type": detect_type,
                        "face_detect": face_detect,
                        "face_detect_type": "全图",
                    }
                    face_results = []
                    if face_detect:
                        face_results = face_recognition.recognition(im0)
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
                        f"线程{thread_id}监测到类别为[{detect_type}]的相似,向{callback_url}发送检测结果"
                    )
                    try:
                        response = requests.post(
                            callback_url,
                            json=json_cls,
                            timeout=3,
                        )
                        if not response or response.status_code != 200:
                            logger.error(f"错误：线程 {thread_id} 回调失败。")
                        else:
                            logger.info(
                                f"线程{thread_id}发送类别为[{detect_type}]相似检测结果成功"
                            )
                    except Exception as e:
                        logger.error(
                            f"线程{thread_id}发送检测结果给回调地址{callback_url}时抛出异常，异常信息{e}"
                        )

            time.sleep(sleep_time)


__initialized = False
"""这是一个模块级别的变量"""
threads: Dict[str, ClipDetectionThread] = {}
"""进程共享数据"""
shared_data = None


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
    config = threads_config.clip_load_config()
    if thread_name not in config:
        return responsify({"error": "线程名称不存在"}, 400)

    thread_config = config[thread_name]
    thread_id = thread_config["THREAD_ID"]
    if thread_id in threads and threads[thread_id].is_running():
        return responsify({"error": f"线程 {thread_name} 已经在运行中"}, 400)

    args = (
        thread_config["STREAMS"],
        thread_config["CALLBACK_URL"],
        thread_config["CLASSES"],
        thread_config["SLEEP_TIME"],
        thread_id,
    )

    detection_thread = ClipDetectionThread(thread_id=thread_id, config=thread_config)
    detection_thread.start(args)
    threads[thread_id] = detection_thread

    return responsify({"message": f"线程 {thread_name} 已启动"}, 200)


def stop_thread(thread_name):
    config = threads_config.clip_load_config()

    if thread_name not in config:
        return responsify({"error": "线程名称不存在"}, 400)

    thread_id = config[thread_name]["THREAD_ID"]
    if thread_id not in threads or not threads[thread_id]._running_status:
        return responsify({"error": f"线程 {thread_name} 已经停止或从未启动"}, 400)
    threads[thread_id].stop()
    del threads[thread_id]
    return responsify({"message": f"线程 {thread_name} 已停止"}, 200)


def responsify(txt_obj, code):
    """构造http模拟响应"""
    return SimpleNamespace(
        text=json.dumps(txt_obj, ensure_ascii=False), status_code=code
    )


def __do_init__(process_share):
    logger.info(f"模块 {__name__} 执行__do_init__...")
    global config
    config = threads_config.clip_load_config()
    global shared_data
    shared_data = process_share


def initialize(process_share):
    """初始化本模块"""
    global __initialized  # 声明 __initialized 为全局变量
    if __initialized is False:
        # 执行初始化操作
        __do_init__(process_share)
    __initialized = True
