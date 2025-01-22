"""
人脸识别线程管理.

@Time    :   2024/03/27 14:26:06
@Author  :   creativor,liuxiao
@Version :   2.0
"""

import base64
import ctypes
import os
import pickle
import time
import json
from types import SimpleNamespace
from enum import Enum, unique
import sqlite3
from typing import Dict, List
from multiprocessing import Process, Value
import cv2
import numpy as np
import pandas as pd
import requests

from loguru import logger
from ultralytics.utils.plotting import save_one_box
import threads_config
from face.facefunction import FaceRecognition
from mods.yolo_loaders import *
from mods.av_load_streams import *




class FaceDetectionThread:
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
        process_method = self.process_face_detection
        # 启动子进程去执行
        process = Process(
            target=self.process_detection,
            args=args,
            kwargs={"process_method": process_method, "process_share":shared_data},
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
            real_process_method(args,kwargs["process_share"])
        except BaseException as e:
            # 捕获所有异常，包括系统退出和中断
            logger.exception(f"检测线程{self.thread_id}发生异常[{e}],退出...")
        finally:
            self._running_status.value = False  # 无论如何重置运行状态

    def process_face_detection(self,args,process_share):
        (
            streams,
            callback_url,
            sleep_time,
            thread_id,
            det_score,
            expression_conf,
        ) = args
        # 创建人脸识别类
        face_recognition = FaceRecognition(process_share)

        rtsp_url_sn_map = build_rtsp_sn_map(streams)
        rtsp_urls = [x["RTSP_URL"] for x in streams]
        dataset=AvLoadStreams(sources=make_streams_temp_file(rtsp_urls), grab_interval=sleep_time)
        for batch in dataset:
            paths, im0s, s = batch
            n = len(im0s)
            formatted_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            for i in range(n):
                im0 = im0s[i]
                path = paths[i]
                if path in rtsp_url_sn_map:
                    camera_sn = rtsp_url_sn_map[path]
                else:
                    continue
                face_results = face_recognition.recognition_advance(im0, det_score=det_score, expression_conf=expression_conf, emotion_detect=True)
                if len(face_results) > 0:
                    logger.info(f"线程{thread_id}检测到人脸")
                    result_list = []
                    for result in face_results:
                        
                        # 假设 result["bbox"] 的结构是 [x1, y1, x2, y2]
                        bbox = result["bbox"]
                        x1, y1, x2, y2 = bbox

                        # 计算扩大后的边界框坐标
                        margin_x = int((x2 - x1) * 0.75)  # 计算x方向扩大的0.75倍范围
                        margin_y = int((y2 - y1) * 0.75)  # 计算y方向扩大的0.75倍范围

                        # 确保新的坐标在图像范围内
                        new_x1 = max(0, x1 - margin_x)
                        new_y1 = max(0, y1 - margin_y)
                        new_x2 = min(im0.shape[1], x2 + margin_x)  # frame.shape[1] 是图像的宽度
                        new_y2 = min(im0.shape[0], y2 + margin_y)  # frame.shape[0] 是图像的高度

                        # 使用新的边界框坐标来截取 face_img
                        face_img = im0[new_y1:new_y2, new_x1:new_x2]
                        # 将图片编码成 base64 格式
                        _, encoded_img = cv2.imencode(".jpg", face_img)
                        encoded_img_base64 = base64.b64encode(encoded_img).decode("utf-8")
                        result_list.append({
                                "user_name": result["user_name"],
                                "face_image": encoded_img_base64,
                                "camera_sn": camera_sn,
                                "detection_time": formatted_time,
                                "type": result["type"],
                                "emotion": result["emotion"],
                                "similarity": result["similarity"]
                        })
                    logger.info(f"线程{thread_id}监测到人脸,向{callback_url}发送检测结果")
                    try:
                        response = requests.post(
                            callback_url,
                            json=result_list,
                            timeout=3,
                        )
                        if not response or response.status_code != 200 :
                            logger.error(f"错误：线程 {thread_id} 回调失败。")
                        else:
                            logger.info(f"线程{thread_id}发送人脸检测结果成功")
                    except Exception as e:
                        logger.error(f"线程{thread_id}发送检测结果给回调地址{callback_url}时抛出异常，异常信息{e}")
                

            time.sleep(sleep_time)


def _create_database():
    with sqlite3.connect(DATABASE_FILE) as conn:
        c = conn.cursor()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS faces_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id TEXT,
            user_name TEXT NOT NULL,
            face_img BLOB NOT NULL,
            dist TEXT,
            detection_time TEXT,
            camera_sn TEXT
            )
            """
        )
        conn.commit()



__initialized = False
DATABASE_FILE = "face_detection_results.db"
"""这是一个模块级别的变量"""
threads: Dict[str, FaceDetectionThread] = {}
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
    config = threads_config.face_load_config()
    if thread_name not in config:
        return responsify({"error": "线程名称不存在"}, 400)

    thread_config = config[thread_name]
    thread_id = thread_config["THREAD_ID"]
    if thread_id in threads and threads[thread_id].is_running():
        return responsify({"error": f"线程 {thread_name} 已经在运行中"}, 400)

    
    args = (
        thread_config["STREAMS"],
        thread_config["CALLBACK_URL"],
        thread_config["SLEEP_TIME"],
        thread_id,
        thread_config["DET_SCORE"] if "DET_SCORE" in thread_config else 0.7,
        thread_config["EXPRESSION_CONF"] if "EXPRESSION_CONF" in thread_config else 0.6,
    )

    detection_thread = FaceDetectionThread(thread_id=thread_id, config=thread_config)
    detection_thread.start(args)
    threads[thread_id] = detection_thread

    return responsify({"message": f"线程 {thread_name} 已启动"}, 200)


def stop_thread(thread_name):
    config = threads_config.face_load_config()

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
    config = threads_config.face_load_config()
    global shared_data
    shared_data = process_share
    _create_database()


def initialize(process_share):
    """初始化本模块"""
    global __initialized  # 声明 __initialized 为全局变量
    if __initialized is False:
        # 执行初始化操作
        __do_init__(process_share)
    __initialized = True
