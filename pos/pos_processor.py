"""
姿势-检测-处理程序.

@Time    :   2024/11/04 14:26:06
@Author  :   creativor
@Version :   1.0
"""

import multiprocessing
import time
from typing import Tuple, List
from numpy import ndarray
from multiprocessing import Process
from loguru import logger


SUCCESS_CODE = 200
"""代表成功返回的代码"""
MAX_WAIT_TIME_OUT = 60
"""最大等待检测结果时间"""


class DetectionResultData:
    """姿势检测结果数据"""

    orig_img: ndarray
    """原始图片"""
    boxes: List[List[int]]
    """检测到的框坐标列表，每个元素都是一个长度为4的列表，表示左上角x、y和右下角x、y坐标"""
    poses: List[List[str]]
    """检测到的姿势列表，代表在对应框中检测到的姿势(列表)"""
    timestamp: int
    """linux时间戳,单位ms"""

    def __init__(self, orig_img, boxes, poses, timestamp):
        self.orig_img = orig_img
        self.boxes = boxes
        self.poses = poses
        self.timestamp = timestamp


class DetectionResult:
    """姿势检测结果"""

    status_code: int = SUCCESS_CODE
    """状态码，200代表成功，其他代表失败"""
    message: str = None
    """消息，成功时候为None，失败时候代表失败信息"""
    data: DetectionResultData
    """姿势检测结果数据，失败时候为None"""

    def success(self, data: DetectionResultData):
        """设置成功返回的data

        Args:
            data (DetectionData): 成功检测的data
        """
        self.status_code = SUCCESS_CODE
        self.data = data

    def fail(self, status_code: int, message: str):
        """设置失败返回

        Args:
            status_code (int): 失败状态码
            message (str): 失败消息
        """
        self.status_code = status_code
        self.message = message

    def is_successful(self) -> bool:
        """是否成功返回"""
        return self.status_code == SUCCESS_CODE


class PosProcessor:
    """姿势检测处理程序"""

    process: Process = None
    """子进程"""

    def __init__(self, rtsp_url: str):
        """

        Args:
            rtsp_url (str): 目标rtsp流地址
        """
        self.rtsp_url = rtsp_url

        # 创建管道用于父子进程通信
        self.parent_conn, self.child_conn = multiprocessing.Pipe()

    def do_detect(self):
        """子进程执行检测并将结果发送到父进程"""
        from ultralytics import YOLO
        from pos.pos_detector import PosDetector
        from mods.utils import capture_stream_screen
        import cv2

        result: DetectionResult = DetectionResult()
        try:
            orig_img = capture_stream_screen(self.rtsp_url)
            timestamp = int(time.time() * 1000)
            # 检测姿势
            pos_detector = PosDetector()
            boxes, poses = pos_detector.detect(orig_img)
            resultData = DetectionResultData(orig_img, boxes, poses, timestamp)
            result.success(resultData)
        except Exception as e:
            logger.exception("检测姿势失败:", e)
            # 检测失败
            result.fail(500, str(e))

        # 发送检测结果
        self.child_conn.send(result)
        self.child_conn.close()

    def receive_result(self) -> DetectionResult:
        """父进程接收检测结果"""
        result: DetectionResult
        if self.parent_conn.poll(MAX_WAIT_TIME_OUT):
            result = self.parent_conn.recv()
        else:
            result = DetectionResult()
            result.fail(
                500, f"超过最大等待时间[f{MAX_WAIT_TIME_OUT}]s,结果仍然没有返回"
            )
        self.process.terminate()
        return result

    def run(self):
        """启动子进程执行姿势检测"""
        process = multiprocessing.Process(target=self.do_detect)
        self.process = process
        process.start()
