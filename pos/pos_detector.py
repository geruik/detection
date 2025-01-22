"""
姿势-检测器.

@Time    :   2024/11/04 09:26:06
@Author  :   creativor
@Version :   1.0
"""

from ultralytics import YOLO
from pos.ph_classifier import *
from PIL import Image
import cv2
from typing import Tuple, List

DEFAULT_MODELS_DIR = "default_models"
"""默认模型目录"""
DEFAULT_MODEL_FILE = "yolov8m-pose.pt"
"""默认模型文件"""

_INFERENCE_ARGS = {"device": "cuda", "conf": 0.5, "verbose": False}


class PosDetector:
    """姿势检测器."""

    model: YOLO
    """YOLO模型."""
    classifier: PH_Classifier
    """姿势分类器."""

    def __init__(self):
        # 初始化YOLO模型和分类器
        self.model = YOLO(DEFAULT_MODELS_DIR + "/" + DEFAULT_MODEL_FILE)
        self.classifier = PH_Classifier()

    def detect(self, im_array) -> Tuple[List[List[int]], List[List[str]]]:
        """
        根据输入图像进行姿势检测
        Args:
            im_array (array): 输入图像数组

        Returns:
             Tuple: 返回两个列表，分别是检测到的框和对应的姿势
                 List[List[int]]: 检测到的框坐标列表，每个元素都是一个长度为4的列表，表示左上角x、y和右下角x、y坐标
                 List[List[str]]: 检测到的姿势列表，代表在对应框中检测到的姿势(列表)
        """
        # 使用YOLO模型进行检测
        results = self.model(im_array, **_INFERENCE_ARGS)
        if len(results) == 0:
            return [], []
        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        result_boxes = []
        result_poses = []
        keypoints_data = result.keypoints.data

        for i, keypoints in enumerate(keypoints_data):
            if keypoints.shape[0] > 0:
                status = self.classifier.classify(keypoints, boxes[i])
                if status[0] != "other" or status[1] != "unknown":
                    result_boxes.append(boxes[i])
                    result_poses.append(status)

        return result_boxes, result_poses
