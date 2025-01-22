# -*- encoding: utf-8 -*-
"""
相似-检测-功能.

@Time    :   2024/09/12 10:38:06
@Author  :   creativor 
@Version :   1.0
"""
import cv2
import torch
from PIL import Image
import config

from myclip.transforms import Resize, CenterCrop, ToTensor, Normalize
from myclip.clip_loader import CLIPVisionModelWithProjection

DEFAULT_MODELS_DIR = "default_models"
"""默认模型目录"""

CONFIGED_CLASSES = config.clip["classes"]
"""获取配置文件中的clip模型类别信息"""


# 这里需要修改！！！
def image2tensor(images, size=224):
    """燕进提供的原代码"""

    def preprocess(size, img):
        return Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        )(ToTensor()(CenterCrop(size)(Resize(size)(img)).convert("RGB")))

    processed_images = [
        preprocess(size, Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        for img in images
    ]
    return torch.stack(processed_images)


def getClassPromptFile(class_name):
    """获取指定检测类别的prompt文件路径"""
    for class_config in CONFIGED_CLASSES:
        if class_config["class"] == class_name:
            return class_config["file"]
    return None


class ClipClass:
    """
    相似检测类
    """

    def __init__(self, class_config):
        self.class_name = class_config["CLASS"]
        self.prompt_ternsor = torch.load(getClassPromptFile(self.class_name))
        self.prompt_tensor_transpose = self.prompt_ternsor.t()  # 转置
        """检测Prompt对应的张量"""
        self.config = class_config
        """配置信息"""

    def needFaceDetection(self) -> bool:
        """是否需要人脸检测

        Returns:
            bool: True 如果需要，否则False
        """
        return self.config.get("FACE_DETECTION")

    def detect(self, image_visual_output) -> bool:
        """检测图片中是否有相似行为

        Args:
            visual_output: 用于检测图像的视觉输出张量

        Returns:
            boolean: 是否相似
        """
        similarity = torch.matmul(image_visual_output, self.prompt_tensor_transpose)
        final_similarity = similarity.cpu().detach().numpy()[0]
        similarity_threshold = self.config["SIMILARITY_THRESHOLD"]
        return final_similarity > similarity_threshold


class ClipRecognition:
    """
    相似检测类
    """

    model: CLIPVisionModelWithProjection
    """模型"""
    # clip_detect: torch.nn.Module
    # """相似检测文本prompts"""

    def __init__(self, process_share):
        # 预留的参数
        self.process_share = process_share
        ## 不用tokenize，加载模型直接加载bin,可以把其他文件删掉了。。。。。
        self.model = CLIPVisionModelWithProjection(
            DEFAULT_MODELS_DIR + "/clipmodel.bin"
        )
        # self.clip_detect = torch.load("myclip/clip.pt")

    def get_visual_output(self, image):
        """获取图片的视觉输出张量

        Args:
            image: 用于检测的图像

        Returns:
            Tensor: 视觉输出张量
        """
        images = [image]
        image_tensors = image2tensor(images)
        visual_output = self.model(image_tensors)
        # 这里修改这个visual_output.image_embeds
        visual_output = visual_output.image_embeds
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        visual_output = torch.mean(visual_output, dim=0)
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        return visual_output

    # def detect(self, image):
    #     """检测图片中是否有相似行为

    #     Args:
    #         image: 用于检测的图像

    #     Returns:
    #         long: 相似度值
    #     """
    #     images = [image]
    #     image_tensors = image2tensor(images)
    #     visual_output = self.model(image_tensors)
    #     # 这里修改这个visual_output.image_embeds
    #     visual_output = visual_output.image_embeds
    #     visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
    #     visual_output = torch.mean(visual_output, dim=0)
    #     visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
    #     similarity = torch.matmul(visual_output, self.clip_detect.t())
    #     final_similarity = similarity.cpu().detach().numpy()[0]
    #     return final_similarity
