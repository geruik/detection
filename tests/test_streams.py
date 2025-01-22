# -*- encoding: utf-8 -*-
"""
单检测进程接入多路摄像头性能测试脚本.
"""
import gc
import multiprocessing
from typing import Dict, List
import time
import random
import tempfile
import sys, os
from ultralytics import YOLO
from ultralytics.utils import ops
from ultralytics.engine.results import Results
from mods.yolo_loaders import *
from mods.av_load_streams import *
from face.facefunction import FaceRecognition


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function {func.__name__} executed in: {execution_time:.6f} seconds")
        return result

    return wrapper


RTSP_URL_LIST = [
    "rtsp://192.168.0.213:8554/rtsp/185",
    "rtsp://admin:weile123456@192.168.0.106/Streaming/Channels/101",
    "rtsp://admin:weile123456@192.168.0.145/Streaming/Channels/101",
    "rtsp://admin:weile123456@192.168.0.182/Streaming/Channels/101",
    "rtsp://admin:wl123456@192.168.0.183/Streaming/Channels/101",
    "rtsp://admin:weile123456@192.168.0.185/Streaming/Channels/101",
    "rtsp://admin:weile123456@192.168.0.220/Streaming/Channels/101",
    "rtsp://admin:12345@192.168.0.120:554/live/main",
    "rtsp://admin:weile123456@192.168.0.103/cam/realmonitor?channel=1&subtype=0",
]
"""已知的视频流列表
"""


def generate_test_streams_conf(count: int = -1) -> List[object]:
    """生成测试的视频流配置列表

    Args:
        count (_type_, optional): 生成的视频流数目.
    Returns:
        list[object]: 生成的视频流配置列表
    """
    CAMERA_SN = 310
    if count is None or count < 1:
        count = len(RTSP_URL_LIST)
    is_random = count > len(RTSP_URL_LIST)
    i = 0
    result = []
    while i < count:
        random_rtsp = random.choice(RTSP_URL_LIST) if is_random else RTSP_URL_LIST[i]
        result.append({"RTSP_URL": random_rtsp, "CAMERA_SN": str(CAMERA_SN + i)})
        i += 1
    return result


# 创建输出目录
result_output_directory = "runs/detect/predict"
os.makedirs(result_output_directory, exist_ok=True)


# 解析摄像头数量参数
argv = sys.argv
stream_count = 0 if len(argv) < 2 else int(argv[1])
STREAMS_CONF = generate_test_streams_conf()


# 建立 RTSP_URL 与 CAMERA_SN 的映射关系字典
rtsp_url_sn_map = build_rtsp_sn_map(STREAMS_CONF)
# Load a model
model: YOLO = YOLO("models/drink20240527.pt")  # pretrained YOLOv8n model

INFERENCE_ARGS = {
    "imgsz": 512,
    "conf": 0.1,
    "iou": 0.5,
    "device": "cuda",
    # "classes": [0],
    "verbose": False,
}


streams = [x["RTSP_URL"] for x in STREAMS_CONF]

# 模拟人脸识别
face_recognition = FaceRecognition(multiprocessing.Value("i", 0))

# Run batched inference on a list of images
my_load_streams = AvLoadStreams(make_streams_temp_file(streams))


# results = model(
#     source=my_load_streams, stream=True, **INFERENCE_ARGS
# )  # return a generator of Results objects


# # 模拟暂时不进行检测
# while True:
#     my_load_streams.__next__()
#     # 强制启动垃圾回收
#     gc.collect()
#     time.sleep(1)

# Process results generator
i = 0
result: Results

for sources_result, images_result, bss_result in my_load_streams:
    for img_idx, image_result in enumerate(images_result):
        # 人脸识别
        face_results = face_recognition.recognition(image_result)
        if len(face_results) > 0:
            face_result = face_results[0]
            dist_str = (
                str(face_result["dist"].item())
                if isinstance(face_result["dist"], np.ndarray)
                else str(face_result["dist"])
            )
            user_name = face_result["user_name"]
        path = sources_result[img_idx]
        start_time = time.time()
        # 物件识别
        results = model(
            image_result, **INFERENCE_ARGS
        )  # return a generator of Results objects
        for result in results:
            count = len(result)
            if count <= 1:
                # print(f"[{path}]--无检测结果nnnn...")
                continue
            camera_sn = rtsp_url_sn_map.get(path)
            print(f"摄像头[{camera_sn}]--检测到物件数量: {count}")
            boxes = result.boxes  # Boxes object for bounding box outputs
            masks = result.masks  # Masks object for segmentation masks outputs
            keypoints = result.keypoints  # Keypoints object for pose outputs
            probs = result.probs  # Probs object for classification outputs
            obb = result.obb  # Oriented boxes object for OBB outputs
            # result.show()  # display to screen
            i = i + 1

            result.save(
                filename=f"{result_output_directory}/result-{camera_sn}-{i}.jpg"
            )  # save to disk

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"一次检测执行时间: {execution_time*1000:.6f} 毫秒")
        # 强制启动垃圾回收
        # gc.collect()
        # torch.cuda.empty_cache()
    time.sleep(1)

# # 获取迭代器对象
# iterator = iter(results)

# while True:
#     try:
#         start_time = time.time()

#         # 获取下一个元素
#         result = next(iterator)
#         # 人脸识别
#         face_results = face_recognition.recognition(result.orig_img)
#         if len(face_results) > 0:
#             face_result = face_results[0]
#             dist_str = (
#                 str(face_result["dist"].item())
#                 if isinstance(face_result["dist"], np.ndarray)
#                 else str(face_result["dist"])
#             )
#             user_name = face_result["user_name"]

#         # 物件识别
#         path = result.path
#         count = len(result)
#         if count <= 1:
#             # print(f"[{path}]--无检测结果nnnn...")
#             continue
#         camera_sn = rtsp_url_sn_map.get(path)
#         print(f"摄像头[{camera_sn}]--检测到物件数量: {count}")
#         boxes = result.boxes  # Boxes object for bounding box outputs
#         masks = result.masks  # Masks object for segmentation masks outputs
#         keypoints = result.keypoints  # Keypoints object for pose outputs
#         probs = result.probs  # Probs object for classification outputs
#         obb = result.obb  # Oriented boxes object for OBB outputs
#         # result.show()  # display to screen
#         i = i + 1

#         result.save(
#             filename=f"{result_output_directory}/result-{camera_sn}-{i}.jpg"
#         )  # save to disk

#         end_time = time.time()
#         execution_time = end_time - start_time
#         print(f"一次检测执行时间: {execution_time*1000:.6f} 毫秒")
#         # 强制启动垃圾回收
#         # gc.collect()
#         torch.cuda.empty_cache()
#         time.sleep(1)
#     except StopIteration:
#         # 迭代结束
#         break
#     except ConnectionError:
#         # 连接视频流失败
#         break
