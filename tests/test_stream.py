# -*- encoding: utf-8 -*-
"""
yolo检测-流模式-测试脚本.
"""
import time
import os
from ultralytics import YOLO
from ultralytics.engine.results import Results

# 创建输出目录
result_output_directory = "runs/detect/predict"
os.makedirs(result_output_directory, exist_ok=True)


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


# Run batched inference on a list of images
results = model(
    source="rtsp://admin:12345@192.168.0.120:554/live/main",
    stream=True,
    **INFERENCE_ARGS,
)  # return a generator of Results objects

# Process results generator
i = 0
result: Results

# 获取迭代器对象
iterator = iter(results)

while True:
    try:
        start_time = time.time()

        # 获取下一个元素
        result = next(iterator)

        path = result.path
        count = len(result)
        if count <= 1:
            print(f"[{path}]--无检测结果nnnn...")
            continue
        print(f"检测到物件数量: {count}")
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        # result.show()  # display to screen
        i = i + 1

        result.save(
            filename=f"{result_output_directory}/result-stream-{i}.jpg"
        )  # save to disk

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"一次检测执行时间: {execution_time*1000:.6f} 毫秒")

        time.sleep(1)
    except StopIteration:
        # 迭代结束
        break
    except ConnectionError:
        # 连接视频流失败
        break
