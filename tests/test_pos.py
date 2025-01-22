# -*- encoding: utf-8 -*-
"""
测试姿势检测模块
"""

import time

import cv2
from pos.pos_processor import PosProcessor
from PIL import Image


test_rtsp_url = "rtsp://192.168.0.213:8554/rtsp/pos"
pos_processor = PosProcessor(test_rtsp_url)
pos_processor.run()
result = pos_processor.receive_result()
if result.is_successful():  # 检测成功
    print("检测姿势成功:"+str(result))
    orig_img = result.data.orig_img
    boxes = result.data.boxes
    poses = result.data.poses
    result_img = None
    # 绘制检测结果
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        result_img = cv2.rectangle(
            orig_img,
            (x1, y1),
            (x2, y2),
            (255, 255, 0),
            1,
        )
        cv2.putText(
            result_img,
            f"{poses[i]}",
            (x1 + 8, y2 - 8),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            thickness=0,
            color=(255, 255, 255),
        )
    # 显示结果
    im = Image.fromarray(result_img)
    im.save("pos_result.jpg")
else:  # 检测失败
    print("检测姿势失败：", result.message)
