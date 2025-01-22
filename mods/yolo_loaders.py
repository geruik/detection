# -*- encoding: utf-8 -*-
"""
定制的yolo data loaders.

@Time    :   2024/05/28 14:26:06
@Author  :   creativor 
@Version :   1.0
"""
import tempfile
import time
from ultralytics.data.loaders import LoadStreams, SourceTypes
import math
import os
from pathlib import Path
from loguru import logger
from threading import Thread
import numpy as np
from typing import Dict, List

import cv2
import torch


from ultralytics.utils import LOGGER, is_colab, is_kaggle, ops


class MyLoadStreams(LoadStreams):
    def __init__(
        self, sources="file.streams", vid_stride=1, buffer=False, grab_interval: int = 1
    ):  # 拷贝父类的代码并修改
        """Initialize instance variables and check for consistent input stream shapes.

        Args:
            sources (str, optional): _description_. Defaults to "file.streams".
            vid_stride (int, optional): _description_. Defaults to 1.
            buffer (bool, optional): _description_. Defaults to False.
            grab_interval (int, optional): 抓取图片时间间隔（秒）. Defaults to 1.

        Raises:
            NotImplementedError: _description_
            ConnectionError: _description_

        Returns:
            _type_: _description_
        """
        torch.backends.cudnn.benchmark = True  # faster for fixed-size inference
        self.source_type = SourceTypes(stream=True)  # 添加source_type属性
        self.grab_interval = grab_interval
        self.buffer = buffer  # buffer input streams
        self.running = True  # running flag for Thread
        self.mode = "stream"
        self.vid_stride = vid_stride  # video frame-rate stride

        sources = (
            Path(sources).read_text().rsplit() if os.path.isfile(sources) else [sources]
        )
        n = len(sources)
        self.bs = n
        self.fps = [0] * n  # frames per second
        self.frames = [0] * n
        self.threads = [None] * n
        self.caps = [None] * n  # video capture objects
        self.imgs = [[] for _ in range(n)]  # images
        self.shape = [[] for _ in range(n)]  # image shapes
        self.sources = [
            ops.clean_str(x) for x in sources
        ]  # clean source names for later
        self.info = [""] * n
        self.is_video = [True] * n
        cap_opened_count = n  # 视频流打开的个数
        for i, s in enumerate(sources):  # index, source
            # Start thread to read frames from video stream
            st = f"{i + 1}/{n}: {s}... "
            s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
            if s == 0 and (is_colab() or is_kaggle()):
                raise NotImplementedError(
                    "'source=0' webcam not supported in Colab and Kaggle notebooks. "
                    "Try running 'source=0' in a local environment."
                )
            self.caps[i] = cv2.VideoCapture(s)  # store video capture object
            cap_opened = self.caps[i].isOpened()  # 添加视频流异常检测
            if cap_opened:
                w = int(self.caps[i].get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(self.caps[i].get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = self.caps[i].get(cv2.CAP_PROP_FPS)  # warning: may return 0 or nan
                self.frames[i] = max(
                    int(self.caps[i].get(cv2.CAP_PROP_FRAME_COUNT)), 0
                ) or float(
                    "inf"
                )  # infinite stream fallback
                self.fps[i] = (
                    max((fps if math.isfinite(fps) else 0) % 100, 0) or 30
                )  # 30 FPS fallback

                success, im = self.caps[i].read()  # guarantee first frame
                if not success or im is None:
                    cap_opened = False
                    logger.error(ConnectionError(f"{st}Failed to read images from {s}"))
                if cap_opened:  # 确保视频流可读才启动线程
                    self.imgs[i].append(im)
                    self.shape[i] = im.shape
                    self.threads[i] = Thread(
                        target=self.update, args=([i, self.caps[i], s]), daemon=True
                    )
                    logger.info(
                        f"{st}Success ✅ ({self.frames[i]} frames of shape {w}x{h} at {self.fps[i]:.2f} FPS)"
                    )
                    self.threads[i].start()
            if not cap_opened:
                cap_opened_count -= 1
                logger.error(
                    ConnectionError(f"{st}Failed to open {s}, so ignore it...")
                )
                # 需要设置默认值，否则检测时候会报错
                self.fps[i] = 30
                self.shape[i] = (24, 24, 3)
                # 创建一个并不启动的线程
                self.threads[i] = Thread(
                    target=self.update, args=([i, self.caps[i], s]), daemon=True
                )
        # 检测视频流是否全部失败
        if cap_opened_count == 0:
            raise ConnectionError(
                "ERROR ❌ No video streams could be opened. Please check your source names."
            )
        LOGGER.info("")  # newline

    def update(self, i, cap, stream):
        """Read stream `i` frames in daemon thread."""
        n, f = 0, self.frames[i]  # frame number, frame array
        retry_num = 0
        buffer_count = 30 if self.buffer else 1  # 缓存的图片数量
        while self.running and n < (f - 1):
            retry_num = 0
            while not cap.isOpened() and retry_num <= 30:
                # 进入快速重连，超过重连次数限制后，进入下面的长等待
                logger.warning(
                    f"WARNING ⚠️ Video stream {stream} unresponsive, re-opening..."
                )
                cap.open(stream)  # re-open stream if signal was lost
                retry_num += 1
                time.sleep(1)
            if retry_num > 30:
                logger.error(
                    f"ERROR ❌ Video stream {stream} failed to open, thread frozen"
                )
                while True:
                    # 进入长等待，每隔一分钟尝试重连，不放弃该视频流
                    time.sleep(60)
                    cap.open(stream)
                    if cap.isOpened():
                        logger.info(f"Video stream {stream} recovery link, thread continues to update")
                        break
            # fps在初始化时候已经设置好了，这里不需要再设置。 Comments by creativor. 2024-08-14
            # fps = cap.get(cv2.CAP_PROP_FPS) # stream fps
            # if not math.isfinite(fps) or fps > 144:
            #     fps = 30 # If the obtained fps is infinite, the default is 30
            if len(self.imgs[i]) < buffer_count:  # keep a <=30-image buffer
                n += 1
                cap.grab()  # .read() = .grab() followed by .retrieve()
                if n % (self.fps[i] * self.grab_interval) == 0:
                    success, im = cap.retrieve()
                    if not success:
                        im = np.zeros(self.shape[i], dtype=np.uint8)
                        logger.warning(
                            f"WARNING ⚠️ Video stream {stream}  unresponsive, please check your IP camera connection."
                        )
                        cap.open(stream)  # re-open stream if signal was lost
                    # 过滤模糊图片
                    if fuzzy_filter(im):
                        n -= 1
                        continue
                    if self.buffer:
                        self.imgs[i].append(im)
                    else:
                        self.imgs[i] = [im]
            else:  # 图片没有被消费，则等待
                time.sleep(0.01)  # wait until the buffer is empty

    def __next__(self):
        """Returns source paths, transformed and original images for processing."""
        self.count += 1

        image = None
        sources_result = (
            []
        )  # 构造resource结果列表，只有有图片的线程才会在结果列表中，避免对self.sources的修改

        images_result = []
        bss_result = []

        def append_result(resource, image):
            """添加图片到结果列表中"""
            sources_result.append(resource)
            images_result.append(image)
            bss_result.append("")

        live_thread_count = self.bs
        for i, x in enumerate(self.imgs):
            if not self.threads[i].is_alive():  # 忽略没有启动的线程
                live_thread_count -= 1
                continue  # next i
            if not x:  # 忽略没有图片的线程
                continue  # next i
            # # Wait until a frame is available in each buffer
            # while not x:
            #     # 改为允许部分线程不启动
            #     # if not self.threads[i].is_alive() or cv2.waitKey(1) == ord("q"):  # q to quit
            #     #     self.close()
            #     #     raise StopIteration
            #     if not self.threads[i].is_alive():
            #         break
            #     time.sleep(1 / min(self.fps))
            #     x = self.imgs[i]
            #     if not x:
            #         # LOGGER.warning(f"WARNING ⚠️ Waiting for stream {i}")
            #         pass

            # Get and remove the first frame from imgs buffer
            if self.buffer:
                # images.append(x.pop(0))
                image = x.pop(0)

            # Get the last frame, and clear the rest from the imgs buffer
            else:
                # images.append(
                #     x.pop(-1) if x else np.zeros(self.shape[i], dtype=np.uint8)
                # )
                image = x.pop(-1)
                x.clear()
            # 构造结果
            append_result(self.sources[i], image)

        # 如果所有线程都结束，则结束检测
        if live_thread_count == 0:
            logger.error(f"ERROR ❌ No live thread, task exit")
            self.close()
            # 这样调用的迭代器才会抛出异常
            raise ConnectionError("ERROR ❌ No live thread, task exit") 

        # 如果没有结果，则构造一个空的
        if not images_result:
            append_result("", np.zeros((24, 24, 3), dtype=np.uint8))
        # return self.sources, images, [""] * self.bs
        return sources_result, images_result, bss_result


def make_streams_temp_file(stream_urls: List[str]) -> str:
    """创建视频流列表临时文件

    Args:
        stream_urls (List[str]): 视频流url列表

    Returns:
        str: 临时文件的名称
    """
    # 写入多行数据
    lines = [(stream_url + "\n").encode("utf-8") for stream_url in stream_urls]
    file = tempfile.NamedTemporaryFile(mode="+bw", suffix=".streams", delete=False)
    file.writelines(lines)
    file.flush()
    return file.name


def build_rtsp_sn_map(streams_conf: List[object]) -> Dict[str, str]:
    """建立 RTSP_URL 与 CAMERA_SN 的映射关系字典

    Args:
        streams_conf (List[object]): 视频流配置列表

    Returns:CAMERA_SN
        Dict[str, str]: RTSP_URL->CAMERA_SN的映射列表
    """
    return {
        ops.clean_str(stream["RTSP_URL"]): stream["CAMERA_SN"]
        for stream in streams_conf
    }


def fuzzy_filter(frame, threshold=100.0) -> bool:
    """判断图片是否模糊，得分小于阀值被判断为模糊图片.

    Args:
        frame (cv2.typing.MatLike): 源图片
        threshold (float): 得分阀值. Defaults to 100.0.

    Returns:
        bool: 图片是否模糊
    """
    # 获取下半部分的图片，因为大多数情况都是下半部分图片模糊
    height, width = frame.shape[:2]
    y_start = height // 2  # 使用整除得到中间行
    y_end = height
    frame_copy = frame[y_start:y_end, :]
    img2gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(img2gray, cv2.CV_64F).var()
    return score < threshold
