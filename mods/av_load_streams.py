# -*- encoding: utf-8 -*-
"""
PyAv版本的 yolo  LoadStreams. 

@Time    :   2024/09/14 09:26:06
@Author  :   creativor 
@Version :   1.0
"""
import time
from ultralytics.data.loaders import LoadStreams, SourceTypes
import math
import os
from pathlib import Path
from loguru import logger
from threading import Thread
import numpy as np
import cv2

import av
from av.container.input import InputContainer
import torch
from ultralytics.utils import LOGGER, is_colab, is_kaggle, ops


# 设置日志级别为 error
av.logging.set_level(av.logging.ERROR)


def _get_frame_count(video_stream):
    # 检查时长和帧率是否可用
    if video_stream.duration is not None and video_stream.average_rate is not None:
        # 如果 duration 和帧率可用，计算总帧数
        frame_count = int(video_stream.duration * video_stream.average_rate)
        return frame_count
    else:
        # 如果无法获取时长，
        return float("inf")


def _reconnect_stream(old_cap: InputContainer, url: str, max_wait=3600):
    """关闭旧的视频流并重连视频流

    Args:
        old_cap (InputContainer):  旧的输入流
        url (string):  rtsp视频流地址
        max_wait (int, optional): 最大重连间隔时间. Defaults to 3600.

    Raises:
        ConnectionError: 如何超过重试次数仍然失败

    Returns:
        InputContainer: 新的流对象
    """
    # 先尝试关掉旧的输入流
    try:
        old_cap.close()
    except BaseException as e:
        logger.exception(f"关闭旧的输入流[{url}]发生异常:", e)
    attempt = 1
    while True:
        logger.info(f"Try to reconnect [{url}], attempt {attempt} ...")
        try:
            new_cap = av.open(url,timeout=5)
            logger.info(f"Reconnect attempt {attempt} succeeded.")
            return new_cap
        except av.AVError as e:
            logger.error(f"Reconnect attempt {attempt} failed: {e}")
            if (2 ** attempt) >= max_wait:
                time.sleep(max_wait)
            else:
                time.sleep(2 ** attempt)
            attempt += 1
    # raise ConnectionError(f"Unable to reconnect [{url}] after multiple attempts")


class AvLoadStreams(LoadStreams):
    """PyAv版本的LoadStreams"""

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
            try:
                cap = av.open(s,timeout=10)
                self.caps[i] = cap  # store video capture object
                self.threads[i] = Thread(
                    target=self.update, args=([i, self.caps[i], s]), daemon=True
                )
                self.threads[i].start()
            except av.AVError as e:
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

    def update(self, i, cap: InputContainer, stream):
        video_stream = next((s for s in cap.streams if s.type == "video"), None)
        
        # Signal that we only want to look at keyframes.
        video_stream.codec_context.skip_frame = "NONKEY"
        
        w = video_stream.codec_context.width
        h = video_stream.codec_context.height
        fps = int(
            video_stream.average_rate if video_stream.average_rate is not None else 0
        )
        self.frames[i] = _get_frame_count(video_stream)
        self.fps[i] = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30
        st = f"{i + 1}/{self.bs}: {stream}... "
        logger.info(
            f"{st}Success ✅ ({self.frames[i]} frames of shape {w}x{h} at {self.fps[i]:.2f} FPS)"
        )

        n, f = 0, self.frames[i]
        buffer_count = 30 if self.buffer else 1
        last_pts = -1  # Track the PTS of the last processed frame
        
        while self.running and n < (f - 1):
            try:
                for packet in cap.demux(video=0):
                    try:
                        for frame in packet.decode():
                            if frame is None or not frame.key_frame or frame.pts is None or frame.pts <= last_pts:
                                continue
                            
                            last_pts = frame.pts
                            n += 1
                            if n % self.grab_interval == 0:
                                im = frame.to_ndarray(format="bgr24")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
                                if im is not None and not fuzzy_filter(im):
                                    if len(self.imgs[i]) < buffer_count:
                                        if self.buffer:
                                            self.imgs[i].append(im)
                                        else:
                                            self.imgs[i] = [im]
                                    else:
                                        time.sleep(0.3)
                            
                            del frame
                    except av.error.InvalidDataError as e:
                        logger.error(f"packet.decode() error:{e}")
                        continue
                    except av.AVError as e:
                        logger.error(f"packet.decode() error:{e}")
                        cap = _reconnect_stream(cap, stream)
                        self.caps[i] = cap
                        last_pts = -1
                        break
                    finally:
                        del packet
            except av.AVError as e:
                logger.error(f"cap.demux() error: {e}")
                cap = _reconnect_stream(cap, stream)
                self.caps[i] = cap
                last_pts = -1
            
            time.sleep(self.grab_interval)

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
            logger.error("ERROR ❌ No live thread, task exit")
            self.close()
            # 这样调用的迭代器才会抛出异常
            raise ConnectionError("ERROR ❌ No live thread, task exit")

        # 如果没有结果，则构造一个空的
        if not images_result:
            return [], [], []
        # return self.sources, images, [""] * self.bs
        return sources_result, images_result, bss_result

    def close(self):
        """Close stream loader and release resources."""
        self.running = False  # stop flag for Thread
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=5)  # Add timeout
        for cap in self.caps:  # Iterate through the stored VideoCapture objects
            try:
                cap.close()  # release video capture
            except Exception as e:
                LOGGER.warning(f"WARNING ⚠️ Could not release VideoCapture object: {e}")

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
