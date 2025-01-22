# -*- encoding: utf-8 -*-
"""
工具类. 
包含系统共享的工具函数.

@Time    :   2024/11/04 11:07:06
@Author  :   creativor 
@Version :   1.0
"""


from numpy import ndarray
import av


def capture_stream_screen(stream_url: str) -> ndarray:
    """截取流媒体屏幕图片.

    Args:
        stream_url (str): 流媒体地址

    Returns:
        ndarray: 返回截取的图片数据
    Raises:
        av.AVError: 当视频流异常。
    """
    with av.open(stream_url,timeout=10) as cap:
        for packet in cap.demux(video=0):
            for frame in packet.decode():
                if frame is None:
                    continue
                if not frame.key_frame:
                    continue
                im_array = frame.to_ndarray(format="rgb24")
                return im_array
