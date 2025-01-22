# 这里用到了ffmpegd的命令 ，需要安装ffmpeg的命令
import streamlit as st
import cv2
from PIL import Image, ImageDraw
import base64
import numpy as np
from streamlit_image_coordinates import streamlit_image_coordinates
from typing import List, Tuple
import subprocess
import json


points = []


def draw_points_and_shapes(
    image: Image.Image, points: List[Tuple[int, int]]
) -> Image.Image:
    draw = ImageDraw.Draw(image)
    num_points = len(points)
    for point in points:
        draw.ellipse(
            (point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5), fill="red"
        )
    if num_points >= 2:
        draw.line(points + [points[0]], fill="blue", width=2)
    if num_points >= 3:
        draw.polygon(points, outline="green", fill=None)

    return image


def normalize_points(points: List[Tuple[int, int]], width: int, height: int) -> str:
    normalized_points = []
    for point in points:
        normalized_x = round(point[0] / width, 5)
        normalized_y = round(point[1] / height, 5)
        normalized_points.append((normalized_x, normalized_y))
    return str(normalized_points)


def get_video_info(rtsp_link):
    # 构建 ffprobe 命令
    cmd = [
        "ffprobe",
        "-i",
        rtsp_link,
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
    ]
    # 运行FFmpeg命令
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        # 等待一定时间
        output, _ = process.communicate(timeout=2)
    except subprocess.TimeoutExpired:
        # 如果超时，则终止进程
        process.kill()
        output, _ = process.communicate()
    
    # 直接返回原始的输出
    return output

    # # 解析 JSON 输出
    # info = {}
    # try:
    #     info = json.loads(output)
    # except json.JSONDecodeError as e:
    #     st.error("Error decoding JSON output:" + str(e))
    #     return None

    # return info


def pagertsp():
    st.header("📹  RTSP测试和检测框获取", divider=True)
    input_size = st.number_input(
        "请输入图片尺寸（宽度），尺寸尽可能需要符合线程任务的图片大小",
        min_value=1,
        value=1280,
    )
    rtsp_link = st.text_input(
        "请输入RTSP链接，用于测试rtsp链接的链接情况，按enter测试",
        "rtsp://admin:12345@192.168.0.120:554/live/main",
    )
    frame = None

    if rtsp_link:
        cap = cv2.VideoCapture(rtsp_link)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                st.success("rtsp连接成功")
                if st.button("查看rtsp属性，需要保证安装系统ffmpeg的命令库"):
                    video_info = get_video_info(rtsp_link)
                    st.info(video_info,icon="ℹ️")
                    # st.write(
                    #     f"视频编码: {video_info.get('codec', '未知')},  分辨率: {video_info.get('resolution', '未知')},  帧率: {video_info.get('frame_rate', '未知')} fps, 色彩: {video_info.get('color_format', '未知')},"
                    # )
            else:
                st.warning("未能获取到图片，请检查RTSP链接是否正确或者是否有权限访问。")
        else:
            st.warning("无法打开RTSP链接，请检查链接是否正确。")

    if input_size and frame is not None:
        height, width, _ = frame.shape
        ratio = input_size / width        
        #image = Image.fromarray(resized_frame)

        st.markdown(
            "**:writing_hand:  如有画面圈定需要，则在这张图片上点击至少四个点，来组成一个多边形，获取最下面的坐标，没有需要则不需要管下面的任何操作**"
        )

        if frame is not None:
            resized_frame = cv2.resize(frame, (input_size, int(height * ratio)))
            #将RGB图像转换为BGR图像
            bgr_frame = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR)
            image = Image.fromarray(bgr_frame)
            value = streamlit_image_coordinates(
                image,
                key="pil",
            )
            if value:
                points.append((value["x"], value["y"]))
                image = draw_points_and_shapes(image, points)
            st.markdown(
                "**:point_down:  下面这张图片上显示已点击的坐标，上面点乱了就刷新页面重新点**"
            )
            st.image(image, channels="BGR", caption=f"宽度为{input_size}")
            st.write(points)

        else:
            st.warning("请先输入RTSP链接并确保能获取到图片。")

        # 显示归一化的坐标点
        if points:
            normalized_points_str = normalize_points(points, width * ratio, height * ratio)
            st.markdown("**:v:  复制这个坐标粘贴到线程编辑里**")
            st.write(f"归一化的坐标点: {normalized_points_str}")

    if st.button("清空points"):
        # 当按钮被点击时，清空points列表
        points.clear()
        st.write("坐标点已被清空。")
