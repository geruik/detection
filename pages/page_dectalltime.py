import cv2
from ultralytics import YOLO
import streamlit as st
import numpy as np
import json
import time


def process_frame(cap, model, conf_threshold, selected_classes, time_interval, st_frame):
    if cap.isOpened():
        if not is_paused:
            success, frame = cap.read()
            if success:
                results = model(frame, conf=conf_threshold, classes=selected_classes)
                annotated_frame = results[0].plot()
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                st_frame.image(annotated_frame)
                time.sleep(time_interval)
            else:
                st.write("Video stream ended.")
        else:
            time.sleep(0.1) 

def yolo_callback():
    global is_started
    global is_paused
    is_started = False
    is_paused = False
    with open('models/models.json', 'r', encoding='utf-8') as f:
        MODELS_INFO = json.load(f)
    
    st.header(':leafy_green:  实时检测任务', divider=True)
    
    selected_model = st.selectbox('选择模型', list(MODELS_INFO.keys()), 0)
    model_path = MODELS_INFO[selected_model]['path']
    conf_threshold = st.slider('置信度', 0.0, 1.0, 0.25, 0.01)
    time_interval = st.slider('检测时间间隔(秒)', 0, 5, 0, 1)
    selected_classes = st.multiselect('选择类别', [0, 1], default=[0, 1])
    model = YOLO(model_path)
    
    video_source = st.radio("选择视频源", ("RTSP", "本地视频"))
    cap = None
    if video_source == "RTSP":
        rtsp_url = st.text_input("输入RTSP地址", "rtsp://admin:12345@192.168.0.120:554/live/main")
        cap = cv2.VideoCapture(rtsp_url)
    else:
        video_file = st.file_uploader("上传本地视频", type=['mp4', 'avi', 'mkv'])
        if video_file is not None:
            file_bytes = np.asarray(bytearray(video_file.read()), dtype=np.uint8)
            cap = cv2.VideoCapture(file_bytes)
    col1, col2 = st.columns(2) 
    start_button = col1.button("启动检测")
    stop_button = col2.button("暂停检测")    
    st_frame = st.empty()


    if start_button:
        if not is_started:
            is_started = True
            is_paused = False
            st_frame = st.empty()
            while is_started and cap.isOpened():
                process_frame(cap, model, conf_threshold, selected_classes, time_interval, st_frame)
        else:
            st.warning('检测已经启动。')

    if stop_button:
        if is_started:
            is_paused = not is_paused
            if is_paused:
                st.info('检测已暂停。')
            else:
                st.info('检测已恢复。')
        else:
            st.warning('检测尚未启动。')

