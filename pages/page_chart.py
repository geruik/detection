import streamlit as st
import psutil
from PIL import Image
import time
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pynvml

DATABASE_FILE = "detection_data.db"

def read_data(thread_id):
    with sqlite3.connect(DATABASE_FILE) as conn:
        df = pd.read_sql_query(f"SELECT * FROM detection_data WHERE thread_id = '{thread_id}'", conn)
    # 将时间戳转换为datetime对象
    df['detection_time'] = pd.to_datetime(df['detection_time'], unit='ms')
    df.set_index('detection_time', inplace=True)  # 设置detection_time为索引
    return df


def get_gpu_memory():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    pynvml.nvmlShutdown()
    return round(info.used / info.total * 100, 2)

# 实时监控函数
def pagechart():
    st.header(":green_salad:  Dashboard", divider=True)
    
    with st.spinner('资源使用情况...'):
            cpu_usage = psutil.cpu_percent()
            mem_usage = psutil.virtual_memory().percent
            gpu_memory = get_gpu_memory()
            col1, col2, col3 = st.columns(3)
            col1.metric("CPU 占用率", f"{cpu_usage}%", delta=None, delta_color="normal")
            col2.metric("内存占用率", f"{mem_usage}%", delta=None, delta_color="normal")
            col3.metric("显存占用率", f"{gpu_memory}%", delta=None, delta_color="normal")
        
    st.write("  ")
    st.write("  ")
    st.write("  ")
    with sqlite3.connect(DATABASE_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT thread_id FROM detection_data")
    thread_ids = [row[0] for row in cursor.fetchall()]
    col4, col5= st.columns(2)
    thread_id = col5.selectbox("选择 Thread ID", thread_ids)
    df = read_data(thread_id)
    st.subheader("检测数量变化趋势")
    line_chart = st.line_chart(df['box_count'])  
    st.subheader("最大和最小置信度变化趋势")
    scatter_chart = st.scatter_chart(df[['max_confidence', 'min_confidence']])  
