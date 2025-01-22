# -*- encoding: utf-8 -*-
"""
st.data_editor测试脚本.
"""
import streamlit as st
import pandas as pd
from mods.constants import *
from datetime import datetime

# 初始化数据
streams = [
    {
        "RTSP_URL": "rtsp://admin:weile123456@192.168.0.185/Streaming/Channels/101",
        "CAMERA_SN": "310",
    },
    {"RTSP_URL": "rtsp://admin:12345@192.168.0.120:554/live/main", "CAMERA_SN": "311"},
]

column_config = {
    "RTSP_URL": st.column_config.TextColumn(
        label=Tokens.RTSP_URL, required=True, validate="^rtsp:\/\/.+$"
    ),
    "CAMERA_SN": st.column_config.TextColumn(
        label=Tokens.CAMERA_SN, required=True, validate="^\S+$"
    ),
}

# streams=[
#     "rtsp://admin:weile123456@192.168.0.106/Streaming/Channels/101",
#     "rtsp://admin:weile123456@192.168.0.145/Streaming/Channels/101",
#     "rtsp://admin:weile123456@192.168.0.182/Streaming/Channels/101",
#     "rtsp://admin:wl123456@192.168.0.183/Streaming/Channels/101",
#     "rtsp://admin:weile123456@192.168.0.185/Streaming/Channels/101",
#     "rtsp://admin:weile123456@192.168.0.220/Streaming/Channels/101",
#     "rtsp://admin:12345@192.168.0.120:554/live/main",
#     "rtsp://admin:weile123456@192.168.0.103/cam/realmonitor?channel=1&subtype=0",
# ]

# data = {
#     "Name": ["Alice", "Bob", "Charlie"],
#     "Age": [25, 30, 35],
#     "City": ["New York", "Los Angeles", "Chicago"],
# }

# 创建 DataFrame
df = pd.DataFrame(streams)

edited_df = st.data_editor(data=df, column_config=column_config, num_rows="dynamic")

# 获取当前时间
current_time = datetime.now()
st.write(f"编辑结果：{current_time}")
st.write(edited_df.to_dict(orient="records"))
