from enum import Enum
import json
import streamlit as st
import config
import time
from mods.constants import *
import threads_config
import pandas as pd
import face_detection
from typing import List, Dict
from st_aggrid import (
    GridOptionsBuilder,
    AgGrid,
    GridUpdateMode,
    JsCode,
    ColumnsAutoSizeMode,
    AgGridReturn,
)

class Event_Type(Enum):
    """用户操作-事件-类型"""

    HOME = 1
    """主页,默认值"""
    NEW = 2
    """编辑新线程"""
    EDIT = 3
    """编辑原有线程"""
    START = 4
    """启动人脸线程"""
    STOP = 5
    """停止人脸线程"""


def facethread():
    st.header("😀人脸检测线程", divider=True)

    (
        col1,
        col2,
        col3,
        col4,
        col5,
        col6,
        col7,
        col8,
        col9,
        col10
    ) = st.columns(10)

    col1.button("新建线程", on_click=lambda: update_event_type(Event_Type.NEW))
    start_button = col2.button("启动线程")
    stop_button = col3.button("停止线程")


    load_config = threads_config.face_load_config()
    config_values = load_config.values()
    df = pd.DataFrame(
        config_values,
        columns=[
            "CALLBACK_URL",
            "SLEEP_TIME",
        ]
    )
    df.insert(loc=0, column="THREAD_NAME", value=load_config.keys())
    df.insert(
        loc=1,
        column="STREAMS",
        value=[json.dumps(x.get("STREAMS")) for x in config_values],
    )
    df.insert(
        loc=4,
        column="RUNNING_STATUS",
        value=[
            face_detection.is_thread_running(x["THREAD_ID"]) for x in config_values
        ],
    )

    checkbox_selection_code = JsCode(
        """
        function(params) {
            var running_status=params.data.RUNNING_STATUS;
            return !running_status;
        }
        """
    )

    status_cell_render = JsCode(
        """
        class StatusCellRenderer {
            init(params) {
                this.eGui = document.createElement('span');
                this.eGui.innerHTML = params.value?"🟢 正在运行":"🔴 停止";
            }
            getGui() {
                return this.eGui;
            }       
        }
        """
    )

    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_selection(selection_mode="single", use_checkbox=False)  # 允许用户多选
    gb.configure_pagination(paginationAutoPageSize=False)
    gb.configure_column(
        "THREAD_NAME",
        Tokens.THREAD_NAME,
        checkboxSelection= True,
        filter="agTextColumnFilter",
    )
    gb.configure_column(
        "STREAMS", Tokens.STREAMS, tooltipField="STREAMS", filter="agTextColumnFilter"
    )
    gb.configure_column(
        "CALLBACK_URL",
        Tokens.CALLBACK_URL,
        tooltipField="CALLBACK_URL",
        filter="agTextColumnFilter",
    )
    gb.configure_column("SLEEP_TIME", Tokens.SLEEP_TIME)
    gb.configure_column(
        "RUNNING_STATUS", Tokens.RUNNING_STATUS, cellRenderer=status_cell_render
    )
    gb.configure_grid_options(
        tooltipShowDelay=500, 
        # domLayout="autoHeight", 
        localeText=AG_GRID_LOCALE_CN
    )
    gridOptions = gb.build()
    grid_response: AgGridReturn = AgGrid(
        df,
        gridOptions=gridOptions,
        allow_unsafe_jscode=True,
        data_return_mode="AS_INPUT",
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        fit_columns_on_grid_load=True,
        theme="streamlit",
        width="100%",
        update_on=[],  # 在此注册要后台处理的事件列表，参考https://www.ag-grid.com/javascript-data-grid/grid-events/
    )

    handle_grid_events(grid_response,start_button,stop_button)
    
    
    # 当前事件类型
    event_type: Event_Type = get_event_type()

    if event_type == Event_Type.HOME:
        pass
    elif event_type == Event_Type.NEW:
        if grid_response.selected_data is None:
            # 新建线程
            st.subheader("请填写新线程的配置参数：")
            THREAD_ID = threads_config.generate_thread_id()
            col1,col2 = st.columns(2)
            THREAD_NAME = col1.text_input(Tokens.THREAD_NAME)
            SLEEP_TIME = col1.number_input(Tokens.SLEEP_TIME, value=10)
            EXPRESSION_CONF = st.slider(
                    "表情检测阈值",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.6,
                )
            DET_SCORE = st.slider(
                    "人脸检测置信度",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                )
            CALLBACK_URL = col1.text_input(
                Tokens.CALLBACK_URL, "http://192.168.0.85:5006/aiFaceLibrary/receive"
            )
            STREAMS = get_edited_streams_data([],None)
            if st.button("新建并提交线程配置"):
                if THREAD_NAME.strip() == "":
                    st.error("线程名称不能为空。")
                else:
                    if threads_config.isSameNameExists(THREAD_NAME, THREAD_ID, face=True):
                        st.error("线程名称已存在，请修改。")
                    else:
                        update_event_type(Event_Type.HOME)
                        update_thread_config(
                            load_config,
                            THREAD_ID,
                            THREAD_NAME,
                            STREAMS,
                            CALLBACK_URL,
                            SLEEP_TIME,
                            DET_SCORE,
                            EXPRESSION_CONF,
                            is_new_thread=True,
                        )
                        st.success("成功新建线程")
                        time.sleep(1)
                        st.rerun()
            st.button("取消新建", on_click=lambda: update_event_type(Event_Type.HOME))
    elif event_type == Event_Type.START:
        if grid_response.selected_data is None:
            # 没有被选中的线程
            return
        thread_name = grid_response.selected_data["THREAD_NAME"].values[0]
        status_placeholder = st.empty()
        status_placeholder.text(f"线程 {thread_name} 启动中")
        response = face_detection.start_thread(thread_name)
        if response.status_code == 200:
            st.success(f"线程 {thread_name} 已成功启动", icon="🎉")
            # config[thread_name]["status"] = "on"
            # threads_config.save_config(config)
            time.sleep(2)
            st.rerun()
        else:
            st.error(f"启动线程 {thread_name} 失败: {response.text}")
    elif event_type == Event_Type.STOP:
        if grid_response.selected_data is None:
            # 没有被选中的线程
            return
        thread_name = grid_response.selected_data["THREAD_NAME"].values[0]
        response = face_detection.stop_thread(thread_name)
        if response.status_code == 200:
            st.success(f"线程 {thread_name} 已成功停止")
            # config[thread_name]["status"] = "off"
            # threads_config.save_config(config)
            time.sleep(2)
            st.rerun()
        else:
            st.error(f"停止线程 {thread_name} 失败: {response.text}")

    elif event_type == Event_Type.EDIT:
        if grid_response.selected_data is None:
            # 没有被选中的线程
            return
        selected_thread = grid_response.selected_data["THREAD_NAME"].values[0]
        selected_thread_config = load_config[selected_thread]
        thread_id = selected_thread_config["THREAD_ID"]
        if face_detection.is_thread_running(thread_id):
            st.error("线程正在运行中，无法修改参数。")
        else:
            col_edit_1, col_edit_2 = st.columns(2)
            with col_edit_1:
                st.subheader(f"正在编辑线程：{selected_thread}")

            THREAD_NAME = st.text_input("当前线程名称", selected_thread)
            THREAD_ID = selected_thread_config["THREAD_ID"]
            SLEEP_TIME = st.number_input(
                Tokens.SLEEP_TIME, value=selected_thread_config["SLEEP_TIME"]
            )
            EXPRESSION_CONF = st.slider(
                    "表情检测阈值",
                    min_value=0.0,
                    max_value=1.0,
                    value=selected_thread_config["EXPRESSION_CONF"] if "EXPRESSION_CONF" in selected_thread_config else 0.6,
                )
            DET_SCORE = st.slider(
                    "人脸检测置信度",
                    min_value=0.0,
                    max_value=1.0,
                    value=selected_thread_config["DET_SCORE"] if "DET_SCORE" in selected_thread_config else 0.7,
                )
            CALLBACK_URL = st.text_input(
                Tokens.CALLBACK_URL, selected_thread_config["CALLBACK_URL"]
            )
            STREAMS = get_edited_streams_data(
                    selected_thread_config["STREAMS"],
                    selected_thread_config.get("THREAD_ID")
                )
            

            if st.button("保存线程修改"):
                if THREAD_NAME.strip() == "":
                    st.error("线程名称不能为空。")
                else:
                    if threads_config.isSameNameExists(THREAD_NAME, THREAD_ID, face=True):
                        st.error("线程名称已存在，请修改。")
                    else:
                        # 保存
                        update_thread_config(
                            load_config,
                            THREAD_ID,
                            THREAD_NAME,
                            STREAMS,
                            CALLBACK_URL,
                            SLEEP_TIME,
                            DET_SCORE,
                            EXPRESSION_CONF,
                            is_new_thread=False,
                        )
                        update_event_type(Event_Type.HOME)
                        st.success("成功修改线程")
                        st.session_state.pop(f"temp_streams_{THREAD_ID}", None)  # 清除对应的临时数据
                        time.sleep(1)
                        st.rerun()
            # Delete thread
            if st.button("删除当前线程", type="primary"):
                if face_detection.is_thread_running(thread_id):
                    st.error("线程正在运行中，无法删除。")
                else:
                    del load_config[selected_thread]
                    threads_config.face_save_config(load_config)
                    st.success("线程已删除！稍等刷新")
                    time.sleep(2)
                    st.rerun()


def update_thread_config(
    config,
    thread_id,
    thread_name,
    streams,
    callback_url,
    sleep_time,
    det_score,
    expression_conf,
    is_new_thread=False,
):
    if is_new_thread and thread_name in config:
        raise Exception("线程名称已存在...")
    if not is_new_thread:
        # 编辑线程，通过thread_id找到原线程配置项，然后删除它
        # Remove the old thread if the name has changed
        old_thread_name = None
        for name, thread in config.items():
            if thread.get("THREAD_ID") == thread_id:
                old_thread_name = name
                break
        if old_thread_name:
            del config[old_thread_name]
    config[thread_name] = {
        "THREAD_ID": thread_id,
        "CALLBACK_URL": callback_url,
        "SLEEP_TIME": sleep_time,
        "STREAMS": streams,
        "DET_SCORE": det_score,
        "EXPRESSION_CONF": expression_conf,
        "running_status": "off",
    }
    threads_config.face_save_config(config)



def update_event_type(event_type: Event_Type):
    """在session_state中更新事件类型"""
    st.session_state.page_face_thread_event_type = event_type

def get_event_type() -> Event_Type:
    """在session_state中获取事件类型"""
    event_type = st.session_state.get("page_face_thread_event_type")
    return event_type if event_type is not None else Event_Type.HOME

def handle_grid_events(grid_response: AgGridReturn,start_button, stop_button):
    """处理表格事件"""
    # 只要表格行被选中，都会触发"selectionChanged"事件
    if grid_response.event_data is not None:
        if (
            "selectionChanged" == grid_response.event_data["type"]
            and grid_response.selected_data is not None
        ):
            if start_button:
                update_event_type(Event_Type.START)
            elif stop_button:
                update_event_type(Event_Type.STOP)
            else:
                """row被选中,且是编辑模式"""
                update_event_type(Event_Type.EDIT)


def get_edited_streams_data(streams: List[Dict[str, str]], thread_id: str) -> List[Dict[str, str]]:
    """
    输入视频流列表,返回经界面编辑后的视频流列表。

    Args:
        streams (List[Dict[str, str]]): 视频流列表，每个元素是一个视频流配置项。

    Returns:
        List[Dict[str, str]]: 编辑后的视频流列表，每个元素是一个视频流配置项。

    """
    if thread_id is not None:
        # 初始化临时存储视频流数据
        temp_key = f"temp_streams_{thread_id}"
        if temp_key not in st.session_state:
            st.session_state[temp_key] = streams.copy()

        # 添加 JSON 输入框
        st.subheader("添加 JSON 数据到视频流列表")
        json_input = st.text_area("输入 JSON 数据：", height=200, placeholder='例如：[{"CAMERA_SN": "3", "RTSP_URL": "rtsp://..."}, ...]')

        if st.button("添加 JSON 数据"):
            try:
                new_streams = json.loads(json_input)
                if isinstance(new_streams, list) and all(isinstance(item, dict) for item in new_streams):
                    st.session_state[temp_key].extend(new_streams)
                    st.success("JSON 数据已成功添加到视频流列表，点击保存线程修改按钮保存所有更改。")
                else:
                    st.error("输入的 JSON 格式不正确，请输入列表形式的字典。")
            except json.JSONDecodeError:
                st.error("解析 JSON 数据时出错，请检查格式是否正确。")

        # 配置编辑表格
        column_config = {
            "RTSP_URL": st.column_config.TextColumn(
                label=Tokens.RTSP_URL,
                width="large",
                required=True,
                default="rtsp://",
                validate="^rtsp:\/\/.+$",
            ),
            "CAMERA_SN": st.column_config.TextColumn(
                label=Tokens.CAMERA_SN, required=True, validate="^\S+$"
            ),
        }
        # 创建 DataFrame
        df = pd.DataFrame(data=st.session_state[temp_key], columns=["RTSP_URL", "CAMERA_SN"])
        # 展现编辑表格并返回编辑后的结果
        st.subheader("视频流列表")
        edited_df = st.data_editor(data=df, column_config=column_config, num_rows="dynamic")
        st.caption("按:red[delete]键可以删除被选中的行.")

        # 更新临时数据
        st.session_state[temp_key] = edited_df.to_dict(orient="records")

        
        # 返回最终数据，但不保存
        return st.session_state[temp_key]
    else:
        column_config = {
            "RTSP_URL": st.column_config.TextColumn(
                label=Tokens.RTSP_URL,
                width="large",
                required=True,
                default="rtsp://",
                validate="^rtsp:\/\/.+$",
            ),
            "CAMERA_SN": st.column_config.TextColumn(
                label=Tokens.CAMERA_SN, required=True, validate="^\S+$"
            ),
        }
        # 创建 DataFrame
        df = pd.DataFrame(data=streams, columns=["RTSP_URL", "CAMERA_SN"])
        # 展现编辑表格并返回编辑后的结果
        st.subheader("视频流列表")
        edited_df = st.data_editor(data=df, column_config=column_config, num_rows="dynamic")
        st.caption("按:red[delete]键可以删除被选中的行.")
        # 封装返回结果
        return edited_df.to_dict(orient="records")