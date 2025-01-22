"""
相似检测线程管理页面.

@Time    :   2024/10/16 15:26:06
@Author  :   creativor
@Version :   1.0
"""

from enum import Enum
import json
import streamlit as st
import config
import time
from mods.constants import *
import threads_config
import pandas as pd
import myclip.clip_detection as clip_detection
from typing import List, Dict
from st_aggrid import (
    GridOptionsBuilder,
    AgGrid,
    GridUpdateMode,
    JsCode,
    ColumnsAutoSizeMode,
    AgGridReturn,
)


CONFIGED_CLASSES = config.clip["classes"]
"""获取配置文件中的clip模型类别信息"""


class Event_Type(Enum):
    """用户操作-事件-类型"""

    HOME = 1
    """主页,默认值"""
    NEW = 2
    """编辑新线程"""
    VIEW = 3
    """查看线程"""
    EDIT = 4
    """编辑原有线程"""
    START = 5
    """启动线程"""
    STOP = 6
    """停止线程"""


def clip_thread():
    st.header("📏相似检测线程", divider=True)

    (col1, col2, col3, col4, col5, col6, col7, col8, col9, col10) = st.columns(10)

    col1.button("新建线程", on_click=lambda: update_event_type(Event_Type.NEW))
    col2.button("编辑线程", on_click=lambda: update_event_type(Event_Type.EDIT))
    col4.button("启动线程", on_click=lambda: update_event_type(Event_Type.START))
    col5.button("停止线程", on_click=lambda: update_event_type(Event_Type.STOP))

    load_config = threads_config.clip_load_config()
    config_values = load_config.values()

    df = pd.DataFrame(
        config_values,
        columns=[
            "CALLBACK_URL",
            "SLEEP_TIME",
        ],
    )
    df.insert(loc=0, column="THREAD_NAME", value=load_config.keys())
    # 字段名称映射关系（旧名称 -> 新名称）
    class_field_mapping = {
        "CLASS": "类别",
        "SIMILARITY_THRESHOLD": "相似度",
        "FACE_DETECTION": "人脸检测",
    }
    df.insert(
        loc=1,
        column="CLASSES",
        value=[
            ",".join([item["CLASS"] for item in x.get("CLASSES")])
            for x in config_values
        ],
    )
    df.insert(
        loc=2,
        column="STREAMS",
        value=[json.dumps(x.get("STREAMS")) for x in config_values],
    )
    df.insert(
        loc=5,
        column="RUNNING_STATUS",
        value=[clip_detection.is_thread_running(x["THREAD_ID"]) for x in config_values],
    )
    df.insert(
        loc=6,
        column="CLASSES_TOOLTIP",
        value=[
            json.dumps(
                [
                    {
                        new_key: d[old_key]
                        for old_key, new_key in class_field_mapping.items()
                    }
                    for d in x.get("CLASSES")
                ],
                ensure_ascii=False,
            )
            for x in config_values
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
        checkboxSelection=True,
        filter="agTextColumnFilter",
    )
    gb.configure_column(
        "CLASSES",
        "检测类别",
        tooltipField="CLASSES_TOOLTIP",
        filter="agTextColumnFilter",
    )
    gb.configure_column(
        "CLASSES_TOOLTIP",
        "检测类别",
        filter="agTextColumnFilter",
        hide=True,
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
        localeText=AG_GRID_LOCALE_CN,
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

    handle_grid_events(grid_response)

    # 当前事件类型
    event_type: Event_Type = get_event_type()

    if event_type == Event_Type.HOME:
        pass
    elif event_type == Event_Type.NEW:
        # 新建线程
        st.subheader("请填写新线程的配置参数：")
        THREAD_ID = threads_config.generate_thread_id()
        col1, col2 = st.columns(2)
        THREAD_NAME = col1.text_input(Tokens.THREAD_NAME)
        CLASSES = get_edited_classes_data([])
        SLEEP_TIME = col1.number_input(Tokens.SLEEP_TIME, value=10)
        CALLBACK_URL = col1.text_input(Tokens.CALLBACK_URL, "http://192.168.0.85:5006/znyj/violation/recognize")
        STREAMS = get_edited_streams_data([],None)
        if st.button("新建并提交线程配置"):
            if THREAD_NAME.strip() == "":
                st.error("线程名称不能为空。")
            else:
                if threads_config.isSameNameExists(THREAD_NAME, THREAD_ID, clip=True):
                    st.error("线程名称已存在，请修改。")
                else:
                    update_event_type(Event_Type.HOME)
                    update_thread_config(
                        load_config,
                        THREAD_ID,
                        THREAD_NAME,
                        CLASSES,
                        STREAMS,
                        CALLBACK_URL,
                        SLEEP_TIME,
                        is_new_thread=True,
                    )
                    st.success("成功新建线程")
                    time.sleep(1)
                    st.rerun()
        st.button("取消新建", on_click=lambda: update_event_type(Event_Type.HOME))

    elif event_type == Event_Type.VIEW:
        if grid_response.selected_data is None:
            # st.error("请选择一个线程")
            return
        selected_thread = grid_response.selected_data["THREAD_NAME"].values[0]
        selected_thread_config = load_config[selected_thread]
        thread_id = selected_thread_config["THREAD_ID"]
        col_edit_1, col_edit_2 = st.columns(2)
        with col_edit_1:
            st.subheader(f"正在查看线程：{selected_thread}")

        THREAD_NAME = st.text("当前线程名称")
        st.subheader(selected_thread)
        THREAD_ID = selected_thread_config["THREAD_ID"]
        CLASSES = get_edited_classes_data(selected_thread_config.get("CLASSES"))
        SLEEP_TIME = st.text(Tokens.SLEEP_TIME)
        st.subheader(str(selected_thread_config["SLEEP_TIME"]))
        CALLBACK_URL = st.text(Tokens.CALLBACK_URL)
        st.subheader(selected_thread_config["CALLBACK_URL"])
        STREAMS = get_edited_streams_data(selected_thread_config["STREAMS"],
                                          selected_thread_config.get("THREAD_ID"))

    elif event_type == Event_Type.EDIT:
        if grid_response.selected_data is None:
            st.error("请选择一个线程")
            return
        selected_thread = grid_response.selected_data["THREAD_NAME"].values[0]
        selected_thread_config = load_config[selected_thread]
        thread_id = selected_thread_config["THREAD_ID"]
        if clip_detection.is_thread_running(thread_id):
            update_event_type(Event_Type.HOME)
            st.error("线程正在运行中，无法修改参数。")
        else:
            col_edit_1, col_edit_2 = st.columns(2)
            with col_edit_1:
                st.subheader(f"正在编辑线程：{selected_thread}")

            THREAD_NAME = st.text_input("当前线程名称", selected_thread)
            THREAD_ID = selected_thread_config["THREAD_ID"]
            CLASSES = get_edited_classes_data(selected_thread_config.get("CLASSES"))
            SLEEP_TIME = st.number_input(
                Tokens.SLEEP_TIME, value=selected_thread_config["SLEEP_TIME"]
            )
            CALLBACK_URL = st.text_input(
                Tokens.CALLBACK_URL, selected_thread_config["CALLBACK_URL"]
            )
            STREAMS = get_edited_streams_data(selected_thread_config["STREAMS"],
                                              selected_thread_config.get("THREAD_ID"))

            if st.button("保存线程修改"):
                if THREAD_NAME.strip() == "":
                    st.error("线程名称不能为空。")
                else:
                    if threads_config.isSameNameExists(
                        THREAD_NAME, THREAD_ID, clip=True
                    ):
                        st.error("线程名称已存在，请修改。")
                    else:
                        # 保存
                        update_thread_config(
                            load_config,
                            THREAD_ID,
                            THREAD_NAME,
                            CLASSES,
                            STREAMS,
                            CALLBACK_URL,
                            SLEEP_TIME,
                            is_new_thread=False,
                        )
                        update_event_type(Event_Type.HOME)
                        st.success("成功修改线程")
                        st.session_state.pop(f"temp_streams_{THREAD_ID}", None)  # 清除对应的临时数据
                        time.sleep(1)
                        st.rerun()
            st.button("取消编辑", on_click=lambda: update_event_type(Event_Type.HOME))
            # Delete thread
            del_expander = st.expander(":red[删除当前模型]")
            if del_expander.button("确定删除", type="primary"):
                if clip_detection.is_thread_running(thread_id):
                    st.error("线程正在运行中，无法删除。")
                else:
                    del load_config[selected_thread]
                    threads_config.clip_save_config(load_config)
                    update_event_type(Event_Type.HOME)
                    st.success("线程已删除！稍等刷新")
                    time.sleep(2)
                    st.rerun()
    elif event_type == Event_Type.START:
        if grid_response.selected_data is None:
            st.error("请选择一个线程")
            return
        thread_name = grid_response.selected_data["THREAD_NAME"].values[0]
        status_placeholder = st.empty()
        status_placeholder.text(f"线程 {thread_name} 启动中")
        response = clip_detection.start_thread(thread_name)
        update_event_type(Event_Type.HOME)
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
            st.error("请选择一个线程")
            return
        thread_name = grid_response.selected_data["THREAD_NAME"].values[0]
        response = clip_detection.stop_thread(thread_name)
        update_event_type(Event_Type.HOME)
        if response.status_code == 200:
            st.success(f"线程 {thread_name} 已成功停止")
            # config[thread_name]["status"] = "off"
            # threads_config.save_config(config)
            time.sleep(2)
            st.rerun()
        else:
            st.error(f"停止线程 {thread_name} 失败: {response.text}")


def filter_by_key(data, key):
    """
    根据指定键过滤重复项

    Args:
      data: 字典列表
      key: 用于判断重复的键

    Returns:
      过滤后的字典列表
    """

    seen = set()
    result = []
    for d in data:
        val = d[key]
        if val not in seen:
            seen.add(val)
            result.append(d)
    return result


def update_thread_config(
    config,
    thread_id,
    thread_name,
    classes,
    streams,
    callback_url,
    sleep_time,
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
        "CLASSES": filter_by_key(classes, "CLASS"),  # 过滤重复项
        "SLEEP_TIME": sleep_time,
        "STREAMS": streams,
        "running_status": "off",
    }
    threads_config.clip_save_config(config)


def update_event_type(event_type: Event_Type):
    """在session_state中更新事件类型"""
    st.session_state.page_clip_thread_event_type = event_type


def get_event_type() -> Event_Type:
    """在session_state中获取事件类型"""
    event_type = st.session_state.get("page_clip_thread_event_type")
    return event_type if event_type is not None else Event_Type.HOME


def is_view_event() -> bool:
    """判断当前是否处于查看页面事件"""
    return get_event_type() == Event_Type.VIEW


def handle_grid_events(grid_response: AgGridReturn):
    """处理表格事件"""
    # 只要表格行被选中，都会触发"selectionChanged"事件
    if grid_response.event_data is not None:
        if (
            "selectionChanged" == grid_response.event_data["type"]
            and grid_response.selected_data is not None
        ):
            # row被选中
            event_type = get_event_type()
            if event_type == Event_Type.HOME:
                update_event_type(Event_Type.VIEW)
            pass


def get_edited_streams_data(streams: List[Dict[str, str]], thread_id: str) -> List[Dict[str, str]]:
    """
    输入视频流列表,返回经界面编辑后的视频流列表。

    Args:
        streams (List[Dict[str, str]]): 视频流列表，每个元素是一个视频流配置项。
        thread_id (str): 当前线程的唯一标识符。

    Returns:
        List[Dict[str, str]]: 编辑后的视频流列表，每个元素是一个视频流配置项。
    """
    if thread_id is not None:
        # 使用 thread_id 初始化临时存储视频流数据
        temp_key = f"temp_streams_{thread_id}"
        if temp_key not in st.session_state:
            st.session_state[temp_key] = streams.copy()

        # 添加 JSON 输入框
        st.subheader("添加 JSON 数据到视频流列表")
        json_input = st.text_area(
            "输入 JSON 数据：",
            height=200,
            placeholder='例如：[{"CAMERA_SN": "3", "RTSP_URL": "rtsp://..."}, ...]'
        )
        
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
        bview = is_view_event()  # 判断当前是否为查看模式
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
        edited_df = st.data_editor(
            data=df, column_config=column_config, num_rows="dynamic", disabled=bview
        )
        if not bview:
            st.caption("按:red[delete]键可以删除被选中的行.")

        # 更新临时数据
        st.session_state[temp_key] = edited_df.to_dict(orient="records")
        
        # 返回最终数据，但不保存
        return st.session_state[temp_key]
    else:
        # 配置编辑表格
        bview = is_view_event()  # 判断当前是否为查看模式
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
        edited_df = st.data_editor(
            data=df, column_config=column_config, num_rows="dynamic", disabled=bview
        )
        if not bview:
            st.caption("按:red[delete]键可以删除被选中的行.")

        return edited_df.to_dict(orient="records")


def get_edited_classes_data(classes: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    输入检测类别列表,返回经界面编辑后的检测类别列表。

    Args:
        classes (List[Dict[str, str]]): 检测类别列表，每个元素是一个检测类别配置项。

    Returns:
        List[Dict[str, str]]: 编辑后的检测类别列表，每个元素是一个检测类别配置项。

    """
    bview = is_view_event()
    # 设置编辑表格
    column_config = {
        "CLASS": st.column_config.SelectboxColumn(
            "检测类别",
            width="medium",
            options=[item["class"] for item in CONFIGED_CLASSES],
            required=True,
        ),
        "SIMILARITY_THRESHOLD": st.column_config.NumberColumn(
            "最小相似度",
            min_value=0.01,
            max_value=0.99,
            step=0.01,
            format="%0.2f",
            required=True,
        ),
        "FACE_DETECTION": st.column_config.CheckboxColumn(
            "人脸检测",
            default=False,
        ),
    }

    # 创建 DataFrame
    df = pd.DataFrame(
        data=classes, columns=["CLASS", "SIMILARITY_THRESHOLD", "FACE_DETECTION"]
    )
    # 展现编辑表格并返回编辑后的结果
    st.subheader("检测类别设置")
    edited_df = st.data_editor(
        data=df,
        column_config=column_config,
        num_rows="dynamic",
        # on_change=handle_edited_classes_events,
        disabled=bview,
    )
    if not bview:
        st.caption("按:red[delete]键可以删除被选中的行.")
    # 展示支持的检测类别
    st.caption(":red[支持的检测类别]")
    st.dataframe(
        [
            {
                "name": d["name"],
                "class": d["class"],
                "similarity_threshold": d["similarity_threshold"],
            }
            for d in CONFIGED_CLASSES
        ],
        column_config={
            "name": "名称",
            "classs": "类别",
            "similarity_threshold": st.column_config.NumberColumn("默认相似度"),
        },
        hide_index=True,
    )
    # 封装返回结果
    return edited_df.to_dict(orient="records")


def handle_edited_classes_events():
    """处理编辑检测类别表格事件"""
    pass
