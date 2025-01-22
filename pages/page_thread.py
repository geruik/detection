import os
import json
import streamlit as st
import uuid
import config
import time
from mods.constants import *
import threads_config
import models_config
import detection
import pandas as pd
from st_aggrid import (
    GridOptionsBuilder,
    AgGrid,
    GridUpdateMode,
    DataReturnMode,
    JsCode,
    ColumnsAutoSizeMode,
    AgGridReturn,
)
from enum import Enum
from typing import List, Dict


class Event_Type(Enum):
    """用户操作-事件-类型"""

    HOME = 1
    """主页,默认值"""
    NEW = 2
    """编辑新线程"""
    EDIT = 3
    """编辑原有线程"""
    COPY = 4
    """复制线程"""


DEFULT_MQTT_SUBSCRIBE = config.mqtt.get("subscribe")
"""默认的MQTT订阅主题"""


def update_thread_config(
    config,
    thread_id,
    thread_name,
    MODEL_INFO,
    selected_model_name,
    inference_args,
    plot_args,
    detection_source,
    subscribes,
    streams,
    callback_url,
    sleep_time,
    imgquality,
    restrictarea,
    restrictareajson,
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
        "MODEL_NAME": selected_model_name,
        "MODEL_PATH": MODEL_INFO["path"],
        "INFERENCE_ARGS": inference_args,
        "PLOT_ARGS": plot_args,
        "SOURCE": detection_source,
        "SUBSCRIBES": subscribes,
        "STREAMS": streams,
        "CALLBACK_URL": callback_url,
        "SLEEP_TIME": sleep_time,
        "IMGQUALITY": imgquality,
        "status": "off",
        "RESTRICTAREA": restrictarea,
        "RESTRICTAREAJSON": restrictareajson,
    }
    threads_config.save_config(config)


def pagethread():
    st.header("🧵 编辑检测线程", divider=True)
    # Load thread and model information
    loaded_config = threads_config.load_config()

    # Load model information
    model_info = models_config.load_config()
    model_names = list(model_info.keys())

    futures = {}
    (
        col1,
        col2,
        col3,
    ) = st.columns(3)

    col1.button("新建线程", on_click=lambda: update_event_type(Event_Type.NEW))
    loaded_config_values = loaded_config.values()
    df = pd.DataFrame(
        loaded_config_values,
        columns=[
            "MODEL_NAME",
            "CALLBACK_URL",
            "SUBSCRIBES",
            "SLEEP_TIME",
            "SOURCE",
        ],
    )
    df.insert(loc=0, column="THREAD_NAME", value=loaded_config.keys())
    df.insert(
        loc=3,
        column="STREAMS",
        value=[json.dumps(x.get("STREAMS")) for x in loaded_config_values],
    )
    df.insert(
        loc=7,
        column="RUNNING_STATUS",
        value=[
            detection.is_thread_running(x["THREAD_ID"]) for x in loaded_config_values
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

    # 关于配置选项, 参考: https://staggrid-examples.streamlit.app/
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_selection(selection_mode="single", use_checkbox=False)  # 允许用户多选
    gb.configure_pagination(paginationAutoPageSize=False)
    gb.configure_column(
        "THREAD_NAME",
        Tokens.THREAD_NAME,
        # 即使是运行中的线程，也允许用户选择，方便查看线程配置信息。Modified by creativor 2024-08-12
        checkboxSelection= True, #checkbox_selection_code,
        tooltipField="THREAD_NAME",
        filter="agTextColumnFilter",
    )
    gb.configure_column("MODEL_NAME", Tokens.MODEL_NAME, filter="agTextColumnFilter")
    gb.configure_column(
        "CALLBACK_URL",
        Tokens.CALLBACK_URL,
        tooltipField="CALLBACK_URL",
        filter="agTextColumnFilter",
    )
    gb.configure_column("SLEEP_TIME", Tokens.SLEEP_TIME)
    gb.configure_column(
        "STREAMS", Tokens.STREAMS, tooltipField="STREAMS", filter="agTextColumnFilter"
    )
    gb.configure_column(
        "SUBSCRIBES",
        Tokens.SUBSCRIBES,
        tooltipField="SUBSCRIBES",
        filter="agTextColumnFilter",
    )
    gb.configure_column("SOURCE", Tokens.DETECTION_SOURCE)
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
        if grid_response.selected_data is None:
            # 新建线程
            st.subheader("请填写新线程的配置参数：")
            THREAD_ID = threads_config.generate_thread_id()
            col1, col2 = st.columns(2)
            THREAD_NAME = col1.text_input(Tokens.THREAD_NAME)
            if model_names is None or len(model_names) == 0:
                st.error("模型不能为空")
                return
            selected_model_name = col2.selectbox("选择模型", model_names)
            MODEL_INFO = model_info[selected_model_name]
            class_names = MODEL_INFO["class_names"]
            class_options = list(class_names.values())

            with st.container(border=True):  # 检测设置
                st.subheader("检测设置：")
                col3, col4, col5 = st.columns(3)
                INFERENCE_ARGS = {
                    "imgsz": col3.number_input(
                        "输入到模型里图片的size,会自动压缩", value=1280
                    ),
                    "conf": col4.slider(
                        "最小置信度限制", min_value=0.0, max_value=1.0, value=0.1
                    ),
                    "iou": col5.slider(
                        "IOU阈值", min_value=0.0, max_value=1.0, value=0.5
                    ),
                    "device": col3.selectbox("检测的硬件设备", ["cuda", "cpu"]),
                    #'classes': col4.multiselect("筛选类别", [0, 1], default=[0, 1]),
                    "classes": col4.multiselect(
                        "筛选类别", class_options, default=class_options
                    ),
                }
                SLEEP_TIME = col5.number_input(Tokens.SLEEP_TIME, value=10)

            with st.container(border=True):  # 检测后返回的图片设置
                st.subheader("检测后返回的图片设置：")
                col7, col8, col9, col15, col16, col17 = st.columns(6)
                IMGQUALITY = col7.slider(
                    "输出图片压缩质量比例，减小文件大小",
                    min_value=1,
                    max_value=100,
                    value=40,
                )
                PLOT_ARGS = {
                    "conf": col8.checkbox("是否显示置信度", value=True),
                    "smoke_detect": col8.checkbox("是否是抽烟检测", value=False),
                    "labels": col9.checkbox("显示类别标签", value=True),
                    "boxes": col15.checkbox("是否显示边框", value=True),
                    "face_detect": col16.checkbox("是否进行人脸识别", value=False),
                    "face_detect_type": col17.selectbox(
                        "人脸识别范围", ["全图", "检测范围内"]
                    ),
                }

            with st.container(border=True):  # 视频流设置
                st.subheader("视频流设置：")
                # 检测来源
                (col_source,) = st.columns(1)
                detection_source = col_source.selectbox(
                    Tokens.DETECTION_SOURCE, detection.DetectionSource._member_names_
                )
                SUBSCRIBES = None
                edited_streams_data = []
                if detection_source == detection.DetectionSource.MQTT.value:
                    col10, col12 = st.columns(2)
                    SUBSCRIBES = col10.text_input(
                        Tokens.SUBSCRIBES,
                        DEFULT_MQTT_SUBSCRIBE,
                        help=Tokens.SUBSCRIBES_HELP,
                    )
                else:
                    edited_streams_data = get_edited_streams_data([],None)
                CALLBACK_URL = st.text_input(
                    Tokens.CALLBACK_URL, "http://192.168.0.85:5006/znyj/violation/recognize"
                )

            with st.container(border=True):  # 检测限制区域设置
                st.subheader("检测限制区域设置：")
                RESTRICTAREA = st.checkbox("是否限制检测区域", value=False)
                RESTRICTAREAJSON = st.text_input("填写限制区域的坐标：")
            if st.button("新建并提交线程配置"):
                if THREAD_NAME.strip() == "":
                    st.error("线程名称不能为空。")
                else:
                    if threads_config.isSameNameExists(THREAD_NAME, THREAD_ID):
                        st.error("线程名称已存在，请修改。")
                    else:
                        selected_classes_ids = [
                            class_id
                            for class_id, class_name in class_names.items()
                            if class_name in INFERENCE_ARGS["classes"]
                        ]
                        # selected_classes_ids = [item[0] for item in selected_classes_ids]
                        INFERENCE_ARGS["classes"] = selected_classes_ids
                        update_event_type(Event_Type.HOME)
                        update_thread_config(
                            loaded_config,
                            THREAD_ID,
                            THREAD_NAME,
                            MODEL_INFO,
                            selected_model_name,
                            INFERENCE_ARGS,
                            PLOT_ARGS,
                            detection_source,
                            SUBSCRIBES,
                            edited_streams_data,
                            CALLBACK_URL,
                            SLEEP_TIME,
                            IMGQUALITY,
                            RESTRICTAREA,
                            RESTRICTAREAJSON,
                            is_new_thread=True,
                        )
                        st.success("成功新建线程")
                        time.sleep(1)
                        st.rerun()
            st.button("取消新建", on_click=lambda: update_event_type(Event_Type.HOME))

    elif event_type == Event_Type.EDIT:
        if grid_response.selected_data is None:
            # 没有被选中的线程
            return
        selected_thread = grid_response.selected_data["THREAD_NAME"].values[0]
        selected_thread_config = loaded_config[selected_thread]
        thread_id = selected_thread_config["THREAD_ID"]
        col_edit_1, col_edit_2 = st.columns(2)
        with col_edit_1:
            st.subheader(f"正在编辑线程：{selected_thread}")
        with col_edit_2:
            st.button(
                "复制本线程",
                on_click=lambda: update_event_type(Event_Type.COPY),
            )
        # Edit thread configuration
        selected_model_name = st.selectbox(
            "选择模型",
            model_names,
            index=(
                model_names.index(selected_thread_config["MODEL_NAME"])
                if selected_thread_config.get("MODEL_NAME") in model_names
                else 0
            ),
        )
        MODEL_INFO = model_info.get(selected_model_name)

        # id反向查名称
        class_names = MODEL_INFO.get("class_names") if MODEL_INFO is not None else {}
        class_options = list(class_names.values())
        selected_classes_names = (
            list(
                map(
                    lambda class_id: class_names.get(str(class_id)),
                    selected_thread_config["INFERENCE_ARGS"]["classes"],
                )
            )
            if len(class_options) > 0
            else []
        )

        THREAD_NAME = st.text_input("当前线程名称", selected_thread)
        THREAD_ID = selected_thread_config["THREAD_ID"]

        with st.container(border=True):  # INFERENCE_ARGS
            INFERENCE_ARGS = {
                "imgsz": st.number_input(
                    "输入图片的size",
                    value=selected_thread_config["INFERENCE_ARGS"]["imgsz"],
                ),
                "conf": st.slider(
                    "置信度限制",
                    min_value=0.0,
                    max_value=1.0,
                    value=selected_thread_config["INFERENCE_ARGS"]["conf"],
                ),
                "iou": st.slider(
                    "IOU阈值",
                    min_value=0.0,
                    max_value=1.0,
                    value=selected_thread_config["INFERENCE_ARGS"]["iou"],
                ),
                "device": st.selectbox(
                    "设备",
                    ["cuda", "cpu"],
                    index=["cuda", "cpu"].index(
                        selected_thread_config["INFERENCE_ARGS"]["device"]
                    ),
                ),
                "classes": st.multiselect(
                    "筛选类别", class_options, default=selected_classes_names
                ),
            }
        with st.container(border=True):  # PLOT_ARGS
            PLOT_ARGS = {
                "conf": st.checkbox(
                    "显示置信度",
                    value=selected_thread_config["PLOT_ARGS"]["conf"],
                ),
                "labels": st.checkbox(
                    "显示类别标签",
                    value=selected_thread_config["PLOT_ARGS"]["labels"],
                ),
                "boxes": st.checkbox(
                    "显示边框",
                    value=selected_thread_config["PLOT_ARGS"]["boxes"],
                ),
                "face_detect": st.checkbox(
                    "是否进行人脸识别",
                    value=(
                        selected_thread_config["PLOT_ARGS"]["face_detect"]
                        if "face_detect" in selected_thread_config["PLOT_ARGS"]
                        else False
                    ),
                ),
                "face_detect_type": st.selectbox(
                    "人脸识别范围",
                    ["全图", "检测范围内"],
                    index=["全图", "检测范围内"].index(
                        selected_thread_config["PLOT_ARGS"]["face_detect_type"]
                        if "face_detect_type" in selected_thread_config["PLOT_ARGS"]
                        else "检测范围内"
                    ),
                ),
                "smoke_detect": st.checkbox(
                    "是否是抽烟检测",
                    value=(
                        selected_thread_config["PLOT_ARGS"]["smoke_detect"]
                        if "smoke_detect" in selected_thread_config["PLOT_ARGS"]
                        else False
                    ),
                ),
            }
        with st.container(border=True):  # 视频流设置
            source_names = detection.DetectionSource._member_names_
            config_source = selected_thread_config.get("SOURCE")
            detection_source = st.selectbox(
                Tokens.DETECTION_SOURCE,
                source_names,
                index=(
                    source_names.index(config_source)
                    if config_source is not None
                    else 0
                ),
            )
            # MQTT配置与其它不一样
            edited_streams_data = []
            SUBSCRIBES = selected_thread_config.get("SUBSCRIBES")
            if detection_source == detection.DetectionSource.MQTT.value:
                configed_subscribes = (
                    SUBSCRIBES if SUBSCRIBES is not None else DEFULT_MQTT_SUBSCRIBE
                )
                SUBSCRIBES = st.text_input(
                    Tokens.SUBSCRIBES,
                    configed_subscribes,
                    help=Tokens.SUBSCRIBES_HELP,
                )
            else:
                edited_streams_data = get_edited_streams_data(
                    selected_thread_config.get("STREAMS"),
                    selected_thread_config.get("THREAD_ID")
                )

        CALLBACK_URL = st.text_input(
            Tokens.CALLBACK_URL, selected_thread_config["CALLBACK_URL"]
        )
        SLEEP_TIME = st.number_input(
            Tokens.SLEEP_TIME, value=selected_thread_config["SLEEP_TIME"]
        )
        IMGQUALITY = st.slider(
            "输出图片压缩质量比例",
            min_value=1,
            max_value=100,
            value=selected_thread_config["IMGQUALITY"],
        )
        with st.container(border=True):  # 限制检测区域设置
            RESTRICTAREA = st.checkbox(
                "是否限制检测区域",
                value=selected_thread_config["RESTRICTAREA"],
            )
            RESTRICTAREAJSON = st.text_input(
                "填写限制区域的坐标",
                selected_thread_config["RESTRICTAREAJSON"],
            )
        st.divider()
        if detection.is_thread_running(thread_id):
            st.error("线程正在运行中，无法修改参数。")
        else:
            col_edit_3, col_edit_4 = st.columns(2)
            if col_edit_3.button("保存线程修改"):
                if THREAD_NAME.strip() == "":
                    st.error("线程名称不能为空。")
                else:
                    if threads_config.isSameNameExists(THREAD_NAME, THREAD_ID):
                        st.error("线程名称已存在，请修改。")
                    else:
                        # 名称查id
                        selected_classes_ids = [
                            class_id
                            for class_id, class_name in class_names.items()
                            if class_name in INFERENCE_ARGS["classes"]
                        ]
                        INFERENCE_ARGS["classes"] = selected_classes_ids
                        # 保存
                        update_thread_config(
                            loaded_config,
                            THREAD_ID,
                            THREAD_NAME,
                            MODEL_INFO,
                            selected_model_name,
                            INFERENCE_ARGS,
                            PLOT_ARGS,
                            detection_source,
                            SUBSCRIBES,
                            edited_streams_data,
                            CALLBACK_URL,
                            SLEEP_TIME,
                            IMGQUALITY,
                            RESTRICTAREA,
                            RESTRICTAREAJSON,
                            is_new_thread=False,
                        )
                        update_event_type(Event_Type.HOME)
                        st.success("成功修改线程")
                        st.session_state.pop(f"temp_streams_{THREAD_ID}", None)  # 清除对应的临时数据
                        time.sleep(1)
                        st.rerun()
            # Delete thread
            del_expander = col_edit_4.expander(":red[删除当前线程]")
            if del_expander.button("确定删除", type="primary"):
                if detection.is_thread_running(thread_id):
                    st.error("线程正在运行中，无法删除。")
                else:
                    del loaded_config[selected_thread]
                    threads_config.save_config(loaded_config)
                    st.success("线程已删除！稍等刷新")
                    time.sleep(2)
                    st.rerun()
    elif event_type == Event_Type.COPY:
        if grid_response.selected_data is None:
            # 没有被选中的线程
            return
        selected_thread = grid_response.selected_data["THREAD_NAME"].values[0]
        selected_thread_config = loaded_config[selected_thread]
        thread_id = selected_thread_config["THREAD_ID"]
        st.warning(f"正在复制线程：{selected_thread}")
        # Edit thread configuration
        selected_model_name = st.selectbox(
            "选择模型",
            model_names,
            index=(
                model_names.index(selected_thread_config["MODEL_NAME"])
                if selected_thread_config.get("MODEL_NAME") in model_names
                else 0
            ),
        )
        MODEL_INFO = model_info.get(selected_model_name)

        # id反向查名称
        class_names = MODEL_INFO.get("class_names") if MODEL_INFO is not None else {}
        class_options = list(class_names.values())
        selected_classes_names = (
            list(
                map(
                    lambda class_id: class_names.get(str(class_id)),
                    selected_thread_config["INFERENCE_ARGS"]["classes"],
                )
            )
            if len(class_options) > 0
            else []
        )

        THREAD_NAME = st.text_input("当前线程名称", selected_thread + "_copy")
        THREAD_ID = threads_config.generate_thread_id()
        with st.container(border=True):  # INFERENCE_ARGS
            INFERENCE_ARGS = {
                "imgsz": st.number_input(
                    "输入图片的size",
                    value=selected_thread_config["INFERENCE_ARGS"]["imgsz"],
                ),
                "conf": st.slider(
                    "置信度限制",
                    min_value=0.0,
                    max_value=1.0,
                    value=selected_thread_config["INFERENCE_ARGS"]["conf"],
                ),
                "iou": st.slider(
                    "IOU阈值",
                    min_value=0.0,
                    max_value=1.0,
                    value=selected_thread_config["INFERENCE_ARGS"]["iou"],
                ),
                "device": st.selectbox(
                    "设备",
                    ["cuda", "cpu"],
                    index=["cuda", "cpu"].index(
                        selected_thread_config["INFERENCE_ARGS"]["device"]
                    ),
                ),
                "classes": st.multiselect(
                    "筛选类别", class_options, default=selected_classes_names
                ),
            }
        with st.container(border=True):  # PLOT_ARGS
            PLOT_ARGS = {
                "conf": st.checkbox(
                    "显示置信度",
                    value=selected_thread_config["PLOT_ARGS"]["conf"],
                ),
                "labels": st.checkbox(
                    "显示类别标签",
                    value=selected_thread_config["PLOT_ARGS"]["labels"],
                ),
                "boxes": st.checkbox(
                    "显示边框",
                    value=selected_thread_config["PLOT_ARGS"]["boxes"],
                ),
                "face_detect": st.checkbox(
                    "是否进行人脸识别",
                    value=(
                        selected_thread_config["PLOT_ARGS"]["face_detect"]
                        if "face_detect" in selected_thread_config["PLOT_ARGS"]
                        else False
                    ),
                ),
                "face_detect_type": st.selectbox(
                    "人脸识别范围",
                    ["全图", "检测范围内"],
                    index=["全图", "检测范围内"].index(
                        selected_thread_config["PLOT_ARGS"]["face_detect_type"]
                        if "face_detect_type" in selected_thread_config["PLOT_ARGS"]
                        else "检测范围内"
                    ),
                ),
                "smoke_detect": st.checkbox(
                    "是否是抽烟检测",
                    value=(
                        selected_thread_config["PLOT_ARGS"]["smoke_detect"]
                        if "smoke_detect" in selected_thread_config["PLOT_ARGS"]
                        else False
                    ),
                ),
            }
        with st.container(border=True):  # 视频流检测设置
            source_names = detection.DetectionSource._member_names_
            config_source = selected_thread_config.get("SOURCE")
            detection_source = st.selectbox(
                Tokens.DETECTION_SOURCE,
                source_names,
                index=(
                    source_names.index(config_source)
                    if config_source is not None
                    else 0
                ),
            )
            # MQTT配置与其它不一样
            edited_streams_data = []
            SUBSCRIBES = selected_thread_config.get("SUBSCRIBES")
            if detection_source == detection.DetectionSource.MQTT.value:
                configed_subscribes = (
                    SUBSCRIBES if SUBSCRIBES is not None else DEFULT_MQTT_SUBSCRIBE
                )
                SUBSCRIBES = st.text_input(
                    Tokens.SUBSCRIBES,
                    configed_subscribes,
                    help=Tokens.SUBSCRIBES_HELP,
                )
            else:
                st.session_state.pop(f"temp_streams_{THREAD_ID}", None)  # 清除对应的临时数据
                edited_streams_data = get_edited_streams_data(
                    selected_thread_config.get("STREAMS"),
                    selected_thread_config.get("THREAD_ID")
                )
        CALLBACK_URL = st.text_input(
            Tokens.CALLBACK_URL, selected_thread_config["CALLBACK_URL"]
        )
        SLEEP_TIME = st.number_input(
            Tokens.SLEEP_TIME, value=selected_thread_config["SLEEP_TIME"]
        )
        IMGQUALITY = st.slider(
            "输出图片压缩质量比例",
            min_value=1,
            max_value=100,
            value=selected_thread_config["IMGQUALITY"],
        )
        with st.container(border=True):  # 限制检测区域设置
            RESTRICTAREA = st.checkbox(
                "是否限制检测区域",
                value=selected_thread_config["RESTRICTAREA"],
            )
            RESTRICTAREAJSON = st.text_input(
                "填写限制区域的坐标",
                selected_thread_config["RESTRICTAREAJSON"],
            )
        st.divider()
        if st.button("新建复制线程"):
            if THREAD_NAME.strip() == "":
                st.error("线程名称不能为空。")
            else:
                if threads_config.isSameNameExists(THREAD_NAME, THREAD_ID):
                    st.error("线程名称已存在，请修改。")
                else:
                    # 名称查id
                    selected_classes_ids = [
                        class_id
                        for class_id, class_name in class_names.items()
                        if class_name in INFERENCE_ARGS["classes"]
                    ]
                    INFERENCE_ARGS["classes"] = selected_classes_ids
                    # 保存
                    update_thread_config(
                        loaded_config,
                        THREAD_ID,
                        THREAD_NAME,
                        MODEL_INFO,
                        selected_model_name,
                        INFERENCE_ARGS,
                        PLOT_ARGS,
                        detection_source,
                        SUBSCRIBES,
                        edited_streams_data,
                        CALLBACK_URL,
                        SLEEP_TIME,
                        IMGQUALITY,
                        RESTRICTAREA,
                        RESTRICTAREAJSON,
                        is_new_thread=False,
                    )
                    update_event_type(Event_Type.HOME)
                    st.success("成功复制线程...")
                    time.sleep(1)
                    st.rerun()
        st.button("取消复制", on_click=lambda: update_event_type(Event_Type.HOME))
    else:  # if event_type == Event_Type.HOME
        pass


def update_event_type(event_type: Event_Type):
    """在session_state中更新事件类型"""
    st.session_state.page_thread_event_type = event_type


def get_event_type() -> Event_Type:
    """在session_state中获取事件类型"""
    event_type = st.session_state.get("page_thread_event_type")
    return event_type if event_type is not None else Event_Type.HOME


def handle_grid_events(grid_response: AgGridReturn):
    """处理表格事件"""
    if grid_response.event_data is not None:
        if (
            "selectionChanged" == grid_response.event_data["type"]
            and grid_response.selected_data is not None
        ):
            if (
                get_event_type() != Event_Type.COPY
            ):  # 只要表格行被选中，都会触发"selectionChanged"事件
                """row被选中,且是编辑模式"""
                update_event_type(Event_Type.EDIT)


def get_edited_streams_data(streams: List[Dict[str, str]],thread_id: str) -> List[Dict[str, str]]:
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
