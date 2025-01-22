# -*- encoding: utf-8 -*-
"""
线程-任务-管理-页面

"""
import time
from PIL import Image
import streamlit as st
from st_aggrid import (
    GridOptionsBuilder,
    AgGrid,
    GridUpdateMode,
    DataReturnMode,
    JsCode,
    AgGridReturn,
)
import pandas as pd
from ultralytics import YOLO
import detection
import threads_config
import json
from mods.constants import *


def pagecontrol():
    st.header("⏯️ 检测线程管理", divider=True)
    st.caption("控制检测线程开关的地方")
    config = threads_config.load_config()
    futures = {}
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
        col10,
        col11,
        col12,
        col13,
        col14,
    ) = st.columns(14)

    start_button = col1.button("启动")
    stop_button = col2.button("停止")

    table_data = []
    for thread_name, thread_config in config.items():
        table_data.append(
            {
                Tokens.THREAD_NAME: thread_name,
                Tokens.MODEL_NAME: thread_config["MODEL_NAME"],
                Tokens.DETECTION_SOURCE: thread_config.get("SOURCE"),
                Tokens.STREAMS: json.dumps(thread_config.get("STREAMS")),
                Tokens.SUBSCRIBES: thread_config.get("SUBSCRIBES"),
                Tokens.CALLBACK_URL: thread_config.get("CALLBACK_URL"),
                Tokens.SLEEP_TIME: thread_config.get("SLEEP_TIME"),
                Tokens.RUNNING_STATUS: detection.is_thread_running(
                    thread_config["THREAD_ID"]
                ),
            }
        )
    df = pd.DataFrame(table_data)

    tooltip_value_getter_code = JsCode(
        """
        function(params) {
            return params.value ? params.value : '';
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
    gb.configure_default_column(
        tooltipValueGetter=tooltip_value_getter_code, filter="agTextColumnFilter"
    )
    gb.configure_column(Tokens.THREAD_NAME, checkboxSelection=True)
    gb.configure_column(
        Tokens.RUNNING_STATUS, cellRenderer=status_cell_render, filter=""
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

    selected_threads = grid_response.selected_rows
    selected_thread_names = (
        [row[Tokens.THREAD_NAME] for index, row in selected_threads.iterrows()]
        if selected_threads is not None
        else []
    )

    #preview_button = None
    # 单个线程被选中
    one_thread_selected = len(selected_thread_names) == 1
    # # 如果线程被选中,则显示实时检测结果
    # if one_thread_selected:
    #     preview_button = col4.button("预览", use_container_width=True)

    if start_button and one_thread_selected:
        status_placeholder = st.empty()
        status_placeholder.text(f"线程 {selected_thread_names} 启动中")
        for thread_name in selected_thread_names:
            # response = requests.post(f"{base_url}/api/start_thread", json={"thread_name": thread_name})
            response = detection.start_thread(thread_name)
            if response.status_code == 200:
                st.success(f"线程 {thread_name} 已成功启动", icon="🎉")
                # config[thread_name]["status"] = "on"
                # threads_config.save_config(config)
                time.sleep(2)
                st.rerun()
            else:
                st.error(f"启动线程 {thread_name} 失败: {response.text}")

    if stop_button and one_thread_selected:
        for thread_name in selected_thread_names:
            # response = requests.post(f"{base_url}/api/stop_thread", json={"thread_name": thread_name})
            response = detection.stop_thread(thread_name)
            if response.status_code == 200:
                st.success(f"线程 {thread_name} 已成功停止")
                # config[thread_name]["status"] = "off"
                # threads_config.save_config(config)
                time.sleep(2)
                st.rerun()
            else:
                st.error(f"停止线程 {thread_name} 失败: {response.text}")

    # if preview_button and one_thread_selected:  # 点击了预览按钮
    #     preview_thread(config[selected_thread_names[0]])


def preview_thread(thread):
    """
    预览线程的预测结果
    """
    (
        rtsp_url,
        callback_url,
        camera_sn,
        model_path,
        inference,
        plot,
        sleeptime,
        imgquality,
        thread_id,
    ) = (
        thread["STREAMS"][0]["RTSP_URL"],
        thread["CALLBACK_URL"],
        thread["STREAMS"][0]["CAMERA_SN"],
        thread["MODEL_PATH"],
        thread["INFERENCE_ARGS"],
        thread["PLOT_ARGS"],
        thread["SLEEP_TIME"],
        thread["IMGQUALITY"],
        thread["THREAD_ID"],
    )
    inference["classes"] = list(map(int, inference["classes"]))
    local_model = YOLO(model_path)
    # Run inference on the source
    results = local_model(rtsp_url, stream=True, **inference)
    # 放置图片
    image_placeholder = st.empty()
    # generator of Results objects
    for result in results:
        im_array = result.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        # 只要用户退出本页面 本行就会抛出异常 所以本过程不会一直循环下去
        image_placeholder.image(im, caption="检测实时预览", use_column_width=True)
        # TODO: 可以配置睡眠时间
        time.sleep(10)
