from enum import Enum
import time
import streamlit as st
import json
import os
from PIL import Image
from ultralytics import YOLO
from mods.constants import *
import models_config
import threads_config
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


class Event_Type(Enum):
    """用户操作-事件-类型"""

    HOME = 1
    """主页,默认值"""
    VIEW = 2
    """查看模型"""
    NEW = 3
    """新建模型"""


def update_event_type(event_type: Event_Type):
    """在session_state中更新事件类型"""
    st.session_state.page_model_event_type = event_type


def get_event_type() -> Event_Type:
    """在session_state中获取事件类型"""
    event_type = st.session_state.get("page_model_event_type")
    return event_type if event_type is not None else Event_Type.HOME


def handle_grid_events(grid_response: AgGridReturn):
    """处理表格事件"""
    if grid_response.event_data is not None:
        if (
            "selectionChanged" == grid_response.event_data["type"]
            and grid_response.selected_data is not None
        ):
            # 只要表格行被选中，都会触发"selectionChanged"事件
            """row被选中"""
            update_event_type(Event_Type.VIEW)


def save_model_info(model_name, model_path, notes, class_names):
    models_info = models_config.load_config()
    models_info[model_name] = {
        "path": model_path,
        "class_names": class_names,
        "notes": notes,
    }
    models_config.save_config(models_info)


def pagemodel():
    st.header("🤖  模型管理", divider=True)
    models_info = models_config.load_config()
    yolomodel_names = list(models_info.keys())
    futures = {}
    (
        col1,
        col2,
    ) = st.columns(2)
    col1.button("新建模型", on_click=lambda: update_event_type(Event_Type.NEW))
    loaded_config_values = models_info.values()
    df = pd.DataFrame(
        loaded_config_values,
        columns=["path", "notes"],
    )
    df.insert(loc=0, column="MODEL_NAME", value=yolomodel_names)
    df.insert(
        loc=2,
        column="class_names",
        value=[json.dumps(x.get("class_names")) for x in loaded_config_values],
    )
    # 关于配置选项, 参考: https://staggrid-examples.streamlit.app/
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_selection(selection_mode="single", use_checkbox=False)  # 允许用户多选
    gb.configure_pagination(paginationAutoPageSize=False)
    gb.configure_column("MODEL_NAME", Tokens.MODEL_NAME, checkboxSelection=True)
    gb.configure_column("path", "文件路径", tooltipField="path")
    gb.configure_column("notes", "备注", tooltipField="notes")
    gb.configure_column("class_names", "检测类别", tooltipField="class_names")
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
    elif event_type == Event_Type.VIEW:
        if grid_response.selected_data is None:
            return
        """row被选中"""
        selected_yolomodel = grid_response.selected_data["MODEL_NAME"].values[0]

        if selected_yolomodel == "":
            return
        elif selected_yolomodel:
            model_path = models_info[selected_yolomodel]["path"]
            class_names = models_info[selected_yolomodel]["class_names"]
            st.write(f"模型地址：{model_path}")
            # Display class names
            st.write("检测类别:")
            st.write(class_names)
            model = YOLO(model_path)
            uploaded_image = st.file_uploader(
                "上传图片进行检测", type=["jpg", "jpeg", "png"]
            )
            if uploaded_image is not None:
                image = Image.open(uploaded_image)
                results = model.predict(image)
                im_array = results[0].plot()
                im = Image.fromarray(im_array[..., ::-1])
                st.image(im, caption="检测结果", use_column_width=True)
        # Delete thread
        del_expander = st.expander(":red[删除当前模型]")
        if del_expander.button("确定删除", type="primary"):
            if threads_config.is_model_used(selected_yolomodel):
                st.error(f"模型[{selected_yolomodel}]正在使用中，无法删除！")
            else:
                models_config.delete(selected_yolomodel)
                st.success("模型已删除！稍等刷新")
                time.sleep(2)
                st.rerun()
    elif event_type == Event_Type.NEW:
        # 新建模型
        uploaded_file = st.file_uploader("上传模型文件", type=["pt"])
        if uploaded_file is not None:
            # 检查目录是否存在，如果不存在则创建
            if not os.path.exists(models_config.MODELS_DIR):
                os.makedirs(models_config.MODELS_DIR)
            model_path = os.path.join(models_config.MODELS_DIR, uploaded_file.name)
            # 判断同名模型文件是否已经存在
            if os.path.exists(model_path):
                st.error(f"同名模型文件{uploaded_file.name}已经存在")
                # update_event_type(Event_Type.NEW)
                return
            model_name = st.text_input("模型名称")
            notes = st.text_input("备注")
            if st.button("保存模型"):
                if model_name != "":
                    with open(model_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    # Load the model to get class names
                    model = YOLO(model_path)
                    class_names = model.model.names
                    save_model_info(model_name, model_path, notes, class_names)
                    st.success("模型已保存！")
                    yolomodel_names.append(model_name)
                    update_event_type(Event_Type.HOME)
                    st.rerun()
                else:
                    st.error("模型名称不能为空啊")
        st.button("取消新建", on_click=lambda: update_event_type(Event_Type.HOME))
    else:  # if event_type == Event_Type.HOME:
        pass
