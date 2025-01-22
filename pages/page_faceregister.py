import io
import time
import cv2
from loguru import logger
import numpy as np
import streamlit as st
import sqlite3
import base64
import pandas as pd
import face.facefunction
from st_aggrid import (
    GridOptionsBuilder,
    AgGrid,
    GridUpdateMode,
    DataReturnMode,
    JsCode,
    ColumnsAutoSizeMode,
    AgGridReturn,
)
from mods.constants import *


def faceregister():
    st.header("👦 人脸注册", divider=True)

    col1,col2 = st.columns(2)
    delete_button = col1.button("删除")


    # 加载数据表到DataFrame
    conn = face.facefunction.create_connection()
    if conn:
        frc = conn.cursor()
        frc.execute("SELECT id,user_name,face_image,age,gender,feature FROM faces")
        rows = frc.fetchall()
        face.facefunction.close_connection(conn)
        df = pd.DataFrame(
            rows, columns=["id", "user_name", "face_image", "age", "gender", "feature"]
        )

    image_cell_render = JsCode(
        """
        class ImgCellRenderer {
            init(params) {
                this.eGui = document.createElement('span');
                this.eGui.innerHTML = `<img src="${params.value}" height="24" width="24"/>`;
            }
            getGui() {
                return this.eGui;
            }       
        }
        """
    )
    tooltip_component_code = JsCode(
        """
        class CustomTooltip {            
            init(params) {
                this.eGui = document.createElement('span');
                this.eGui.innerHTML = (params.colDef.field == 'face_image') ? `<img src="${params.value}" height="128" width="128"/>`:params.value;
            }
            getGui() {
                return this.eGui;
            }       
        }
        """
    )


    # 关于配置选项, 参考: https://staggrid-examples.streamlit.app/
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_selection(selection_mode="multiple", use_checkbox=True)  # 允许用户多选
    gb.configure_pagination(paginationAutoPageSize=False)
    gb.configure_default_column(tooltipComponent=tooltip_component_code)
    gb.configure_column("id", "ID", checkboxSelection=True)
    gb.configure_column("user_name", "名称", filter="agTextColumnFilter")
    gb.configure_column("age", "年龄")
    gb.configure_column("gender", "性别")
    gb.configure_column("feature", "特征向量", tooltipField="feature")
    gb.configure_column(
        "face_image", "头像", cellRenderer=image_cell_render, tooltipField="face_image"
    )
    gb.configure_grid_options(
        tooltipShowDelay=1000,
        # domLayout="autoHeight",
        rowStyle={"text-align": "center"},
        localeText=AG_GRID_LOCALE_CN,
    )
    # 关于ag-grid的配置选项，参考: https://www.ag-grid.com/javascript-data-grid/grid-options/
    gridOptions = gb.build()
    grid_response: AgGridReturn = AgGrid(
        df,
        gridOptions=gridOptions,
        allow_unsafe_jscode=True,
        data_return_mode="AS_INPUT",
        update_mode=GridUpdateMode.MODEL_CHANGED,
        fit_columns_on_grid_load=True,
        theme="streamlit",
        # width="100%",
        update_on=[],  # 在此注册要后台处理的事件列表，参考https://www.ag-grid.com/javascript-data-grid/grid-events/
        columns_auto_size_mode=ColumnsAutoSizeMode.FIT_ALL_COLUMNS_TO_VIEW,
    )

    selected_data = grid_response["selected_rows"]

    if delete_button:
        if selected_data is not None and not selected_data.empty:
            selected_ids = selected_data.loc[:, "id"].tolist()
            if selected_ids:
                logger.info(f"删除人脸信息: {selected_ids}")
                # 处理删除操作
                conn = face.facefunction.create_connection()
                if conn:
                    cur = conn.cursor()
                    sql = "DELETE FROM faces WHERE id IN ({})".format(','.join(['%s'] * len(selected_ids)))
                    cur.execute(sql, selected_ids)
                    conn.commit()
                    face.facefunction.close_connection(conn)
                st.info("删除成功")
                time.sleep(2)
                st.rerun()
        else:
            st.error("请选择要删除的人脸")

    uploaded_file = st.file_uploader("上传图片", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        # print(FaceRecognition.__dict__)

        # 检测人脸
        frd = (
            face.facefunction.getRecognizaObject(None)
        )  
        results = frd.detect(image)
        if len(results) > 0:
            for i, result in enumerate(results):
                # 假设 result["bbox"] 的结构是 [x1, y1, x2, y2]
                bbox = result["bbox"]
                x1, y1, x2, y2 = bbox

                # 计算扩大后的边界框坐标
                margin_x = int((x2 - x1) * 0.5)  # 计算x方向扩大的0.5倍范围
                margin_y = int((y2 - y1) * 0.5)  # 计算y方向扩大的0.5倍范围

                # 确保新的坐标在图像范围内
                new_x1 = max(0, x1 - margin_x)
                new_y1 = max(0, y1 - margin_y)
                new_x2 = min(image.shape[1], x2 + margin_x)  # frame.shape[1] 是图像的宽度
                new_y2 = min(image.shape[0], y2 + margin_y)  # frame.shape[0] 是图像的高度

                # 使用新的边界框坐标来截取 face_img
                face_img = image[new_y1:new_y2, new_x1:new_x2]
                st.image(face_img, caption=f"人脸{i+1}",channels="BGR")
                user_name = st.text_input(f"输入人脸{i+1}的名称")

                if st.button(f"注册人脸{i+1}"):
                    if user_name:
                        _, encoded_img = cv2.imencode(".jpg", face_img)
                        encoded_img_base64 = base64.b64encode(encoded_img).decode(
                            "utf-8"
                        )
                        face_image_base64 = (
                            f"data:image/jpg;base64,{encoded_img_base64}"
                        )

                        age = result.get("age")
                        gender = result.get(
                            "gender"
                        )

                        register_result = frd.register(
                            user_name,
                            result["embedding"],
                            face_image_base64,
                            age,
                            gender,
                            "common"
                        )
                        st.info(register_result)
                        time.sleep(2)
                        st.rerun()
        else:
            st.error("未识别到人脸")
