import streamlit as st
import sqlite3
import pandas as pd
from st_aggrid import (
    GridOptionsBuilder,
    AgGrid,
    GridUpdateMode,
    JsCode,
    ColumnsAutoSizeMode,
    AgGridReturn,
)

from mods.constants import AG_GRID_LOCALE_CN

DATABASE_FILE = "face_detection_results.db"

def face_results_show():
    st.header("👦 人脸检测结果展示", divider=True)

    # 加载数据表到DataFrame
    with sqlite3.connect(DATABASE_FILE) as frconn:
        frc = frconn.cursor()
        frc.execute("SELECT id,user_name,face_img,dist,detection_time,camera_sn FROM faces_results order by id desc")
        rows = frc.fetchall()
        df = pd.DataFrame(
            rows, columns=["id", "user_name", "face_img", "dist", "detection_time", "camera_sn"]
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
                this.eGui.innerHTML = (params.colDef.field == 'face_img') ? `<img src="${params.value}" height="128" width="128"/>`:params.value;
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
    gb.configure_column("dist", "距离值")
    gb.configure_column("detection_time", "检测时间")
    gb.configure_column("camera_sn", "摄像头编号")
    gb.configure_column(
        "face_img", "头像", cellRenderer=image_cell_render, tooltipField="face_img"
    )
    gb.configure_grid_options(
        tooltipShowDelay=1000,
        #domLayout="autoHeight",
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
        update_mode=GridUpdateMode.VALUE_CHANGED,
        fit_columns_on_grid_load=True,
        theme="streamlit",
        # width="100%",
        update_on=[],  # 在此注册要后台处理的事件列表，参考https://www.ag-grid.com/javascript-data-grid/grid-events/
        columns_auto_size_mode=ColumnsAutoSizeMode.FIT_ALL_COLUMNS_TO_VIEW,
    )






