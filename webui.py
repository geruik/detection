"""
AI 物件检测 WEB UI 程序入口.
"""

import mods.app_initializer

def main():
    """
    streamlit webui 入口函数
    """
    import streamlit as st
    from streamlit_option_menu import option_menu
    from pages.page_thread import pagethread
    from pages.page_model import pagemodel
    from pages.page_control import pagecontrol
    from pages.page_rtsp import pagertsp
    from pages.page_faceregister import faceregister
    from pages.page_facerecognition import facerecognition
    from pages.page_face_thread import facethread
    from pages.page_clip_thread import clip_thread
    #from pages.page_face_thread_results import face_results_show
    st.set_page_config(
        page_title=" AI检测系统",
        page_icon="static/favicon.ico",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://www.weileit.com",
            "Report a bug": None,
            "About": "欢迎使用 AI检测系统！",
        },
    )
    # 图标请参考：https://icons.getbootstrap.com/
    pages = {
        "线程管理": {"icon": "bi-toggles", "func": pagecontrol},
        "编辑线程": {"icon": "bi-card-checklist", "func": pagethread},
        "模型管理": {"icon": "bi-robot", "func": pagemodel},
        "人脸注册": {"icon": "bi-person-fill-add", "func": faceregister},
        "人脸检测": {"icon": "bi-person-bounding-box", "func": facerecognition},
        "人脸线程": {"icon": "bi-cpu-fill", "func": facethread},
        "相似检测": {"icon": "bi-rulers", "func": clip_thread},
        "推流检查": {"icon": "bi-camera-reels", "func": pagertsp},
        # "人脸检测结果": {"icon": "bi-file-ruled", "func": face_results_show},
        # "实时测试": {"icon": "bi-router", "func": yolo_callback},
        # "晴雨盘": {"icon": "bi-speedometer", "func": pagechart}
    }

    with st.sidebar:
        st.image("static/banner.png", use_column_width=True)
        st.caption(
            f"""<p align="right">当前版本：2</p>""",
            unsafe_allow_html=True,
        )
        options = list(pages)
        icons = [x["icon"] for x in pages.values()]
        selected_page = option_menu("", options=options, icons=icons, default_index=0)

    if selected_page in pages:
        pages[selected_page]["func"]()

    # hide_streamlit_style = """
    #         <style>
    #         /* 隐藏Streamlit的侧边栏链接 */
    #         [data-testid="stSidebarNavLink"] {
    #             display: none !important;
    #         }
    #         </style>
    #         """
    # st.markdown(hide_streamlit_style, unsafe_allow_html=True)


if __name__ == "__main__":
    # 首先lazy启动app 
    mods.app_initializer.app_startup()
    main()
