import cv2
import numpy as np
import streamlit as st
import face.facefunction


def facerecognition():
    st.header("👤 人脸识别", divider=True)

    uploaded_file = st.file_uploader("上传检测图片", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        # st.image(img, caption="上传的图片", channels="BGR")
        # 人脸识别
        frg = face.facefunction.getRecognizaObject(None)
        results = frg.recognition_advance(img,register=False)

        for result in results:
            st.write("识别结果：{}".format(result["user_name"]))
            st.write("距离值：{}".format(result["dist"]))

            if result["id"] is not None:
                face_image = frg.get_face_img(result["id"])
                if face_image is not None:
                    st.image(face_image, caption="数据库中匹配的人脸",channels="BGR")

            face_img = img[
                result["bbox"][1] : result["bbox"][3],
                result["bbox"][0] : result["bbox"][2],
            ]
            st.image(face_img, caption="检测到的人脸",channels="BGR")
