import av
import os
import cv2
import time
import numpy as np  # 添加numpy库导入


def save_frames_from_video(video_stream_url, output_folder):
    """
    从给定的视频流中提取帧并保存为图片到本地文件夹

    参数:
    video_stream_url (str): 视频流的地址，可以是本地视频文件路径或者网络视频流地址
    output_folder (str): 保存图片的本地文件夹路径
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    container = av.open(video_stream_url)
    frame_count = 0
    prev_frame = None
    threshold = 10000  # 画面变化阈值，可根据实际情况调整
    for frame in container.decode(video=0):
        if frame is None:
            continue
        image = frame.to_ndarray(format="bgr24")
        if prev_frame is None:
            prev_frame = image
            continue
        # 计算图像差分（简单示例，可采用更复杂的算法比如基于特征点匹配等）
        diff = cv2.absdiff(prev_frame, image)
        diff_sum = diff.sum()
        if diff_sum > threshold:
            # 检查色彩空间是否需要转换，这里简单判断是否为RGB格式（实际需更精确判断）
            if image.shape[2] == 3 and image.dtype == np.uint8 and image[0, 0, 0] > image[0, 0, 2]:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if fuzzy_filter(image):
                continue

            image_name = os.path.join(output_folder, f"img_{frame_count}.jpg")
            cv2.imwrite(image_name, image)
            frame_count += 1
            prev_frame = image
            if frame_count >= 10:
                break
    container.close()
    print(f"共保存了{frame_count}张图片到 {output_folder} 文件夹中。")


def fuzzy_filter(frame, threshold=100.0) -> bool:
    """判断图片是否模糊，得分小于阀值被判断为模糊图片.

    Args:
        frame (cv2.typing.MatLike): 源图片
        threshold (float): 得分阀值. Defaults to 100.0.

    Returns:
        bool: 图片是否模糊
    """
    height, width = frame.shape[:2]
    if height < 10 or width < 10:  # 简单设定尺寸阈值，可根据实际调整
        return True
    # 计算图像像素值的整体方差，初步判断是否模糊
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    overall_variance = np.var(gray_frame)
    if overall_variance < 10:  # 设定一个较小的方差阈值，可调整
        return True
    y_start = height // 2  # 使用整除得到中间行
    y_end = height
    img2gray = cv2.cvtColor(frame[y_start:y_end, :], cv2.COLOR_BGR2GRAY)  # 修改此处，去掉.view()方法的使用
    score = cv2.Laplacian(img2gray, cv2.CV_64F).var()
    return score < threshold


def main():
    video_stream_url = "rtsp://admin:xsjsxy12345@192.168.1.12/Streaming/Channels/101"
    output_folder = "output_images_yuanshi"
    save_frames_from_video(video_stream_url, output_folder)

    video_stream_url_daili = "rtsp://192.168.1.206:8554/rtsp/12"
    output_folder_daili = "output_images_daili"
    save_frames_from_video(video_stream_url_daili, output_folder_daili)


if __name__ == "__main__":
    main()
    