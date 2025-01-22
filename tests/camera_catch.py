import multiprocessing
import pandas as pd
from ultralytics import YOLO
from mods.yolo_loaders import *
import threads_config

__CAMERA_LIST_PATH = "tests/camera_list.json"




def startProcess(photos_info):
    camera_process = multiprocessing.Process(
            target=camera_catch_start,
            args=(photos_info,)
        )
    camera_process.start()
    detection_process_1 = multiprocessing.Process(
            target=detection_start,
            args=(1,photos_info)
        )
    detection_process_1.start();
    detection_process_2 = multiprocessing.Process(
            target=detection_start,
            args=(2,photos_info)
        )
    detection_process_2.start();
    while True:
        time.sleep(10)




def camera_catch_start(photos_info):
    config = threads_config.__base_load_config(__CAMERA_LIST_PATH)
    items = dict(config.items())
    streams = items.get("cameras")["STREAMS"]
    
    rtsp_url_sn_map = build_rtsp_sn_map(streams)
    rtsp_urls = [x["RTSP_URL"] for x in streams]
    dataset=MyLoadStreams(sources=make_streams_temp_file(rtsp_urls), grab_interval=0.5)
    for batch in dataset:
            paths, im0s, s = batch
            n = len(im0s)
            formatted_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            for i in range(n):
                im0 = im0s[i]
                path = paths[i]
                if path in rtsp_url_sn_map:
                    camera_sn = rtsp_url_sn_map[path]
                else:
                    continue
                photos_info[camera_sn] = {
                    'image': im0,
                    'timestamp': formatted_time
                }
    
    

def detection_start(n,photos_info):
    sleep_time = 2
    config = threads_config.load_config()
    items = dict(config.items())
    detect_args = items.get("人员聚集检测")
    inference = detect_args["INFERENCE_ARGS"]
    inference["verbose"] = False  # 禁止啰嗦模式，以免控制台失控
    if "classes" in inference:
        classes = list(map(int, inference["classes"]))
        if len(classes) > 0:
            inference["classes"] = classes
        else:
            del inference["classes"]
    yolo_model = YOLO(detect_args["MODEL_PATH"])
    streams = detect_args["STREAMS"]
    print(f"进程{n}开始检测")
    while True:
        for stream in streams:
            if stream["CAMERA_SN"] not in photos_info:
                continue
            photo_info = photos_info[stream["CAMERA_SN"]]
            if abs(pd.Timestamp.now() - pd.Timestamp(photo_info["timestamp"])).total_seconds() > (2 * sleep_time):
                continue
            image = photo_info["image"]
            results = yolo_model(image, **inference)
            for result in results:
                print(f"进程{n}进行检测，识别到{len(result)}个目标")
            
        time.sleep(sleep_time)

if __name__ == '__main__':
    photos_info = multiprocessing.Manager().dict()  
    startProcess(photos_info)
