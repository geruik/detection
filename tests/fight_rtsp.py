"""
燕进提供的原代码
"""
import threading
import time
import cv2
import torch
import numpy as np
## 不用tokenizer，只有这几个了。。
from fight.transforms import Resize, CenterCrop, ToTensor, Normalize
from fight.clip_loader import CLIPVisionModelWithProjection
from PIL import Image
import logging
logging.basicConfig(filename='app2.log', level=logging.INFO, format='%(asctime)s - %(message)s')

class VideoStream:
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.frame = None
        self.stopped = False

    def start(self):
        threading.Thread(target=self._update, daemon=True).start()
        return self

    def _update(self):
        cap = cv2.VideoCapture(self.rtsp_url)
        if not cap.isOpened():
            print(f"错误：线程无法打开RTSP视频流。")
            return
        while not self.stopped:
            ret, frame = cap.read()
            if not ret:
                print(f"错误：线程 无法读取视频流。5s后重启Videocapture")
                time.sleep(5)
                cap = cv2.VideoCapture(self.rtsp_url)
                continue 
            if ret:
                self.frame = frame
        cap.release()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True




#这里需要修改！！！
def image2tensor(images, size=224):
    def preprocess(size, img):
        return Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))(
            ToTensor()(
                CenterCrop(size)(
                    Resize(size)(img)
                ).convert("RGB")
            )
        )
    processed_images = [preprocess(size, Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))) for img in images]
    return torch.stack(processed_images)

## 不用tokenize，加载模型直接加载bin,可以把其他文件删掉了。。。。。
model = CLIPVisionModelWithProjection("default_models/pytorch_model.bin")
fight_detect =  torch.load('default_models/fight.pt') 
rtsp = "rtsp://192.168.0.213:8515/mystream"
images_count=15
sleepy = 0.5
video_stream = VideoStream(rtsp).start()


while True:
    images = []
    for _ in range(images_count):
        frame = video_stream.read()
        time.sleep(sleepy)
        if frame is not None:
            images.append(frame)
    image_tensors = image2tensor(images)

    visual_output = model(image_tensors)
     #这里修改这个visual_output.image_embeds
    visual_output = visual_output.image_embeds
    visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
    visual_output = torch.mean(visual_output, dim=0)
    visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
    similarity = torch.matmul(visual_output, fight_detect.t())
    final_score = similarity.cpu().detach().numpy()[0]
    
    current_time = time.strftime("%Y%m%d-%H%M%S")
    logging.info(f"{final_score}")
    print(current_time,final_score)
    if final_score > 0.23:
        print('figtttbbbt')
        height, width = images[0].shape[:2]
        out = cv2.VideoWriter(f"{current_time}_{final_score:.2f}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 1, (width, height))
        for img in images:
            cv2.putText(img, 'VIOLENCE Detected', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
            out.write(img)
    continue