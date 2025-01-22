from PIL import Image
from mods.utils import *

test_rtsp_url = "rtsp://192.168.0.213:8554/rtsp/185"

im_array = capture_stream_screen(test_rtsp_url)

Image.fromarray(im_array).show()
