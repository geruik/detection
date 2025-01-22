import cv2
import torch
from PIL import Image
from fight.fightfunction import FightRecognition


fightRecognition = FightRecognition()


def print_similarity(similarity, img_path):
    if similarity > 0.23:
        print("Fight Detected on [  " + img_path + " ]")
    else:
        print("No Fight Detected on [  " + img_path + " ]")


def detect(img_path):
    image = cv2.imread(img_path)
    similarity = fightRecognition.detect(image)
    print_similarity(similarity, img_path)


img_path = "tests/eight_col_carjack.jpg"
detect(img_path)
img_path = "tests/CCTV-from-Liverpool.jpg"
detect(img_path)
