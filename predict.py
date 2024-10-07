import os
#Predict.py untuk mengetes algoritmanya ke video, tapi belum selesai di koding
from ultralytics import YOLO
import cv2

VIDEOS_PATH = os.path.join('.', 'videos', 'dogvids.mp4')

cap = cv2.VideoCapture(VIDEOS_PATH)

