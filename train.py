import os

from ultralytics import YOLO

model = YOLO("yolov8n.pt")
#imagenya susah untuk di push,jadi download dari sini https://universe.roboflow.com/dogbreeddetection/actual-dog-breed-detection
results = model.train(data=os.path.join("C:/Users/HP/Documents/GitHub/speciesDetection/data", "data.yaml"), epochs=20)