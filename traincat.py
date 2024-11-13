import os
from ultralytics import YOLO

model = YOLO(os.path.join('.', 'runs', 'detect', 'train6', 'weights', 'best.pt'))
results = model.train(data=os.path.join("C:/Users/HP/Documents/GitHub/speciesDetection/data", "data.yaml"), epochs=100)