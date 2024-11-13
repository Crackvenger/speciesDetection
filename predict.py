import os
#Predict.py untuk mengetes algoritmanya ke video, tapi belum selesai di koding
from ultralytics import YOLO
import cv2
import numpy as np

VIDEOS_PATH = os.path.join('.', 'videos', 'dogvids.mp4')
VIDEOS_PATH_OUT = '{}_out.mp4'.format(VIDEOS_PATH)
cap = cv2.VideoCapture(VIDEOS_PATH)

ret, frame = cap.read()

H, W, _ = frame.shape
out = cv2.VideoWriter(VIDEOS_PATH_OUT, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model_path = os.path.join('.', 'runs', 'detect', 'train4', 'weights', 'last.pt')


# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.5

while ret:

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()