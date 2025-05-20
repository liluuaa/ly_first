#coding:utf-8
import cv2
from ultralytics import YOLO

# 所需加载的模型目录
path = 'models/best.pt'
# 需要检测的图片地址
video_path = "E:/Py Code/fire/datasets/fire data/test/images/727955741-1-208.mp4"

model = YOLO(path)
cap = cv2.VideoCapture(video_path)
while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        cv2.imshow("YOLOv5 Inference", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()