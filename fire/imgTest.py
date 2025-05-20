#coding:utf-8
from ultralytics import YOLO
import cv2

# 所需加载的模型目录
path = 'E:/Py Code/fire/models/exp/weights/best.pt'
# 需要检测的图片地址
img_path = "E:/Py Code/fire/TestFiles/WEB03966.jpg"

# 加载预训练模型
model = YOLO(path, task='detect')

# 检测图片
results = model(img_path)
res = results[0].plot()
cv2.imshow("YOLOv5 Detection", res)
cv2.waitKey(0)