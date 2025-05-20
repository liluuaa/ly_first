# coding:utf-8
from ultralytics import YOLO

model = YOLO("yolov5s.pt")

if __name__ == '__main__':
    try:
        results = model.train(data='E:/Py Code/fire/datasets/fire data/data.yaml', epochs=100, batch=4)  # 训练100个epochs
        print("训练完成！")

        success = model.export(format='onnx')
        if success:
            print("模型导出成功。")
        else:
            print("模型导出失败。")

    except Exception as e:
        print(f"训练过程中出错: {e}")
