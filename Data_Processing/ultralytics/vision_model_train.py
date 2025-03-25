import sys
import os

# 将 ultralytics 添加到 Python 的路径中
sys.path.append(os.path.join(os.path.dirname(__file__), "ultralytics", "ultralytics"))
from ultralytics import YOLO

####################################
# yolov8n.pt（Nano 版，最小、速度最快）
# yolov8s.pt（Small 版，适合嵌入式设备）
# yolov8m.pt（Medium 版，精度更高）
# yolov8l.pt（Large 版，适合服务器端）
# yolov8x.pt（Extra-large 版，最高精度）
####################################

model = YOLO("yolov8n.pt")  # 加载预训练模型

# 训练模型
model.train(data="data.yaml", epochs=50, batch=16, imgsz=640)

####################################
# data="data.yaml" → 指定数据集路径
# epochs=50 → 训练 50 轮
# batch=16 → 每个批次大小 16
# imgsz=640 → 输入图片尺寸 640×640
# 训练完成后，权重文件会保存在 runs/detect/train/weights/best.pt
####################################

model.val()

####################################
# mAP@50：目标检测的精确度
# Precision：模型的 精确率
# Recall：模型的 召回率
####################################