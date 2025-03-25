#####################################################################################
# 作用：处理两份图像，得到门把手中心相对于机械臂基坐标的三维坐标。
# 方法：使用棋盘格进行标定
# 输入：两个摄像头对于门的把手的拍摄图像
# 输出：门把手中心相对于机械臂基坐标的三维坐标
#####################################################################################
import cv2
import sys
import torch
import numpy as np
import os
# 将 ultralytics 添加到 Python 的路径中
sys.path.append(os.path.join(os.path.dirname(__file__), "ultralytics", "ultralytics"))
from ultralytics import YOLO

# 加载微调后的YOLOv8模型
model = YOLO("runs/detect/train/weights/best.pt")  # 加载训练好的模型

def detect_handle(image):
    results = model(image)
    detections = results.pandas().xyxy[0]
    # 筛选置信度最高的门把手检测结果
    handle = detections[detections['name'] == 'door_handle']
    if handle.empty:
        return None
    x_center = (handle.xmin.values[0] + handle.xmax.values[0]) / 2
    y_center = (handle.ymin.values[0] + handle.ymax.values[0]) / 2
    return (x_center, y_center)

# 读取左右摄像头图像（图像传输和读取方式可能需要更改）
left_img = cv2.imread('left_image.jpg')
right_img = cv2.imread('right_image.jpg')

left_point = detect_handle(left_img)
right_point = detect_handle(right_img)

if left_point is None or right_point is None:
    raise Exception("门把手未在图像中检测到")

#################################################
### edit below ##################################
# 相机参数（示例，需替换为实际标定结果）
K_left = np.array([[fx_left, 0, cx_left],
                  [0, fy_left, cy_left],
                  [0, 0, 1]])
K_right = np.array([[fx_right, 0, cx_right],
                   [0, fy_right, cy_right],
                   [0, 0, 1]])
R = np.array([[r11, r12, r13],  # 右相机相对于左的旋转矩阵
              [r21, r22, r23],
              [r31, r32, r33]])
T = np.array([tx, ty, tz])      # 平移向量
### edit above ##################################
#################################################

# 构建投影矩阵
P_left = K_left @ np.hstack((np.eye(3), np.zeros((3, 1))))
P_right = K_right @ np.hstack((R, T.reshape(3, 1)))

# 转换为cv2需要的格式
points_left = np.array([[left_point]], dtype=np.float32).T
points_right = np.array([[right_point]], dtype=np.float32).T

# 三角化三维点
point_4d = cv2.triangulatePoints(P_left, P_right, points_left, points_right)
point_3d = (point_4d[:3] / point_4d[3]).flatten()  # 左相机坐标系下的坐标

#################################################
### edit below ##################################
# 手眼标定矩阵（示例）
T_eye_to_arm = np.array([
    [r11_arm, r12_arm, r13_arm, tx_arm],
    [r21_arm, r22_arm, r23_arm, ty_arm],
    [r31_arm, r32_arm, r33_arm, tz_arm],
    [0, 0, 0, 1]
])
### edit above ##################################
#################################################

# 坐标转换
point_3d_homo = np.append(point_3d, 1)
arm_coord = T_eye_to_arm @ point_3d_homo
x, y, z = arm_coord[:3]

print(f"门把手在机械臂坐标系中的位置: X={x:.2f}, Y={y:.2f}, Z={z:.2f} 毫米")