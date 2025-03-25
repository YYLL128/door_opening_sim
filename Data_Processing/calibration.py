#####################################################################################
# 作用：相机标定。进行摄像头内参和外参矩阵的获取，
# 方法：使用棋盘格进行标定
# 输入：两个摄像头对同一棋盘格的拍摄图组若干（15~20）
# 输出：内参矩阵、外参矩阵、平均重投影误差（一般应小于 0.5像素）
#####################################################################################
import numpy as np
import cv2
import glob
import os

# ==================== 参数配置（按需修改） ====================
CHECKERBOARD = (8, 6)  # 棋盘格内部角点数 (cols-1, rows-1)
SQUARE_SIZE = 0.025  # 棋盘格方格边长（单位：米，需实际测量）
CAM1_ID, CAM2_ID = 0, 1  # 两个相机的设备ID（或视频文件路径）
SAVE_DIR = "calibration_photos"  # 保存标定图像的文件夹
NUM_CAPTURES = 20  # 需要采集的图像对数
SHOW_LIVE = True  # 是否实时显示摄像头画面

# ==================== 准备对象点 ====================
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE

# ==================== 初始化摄像头 ====================
cap1 = cv2.VideoCapture(CAM1_ID)
cap2 = cv2.VideoCapture(CAM2_ID)
assert cap1.isOpened() and cap2.isOpened(), "摄像头打开失败！"

# ==================== 采集标定图像对 ====================
os.makedirs(SAVE_DIR, exist_ok=True)
count = 0

print("按下空格键捕获图像，Esc键退出...")
while count < NUM_CAPTURES:
    # 读取双摄像头帧
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not ret1 or not ret2: break

    # 实时显示
    if SHOW_LIVE:
        display = np.hstack([frame1, frame2])
        cv2.imshow('Dual Camera', display)

    key = cv2.waitKey(1)
    if key == 27:  # Esc键退出
        break
    elif key == 32:  # 空格键捕获
        # 检测棋盘格角点
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        ret1, corners1 = cv2.findChessboardCorners(gray1, CHECKERBOARD, None)
        ret2, corners2 = cv2.findChessboardCorners(gray2, CHECKERBOARD, None)

        if ret1 and ret2:
            # 亚像素优化
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

            # 绘制并保存图像
            cv2.drawChessboardCorners(frame1, CHECKERBOARD, corners1, ret1)
            cv2.drawChessboardCorners(frame2, CHECKERBOARD, corners2, ret2)
            cv2.imwrite(f"{SAVE_DIR}/cam1_{count}.jpg", frame1)
            cv2.imwrite(f"{SAVE_DIR}/cam2_{count}.jpg", frame2)
            print(f"已捕获图像对: {count}")
            count += 1

# 释放摄像头
cap1.release()
cap2.release()
cv2.destroyAllWindows()


# ==================== 标定内参和外参 ====================
def calibrate_single_camera(image_paths, objpoints):
    imgpoints = []
    rvecs = []
    tvecs = []
    for fname in image_paths:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        if ret:
            imgpoints.append(corners)
    # 标定内参并返回外参
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)
    return ret, mtx, dist, rvecs, tvecs


# 加载所有标定图像
cam1_images = sorted(glob.glob(f"{SAVE_DIR}/cam1_*.jpg"))
cam2_images = sorted(glob.glob(f"{SAVE_DIR}/cam2_*.jpg"))
assert len(cam1_images) == len(cam2_images), "图像对数不匹配！"

# 提取所有图像点
objpoints = [objp] * len(cam1_images)  # 所有图像共享同一组对象点
# 标定相机1和相机2，并保存外参
_, mtx1, dist1, rvecs1, tvecs1 = calibrate_single_camera(cam1_images, objpoints)
_, mtx2, dist2, rvecs2, tvecs2 = calibrate_single_camera(cam2_images, objpoints)

# 提取图像点对
imgpoints1, imgpoints2 = [], []
for fname1, fname2 in zip(cam1_images, cam2_images):
    gray1 = cv2.imread(fname1, 0)
    gray2 = cv2.imread(fname2, 0)
    ret1, corners1 = cv2.findChessboardCorners(gray1, CHECKERBOARD, None)
    ret2, corners2 = cv2.findChessboardCorners(gray2, CHECKERBOARD, None)
    if ret1 and ret2:
        imgpoints1.append(corners1)
        imgpoints2.append(corners2)

# 立体标定获取外参
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints1, imgpoints2,
    mtx1, dist1, mtx2, dist2,
    gray1.shape[::-1],
    criteria=criteria,
    flags=cv2.CALIB_FIX_INTRINSIC
)

# ==================== 输出结果 ====================
print("\n====== 相机1内参 ======")
print("mtx1:\n", mtx1)
print("dist1:\n", dist1)

print("\n====== 相机2内参 ======")
print("mtx2:\n", mtx2)
print("dist2:\n", dist2)

print("\n====== 外参 ======")
print("旋转矩阵 R (cam1 -> cam2):\n", R)
print("平移向量 T (cam1 -> cam2):\n", T)

# 保存标定结果
np.savez("stereo_calib.npz",
         mtx1=mtx1, dist1=dist1,
         mtx2=mtx2, dist2=dist2,
         R=R, T=T
         )

# ==================== 验证重投影误差 ====================
mean_error = 0
for i in range(len(objpoints)):
    # 投影到两个相机
    imgpoints1_proj, _ = cv2.projectPoints(objpoints[i], rvecs1[i], tvecs1[i], mtx1, dist1)
    imgpoints2_proj, _ = cv2.projectPoints(objpoints[i], rvecs2[i], tvecs2[i], mtx2, dist2)
    # 计算误差
    error1 = cv2.norm(imgpoints1[i], imgpoints1_proj, cv2.NORM_L2) / len(imgpoints1_proj)
    error2 = cv2.norm(imgpoints2[i], imgpoints2_proj, cv2.NORM_L2) / len(imgpoints2_proj)
    mean_error += (error1 + error2)
print(f"\n平均重投影误差: {mean_error / (2 * len(objpoints)):.4f} 像素")