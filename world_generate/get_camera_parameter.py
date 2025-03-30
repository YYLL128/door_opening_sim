import os
import random
import argparse
import cv2
import math
import numpy as np
import mujoco_py
from mujoco_py import GlfwContext


def find_xml_files(root_folder):
    """
    在给定的 root_folder 下，查找所有后缀为 .xml 的文件并返回它们的完整路径列表。
    """
    xml_paths = []
    for foldername, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith(".xml"):
                full_path = os.path.join(foldername, filename)
                xml_paths.append(full_path)
    return xml_paths


def print_camera_parameters(model, cam_name, width, height):
    """
    根据相机名称打印相机的内参和外参。

    参数:
      model   : Mujoco 模型对象
      cam_name: 相机名称（字符串），例如 "front_cam"
      width   : 渲染图像宽度（像素）
      height  : 渲染图像高度（像素）
    """
    # 从模型中的 camera_names 列表中查找相机的索引 ID
    try:
        cam_id = model.camera_names.index(cam_name)
    except ValueError:
        print(f"相机名称 {cam_name} 不存在于模型中。")
        return

    # --------------------- 获取外参 ---------------------
    # 获取相机在世界坐标系中的位置（平移向量）
    cam_pos = model.cam_pos[cam_id]

    # 从 model.cam_quat 中获取相机的四元数
    # 注意：model.cam_quat 是一个二维数组，每一行对应一个相机
    quat = model.cam_quat[cam_id]

    # 使用 mujoco_py 提供的函数将四元数转换为旋转矩阵
    # 创建一个长度为 9 的空数组，用于存储平铺的 3x3 旋转矩阵
    R_flat = np.empty(9, dtype=np.float64)
    mujoco_py.functions.mju_quat2Mat(R_flat, quat)
    # 将平铺数组转换为 3x3 矩阵
    R = R_flat.reshape(3, 3)

    print(f"相机 {cam_name} 的外参：")
    print("位置 (translation):", cam_pos)
    print("旋转矩阵 (camera->world，从四元数转换):\n", R)
    # 若需要将外参转换为 world->camera，则旋转矩阵为 R 的转置，
    # 平移向量 t = -R.T * cam_pos
    R_world2cam = R.T
    t_world2cam = -R_world2cam.dot(cam_pos)
    print("转换为 world->camera 的旋转矩阵 R:\n", R_world2cam)
    print("相机坐标系下的平移向量 t (即 -R.T * cam_pos):\n", t_world2cam)
    print("------------------------------------------------")

    # --------------------- 获取内参 ---------------------
    # 从模型中获取垂直视场角 fovy（单位：度）
    fovy_deg = model.cam_fovy[cam_id]
    fovy_rad = math.radians(fovy_deg)
    # 根据公式计算焦距：
    # fy = height / (2 * tan(fovy/2))，而 fx = fy * (width / height)
    fy = height / (2 * math.tan(fovy_rad / 2))
    fx = fy * (width / height)
    # 主点通常设为图像中心
    cx = width / 2
    cy = height / 2
    # 构造内参矩阵 K
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])
    print(f"相机 {cam_name} 的内参：")
    print("焦距 fx, fy:", fx, fy)
    print("主点 cx, cy:", cx, cy)
    print("内参矩阵 K:\n", K)
    print("================================================")


def run_simulation(xml_path):
    """
    运行模拟，显示摄像机图像，同时打印摄像机的内参和外参。
    """
    # 初始化 offscreen OpenGL 上下文
    GlfwContext(offscreen=True)

    # 加载 XML 模型并创建模拟对象
    model = mujoco_py.load_model_from_path(xml_path)
    sim = mujoco_py.MjSim(model)

    # 定义要使用的摄像机名称列表（需与 XML 文件中定义的摄像机名称一致）
    camera_names = ["front_cam", "side_cam", "top_cam"]
    current = 0  # 当前选中的摄像机索引
    width, height = 640, 480  # 渲染图像的宽和高

    # 提示当前使用的摄像机视角
    print(f"使用摄像机视角：{camera_names[current]}（按 a/d 键切换，ESC 键退出）")

    # 预先打印所有指定摄像机的内参和外参
    for cam_name in camera_names:
        print_camera_parameters(model, cam_name, width, height)

    # 模拟主循环
    for _ in range(1000):
        # 进行一步模拟更新
        sim.step()

        # 根据当前选中的摄像机名称渲染图像
        cam_name = camera_names[current]
        img = sim.render(width, height, camera_name=cam_name)
        # 将图像从 RGB 转换为 BGR 格式（OpenCV 默认使用 BGR）
        img_bgr = img[..., ::-1]
        cv2.imshow("Camera View", img_bgr)

        # 监听键盘事件：ESC 退出，a/d 切换摄像机视角
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC 键
            break
        elif key == ord('a'):
            current = (current - 1) % len(camera_names)
        elif key == ord('d'):
            current = (current + 1) % len(camera_names)

    # 退出时关闭所有 OpenCV 窗口
    cv2.destroyAllWindows()


def main():
    """
    主函数：解析命令行参数，选择 XML 模型文件，并启动模拟。
    """
    parser = argparse.ArgumentParser(
        description="使用 mujoco_py 显示摄像头图像，并获取摄像机的内参和外参。"
    )
    parser.add_argument(
        "--xml",
        type=str,
        default=None,
        help="要运行的 XML 文件路径（相对于 world 目录或绝对路径）。若不指定则随机运行。"
    )
    args = parser.parse_args()

    # 定义存放 XML 模型文件的目录（假设在脚本同目录下的 world 文件夹）
    world_folder = os.path.join(os.path.dirname(__file__), "world")
    all_xml_files = find_xml_files(world_folder)

    if not all_xml_files:
        print("在 world 文件夹下未找到任何 .xml 文件，请检查路径。")
        return

    chosen_xml = None
    if args.xml is not None:
        possible_path = os.path.join(world_folder, args.xml)
        if os.path.isfile(args.xml):
            chosen_xml = os.path.abspath(args.xml)
        elif os.path.isfile(possible_path):
            chosen_xml = os.path.abspath(possible_path)
        else:
            print(f"无法找到指定的文件: {args.xml}")
            return
    else:
        # 随机选择一个 XML 模型文件
        chosen_xml = random.choice(all_xml_files)

    print(f"即将运行模型: {chosen_xml}")
    run_simulation(chosen_xml)


if __name__ == "__main__":
    main()
