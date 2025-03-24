import os
import random
import argparse
import cv2
import mujoco_py
import numpy as np
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

def run_simulation(xml_path):
    GlfwContext(offscreen=True)

    model = mujoco_py.load_model_from_path(xml_path)
    sim = mujoco_py.MjSim(model)

    # 获取 sensor 的 ID
    try:
        right_idx = model.sensor_name2id("right_tip_force")
        left_idx = model.sensor_name2id("left_tip_force")
    except KeyError as e:
        print(f"找不到传感器名称：{e}")
        return

    camera_names = ["front_cam", "side_cam", "top_cam"]
    current = 0
    width, height = 640, 480

    print(f"使用摄像机视角：{camera_names[current]}（按 a/d 切换，ESC 退出）")

    viewer = mujoco_py.MjRenderContextOffscreen(sim)

    for step in range(1000):
        sim.step()

        # 渲染当前视角图像
        cam_name = camera_names[current]
        cam_id = sim.model.camera_name2id(cam_name)
        viewer.cam.fixedcamid = cam_id
        viewer.cam.type = 2  # 2 表示使用固定相机
        viewer.render(width, height)
        img = viewer.read_pixels(width, height, depth=False)
        img_bgr = img[..., ::-1]  # 上下翻转 + RGB 转 BGR

        # 显示图像
        cv2.imshow("Camera View", img_bgr)

        # 读取左右指尖的力数据
        right_force = sim.data.sensordata[right_idx : right_idx + 3]
        left_force = sim.data.sensordata[left_idx : left_idx + 3]

        print(f"[{step:04d}]")
        print("  ➤ Right Tip Force:", np.round(right_force, 4))
        print("  ➤ Left Tip Force :", np.round(left_force, 4))
        print("-" * 40)

        # 键盘控制
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('a'):
            current = (current - 1) % len(camera_names)
        elif key == ord('d'):
            current = (current + 1) % len(camera_names)

    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(
        description="使用 mujoco_py 显示摄像头图像并读取指尖受力，可指定或随机选择一个 XML 模型文件。"
    )
    parser.add_argument(
        "--xml",
        type=str,
        default=None,
        help="要运行的xml文件路径（相对于world目录或绝对路径）。若不指定则随机运行。"
    )
    args = parser.parse_args()

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
        chosen_xml = random.choice(all_xml_files)

    print(f"即将运行模型: {chosen_xml}")
    run_simulation(chosen_xml)

if __name__ == "__main__":
    main()
