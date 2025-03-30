import os
import random
import argparse
import cv2
import mujoco_py
import numpy as np
import time
from mujoco_py import GlfwContext


def find_xml_files(root_folder):
    xml_paths = []
    for foldername, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith(".xml"):
                full_path = os.path.join(foldername, filename)
                xml_paths.append(full_path)
    print(f"[DEBUG] Found {len(xml_paths)} xml files in {root_folder}")
    return xml_paths


def create_joint_trackbars(window_name, n_joints):
    """
    为每个关节创建一个滑块，范围为0~200，初始值为100（对应0速度）。
    滑块值转换为速度：(value - 100) / 100 得到 -1.0 ~ 1.0 的速度指令
    """
    print(f"[DEBUG] Creating joint trackbars in window '{window_name}' for {n_joints} joints")
    cv2.namedWindow(window_name)
    for i in range(n_joints):
        trackbar_name = f"Joint {i + 1}"
        cv2.createTrackbar(trackbar_name, window_name, 100, 200, lambda x: None)
        print(f"[DEBUG] Created trackbar '{trackbar_name}' with initial value 100")


def get_joint_velocities(window_name, n_joints):
    """
    从滑块中获取当前关节速度指令，返回一个长度为 n_joints 的 numpy 数组
    """
    velocities = np.zeros(n_joints, dtype=np.float32)
    for i in range(n_joints):
        trackbar_name = f"Joint {i + 1}"
        pos = cv2.getTrackbarPos(trackbar_name, window_name)
        velocities[i] = (pos - 100) / 100.0  # 默认中间值100对应0速度
    print(f"[DEBUG] Current joint velocities: {velocities}")
    return velocities


def run_simulation(xml_path, control_gain=100.0):
    print(f"[DEBUG] Initializing GlfwContext with offscreen=True")
    GlfwContext(offscreen=True)
    try:
        model = mujoco_py.load_model_from_path(xml_path)
        print(f"[DEBUG] Loaded model from {xml_path}")
    except Exception as e:
        print(f"[ERROR] Failed to load model from {xml_path}: {e}")
        return

    sim = mujoco_py.MjSim(model)
    try:
        sim.forward()
        print("[DEBUG] sim.forward() called successfully")
    except Exception as e:
        print(f"[ERROR] sim.forward() failed: {e}")

    print(f"[DEBUG] Model timestep: {model.opt.timestep}")

    # 定义摄像机名称
    camera_names = ["front_cam", "side_cam", "top_cam"]
    current_camera = 0
    width, height = 640, 480
    print(f"[DEBUG] Using camera view: {camera_names[current_camera]}")

    try:
        viewer = mujoco_py.MjRenderContextOffscreen(sim)
        print("[DEBUG] Created MjRenderContextOffscreen successfully")
    except Exception as e:
        print(f"[ERROR] Failed to create viewer: {e}")
        return

    # 创建关节控制的窗口和滑块（控制前 7 个关节）
    joint_control_window = "Joint Control"
    n_joints = 7
    create_joint_trackbars(joint_control_window, n_joints)

    # 初始化关节目标位置（控制输入）
    joint_targets = np.copy(sim.data.ctrl[:n_joints])
    print(f"[DEBUG] Initial joint_targets: {joint_targets}")

    cv2.namedWindow("Camera View")

    for step in range(10000):
        start_time = time.time()

        try:
            joint_velocities = get_joint_velocities(joint_control_window, n_joints)
            # 通过积分更新关节目标位置，加上控制增益使运动更明显
            joint_targets += joint_velocities * model.opt.timestep * control_gain
            sim.data.ctrl[:n_joints] = joint_targets
        except Exception as e:
            print(f"[ERROR] Error updating joint controls at step {step}: {e}")

        try:
            sim.step()
        except Exception as e:
            print(f"[ERROR] sim.step() failed at step {step}: {e}")
            break

        try:
            # 根据当前摄像机视角进行渲染
            cam_name = camera_names[current_camera]
            cam_id = sim.model.camera_name2id(cam_name)
            viewer.cam.fixedcamid = cam_id
            viewer.cam.type = 2  # 固定摄像机模式
            viewer.render(width, height)
            img = viewer.read_pixels(width, height, depth=False)
            img_bgr = img[..., ::-1]  # 转换为 BGR 格式
            cv2.imshow("Camera View", img_bgr)
        except Exception as e:
            print(f"[ERROR] Rendering error at step {step}: {e}")

        elapsed = time.time() - start_time
        if step % 100 == 0 and step > 0:
            print(f"[DEBUG] Step {step}, elapsed time: {elapsed:.5f} sec, joint_targets: {joint_targets}")

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC 退出
            print("[DEBUG] ESC pressed. Exiting simulation loop.")
            break
        elif key == ord('a'):
            current_camera = (current_camera - 1) % len(camera_names)
            print(f"[DEBUG] Switched to camera view: {camera_names[current_camera]}")
        elif key == ord('d'):
            current_camera = (current_camera + 1) % len(camera_names)
            print(f"[DEBUG] Switched to camera view: {camera_names[current_camera]}")

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="使用 mujoco_py 随机选择 xml 文件运行模拟，并通过滑块手动调整机械臂关节速度。"
    )
    parser.add_argument(
        "--xml",
        type=str,
        default=None,
        help="要运行的 xml 文件路径（相对于 world 目录或绝对路径）。若不指定则随机运行。"
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=100.0,
        help="控制增益，用于放大每步积分的效果（默认100.0）"
    )
    args = parser.parse_args()

    world_folder = os.path.join(os.path.dirname(__file__), "world")
    all_xml_files = find_xml_files(world_folder)

    if not all_xml_files:
        print("[ERROR] 在 world 文件夹下未找到任何 .xml 文件，请检查路径。")
        return

    if args.xml is not None:
        possible_path = os.path.join(world_folder, args.xml)
        if os.path.isfile(args.xml):
            chosen_xml = os.path.abspath(args.xml)
        elif os.path.isfile(possible_path):
            chosen_xml = os.path.abspath(possible_path)
        else:
            print(f"[ERROR] 无法找到指定的文件: {args.xml}")
            return
    else:
        chosen_xml = random.choice(all_xml_files)

    print(f"[DEBUG] 即将运行模型: {chosen_xml}")
    run_simulation(chosen_xml, control_gain=args.gain)


if __name__ == "__main__":
    main()
