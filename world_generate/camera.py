import mujoco_py
import cv2

model = mujoco_py.load_model_from_path("./world/pull_gen3/1555114732_pull_gen3.xml")
sim = mujoco_py.MjSim(model)

from mujoco_py import GlfwContext
GlfwContext(offscreen=True)

camera_names = ["front_cam", "side_cam", "top_cam"]
current = 0  # 当前摄像头索引
width, height = 640, 480

print("按左右箭头切换摄像头，ESC退出")

for _ in range(1000):
    sim.step()

    name = camera_names[current]
    img = sim.render(width, height, camera_name=name)
    img_bgr = img[..., ::-1]
    cv2.imshow("Camera View", img_bgr)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('a'):  # 'a' 切换到前一个摄像头
        current = (current - 1) % len(camera_names)
    elif key == ord('d'):  # 'd' 切换到下一个摄像头
        current = (current + 1) % len(camera_names)

cv2.destroyAllWindows()
