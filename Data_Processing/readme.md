### vision_network

***

#### procedure

- [ ] 相机标定（畸变矫正）：棋盘格图片、python代码 $\rightarrow$ **相机内参**矩阵（K，包括焦距、光心）
  - 两个相机拍摄的若干张==棋盘格图片==
- [ ] **相机外参** ：两个相机之间的平移（T）、旋转（R）变换矩阵
  - 两个相机拍摄==同一棋盘格的若干照片对==（共视标定法）
- [ ] **目标点二维定位**：两张图像中分别获取门把手中心的二维坐标（使用yolo v5？还是自训练神经网络模型？）
  - 暂定使用[yolov8](https://blog.csdn.net/zyw2002/article/details/128732494)
- [ ] **3D坐标**获取：得到[4D齐次坐标](https://zhuanlan.zhihu.com/p/110503121) $\rightarrow$ 除以w得到3D坐标（相机坐标系下的坐标，原点为**左摄像头的光心**）
- [ ] **手眼标定**：摄像头坐标系  $\rightarrow$ 机械臂基坐标系（齐次变换矩阵） $\rightarrow$  机械臂原点坐标下的门把手三维坐标
  - 机械臂==各关节参数==、机械臂==末端的坐标和朝向向量&左摄像头的坐标和朝向向量==

#### details

- [ ] [yolov8模型](https://github.com/ultralytics/ultralytics)（上传的文件中已包含，见`ultralytics`文件夹）

  ```python
  # Clone the ultralytics repository
  git clone https://github.com/ultralytics/ultralytics
  
  # Navigate to the cloned directory
  cd ultralytics
  
  # Install the package in editable mode for development
  pip install -e .
  ```

- [ ] 外参和内参的标定（见`calibration.py`）

- [ ] 3D坐标的获取（见`ultralytics/vision_network.py`）

- [ ] 门把手识别模型的训练（见`ultralytics/vision_model_train.py`）

  - 训练集和测试集：见`ultralytics/dataset`目录下，其中`images`为图像，`label`为`.txt`文件，标注特征框信息

    - 如果 `images/train/001.jpg` 是一张门把手的图片，则它的标签文件 `labels/train/001.txt` 可能如下：

      ```python
      0 0.45 0.67 0.2 0.3
      ```

      其中：

      - `0` **类别ID**（门把手，唯一类别）
      - `0.45` **x_center**（物体中心 x 坐标，相对图像宽度归一化）
      - `0.67` **y_center**（物体中心 y 坐标，相对图像高度归一化）
      - `0.2` **width**（物体宽度，相对图像宽度归一化）
      - `0.3` **height**（物体高度，相对图像高度归一化）

  - `data.yaml`文件标注数据路径信息。

> 目前尚在寻找更适合的门把手特征提取方法，目标点二维定位的模型训练尚待商榷。
