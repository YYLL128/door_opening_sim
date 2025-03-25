## README

### WORLD_GENERATOR

#### /door

存储三种不同的门把手，分别为lever_knob, pull_knob, roundknob, 在此处被world_generate调用

#### /mjcf

在生成世界时，代码中需要调用的库

#### /robot

存储机器人信息，其中正在使用的为gen3_lite.xml与gripper.xml ，他们对应的3D建模文件均存储在gen3_lite_mesh中

#### /world

经过world_generator生成的世界存储在这里

---

#### 代码部分

**world_generator**：世界生成器

**run_world**：世界运行器，在mujoco_py中运行

**sensor2.py**：包含三种视角的opencv返回图像，以及力传感器信息和力传感器的频率信息，运行时将随即运行一个世界，若想运行指定世界则需要在运行时指定具体世界路径