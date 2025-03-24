#!/usr/bin/python3
# 以上行指定使用 Python 3 解释器

from mjcf import elements as e
# 从 mjcf 库中导入 elements 模块并简写为 e
from mjcf.elements import visual as v
# 从 mjcf.elements 中导入 visual 模块并简写为 v
from random import randrange
# 从 Python 标准库的 random 模块中导入 randrange 方法，用于生成随机数
from tqdm import tqdm
# 从 tqdm 库中导入 tqdm，用于显示进度条
import argparse
# 导入 argparse 库，用于命令行参数解析
import os
# 导入 os 模块，用于文件和路径操作
import json


# 导入 json 模块，用于读取和解析 JSON 文件

def main(dataset_file, args, knob_type):
    """
    main 函数：
    - 接收单个门把手数据文件名、命令行参数、门把手类型
    - 生成对应的 MJCF（MuJoCo）XML 仿真世界文件
    """
    # 构造路径：handle_folder 是输入门把手的文件夹，stl_path 是网格文件所在路径（未在本函数内使用）
    handle_folder = os.path.join(args.input_dirname.format(knob_type), dataset_file)
    stl_path = os.path.join('door/{}knobs'.format(knob_type), dataset_file)

    # 打开并读取 info.json，以获取当前门把手的额外信息（model_count 等）
    with open(os.path.join(handle_folder, 'info.json'), 'r') as f:
        params_dict = json.load(f)

    # 定义各部分构件数量
    door_parts_n = 1  # 门部件数量（示例中只有一块大门板）
    frame_parts_n = 6  # 门框部件数量
    wall_parts_n = 3  # 墙体部件数量
    knob_parts_n = params_dict['model_count']  # 根据 info.json 读取门把手部件数量（多个零件组合成一个把手）

    # 固定使用 gen3.xml 作为机械臂的 include 文件
    robot_include_file = "../../robot/gen3_lite.xml"

    ###################### 随机化参数 ######################
    # 随机生成灯光属性
    light_n = randrange(2, 6)  # 随机生成灯光数量 2~5
    light_property_list = []
    for i in range(light_n):
        # 随机灯光漫反射系数、位置和方向
        light_property = dict(
            light_diffuse=[randrange(9, 11) / 10, randrange(9, 11) / 10, randrange(9, 11) / 10],
            light_pos=[randrange(0, 500) / 100.0, randrange(-500, 500) / 100.0, randrange(300, 700) / 100.0],
            light_dir=[randrange(-50, 50) / 100.0, randrange(-50, 50) / 100.0, randrange(-50, -25) / 100.0]
        )
        light_property_list.append(light_property)

    # 相机属性
    camera1_pos = [0.99, 0.5, 1.0]  # 相机1的位置
    camera1_ori = [0.0, 1.57, 1.57]  # 相机1的欧拉角
    camera2_pos = [0.5, 0.0, 1.99]  # 相机2的位置
    camera2_ori = [0, 0, 0]  # 相机2的欧拉角
    camera_fieldview = 60  # 相机视场角 (FOV)
    camera_randomization = False  # 是否开启相机随机化

    # 若开启相机随机化，对上述相机位置和角度加随机扰动
    if camera_randomization:
        for i in range(len(camera1_pos)):
            camera1_pos[i] += randrange(-15, 15) / 1000
            camera2_pos[i] += randrange(-15, 15) / 1000
        for i in range(len(camera1_ori)):
            camera1_ori[i] += randrange(-17, 17) / 1000
            camera2_ori[i] += randrange(-17, 17) / 1000
        camera_fieldview_noise = [randrange(-100, 100) / 100, randrange(-100, 100) / 100]
    else:
        camera_fieldview_noise = [0, 0]

    # 收集两台相机的位置和欧拉角
    camera_poses = [camera1_pos, camera2_pos]
    camera_ories = [camera1_ori, camera2_ori]

    # 材质名称列表
    material_name_list = ['Paint', 'Wood', 'Carpet', 'Metal']
    # 随机化不同材质的高光(shininess)与镜面(specular)属性
    paint_shininess = randrange(1, 50) / 100.0
    paint_specular = randrange(1, 50) / 100.0
    wood_shininess = randrange(1, 20) / 100.0
    wood_specular = randrange(1, 20) / 100.0
    carpet_shininess = randrange(1, 5) / 100.0
    carpet_specular = randrange(1, 5) / 100.0
    metal_shininess = randrange(80, 100) / 100.0
    metal_specular = randrange(80, 100) / 100.0

    # 地面颜色随机化（rgb1、rgb2 用于渐变）
    floor_rgb1 = [randrange(0, 50) / 100.0, randrange(0, 50) / 100.0, randrange(0, 50) / 100.0]
    floor_rgb2 = [randrange(0, 50) / 100.0, randrange(0, 50) / 100.0, randrange(0, 50) / 100.0]

    # 墙面属性：颜色、位置、材质等
    wall_rgba = [randrange(1, 85) / 100.0, randrange(1, 85) / 100.0, randrange(1, 85) / 100.0, 1.0]
    wall_location = [0.0, 0.0, 0.0]
    wall_material = material_name_list[randrange(0, 3)]

    # 门框属性
    frame_rgba = [randrange(70, 85) / 100.0, randrange(70, 85) / 100.0, randrange(70, 85) / 100.0, 1.0]
    frame_material = material_name_list[randrange(0, 3)]

    # 门框关节（门铰链）的阻尼、弹簧和摩擦损失
    door_frame_damper = randrange(10, 20) / 10.0
    door_frame_spring = randrange(10, 20) / 100.0
    door_frame_frictionloss = randrange(0, 1)

    # 门铰链位置（左铰链、右铰链）和开门方向（推、拉）
    hinge_loc = "righthinge"
    opendir = "pull"

    # 若使用左铰链，可在此调整相机位置，此处保持原样以示例
    if hinge_loc == "lefthinge":
        camera_poses[0] = [0.99, 0.5, 1.0]

    # 门属性的随机化：高度、宽度、厚度等
    door_height = randrange(2000, 2500) / 1000.0  # 2.0~2.5 m
    door_width = 1  # 固定设为 1 m
    door_thickness = randrange(20, 30) / 1000.0  # 0.02~0.03 m
    knob_height = 1  # 把手在门上的高度（与地面距离）为 1 m
    knob_horizontal_location_ratio = 0.15  # 把手在门宽方向的位置比例
    # 门的质量计算：近似（体积 x 密度(此处固定为300)）
    door_mass = door_height * door_width * door_thickness * 300
    # 门表面随机颜色
    door_rgba = [randrange(1, 100) / 100.0, randrange(1, 100) / 100.0, randrange(1, 100) / 100.0, 1.0]
    # 门材质随机
    door_material = material_name_list[randrange(0, 3)]

    # 门和门把手的关节属性（如把手旋转的阻尼、弹簧、摩擦、旋转范围）
    knob_door_damper = randrange(100, 200) / 100.0
    knob_door_spring = randrange(100, 150) / 100.0
    knob_door_frictionloss = randrange(0, 1)
    knob_rot_range = randrange(75, 80) * 3.14 / 180  # 75~80度，转换为弧度

    # 根据门铰链侧，确定把手的初始位置、旋转角等
    if hinge_loc == "righthinge":
        doorknob_pos = [0, 0, 0]
        knob_euler = [-1.57, 1.57, 0]
    elif hinge_loc == "lefthinge":
        doorknob_pos = [0, (0.5 - knob_horizontal_location_ratio) * door_width * 2, 0]
        knob_euler = [1.57, 1.57, 0]
    else:
        raise Exception("door direction undefined")

    # 门把手自身属性：质量、颜色、表面摩擦等
    knob_mass = randrange(1, 2)
    knob_rgba = [randrange(1, 100) / 100.0, randrange(1, 100) / 100.0, randrange(1, 100) / 100.0, 1.0]
    knob_surface_friction = [randrange(50, 100) / 100.0, randrange(1, 5) / 1000.0, randrange(1, 5) / 1000.0]
    doorknob_material = material_name_list[randrange(0, 3)]

    # 机械臂关节阻尼参数
    robot_damping = randrange(9, 11) / 100

    ###################### 常量参数 ######################
    # 重力向量
    gravity_vector = [0, 0, 0]
    # 相机数量
    camera_n = 2

    # 墙体与世界之间的关节/连接属性（此处基本设为非常大的刚度或阻尼，表示不可动）
    wall_world_armature = 0.0001
    wall_world_damper = 100000000
    wall_world_spring = 1000
    wall_world_frictionloss = 0
    wall_world_joint_pos = [0, 0, 0]

    # 门框与墙体的关节/连接属性（同理，设为几乎不可动）
    frame_wall_armature = 0.0001
    frame_wall_damper = 100000000
    frame_wall_spring = 1000
    frame_wall_frictionloss = 0
    frame_wall_joint_pos = [0, 0, 0]

    # 门与门框的关节属性
    door_frame_armature = 0.0001
    door_frame_limited = True  # 门的转动范围是否有限制

    # 根据铰链位置和开门方向确定门转动的范围
    if hinge_loc == "lefthinge":
        if opendir == "push":
            door_frame_range = [-0.00, 1.57]
        else:
            door_frame_range = [-1.57, 0.00]
    else:
        if opendir == "push":
            door_frame_range = [-1.57, 0.00]
        else:
            door_frame_range = [-0.00, 1.57]

    # 若是右铰链，把手位置在门板右侧
    if hinge_loc == "righthinge":
        door_frame_joint_pos = [0, door_width - door_width * knob_horizontal_location_ratio, 0]
    else:
        door_frame_joint_pos = [0, -door_width * knob_horizontal_location_ratio, 0]

    door_front_frame = True  # 门是否安装在门框前沿处（影响墙体位置的计算）

    # 计算门的惯性矩：矩形块三轴惯性（近似分布）
    door_diaginertia = [
        door_mass / 12.0 * (door_height ** 2 + door_width ** 2),
        door_mass / 12.0 * (door_height ** 2 + door_thickness ** 2),
        door_mass / 12.0 * (door_width ** 2 + door_thickness ** 2)
    ]
    door_euler = [0, 0, 0]  # 门的欧拉角
    door_pos = [0, 0, 0]  # 门在父级坐标系下的位置

    # 门框几何尺寸与位置
    frame_width = 100 / 1000.0
    overlap_door_frame = 10 / 1000.0
    frame_world_pos = [0, -(0.5 - knob_horizontal_location_ratio) * door_width, knob_height]
    frame_euler = [0, 0, 0]
    frame_mass = 500
    frame_diaginertia = [0.0001, 0.0001, 0.0001]

    # 门把手和门的关节位置
    knob_door_joint_pos = [0, 0, 0]
    knob_door_armature = 0.0001
    knob_door_limited = True
    knob_door_range = [-knob_rot_range, knob_rot_range]

    # 门闩相关尺寸
    latch_thickness = 0.015
    latch_height = 0.05
    latch_width = door_width * knob_horizontal_location_ratio + frame_width / 3.0
    latch_gap = latch_thickness * 1.3

    # 根据门把手类型决定把手在门上的位置
    if knob_type == "pull":
        knob_pos = [door_thickness / 2 - 0.006, 0, 0]
    else:
        knob_pos = [door_thickness / 2, 0, 0]

    # 把手惯性
    knob_diaginertia = [
        knob_mass / 12.0 * (latch_height ** 2 + latch_width ** 2),
        knob_mass / 12.0 * (latch_height ** 2 + latch_thickness ** 2),
        knob_mass / 12.0 * (latch_width ** 2 + latch_thickness ** 2)
    ]

    # 门框中各部件几何中心位置列表
    frame_pos_list = [
        [0, -door_width * knob_horizontal_location_ratio - frame_width / 2.0 - overlap_door_frame,
         (door_height + frame_width) / 2.0 - knob_height],
        [0, door_width - door_width * knob_horizontal_location_ratio + frame_width / 2.0 + overlap_door_frame,
         (door_height + frame_width) / 2.0 - knob_height],
        [0, door_width / 2 - door_width * knob_horizontal_location_ratio, door_height - knob_height + frame_width / 2],
        [-(door_thickness + latch_gap),
         -door_width * knob_horizontal_location_ratio - frame_width / 2.0 - overlap_door_frame,
         (door_height + frame_width) / 2.0 - knob_height],
        [-(door_thickness + latch_gap),
         door_width - door_width * knob_horizontal_location_ratio + frame_width / 2.0 + overlap_door_frame,
         (door_height + frame_width) / 2.0 - knob_height],
        [-(door_thickness + latch_gap), door_width / 2 - door_width * knob_horizontal_location_ratio,
         door_height - knob_height + frame_width / 2]
    ]
    # 门框各部件的大小
    frame_size_list = [
        [door_thickness / 2.0, frame_width / 2.0, (door_height + frame_width) / 2.0],
        [door_thickness / 2.0, frame_width / 2.0, (door_height + frame_width) / 2.0],
        [door_thickness / 2.0, door_width / 2.0 + frame_width, frame_width / 2.0],
        [door_thickness / 2.0, frame_width / 2.0, (door_height + frame_width) / 2.0],
        [door_thickness / 2.0, frame_width / 2.0, (door_height + frame_width) / 2.0],
        [door_thickness / 2.0, door_width / 2.0 + frame_width, frame_width / 2.0]
    ]

    # 墙体尺寸与位置
    sidewall_len = 2000 / 1000.0  # 2m
    topwall_width = 1000 / 1000.0  # 1m
    wall_thickness = 300 / 1000.0  # 0.3m
    wall_euler = [0, 0, 0]
    wall_mass = 100
    wall_diaginertia = [0.0001, 0.0001, 0.0001]

    # 根据门是否安装在前沿来计算墙体块的位置
    if door_front_frame:
        wall_pos_list = [
            [-wall_thickness / 2.0, -door_width / 2.0 - frame_width - sidewall_len / 2.0 + 0.03,
             (door_height + frame_width) / 2.0],
            [-wall_thickness / 2.0, door_width / 2.0 + frame_width + sidewall_len / 2.0,
             (door_height + frame_width) / 2.0],
            [-wall_thickness / 2.0, 0, door_height + frame_width + topwall_width / 2.0]
        ]
    else:
        wall_pos_list = [
            [0, -door_width / 2.0 - frame_width - sidewall_len / 2.0 + 0.03, (door_height + frame_width) / 2.0],
            [0, door_width / 2.0 + frame_width + sidewall_len / 2.0, (door_height + frame_width) / 2.0],
            [0, 0, door_height + frame_width + topwall_width / 2.0]
        ]
    wall_size_list = [
        [wall_thickness / 2.0, sidewall_len / 2.0, (door_height + frame_width) / 2.0],
        [wall_thickness / 2.0, sidewall_len / 2.0, (door_height + frame_width) / 2.0],
        [wall_thickness / 2.0, door_width / 2.0 + frame_width + sidewall_len, topwall_width / 2.0]
    ]

    ###################### XML生成 ######################
    # 创建一个根元素 Mujoco，表示 MJCF 文件的根节点
    mujoco = e.Mujoco(model="simulation_world")

    # 创建编译器元素，指定角度单位为弧度
    compiler = e.Compiler(angle="radian")
    # 创建 Include 元素，用于包含机械臂的 XML 文件
    include = e.Include(file=robot_include_file)
    # 创建 Option 元素，设定全局物理选项，如重力和时间步
    option = e.Option(gravity=gravity_vector, timestep=0.001)
    # 创建 Visual 元素，存放可视化相关设置
    visual = e.Visual()
    # 创建 Asset 元素，用于定义几何网格、纹理和材质等资源
    asset = e.Asset()
    # 创建 Contact 元素，用于定义接触对等信息
    contact = e.Contact()
    # 创建 Default 元素，用于定义默认属性
    default = e.Default()
    # 创建 Worldbody 元素，作为场景中的最顶层物理实体容器
    worldbody = e.Worldbody()

    # 将上述元素加入到 mujoco 根元素中
    mujoco.add_children([
        compiler,
        include,
        option,
        visual,
        asset,
        contact,
        default,
        worldbody
    ])

    # 下面配置一些 Visual（可视化）相关的子元素
    v_map = e.visual.Map(fogstart=3, fogend=5, force=0.1, znear=0.01, zfar=10)
    # 配置阴影分辨率
    v_quality = v.Quality(shadowsize=2048)
    # 配置全局渲染参数
    v_global = v.Global(offwidth=256, offheight=256)
    # 将这三个可视化元素加到 visual 中
    visual.add_children([v_map, v_quality, v_global])

    # Asset 部分：导入门把手网格（stl 文件）、纹理和材质
    if knob_type == "lever":
        knob_scale = [0.0015, 0.0015, 0.0015]  # lever 型把手的缩放比例
    else:
        knob_scale = [0.001, 0.001, 0.001]  # round/pull 型把手的缩放比例

    # 读取门把手各部件的 stl 网格文件
    mesh_knob = []
    fname = '../../door/{0}knobs/{1}/body_{2}.stl'  # STL 文件的路径格式
    mesh_name = 'door_knob_{}'
    for i in range(1, knob_parts_n + 1):
        mesh_knob.append(e.Mesh(
            file=fname.format(knob_type, dataset_file, i),
            name=mesh_name.format(i),
            scale=knob_scale
        ))

    # 定义纹理列表，用于墙、框、门、把手等
    texture_list = []
    texture_name_list = ['wall_geom', 'frame_geom', 'door_geom', 'knob_geom']
    texture_type_list = ['2d', '2d', '2d', '2d']
    for i in range(len(texture_name_list)):
        texture_list.append(e.Texture(
            builtin="flat",
            name=texture_name_list[i],
            height=32,
            width=32,
            type=texture_type_list[i]
        ))
    # 再添加一个天空盒纹理，使用渐变模拟天空
    texture_list.append(e.Texture(
        type="skybox",
        builtin="gradient",
        width=128,
        height=128,
        rgb1=[.4, .6, .8],
        rgb2=[0, 0, 0]
    ))
    # 添加一个地面纹理（渐变）
    texture_list.append(e.Texture(
        name='texplane',
        type='2d',
        builtin='gradient',
        rgb1=floor_rgb1,
        rgb2=floor_rgb2,
        width=512,
        height=512
    ))

    # 根据之前随机生成的 shininess/specular，为每种材质创建 Material
    material_list = []
    material_shininess_list = [paint_shininess, wood_shininess, carpet_shininess, metal_shininess]
    material_specular_list = [paint_specular, wood_specular, carpet_specular, metal_specular]
    for i in range(len(material_name_list)):
        material_list.append(e.Material(
            name=material_name_list[i],
            texture=texture_name_list[i],
            shininess=material_shininess_list[i],
            specular=material_specular_list[i]
        ))
    # 另外一个材质给地面
    material_list.append(e.Material(name="Floor", texture="texplane"))

    # 将门把手网格、纹理和材质添加到 asset 元素
    asset.add_children(mesh_knob + texture_list + material_list)

    # Contact 部分：门把手与门框之间的接触设置（仅在 knob_type 不是 "pull" 时添加门闩的接触对）
    contact_pairs = []
    if knob_type != "pull":
        for i in range(4):
            contact_pairs.append(e.Pair(
                geom1="knob_latch",
                geom2=f"door_frame_{i}",
                solref="0.01 1"
            ))
    contact.add_children(contact_pairs)

    # Default 元素：定义各类默认属性（joint、geom 等）
    joint_default = e.Joint(armature=1, damping=1, limited="true")
    # 为墙体默认属性创建一个 class='wall'
    wall_default = e.Default(class_='wall')
    wall_geom = e.Geom(type='mesh', rgba=wall_rgba)
    wall_default.add_child(wall_geom)
    # 门框默认属性
    frame_default = e.Default(class_='frame')
    frame_geom = e.Geom(type='mesh', rgba=frame_rgba)
    frame_default.add_child(frame_geom)
    # 门默认属性
    door_default = e.Default(class_='door')
    door_geom = e.Geom(type='mesh', rgba=door_rgba)
    door_default.add_child(door_geom)
    # 门把手默认属性
    knob_default = e.Default(class_='door_knob')
    knob_geom = e.Geom(condim=4, type="mesh", rgba=knob_rgba)
    knob_default.add_child(knob_geom)
    # 机械臂默认属性
    robot_default = e.Default(class_='robot')
    robot_joint = e.Joint(damping=robot_damping)
    robot_default.add_children([robot_joint])

    # 将所有 default 定义加入到 default 元素
    default.add_children([joint_default, wall_default, frame_default, door_default, knob_default, robot_default])

    # Worldbody 部分：添加灯光、地面、相机等
    for i in range(light_n):
        # 使用前面生成的随机灯光参数
        worldbody.add_child(e.Light(
            directional=True,
            diffuse=light_property_list[i]['light_diffuse'],
            pos=light_property_list[i]['light_pos'],
            dir=light_property_list[i]['light_dir']
        ))
    # 添加地面
    worldbody.add_child(e.Geom(
        name='floor',
        material="Floor",
        pos=[0, 0, -0.05],
        size=[15.0, 15.0, 0.05],
        type='plane'
    ))
    # 添加两个相机
    camera_names = ["camera1", "camera2"]
    for i in range(camera_n):
        worldbody.add_child(e.Camera(
            name=camera_names[i],
            mode="fixed",
            fovy=camera_fieldview + camera_fieldview_noise[i],
            pos=camera_poses[i],
            euler=camera_ories[i]
        ))

    # 创建分层的 Body：wall_link -> frame_link -> door_link -> knob_link -> (lever/round/pull)knob_link
    body0 = e.Body(name='wall_link', pos=wall_location, childclass='wall')
    body1 = e.Body(name='frame_link', pos=frame_world_pos, childclass='frame')
    body2 = e.Body(name='door_link', pos=door_pos, childclass='door')
    body3 = e.Body(name='knob_link', pos=doorknob_pos, childclass='door_knob')
    body4 = e.Body(name=f'{knob_type}knob_link', pos=knob_pos, childclass='door_knob')

    # 将子Body连接到 worldbody
    worldbody.add_child(body0)
    body0.add_child(body1)
    body1.add_child(body2)
    body2.add_child(body3)
    body3.add_child(body4)

    # Level 0: wall_link 的惯性与几何体
    body0.add_child(e.Inertial(pos=[0, 0, 0], mass=wall_mass, diaginertia=wall_diaginertia))
    # 在墙体 body 上添加各个 wall geom
    for i in range(wall_parts_n):
        body0.add_child(e.Geom(
            name=f"wall_{i}",
            material=wall_material,
            pos=wall_pos_list[i],
            size=wall_size_list[i],
            type='box',
            euler=frame_euler
        ))

    # Level 1: frame_link 的惯性与几何体
    body1.add_child(e.Inertial(pos=[0, 0, 0], mass=frame_mass, diaginertia=frame_diaginertia))
    # 在门框 body 上添加各个 frame geom
    for i in range(frame_parts_n):
        body1.add_child(e.Geom(
            name=f"door_frame_{i}",
            material=frame_material,
            pos=frame_pos_list[i],
            size=frame_size_list[i],
            type='box',
            euler=frame_euler
        ))

    # Level 2: door_link 添加门的关节、几何体和惯性
    door_joint = e.Joint(
        name='hinge0',
        type='hinge',
        axis=[0, 0, 1],
        armature=door_frame_armature,
        stiffness=door_frame_spring,
        damping=door_frame_damper,
        frictionloss=door_frame_frictionloss,
        limited=door_frame_limited,
        range=door_frame_range,
        pos=door_frame_joint_pos
    )
    body2.add_child(door_joint)
    # 门的几何体
    body2.add_child(e.Geom(
        name='door0',
        material=door_material,
        pos=[0, door_width / 2.0 - door_width * knob_horizontal_location_ratio, door_height / 2.0 - knob_height],
        size=[door_thickness / 2.0, door_width / 2.0, door_height / 2.0 * 0.99],
        type="box",
        euler=door_euler
    ))
    # 门的惯性
    body2.add_child(e.Inertial(
        pos=[0, door_width / 2.0 - door_width * knob_horizontal_location_ratio, door_height / 2.0 - knob_height],
        mass=door_mass,
        diaginertia=door_diaginertia
    ))

    # Level 3: knob_link 添加滑动和铰链关节（用于做“抓取”测试的把手平移）
    axes = [[0, 1, 0], [0, 0, 1]]  # 分别对 Y 轴和 Z 轴进行滑动
    ranges = [[-0.2, 0.3], [-0.5, 0.5]]  # 对应上面的滑动范围
    for i in range(2):
        body3.add_child(e.Joint(
            name=f'target{i}',
            type='slide',
            axis=axes[i],
            armature=0,
            stiffness=0,
            damping=30000,
            frictionloss=0,
            limited="true",
            range=ranges[i]
        ))
    # 如果不是 pull 类型，把手还能绕 X 轴转动（即真实的门把手扭动）
    if knob_type != "pull":
        body3.add_child(e.Joint(
            name='hinge1',
            type='hinge',
            axis=[1, 0, 0],
            armature=knob_door_armature,
            stiffness=knob_door_spring,
            damping=knob_door_damper,
            frictionloss=knob_door_frictionloss,
            limited=knob_door_limited,
            range=knob_door_range,
            pos=knob_door_joint_pos
        ))
    # 给 knob_link 一个小的惯性
    body3.add_child(e.Inertial(pos=[0, 0, 0], mass=1, diaginertia=[0.001, 0.001, 0.001]))

    # Level 4: 在 knob_link 下添加具体的门把手几何
    for i in range(1, knob_parts_n + 1):
        body4.add_child(e.Geom(
            name=f"door_knob_{i}",
            material=doorknob_material,
            mesh=f"door_knob_{i}",
            euler=knob_euler,
            friction=knob_surface_friction
        ))
    # 如果不是 pull，把手带有一个 latch（门闩）
    if knob_type != "pull":
        pos_ = [-(latch_gap / 2.0 + door_thickness * 1.0), 0, 0]
        body4.add_child(e.Geom(
            name='knob_latch',
            material=material_name_list[randrange(0, 3)],
            pos=pos_,
            size=[latch_thickness / 2.0, latch_width, latch_height],
            type="box",
            euler=[0, 0, 0]
        ))
        body4.add_child(e.Inertial(
            pos=[-(latch_gap / 2.0 + door_thickness * 2.1), 0, 0],
            mass=knob_mass,
            diaginertia=knob_diaginertia
        ))

    # 最后，将生成的 XML 字符串写入文件
    model_xml = mujoco.xml()
    output_path = os.path.join(args.output_dirname, f"{knob_type}_gen3")
    os.makedirs(output_path, exist_ok=True)
    output_filename = os.path.join(output_path, f"{dataset_file}_{knob_type}_gen3.xml")
    with open(output_filename, 'w') as fh:
        fh.write(model_xml)


if __name__ == '__main__':
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description='生成仿真世界XML文件')
    parser.add_argument('--knob-type', default=None, choices=['lever', 'round', 'pull'], help='门把手类型')
    parser.add_argument('--input-dirname', type=str, default='./door/{}knobs',
                        help='包含门把手数据集的目录（含 {} 占位符）')
    parser.add_argument('--output-dirname', type=str, default='./world', help='输出仿真世界XML文件的目录')
    parser.add_argument('--output-name-extention', type=str, default="", help='输出文件名附加信息')
    args = parser.parse_args()

    # 若未指定门把手类型，则生成所有三种类型（lever, round, pull）
    if args.knob_type is None:
        for knob_type in ['lever', 'round', 'pull']:
            print(f"生成门把手类型: {knob_type}")
            dataset_files = os.listdir(args.input_dirname.format(knob_type))
            for dataset_file in tqdm(dataset_files, desc=f"生成 {knob_type} 类型仿真世界"):
                main(dataset_file, args, knob_type)
    else:
        # 若指定了门把手类型，则只生成该类型
        knob_type = args.knob_type
        dataset_files = os.listdir(args.input_dirname.format(knob_type))
        for dataset_file in tqdm(dataset_files, desc=f"生成 {knob_type} 类型仿真世界"):
            main(dataset_file, args, knob_type)
