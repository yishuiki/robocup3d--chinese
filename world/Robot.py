from collections import deque
from math import atan, pi, sqrt, tan
from math_ops.Math_Ops import Math_Ops as M
from math_ops.Matrix_3x3 import Matrix_3x3
from math_ops.Matrix_4x4 import Matrix_4x4
from world.commons.Body_Part import Body_Part
from world.commons.Joint_Info import Joint_Info
import numpy as np
import xml.etree.ElementTree as xmlp

class Robot():
    # 定义机器人的常量参数
    STEPTIME = 0.02   # 固定的时间步长
    VISUALSTEP = 0.04 # 固定的视觉更新时间步长
    SQ_STEPTIME = STEPTIME * STEPTIME  # 时间步长的平方，用于计算
    GRAVITY = np.array([0, 0, -9.81])  # 重力加速度（m/s²）
    IMU_DECAY = 0.996 # 惯性测量单元（IMU）的速度衰减因子

    # 定义感知器到索引的映射，用于快速访问关节信息
    MAP_PERCEPTOR_TO_INDEX = {
        "hj1": 0, "hj2": 1, "llj1": 2, "rlj1": 3,
        "llj2": 4, "rlj2": 5, "llj3": 6, "rlj3": 7,
        "llj4": 8, "rlj4": 9, "llj5": 10, "rlj5": 11,
        "llj6": 12, "rlj6": 13, "laj1": 14, "raj1": 15,
        "laj2": 16, "raj2": 17, "laj3": 18, "raj3": 19,
        "laj4": 20, "raj4": 21, "llj7": 22, "rlj7": 23
    }

    # 定义需要修正对称性的关节集合和索引列表
    FIX_PERCEPTOR_SET = {'rlj2', 'rlj6', 'raj2', 'laj3', 'laj4'}
    FIX_INDICES_LIST = [5, 13, 17, 18, 20]

    # 定义不同机器人类型对应的官方横梁高度（单位：米）
    BEAM_HEIGHTS = [0.4, 0.43, 0.4, 0.46, 0.4]
   
    def __init__(self, unum: int, robot_type: int) -> None:
        """
        初始化一个机器人实例。
        参数：
        - unum (int): 机器人的编号。
        - robot_type (int): 机器人的类型，用于确定其配置文件和特性。
        """
        # 根据机器人类型确定配置文件名
        robot_xml = "nao" + str(robot_type) + ".xml"  # 典型的NAO机器人配置文件名
        self.type = robot_type  # 机器人类型
        self.beam_height = Robot.BEAM_HEIGHTS[robot_type]  # 根据机器人类型获取横梁高度
        self.no_of_joints = 24 if robot_type == 4 else 22  # 根据机器人类型确定关节数量

        # 初始化关节速度修正掩码，用于修正对称性问题
        self.FIX_EFFECTOR_MASK = np.ones(self.no_of_joints)
        self.FIX_EFFECTOR_MASK[Robot.FIX_INDICES_LIST] = -1

        # 初始化机器人的身体部位字典，键为身体部位名称，值为Body_Part对象
        self.body_parts = dict()
        self.unum = unum  # 机器人编号
        self.gyro = np.zeros(3)  # 陀螺仪数据（机器人躯干的角速度，单位：deg/s）
        self.acc = np.zeros(3)  # 加速度计数据（机器人躯干的加速度，单位：m/s²）
        self.frp = dict()  # 脚和脚趾的阻力感知器数据
        self.feet_toes_last_touch = {"lf": 0, "rf": 0, "lf1": 0, "rf1": 0}  # 脚和脚趾上次接触地面的时间
        self.feet_toes_are_touching = {"lf": False, "rf": False, "lf1": False, "rf1": False}  # 脚和脚趾是否接触地面
        self.fwd_kinematics_list = None  # 正向运动学列表，用于存储身体部位的依赖顺序
        self.rel_cart_CoM_position = np.zeros(3)  # 相对质心位置（相对于头部，单位：米）

        # 初始化关节变量
        self.joints_position = np.zeros(self.no_of_joints)  # 关节角度位置（单位：度）
        self.joints_speed = np.zeros(self.no_of_joints)  # 关节角速度（单位：rad/s）
        self.joints_target_speed = np.zeros(self.no_of_joints)  # 关节目标速度（单位：rad/s）
        self.joints_target_last_speed = np.zeros(self.no_of_joints)  # 关节上次目标速度
        self.joints_info = [None] * self.no_of_joints  # 关节信息（Joint_Info对象）
        self.joints_transform = [Matrix_4x4() for _ in range(self.no_of_joints)]  # 关节变换矩阵

        # 初始化定位相关变量（相对于头部）
        self.loc_head_to_field_transform = Matrix_4x4()  # 头部到场地的变换矩阵
        self.loc_field_to_head_transform = Matrix_4x4()  # 场地到头部的变换矩阵
        self.loc_rotation_head_to_field = Matrix_3x3()  # 头部到场地的旋转矩阵
        self.loc_rotation_field_to_head = Matrix_3x3()  # 场地到头部的旋转矩阵
        self.loc_head_position = np.zeros(3)  # 绝对头部位置（单位：米）
        self.loc_head_position_history = deque(maxlen=40)  # 头部位置历史记录（最多存储40个位置）
        self.loc_head_velocity = np.zeros(3)  # 绝对头部速度（单位：m/s，可能较嘈杂）
        self.loc_head_orientation = 0  # 头部方向（单位：度）
        self.loc_is_up_to_date = False  # 是否为视觉更新步骤，且定位信息有效
        self.loc_last_update = 0  # 定位信息上次更新的时间（单位：毫秒）
        self.loc_head_position_last_update = 0  # 头部位置上次更新的时间
        self.radio_fallen_state = False  # 是否通过无线电判断机器人摔倒
        self.radio_last_update = 0  # 无线电信息上次更新的时间

        # 初始化定位相关变量（相对于躯干）
        self.loc_torso_to_field_rotation = Matrix_3x3()  # 躯干到场地的旋转矩阵
        self.loc_torso_to_field_transform = Matrix_4x4()  # 躯干到场地的变换矩阵
        self.loc_torso_roll = 0  # 躯干翻滚角（单位：度）
        self.loc_torso_pitch = 0  # 躯干俯仰角（单位：度）
        self.loc_torso_orientation = 0  # 躯干方向（单位：度）
        self.loc_torso_inclination = 0  # 躯干倾斜角（单位：度）
        self.loc_torso_position = np.zeros(3)  # 绝对躯干位置（单位：米）
        self.loc_torso_velocity = np.zeros(3)  # 绝对躯干速度（单位：m/s）
        self.loc_torso_acceleration = np.zeros(3)  # 绝对躯干加速度（单位：m/s²）

        # 初始化其他定位相关变量
        self.cheat_abs_pos = np.zeros(3)  # 服务器提供的绝对头部位置（作弊用，单位：米）
        self.cheat_ori = 0.0  # 服务器提供的绝对头部方向（作弊用，单位：度）
        self.loc_CoM_position = np.zeros(3)  # 绝对质心位置（单位：米）
        self.loc_CoM_velocity = np.zeros(3)  # 绝对质心速度（单位：m/s）

        # 初始化特殊定位变量（如头部高度）
        self.loc_head_z = 0  # 绝对头部高度（单位：米）
        self.loc_head_z_is_up_to_date = False  # 头部高度是否更新
        self.loc_head_z_last_update = 0  # 头部高度上次更新的时间
        self.loc_head_z_vel = 0  # 头部高度速度（单位：m/s）

            # 初始化基于IMU（惯性测量单元）的定位变量
        self.imu_torso_roll = 0  # 躯干翻滚角（单位：度，来源：定位 + 陀螺仪）
        self.imu_torso_pitch = 0  # 躯干俯仰角（单位：度，来源：定位 + 陀螺仪）
        self.imu_torso_orientation = 0  # 躯干方向（单位：度，来源：定位 + 陀螺仪）
        self.imu_torso_inclination = 0  # 躯干倾斜角（单位：度，来源：定位 + 陀螺仪）
        self.imu_torso_to_field_rotation = Matrix_3x3()  # 躯干到场地的旋转矩阵（来源：定位 + 陀螺仪）
        self.imu_last_visual_update = 0  # IMU数据上次通过视觉信息更新的时间（单位：毫秒）

        # 初始化基于IMU + 加速度计的定位变量（注意：这些变量可能不可靠）
        self.imu_weak_torso_to_field_transform = Matrix_4x4()  # 躯干到场地的变换矩阵（来源：定位 + 陀螺仪 + 加速度计）
        self.imu_weak_head_to_field_transform = Matrix_4x4()  # 头部到场地的变换矩阵（来源：定位 + 陀螺仪 + 加速度计）
        self.imu_weak_field_to_head_transform = Matrix_4x4()  # 场地到头部的变换矩阵（来源：定位 + 陀螺仪 + 加速度计）
        self.imu_weak_torso_position = np.zeros(3)  # 绝对躯干位置（单位：米，来源：定位 + 陀螺仪 + 加速度计）
        self.imu_weak_torso_velocity = np.zeros(3)  # 绝对躯干速度（单位：m/s，来源：定位 + 陀螺仪 + 加速度计）
        self.imu_weak_torso_acceleration = np.zeros(3)  # 绝对躯干加速度（单位：m/s²，来源：定位 + 陀螺仪 + 加速度计）
        self.imu_weak_torso_next_position = np.zeros(3)  # 预测的下一次躯干位置（单位：米）
        self.imu_weak_torso_next_velocity = np.zeros(3)  # 预测的下一次躯干速度（单位：m/s）
        self.imu_weak_CoM_position = np.zeros(3)  # 绝对质心位置（单位：米，来源：定位 + 陀螺仪 + 加速度计）
        self.imu_weak_CoM_velocity = np.zeros(3)  # 绝对质心速度（单位：m/s，来源：定位 + 陀螺仪 + 加速度计）
    
        # 定义关节索引的显式变量，以便IDE提供自动补全建议
        self.J_HEAD_YAW = 0  # 头部偏航角
        self.J_HEAD_PITCH = 1  # 头部俯仰角
        self.J_LLEG_YAW_PITCH = 2  # 左腿偏航角和俯仰角
        self.J_RLEG_YAW_PITCH = 3  # 右腿偏航角和俯仰角
        self.J_LLEG_ROLL = 4  # 左腿翻滚角
        self.J_RLEG_ROLL = 5  # 右腿翻滚角
        self.J_LLEG_PITCH = 6  # 左腿俯仰角
        self.J_RLEG_PITCH = 7  # 右腿俯仰角
        self.J_LKNEE = 8  # 左膝关节
        self.J_RKNEE = 9  # 右膝关节
        self.J_LFOOT_PITCH = 10  # 左脚俯仰角
        self.J_RFOOT_PITCH = 11  # 右脚俯仰角
        self.J_LFOOT_ROLL = 12  # 左脚翻滚角
        self.J_RFOOT_ROLL = 13  # 右脚翻滚角
        self.J_LARM_PITCH = 14  # 左臂俯仰角
        self.J_RARM_PITCH = 15  # 右臂俯仰角
        self.J_LARM_ROLL = 16  # 左臂翻滚角
        self.J_RARM_ROLL = 17  # 右臂翻滚角
        self.J_LELBOW_YAW = 18  # 左肘偏航角
        self.J_RELBOW_YAW = 19  # 右肘偏航角
        self.J_LELBOW_ROLL = 20  # 左肘翻滚角
        self.J_RELBOW_ROLL = 21  # 右肘翻滚角
        self.J_LTOE_PITCH = 22  # 左脚趾俯仰角
        self.J_RTOE_PITCH = 23  # 右脚趾俯仰角

            # 解析机器人配置文件
        dir = M.get_active_directory("/world/commons/robots/")  # 获取机器人配置文件所在的目录
        robot_xml_root = xmlp.parse(dir + robot_xml).getroot()  # 解析配置文件

        joint_no = 0  # 当前处理的关节编号
        for child in robot_xml_root:
            if child.tag == "bodypart":
                # 如果是身体部位节点，初始化Body_Part对象
                self.body_parts[child.attrib['name']] = Body_Part(child.attrib['mass'])
            elif child.tag == "joint":
                # 如果是关节节点，初始化Joint_Info对象
                self.joints_info[joint_no] = Joint_Info(child)
                self.joints_position[joint_no] = 0.0  # 初始化关节角度为0
                ji = self.joints_info[joint_no]

                # 将关节添加到身体部位的关节列表中
                self.body_parts[ji.anchor0_part].joints.append(Robot.MAP_PERCEPTOR_TO_INDEX[ji.perceptor])

                joint_no += 1
                if joint_no == self.no_of_joints:
                    break  # 如果处理完所有关节，则退出循环
            else:
                raise NotImplementedError("未实现的XML节点类型")

        # 检查关节数量是否匹配
        assert joint_no == self.no_of_joints, "机器人配置文件与机器人类型不匹配！"
    
    def get_head_abs_vel(self, history_steps: int):
        """
        获取机器人头部的绝对速度（单位：m/s）。

        参数：
        - history_steps (int): 考虑的历史步数，范围为 [1, 40]。

        示例：
        - history_steps=1：计算当前绝对位置与上一次绝对位置的差值，除以时间间隔（0.04秒）。
        - history_steps=2：计算当前绝对位置与0.08秒前的绝对位置的差值，除以时间间隔（0.08秒）。
        """
        assert 1 <= history_steps <= 40, "参数 'history_steps' 必须在范围 [1, 40] 内"

        if len(self.loc_head_position_history) == 0:
            return np.zeros(3)  # 如果历史记录为空，则返回零速度

        h_step = min(history_steps, len(self.loc_head_position_history))  # 获取有效的历史步数
        t = h_step * Robot.VISUALSTEP  # 计算时间间隔

        # 计算当前头部位置与历史位置的差值，除以时间间隔，得到速度
        return (self.loc_head_position - self.loc_head_position_history[h_step - 1]) / t
   
    def _initialize_kinematics(self):
        """
        初始化正向运动学链。

        从头部开始，构建一个有序的身体部位列表，用于后续的运动学计算。
        """
        # 从头部开始
        parts = {"head"}
        sequential_body_parts = ["head"]

        while len(parts) > 0:
            part = parts.pop()  # 获取当前身体部位

            for j in self.body_parts[part].joints:
                # 获取当前关节连接的下一个身体部位
                p = self.joints_info[j].anchor1_part

                if len(self.body_parts[p].joints) > 0:  # 如果该身体部位是其他关节的锚点
                    parts.add(p)  # 将其加入待处理列表
                    sequential_body_parts.append(p)  # 将其加入正向运动学链
        # 构建正向运动学链，存储身体部位、关节及其连接关系
        self.fwd_kinematics_list = [
            (self.body_parts[part], j, self.body_parts[self.joints_info[j].anchor1_part])
            for part in sequential_body_parts
            for j in self.body_parts[part].joints
        ]

        # 修正对称性问题（运动学部分）
        for i in Robot.FIX_INDICES_LIST:
            self.joints_info[i].axes *= -1  # 反转关节轴向量
            aux = self.joints_info[i].min
            self.joints_info[i].min = -self.joints_info[i].max
            self.joints_info[i].max = -aux  # 交换关节角度范围的最小值和最大值

    def update_localization(self, localization_raw, time_local_ms):
        """
        更新机器人的定位信息。

        参数：
        - localization_raw：从传感器或服务器接收到的原始定位数据。
        - time_local_ms：当前时间戳（单位：毫秒）。
        """
        # 将原始数据转换为64位浮点数，确保数据一致性
        loc = localization_raw.astype(float)
        self.loc_is_up_to_date = bool(loc[32])  # 检查定位数据是否有效
        self.loc_head_z_is_up_to_date = bool(loc[34])  # 检查头部高度数据是否有效

        if self.loc_head_z_is_up_to_date:
            # 如果头部高度数据有效，更新头部高度及其速度
            time_diff = (time_local_ms - self.loc_head_z_last_update) / 1000  # 时间差（秒）
            self.loc_head_z_vel = (loc[33] - self.loc_head_z) / time_diff  # 更新头部高度速度
            self.loc_head_z = loc[33]  # 更新头部高度
            self.loc_head_z_last_update = time_local_ms  # 更新时间戳

        # 将当前头部位置保存到历史记录中（即使定位数据无效，也会在视觉更新周期中保存）
        self.loc_head_position_history.appendleft(np.copy(self.loc_head_position))

        if self.loc_is_up_to_date:
            # 如果定位数据有效，更新所有相关定位信息
            time_diff = (time_local_ms - self.loc_last_update) / 1000  # 时间差（秒）
            self.loc_last_update = time_local_ms  # 更新时间戳

            # 更新头部到场地的变换矩阵
            self.loc_head_to_field_transform.m[:] = loc[0:16].reshape((4, 4))
            self.loc_field_to_head_transform.m[:] = loc[16:32].reshape((4, 4))

            # 提取与头部相关的数据
            self.loc_rotation_head_to_field = self.loc_head_to_field_transform.get_rotation()
            self.loc_rotation_field_to_head = self.loc_field_to_head_transform.get_rotation()
            p = self.loc_head_to_field_transform.get_translation()
            self.loc_head_velocity = (p - self.loc_head_position) / time_diff  # 更新头部速度
            self.loc_head_position = p  # 更新头部位置
            self.loc_head_position_last_update = time_local_ms  # 更新时间戳
            self.loc_head_orientation = self.loc_head_to_field_transform.get_yaw_deg()  # 更新头部方向
            self.radio_fallen_state = False  # 重置摔倒状态标志

            # 更新质心相关数据
            p = self.loc_head_to_field_transform(self.rel_cart_CoM_position)
            self.loc_CoM_velocity = (p - self.loc_CoM_position) / time_diff  # 更新质心速度
            self.loc_CoM_position = p  # 更新质心位置

            # 更新躯干相关数据
            t = self.get_body_part_to_field_transform('torso')
            self.loc_torso_to_field_transform = t
            self.loc_torso_to_field_rotation = t.get_rotation()
            self.loc_torso_orientation = t.get_yaw_deg()
            self.loc_torso_pitch = t.get_pitch_deg()
            self.loc_torso_roll = t.get_roll_deg()
            self.loc_torso_inclination = t.get_inclination_deg()
            p = t.get_translation()
            self.loc_torso_velocity = (p - self.loc_torso_position) / time_diff  # 更新躯干速度
            self.loc_torso_position = p  # 更新躯干位置
            self.loc_torso_acceleration = self.loc_torso_to_field_rotation.multiply(self.acc) + Robot.GRAVITY  # 更新躯干加速度
   
    def head_to_body_part_transform(self, body_part_name, coords, is_batch=False):
        """
        将相对于头部的坐标转换为相对于指定身体部位的坐标。

        参数：
        - body_part_name（str）：目标身体部位的名称。
        - coords：一个或多个三维坐标点。
        - is_batch（bool）：是否为批量转换。

        返回：
        - 转换后的坐标（单个坐标或坐标列表）。
        """
        head_to_bp_transform = self.body_parts[body_part_name].transform.invert()  # 获取头部到目标身体部位的逆变换矩阵

        if is_batch:
            # 如果是批量转换，对每个坐标点应用变换
            return [head_to_bp_transform(c) for c in coords]
        else:
            # 如果是单个坐标，直接应用变换
            return head_to_bp_transform(coords)

    def get_body_part_to_field_transform(self, body_part_name) -> Matrix_4x4:
        """
        计算指定身体部位到场地的变换矩阵。

        参数：
        - body_part_name（str）：目标身体部位的名称。

        返回：
        - 身体部位到场地的变换矩阵（Matrix_4x4）。
        """
        # 通过头部到场地的变换矩阵和身体部位的局部变换矩阵，计算全局变换矩阵
        return self.loc_head_to_field_transform.multiply(self.body_parts[body_part_name].transform)
    
    def get_body_part_abs_position(self, body_part_name) -> np.ndarray:
        """
        计算指定身体部位的绝对位置。

        参数：
        - body_part_name（str）：目标身体部位的名称。

        返回：
        - 身体部位的绝对位置（三维坐标，单位：米）。
        """
        # 从变换矩阵中提取平移部分，即为绝对位置
        return self.get_body_part_to_field_transform(body_part_name).get_translation()

    def get_joint_to_field_transform(self, joint_index) -> Matrix_4x4:
        """
        计算指定关节到场地的变换矩阵。

        参数：
        - joint_index（int）：关节的索引。

        返回：
        - 关节到场地的变换矩阵（Matrix_4x4）。
        """
        # 通过头部到场地的变换矩阵和关节的局部变换矩阵，计算全局变换矩阵
        return self.loc_head_to_field_transform.multiply(self.joints_transform[joint_index])

    def get_joint_abs_position(self, joint_index) -> np.ndarray:
        """
        计算指定关节的绝对位置。

        参数：
        - joint_index（int）：关节的索引。

        返回：
        - 关节的绝对位置（三维坐标，单位：米）。
        """
        # 从变换矩阵中提取平移部分，即为绝对位置
        return self.get_joint_to_field_transform(joint_index).get_translation()

    def update_pose(self):
        """
        更新机器人的姿态（正向运动学）。

        通过关节角度和身体部位的依赖关系，计算每个身体部位的全局位置和姿态。
        """
        if self.fwd_kinematics_list is None:
            self._initialize_kinematics()  # 如果尚未初始化运动学链，则先初始化

        # 遍历正向运动学链，更新每个身体部位的变换矩阵
        for body_part, j, child_body_part in self.fwd_kinematics_list:
            ji = self.joints_info[j]
            self.joints_transform[j].m[:] = body_part.transform.m  # 复制当前身体部位的变换矩阵
            self.joints_transform[j].translate(ji.anchor0_axes, True)  # 平移到关节的锚点位置
            child_body_part.transform.m[:] = self.joints_transform[j].m  # 更新子身体部位的变换矩阵
            child_body_part.transform.rotate_deg(ji.axes, self.joints_position[j], True)  # 绕关节轴旋转
            child_body_part.transform.translate(ji.anchor1_axes_neg, True)  # 平移到子身体部位的位置

        # 计算质心的相对位置
        self.rel_cart_CoM_position = np.average(
            [b.transform.get_translation() for b in self.body_parts.values()],  # 获取所有身体部位的绝对位置
            axis=0,
            weights=[b.mass for b in self.body_parts.values()]  # 使用身体部位的质量作为权重
        )
   
    def update_imu(self, time_local_ms):
        """
        更新惯性测量单元（IMU）数据。

        如果有新的视觉数据，则直接更新IMU数据；否则，根据陀螺仪和加速度计数据进行预测更新。
        """
        # 如果定位数据有效，直接更新IMU数据
        if self.loc_is_up_to_date:
            self.imu_torso_roll = self.loc_torso_roll
            self.imu_torso_pitch = self.loc_torso_pitch
            self.imu_torso_orientation = self.loc_torso_orientation
            self.imu_torso_inclination = self.loc_torso_inclination
            self.imu_torso_to_field_rotation.m[:] = self.loc_torso_to_field_rotation.m
            self.imu_weak_torso_to_field_transform.m[:] = self.loc_torso_to_field_transform.m
            self.imu_weak_head_to_field_transform.m[:] = self.loc_head_to_field_transform.m
            self.imu_weak_field_to_head_transform.m[:] = self.loc_field_to_head_transform.m
            self.imu_weak_torso_position[:] = self.loc_torso_position
            self.imu_weak_torso_velocity[:] = self.loc_torso_velocity
            self.imu_weak_torso_acceleration[:] = self.loc_torso_acceleration

            # 预测下一步的位置和速度
            self.imu_weak_torso_next_position = (
                self.loc_torso_position + self.loc_torso_velocity * Robot.STEPTIME +
                self.loc_torso_acceleration * (0.5 * Robot.SQ_STEPTIME)
            )
            self.imu_weak_torso_next_velocity = (
                self.loc_torso_velocity + self.loc_torso_acceleration * Robot.STEPTIME
            )
            self.imu_weak_CoM_position[:] = self.loc_CoM_position
            self.imu_weak_CoM_velocity[:] = self.loc_CoM_velocity
            self.imu_last_visual_update = time_local_ms
        else:
            # 如果没有新的视觉数据，根据陀螺仪数据更新旋转
            g = self.gyro / 50  # 将陀螺仪数据从度/秒转换为度/步
            self.imu_torso_to_field_rotation.multiply(
                Matrix_3x3.from_rotation_deg(g), in_place=True, reverse_order=True
            )

            # 更新IMU的姿态信息
            self.imu_torso_orientation = self.imu_torso_to_field_rotation.get_yaw_deg()
            self.imu_torso_pitch = self.imu_torso_to_field_rotation.get_pitch_deg()
            self.imu_torso_roll = self.imu_torso_to_field_rotation.get_roll_deg()

            # 计算倾斜角度
            self.imu_torso_inclination = (
                atan(
                    sqrt(
                        tan(self.imu_torso_roll / 180 * pi) ** 2 +
                        tan(self.imu_torso_pitch / 180 * pi) ** 2
                    )
                ) * 180 / pi
            )

            # 如果距离上次视觉更新不超过0.2秒，继续更新位置和速度
            if time_local_ms < self.imu_last_visual_update + 200:
                self.imu_weak_torso_position[:] = self.imu_weak_torso_next_position
                if self.imu_weak_torso_position[2] < 0:
                    self.imu_weak_torso_position[2] = 0  # 限制z坐标为正值
                self.imu_weak_torso_velocity[:] = (
                    self.imu_weak_torso_next_velocity * Robot.IMU_DECAY
                )  # 速度衰减以提高稳定性
            else:
                # 如果超过0.2秒没有视觉更新，锁定位置并逐渐衰减速度
                self.imu_weak_torso_velocity *= 0.97

            # 将加速度计数据转换为坐标加速度，并修正误差
            self.imu_weak_torso_acceleration = (
                self.imu_torso_to_field_rotation.multiply(self.acc) + Robot.GRAVITY
            )
            self.imu_weak_torso_to_field_transform = Matrix_4x4.from_3x3_and_translation(
                self.imu_torso_to_field_rotation, self.imu_weak_torso_position
            )
            self.imu_weak_head_to_field_transform = (
                self.imu_weak_torso_to_field_transform.multiply(
                    self.body_parts["torso"].transform.invert()
                )
            )
            self.imu_weak_field_to_head_transform = self.imu_weak_head_to_field_transform.invert()

            # 更新头部位置和速度
            p = self.imu_weak_head_to_field_transform(self.rel_cart_CoM_position)
            self.imu_weak_CoM_velocity = (
                (p - self.imu_weak_CoM_position) / Robot.STEPTIME
            )
            self.imu_weak_CoM_position = p

            # 预测下一步的位置和速度
            self.imu_weak_torso_next_position = (
                self.imu_weak_torso_position + self.imu_weak_torso_velocity * Robot.STEPTIME +
                self.imu_weak_torso_acceleration * (0.5 * Robot.SQ_STEPTIME)
            )
            self.imu_weak_torso_next_velocity = (
                self.imu_weak_torso_velocity + self.imu_weak_torso_acceleration * Robot.STEPTIME
            )

    def set_joints_target_position_direct(
        self, indices, values: np.ndarray, harmonize=True, max_speed=7.03, tolerance=0.012, limit_joints=True
    ) -> int:
        """
        设置关节的目标位置，并计算所需的速度。

        参数：
        - indices：关节索引（可以是单个索引、列表、切片或numpy数组）。
        - values：目标位置数组（单位：度）。
        - harmonize：是否使所有关节同时到达目标位置。
        - max_speed：最大速度（单位：度/步，默认为7.03，对应351.77度/秒）。
        - tolerance：角度误差容差（单位：度，用于判断是否达到目标）。
        - limit_joints：是否限制关节角度在允许范围内。

        返回：
        - 剩余步数（如果目标已经到达，则返回-1）。
        """
        assert type(values) == np.ndarray, "'values' 参数必须是numpy数组"
        np.nan_to_num(values, copy=False)  # 将NaN替换为零，将无穷大替换为有限值

        # 如果启用，限制关节角度在允许范围内
        if limit_joints:
            if type(indices) == list or type(indices) == np.ndarray:
                for i in range(len(indices)):
                    values[i] = np.clip(
                        values[i], self.joints_info[indices[i]].min, self.joints_info[indices[i]].max
                    )
            elif type(indices) == slice:
                info = self.joints_info[indices]
                for i in range(len(info)):
                    values[i] = np.clip(values[i], info[i].min, info[i].max)
            else:  # 单个索引
                values[0] = np.clip(
                    values[0], self.joints_info[indices].min, self.joints_info[indices].max
                )

        # 预测关节位置与实际位置的差异
        predicted_diff = self.joints_target_last_speed[indices] * 1.1459156  # 将弧度/秒转换为度/步
        predicted_diff = np.asarray(predicted_diff)
        np.clip(predicted_diff, -7.03, 7.03, out=predicted_diff)  # 限制预测运动范围

        # 计算目标位置与当前位置的差异
        reported_dist = values - self.joints_position[indices]
        if (
            np.all((np.abs(reported_dist) < tolerance)) and
            np.all((np.abs(predicted_diff) < tolerance))
        ):
            self.joints_target_speed[indices] = 0
            return -1  # 如果目标已经到达，返回-1

        # 计算每个关节的运动量
        deg_per_step = reported_dist - predicted_diff

        # 如果启用同步，计算最大步数
        relative_max = np.max(np.abs(deg_per_step)) / max_speed
        remaining_steps = np.ceil(relative_max)

        if remaining_steps == 0:
            self.joints_target_speed[indices] = 0
            return 0  # 如果不需要额外步骤，返回0

        # 如果启用同步，调整每个关节的速度，使它们同时到达目标位置
        if harmonize:
            deg_per_step /= remaining_steps
        else:
            # 如果不启用同步，直接限制每个关节的最大速度
            np.clip(deg_per_step, -max_speed, max_speed, out=deg_per_step)

        # 将速度从度/步转换为弧度/秒
        self.joints_target_speed[indices] = deg_per_step * 0.87266463  # 1度/步 ≈ 0.87266463 弧度/秒

        return int(remaining_steps)  # 返回剩余步数

    def get_command(self) -> bytes:
        """
        构建并返回机器人关节速度的命令字符串。

        返回值：
        - 命令字符串（字节形式）。
        """
        # 根据关节速度修正掩码调整目标速度，以修正对称性问题
        j_speed = self.joints_target_speed * self.FIX_EFFECTOR_MASK

        # 构建命令字符串
        cmd = "".join(
            f"({self.joints_info[i].effector} {j_speed[i]:.5f})"  # 格式化为字符串，保留5位小数
            for i in range(self.no_of_joints)
        ).encode('utf-8')  # 将字符串编码为字节

        # 更新关节速度状态
        self.joints_target_last_speed = self.joints_target_speed  # 保存当前目标速度
        self.joints_target_speed = np.zeros_like(self.joints_target_speed)  # 重置目标速度数组

        return cmd  # 返回命令字节


