from math import asin, atan, atan2, pi, sqrt
from math_ops.Matrix_3x3 import Matrix_3x3
from math_ops.Math_Ops import Math_Ops as M
import numpy as np

class Inverse_Kinematics():
    """
    逆运动学类，用于计算机器人的腿部关节角度。
    """

    # 不同机器人的腿部参数（腿的y偏移量、大腿高度、大腿深度、小腿长度、膝盖额外角度、最大脚踝z值）
    NAO_SPECS_PER_ROBOT = ((0.055,      0.12,        0.005, 0.1,         atan(0.005/0.12),        -0.091),
                           (0.055,      0.13832,     0.005, 0.11832,     atan(0.005/0.13832),     -0.106),
                           (0.055,      0.12,        0.005, 0.1,         atan(0.005/0.12),        -0.091),
                           (0.072954143,0.147868424, 0.005, 0.127868424, atan(0.005/0.147868424), -0.114),
                           (0.055,      0.12,        0.005, 0.1,         atan(0.005/0.12),        -0.091))

    TORSO_HIP_Z = 0.115 # 躯干与髋关节中心之间的z轴距离（所有机器人相同）
    TORSO_HIP_X = 0.01  # 躯干与髋关节中心之间的x轴距离（所有机器人相同）（髋关节在躯干后方0.01米）

    def __init__(self, robot) -> None:
        """
        初始化逆运动学类。

        参数：
        robot：机器人对象，包含机器人的类型和其他相关信息。
        """
        self.robot = robot
        self.NAO_SPECS = Inverse_Kinematics.NAO_SPECS_PER_ROBOT[robot.type]

    def torso_to_hip_transform(self, coords, is_batch=False):
        """
        将相对于躯干的笛卡尔坐标转换为相对于两个髋关节中心的坐标。

        参数：
        coords：一个3D位置或3D位置列表。
        is_batch：布尔值，表示coords是否为3D位置列表。

        返回值：
        如果is_batch为False，则返回一个numpy数组；否则返回一个数组列表。
        """
        if is_batch:
            return [c + (Inverse_Kinematics.TORSO_HIP_X, 0, Inverse_Kinematics.TORSO_HIP_Z) for c in coords]
        else:
            return coords + (Inverse_Kinematics.TORSO_HIP_X, 0, Inverse_Kinematics.TORSO_HIP_Z)
        

    def head_to_hip_transform(self, coords, is_batch=False):
        """
        将相对于头部的笛卡尔坐标转换为相对于两个髋关节中心的坐标。

        参数：
        coords：一个3D位置或3D位置列表。
        is_batch：布尔值，表示coords是否为3D位置列表。

        返回值：
        如果is_batch为False，则返回一个numpy数组；否则返回一个数组列表。
        """
        coords_rel_torso = self.robot.head_to_body_part_transform( "torso", coords, is_batch )
        return self.torso_to_hip_transform(coords_rel_torso, is_batch)

    def get_body_part_pos_relative_to_hip(self, body_part_name):
        """
        获取身体部位相对于两个髋关节中心的位置。
        """
        bp_rel_head = self.robot.body_parts[body_part_name].transform.get_translation()
        return self.head_to_hip_transform(bp_rel_head)

    def get_ankle_pos_relative_to_hip(self, is_left):
        """
        获取脚踝相对于两个髋关节中心的位置（内部调用get_body_part_pos_relative_to_hip()）。
        """
        return self.get_body_part_pos_relative_to_hip("lankle" if is_left else "rankle")

    def get_linear_leg_trajectory(self, is_left:bool, p1, p2=None, foot_ori3d=(0,0,0), dynamic_pose:bool=True, resolution=100):
        """
        计算腿部轨迹，使脚踝在两个3D点之间线性移动（相对于髋关节）。

        参数：
        is_left：布尔值，表示选择左腿（True）还是右腿（False）。
        p1：如果p2为None，则p1是目标位置（相对于髋关节），初始点由脚踝当前位置给出；如果p2不为None，则p1是初始点（相对于髋关节）。
        p2：目标位置（相对于髋关节）或None（见p1）。
        foot_ori3d：绕x、y、z轴的旋转角度（x和y轴的旋转是相对于垂直姿态或动态姿态的偏差，如果启用动态姿态）。
        dynamic_pose：布尔值，表示是否启用动态脚部旋转以与地面平行（基于IMU）。
        resolution：插值分辨率；分辨率越高越好，但计算时间也会越长；如果点过多，会在分析优化过程中移除多余的点。

        返回值：
        轨迹：一个元组，包含索引和一个列表，列表中包含多个值和错误代码对（见leg()方法）。
        """
        if p2 is None:
            p2 = np.asarray(p1, float)
            p1 = self.get_body_part_pos_relative_to_hip('lankle' if is_left else 'rankle')
        else:
            p1 = np.asarray(p1, float)
            p2 = np.asarray(p2, float)

        vec = (p2 - p1) / resolution


        hip_points = [p1 + vec * i for i in range(1,resolution+1)]
        interpolation = [self.leg(p, foot_ori3d, is_left, dynamic_pose) for p in hip_points]

        indices = [2,4,6,8,10,12] if is_left else [3,5,7,9,11,13]

        last_joint_values = self.robot.joints_position[indices[0:4]] #排除脚部关节以计算脚踝轨迹
        next_step = interpolation[0]
        trajectory = []

        for p in interpolation[1:-1]:
            if np.any(np.abs(p[1][0:4]-last_joint_values) > 7.03): 
                trajectory.append(next_step[1:3])
                last_joint_values = next_step[1][0:4]
                next_step = p
            else:
                next_step = p

        trajectory.append(interpolation[-1][1:3])

        return indices, trajectory



    def leg(self, ankle_pos3d, foot_ori3d, is_left:bool, dynamic_pose:bool):
        """
        计算腿部的逆运动学，输入是脚踝的相对3D位置和脚部的3D方向*（*脚部的偏航角可以直接控制，但俯仰角和滚转角是偏差，见下文）。

        参数：
        ankle_pos3d：脚踝的3D位置，相对于两个髋关节中心。
        foot_ori3d：绕x、y、z轴的旋转角度（x和y轴的旋转是相对于垂直姿态或动态姿态的偏差，如果启用动态姿态）。
        is_left：布尔值，表示选择左腿（True）还是右腿（False）。
        dynamic_pose：布尔值，表示是否启用动态脚部旋转以与地面平行（基于IMU）。

        返回值：
        索引：计算的关节索引列表。
        值：计算的关节值列表。
        错误代码：错误代码列表：
            (-1) 脚部距离过远（无法到达）
            (x)  关节x超出范围
        """
        error_codes = []
        leg_y_dev, upper_leg_height, upper_leg_depth, lower_leg_len, knee_extra_angle, _ = self.NAO_SPECS
        sign = -1 if is_left else 1

        # 将脚踝位置平移到腿部原点，通过平移y坐标
        ankle_pos3d = np.asarray(ankle_pos3d) + (0,sign*leg_y_dev,0)

        # 首先旋转腿部，然后旋转坐标以抽象出旋转
        ankle_pos3d = Matrix_3x3().rotate_z_deg(-foot_ori3d[2]).multiply(ankle_pos3d)

        # 使用几何解法计算膝盖角度和脚部俯仰角
        # 计算髋关节到脚踝的距离
        dist = np.linalg.norm(ankle_pos3d)  # 髋关节与脚踝之间的距离
        sq_dist = dist * dist
        sq_upper_leg_h = upper_leg_height * upper_leg_height
        sq_lower_leg_l = lower_leg_len * lower_leg_len
        sq_upper_leg_l = upper_leg_depth * upper_leg_depth + sq_upper_leg_h
        upper_leg_len = sqrt(sq_upper_leg_l)
        # 使用余弦定理计算膝盖角度
        knee = M.acos((sq_upper_leg_l + sq_lower_leg_l - sq_dist) / (2 * upper_leg_len * lower_leg_len)) + knee_extra_angle
        # 计算脚部与髋关节到脚踝向量的夹角
        foot = M.acos((sq_lower_leg_l + sq_dist - sq_upper_leg_l) / (2 * lower_leg_len * dist))

        # 检查目标是否可达
        if dist > upper_leg_len + lower_leg_len: 
            error_codes.append(-1)  # 添加错误代码：目标位置超出范围

        # 计算膝盖和脚部角度
        knee_angle = pi - knee
        foot_pitch = foot - atan(ankle_pos3d[0] / np.linalg.norm(ankle_pos3d[1:3]))  # 脚部俯仰角
        # 计算脚部滚转角，避免在z值小于-0.05时出现不稳定
        foot_roll = atan(ankle_pos3d[1] / min(-0.05, ankle_pos3d[2])) * -sign  

        # 假设所有关节都是直线运动时的原始髋关节角度
        raw_hip_yaw = foot_ori3d[2]  # 偏航角
        raw_hip_pitch = foot_pitch - knee_angle  # 俯仰角
        raw_hip_roll = -sign * foot_roll  # 滚转角

        # 由于偏航关节的方向，需要旋转45度，然后依次旋转偏航、滚转和俯仰
        m = Matrix_3x3().rotate_y_rad(raw_hip_pitch).rotate_x_rad(raw_hip_roll).rotate_z_deg(raw_hip_yaw).rotate_x_deg(-45*sign)

        # 考虑偏航关节方向后，计算实际的髋关节角度
        hip_roll = (pi/4) - (sign * asin(m.m[1,2]))  # 滚转角，加上45度的旋转
        hip_pitch = - atan2(m.m[0,2],m.m[2,2])  # 俯仰角
        hip_yaw = sign * atan2(m.m[1,0],m.m[1,1])  # 偏航角

        # 将弧度转换为度
        values = np.array([hip_yaw, hip_roll, hip_pitch, -knee_angle, foot_pitch, foot_roll]) * 57.2957795  # 弧度转度

        # 根据垂直姿态或动态姿态设置脚部旋转偏差
        values[4] -= foot_ori3d[1]  # 脚部俯仰偏差
        values[5] -= foot_ori3d[0] * sign  # 脚部滚转偏差

        indices = [2,4,6,8,10,12] if is_left else [3,5,7,9,11,13]  # 左腿或右腿的关节索引

        if dynamic_pose:
            # 根据IMU数据计算躯干相对于脚部的旋转
            m : Matrix_3x3 = Matrix_3x3.from_rotation_deg((self.robot.imu_torso_roll, self.robot.imu_torso_pitch, 0))
            m.rotate_z_deg(foot_ori3d[2], True)

            roll =  m.get_roll_deg()  # 滚转角
            pitch = m.get_pitch_deg()  # 俯仰角

            # 简单的平衡算法
            correction = 1  # 用于激励躯干保持垂直的修正值（单位：度）
            roll  = 0 if abs(roll)  < correction else roll  - np.copysign(correction,roll)  # 限制滚转角
            pitch = 0 if abs(pitch) < correction else pitch - np.copysign(correction,pitch)  # 限制俯仰角
     
            values[4] += pitch  # 调整脚部俯仰角
            values[5] += roll * sign  # 调整脚部滚转角


        # 检查并限制关节角度范围
        for i in range(len(indices)):
            if values[i] < self.robot.joints_info[indices[i]].min or values[i] > self.robot.joints_info[indices[i]].max: 
                error_codes.append(indices[i])  # 添加超出范围的关节错误代码
                values[i] = np.clip(values[i], self.robot.joints_info[indices[i]].min, self.robot.joints_info[indices[i]].max)  # 限制关节角度

        return indices, values, error_codes  # 返回关节索引、值和错误代码
    def torso_to_foot_transform(self, coords, is_left: bool, is_batch=False):
        """
        将相对于躯干的坐标转换为相对于脚部的坐标。

        参数：
        coords：一个3D位置或3D位置列表。
        is_left：布尔值，表示选择左脚（True）还是右脚（False）。
        is_batch：布尔值，表示coords是否为3D位置列表。

        返回值：
        如果is_batch为False，则返回一个numpy数组；否则返回一个数组列表。
        """
        hip_coords = self.torso_to_hip_transform(coords, is_batch)
        if is_batch:
            return [self.hip_to_foot_transform(h, is_left) for h in hip_coords]
        else:
            return self.hip_to_foot_transform(hip_coords, is_left)

    def hip_to_foot_transform(self, coords, is_left: bool):
        """
        将相对于髋关节的坐标转换为相对于脚部的坐标。

        参数：
        coords：一个3D位置。
        is_left：布尔值，表示选择左脚（True）还是右脚（False）。

        返回值：
        转换后的坐标。
        """
        leg_y_dev, _, _, _, _, _ = self.NAO_SPECS
        sign = -1 if is_left else 1
        return coords + (0, sign * leg_y_dev, 0)

    def get_foot_pos_relative_to_torso(self, is_left: bool):
        """
        获取脚部相对于躯干的位置。

        参数：
        is_left：布尔值，表示选择左脚（True）还是右脚（False）。

        返回值：
        脚部相对于躯干的位置。
        """
        foot_pos_rel_hip = self.get_body_part_pos_relative_to_hip("lankle" if is_left else "rankle")
        return self.hip_to_torso_transform(foot_pos_rel_hip)

    def hip_to_torso_transform(self, coords):
        """
        将相对于髋关节的坐标转换为相对于躯干的坐标。

        参数：
        coords：一个3D位置。

        返回值：
        转换后的坐标。
        """
        return coords - (Inverse_Kinematics.TORSO_HIP_X, 0, Inverse_Kinematics.TORSO_HIP_Z)

    def get_foot_trajectory(self, is_left: bool, p1, p2=None, foot_ori3d=(0, 0, 0), dynamic_pose=True, resolution=100):
        """
        计算脚部的轨迹，使脚部在两个3D点之间线性移动（相对于躯干）。

        参数：
        is_left：布尔值，表示选择左脚（True）还是右脚（False）。
        p1：如果p2为None，则p1是目标位置（相对于躯干），初始点由脚踝当前位置给出；如果p2不为None，则p1是初始点（相对于躯干）。
        p2：目标位置（相对于躯干）或None（见p1）。
        foot_ori3d：绕x、y、z轴的旋转角度（x和y轴的旋转是相对于垂直姿态或动态姿态的偏差，如果启用动态姿态）。
        dynamic_pose：布尔值，表示是否启用动态脚部旋转以与地面平行（基于IMU）。
        resolution：插值分辨率；分辨率越高越好，但计算时间也会越长；如果点过多，会在分析优化过程中移除多余的点。

        返回值：
        轨迹：一个元组，包含索引和一个列表，列表中包含多个值和错误代码对（见leg()方法）。
        """
        hip_p1 = self.torso_to_hip_transform(p1)
        if p2 is not None:
            hip_p2 = self.torso_to_hip_transform(p2)
        else:
            hip_p2 = None
        return self.get_linear_leg_trajectory(is_left, hip_p1, hip_p2, foot_ori3d, dynamic_pose, resolution)
