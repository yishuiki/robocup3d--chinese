from agent.Base_Agent import Base_Agent
from behaviors.custom.Step.Step_Generator import Step_Generator
from math_ops.Math_Ops import Math_Ops as M
import math
import numpy as np

class Env():
    """
    Env 类实现了一个用于控制机器人行走的环境，包括状态观测和动作执行。
    """

    def __init__(self, base_agent: Base_Agent, step_width) -> None:
        """
        初始化环境。

        参数：
        - base_agent: Base_Agent 类型，基础智能体对象，用于访问机器人和世界状态。
        - step_width: float 类型，步幅宽度。
        """
        self.world = base_agent.world  # 获取世界状态
        self.ik = base_agent.inv_kinematics  # 获取逆运动学模块

        # 状态空间
        self.obs = np.zeros(76, np.float32)  # 初始化观测值数组

        # 步行动作的默认参数
        self.STEP_DUR = 8  # 步长持续时间
        self.STEP_Z_SPAN = 0.02  # 垂直移动范围
        self.STEP_Z_MAX = 0.70  # 最大 Z 轴高度

        # 逆运动学
        r = self.world.robot  # 获取机器人对象
        nao_specs = self.ik.NAO_SPECS  # 获取 NAO 机器人的规格参数
        self.leg_length = nao_specs[1] + nao_specs[3]  # 大腿和小腿的总长度
        feet_y_dev = nao_specs[0] * step_width  # 步幅宽度
        sample_time = r.STEPTIME  # 采样时间
        max_ankle_z = nao_specs[5]  # 踝关节的最大 Z 轴高度

        # 初始化步态生成器
        self.step_generator = Step_Generator(feet_y_dev, sample_time, max_ankle_z)
        self.DEFAULT_ARMS = np.array([-90, -90, 8, 8, 90, 90, 70, 70], np.float32)  # 默认手臂姿态

        self.dribble_rel_orientation = None  # 带球的相对方向（相对于躯干 IMU）
        self.dribble_speed = 1  # 带球速度


    def observe(self, init=False, virtual_ball=False):
        """
        获取环境的观测值。

        参数：
        - init: bool 类型，是否为初始化状态。
        - virtual_ball: bool 类型，是否使用虚拟球（用于训练）。
        """
        w = self.world  # 获取世界状态
        r = self.world.robot  # 获取机器人对象

        if init:  # 重置变量
            self.step_counter = 0  # 步数计数器
            self.act = np.zeros(16, np.float32)  # 动作记忆变量

        # index       observation              naive normalization
        self.obs[0] = min(self.step_counter, 12 * 8) / 100  # 简单计数器：0,1,2,3...
        self.obs[1] = r.loc_head_z * 3  # 头部 Z 轴坐标（躯干）
        self.obs[2] = r.loc_head_z_vel / 2  # 头部 Z 轴速度（躯干）
        self.obs[3] = r.imu_torso_roll / 15  # 躯干滚转角（单位：度）
        self.obs[4] = r.imu_torso_pitch / 15  # 躯干俯仰角（单位：度）
        self.obs[5:8] = r.gyro / 100  # 陀螺仪数据
        self.obs[8:11] = r.acc / 10  # 加速度计数据

        self.obs[11:17] = r.frp.get('lf', np.zeros(6)) * (10, 10, 10, 0.01, 0.01, 0.01)  # 左脚：相对原点位置和力向量
        self.obs[17:23] = r.frp.get('rf', np.zeros(6)) * (10, 10, 10, 0.01, 0.01, 0.01)  # 右脚：相对原点位置和力向量
        # *如果脚未接触地面，则 (px=0,py=0,pz=0,fx=0,fy=0,fz=0)

        self.obs[23:43] = r.joints_position[2:22] / 100  # 除头部和脚趾外的所有关节位置
        self.obs[43:63] = r.joints_speed[2:22] / 6.1395  # 除头部和脚趾外的所有关节速度

        '''
        预期的行走状态观测值：
        Time step        R  0   1   2   3   4   5   6   7   0
        Progress         1  0 .14 .28 .43 .57 .71 .86   1   0
        Left leg active  T  F   F   F   F   F   F   F   F   T
        '''

        if init:  # 行走参数在重置后无效
            self.obs[63] = 1  # 步行进度
            self.obs[64] = 1  # 左腿是否活跃
            self.obs[65] = 0  # 右腿是否活跃
            self.obs[66] = 0
        else:
            self.obs[63] = self.step_generator.external_progress  # 步行进度
            self.obs[64] = float(self.step_generator.state_is_left_active)  # 左腿是否活跃
            self.obs[65] = float(not self.step_generator.state_is_left_active)  # 右腿是否活跃
            self.obs[66] = math.sin(self.step_generator.state_current_ts / self.step_generator.ts_per_step * math.pi)

        # 球
        ball_rel_hip_center = self.ik.torso_to_hip_transform(w.ball_rel_torso_cart_pos)
        ball_dist_hip_center = np.linalg.norm(ball_rel_hip_center)

        if init:
            self.obs[67:70] = (0, 0, 0)  # 初始速度为 0
        elif w.ball_is_visible:
            self.obs[67:70] = (ball_rel_hip_center - self.obs[70:73]) * 10  # 球的速度，相对于踝关节中点

        self.obs[70:73] = ball_rel_hip_center  # 球的位置，相对于髋关节
        self.obs[73] = ball_dist_hip_center * 2

        if virtual_ball:  # 在机器人双脚之间模拟球
            self.obs[67:74] = (0, 0, 0, 0.05, 0, -0.175, 0.36)

        '''
        创建内部目标，使变化更加平滑
        '''

        MAX_ROTATION_DIFF = 20  # 每个视觉步骤的最大角度差（单位：度）
        MAX_ROTATION_DIST = 80

        if init:
            self.internal_rel_orientation = 0
            self.internal_target_vel = 0
            self.gym_last_internal_abs_ori = r.imu_torso_orientation  # 用于训练（奖励）

        #---------------------------------------------------------------- 计算内部目标

        if w.vision_is_up_to_date:

            previous_internal_rel_orientation = np.copy(self.internal_rel_orientation)

            internal_ori_diff = np.clip(M.normalize_deg(self.dribble_rel_orientation - self.internal_rel_orientation), -MAX_ROTATION_DIFF, MAX_ROTATION_DIFF)
            self.internal_rel_orientation = np.clip(M.normalize_deg(self.internal_rel_orientation + internal_ori_diff), -MAX_ROTATION_DIST, MAX_ROTATION_DIST)

            # 观测值
            self.internal_target_vel = self.internal_rel_orientation - previous_internal_rel_orientation

            self.gym_last_internal_abs_ori = self.internal_rel_orientation + r.imu_torso_orientation

        #----------------------------------------------------------------- 观测值

        self.obs[74] = self.internal_rel_orientation / MAX_ROTATION_DIST
        self.obs[75] = self.internal_target_vel / MAX_ROTATION_DIFF

        return self.obs


    def execute_ik(self, l_pos, l_rot, r_pos, r_rot):
        """
        对每条腿应用逆运动学并设置关节目标位置。

        参数：
        - l_pos: 左脚踝的目标位置。
        - l_rot: 左脚踝的目标旋转。
        - r_pos: 右脚踝的目标位置。
        - r_rot: 右脚踝的目标旋转。
        """
        r = self.world.robot  # 获取机器人对象
        # 对每条腿应用逆运动学 + 设置关节目标位置
          
        # 左腿 
        indices, self.values_l, error_codes = self.ik.leg(l_pos, l_rot, True, dynamic_pose=False)
        for i in error_codes:
            if i != -1:
                print(f"Joint {i} is out of range!")  # 如果关节超出范围
            else:
                print("Position is out of reach!")  # 如果位置不可达

        r.set_joints_target_position_direct(indices, self.values_l, harmonize=False)

        # 右腿
        indices, self.values_r, error_codes = self.ik.leg(r_pos, r_rot, False, dynamic_pose=False)
        for i in error_codes:
            if i != -1:
                print(f"Joint {i} is out of range!")  # 如果关节超出范围
            else:
                print("Position is out of reach!")  # 如果位置不可达

        r.set_joints_target_position_direct(indices, self.values_r, harmonize=False)


    def execute(self, action):
        """
        执行动作。

        参数：
        - action: 动作向量。
        """
        r = self.world.robot  # 获取机器人对象

        # 动作：
        # 0,1,2    左脚踝位置
        # 3,4,5    右脚踝位置
        # 6,7,8    左脚旋转
        # 9,10,11  右脚旋转
        # 12,13    左/右手臂俯仰角
        # 14,15    左/右手臂翻滚角

        # 指数移动平均
        self.act = 0.85 * self.act + 0.15 * action * 0.7 * 0.95 * self.dribble_speed

        # 执行步行动作以获取每条腿的目标位置（我们将覆盖这些目标）
        lfy, lfz, rfy, rfz = self.step_generator.get_target_positions(self.step_counter == 0, self.STEP_DUR, self.STEP_Z_SPAN, self.leg_length * self.STEP_Z_MAX)

        # 腿部逆运动学
        a = self.act
        l_ankle_pos = (a[0] * 0.025 - 0.01, a[1] * 0.01 + lfy, a[2] * 0.01 + lfz)  # 左脚踝位置
        r_ankle_pos = (a[3] * 0.025 - 0.01, a[4] * 0.01 + rfy, a[5] * 0.01 + rfz)  # 右脚踝位置
        l_foot_rot = a[6:9] * (2, 2, 3)  # 左脚旋转
        r_foot_rot = a[9:12] * (2, 2, 3)  # 右脚旋转

        # 限制腿部偏航角/俯仰角（并添加偏差）
        l_foot_rot[2] = max(0, l_foot_rot[2] + 18.3)  # 左脚偏航角
        r_foot_rot[2] = min(0, r_foot_rot[2] - 18.3)  # 右脚偏航角

        # 手臂动作
        arms = np.copy(self.DEFAULT_ARMS)  # 默认手臂姿态
        arms[0:4] += a[12:16] * 4  # 手臂俯仰角 + 翻滚角

        # 设置目标位置
        self.execute_ik(l_ankle_pos, l_foot_rot, r_ankle_pos, r_foot_rot)  # 腿部
        r.set_joints_target_position_direct(slice(14, 22), arms, harmonize=False)  # 手臂

        self.step_counter += 1  # 步数计数器加 1