from agent.Base_Agent import Base_Agent
from behaviors.custom.Step.Step_Generator import Step_Generator
from math_ops.Math_Ops import Math_Ops as M
import math
import numpy as np


class Env():
    def __init__(self, base_agent: Base_Agent) -> None:
        """
        环境初始化函数。
        :param base_agent: 基础智能体对象。
        """
        self.world = base_agent.world  # 获取基础智能体的世界对象
        self.ik = base_agent.inv_kinematics  # 获取基础智能体的逆运动学对象

        # 状态空间
        self.obs = np.zeros(63, np.float32)  # 初始化观测值数组，长度为63

        # 步态行为默认参数
        self.STEP_DUR = 8  # 步态持续时间
        self.STEP_Z_SPAN = 0.02  # Z轴跨度
        self.STEP_Z_MAX = 0.70  # Z轴最大值

        # 逆运动学参数
        nao_specs = self.ik.NAO_SPECS  # 获取NAO机器人的规格参数
        self.leg_length = nao_specs[1] + nao_specs[3]  # 腿的长度（大腿高度 + 小腿高度）
        feet_y_dev = nao_specs[0] * 1.12  # 步幅宽度
        sample_time = self.world.robot.STEPTIME  # 采样时间
        max_ankle_z = nao_specs[5]  # 踝关节Z轴最大值

        self.step_generator = Step_Generator(feet_y_dev, sample_time, max_ankle_z)  # 初始化步态生成器
        self.DEFAULT_ARMS = np.array([-90, -90, 8, 8, 90, 90, 70, 70], np.float32)  # 默认手臂姿势

        self.walk_rel_orientation = None  # 行走相对方向
        self.walk_rel_target = None  # 行走相对目标
        self.walk_distance = None  # 行走距离


    def observe(self, init=False):
        """
        观测函数，用于获取环境的状态信息。
        :param init: 是否为初始化观测。
        :return: 观测值数组。
        """
        r = self.world.robot  # 获取机器人对象

        if init:  # 如果是初始化观测，重置变量
            self.act = np.zeros(16, np.float32)  # 初始化动作记忆变量
            self.step_counter = 0  # 初始化步态计数器

        # 索引       观测值                     归一化因子
        self.obs[0] = min(self.step_counter, 15 * 8) / 100  # 步态计数器（简单计数：0,1,2,3...）
        self.obs[1] = r.loc_head_z * 3  # 躯干Z坐标
        self.obs[2] = r.loc_head_z_vel / 2  # 躯干Z速度
        self.obs[3] = r.imu_torso_roll / 15  # 躯干翻滚角（绝对值，单位：度）
        self.obs[4] = r.imu_torso_pitch / 15  # 躯干俯仰角（绝对值，单位：度）
        self.obs[5:8] = r.gyro / 100  # 陀螺仪数据
        self.obs[8:11] = r.acc / 10  # 加速度计数据

        self.obs[11:17] = r.frp.get('lf', np.zeros(6)) * (10, 10, 10, 0.01, 0.01, 0.01)  # 左脚：相对原点位置（p）和力向量（f）-> (px,py,pz,fx,fy,fz)
        self.obs[17:23] = r.frp.get('rf', np.zeros(6)) * (10, 10, 10, 0.01, 0.01, 0.01)  # 右脚：相对原点位置（p）和力向量（f）-> (px,py,pz,fx,fy,fz)
        # 如果脚没有接触地面，则 (px=0,py=0,pz=0,fx=0,fy=0,fz=0)

        # 关节：踝关节的前向运动学 + 脚部旋转 + 手臂（俯仰角 + 横滚角）
        rel_lankle = self.ik.get_body_part_pos_relative_to_hip("lankle")  # 左踝关节相对于髋关节中心的位置
        rel_rankle = self.ik.get_body_part_pos_relative_to_hip("rankle")  # 右踝关节相对于髋关节中心的位置
        lf = r.head_to_body_part_transform("torso", r.body_parts['lfoot'].transform)  # 左脚相对于躯干的变换
        rf = r.head_to_body_part_transform("torso", r.body_parts['rfoot'].transform)  # 右脚相对于躯干的变换
        lf_rot_rel_torso = np.array([lf.get_roll_deg(), lf.get_pitch_deg(), lf.get_yaw_deg()])  # 左脚相对于躯干的旋转
        rf_rot_rel_torso = np.array([rf.get_roll_deg(), rf.get_pitch_deg(), rf.get_yaw_deg()])  # 右脚相对于躯干的旋转

        # 姿态
        self.obs[23:26] = rel_lankle * (8, 8, 5)
        self.obs[26:29] = rel_rankle * (8, 8, 5)
        self.obs[29:32] = lf_rot_rel_torso / 20
        self.obs[32:35] = rf_rot_rel_torso / 20
        self.obs[35:39] = r.joints_position[14:18] / 100  # 手臂（俯仰角 + 横滚角）

        # 速度
        self.obs[39:55] = r.joints_target_last_speed[2:18]  # 预测值 == 上一次动作

        '''
        行走状态的预期观测值：
        时间步        R  0   1   2   3   4   5   6   7   0
        进度          1  0 .14 .28 .43 .57 .71 .86   1   0
        左腿活动      T  F   F   F   F   F   F   F   F   T
        '''

        if init:  # 如果是初始化观测，行走参数引用最后生效的参数（在重置后，这些参数没有意义）
            self.obs[55] = 1  # 步态进度
            self.obs[56] = 1  # 左腿活动（1 表示活动）
            self.obs[57] = 0  # 右腿活动（1 表示活动）
        else:
            self.obs[55] = self.step_generator.external_progress  # 步态进度
            self.obs[56] = float(self.step_generator.state_is_left_active)  # 左腿活动（1 表示活动）
            self.obs[57] = float(not self.step_generator.state_is_left_active)  # 右腿活动（1 表示活动）

        '''
        创建内部目标，使其变化更加平滑
        '''

        MAX_LINEAR_DIST = 0.5  # 最大线性距离
        MAX_LINEAR_DIFF = 0.014  # 每步最大线性差异（单位：米）
        MAX_ROTATION_DIFF = 1.6  # 每步最大旋转差异（单位：度）
        MAX_ROTATION_DIST = 45  # 最大旋转距离

        if init:  # 如果是初始化观测
            self.internal_rel_orientation = 0  # 初始化内部相对方向
            self.internal_target = np.zeros(2)  # 初始化内部目标

        previous_internal_target = np.copy(self.internal_target)  # 复制上一次内部目标

        #---------------------------------------------------------------- 计算内部线性目标

        rel_raw_target_size = np.linalg.norm(self.walk_rel_target)  # 相对原始目标的大小

        if rel_raw_target_size == 0:  # 如果相对原始目标大小为0
            rel_target = self.walk_rel_target
        else:
            rel_target = self.walk_rel_target / rel_raw_target_size * min(self.walk_distance, MAX_LINEAR_DIST)  # 归一化目标

        internal_diff = rel_target - self.internal_target  # 内部目标差异
        internal_diff_size = np.linalg.norm(internal_diff)  # 内部目标差异的大小

        if internal_diff_size > MAX_LINEAR_DIFF:  # 如果差异超过最大线性差异
            self.internal_target += internal_diff * (MAX_LINEAR_DIFF / internal_diff_size)  # 按比例调整内部目标
        else:
            self.internal_target[:] = rel_target  # 直接更新内部目标

        #---------------------------------------------------------------- 计算内部旋转目标

        internal_ori_diff = np.clip(M.normalize_deg(self.walk_rel_orientation - self.internal_rel_orientation), 
                                    -MAX_ROTATION_DIFF, MAX_ROTATION_DIFF)  # 限制旋转差异范围
        self.internal_rel_orientation = np.clip(M.normalize_deg(self.internal_rel_orientation + internal_ori_diff), 
                                                -MAX_ROTATION_DIST, MAX_ROTATION_DIST)  # 更新内部相对方向

        #---------------------------------------------------------------- 观测值更新

        internal_target_vel = self.internal_target - previous_internal_target  # 内部目标速度

        self.obs[58] = self.internal_target[0] / MAX_LINEAR_DIST  # 内部目标X分量
        self.obs[59] = self.internal_target[1] / MAX_LINEAR_DIST  # 内部目标Y分量
        self.obs[60] = self.internal_rel_orientation / MAX_ROTATION_DIST  # 内部相对方向
        self.obs[61] = internal_target_vel[0] / MAX_LINEAR_DIFF  # 内部目标速度X分量
        self.obs[62] = internal_target_vel[1] / MAX_LINEAR_DIFF  # 内部目标速度Y分量

        return self.obs  # 返回观测值数组


    def execute_ik(self, l_pos, l_rot, r_pos, r_rot):
        """
        执行逆运动学函数，用于计算关节目标位置。
        :param l_pos: 左腿目标位置。
        :param l_rot: 左腿目标旋转。
        :param r_pos: 右腿目标位置。
        :param r_rot: 右腿目标旋转。
        """
        r = self.world.robot  # 获取机器人对象

        # 左腿逆运动学
        indices, self.values_l, error_codes = self.ik.leg(l_pos, l_rot, True, dynamic_pose=False)  # 计算左腿关节目标位置
        r.set_joints_target_position_direct(indices, self.values_l, harmonize=False)  # 设置左腿关节目标位置

        # 右腿逆运动学
        indices, self.values_r, error_codes = self.ik.leg(r_pos, r_rot, False, dynamic_pose=False)  # 计算右腿关节目标位置
        r.set_joints_target_position_direct(indices, self.values_r, harmonize=False)  # 设置右腿关节目标位置


    def execute(self, action):
        """
        执行动作函数，用于根据动作更新机器人的关节目标位置。
        :param action: 动作数组。
        """
        r = self.world.robot  # 获取机器人对象

        # 动作解析：
        # 0,1,2    左脚踝位置
        # 3,4,5    右脚踝位置
        # 6,7,8    左脚旋转
        # 9,10,11  右脚旋转
        # 12,13    左/右手臂俯仰角
        # 14,15    左/右手臂横滚角

        internal_dist = np.linalg.norm(self.internal_target)  # 内部目标距离
        action_mult = 1 if internal_dist > 0.2 else (0.7 / 0.2) * internal_dist + 0.3  # 动作缩放因子

        # 指数移动平均
        self.act = 0.8 * self.act + 0.2 * action * action_mult * 0.7  # 更新动作记忆变量

        # 执行步态行为以获取每条腿的目标位置（我们将覆盖这些目标）
        lfy, lfz, rfy, rfz = self.step_generator.get_target_positions(self.step_counter == 0, self.STEP_DUR, 
                                                                      self.STEP_Z_SPAN, self.leg_length * self.STEP_Z_MAX)

        # 腿部逆运动学
        a = self.act
        l_ankle_pos = (a[0] * 0.02, max(0.01, a[1] * 0.02 + lfy), a[2] * 0.01 + lfz)  # 限制Y值以避免自碰撞
        r_ankle_pos = (a[3] * 0.02, min(a[4] * 0.02 + rfy, -0.01), a[5] * 0.01 + rfz)  # 限制Y值以避免自碰撞
        l_foot_rot = a[6:9] * (3, 3, 5)
        r_foot_rot = a[9:12] * (3, 3, 5)

        # 限制腿部偏航角/俯仰角
        l_foot_rot[2] = max(0, l_foot_rot[2] + 7)
        r_foot_rot[2] = min(0, r_foot_rot[2] - 7)

        # 手臂动作
        arms = np.copy(self.DEFAULT_ARMS)  # 默认手臂姿势
        arm_swing = math.sin(self.step_generator.state_current_ts / self.STEP_DUR * math.pi) * 6  # 手臂摆动幅度
        inv = 1 if self.step_generator.state_is_left_active else -1  # 左腿活动时为1，右腿活动时为-1
        arms[0:4] += a[12:16] * 4 + (-arm_swing * inv, arm_swing * inv, 0, 0)  # 更新手臂俯仰角和横滚角

        # 设置目标位置
        self.execute_ik(l_ankle_pos, l_foot_rot, r_ankle_pos, r_foot_rot)  # 腿部
        r.set_joints_target_position_direct(slice(14, 22), arms, harmonize=False)  # 手臂

        self.step_counter += 1  # 步态计数器加1