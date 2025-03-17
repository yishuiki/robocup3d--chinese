from agent.Base_Agent import Base_Agent
from behaviors.custom.Step.Step_Generator import Step_Generator
from math_ops.Math_Ops import Math_Ops as M
import math
import numpy as np

class Env():
    """
    环境类，用于提供机器人跑步的观测值和执行动作。
    """
    def __init__(self, base_agent: Base_Agent):
        """
        初始化环境。
        :param base_agent: 基础智能体对象。
        """
        self.world = base_agent.world  # 获取基础智能体的世界对象
        self.ik = base_agent.inv_kinematics  # 获取基础智能体的逆运动学对象

        # 状态空间
        self.obs = np.zeros(63, dtype=np.float32)  # 初始化观测值数组

        # 跑步行为默认参数
        self.STEP_DUR = 4  # 跑步步态持续时间（更短的步态周期）
        self.STEP_Z_SPAN = 0.05  # Z轴跨度（更大的抬腿高度）
        self.STEP_Z_MAX = 0.80  # Z轴最大值

        # 逆运动学参数
        nao_specs = self.ik.NAO_SPECS
        self.leg_length = nao_specs[1] + nao_specs[3]  # 腿的长度
        feet_y_dev = nao_specs[0] * 1.2  # 步幅宽度
        sample_time = self.world.robot.STEPTIME
        max_ankle_z = nao_specs[5]

        self.step_generator = Step_Generator(feet_y_dev, sample_time, max_ankle_z)  # 初始化步态生成器
        self.DEFAULT_ARMS = np.array([-90, -90, 8, 8, 90, 90, 70, 70], dtype=np.float32)  # 默认手臂姿势

        # 初始化内部目标和方向
        self.internal_target = np.zeros(2)
        self.internal_rel_orientation = 0.0

        # 初始化动作记忆变量
        self.act = np.zeros(16, dtype=np.float32)
        self.step_counter = 0

    def observe(self, init=False):
        """
        获取环境的观测值。
        :param init: 是否为初始化观测。
        :return: 观测值数组。
        """
        r = self.world.robot  # 获取机器人对象

        if init:
            self.act = np.zeros(16, dtype=np.float32)
            self.step_counter = 0

        # 填充观测值数组
        self.obs[0] = self.step_counter / 100  # 步态计数器
        self.obs[1] = r.loc_head_z * 3  # 躯干Z坐标
        self.obs[2] = r.loc_head_z_vel / 2  # 躯干Z速度
        self.obs[3] = r.imu_torso_roll / 15  # 躯干翻滚角
        self.obs[4] = r.imu_torso_pitch / 15  # 躯干俯仰角
        self.obs[5:8] = r.gyro / 100  # 陀螺仪数据
        self.obs[8:11] = r.acc / 10  # 加速度计数据

        # 脚部接触力和位置
        self.obs[11:17] = r.frp.get('lf', np.zeros(6)) * (10, 10, 10, 0.01, 0.01, 0.01)  # 左脚
        self.obs[17:23] = r.frp.get('rf', np.zeros(6)) * (10, 10, 10, 0.01, 0.01, 0.01)  # 右脚

        # 更新内部目标和方向
        self.obs[58] = self.internal_target[0]  # 内部目标X分量
        self.obs[59] = self.internal_target[1]  # 内部目标Y分量
        self.obs[60] = self.internal_rel_orientation  # 内部相对方向

        return self.obs

    def execute(self, action):
        """
        执行动作。
        :param action: 动作数组。
        """
        r = self.world.robot  # 获取机器人对象

        # 更新动作记忆变量
        self.act = 0.8 * self.act + 0.2 * action

        # 获取步态生成器的目标位置
        lfy, lfz, rfy, rfz = self.step_generator.get_target_positions(
            self.step_counter == 0, self.STEP_DUR, self.STEP_Z_SPAN, self.leg_length * self.STEP_Z_MAX
        )

        # 计算腿部目标位置
        l_ankle_pos = (self.act[0] * 0.02, max(0.01, self.act[1] * 0.02 + lfy), self.act[2] * 0.01 + lfz)
        r_ankle_pos = (self.act[3] * 0.02, min(-0.01, self.act[4] * 0.02 + rfy), self.act[5] * 0.01 + rfz)

        # 执行逆运动学
        self.execute_ik(l_ankle_pos, self.act[6:9], r_ankle_pos, self.act[9:12])

        # 更新手臂姿势
        arms = self.DEFAULT_ARMS + self.act[12:16] * 4
        r.set_joints_target_position_direct(slice(14, 22), arms, harmonize=False)

        # 更新步态计数器
        self.step_counter += 1

    def execute_ik(self, l_pos, l_rot, r_pos, r_rot):
        """
        执行逆运动学。
        """
        # 左腿
        indices, values, errors = self.ik.leg(l_pos, l_rot, True, dynamic_pose=False)
        self.world.robot.set_joints_target_position_direct(indices, values)

        # 右腿
        indices, values, errors = self.ik.leg(r_pos, r_rot, False, dynamic_pose=False)
        self.world.robot.set_joints_target_position_direct(indices, values)