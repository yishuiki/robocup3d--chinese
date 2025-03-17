from agent.Base_Agent import Base_Agent
from behaviors.custom.Walk.Env import Env
from math_ops.Math_Ops import Math_Ops as M
from math_ops.Neural_Network import run_mlp
import numpy as np
import pickle

class Walk():
    """
    Walk 类实现了一个全向行走行为，使用强化学习模型控制机器人的行走。
    """

    def __init__(self, base_agent: Base_Agent) -> None:
        """
        初始化 Walk 行为。

        参数：
        - base_agent: Base_Agent 类型，基础智能体对象，用于访问机器人和世界状态。
        """
        self.world = base_agent.world  # 获取世界状态
        self.description = "Omnidirectional RL walk"  # 行为描述
        self.auto_head = True  # 是否自动控制头部
        self.env = Env(base_agent)  # 初始化环境对象
        self.last_executed = 0  # 上一次执行的时间戳

        # 根据机器人类型加载预训练的强化学习模型
        with open(M.get_active_directory([
            "/behaviors/custom/Walk/walk_R0.pkl",
            "/behaviors/custom/Walk/walk_R1_R3.pkl",
            "/behaviors/custom/Walk/walk_R2.pkl",
            "/behaviors/custom/Walk/walk_R1_R3.pkl",
            "/behaviors/custom/Walk/walk_R4.pkl"
        ][self.world.robot.type]), 'rb') as f:
            self.model = pickle.load(f)  # 加载模型

    def execute(self, reset, target_2d, is_target_absolute, orientation, is_orientation_absolute, distance):
        """
        执行行走行为。

        参数：
        - reset: bool 类型，是否重置行为。
        - target_2d: array_like 类型，二维目标位置（绝对或相对坐标，具体取决于 is_target_absolute 参数）。
        - is_target_absolute: bool 类型，是否为绝对坐标。如果为 True，则 target_2d 为绝对坐标；如果为 False，则为相对于机器人躯干的相对坐标。
        - orientation: float 类型，躯干的方向（绝对或相对方向，单位：度）。如果设置为 None，则机器人将朝向目标（忽略 is_orientation_absolute 参数）。
        - is_orientation_absolute: bool 类型，是否为绝对方向。如果为 True，则 orientation 为相对于场地的方向；如果为 False，则为相对于机器人躯干的相对方向。
        - distance: float 类型，到最终目标的距离（范围：[0, 0.5]，在接近最终目标时影响行走速度）。如果设置为 None，则将 target_2d 视为最终目标。
        """
        r = self.world.robot  # 获取机器人对象

        #------------------------ 0. 重置行为（因为某些行为可能将此行为作为子行为使用）
        if reset and self.world.time_local_ms - self.last_executed == 20:
            reset = False  # 如果在 20 毫秒内重复调用 reset，则忽略重置
        self.last_executed = self.world.time_local_ms  # 更新最后一次执行的时间

        #------------------------ 1. 定义行走参数
        if is_target_absolute:  # 如果目标是绝对坐标
            raw_target = target_2d - r.loc_head_position[:2]  # 计算目标相对于头部的位置
            self.env.walk_rel_target = M.rotate_2d_vec(raw_target, -r.imu_torso_orientation)  # 转换为相对于躯干的方向
        else:
            self.env.walk_rel_target = target_2d  # 如果目标是相对坐标，直接使用

        if distance is None:  # 如果未指定距离
            self.env.walk_distance = np.linalg.norm(self.env.walk_rel_target)  # 计算目标距离
        else:
            self.env.walk_distance = distance  # 使用指定的距离

        if orientation is None:  # 如果未指定方向
            self.env.walk_rel_orientation = M.vector_angle(self.env.walk_rel_target) * 0.3  # 计算目标方向并减小值以避免过度调整
        elif is_orientation_absolute:  # 如果方向是绝对的
            self.env.walk_rel_orientation = M.normalize_deg(orientation - r.imu_torso_orientation)  # 转换为相对于躯干的方向
        else:
            self.env.walk_rel_orientation = orientation * 0.3  # 如果方向是相对的，直接使用并减小值

        #------------------------ 2. 执行行为
        obs = self.env.observe(reset)  # 获取环境观测值
        action = run_mlp(obs, self.model)  # 使用强化学习模型生成动作
        self.env.execute(action)  # 执行动作

        return False  # 返回 False 表示行为未完成

    def is_ready(self):
        """
        检查 Walk 行为是否准备好在当前游戏/机器人条件下开始。

        返回：
        - bool 类型，如果行为准备好，则返回 True。
        """
        return True  # 假设行为总是准备好