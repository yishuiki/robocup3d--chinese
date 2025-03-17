from agent.Base_Agent import Base_Agent
from behaviors.custom.Dribble.Env import Env
from math_ops.Math_Ops import Math_Ops as M
from math_ops.Neural_Network import run_mlp
import numpy as np
import pickle


class Dribble():
    """
    Dribble 类实现了一个基于强化学习的带球行为，包括走到球的位置并执行带球动作。
    """

    def __init__(self, base_agent: Base_Agent) -> None:
        """
        初始化 Dribble 行为。

        参数：
        - base_agent: Base_Agent 类型，基础智能体对象，用于访问机器人和世界状态。
        """
        self.behavior = base_agent.behavior  # 获取行为管理器
        self.path_manager = base_agent.path_manager  # 获取路径管理器
        self.world = base_agent.world  # 获取世界状态
        self.description = "RL dribble"  # 行为描述
        self.auto_head = True  # 是否自动控制头部
        self.env = Env(base_agent, 0.9 if self.world.robot.type == 3 else 1.2)  # 初始化环境对象

        # 根据机器人类型加载预训练的强化学习模型
        with open(M.get_active_directory([
            "/behaviors/custom/Dribble/dribble_R0.pkl",
            "/behaviors/custom/Dribble/dribble_R1.pkl",
            "/behaviors/custom/Dribble/dribble_R2.pkl",
            "/behaviors/custom/Dribble/dribble_R3.pkl",
            "/behaviors/custom/Dribble/dribble_R4.pkl"
        ][self.world.robot.type]), 'rb') as f:
            self.model = pickle.load(f)  # 加载模型


    def define_approach_orientation(self):
        """
        定义接近球的方向。
        """
        w = self.world  # 获取世界状态
        b = w.ball_abs_pos[:2]  # 球的绝对位置
        me = w.robot.loc_head_position[:2]  # 机器人头部的位置

        self.approach_orientation = None  # 初始化接近方向

        MARGIN = 0.8  # 安全边距（如果球接近场地边界，考虑接近方向）
        M90 = 90 / MARGIN  # 辅助变量
        DEV = 25  # 当站在边线或端线上时，接近方向偏离该线的角度
        MDEV = (90 + DEV) / MARGIN  # 辅助变量

        a1 = -180  # 角度范围起始值（逆时针旋转）
        a2 = 180  # 角度范围结束值（逆时针旋转）

        # 根据球的位置调整接近方向
        if b[1] < -10 + MARGIN:
            if b[0] < -15 + MARGIN:
                a1 = DEV - M90 * (b[1] + 10)
                a2 = 90 - DEV + M90 * (b[0] + 15)
            elif b[0] > 15 - MARGIN:
                a1 = 90 + DEV - M90 * (15 - b[0])
                a2 = 180 - DEV + M90 * (b[1] + 10)
            else:
                a1 = DEV - MDEV * (b[1] + 10)
                a2 = 180 - DEV + MDEV * (b[1] + 10)
        elif b[1] > 10 - MARGIN:
            if b[0] < -15 + MARGIN:
                a1 = -90 + DEV - M90 * (b[0] + 15)
                a2 = -DEV + M90 * (10 - b[1])
            elif b[0] > 15 - MARGIN:
                a1 = 180 + DEV - M90 * (10 - b[1])
                a2 = 270 - DEV + M90 * (15 - b[0])
            else:
                a1 = -180 + DEV - MDEV * (10 - b[1])
                a2 = -DEV + MDEV * (10 - b[1])
        elif b[0] < -15 + MARGIN:
            a1 = -90 + DEV - MDEV * (b[0] + 15)
            a2 = 90 - DEV + MDEV * (b[0] + 15)
        elif b[0] > 15 - MARGIN and abs(b[1]) > 1.2:
            a1 = 90 + DEV - MDEV * (15 - b[0])
            a2 = 270 - DEV + MDEV * (15 - b[0])

        cad = M.vector_angle(b - me)  # 当前接近方向

        a1 = M.normalize_deg(a1)  # 角度归一化
        a2 = M.normalize_deg(a2)  # 角度归一化

        # 检查当前接近方向是否在允许范围内
        if a1 < a2:
            if a1 <= cad <= a2:
                return  # 当前接近方向在允许范围内
        else:
            if a1 <= cad or cad <= a2:
                return  # 当前接近方向在允许范围内

        a1_diff = abs(M.normalize_deg(a1 - cad))
        a2_diff = abs(M.normalize_deg(a2 - cad))

        self.approach_orientation = a1 if a1_diff < a2_diff else a2  # 选择更接近的角度


    def execute(self, reset, orientation, is_orientation_absolute, speed=1, stop=False):
        """
        执行带球行为。

        参数：
        - reset: bool 类型，是否重置行为。
        - orientation: float 类型，躯干的方向（绝对或相对方向，单位：度）。
        - is_orientation_absolute: bool 类型，是否为绝对方向。如果为 True，则 orientation 为相对于场地的方向；如果为 False，则为相对于机器人躯干的相对方向。
        - speed: float 类型，速度范围为 0 到 1（非线性比例）。
        - stop: bool 类型，如果为 True，立即返回 True（如果正在行走），或者在带球完成后返回 True。
        """
        w = self.world  # 获取世界状态
        r = self.world.robot  # 获取机器人对象
        me = r.loc_head_position[:2]  # 机器人头部的位置
        b = w.ball_abs_pos[:2]  # 球的绝对位置
        b_rel = w.ball_rel_torso_cart_pos[:2]  # 球相对于躯干的位置
        b_dist = np.linalg.norm(b - me)  # 球与机器人的距离
        behavior = self.behavior  # 获取行为管理器
        reset_dribble = False  # 是否重置带球行为
        lost_ball = (w.ball_last_seen <= w.time_local_ms - w.VISUALSTEP_MS) or np.linalg.norm(b_rel) > 0.4  # 是否丢失球

        if reset:
            self.phase = 0  # 重置行为阶段
            if behavior.previous_behavior == "Push_RL" and 0 < b_rel[0] < 0.25 and abs(b_rel[1]) < 0.07:
                self.phase = 1  # 如果上一个行为是 Push_RL 且球在合适位置，直接进入带球阶段
                reset_dribble = True

        if self.phase == 0:  # 走到球的位置
            reset_walk = reset and behavior.previous_behavior not in ["Walk", "Push_RL"]  # 如果上一个行为不是 Walk 或 Push_RL，则重置 Walk

            #------------------------ 1. 决定是否需要更好的接近方向（当球接近场地边界时）
            if reset or b_dist > 0.4:  # 在接近球之前停止定义方向，以避免噪声
                self.define_approach_orientation()

            #------------------------ 2A. 需要更好的接近方向（球接近场地边界）
            if self.approach_orientation is not None:
                next_pos, next_ori, dist_to_final_target = self.path_manager.get_path_to_ball(
                    x_ori=self.approach_orientation, x_dev=-0.24, torso_ori=self.approach_orientation, safety_margin=0.4)

                if b_rel[0] < 0.26 and b_rel[0] > 0.18 and abs(b_rel[1]) < 0.04 and w.ball_is_visible:  # 准备开始带球
                    self.phase += 1
                    reset_dribble = True
                else:
                    dist = max(0.08, dist_to_final_target * 0.7)
                    behavior.execute_sub_behavior("Walk", reset_walk, next_pos, True, next_ori, True, dist)  # target, is_target_abs, ori, is_ori_abs, distance

            #------------------------ 2B. 不需要更好的接近方向，但机器人看不到球
            elif w.time_local_ms - w.ball_last_seen > 200:  # 如果球不可见，走到球的绝对位置
                abs_ori = M.vector_angle(b - me)  # 计算球的绝对方向
                behavior.execute_sub_behavior("Walk", reset_walk, b, True, abs_ori, True, None)  # target, is_target_abs, ori, is_ori_abs, distance

            #------------------------ 2C. 不需要更好的接近方向，且机器人可以看到球
            else:  # 走到相对目标位置
                if 0.18 < b_rel[0] < 0.25 and abs(b_rel[1]) < 0.05 and w.ball_is_visible:  # 准备开始带球
                    self.phase += 1
                    reset_dribble = True
                else:
                    rel_target = b_rel + (-0.23, 0)  # 相对目标位置（以球为中心的圆，半径为 0.23 米）
                    rel_ori = M.vector_angle(b_rel)  # 球的相对方向
                    dist = max(0.08, np.linalg.norm(rel_target) * 0.7)  # 缓慢接近
                    behavior.execute_sub_behavior("Walk", reset_walk, rel_target, False, rel_ori, False, dist)  # target, is_target_abs, ori, is_ori_abs, distance

            if stop:
                return True

        if self.phase == 1 and (stop or (b_dist > 0.5 and lost_ball)):  # 如果停止或球丢失，回到行走阶段
            self.phase += 1
        elif self.phase == 1:  # 执行带球动作
            #------------------------ 1. 定义带球参数
            self.env.dribble_speed = speed  # 设置带球速度

            # 如果方向为 None，则向对手球门带球
            if orientation is None:
                if b[0] < 0:  # 向两侧带球
                    if b[1] > 0:
                        dribble_target = (15, 5)
                    else:
                        dribble_target = (15, -5)
                else:
                    dribble_target = None  # 向球门带球
                self.env.dribble_rel_orientation = self.path_manager.get_dribble_path(optional_2d_target=dribble_target)[1]
            elif is_orientation_absolute:
                self.env.dribble_rel_orientation = M.normalize_deg(orientation - r.imu_torso_orientation)  # 转换为相对方向
            else:
                self.env.dribble_rel_orientation = float(orientation)  # 使用传入的方向值

            #------------------------ 2. 执行带球行为
            obs = self.env.observe(reset_dribble)  # 获取环境观测值
            action = run_mlp(obs, self.model)  # 使用强化学习模型生成动作
            self.env.execute(action)  # 执行动作

        # 带球动作结束，逐渐减速并重置阶段
        if self.phase > 1:
            WIND_DOWN_STEPS = 60  # 减速步数
            #------------------------ 1. 定义带球减速参数
            self.env.dribble_speed = 1 - self.phase / WIND_DOWN_STEPS  # 逐渐降低速度
            self.env.dribble_rel_orientation = 0  # 设置相对方向为 0

            #------------------------ 2. 执行带球行为
            obs = self.env.observe(reset_dribble, virtual_ball=True)  # 获取环境观测值
            action = run_mlp(obs, self.model)  # 使用强化学习模型生成动作
            self.env.execute(action)  # 执行动作

            #------------------------ 3. 重置行为
            self.phase += 1
            if self.phase >= WIND_DOWN_STEPS - 5:  # 如果减速阶段结束
                self.phase = 0  # 重置阶段
                return True  # 返回 True 表示行为完成

        return False  # 如果行为未完成，返回 False


    def is_ready(self):
        """
        检查当前游戏/机器人条件下，该行为是否准备好开始/继续。

        返回：
        - bool 类型，如果行为准备好，则返回 True。
        """
        return True  # 假设行为总是准备好