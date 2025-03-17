from agent.Base_Agent import Base_Agent
from behaviors.custom.Run.Env import Env  # 确保导入的是跑步专用的 Env 类
from math_ops.Math_Ops import Math_Ops as M
from math_ops.Neural_Network import run_mlp
import numpy as np
import pickle

class Basic_Run():
    """
    基本跑步类，使用环境类 `Env` 和预训练的强化学习模型来控制机器人跑步。
    """
    def __init__(self, base_agent: Base_Agent):
        """
        初始化基本跑步行为。
        :param base_agent: 基础智能体对象。
        """
        self.world = base_agent.world  # 获取世界状态
        self.env = Env(base_agent)  # 初始化跑步环境
        self.description = "Basic Run Behavior with RL Model"
        self.auto_head = True  # 是否自动控制头部

        # 根据机器人类型加载预训练的强化学习模型
        model_paths = [
            "/behaviors/custom/Run/run_R0.pkl",          # 机器人类型 0
            "/behaviors/custom/Run/run_R1_R3.pkl",       # 机器人类型 1
            "/behaviors/custom/Run/run_R2.pkl",          # 机器人类型 2
            "/behaviors/custom/Run/run_R1_R3.pkl",       # 机器人类型 3
            "/behaviors/custom/Run/run_R4.pkl"           # 机器人类型 4
        ]
        model_path = M.get_active_directory(model_paths[self.world.robot.type])

        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)  # 加载模型

    def execute(self, reset=False, target_2d=None, orientation=None):
        """
        执行跑步行为。
        :param reset: 是否重置行为。
        :param target_2d: 二维目标位置（相对或绝对）。
        :param orientation: 目标方向（相对或绝对）。
        """
        # 更新环境的目标和方向
        if target_2d is not None:
            self.env.internal_target = np.array(target_2d, dtype=np.float32)
        if orientation is not None:
            self.env.internal_rel_orientation = orientation

        # 获取环境观测值
        obs = self.env.observe(reset)

        # 使用强化学习模型生成动作
        action = run_mlp(obs, self.model)  # 使用预训练模型生成动作

        # 执行动作
        self.env.execute(action)

        return False  # 返回 False 表示行为未完成

    def is_ready(self):
        """
        检查跑步行为是否准备好在当前游戏/机器人条件下开始。

        返回：
        - bool 类型，如果行为准备好，则返回 True。
        """
        # 检查机器人是否处于可跑步的状态（例如，是否站立且无错误）
        r = self.world.robot
        if r.is_standing and not r.has_error:
            return True
        else:
            return False