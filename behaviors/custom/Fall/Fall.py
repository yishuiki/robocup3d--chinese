from agent.Base_Agent import Base_Agent
from math_ops.Math_Ops import Math_Ops as M
from math_ops.Neural_Network import run_mlp
import pickle, numpy as np

class Fall():
    """
    Fall 类实现了一个示例性的摔倒行为，用于控制机器人在摔倒时的动作。
    """

    def __init__(self, base_agent: Base_Agent) -> None:
        """
        初始化 Fall 行为。

        参数：
        - base_agent: Base_Agent 类型，基础智能体对象，用于访问机器人和世界状态。
        """
        self.world = base_agent.world  # 获取世界状态
        self.description = "Fall example"  # 行为描述
        self.auto_head = False  # 是否自动控制头部

        # 加载预训练的神经网络模型
        with open(M.get_active_directory("/behaviors/custom/Fall/fall.pkl"), 'rb') as f:
            self.model = pickle.load(f)

        # 从神经网络的最后一层偏置中提取动作大小
        self.action_size = len(self.model[-1][0])
        self.obs = np.zeros(self.action_size + 1, np.float32)  # 初始化观测值数组

        # 确保不同机器人类型之间的兼容性
        self.controllable_joints = min(self.world.robot.no_of_joints, self.action_size)


    def observe(self):
        """
        获取环境的观测值。
        """
        r = self.world.robot  # 获取机器人对象

        # 将关节位置归一化并存储到观测值数组中
        for i in range(self.action_size):
            self.obs[i] = r.joints_position[i] / 100  # 简单的比例归一化

        # 添加头部的 Z 轴位置作为观测值
        self.obs[self.action_size] = r.cheat_abs_pos[2]  # 头部的 Z 轴位置（替代：r.loc_head_z）


    def execute(self, reset) -> bool:
        """
        执行摔倒行为。

        参数：
        - reset: bool 类型，是否重置行为。

        返回：
        - bool 类型，如果行为完成（头部高度 < 0.15 米），则返回 True。
        """
        self.observe()  # 获取当前观测值
        action = run_mlp(self.obs, self.model)  # 使用神经网络生成动作

        # 设置关节目标位置
        self.world.robot.set_joints_target_position_direct(
            slice(self.controllable_joints),  # 作用于训练过的关节
            action * 10,  # 放大动作以促进早期探索
            harmonize=False  # 如果目标在每一步都改变，则无需协调动作
        )

        # 如果头部高度小于 0.15 米，则认为行为完成
        return self.world.robot.loc_head_z < 0.15


    def is_ready(self) -> bool:
        """
        检查当前游戏/机器人条件下，该行为是否准备好开始/继续。

        返回：
        - bool 类型，如果行为准备好，则返回 True。
        """
        return True  # 假设行为总是准备好