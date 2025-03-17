from agent.Base_Agent import Base_Agent
from collections import deque
import numpy as np

class Get_Up():
    """
    Get_Up 类实现了一个让机器人站起来的行为，根据机器人的状态选择最合适的技能。
    """

    def __init__(self, base_agent: Base_Agent) -> None:
        """
        初始化 Get_Up 行为。

        参数：
        - base_agent: Base_Agent 类型，基础智能体对象，用于访问机器人和世界状态。
        """
        self.behavior = base_agent.behavior  # 获取行为管理器
        self.world = base_agent.world  # 获取世界状态
        self.description = "Get Up using the most appropriate skills"  # 行为描述
        self.auto_head = False  # 是否自动控制头部
        self.MIN_HEIGHT = 0.3  # 头部的最小高度
        self.MAX_INCLIN = 50  # 躯干的最大倾斜角度（单位：度）
        self.STABILITY_THRESHOLD = 4  # 稳定性阈值


    def reset(self):
        """
        重置 Get_Up 行为。
        """
        self.state = 0  # 重置行为状态
        self.gyro_queue = deque(maxlen=self.STABILITY_THRESHOLD)  # 初始化陀螺仪数据队列
        self.watchdog = 0  # 当机器人出现抖动错误时，它永远不会稳定到足以站起来


    def execute(self, reset):
        """
        执行站起来的行为。

        参数：
        - reset: bool 类型，是否重置行为。
        """
        r = self.world.robot  # 获取机器人对象
        execute_sub_behavior = self.behavior.execute_sub_behavior  # 获取子行为执行函数

        if reset:
            self.reset()  # 如果需要重置，则调用 reset 方法

        if self.state == 0:  # 状态 0：进入 "Zero" 姿态
            self.watchdog += 1  # 增加看门狗计数
            self.gyro_queue.append(max(abs(r.gyro)))  # 记录最近的 STABILITY_THRESHOLD 个陀螺仪值

            # 如果行为完成且机器人稳定，则进入下一个状态
            if (execute_sub_behavior("Zero", None) and len(self.gyro_queue) == self.STABILITY_THRESHOLD 
                and all(g < 10 for g in self.gyro_queue)) or self.watchdog > 100:

                # 根据机器人的状态决定如何站起来
                if r.acc[0] < -4 and abs(r.acc[1]) < 2 and abs(r.acc[2]) < 3:
                    execute_sub_behavior("Get_Up_Front", True)  # 重置行为
                    self.state = 1
                elif r.acc[0] > 4 and abs(r.acc[1]) < 2 and abs(r.acc[2]) < 3:
                    execute_sub_behavior("Get_Up_Back", True)  # 重置行为
                    self.state = 2
                elif r.acc[2] > 8:  # 如果视觉信息未更新：如果姿态为 'Zero' 且躯干直立，则机器人已经站起来了
                    return True
                else:
                    execute_sub_behavior("Flip", True)  # 重置行为
                    self.state = 3

        elif self.state == 1:
            if execute_sub_behavior("Get_Up_Front", False):
                return True
        elif self.state == 2:
            if execute_sub_behavior("Get_Up_Back", False):
                return True
        elif self.state == 3:
            if execute_sub_behavior("Flip", False):
                self.reset()

        return False


    def is_ready(self):
        """
        检查当前游戏/机器人条件下，该行为是否准备好开始/继续。

        返回：
        - bool 类型，如果行为准备好，则返回 True（即机器人已经摔倒）。
        """
        r = self.world.robot  # 获取机器人对象
        # 检查是否 z < 5 且加速度计的模长 > 8 且任何视觉指示器表明机器人摔倒
        return r.acc[2] < 5 and np.dot(r.acc, r.acc) > 64 and (r.loc_head_z < self.MIN_HEIGHT or r.imu_torso_inclination > self.MAX_INCLIN)