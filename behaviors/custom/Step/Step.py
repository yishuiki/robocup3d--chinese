from agent.Base_Agent import Base_Agent
from behaviors.custom.Step.Step_Generator import Step_Generator
import numpy as np

class Step():
    """
    Step 类实现了一个步行动作原语（Skill-Set-Primitive），用于控制机器人的行走。
    """

    def __init__(self, base_agent: Base_Agent) -> None:
        """
        初始化 Step 行为。

        参数：
        - base_agent: Base_Agent 类型，基础智能体对象，用于访问机器人和世界状态。
        """
        self.world = base_agent.world  # 获取世界状态
        self.ik = base_agent.inv_kinematics  # 获取逆运动学模块
        self.description = "Step (Skill-Set-Primitive)"  # 行为描述
        self.auto_head = True  # 是否自动控制头部

        nao_specs = self.ik.NAO_SPECS  # 获取 NAO 机器人的规格参数
        self.leg_length = nao_specs[1] + nao_specs[3]  # 大腿和小腿的总长度

        feet_y_dev = nao_specs[0] * 1.2  # 步幅宽度（比默认值稍宽）
        sample_time = self.world.robot.STEPTIME  # 采样时间
        max_ankle_z = nao_specs[5]  # 踝关节的最大 Z 轴高度

        # 使用常量初始化步态生成器
        self.step_generator = Step_Generator(feet_y_dev, sample_time, max_ankle_z)


    def execute(self, reset, ts_per_step=7, z_span=0.03, z_max=0.8):
        """
        执行步行动作。

        参数：
        - reset: bool 类型，是否重置行为。
        - ts_per_step: int 类型，每步的时间步数。
        - z_span: float 类型，Z 轴移动范围。
        - z_max: float 类型，Z 轴最大值。
        """
        # 获取步态生成器的目标位置
        lfy, lfz, rfy, rfz = self.step_generator.get_target_positions(reset, ts_per_step, z_span, self.leg_length * z_max)

        #----------------- 对每条腿应用逆运动学并设置关节目标位置

        # 左腿
        indices, self.values_l, error_codes = self.ik.leg((0, lfy, lfz), (0, 0, 0), True, dynamic_pose=False)
        for i in error_codes:
            print(f"Joint {i} is out of range!" if i != -1 else "Position is out of reach!")

        self.world.robot.set_joints_target_position_direct(indices, self.values_l)

        # 右腿
        indices, self.values_r, error_codes = self.ik.leg((0, rfy, rfz), (0, 0, 0), False, dynamic_pose=False)
        for i in error_codes:
            print(f"Joint {i} is out of range!" if i != -1 else "Position is out of reach!")

        self.world.robot.set_joints_target_position_direct(indices, self.values_r)

        # ----------------- 固定手臂位置

        indices = [14, 16, 18, 20]
        values = np.array([-80, 20, 90, 0])
        self.world.robot.set_joints_target_position_direct(indices, values)

        indices = [15, 17, 19, 21]
        values = np.array([-80, 20, 90, 0])
        self.world.robot.set_joints_target_position_direct(indices, values)

        return False


    def is_ready(self):
        """
        检查当前游戏/机器人条件下，该行为是否准备好开始/继续。

        返回：
        - bool 类型，如果行为准备好，则返回 True。
        """
        return True  # 假设行为总是准备好