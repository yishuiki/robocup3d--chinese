from agent.Base_Agent import Base_Agent
from behaviors.custom.Step.Step_Generator import Step_Generator
from math_ops.Math_Ops import Math_Ops as M


class Basic_Kick():
    """
    Basic_Kick 类实现了一个基本的踢球行为，包括走到球的位置并执行踢球动作。
    """

    def __init__(self, base_agent: Base_Agent) -> None:
        """
        初始化 Basic_Kick 行为。

        参数：
        - base_agent: Base_Agent 类型，基础智能体对象，用于访问机器人和世界状态。
        """
        self.behavior = base_agent.behavior  # 获取行为管理器
        self.path_manager = base_agent.path_manager  # 获取路径管理器
        self.world = base_agent.world  # 获取世界状态
        self.description = "Walk to ball and perform a basic kick"  # 行为描述
        self.auto_head = True  # 是否自动控制头部

        r_type = self.world.robot.type  # 获取机器人类型
        self.bias_dir = [22, 29, 26, 29, 22][self.world.robot.type]  # 方向偏差
        self.ball_x_limits = ((0.19, 0.215), (0.2, 0.22), (0.19, 0.22), (0.2, 0.215), (0.2, 0.215))[r_type]  # 球的 X 轴范围
        self.ball_y_limits = ((-0.115, -0.1), (-0.125, -0.095), (-0.12, -0.1), (-0.13, -0.105), (-0.09, -0.06))[r_type]  # 球的 Y 轴范围
        self.ball_x_center = (self.ball_x_limits[0] + self.ball_x_limits[1]) / 2  # 球的 X 轴中心
        self.ball_y_center = (self.ball_y_limits[0] + self.ball_y_limits[1]) / 2  # 球的 Y 轴中心


    def execute(self, reset, direction, abort=False) -> bool:
        """
        执行踢球行为。

        参数：
        - reset: bool 类型，是否重置行为。
        - direction: float 类型，踢球方向，相对于场地的角度（单位：度）。
        - abort: bool 类型，是否终止行为。如果在对齐阶段请求终止，行为将立即返回 True；如果在踢球阶段请求终止，行为将在踢球完成后返回 True。
        """
        w = self.world  # 获取世界状态
        r = self.world.robot  # 获取机器人对象
        b = w.ball_rel_torso_cart_pos  # 获取球相对于躯干的位置
        t = w.time_local_ms  # 获取当前时间
        gait: Step_Generator = self.behavior.get_custom_behavior_object("Walk").env.step_generator  # 获取步态生成器

        if reset:
            self.phase = 0  # 重置行为阶段
            self.reset_time = t  # 更新重置时间

        if self.phase == 0:
            biased_dir = M.normalize_deg(direction + self.bias_dir)  # 添加偏差以校正方向
            ang_diff = abs(M.normalize_deg(biased_dir - r.loc_torso_orientation))  # 计算方向偏差

            next_pos, next_ori, dist_to_final_target = self.path_manager.get_path_to_ball(
                x_ori=biased_dir, x_dev=-self.ball_x_center, y_dev=-self.ball_y_center, torso_ori=biased_dir)

            if (w.ball_last_seen > t - w.VISUALSTEP_MS and ang_diff < 5 and  # 球可见且方向对齐
                self.ball_x_limits[0] < b[0] < self.ball_x_limits[1] and  # 球在踢球区域（X 轴）
                self.ball_y_limits[0] < b[1] < self.ball_y_limits[1] and  # 球在踢球区域（Y 轴）
                t - w.ball_abs_pos_last_update < 100 and  # 球的绝对位置信息是最近的
                dist_to_final_target < 0.03 and  # 球到最终目标的距离小于 0.03 米
                not gait.state_is_left_active and gait.state_current_ts == 2 and  # 步态阶段合适
                t - self.reset_time > 500):  # 避免立即踢球，确保准备和稳定

                self.phase += 1  # 进入下一阶段
                return self.behavior.execute_sub_behavior("Kick_Motion", True)  # 执行踢球动作
            else:
                dist = max(0.07, dist_to_final_target)  # 计算目标距离
                reset_walk = reset and self.behavior.previous_behavior != "Walk"  # 如果上一个行为不是 Walk，则重置 Walk
                self.behavior.execute_sub_behavior("Walk", reset_walk, next_pos, True, next_ori, True, dist)  # 执行 Walk 行为
                return abort  # 如果在阶段 0，返回 abort 状态

        else:  # 定义踢球参数并执行
            return self.behavior.execute_sub_behavior("Kick_Motion", False)


    def is_ready(self) -> bool:
        """
        检查当前游戏/机器人条件下，该行为是否准备好开始/继续。

        返回：
        - bool 类型，如果行为准备好，则返回 True。
        """
        return True