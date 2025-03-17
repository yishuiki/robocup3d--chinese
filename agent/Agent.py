from agent.Base_Agent import Base_Agent  # 导入基础代理类
from math_ops.Math_Ops import Math_Ops as M  # 导入数学操作模块
import math  # 导入数学库
import numpy as np  # 导入 NumPy 库


class Agent(Base_Agent):
    """
    定义一个代理类，继承自 Base_Agent。
    该类封装了代理的行为逻辑，包括移动、踢球、决策等。
    """

    def __init__(self, host: str, agent_port: int, monitor_port: int, unum: int,
                 team_name: str, enable_log, enable_draw, wait_for_server=True, is_fat_proxy=False) -> None:
        """
        初始化代理。

        参数：
        - host: 服务器地址
        - agent_port: 代理端口
        - monitor_port: 监控端口
        - unum: 代理编号
        - team_name: 队伍名称
        - enable_log: 是否启用日志
        - enable_draw: 是否启用绘图
        - wait_for_server: 是否等待服务器响应
        - is_fat_proxy: 是否使用胖代理模式
        """

        # 定义机器人类型，根据代理编号选择
        robot_type = (0, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4)[unum - 1]

        # 初始化基础代理
        super().__init__(host, agent_port, monitor_port, unum, robot_type, team_name, enable_log, enable_draw, True,
                         wait_for_server, None)

        self.enable_draw = enable_draw  # 是否启用绘图
        self.state = 0  # 当前状态：0-正常，1-起身，2-踢球
        self.kick_direction = 0  # 踢球方向
        self.kick_distance = 0  # 踢球距离
        self.fat_proxy_cmd = "" if is_fat_proxy else None  # 胖代理命令
        self.fat_proxy_walk = np.zeros(3)  # 胖代理的过滤行走参数

        # 初始位置（根据代理编号选择）
        self.init_pos = ([-14, 0], [-9, -5], [-9, 0], [-9, 5], [-5, -5], [-5, 0], [-5, 5], [-1, -6], [-1, -2.5], [-1, 2.5], [-1, 6])[
            unum - 1]

    def beam(self, avoid_center_circle=False):
        """
        将代理传送到初始位置。

        参数：
        - avoid_center_circle: 是否避开中圈
        """
        r = self.world.robot  # 获取机器人状态
        pos = self.init_pos[:]  # 复制初始位置
        self.state = 0  # 重置状态为正常

        # 如果需要避开中圈，调整位置
        if avoid_center_circle and np.linalg.norm(self.init_pos) < 2.5:
            pos[0] = -2.3

        # 如果当前位置与目标位置有较大偏差，或者处于起身状态，则传送
        if np.linalg.norm(pos - r.loc_head_position[:2]) > 0.1 or self.behavior.is_ready("Get_Up"):
            self.scom.commit_beam(pos, M.vector_angle((-pos[0], -pos[1])))  # 传送至初始位置，面向坐标原点
        else:
            # 如果不是胖代理模式，则执行正常行为
            if self.fat_proxy_cmd is None:
                self.behavior.execute("Zero_Bent_Knees_Auto_Head")
            else:  # 胖代理模式
                self.fat_proxy_cmd += "(proxy dash 0 0 0)"  # 发送胖代理命令
                self.fat_proxy_walk = np.zeros(3)  # 重置胖代理行走参数

    def move(self, target_2d=(0, 0), orientation=None, is_orientation_absolute=True,
             avoid_obstacles=True, priority_unums=[], is_aggressive=False, timeout=3000):
        """
        移动到目标位置。

        参数：
        - target_2d: 目标位置（二维坐标）
        - orientation: 目标朝向（角度）
        - is_orientation_absolute: 是否为绝对朝向（相对于场地）
        - avoid_obstacles: 是否避开障碍物
        - priority_unums: 优先避开的队友编号
        - is_aggressive: 是否为进攻模式
        - timeout: 路径规划超时时间（微秒）
        """
        r = self.world.robot  # 获取机器人状态

        # 胖代理模式
        if self.fat_proxy_cmd is not None:
            self.fat_proxy_move(target_2d, orientation, is_orientation_absolute)  # 忽略障碍物
            return

        # 如果需要避开障碍物，则调用路径规划
        if avoid_obstacles:
            target_2d, _, distance_to_final_target = self.path_manager.get_path_to_target(
                target_2d, priority_unums=priority_unums, is_aggressive=is_aggressive, timeout=timeout)
        else:
            distance_to_final_target = np.linalg.norm(target_2d - r.loc_head_position[:2])  # 计算与目标的距离

        # 执行移动行为
        self.behavior.execute("Basic_Run", self.world.ball_abs_pos[:2], True, None, True, None)

    def kick(self, kick_direction=None, kick_distance=None, abort=False, enable_pass_command=False):
        """
        踢球。

        参数：
        - kick_direction: 踢球方向（角度）
        - kick_distance: 踢球距离（米）
        - abort: 是否中止踢球
        - enable_pass_command: 是否启用传球指令
        """
        # 如果对手距离球较近且启用传球指令，则发送传球指令
        if self.min_opponent_ball_dist < 1.45 and enable_pass_command:
            self.scom.commit_pass_command()

        # 更新踢球方向和距离
        self.kick_direction = self.kick_direction if kick_direction is None else kick_direction
        self.kick_distance = self.kick_distance if kick_distance is None else kick_distance

        # 根据是否为胖代理模式执行不同行为
        if self.fat_proxy_cmd is None:  # 正常模式
            return self.behavior.execute("Basic_Kick", self.kick_direction, abort)  # 执行基本踢球行为
        else:  # 胖代理模式
            return self.fat_proxy_kick()

    def think_and_send(self):
        """
        思考并发送命令。
        """
        w = self.world  # 获取世界状态
        r = self.world.robot  # 获取机器人状态
        my_head_pos_2d = r.loc_head_position[:2]  # 机器人头部位置
        my_ori = r.imu_torso_orientation  # 机器人朝向
        ball_2d = w.ball_abs_pos[:2]  # 球的位置
        ball_vec = ball_2d - my_head_pos_2d  # 球的向量
        ball_dir = M.vector_angle(ball_vec)  # 球的方向
        ball_dist = np.linalg.norm(ball_vec)  # 球的距离
        ball_sq_dist = ball_dist * ball_dist  # 球的平方距离（用于快速比较）
        ball_speed = np.linalg.norm(w.get_ball_abs_vel(6)[:2])  # 球的速度
        behavior = self.behavior  # 获取行为模块
        goal_dir = M.target_abs_angle(ball_2d, (15.05, 0))  # 计算射门方向
        path_draw_options = self.path_manager.draw_options  # 路径绘制选项
        PM = w.play_mode  # 当前比赛模式
        PM_GROUP = w.play_mode_group  # 比赛模式组

        # 预测球的未来位置（当球速小于 0.5 m/s 时）
        slow_ball_pos = w.get_predicted_ball_pos(0.5)

        # 计算队友与球的距离（平方）
        teammates_ball_sq_dist = [np.sum((p.state_abs_pos[:2] - slow_ball_pos) ** 2) if p.state_last_update != 0 and (
                w.time_local_ms - p.state_last_update <= 360 or p.is_self) and not p.state_fallen else 1000 for p in
                                  w.teammates]

        # 计算对手与球的距离（平方）
        opponents_ball_sq_dist = [np.sum((p.state_abs_pos[:2] - slow_ball_pos) ** 2) if p.state_last_update != 0 and (
                w.time_local_ms - p.state_last_update <= 360) and not p.state_fallen else 1000 for p in w.opponents]

        # 计算最近的队友和对手与球的距离
        min_teammate_ball_sq_dist = min(teammates_ball_sq_dist)
        self.min_teammate_ball_dist = math.sqrt(min_teammate_ball_sq_dist)  # 最近队友与球的距离
        self.min_opponent_ball_dist = math.sqrt(min(opponents_ball_sq_dist))  # 最近对手与球的距离

        # 找到最近的队友编号（加1是因为编号从1开始）
        active_player_unum = teammates_ball_sq_dist.index(min_teammate_ball_sq_dist) + 1

        #--------------------------------------- 2. 决策行为

        # 根据比赛模式和状态选择行为
        if PM == w.M_GAME_OVER:
            # 比赛结束，不做任何操作
            pass
        elif PM_GROUP == w.MG_ACTIVE_BEAM:
            # 主动传送模式
            self.beam()
        elif PM_GROUP == w.MG_PASSIVE_BEAM:
            # 被动传送模式（避开中圈）
            self.beam(True)
        elif self.state == 1 or (behavior.is_ready("Get_Up") and self.fat_proxy_cmd is None):
            # 如果处于起身状态或准备起身
            self.state = 0 if behavior.execute("Get_Up") else 1  # 完成起身行为后恢复到正常状态
        elif PM == w.M_OUR_KICKOFF:
            # 我方开球
            if r.unum == 9:
                # 如果是9号球员，执行踢球动作
                self.kick(120, 3)  # 方向为120度，距离为3米
            else:
                # 其他球员移动到初始位置
                self.move(self.init_pos, orientation=ball_dir)
        elif PM == w.M_THEIR_KICKOFF:
            # 对方开球
            self.move(self.init_pos, orientation=ball_dir)  # 移动到初始位置
        elif active_player_unum != r.unum:
            # 如果当前球员不是活跃球员
            if r.unum == 1:
                # 如果是守门员
                self.move(self.init_pos, orientation=ball_dir)  # 移动到初始位置
            else:
                # 计算基于球位置的基本站位
                new_x = max(0.5, (ball_2d[0] + 15) / 15) * (self.init_pos[0] + 15) - 15
                if self.min_teammate_ball_dist < self.min_opponent_ball_dist:
                    new_x = min(new_x + 3.5, 13)  # 如果我方控球，向前推进
                self.move((new_x, self.init_pos[1]), orientation=ball_dir, priority_unums=[active_player_unum])
        else:
            # 如果是活跃球员
            path_draw_options(enable_obstacles=True, enable_path=True, use_team_drawing_channel=True)
            enable_pass_command = (PM == w.M_PLAY_ON and ball_2d[0] < 6)

            if r.unum == 1 and PM_GROUP == w.MG_THEIR_KICK:
                # 对方开球时，守门员移动到初始位置
                self.move(self.init_pos, orientation=ball_dir)
            elif PM == w.M_OUR_CORNER_KICK:
                # 我方角球
                self.kick(-np.sign(ball_2d[1]) * 95, 5.5)  # 将球传向对方球门前方
            elif self.min_opponent_ball_dist + 0.5 < self.min_teammate_ball_dist:
                # 如果对手明显更接近球，进行防守
                if self.state == 2:
                    self.state = 0 if self.kick(abort=True) else 2  # 中止踢球动作
                else:
                    # 移动到球和我方球门之间的位置
                    self.move(slow_ball_pos + M.normalize_vec((-16, 0) - slow_ball_pos) * 0.2, is_aggressive=True)
            else:
                # 否则，尝试踢球
                self.state = 0 if self.kick(goal_dir, 9, False, enable_pass_command) else 2

            # 关闭路径绘制选项
            path_draw_options(enable_obstacles=False, enable_path=False)

        #--------------------------------------- 3. 广播信息
        self.radio.broadcast()

        #--------------------------------------- 4. 发送命令到服务器
        if self.fat_proxy_cmd is None:
            # 正常模式：发送机器人命令
            self.scom.commit_and_send(r.get_command())
        else:
            # 胖代理模式：发送胖代理命令
            self.scom.commit_and_send(self.fat_proxy_cmd.encode())
            self.fat_proxy_cmd = ""  # 清空命令

        #---------------------- 调试用的绘图注释
        if self.enable_draw:
            d = w.draw
            if active_player_unum == r.unum:
                # 如果是活跃球员，绘制相关信息
                d.point(slow_ball_pos, 3, d.Color.pink, "status", False)  # 预测的球位置
                d.point(w.ball_2d_pred_pos[-1], 5, d.Color.pink, "status", False)  # 最后一次球预测位置
                d.annotation((*my_head_pos_2d, 0.6), "I've got it!", d.Color.yellow, "status")
            else:
                d.clear("status")  # 清空绘图

    #--------------------------------------- 胖代理辅助方法

    def fat_proxy_kick(self):
        """
        胖代理模式下的踢球方法。
        """
        w = self.world
        r = self.world.robot
        ball_2d = w.ball_abs_pos[:2]  # 球的位置
        my_head_pos_2d = r.loc_head_position[:2]  # 机器人头部位置

        # 如果机器人接近球
        if np.linalg.norm(ball_2d - my_head_pos_2d) < 0.25:
            # 发送踢球命令
            self.fat_proxy_cmd += f"(proxy kick 10 {M.normalize_deg(self.kick_direction - r.imu_torso_orientation):.2f} 20)"
            self.fat_proxy_walk = np.zeros(3)  # 重置行走参数
            return True
        else:
            # 否则，移动到球的位置
            self.fat_proxy_move(ball_2d - (-0.1, 0), None, True)
            return False

    def fat_proxy_move(self, target_2d, orientation, is_orientation_absolute):
        """
        胖代理模式下的移动方法。
        """
        r = self.world.robot

        # 计算目标距离和方向
        target_dist = np.linalg.norm(target_2d - r.loc_head_position[:2])
        target_dir = M.target_rel_angle(r.loc_head_position[:2], r.imu_torso_orientation, target_2d)

        # 如果距离较远且方向偏差较小，向前移动
        if target_dist > 0.1 and abs(target_dir) < 8:
            self.fat_proxy_cmd += f"(proxy dash {100} {0} {0})"
            return

        # 如果接近目标位置
        if target_dist < 0.1:
            if is_orientation_absolute:
                # 如果是绝对方向，计算相对方向
                orientation = M.normalize_deg(orientation - r.imu_torso_orientation)
            target_dir = np.clip(orientation, -60, 60)
            self.fat_proxy_cmd += f"(proxy dash {0} {0} {target_dir:.1f})"
        else:
            # 否则，调整方向并移动
            self.fat_proxy_cmd += f"(proxy dash {20} {0} {target_dir:.1f})"