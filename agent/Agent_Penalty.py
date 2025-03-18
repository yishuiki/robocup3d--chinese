from agent.Base_Agent import Base_Agent
from math_ops.Math_Ops import Math_Ops as M
import numpy as np
import random


class Agent(Base_Agent):
    def __init__(self, host: str, agent_port: int, monitor_port: int, unum: int,
                 team_name: str, enable_log, enable_draw, wait_for_server=True, is_fat_proxy=False) -> None:
        """
        初始化 Agent 类
        :param host: 服务器 IP
        :param agent_port: Agent 端口
        :param monitor_port: 监控端口
        :param unum: 球衣号码
        :param team_name: 队伍名称
        :param enable_log: 是否启用日志
        :param enable_draw: 是否启用绘图
        :param wait_for_server: 是否等待服务器
        :param is_fat_proxy: 是否为 magmaFatProxy
        """
        
        # 定义机器人类型
        robot_type = 0 if unum == 1 else 4  # 假设守门员使用球衣号码 1，其他号码为进攻球员

        # 初始化基础 Agent
        # 参数：服务器 IP、Agent 端口、监控端口、球衣号码、机器人类型、队伍名称、启用日志、启用绘图、是否校正比赛模式、是否等待服务器、听觉回调
        super().__init__(host, agent_port, monitor_port, unum, robot_type, team_name, enable_log, enable_draw, False, wait_for_server, None)

        self.enable_draw = enable_draw  # 是否启用绘图
        self.state = 0  # 状态：0-正常，1-起身，2-向左扑，3-向右扑，4-等待

        self.kick_dir = 0  # 踢球方向
        self.reset_kick = True  # 当为 True 时，生成新的随机踢球方向

    def think_and_send(self):
        """
        思考并发送指令
        """
        w = self.world  # 获取世界状态
        r = self.world.robot  # 获取机器人状态
        my_head_pos_2d = r.loc_head_position[:2]  # 我方头部位置（二维）
        my_ori = r.imu_torso_orientation  # 我方身体朝向
        ball_2d = w.ball_abs_pos[:2]  # 球的位置（二维）
        ball_vec = ball_2d - my_head_pos_2d  # 球与我方头部的向量
        ball_dir = M.vector_angle(ball_vec)  # 球的方向
        ball_dist = np.linalg.norm(ball_vec)  # 球与我方头部的距离
        ball_speed = np.linalg.norm(w.get_ball_abs_vel(6)[:2])  # 球的速度
        behavior = self.behavior  # 获取行为模块
        PM = w.play_mode  # 获取比赛模式

        # --------------------------------------- 1. 决定动作

        if PM in [w.M_BEFORE_KICKOFF, w.M_THEIR_GOAL, w.M_OUR_GOAL]:  # 比赛开始前、对方进球、我方进球
            # 传送至初始位置并等待
            self.state = 0
            self.reset_kick = True
            pos = (-14, 0) if r.unum == 1 else (4.9, 0)  # 守门员和进攻球员的初始位置
            if np.linalg.norm(pos - r.loc_head_position[:2]) > 0.1 or behavior.is_ready("Get_Up"):
                # 如果距离初始位置大于 0.1 或需要起身
                self.scom.commit_beam(pos, 0)  # 传送至初始位置
            else:
                behavior.execute("Zero_Bent_Knees")  # 等待
        elif self.state == 2:  # 向左扑
            self.state = 4 if behavior.execute("Dive_Left") else 2  # 动作完成后进入等待状态
        elif self.state == 3:  # 向右扑
            self.state = 4 if behavior.execute("Dive_Right") else 3  # 动作完成后进入等待状态
        elif self.state == 4:  # 等待（扑球后或对方踢球时）
            pass
        elif self.state == 1 or behavior.is_ready("Get_Up"):  # 如果正在起身或已摔倒
            self.state = 0 if behavior.execute("Get_Up") else 1  # 起身动作完成后进入正常状态
        elif PM == w.M_OUR_KICKOFF and r.unum == 1 or PM == w.M_THEIR_KICKOFF and r.unum != 1:
            # 我方开球且为守门员，或对方开球且为进攻球员
            self.state = 4  # 等待下次传送
        elif r.unum == 1:  # 守门员
            y_coordinate = np.clip(ball_2d[1], -1.1, 1.1)  # 限制 y 坐标范围
            behavior.execute("Walk", (-14, y_coordinate), True, 0, True, None)  # 走向球的位置
            if ball_2d[0] < -10:  # 如果球接近球门
                self.state = 2 if ball_2d[1] > 0 else 3  # 根据球的位置决定扑向左边还是右边
        else:  # 进攻球员
            if PM == w.M_OUR_KICKOFF and ball_2d[0] > 5:  # 我方开球且球在进攻球员前方
                if self.reset_kick: 
                    self.kick_dir = random.choice([-7.5, 7.5])  # 随机选择踢球方向
                    self.reset_kick = False
                behavior.execute("Basic_Kick", self.kick_dir)  # 执行基本踢球动作
            else:
                behavior.execute("Zero_Bent_Knees")  # 等待

        # --------------------------------------- 2. 广播
        self.radio.broadcast()

        # --------------------------------------- 3. 发送至服务器
        self.scom.commit_and_send(r.get_command())

        # ---------------------- 调试用注释
        if self.enable_draw: 
            d = w.draw
            if r.unum == 1:
                d.annotation((*my_head_pos_2d, 0.8), "Goalkeeper", d.Color.yellow, "status")  # 标注守门员状态
            else:
                d.annotation((*my_head_pos_2d, 0.8), "Kicker", d.Color.yellow, "status")  # 标注进攻球员状态
                if PM == w.M_OUR_KICKOFF:  # 绘制踢球方向箭头
                    d.arrow(ball_2d, ball_2d + 5 * M.vector_from_angle(self.kick_dir), 0.4, 3, d.Color.cyan_light, "Target")
