from agent.Agent import Agent
from agent.Base_Agent import Base_Agent
from scripts.commons.Script import Script
import numpy as np

'''
目标：
----------
带球并射门
'''

class Dribble():
    def __init__(self, script: Script) -> None:
        '''
        初始化 Dribble 类
        :param script: Script 对象，包含命令行参数等信息
        '''
        self.script = script

    def execute(self):
        '''
        执行带球和射门行为
        '''
        a = self.script.args  

        # 创建一个带球者（使用 Base_Agent）和一个对手（使用 Agent）
        # 参数：服务器 IP、代理端口、监控端口、球衣号码、机器人类型（对于 Base_Agent）、队伍名称、启用日志、启用绘图
        self.script.batch_create(Base_Agent, ((a.i, a.p, a.m, a.u, a.r, a.t, True, True),))  # 一个带球者
        self.script.batch_create(Agent, ((a.i, a.p, a.m, u, "Opponent", False, False) for u in range(1, 2)))  # 一个对手（普通代理）

        p: Base_Agent = self.script.players[0]  # 获取带球者
        p.path_manager.draw_options(enable_obstacles=True, enable_path=True)  # 启用障碍物和路径绘制

        behavior = p.behavior  # 获取行为模块
        w = p.world  # 获取世界状态
        r = w.robot  # 获取机器人状态
        d = w.draw  # 获取绘图工具

        # 将带球者传送到初始位置
        p.scom.unofficial_beam((-3, 0, r.beam_height), 0)
        p.scom.unofficial_set_play_mode("PlayOn")  # 设置比赛模式为 "PlayOn"
        print("\nPress ctrl+c to return.")

        while True:
            # 如果比赛模式是对方开球，则切换到 "PlayOn" 模式
            if w.play_mode == w.M_THEIR_KICKOFF:
                p.scom.unofficial_set_play_mode("PlayOn")
            
            # 执行带球者的行为
            if behavior.is_ready("Get_Up") or w.play_mode_group in [w.MG_ACTIVE_BEAM, w.MG_PASSIVE_BEAM]:
                # 如果需要起身或处于传送模式，则将带球者传送到球的位置
                p.scom.unofficial_beam((*(w.ball_abs_pos[:2] - (1, 0)), r.beam_height), 0)
                behavior.execute("Zero_Bent_Knees")  # 执行起身行为
            else:
                behavior.execute("Dribble", None, None)  # 执行带球行为
            # 绘制速度注释
            d.annotation(r.loc_head_position + (0, 0, 0.2), f"{np.linalg.norm(r.get_head_abs_vel(40)[:2]):.2f}", d.Color.white, "vel_annotation")
            p.scom.commit_and_send(r.get_command())  # 提交并发送命令

            # 执行对手的行为（作为普通代理）
            self.script.batch_execute_agent(slice(1, None)) 
                        
            # 所有球员等待服务器反馈
            self.script.batch_receive()
