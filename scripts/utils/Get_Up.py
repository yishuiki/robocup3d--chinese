from agent.Base_Agent import Base_Agent as Agent
from itertools import count
from scripts.commons.Script import Script
import numpy as np

'''
目标：
----------
摔倒并起身
'''

class Get_Up():
    def __init__(self, script: Script) -> None:
        '''
        初始化 Get_Up 类
        :param script: Script 对象，包含命令行参数等信息
        '''
        self.script = script
        self.player: Agent = None  # 初始化时未指定代理

    def sync(self):
        '''
        同步代理状态
        '''
        r = self.player.world.robot
        self.player.scom.commit_and_send(r.get_command())  # 提交并发送命令
        self.player.scom.receive()  # 接收服务器响应

    def execute(self):
        '''
        执行摔倒并起身的行为
        '''
        a = self.script.args
        # 初始化代理
        player = self.player = Agent(a.i, a.p, a.m, a.u, a.r, a.t)  # 参数：服务器 IP、代理端口、监控端口、球衣号码、机器人类型、队伍名称
        behavior = player.behavior  # 获取行为模块
        r = player.world.robot  # 获取机器人状态

        # 将代理传送到初始位置
        player.scom.commit_beam((-3, 0), 0)
        print("\nPress ctrl+c to return.")

        for i in count():
            # 随机生成关节目标速度
            rnd = np.random.uniform(-6, 6, r.no_of_joints)

            # 摔倒
            while r.loc_head_z > 0.3 and r.imu_torso_inclination < 50:
                if i < 4:
                    # 前几次摔倒是确定性的
                    behavior.execute(["Fall_Front", "Fall_Back", "Fall_Left", "Fall_Right"][i % 4])
                else:
                    # 后续摔倒是随机的
                    r.joints_target_speed[:] = rnd
                self.sync()

            # 起身
            behavior.execute_to_completion("Get_Up")  # 执行起身行为
            behavior.execute_to_completion("Zero_Bent_Knees")  # 执行直膝行为
