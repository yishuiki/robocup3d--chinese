from agent.Base_Agent import Base_Agent as Agent
from scripts.commons.Script import Script
from scripts.commons.UI import UI

class Behaviors():
    def __init__(self, script: Script) -> None:
        '''
        初始化 Behaviors 类
        :param script: Script 对象，包含命令行参数等信息
        '''
        self.script = script
        self.player: Agent = None  # 初始化时未指定代理

    def ask_for_behavior(self):
        '''
        提示用户选择一个行为
        '''
        names, descriptions = self.player.behavior.get_all_behaviors()  # 获取所有行为的名称和描述

        # 打印行为列表，包含编号和描述
        UI.print_table([names, descriptions], ["Behavior Name", "Description"], numbering=[True, False])
        # 提示用户选择行为
        choice, is_str_opt = UI.read_particle('Choose behavior ("" to skip 2 time steps, "b" to beam, ctrl+c to return): ', ["", "b"], int, [0, len(names)])
        if is_str_opt:  # 如果用户选择的是字符串选项
            return choice  # 返回选择（跳过2个时间步或退出）
        return names[choice]  # 返回行为名称

    def sync(self):
        '''
        同步代理状态
        '''
        self.player.scom.commit_and_send(self.player.world.robot.get_command())  # 提交并发送命令
        self.player.scom.receive()  # 接收服务器响应

    def beam(self):
        '''
        将代理传送到指定位置
        '''
        # 使用非官方命令将代理传送到指定位置
        self.player.scom.unofficial_beam((-2.5, 0, self.player.world.robot.beam_height), 0)
        for _ in range(5):  # 多次同步以确保传送完成
            self.sync()

    def execute(self):
        '''
        执行行为
        '''
        a = self.script.args  # 获取命令行参数
        # 初始化代理
        self.player = Agent(a.i, a.p, a.m, a.u, a.r, a.t)  # 参数：服务器 IP、代理端口、监控端口、球衣号码、机器人类型、队伍名称
        behavior = self.player.behavior  # 获取行为模块

        # 将代理传送到初始位置
        self.beam()
        # 设置比赛模式为 "PlayOn"
        self.player.scom.unofficial_set_play_mode("PlayOn")

        # 特殊行为及其参数
        special_behaviors = {
            "Step": (),
            "Basic_Kick": (0,),
            "Walk": ((0.5, 0), False, 0, False, None),
            "Dribble": (None, None)
        }

        while True:
            behavior_name = self.ask_for_behavior()  # 提示用户选择行为
            if behavior_name == 0:  # 跳过2个时间步（用户请求）
                self.sync()
                self.sync()
            elif behavior_name == 1:  # 传送
                self.beam()
            else:
                if behavior_name in special_behaviors:  # 如果是特殊行为
                    # 提示用户输入行为持续时间
                    duration = UI.read_int("For how many time steps [1,1000]? ", 1, 1001)
                    for _ in range(duration):  # 执行行为
                        if behavior.execute(behavior_name, *special_behaviors[behavior_name]):
                            break  # 如果行为结束，则退出
                        self.sync()
                else:
                    # 执行行为直到完成
                    behavior.execute_to_completion(behavior_name)
