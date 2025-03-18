from agent.Base_Agent import Base_Agent as Agent
from scripts.commons.Script import Script
from time import sleep


class Beam():
    def __init__(self, script: Script) -> None:
        '''
        初始化 Beam 类
        :param script: Script 对象，包含命令行参数等信息
        '''
        self.script = script

    def ask_for_input(self, prompt, default):
        '''
        提示用户输入一个值，如果输入无效则返回默认值

        :param prompt: 提示信息
        :param default: 默认值
        :return: 用户输入的值或默认值
        '''
        try:
            inp = input(prompt)
            return float(inp)
        except ValueError:
            if inp != '':
                print("Illegal input:", inp, "\n")
            return default

    def beam_and_update(self, x, y, rot):
        '''
        将代理传送到指定位置并更新状态

        :param x: x 坐标
        :param y: y 坐标
        :param rot: 旋转角度
        '''
        r = self.player.world.robot
        d = self.player.world.draw

        # 在世界中绘制位置标签
        d.annotation((x, y, 0.7), f"x:{x} y:{y} r:{rot}", d.Color.yellow, "pos_label")

        # 使用非官方命令将代理传送到指定位置
        self.player.scom.unofficial_beam((x, y, r.beam_height), rot)
        # 多次运行以处理可能的碰撞（例如球门柱）
        for _ in range(10):
            sleep(0.03)
            self.player.behavior.execute("Zero")
            self.player.scom.commit_and_send(r.get_command())
            self.player.scom.receive()

    def execute(self):
        '''
        执行传送操作
        '''
        a = self.script.args
        # 初始化代理
        self.player = Agent(a.i, a.p, a.m, a.u, a.r, a.t)  # 参数：服务器 IP、代理端口、监控端口、球衣号码、机器人类型、队伍名称
        d = self.player.world.draw

        # 设置比赛模式为 "PlayOn"
        self.player.scom.unofficial_set_play_mode("PlayOn")

        # 绘制网格
        for x in range(-15, 16):
            for y in range(-10, 11):
                d.point((x, y), 6, d.Color.red, "grid", False)
        d.flush("grid")

        # 初始化代理状态
        for _ in range(10):
            self.player.scom.send()
            self.player.scom.receive()

        print("\nBeam player to coordinates + orientation:")

        x = y = a = 0
        while True:  # 将代理传送到指定位置
            x = self.ask_for_input(f"\nInput x coordinate       ('' to send {x:5} again, ctrl+c to return): ", x)
            self.beam_and_update(x, y, a)
            y = self.ask_for_input(f"Input y coordinate       ('' to send {y:5} again, ctrl+c to return): ", y)
            self.beam_and_update(x, y, a)
            a = self.ask_for_input(f"Orientation -180 to 180  ('' to send {a:5} again, ctrl+c to return): ", a)
            self.beam_and_update(x, y, a)
