from time import sleep
from world.commons.Draw import Draw


class Drawings():
    def __init__(self, script) -> None:
        '''
        初始化 Drawings 类
        :param script: Script 对象，包含命令行参数等信息
        '''
        self.script = script

    def execute(self):
        '''
        执行绘图操作
        '''
        # 创建一个 Draw 对象，通常我们可以通过 player.world.draw 访问该对象
        # 这里是一个快捷方式，用于在不创建代理的情况下绘制图形
        a = self.script.args
        draw = Draw(True, 0, a.i, 32769)  # 参数：启用绘图、代理编号、服务器 IP、端口

        print("\nPress ctrl+c to return.")

        while True:
            for i in range(100):
                sleep(0.02)  # 每次绘制之间暂停 0.02 秒

                # 绘制一个绿色的圆
                draw.circle((0, 0), i / 10, 2, Draw.Color.green_light, "green")
                # 绘制一个红色的圆
                draw.circle((0, 0), i / 9, 2, Draw.Color.red, "red")
                # 绘制一个红色的球
                draw.sphere((0, 0, 5 - i / 20), 0.2, Draw.Color.red, "ball")
                # 绘制一个注释文本
                draw.annotation((0, 0, 1), "Hello!", Draw.Color.cyan, "text")
                # 绘制一个箭头
                draw.arrow((0, 0, 5), (0, 0, 5 - i / 25), 0.5, 4, Draw.Color.get(127, 50, 255), "my_arrow")

                # 绘制一个金字塔
                draw.polygon(((2, 0, 0), (3, 0, 0), (3, 1, 0), (2, 1, 0)), Draw.Color.blue, 255, "solid", False)
                draw.line((2, 0, 0), (2.5, 0.5, 1), 2, Draw.Color.cyan, "solid", False)
                draw.line((3, 0, 0), (2.5, 0.5, 1), 2, Draw.Color.cyan, "solid", False)
                draw.line((2, 1, 0), (2.5, 0.5, 1), 2, Draw.Color.cyan, "solid", False)
                draw.line((3, 1, 0), (2.5, 0.5, 1), 2, Draw.Color.cyan, "solid", True)
