from agent.Agent import Agent as Agent  # 导入代理类
from cpp.localization import localization  # 导入定位模块
from math_ops.Math_Ops import Math_Ops as M  # 导入数学操作类
from scripts.commons.Script import Script  # 导入脚本类
from world.commons.Draw import Draw  # 导入绘图工具类
from world.commons.Other_Robot import Other_Robot  # 导入其他机器人类


class Localization():
    """
    定位类，用于展示机器人的定位和感知功能。
    """

    def __init__(self, script: Script) -> None:
        """
        初始化定位类。
        :param script: 脚本对象，用于获取相关参数。
        """
        self.script = script  # 保存脚本对象

    def execute(self):
        """
        执行定位功能。
        """
        a = self.script.args  # 获取脚本参数
        # 创建一个独立的绘图对象，用于绘制机器人的感知信息
        d = self.draw = Draw(True, 0, a.i, 32769)  # 参数：启用绘图、绘图编号、服务器 IP、绘图端口

        # 创建机器人代理
        # 参数：服务器 IP、代理端口、监控端口、球衣号码、队伍名称、是否启用日志、是否启用绘图
        self.script.batch_create(Agent, ((a.i, a.p, a.m, 1, a.t, False, False),))  # 创建一个队友（虚拟守门员，不进行通信）
        self.script.batch_create(Agent, ((a.i, a.p, a.m, 5, "Opponent", False, False),))  # 创建一个对手
        self.script.batch_create(Agent, ((a.i, a.p, a.m, 9, a.t, False, False),))  # 创建一个主代理（绘制其世界信息）

        # 将虚拟守门员移动到指定位置
        self.script.batch_unofficial_beam(((-14, 0, 0.5, 0),), slice(0, 1))

        p: Agent = self.script.players[-1]  # p 为主代理
        p.scom.unofficial_set_play_mode("PlayOn")  # 设置比赛模式为“进行中”

        # 执行主循环
        while True:
            # 提交并发送虚拟代理的命令（虚拟代理不进行思考）
            self.script.batch_commit_and_send(slice(0, 1))
            # 执行正常代理的行为
            self.script.batch_execute_agent(slice(1, None))
            # 接收虚拟代理的反馈（不更新其世界状态，以节省 CPU 资源）
            self.script.batch_receive(slice(0, 1), False)
            # 接收并更新其他代理的世界状态
            self.script.batch_receive(slice(1, None))

            # 如果主代理的世界状态已更新
            if p.world.vision_is_up_to_date:
                # 如果主代理的定位信息已更新
                if p.world.robot.loc_is_up_to_date:
                    # 打印定位模块接收到的数据
                    localization.print_python_data()
                    # 绘制可见元素
                    localization.draw_visible_elements(not p.world.team_side_is_left)
                    # 打印包含统计信息的报告
                    localization.print_report()
                    print("\nPress ctrl+c to return.")  # 提示用户按 Ctrl+C 返回
                    # 绘制球的位置
                    d.circle(p.world.ball_abs_pos, 0.1, 6, Draw.Color.purple_magenta, "world", False)
                else:
                    # 如果定位信息未更新，绘制提示信息
                    d.annotation(p.world.robot.cheat_abs_pos, "Not enough visual data!", Draw.Color.red, "world", False)

                # 绘制队友信息
                for o in p.world.teammates:
                    if o.state_last_update != 0 and not o.is_self:  # 如果其他机器人已被看到且不是自身
                        self._draw_other_robot(p, o, Draw.Color.white)  # 绘制队友信息

                # 绘制对手信息
                for o in p.world.opponents:
                    if o.state_last_update != 0:  # 如果其他机器人已被看到
                        self._draw_other_robot(p, o, Draw.Color.red)  # 绘制对手信息

                # 刷新绘图
                d.flush("world")

    def _draw_other_robot(self, p: Agent, o: Other_Robot, team_color):
        """
        绘制其他机器人（队友或对手）的信息。
        :param p: 主代理。
        :param o: 其他机器人。
        :param team_color: 队伍颜色。
        """
        d = self.draw  # 获取绘图对象
        white = Draw.Color.white  # 白色
        green = Draw.Color.green_light  # 浅绿色
        gray = Draw.Color.gray_20  # 灰色（20%）

        # 如果其他机器人的状态更新时间过长，调整颜色
        time_diff = p.world.time_local_ms - o.state_last_update
        if time_diff > 0:
            white = Draw.Color.gray_40  # 灰色（40%）
            green = Draw.Color.get(107, 139, 107)  # 自定义绿色
            gray = Draw.Color.gray_50  # 灰色（50%）

        # 绘制其他机器人的朝向
        if len(o.state_abs_pos) == 3:
            # 如果位置信息是三维的，计算朝向的终点
            line_tip = o.state_abs_pos + (0.5 * M.deg_cos(o.state_orientation), 0.5 * M.deg_sin(o.state_orientation), 0)
            d.line(o.state_abs_pos, line_tip, 3, white, "world", False)  # 绘制朝向线
        else:
            # 如果位置信息不是三维的，使用默认高度并绘制黄色朝向线
            temp_pos = M.to_3d(o.state_abs_pos, 0.3)
            line_tip = temp_pos + (0.5 * M.deg_cos(o.state_orientation), 0.5 * M.deg_sin(o.state_orientation), 0)
            d.line(temp_pos, line_tip, 3, Draw.Color.yellow, "world", False)

        # 绘制其他机器人的身体部位
        for pos in o.state_body_parts_abs_pos.values():
            d.sphere(pos, 0.07, green, "world", False)  # 绘制身体部位

        # 绘制其他机器人所在的地面区域
        d.circle(o.state_ground_area[0], o.state_ground_area[1], 6, team_color, "world", False)

        # 绘制主代理与该机器人之间的距离
        midpoint = (o.state_abs_pos[0:2] + p.world.robot.loc_head_position[0:2]) / 2  # 计算中点
        d.line(o.state_abs_pos[0:2], p.world.robot.loc_head_position[0:2], 1, gray, "world", False)  # 绘制连接线
        d.annotation(midpoint, f'{o.state_horizontal_dist:.2f}m', white, "world", False)  # 标注距离

        # 绘制其他机器人的速度向量
        arrow_tip = o.state_abs_pos[0:2] + o.state_filtered_velocity[0:2]  # 计算速度向量的终点
        d.arrow(o.state_abs_pos[0:2], arrow_tip, 0.2, 4, green, "world", False)  # 绘制速度箭头

        # 绘制其他机器人的状态信息
        state_color = white if not o.state_fallen else Draw.Color.yellow  # 如果机器人倒地，则使用黄色
        d.annotation((o.state_abs_pos[0], o.state_abs_pos[1], 1), 
                     f"({o.unum}) {'Fallen' if o.state_fallen else 'Normal'}", state_color, "world", False)
