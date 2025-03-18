from agent.Agent import Agent  # 导入代理类
from itertools import count  # 导入 itertools 模块用于无限循环计数
from scripts.commons.Script import Script  # 导入脚本类
from typing import List  # 导入类型注解模块
from world.commons.Draw import Draw  # 导入绘图工具类
class Radio_Localization():
    def __init__(self, script: Script) -> None:
        """
        初始化无线电定位类。
        :param script: 脚本对象，用于获取相关参数。
        """
        self.script = script  # 保存脚本对象
    def draw_objects(self, p: Agent, pos, is_down, was_seen, last_update, is_self=False):
        """
        绘制对象（球员或球）的状态。
        :param p: 代理对象。
        :param pos: 对象的位置。
        :param is_down: 对象是否倒地。
        :param was_seen: 对象是否被看到。
        :param last_update: 对象最后更新的时间。
        :param is_self: 是否为自身。
        """
        w = p.world  # 获取世界对象
        me = w.robot.loc_head_position  # 获取自身头部位置

        # 获取绘图对象，确保始终覆盖之前的绘图
        d: Draw = self.script.players[0].world.draw

        # 判断对象是否为当前或最近更新
        is_current = last_update > w.time_local_ms - w.VISUALSTEP_MS  # 是否为当前或上一个时间步更新
        is_recent = last_update >= w.time_local_ms - 120  # 是否在过去 0.12 秒内更新

        # 根据对象的状态选择颜色
        if is_current and was_seen:
            c = d.Color.green_light  # 当前或上一个时间步看到的对象
        elif is_recent and was_seen:
            c = d.Color.green  # 最近 0.12 秒内看到的对象
        elif is_current:
            c = d.Color.yellow  # 当前或上一个时间步通过无线电听到的对象
        elif is_recent:
            c = d.Color.yellow_light  # 最近 0.12 秒内通过无线电听到的对象
        else:
            c = d.Color.red  # 超过 0.12 秒未看到或听到的对象

        # 绘制自身状态
        if is_self:
            if w.robot.radio_fallen_state:
                d.annotation(me, "Fallen (radio)", d.Color.yellow, "objects", False)  # 通过无线电听到自己倒地
            elif w.robot.loc_head_z < 0.3:
                d.annotation(me, "Fallen (internal)", d.Color.white, "objects", False)  # 自身检测到倒地
            d.sphere(me, 0.06, c, "objects", False)  # 绘 ‌‍
    def draw(self, p: Agent):
        """
        绘制所有对象的状态。
        :param p: 代理对象。
        """
        w = p.world  # 获取世界对象
        others = w.teammates + w.opponents  # 获取队友和对手

        # 绘制其他球员
        for o in others:
            if o.is_self or o.state_last_update == 0:  # 不绘制自身或从未见过的球员
                continue

            pos = o.state_abs_pos  # 获取球员位置
            is_down = o.state_fallen  # 获取球员是否倒地
            is_3D = pos is not None and len(pos) == 3  # 判断位置是否为 3D（头部可见）

            self.draw_objects(p, pos, is_down, is_3D, o.state_last_update)  # 绘制球员状态

        # 绘制自身状态
        is_pos_from_vision = w.robot.loc_head_position_last_update == w.robot.loc_last_update
        self.draw_objects(p, None, None, is_pos_from_vision, w.robot.loc_head_position_last_update, True)

        # 绘制球的状态
        self.draw_objects(p, w.ball_abs_pos, False, w.is_ball_abs_pos_from_vision, w.ball_abs_pos_last_update)

        # 刷新绘图
        self.script.players[0].world.draw.flush("objects")
    def execute(self):
        """
        执行无线电定位和可视化。
        """
        a = self.script.args  # 获取脚本参数

        # 创建球员代理（包括队友和对手）
        self.script.batch_create(Agent, ((a.i, a.p, a.m, u, a.t, False, u == 1) for u in range(1, 12)))
        self.script.batch_create(Agent, ((a.i, a.p, a.m, u, "Opponent", False, False) for u in range(1, 12)))
        players: List[Agent] = self.script.players  # 获取所有球员

        # 设置对手的初始位置
        beam_pos = [(-(i // 2) - 3, (i % 2 * 2 - 1) * (i // 2 + 1), 0) for i in range(11)]
        self.script.batch_commit_beam(beam_pos, slice(11, None))
        print("\nPress ctrl+c to return.")

        # 主循环
        for j in count():
            self.script.batch_execute_agent(slice(11))  # 执行队友的行为
            self.script.batch_commit_and_send(slice(11, None))  # 提交并发送对手的行为

            # 绘制球员和球的状态，每 15 个时间步绘制一个球员
            self.draw(players[j // 15 % 11])
            self.script.batch_receive(slice(11))  # 接收并更新队友的世界状态
            self.script.batch_receive(slice(11, None), False)  # 接收但不更新对手的世界状态（节省 CPU 资源）
