from agent.Base_Agent import Base_Agent as Agent  # 导入基础代理类
from math_ops.Math_Ops import Math_Ops as M  # 导入数学操作类
from scripts.commons.Script import Script  # 导入脚本类
import numpy as np  # 导入 NumPy 库用于数学计算

'''
目标：
展示踢球动作
'''

class Kick():
    def __init__(self, script: Script) -> None:
        """
        初始化踢球类。
        :param script: 脚本对象，用于获取相关参数。
        """
        self.script = script  # 保存脚本对象

    def execute(self):
        """
        执行踢球动作。
        """
        a = self.script.args  # 获取脚本参数
        player = Agent(a.i, a.p, a.m, a.u, a.r, a.t)  # 创建代理对象，参数分别为服务器 IP、代理端口、监控端口、球衣号码、机器人类型、队伍名称
        player.path_manager.draw_options(enable_obstacles=True, enable_path=True)  # 启用障碍物和路径绘制
        behavior = player.behavior  # 获取行为控制器
        w = player.world  # 获取世界对象
        r = w.robot  # 获取机器人对象

        print("\nThe robot will kick towards the center of the field")  # 提示机器人将向场地中心踢球
        print("Try to manually relocate the ball")  # 提示用户可以手动移动球的位置
        print("Press ctrl+c to return\n")  # 提示用户按 Ctrl+C 返回

        player.scom.unofficial_set_play_mode("PlayOn")  # 设置比赛模式为“进行中”
        player.scom.unofficial_beam((-3, 0, r.beam_height), 0)  # 将机器人移动到指定位置
        vec = (1, 0)  # 初始化踢球方向向量

        while True:
            player.scom.unofficial_set_game_time(0)  # 将游戏时间设置为 0
            b = w.ball_abs_pos[:2]  # 获取球的绝对位置（仅取前两个坐标）

            # 如果球的速度接近 0（表示长时间未看到球），并且球与机器人的相对距离大于 0.5 米，则更新踢球方向
            if 0 < np.linalg.norm(w.get_ball_abs_vel(6)) < 0.02:
                if np.linalg.norm(w.ball_rel_head_cart_pos[:2]) > 0.5:
                    # 如果球的位置在场地中心附近（坐标绝对值小于 0.5），则将踢球方向设置为向场地中心
                    if max(abs(b)) < 0.5:
                        vec = np.array([6, 0])
                    else:
                        # 否则，计算从球到场地中心的单位向量，并将其长度设置为 6
                        vec = M.normalize_vec((0, 0) - b) * 6

            # 在球的位置加上踢球方向向量的位置绘制一个粉色的点，表示踢球目标位置
            w.draw.point(b + vec, 8, w.draw.Color.pink, "target")

            # 执行“基本踢球”行为，传入踢球方向的角度
            behavior.execute("Basic_Kick", M.vector_angle(vec))

            # 提交机器人命令并发送
            player.scom.commit_and_send(r.get_command())
            # 接收仿真环境的反馈
            player.scom.receive()

            # 如果机器人准备好执行“起身”行为，则将机器人移动到当前位置，并执行“零弯膝”行为
            if behavior.is_ready("Get_Up"):
                player.scom.unofficial_beam((*r.loc_head_position[0:2], r.beam_height), 0)
                behavior.execute_to_completion("Zero_Bent_Knees")
