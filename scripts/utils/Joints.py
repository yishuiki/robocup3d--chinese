from agent.Base_Agent import Base_Agent as Agent  # 导入基础代理类
from scripts.commons.Script import Script  # 导入脚本类
from world.commons.Draw import Draw  # 导入绘图工具类
import numpy as np  # 导入 NumPy 库用于数学计算


class Joints():
    """
    关节控制类，用于控制机器人关节的运动。
    """

    def __init__(self, script: Script) -> None:
        """
        初始化关节控制类。
        :param script: 脚本对象，用于获取相关参数。
        """
        self.script = script  # 保存脚本对象
        self.agent_pos = (-3, 0, 0.45)  # 设置代理的位置
        self.enable_pos = True  # 是否启用位置控制，默认启用
        self.enable_gravity = False  # 是否启用重力，默认不启用
        self.enable_labels = True  # 是否启用标签，默认启用
        self.enable_harmonize = True  # 是否启用关节运动同步，默认启用
        self.active_joint = 0  # 当前活动的关节编号
        self.joints_value = None  # 关节值（位置或速度），初始为 None

    def _draw_joints(self, player: Agent):
        """
        绘制关节信息。
        :param player: 代理对象。
        """
        zstep = 0.05  # Z 轴步长
        label_z = [3 * zstep, 5 * zstep, 0, 0, zstep, zstep, 2 * zstep, 2 * zstep, 0, 0, 0, 0, zstep, zstep, 0, 0, zstep, zstep, 4 * zstep, 4 * zstep, 5 * zstep, 5 * zstep, 0, 0]  # 标签的 Z 轴偏移量
        for j, transf in enumerate(player.world.robot.joints_transform):  # 遍历关节变换矩阵
            rp = transf.get_translation()  # 获取关节的平移向量
            pos = player.world.robot.loc_head_to_field_transform(rp, False)  # 将关节位置转换为场地坐标
            j_id = f"{j}"  # 关节编号
            j_name = f"{j}"  # 关节名称
            color = Draw.Color.cyan  # 默认颜色为青色
            if player.world.robot.joints_position[j] != 0:  # 如果关节位置不为零
                j_name += f" ({int(player.world.robot.joints_position[j])})"  # 在关节名称中添加位置值
                color = Draw.Color.red  # 将颜色改为红色
            label_rp = np.array([rp[0] - 0.0001, rp[1] * 0.5, 0])  # 计算标签的位置偏移量
            label_rp /= np.linalg.norm(label_rp) / 0.5  # 将标签位置归一化到距离关节 0.5 米处
            label_rp += (0, 0, label_z[j])  # 添加 Z 轴偏移量
            label = player.world.robot.loc_head_to_field_transform(rp + label_rp, False)  # 计算标签的场地坐标
            player.world.draw.line(pos, label, 2, Draw.Color.green_light, j_id, False)  # 绘制关节到标签的连线
            player.world.draw.annotation(label, j_name, color, j_id)  # 绘制关节标签

    def print_help(self):
        """
        打印帮助信息。
        """
        print(f"""
---------------------- Joints demonstration ----------------------
Command: {{action/actions/option}}
    action : [joint:{{int}}] value 
    actions: value0,value1,...,valueN
             e.g. if N=10, you control all joints from j0 to j10
    option:  {{h,s,g,l,w,r,"",.}}
Examples:
    "6 90"   - move joint 6 to 90deg or move joint 6 at 90deg/step
    "4"      - move last joint to 4deg or apply speed of 4deg/step
    "1,9,-35"- move joints 0,1,2 to 1deg, 9deg, -35deg (or speed)
    "h"      - help, display this message
    "s"      - toggle position/speed control ({"Posi" if self.enable_pos else "Spee"})
    "g"      - toggle gravity                ({self.enable_gravity})
    "l"      - toggle labels                 ({self.enable_labels})
    "w"      - toggle harmonize*             ({self.enable_harmonize})
    "r"      - reset (position mode + reset joints)
    ""       - advance 2 simulation step
    "."      - advance 1 simulation step
    "ctrl+c" - quit demonstration

    *all joints end moving at the same time when harmonize is True
------------------------------------------------------------------""")

    def _user_control_step(self, player: Agent):
        """
        用户控制步骤。
        :param player: 代理对象。
        """
        while True:
            inp = input("Command: ")
            if inp == "s":  # 切换位置/速度控制
                self.enable_pos = not self.enable_pos
                print("Using", "position" if self.enable_pos else "velocity", "control.")
                if self.enable_pos:
                    self.joints_value[:] = player.world.robot.joints_position  # 如果切换到位置控制，将关节值设置为当前关节位置
                else:
                    self.joints_value.fill(0)  # 如果切换到速度控制，将关节值清零
                continue
            elif inp == "g":  # 切换重力
                self.enable_gravity = not self.enable_gravity
                print("Using gravity:",self.enable_gravity)
                continue
            elif inp == "l":  # 切换标签
                self.enable_labels = not self.enable_labels
                print("Using labels:",self.enable_labels)
                continue
            elif inp == "w":  # 切换同步
                self.enable_harmonize = not self.enable_harmonize
                print("Using harmonize:",self.enable_harmonize)
                continue
            elif inp == "r":  # 重置
                self.enable_pos = True
                self.joints_value.fill(0)
                print("Using position control. All joints are set to zero.")
                continue
            elif inp == "h":  # 显示帮助
                self.print_help()
                continue

            elif inp == "":  # 前进 2 个仿真步
                return 1
            elif inp == ".":  # 前进 1 个仿真步
                return 0

            try:
                if " " in inp:  # 如果输入中包含空格，表示指定关节编号和值
                    self.active_joint, value = map(float, inp.split())
                    self.joints_value[int(self.active_joint)] = value
                elif "," in inp:  # 如果输入中包含逗号，表示设置多个关节的值
                    values = inp.split(",")
                    self.joints_value[0:len(values)] = values
                else:  # 如果输入中没有空格或逗号，表示设置当前活动关节的值
                    self.joints_value[self.active_joint] = float(inp)
            except:
                print("非法命令！")
                continue

    def execute(self):
        """
        执行关节控制。
        """
        a = self.script.args  # 获取脚本参数
        player = Agent(a.i, a.p, a.m, a.u, a.r, a.t)  # 创建代理对象

        self.joints_no = player.world.robot.no_of_joints  # 获取关节数量
        self.joints_value = np.zeros(self.joints_no)  # 初始化关节值数组

        player.scom.commit_beam(self.agent_pos[0:2], 0)  # 提交代理位置

        self.print_help()  # 打印帮助信息

        # 初始化（+光束）
        for _ in range(8):
            player.scom.commit_and_send()
            player.scom.receive()
        self._draw_joints(player)  # 绘制关节信息

        skip_next = 0  # 用于前进多个仿真步的变量

        while True:
            if skip_next == 0:
                skip_next = self._user_control_step(player)  # 执行用户控制步骤
            else:
                skip_next -= 1  # 如果需要前进多个仿真步，则减少剩余步
            # 如果启用了标签，则重新绘制关节信息
            if self.enable_labels:
                self._draw_joints(player)

            # 根据当前的控制模式（位置或速度）设置关节目标值
            if self.enable_pos:
                # 如果是位置控制模式，设置关节目标位置
                player.world.robot.set_joints_target_position_direct(slice(self.joints_no), self.joints_value, harmonize=self.enable_harmonize)
            else:
                # 如果是速度控制模式，设置关节目标速度
                player.world.robot.joints_target_speed[:] = self.joints_value * 0.87266463  # 将输入的速度值从度/步转换为弧度/秒

            # 如果未启用重力，使用非官方的光束功能来抵消重力
            if not self.enable_gravity:
                player.scom.unofficial_beam(self.agent_pos, 0)

            # 提交机器人命令并发送
            player.scom.commit_and_send(player.world.robot.get_command())
            # 接收仿真环境的反馈
            player.scom.receive()
