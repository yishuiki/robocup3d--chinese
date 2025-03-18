from agent.Base_Agent import Base_Agent as Agent
from itertools import count
from math_ops.Inverse_Kinematics import Inverse_Kinematics
from scripts.commons.Script import Script
from world.commons.Draw import Draw
import numpy as np


class Inv_Kinematics():
    def __init__(self, script: Script) -> None:
        '''
        初始化逆运动学演示
        :param script: Script 对象，包含命令行参数等信息
        '''
        self.args = script.args
        self.last_action = (0, 0, 0)  # 上一次操作的记录
        self.gravity = True  # 是否启用重力

        # 初始姿态是一个中立姿态，所有角度均为 0
        leg_y_dev, upper_leg_height, upper_leg_depth, lower_leg_len, _, _ = Inverse_Kinematics.NAO_SPECS_PER_ROBOT[self.args.r]
        leg_height = upper_leg_height + lower_leg_len
        self.feet_pose = [  # 初始化双脚的姿态
            [[upper_leg_depth, leg_y_dev, -leg_height], [0, 0, 0]],  # 左脚
            [[upper_leg_depth, -leg_y_dev, -leg_height], [0, 0, 0]]  # 右脚
        ]

    def _user_control(self):
        '''
        用户控制接口，用于输入命令和调整脚部姿态
        '''
        while True:
            inp = input("Command:")
            if inp == "": return 2  # 空输入，前进 2 步
            elif inp == ".": return 1  # 单点，前进 1 步
            elif inp == "h": self.print_help(); continue  # 显示帮助信息
            elif inp == "g": 
                self.gravity = not self.gravity  # 切换重力模式
                print("Using gravity:", self.gravity)
                if self.gravity:
                    return 6  # 额外的步数，以便传送生效
                else:
                    return 1

            # 检查用户输入是否为数值
            try: 
                val = float(inp)
                self.feet_pose[self.last_action[0]][self.last_action[1]][self.last_action[2]] = val
                continue
            except:
                pass

            if inp[0] not in ['l', 'r'] or inp[1] not in ['x', 'y', 'z', 'X', 'Y', 'Z']:
                print("Illegal command!")
                continue

            side = 0 if inp[0] == 'l' else 1  # 左脚或右脚
            pos_rot = 0 if inp[1].islower() else 1  # 位置或旋转
            axis = {'x': 0, 'y': 1, 'z': 2}[inp[1].lower()]  # 轴
            self.last_action = (side, pos_rot, axis)  # 更新上一次操作

            try:
                val = float(inp[2:])
                self.feet_pose[side][pos_rot][axis] = val  # 更新脚部姿态
            except:
                print("Illegal value conversion!")

    def _draw_labels(self, player: Agent):
        '''
        绘制标签和注释
        :param player: Agent 对象
        '''
        r = player.world.robot
        robot_pos = r.loc_head_position
        for i, body_part in enumerate(['lankle', 'rankle']):
            pos = r.get_body_part_abs_position(body_part)
            label_rel_pos = np.array([-0.2, (0.5 - i), 0])
            label_rel_pos /= np.linalg.norm(label_rel_pos) / 1.0  # 标签距离身体部位 1.0 米
            player.world.draw.line(pos, pos + label_rel_pos, 2, Draw.Color.green_light, body_part, False)
            p = self.feet_pose[i]
            pose_text = (
                f"x:{p[0][0]:.4f}    y:{p[0][1]:.4f}    z:{p[0][2]:.4f}",
                f"rol:{p[1][0]:.2f} (bias)   pit:{p[1][1]:.2f} (bias)   yaw:{p[1][2]:.2f}"
            )
            player.world.draw.annotation(pos + label_rel_pos + [0, 0, 0.2], pose_text[0], Draw.Color.cyan, body_part, False)
            player.world.draw.annotation(pos + label_rel_pos + [0, 0, 0.1], pose_text[1], Draw.Color.cyan, body_part, False)

            # 绘制正向运动学数据（踝关节位置 + 脚部旋转）
            p = player.inv_kinematics.get_body_part_pos_relative_to_hip(body_part)  # 踝关节相对于髋关节的位置
            foot_rel_torso = r.head_to_body_part_transform("torso", r.body_parts[['lfoot', 'rfoot'][i]].transform)
            w = foot_rel_torso.get_roll_deg(), foot_rel_torso.get_pitch_deg(), foot_rel_torso.get_yaw_deg()
            pose_text = (
                f"x:{p[0]:.4f}    y:{p[1]:.4f}    z:{p[2]:.4f}",
                f"rol:{w[0]:.4f}    pit:{w[1]:.4f}    yaw:{w[2]:.4f}"
            )

            player.world.draw.annotation(pos + label_rel_pos + [0, 0, -0.2], pose_text[0], Draw.Color.red, body_part, False)
            player.world.draw.annotation(pos + label_rel_pos + [0, 0, -0.3], pose_text[1], Draw.Color.red, body_part, False)
            player.world.draw.annotation(pos + label_rel_pos + [0, 0, -0.4], "(forward kinematics data)", Draw.Color.red, body_part, True)

        note = f"Torso roll: {r.imu_torso_roll:.2f}   Torso pitch: {r.imu_torso_pitch:.2f}"
        player.world.draw.annotation(robot_pos + [0, 0, 0.10], note, Draw.Color.red, "Torso")

    def print_help(self):
        '''
        打印帮助信息
        '''
        print("""
---------------- Inverse kinematics demonstration ----------------
INPUT: ankle positions + feet rotations  (relative coordinates)
OUTPUT: angular positions of both legs' joints 
------------------------------------------------------------------
Command: {action/option}
    action: [side:{l/r} axis*:{x/y/z/X/Y/Z}] value 
            *for position use x/y/z, for rotation use X/Y/Z
    option: {"",.,g,h}
Examples:
    "lz-0.12" - move left ankle to -0.1m in the z-axis
    "rX30.5"  - rotate right foot to 30.5 deg in the x-axis (roll)
    "20"      - repeat last action but change value to 20
    ""        - advance 2 simulation steps
    "."       - advance 1 simulation step
    "g"       - toggle gravity
    "h"       - help, display this message
    "ctrl+c"  - quit demonstration
------------------------------------------------------------------""")

    def execute(self):
        '''
        执行逆运动学演示
        '''
        self.state = 0
        a = self.args
        
        self.print_help()
        player = Agent(a.i, a.p, a.m, a.u, a.r, a.t)  # 参数：服务器 IP、代理端口、监控端口、球衣号码、机器人类型、队伍名称

        player.scom.unofficial_beam((-3, 0, 0.42), 0)  # 将代理传送到初始位置

        next_control_step = 20  # 下一步控制的时间步

        for i in count():
            if self.gravity: 
                player.scom.unofficial_beam((-3, 0, 0.42), 0)  # 启用重力

            self._draw_labels(player)  # 绘制标签和注释

            if i == next_control_step:
                next_control_step += self._user_control()  # 用户控制

            for i in range(2):  # 左腿和右腿
                indices, values, error_codes = player.inv_kinematics.leg(self.feet_pose[i][0], self.feet_pose[i][1], bool(i == 0), False)

                if -1 in error_codes: 
                    print("Position is out of reach!")
                    error_codes.remove(-1)
                for j in error_codes:
                    print(f"Joint {j} is out of range!")

                player.world.robot.set_joints_target_position_direct(indices, values)

            player.scom.commit_and_send(player.world.robot.get_command())  # 提交并发送命令
            player.scom.receive()  # 接收服务器反馈

