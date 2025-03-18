from agent.Base_Agent import Base_Agent as Agent
from scripts.commons.Script import Script
from world.commons.Draw import Draw
import numpy as np

class Fwd_Kinematics():
    def __init__(self, script: Script) -> None:
        '''
        初始化 Fwd_Kinematics 类
        :param script: Script 对象，包含命令行参数等信息
        '''
        self.script = script
        self.cycle_duration = 200  # 每个绘制周期的步数

    def draw_cycle(self):
        '''
        绘制一个完整的周期，包括机器人的身体部位位置、关节位置和身体部位的方向
        '''
        # 绘制身体部位的位置
        for _ in range(self.cycle_duration):
            self.script.batch_execute_behavior("Squat")  # 执行下蹲行为
            self.script.batch_commit_and_send()  # 提交并发送命令
            self.script.batch_receive()  # 接收服务器反馈

            for p in self.script.players:
                if p.world.vision_is_up_to_date and not p.world.robot.loc_is_up_to_date:
                    p.world.draw.annotation(p.world.robot.cheat_abs_pos, "Not enough visual data! Using IMU", Draw.Color.red, "localization")

                for key, val in p.world.robot.body_parts.items():
                    rp = val.transform.get_translation()  # 获取身体部位的绝对位置
                    pos = p.world.robot.loc_head_to_field_transform(rp, False)  # 转换为场地坐标
                    label_rp = np.array([rp[0] - 0.0001, rp[1] * 0.5, 0])
                    label_rp /= np.linalg.norm(label_rp) / 0.4  # 标签位置在身体部位前方 0.4 米处
                    label = p.world.robot.loc_head_to_field_transform(rp + label_rp, False)
                    p.world.draw.line(pos, label, 2, Draw.Color.green_light, key, False)  # 绘制线条
                    p.world.draw.annotation(label, key, Draw.Color.red, key)  # 绘制标签

                # 绘制左脚的四个方向
                rp = p.world.robot.body_parts['lfoot'].transform((0.08, 0, 0))
                ap = p.world.robot.loc_head_to_field_transform(rp, False)
                p.world.draw.line(ap, ap + (0, 0, 0.1), 1, Draw.Color.red, "soup", False)
                rp = p.world.robot.body_parts['lfoot'].transform((-0.08, 0, 0))
                ap = p.world.robot.loc_head_to_field_transform(rp, False)
                p.world.draw.line(ap, ap + (0, 0, 0.1), 1, Draw.Color.red, "soup", False)
                rp = p.world.robot.body_parts['lfoot'].transform((0, 0.04, 0))
                ap = p.world.robot.loc_head_to_field_transform(rp, False)
                p.world.draw.line(ap, ap + (0, 0, 0.1), 1, Draw.Color.red, "soup", False)
                rp = p.world.robot.body_parts['lfoot'].transform((0, -0.04, 0))
                ap = p.world.robot.loc_head_to_field_transform(rp, False)
                p.world.draw.line(ap, ap + (0, 0, 0.1), 1, Draw.Color.red, "soup", True)

        Draw.clear_all()  # 清除所有绘制内容

        # 绘制关节的位置
        for _ in range(self.cycle_duration):
            self.script.batch_execute_behavior("Squat")  # 执行下蹲行为
            self.script.batch_commit_and_send()  # 提交并发送命令
            self.script.batch_receive()  # 接收服务器反馈

            for p in self.script.players:
                if p.world.vision_is_up_to_date and not p.world.robot.loc_is_up_to_date:
                    p.world.draw.annotation(p.world.robot.cheat_abs_pos, "Not enough visual data! Using IMU", Draw.Color.red, "localization")

                zstep = 0.05
                label_z = [0, 0, 0, 0, zstep, zstep, 2 * zstep, 2 * zstep, 0, 0, 0, 0, zstep, zstep, 0, 0, zstep, zstep, 2 * zstep, 2 * zstep, 3 * zstep, 3 * zstep, 0, 0]
                for j, transf in enumerate(p.world.robot.joints_transform):
                    rp = transf.get_translation()  # 获取关节的绝对位置
                    pos = p.world.robot.loc_head_to_field_transform(rp, False)  # 转换为场地坐标
                    j_name = str(j)
                    label_rp = np.array([rp[0] - 0.0001, rp[1] * 0.5, 0])
                    label_rp /= np.linalg.norm(label_rp) / 0.4  # 标签位置在关节前方 0.4 米处
                    label_rp += (0, 0, label_z[j])
                    label = p.world.robot.loc_head_to_field_transform(rp + label_rp, False)
                    p.world.draw.line(pos, label, 2, Draw.Color.green_light, j_name, False)  # 绘制线条
                    p.world.draw.annotation(label, j_name, Draw.Color.cyan, j_name)  # 绘制标签

        Draw.clear_all()  # 清除所有绘制内容

        # 绘制身体部位的方向
        for _ in range(self.cycle_duration):
            self.script.batch_execute_behavior("Squat")  # 执行下蹲行为
            self.script.batch_commit_and_send()  # 提交并发送命令
            self.script.batch_receive()  # 接收服务器反馈

            for p in self.script.players:
                if p.world.vision_is_up_to_date and not p.world.robot.loc_is_up_to_date:
                    p.world.draw.annotation(p.world.robot.cheat_abs_pos, "Not enough visual data! Using IMU", Draw.Color.red, "localization")

                for key in p.world.robot.body_parts:
                    # 仅选择部分身体部位
                    if key not in ['head', 'torso', 'llowerarm', 'rlowerarm', 'lthigh', 'rthigh', 'lshank', 'rshank', 'lfoot', 'rfoot']:
                        continue
                    bpart_abs_pos = p.world.robot.get_body_part_to_field_transform(key).translate((0.1, 0, 0))  # 身体部位前方 10 厘米处的位置
                    x_axis = bpart_abs_pos((0.05, 0, 0), False)  # X 轴方向
                    y_axis = bpart_abs_pos((0, 0.05, 0), False)  # Y 轴方向
                    z_axis = bpart_abs_pos((0, 0, 0.05), False)  # Z 轴方向
                    axes_0 = bpart_abs_pos.get_translation()  # 轴的起始位置
                    p.world.draw.line(axes_0, x_axis, 2, Draw.Color.green_light, key, False)  # 绘制 X 轴
                    p.world.draw.line(axes_0, y_axis, 2, Draw.Color.blue, key, False)  # 绘制 Y 轴
                    p.world.draw.line(axes_0, z_axis, 2, Draw.Color.red, key)  # 绘制 Z 轴

        Draw.clear_all()  # 清除所有绘制内容

    def execute(self):
        '''
        执行正向运动学绘制
        '''
        a = self.script.args

        # 创建 5 个代理
        # 参数：服务器 IP、代理端口、监控端口、球衣号码、机器人类型、队伍名称、启用日志、启用绘图
        self.script.batch_create(Agent, ((a.i, a.p, a.m, u, u - 1, a.t, True, True) for u in range(1, 6)))

        # 将代理传送到指定位置
        self.script.batch_unofficial_beam([(-2, i * 4 - 10, 0.5, i * 45) for i in range(5)])

        print("\nPress ctrl+c to return.")

        while True:
            self.draw_cycle()  # 绘制一个完整的周期
