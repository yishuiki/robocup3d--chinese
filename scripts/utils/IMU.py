from agent.Base_Agent import Base_Agent as Agent
from math_ops.Matrix_3x3 import Matrix_3x3
from math_ops.Matrix_4x4 import Matrix_4x4
from scripts.commons.Script import Script
from world.commons.Draw import Draw
from world.Robot import Robot
import numpy as np

'''
目标：
----------
演示 IMU 的准确性
Robot.imu_(...) 变量基于视觉定位算法和 IMU，当没有视觉数据时。
如果视觉数据不可用超过 0.2 秒，机器人的位置将被冻结，速度将衰减至零。
IMU 计算的旋转如此精确，以至于无论机器人在没有视觉数据的情况下持续多长时间，它都不会被冻结。
在几乎所有情况下，使用 IMU 数据进行旋转都是安全的。
已知问题：加速度计在存在“瞬间”加速度峰值时不可靠，因为其采样率较低（50Hz）
              这一限制影响了在碰撞期间（例如摔倒、与其他球员相撞）的平移估计
'''

class IMU():
    def __init__(self, script: Script) -> None:
        '''
        初始化 IMU 类
        :param script: Script 对象，包含命令行参数等信息
        '''
        self.script = script
        self.player: Agent = None  # 初始化时未指定代理
        self.cycle = 0  # 初始化周期计数器

        # 初始化 IMU 相关的矩阵和变量
        self.imu_torso_to_field_rotation = [Matrix_3x3() for _ in range(3)]
        self.imu_torso_to_field_transform = [Matrix_4x4() for _ in range(3)]
        self.imu_head_to_field_transform = [Matrix_4x4() for _ in range(3)]
        self.imu_torso_position = np.zeros((3, 3))
        self.imu_torso_velocity = np.zeros((3, 3))
        self.imu_torso_acceleration = np.zeros((3, 3))
        self.imu_torso_next_position = np.zeros((3, 3))
        self.imu_torso_next_velocity = np.zeros((3, 3))
        self.imu_CoM_position = np.zeros((3, 3))
        self.colors = [Draw.Color.green_light, Draw.Color.yellow, Draw.Color.red]  # 定义绘制颜色

    def act(self):
        '''
        执行动作，控制机器人进行一系列关节运动
        '''
        r = self.player.world.robot
        joint_indices = [r.J_LLEG_PITCH, 
                         r.J_LKNEE, 
                         r.J_LFOOT_PITCH,
                         r.J_LARM_ROLL,
                         r.J_RLEG_PITCH, 
                         r.J_RKNEE, 
                         r.J_RFOOT_PITCH,
                         r.J_RARM_ROLL]
        
        amplitude = [1, 0.93, 1, 1, 1][r.type]

        self.cycle += 1
        if self.cycle < 50:
            r.set_joints_target_position_direct(joint_indices, np.array([32 + 10, -64, 32, 45, 40 + 10, -80, 40, 0]) * amplitude)
        elif self.cycle < 100:
            r.set_joints_target_position_direct(joint_indices, np.array([-10, 0, 0, 0, -10, 0, 0, 0]) * amplitude)
        elif self.cycle < 150:
            r.set_joints_target_position_direct(joint_indices, np.array([40 + 10, -80, 40, 0, 32 + 10, -64, 32, 45]) * amplitude)
        elif self.cycle < 200:
            r.set_joints_target_position_direct(joint_indices, np.array([-10, 0, 0, 0, -10, 0, 0, 0]) * amplitude)
        else:
            self.cycle = 0

        self.player.scom.commit_and_send(r.get_command())
        self.player.scom.receive()

    def act2(self):
        '''
        执行行走动作
        '''
        r = self.player.world.robot
        self.player.behavior.execute("Walk", (0.2, 0), False, 5, False, None)  # 参数：目标速度、是否绝对目标、方向、是否绝对方向、距离
        self.player.scom.commit_and_send(r.get_command())
        self.player.scom.receive()

    def draw_player_reference_frame(self, i):
        '''
        绘制机器人的参考框架
        :param i: 指定绘制哪个 IMU 的参考框架
        '''
        pos = self.imu_torso_position[i]
        xvec = self.imu_torso_to_field_rotation[i].multiply((1, 0, 0)) + pos
        yvec = self.imu_torso_to_field_rotation[i].multiply((0, 1, 0)) + pos
        zvec = self.imu_torso_to_field_rotation[i].multiply((0, 0, 1)) + pos
        self.player.world.draw.arrow(pos, xvec, 0.2, 2, self.colors[i], "IMU" + str(i), False)
        self.player.world.draw.arrow(pos, yvec, 0.2, 2, self.colors[i], "IMU" + str(i), False)
        self.player.world.draw.arrow(pos, zvec, 0.2, 2, self.colors[i], "IMU" + str(i), False)
        self.player.world.draw.annotation(xvec, "x", Draw.Color.white, "IMU" + str(i), False)
        self.player.world.draw.annotation(yvec, "y", Draw.Color.white, "IMU" + str(i), False)
        self.player.world.draw.annotation(zvec, "z", Draw.Color.white, "IMU" + str(i), False)
        self.player.world.draw.sphere(self.imu_CoM_position[i], 0.04, self.colors[i], "IMU" + str(i), True)

    def compute_local_IMU(self):
        '''
        计算局部 IMU 数据（包括位置和旋转）
        '''
        r = self.player.world.robot
        g = r.gyro / 50  # 将每秒的度数转换为每步的度数
        self.imu_torso_to_field_rotation[2].multiply(Matrix_3x3.from_rotation_deg(g), in_place=True, reverse_order=True)
        self.imu_torso_position[2][:] = self.imu_torso_next_position[2]
        if self.imu_torso_position[2][2] < 0: self.imu_torso_position[2][2] = 0  # 限制 z 坐标为正值
        self.imu_torso_velocity[2][:] = self.imu_torso_next_velocity[2]

        # 将加速度计数据转换为坐标加速度并修正舍入偏差
        self.imu_torso_acceleration[2] = self.imu_torso_to_field_rotation[2].multiply(r.acc) + Robot.GRAVITY
        self.imu_torso_to_field_transform[2] = Matrix_4x4.from_3x3_and_translation(self.imu_torso_to_field_rotation[2], self.imu_torso_position[2])
        self.imu_head_to_field_transform[2] = self.imu_torso_to_field_transform[2].multiply(r.body_parts["torso"].transform.invert())
        self.imu_CoM_position[2][:] = self.imu_head_to_field_transform[2](r.rel_cart_CoM_position)

        # 计算下一位置和速度
        self.imu_torso_next_position[2] = self.imu_torso_position[2] + self.imu_torso_velocity[2] * 0.02 + self.imu_torso_acceleration[2] * 0.0002
        self.imu_torso_next_velocity[2] = self.imu_torso_velocity[2] + self.imu_torso_acceleration[2] * 0.02
        self.imu_torso_next_velocity[2] *= Robot.IMU_DECAY  # 稳定性权衡

    def compute_local_IMU_rotation_only(self):
        '''
        计算局部 IMU 数据（仅旋转）
        '''
        r = self.player.world.robot
        g = r.gyro / 50  # 将每秒的度数转换为每步的度数
        self.imu_torso_to_field_rotation[1].multiply(Matrix_3x3.from_rotation_deg(g), in_place=True, reverse_order=True)
        self.imu_torso_position[1][:] = r.loc_torso_position
        self.imu_torso_to_field_transform[1] = Matrix_4x4.from_3x3_and_translation(self.imu_torso_to_field_rotation[1], self.imu_torso_position[1])
        self.imu_head_to_field_transform[1] = self.imu_torso_to_field_transform[1].multiply(r.body_parts["torso"].transform.invert())
        self.imu_CoM_position[1][:] = self.imu_head_to_field_transform[1](r.rel_cart_CoM_position)

    def update_local_IMU(self, i):
        '''
        更新局部 IMU 数据
        :param i: 指定更新哪个 IMU 的数据
        '''
        r = self.player.world.robot
        self.imu_torso_to_field_rotation[i].m[:] = r.imu_torso_to_field_rotation.m
        self.imu_torso_to_field_transform[i].m[:] = r.imu_weak_torso_to_field_transform.m
        self.imu_head_to_field_transform[i].m[:] = r.imu_weak_head_to_field_transform.m
        self.imu_torso_position[i][:] = r.imu_weak_torso_position
        self.imu_torso_velocity[i][:] = r.imu_weak_torso_velocity
        self.imu_torso_acceleration[i][:] = r.imu_weak_torso_acceleration
        self.imu_torso_next_position[i] = self.imu_torso_position[i] + self.imu_torso_velocity[i] * 0.02 + self.imu_torso_acceleration[i] * 0.0002
        self.imu_torso_next_velocity[i] = self.imu_torso_velocity[i] + self.imu_torso_acceleration[i] * 0.02
        self.imu_CoM_position[i][:] = r.imu_weak_CoM_position

    def execute(self):
        '''
        执行 IMU 演示
        '''
        a = self.script.args    
        self.player = Agent(a.i, a.p, a.m, a.u, a.r, a.t)  # 参数：服务器 IP、代理端口、监控端口、球衣号码、机器人类型、队伍名称

        # 将代理传送到初始位置
        self.player.scom.unofficial_beam((-3, 0, self.player.world.robot.beam_height), 15)

        # 初始化位置
        for _ in range(10): 
            self.player.scom.commit_and_send()
            self.player.scom.receive()

        # 添加注释
        self.player.world.draw.annotation((-3, 1, 1.1), "IMU + Localizer", self.colors[0], "note_IMU_1", True)

        # 第一部分：仅使用 IMU + 定位器
        for _ in range(150):
            self.act()
            self.update_local_IMU(0)
            self.draw_player_reference_frame(0)

        # 添加注释
        self.player.world.draw.annotation((-3, 1, 0.9), "IMU for rotation", self.colors[1], "note_IMU_2", True)
        self.update_local_IMU(1)

        # 第二部分：仅使用 IMU 旋转
        for _ in range(200):
            self.act()   
            self.update_local_IMU(0)
            self.draw_player_reference_frame(0)
            self.compute_local_IMU_rotation_only()
            self.draw_player_reference_frame(1)

        # 添加注释
        self.player.world.draw.annotation((-3, 1, 0.7), "IMU for rotation & position", self.colors[2], "note_IMU_3", True)
        self.update_local_IMU(2)

        # 第三部分：使用 IMU 旋转和位置
        for _ in range(200):
            self.act()
            self.update_local_IMU(0)
            self.draw_player_reference_frame(0)
            self.compute_local_IMU_rotation_only()
            self.draw_player_reference_frame(1)
            self.compute_local_IMU()
            self.draw_player_reference_frame(2)

        print("\nPress ctrl+c to return.")

        # 仍然使用 “IMU for rotation & position”，但现在开始行走
        self.update_local_IMU(2)
        while True:
            self.act2()
            self.update_local_IMU(0)
            self.draw_player_reference_frame(0)
            self.compute_local_IMU_rotation_only()
            self.draw_player_reference_frame(1)
            self.compute_local_IMU()
            self.draw_player_reference_frame(2)

