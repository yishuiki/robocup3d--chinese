from agent.Base_Agent import Base_Agent as Agent  # 导入基础智能体类
from behaviors.custom.Step.Step import Step  # 导入自定义的步态行为类
from world.commons.Draw import Draw  # 导入绘图工具
from stable_baselines3 import PPO  # 导入 PPO 算法
from stable_baselines3.common.vec_env import SubprocVecEnv  # 导入多进程向量化环境
from scripts.commons.Server import Server  # 导入服务器管理类
from scripts.commons.Train_Base import Train_Base  # 导入基础训练类
from time import sleep  # 导入睡眠函数
import os, gym  # 导入操作系统和 Gym 库
import numpy as np  # 导入 NumPy 库

'''
目标：
学习如何使用步态原语（step primitive）向前行走
----------
- 类 Basic_Run：实现一个自定义的 OpenAI Gym 环境
- 类 Train：实现用于训练新模型或测试现有模型的算法
'''

class Basic_Run(gym.Env):

    def __init__(self, ip, server_p, monitor_p, r_type, enable_draw) -> None:
        '''
        初始化自定义 Gym 环境
        参数：
        - ip: 服务器 IP 地址
        - server_p: 服务器端口
        - monitor_p: 监控端口
        - r_type: 机器人类型
        - enable_draw: 是否启用绘图
        '''
        self.robot_type = r_type  # 机器人类型

        # 初始化智能体
        # 参数：服务器 IP、智能体端口、监控端口、机器人类型、团队名称、启用日志、启用绘图
        self.player = Agent(ip, server_p, monitor_p, 1, self.robot_type, "Gym", True, enable_draw)
        self.step_counter = 0  # 用于限制 episode 的大小

        # 获取步态行为对象
        self.step_obj: Step = self.player.behavior.get_custom_behavior_object("Step")

        # 状态空间
        obs_size = 70  # 观测值大小
        self.obs = np.zeros(obs_size, np.float32)  # 初始化观测值
        self.observation_space = gym.spaces.Box(
            low=np.full(obs_size, -np.inf, np.float32), 
            high=np.full(obs_size, np.inf, np.float32), 
            dtype=np.float32
        )  # 定义观测空间

        # 动作空间
        MAX = np.finfo(np.float32).max  # 浮点数的最大值
        self.no_of_actions = act_size = 22  # 动作数量
        self.action_space = gym.spaces.Box(
            low=np.full(act_size, -MAX, np.float32), 
            high=np.full(act_size, MAX, np.float32), 
            dtype=np.float32
        )  # 定义动作空间

        # 步态行为的默认参数
        self.step_default_dur = 7  # 步态持续时间
        self.step_default_z_span = 0.035  # Z 轴移动范围
        self.step_default_z_max = 0.70  # Z 轴最大值

        # 将球放置在远处，以保持地标在视野内（使用步态行为时，头部会跟随球移动）
        self.player.scom.unofficial_move_ball((14, 0, 0.042))

    def observe(self, init=False):
        '''
        获取环境的观测值
        参数：
        - init: 是否为初始化状态
        '''
        r = self.player.world.robot  # 获取机器人对象

        # 观测值索引及其含义（简单归一化处理）
        self.obs[0] = self.step_counter / 100  # 时间步计数器
        self.obs[1] = r.loc_head_z * 3  # 头部 Z 坐标（身体）
        self.obs[2] = r.loc_head_z_vel / 2  # 头部 Z 轴速度
        self.obs[3] = r.imu_torso_orientation / 50  # 躯干绝对方向（度）
        self.obs[4] = r.imu_torso_roll / 15  # 躯干滚转角（度）
        self.obs[5] = r.imu_torso_pitch / 15  # 躯干俯仰角（度）
        self.obs[6:9] = r.gyro / 100  # 陀螺仪数据
        self.obs[9:12] = r.acc / 10  # 加速度计数据

        # 左脚和右脚的相对原点和力向量（px, py, pz, fx, fy, fz）
        self.obs[12:18] = r.frp.get('lf', (0, 0, 0, 0, 0, 0))  # 左脚
        self.obs[18:24] = r.frp.get('rf', (0, 0, 0, 0, 0, 0))  # 右脚
        self.obs[15:18] /= 100  # 归一化力向量
        self.obs[21:24] /= 100  # 归一化力向量
        self.obs[24:44] = r.joints_position[2:22] / 100  # 除头部和脚趾外的所有关节位置（机器人类型 4）
        self.obs[44:64] = r.joints_speed[2:22] / 6.1395  # 除头部和脚趾外的所有关节速度

        # 如果脚未接触地面，则 (px=0, py=0, pz=0, fx=0, fy=0, fz=0)

        if init:  # 如果是初始化状态
            # 步态参数初始化
            self.obs[64] = self.step_default_dur / 10  # 步态持续时间（时间步）
            self.obs[65] = self.step_default_z_span * 20  # Z 轴移动范围
            self.obs[66] = self.step_default_z_max  # 支撑腿的相对伸展
            self.obs[67] = 1  # 步态进度
            self.obs[68] = 1  # 左腿是否活跃
            self.obs[69] = 0  # 右腿是否活跃
        else:
            # 使用步态行为对象的当前状态
            self.obs[64] = self.step_obj.step_generator.ts_per_step / 10  # 步态持续时间（时间步）
            self.obs[65] = self.step_obj.step_generator.swing_height * 20  # Z 轴移动范围
            self.obs[66] = self.step_obj.step_generator.max_leg_extension / self.step_obj.leg_length  # 支撑腿的相对伸展
            self.obs[67] = self.step_obj.step_generator.external_progress  # 步态进度
            self.obs[68] = float(self.step_obj.step_generator.state_is_left_active)  # 左腿是否活跃
            self.obs[69] = float(not self.step_obj.step_generator.state_is_left_active)  # 右腿是否活跃

        return self.obs

    def sync(self):
        '''
        运行单步模拟
        '''
        r = self.player.world.robot
        self.player.scom.commit_and_send(r.get_command())  # 提交并发送命令
        self.player.scom.receive()  # 接收数据

    def reset(self):
        '''
        重置并稳定机器人
        注意：对于某些行为，减少稳定时间或添加噪声可能更好
        '''
        self.step_counter = 0
        r = self.player.world.robot

        # 将机器人连续传送至空中（漂浮在地面上方）
        for _ in range(25):
            self.player.scom.unofficial_beam((-14, 0, 0.50), 0)
            self.player.behavior.execute("Zero_Bent_Knees")  # 执行零弯曲膝盖行为
            self.sync()

        # 将机器人传送至地面
        self.player.scom.unofficial_beam((-14, 0, r.beam_height), 0) 
        r.joints_target_speed[0] = 0.01  # 移动头部以触发物理更新（rcssserver3d 的一个 Bug，当没有关节移动时会出现问题）
        self.sync()

        # 在地面上稳定机器人
        for _ in range(7): 
            self.player.behavior.execute("Zero_Bent_Knees")  # 执行零弯曲膝盖行为
            self.sync()

        # 初始化记忆变量
        self.lastx = r.cheat_abs_pos[0]  # 上一次的 X 轴位置
        self.act = np.zeros(self.no_of_actions, np.float32)  # 初始化动作数组

        return self.observe(True)  # 返回初始化状态的观测值

    def render(self, mode='human', close=False):
        '''
        渲染环境（此处为空实现）
        '''
        return

    def close(self):
        '''
        关闭环境
        '''
        Draw.clear_all()  # 清除所有绘图
        self.player.terminate()  # 终止智能体

    def step(self, action):
        '''
        执行一步动作
        参数：
        - action: 动作向量
        '''
        r = self.player.world.robot  # 获取机器人对象

        # 使用指数移动平均法更新动作
        self.act = 0.4 * self.act + 0.6 * action

        # 执行 Step 行为以获取每条腿的目标位置（我们将覆盖这些目标）
        if self.step_counter == 0:
            '''
            第一次时间步将改变下一步的步态参数。
            使用默认参数，以便智能体可以预测下一步的目标姿态。
            原因：智能体在上一步中决定步态参数。
            '''
            self.player.behavior.execute("Step", self.step_default_dur, self.step_default_z_span, self.step_default_z_max)
        else:
            # 根据动作调整步态参数
            step_zsp = np.clip(self.step_default_z_span + self.act[20] / 300, 0, 0.07)  # Z 轴移动范围
            step_zmx = np.clip(self.step_default_z_max + self.act[21] / 30, 0.6, 0.9)  # Z 轴最大值

            self.player.behavior.execute("Step", self.step_default_dur, step_zsp, step_zmx)

        # 将动作作为步态行为的残差添加（由于排除了头部关节，动作索引与典型索引不同）
        new_action = self.act[:20] * 2  # 放大动作以促进探索
        new_action[[0, 2, 4, 6, 8, 10]] += self.step_obj.values_l  # 左腿目标位置
        new_action[[1, 3, 5, 7, 9, 11]] += self.step_obj.values_r  # 右腿目标位置
        new_action[12] -= 90  # 手臂下垂
        new_action[13] -= 90  # 手臂下垂
        new_action[16] += 90  # 手臂解开扭曲
        new_action[17] += 90  # 手臂解开扭曲
        new_action[18] += 90  # 肘部角度设置为 90 度
        new_action[19] += 90  # 肘部角度设置为 90 度

        # 设置关节目标位置
        r.set_joints_target_position_direct(
            slice(2, 22),  # 作用于除头部和脚趾外的所有关节（机器人类型 4）
            new_action,  # 目标关节位置
            harmonize=False  # 如果目标位置在每一步都变化，则无需谐波调整
        )

        self.sync()  # 运行模拟步骤
        self.step_counter += 1

        # 计算奖励：X 轴的位移（可以是负值）
        reward = r.cheat_abs_pos[0] - self.lastx
        self.lastx = r.cheat_abs_pos[0]

        # 终止条件：机器人跌倒或超时
        terminal = r.cheat_abs_pos[2] < 0.3 or self.step_counter > 300

        return self.observe(), reward, terminal, {}

class Train(Train_Base):
    def __init__(self, script) -> None:
        '''
        初始化训练类
        '''
        super().__init__(script)  # 调用父类初始化

    def train(self, args):
        '''
        训练模型
        参数：
        - args: 训练参数
        '''

        #--------------------------------------- 学习参数
        n_envs = min(16, os.cpu_count())  # 环境数量（最多 16 个，或根据 CPU 核心数）
        n_steps_per_env = 1024  # 每个环境的步数（RolloutBuffer 的大小为 n_steps_per_env * n_envs）
        minibatch_size = 64  # 小批量大小（应为 n_steps_per_env * n_envs 的因数）
        total_steps = 30000000  # 总训练步数
        learning_rate = 3e-4  # 学习率
        folder_name = f'Basic_Run_R{self.robot_type}'  # 日志文件夹名称
        model_path = f'./scripts/gyms/logs/{folder_name}/'  # 模型保存路径

        print("Model path:", model_path)  # 打印模型路径

        #--------------------------------------- 运行算法
        def init_env(i_env):
            '''
            初始化环境
            参数：
            - i_env: 环境索引
            '''
            def thunk():
                return Basic_Run(self.ip, self.server_p + i_env, self.monitor_p_1000 + i_env, self.robot_type, False)
            return thunk

        # 启动服务器（包括 1 个额外的服务器用于测试）
        servers = Server(self.server_p, self.monitor_p_1000, n_envs + 1)

        # 创建向量化环境
        env = SubprocVecEnv([init_env(i) for i in range(n_envs)])
        eval_env = SubprocVecEnv([init_env(n_envs)])  # 测试环境

        try:
            if "model_file" in args:  # 重新训练
                model = PPO.load(args["model_file"], env=env, device="cpu", n_envs=n_envs, n_steps=n_steps_per_env, batch_size=minibatch_size, learning_rate=learning_rate)
            else:  # 训练新模型
                model = PPO("MlpPolicy", env=env, verbose=1, n_steps=n_steps_per_env, batch_size=minibatch_size, learning_rate=learning_rate, device="cpu")

            # 开始训练
            model_path = self.learn_model(model, total_steps, model_path, eval_env=eval_env, eval_freq=n_steps_per_env * 20, save_freq=n_steps_per_env * 200, backup_env_file=__file__)
        except KeyboardInterrupt:
            sleep(1)  # 等待子进程
            print("\nctrl+c pressed, aborting...\n")  # 捕获 Ctrl+C，终止训练
            servers.kill()  # 关闭服务器
            return

        # 关闭环境和服务器
        env.close()
        eval_env.close()
        servers.kill()

    def test(self, args):
        '''
        测试模型
        参数：
        - args: 测试参数
        '''
        # 启动服务器
        server = Server(self.server_p - 1, self.monitor_p, 1)
        env = Basic_Run(self.ip, self.server_p - 1, self.monitor_p, self.robot_type, True)  # 创建测试环境
        model = PPO.load(args["model_file"], env=env)  # 加载模型

        try:
            # 导出模型权重
            self.export_model(args["model_file"], args["model_file"] + ".pkl", False)
            # 测试模型
            self.test_model(model, env, log_path=args["folder_dir"], model_path=args["folder_dir"])
        except KeyboardInterrupt:
            print()

        # 关闭环境和服务器
        env.close()
        server.kill()
'''
The learning process takes several hours.
A video with the results can be seen at:
https://imgur.com/a/dC2V6Et

Stats:
- Avg. reward:     7.7 
- Avg. ep. length: 5.5s (episode is limited to 6s)
- Max. reward:     9.3  (speed: 1.55m/s)    

State space:
- Composed of all joint positions + torso height
- Stage of the underlying Step behavior

Reward:
- Displacement in the x-axis (it can be negative)
- Note that cheat and visual data is only updated every 3 steps
'''
