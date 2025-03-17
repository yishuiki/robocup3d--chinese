# 标准库
import os
import gym
import numpy as np
from time import sleep

# 第三方库
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.base_class import BaseAlgorithm

# 自定义模块
from agent.Base_Agent import Base_Agent as Agent
from scripts.commons.Server import Server
from scripts.commons.Train_Base import Train_Base
from world.commons.Draw import Draw
from pathlib import Path

class Kick_Ball(gym.Env):
    def __init__(self, ip, server_p, monitor_p, r_type, enable_draw):
        '''
        初始化踢球环境
        参数：
        - ip: 服务器 IP 地址
        - server_p: 服务器端口
        - monitor_p: 监控端口
        - r_type: 机器人类型
        - enable_draw: 是否启用绘图
        '''
        self.robot_type = r_type
        self.player = Agent(ip, server_p, monitor_p, 1, self.robot_type, "Gym", True, enable_draw)  # 初始化智能体
        self.step_counter = 0  # 用于限制 episode 的大小

        # 初始化球的位置（在机器人前方）
        self.player.scom.unofficial_move_ball((1.5, 0, 0.042))

        # 状态空间：包括球的位置、机器人的关节角度和速度等
        obs_size = 30  # 观测值大小
        self.obs = np.zeros(obs_size, np.float32)
        self.observation_space = gym.spaces.Box(
            low=np.full(obs_size, -np.inf, np.float32),
            high=np.full(obs_size, np.inf, np.float32),
            dtype=np.float32
        )

        # 动作空间：允许调整腿部关节角度
        MAX = np.finfo(np.float32).max
        self.no_of_actions = 10  # 10个动作（腿部关节角度调整）
        self.action_space = gym.spaces.Box(
            low=np.full(self.no_of_actions, -MAX, np.float32),
            high=np.full(self.no_of_actions, MAX, np.float32),
            dtype=np.float32
        )

    def observe(self):
        '''
        获取环境的观测值
        '''
        r = self.player.world.robot
        ball = self.player.world.ball

        # 观测值索引及其含义
        self.obs[0:3] = ball.pos  # 球的位置
        self.obs[3:6] = ball.vel  # 球的速度
        self.obs[6:16] = r.joints_position[2:12]  # 腿部关节角度
        self.obs[16:26] = r.joints_speed[2:12]  # 腿部关节速度

        return self.obs

    def step(self, action):
        '''
        执行一步动作
        参数：
        - action: 动作数组
        '''
        r = self.player.world.robot
        ball = self.player.world.ball

        # 应用动作到腿部关节
        r.set_joints_target_position_direct(slice(2, 12), action)

        self.player.scom.commit_and_send(r.get_command())  # 提交命令
        self.player.scom.receive()  # 接收数据

        self.step_counter += 1

        # 奖励机制
        reward = 0
        if ball.pos[0] > 5:  # 如果球的 X 轴位置大于 5 米
            reward += 10  # 给予正奖励
        if ball.pos[2] < 0.1:  # 如果球的 Z 轴位置过低
            reward -= 5  # 给予负奖励

        # 终止条件：机器人跌倒或球飞出边界
        terminal = r.cheat_abs_pos[2] < 0.3 or ball.pos[0] > 10 or self.step_counter > 200

        return self.observe(), reward, terminal, {}

    def reset(self):
        '''
        重置环境
        '''
        self.step_counter = 0
        self.player.scom.unofficial_move_ball((1.5, 0, 0.042))  # 将球重置到初始位置
        self.player.scom.commit_beam((-3, 0), 0)  # 将机器人重置到初始位置

        return self.observe()

    def render(self, mode='human', close=False):
        return

    def close(self):
        Draw.clear_all()
        self.player.scom.close()

class Train(Train_Base):
    def __init__(self, script):
        super().__init__(script)

    def train(self, args):
        '''
        训练踢球模型
        参数：
        - args: 训练参数
        '''
        n_envs = min(15, os.cpu_count())  # 环境数量
        n_steps_per_env = 72  # 每个环境的步数
        minibatch_size = 72  # 小批量大小
        total_steps = 1000000  # 总训练步数
        learning_rate = 3e-4  # 学习率
        folder_name = f'Kick_Ball_R{self.robot_type}'
        model_path = f'./scripts/gyms/logs/{folder_name}/'

        print("Model path:", model_path)

        def init_env(i_env):
            def thunk():
                return Kick_Ball(self.ip, self.server_p + i_env, self.monitor_p_1000 + i_env, self.robot_type, False)
            return thunk

        servers = Server(self.server_p, self.monitor_p_1000, n_envs + 1)
        env = SubprocVecEnv([init_env(i) for i in range(n_envs)])
        eval_env = SubprocVecEnv([init_env(n_envs)])

        try:
            if "model_file" in args:
                model = PPO.load(args["model_file"], env=env, n_envs=n_envs, n_steps=n_steps_per_env, batch_size=minibatch_size, learning_rate=learning_rate)
            else:
                model = PPO("MlpPolicy", env=env, verbose=1, n_steps=n_steps_per_env, batch_size=minibatch_size, learning_rate=learning_rate)

            model_path = self.learn_model(
                model, 
                total_steps, 
                model_path, 
                eval_env=eval_env, 
                eval_freq=n_steps_per_env * 10, 
                backup_env_file=__file__
            )
        except KeyboardInterrupt:
            sleep(1)
            print("\nctrl+c pressed, aborting...\n")
            servers.kill()
            return

        env.close()
        eval_env.close()
        servers.kill()

    def test(self, args):
        '''
        测试踢球模型
        参数：
        - args: 测试参数
        '''
        server = Server(self.server_p - 1, self.monitor_p, 1)
        env = Kick_Ball(self.ip, self.server_p - 1, self.monitor_p, self.robot_type, True)
        model = PPO.load(args["model_file"], env=env)

        self.test_model(
            model, 
            env, 
            log_path=args["folder_dir"], 
            model_path=args["folder_dir"]
        )

        env.close()
        server.kill()