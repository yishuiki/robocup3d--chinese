from agent.Base_Agent import Base_Agent as Agent
from pathlib import Path
from scripts.commons.Server import Server
from scripts.commons.Train_Base import Train_Base
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import SubprocVecEnv
from time import sleep
from world.commons.Draw import Draw
import gym
import numpy as np
import os
from behaviors.custom.Basic_Kick.Basic_Kick import Basic_Kick
'''
目标:
学习如何向指定方向踢球
----------
- class Basic_Kick_Gym: 实现OpenAI自定义训练环境
- class Train: 实现训练新模型或测试现有模型的算法
'''

class Basic_Kick_Gym(gym.Env):
    def __init__(self, ip, server_p, monitor_p, r_type, enable_draw) -> None:

        self.robot_type = r_type

        # Args: Server IP, Agent Port, Monitor Port, Uniform No., Robot Type, Team Name, Enable Log, Enable Draw
        self.player = Agent(ip, server_p, monitor_p, 1, self.robot_type, "Gym", True, enable_draw)
        self.step_counter = 0 # to limit episode size

        self.kick_obj : Basic_Kick = self.player.behavior.get_custom_behavior_object("Basic_Kick") # Basic_Kick behavior object

        # 获取机器人关节配置
        r = self.player.world.robot
        self.joint_indices = []
        
        # 根据不同机器人型号获取对应的关节
        if r_type == 1:  # Nao
            self.joint_indices = [
                r.J_LLEG_1, r.J_RLEG_1,
                r.J_LLEG_2, r.J_RLEG_2,
                r.J_LLEG_3, r.J_RLEG_3,
                r.J_LLEG_4, r.J_RLEG_4,
                r.J_LLEG_5, r.J_RLEG_5,
                r.J_LLEG_6, r.J_RLEG_6
            ]
        elif r_type == 3:  # RoboCup 2D
            self.joint_indices = [
                1, 2,  # 左右髋关节 (hip)
                3, 4,  # 左右膝关节 (knee)
                5, 6,  # 左右踝关节 (ankle)
                7, 8,  # 左右脚关节 (foot)
                9, 10, # 左右脚趾关节 (toe)
                11, 12 # 额外关节
            ]
        else:
            # 其他类型机器人的关节配置
            self.joint_indices = list(range(12))  # 默认使用前12个关节
            
        # 调整观测空间大小
        obs_size = 7 + len(self.joint_indices)  # 基础观测 + 关节角度
        self.obs = np.zeros(obs_size, np.float32)
        self.observation_space = gym.spaces.Box(
            low=np.full(obs_size, -np.inf, np.float32),
            high=np.full(obs_size, np.inf, np.float32),
            dtype=np.float32
        )

        # 动作空间定义 
        MAX = np.finfo(np.float32).max
        self.no_of_actions = act_size = 1  # 动作维度:踢球方向调整
        self.action_space = gym.spaces.Box(low=np.full(act_size,-MAX,np.float32), high=np.full(act_size,MAX,np.float32), dtype=np.float32)

        # 初始位置设置
        self.robot_init_pos = (-0.2, 0, self.player.world.robot.beam_height) 
        self.ball_init_pos = (0.15, -0.1, 0.042)  # 放在机器人右前方便于踢球
        self.target_position = (-15, 0, 0.042)    # 目标位置设在场地中心

        self.player.scom.unofficial_move_ball(self.ball_init_pos)


    def observe(self):
        """扩展观测空间
        包含:
        - 球的相对位置
        - 机器人状态
        - 关键关节角度
        - 与球的相对角度
        """
        r = self.player.world.robot
        b = self.player.world.ball_rel_torso_cart_pos

        # 基础观测 - 球的相对位置和机器人朝向
        self.obs[0] = b[0] / 2.0  # 球的x坐标
        self.obs[1] = b[1] / 1.0  # 球的y坐标
        self.obs[2] = b[2] / 0.5  # 球的z坐标
        
        # 机器人状态
        self.obs[3] = r.loc_torso_orientation / 180.0  # 机器人朝向
        self.obs[4] = r.gyro[1] / 100.0  # 机器人角速度
        self.obs[5] = r.acc[2] / 10.0   # 垂直加速度
        
        # 增加与球的相对角度
        self.obs[6] = np.arctan2(b[1], b[0]) / np.pi
        
        # 增加关键关节角度 (归一化到[-1,1])
        self.obs[7:] = r.joints_position[self.joint_indices] / 180.0
        
        return self.obs

    def sync(self):
        ''' Run a single simulation step '''
        r = self.player.world.robot
        self.player.scom.commit_and_send( r.get_command() )
        self.player.scom.receive()


    def reset(self):
        """重置环境到初始状态
        1. 重置计数器
        2. 将机器人和球移动到初始位置
        3. 稳定机器人姿态
        """
        self.step_counter = 0
        self.last_ball_pos = None  # 用于计算球的移动
        r = self.player.world.robot

        # Beam robot and ball to initial positions
        self.player.scom.unofficial_beam(self.robot_init_pos, 0)  # 0度朝向
        self.player.scom.unofficial_move_ball(self.ball_init_pos)

        # Stabilize the robot
        for _ in range(25):
            self.player.behavior.execute("Zero")
            self.sync()

        return self.observe()

    def render(self, mode='human', close=False):
        return

    def close(self):
        Draw.clear_all()
        self.player.terminate()

    def step(self, action):
        """执行一步环境交互
        参数:
        - action: 踢球方向调整量
        
        返回:
        - 观测值
        - 奖励值(负的到目标距离)
        - 是否结束
        - 额外信息
        """
        r = self.player.world.robot

        # Adjust kick direction based on action
        kick_direction_adjustment = action[0] * 10  # Scale up action
        kick_direction = 0 + kick_direction_adjustment # 0 is forward. Adjust as needed.

        # Execute the kick behavior
        reset = self.step_counter == 0
        self.kick_obj.execute(reset, kick_direction)

        self.sync() # run simulation step
        self.step_counter += 1
         
        # 计算奖励
        ball_pos = self.player.world.ball_abs_pos
        distance_to_target = np.linalg.norm(ball_pos[:2] - self.target_position[:2])
        
        # 1. 基础奖励：负的到目标距离
        reward = -distance_to_target / 15.0  # 归一化距离奖励
        
        # 2. 球移动奖励
        if self.last_ball_pos is not None:
            ball_movement = np.linalg.norm(ball_pos[:2] - self.last_ball_pos[:2])
            movement_direction = ball_pos[0] - self.last_ball_pos[0]  # 球的前向移动
            if movement_direction > 0:  # 如果球向前移动
                reward += ball_movement * 5.0  # 增加正向移动奖励
        
        self.last_ball_pos = ball_pos.copy()
        
        # 3. 接近球的奖励
        dist_to_ball = np.linalg.norm(self.player.world.ball_rel_torso_cart_pos[:2])
        approach_reward = -dist_to_ball if dist_to_ball > 0.3 else 0
        reward += approach_reward
        
        # 终止条件优化
        terminal = (distance_to_target < 0.5 or    # 球接近目标
                   self.step_counter > 300 or      # 超时
                   ball_pos[2] > 1.0 or           # 球飞得太高
                   dist_to_ball > 2.0)            # 机器人离球太远
                   
        return self.observe(), reward, terminal, {}





class Train(Train_Base):
    def __init__(self, script) -> None:
        super().__init__(script)


    def train(self, args):
        """训练入口函数
        
        训练参数:
        - n_envs: 并行环境数,默认4个
        - n_steps_per_env: 每个环境的步数
        - minibatch_size: 小批量大小
        - total_steps: 总训练步数 
        - learning_rate: 学习率
        """
        #--------------------------------------- Learning parameters
        n_envs = min(4, os.cpu_count())  # 减少并行环境数提高稳定性
        n_steps_per_env = 256  # 增加每个环境的步数
        minibatch_size = 64    # should be a factor of (n_steps_per_env * n_envs)
        total_steps = 1000000  # 增加总训练步数
        learning_rate = 3e-4
        folder_name = f'Basic_Kick_R{self.robot_type}'
        model_path = f'./scripts/gyms/logs/{folder_name}/'

        print("Model path:", model_path)

        #--------------------------------------- Run algorithm
        def init_env(i_env):
            def thunk():
                return Basic_Kick_Gym( self.ip , self.server_p + i_env, self.monitor_p_1000 + i_env, self.robot_type, False )
            return thunk

        servers = Server( self.server_p, self.monitor_p_1000, n_envs+1 ) #include 1 extra server for testing

        env = SubprocVecEnv( [init_env(i) for i in range(n_envs)] )
        eval_env = SubprocVecEnv( [init_env(n_envs)] )

        try:
            if "model_file" in args: # retrain
                model = PPO.load( args["model_file"], env=env, device="cpu", n_envs=n_envs, n_steps=n_steps_per_env, batch_size=minibatch_size, learning_rate=learning_rate )
            else: # train new model
                model = PPO( "MlpPolicy", env=env, verbose=1, n_steps=n_steps_per_env, batch_size=minibatch_size, learning_rate=learning_rate, device="cpu" )

            model_path = self.learn_model( model, total_steps, model_path, eval_env=eval_env, eval_freq=n_steps_per_env*10, save_freq=n_steps_per_env*100, backup_env_file=__file__ )
        except KeyboardInterrupt:
            sleep(1) # wait for child processes
            print("\nctrl+c pressed, aborting...\n")
            servers.kill()
            return
    
        env.close()
        eval_env.close()
        servers.kill()
        

    def test(self, args):
        """测试入口函数
        
        使用单个环境测试训练好的模型
        可以导出模型为pkl格式用于自定义行为
        """
        # Uses different server and monitor ports
        server = Server( self.server_p-1, self.monitor_p, 1 )
        env = Basic_Kick_Gym( self.ip, self.server_p-1, self.monitor_p, self.robot_type, True )
        model = PPO.load( args["model_file"], env=env )

        try:
            self.export_model( args["model_file"], args["model_file"]+".pkl", False )  # Export to pkl to create custom behavior
            self.test_model( model, env, log_path=args["folder_dir"], model_path=args["folder_dir"] )
        except KeyboardInterrupt:
            print()

        env.close()
        server.kill()
