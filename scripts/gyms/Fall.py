from agent.Base_Agent import Base_Agent as Agent  # 导入基础智能体类
from world.commons.Draw import Draw  # 导入绘图工具
from stable_baselines3 import PPO  # 导入 PPO 算法
from stable_baselines3.common.vec_env import SubprocVecEnv  # 导入多进程向量化环境
from scripts.commons.Server import Server  # 导入服务器管理类
from scripts.commons.Train_Base import Train_Base  # 导入基础训练类
from time import sleep  # 导入睡眠函数
import os, gym  # 导入操作系统模块和 Gym 库
import numpy as np  # 导入 NumPy 库

'''
目标：
学习如何跌倒（最简单的示例）
----------
- 类 Fall：实现一个自定义的 OpenAI Gym 环境
- 类 Train：实现用于训练新模型或测试现有模型的算法
'''

class Fall(gym.Env):
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
        self.player = Agent(ip, server_p, monitor_p, 1, self.robot_type, "Gym", True, enable_draw)
        self.step_counter = 0  # 用于限制 episode 的大小

        # 状态空间：关节位置 + 躯干高度
        self.no_of_joints = self.player.world.robot.no_of_joints
        self.obs = np.zeros(self.no_of_joints + 1, np.float32)  # 关节位置 + 躯干高度
        self.observation_space = gym.spaces.Box(
            low=np.full(len(self.obs), -np.inf, np.float32),
            high=np.full(len(self.obs), np.inf, np.float32),
            dtype=np.float32
        )

        # 动作空间：所有关节的目标位置
        MAX = np.finfo(np.float32).max
        no_of_actions = self.no_of_joints
        self.action_space = gym.spaces.Box(
            low=np.full(no_of_actions, -MAX, np.float32),
            high=np.full(no_of_actions, MAX, np.float32),
            dtype=np.float32
        )

        # 检查是否启用作弊模式（用于获取机器人的绝对位置）
        assert np.any(self.player.world.robot.cheat_abs_pos), "作弊模式未启用！请在 Run_Utils.py -> Server 中启用作弊模式"

    def observe(self):
        '''
        获取环境的观测值
        '''
        r = self.player.world.robot  # 获取机器人对象

        # 观测值：关节位置（归一化处理）+ 躯干高度
        for i in range(self.no_of_joints):
            self.obs[i] = r.joints_position[i] / 100  # 关节位置（简单归一化）
        self.obs[self.no_of_joints] = r.cheat_abs_pos[2]  # 躯干高度（Z 轴位置）

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
            self.player.scom.unofficial_beam((-3, 0, 0.50), 0)  # 持续传送机器人
            self.player.behavior.execute("Zero")  # 执行零关节行为
            self.sync()

        # 将机器人传送至地面
        self.player.scom.unofficial_beam((-3, 0, r.beam_height), 0)
        r.joints_target_speed[0] = 0.01  # 移动头部以触发物理更新（rcssserver3d 的一个 Bug）
        self.sync()

        # 在地面上稳定机器人
        for _ in range(7):
            self.player.behavior.execute("Zero")  # 执行零关节行为
            self.sync()

        return self.observe()

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
        - action: 动作数组
        '''
        r = self.player.world.robot
        r.set_joints_target_position_direct(  # 设置关节目标位置
            slice(self.no_of_joints),  # 作用于所有关节
            action * 10,  # 放大动作以促进早期探索
            harmonize=False  # 如果目标位置在每一步都变化，则无需谐波调整
        )

        self.sync()  # 运行模拟步骤
        self.step_counter += 1
        self.observe()

        # 终止条件和奖励机制
        if self.obs[-1] < 0.15:  # 如果机器人成功跌倒（Z 轴高度小于 0.15）
            return self.obs, 1, True, {}  # 奖励为 1，终止 episode
        elif self.step_counter > 150:  # 如果超过 150 步仍未跌倒
            return self.obs, 0, True, {}  # 奖励为 0，终止 episode
        else:
            return self.obs, 0, False, {}  # 奖励为 0，继续 episode

class Train(Train_Base):
    def __init__(self, script) -> None:
        '''
        初始化训练类
        参数：
        - script: 脚本对象，包含训练配置
        '''
        super().__init__(script)  # 调用父类初始化

    def train(self, args):
        '''
        训练模型
        参数：
        - args: 训练参数
        '''

        #--------------------------------------- 学习参数
        n_envs = min(4, os.cpu_count())  # 环境数量（最多4个，或根据CPU核心数）
        n_steps_per_env = 128  # 每个环境的步数（RolloutBuffer 的大小为 n_steps_per_env * n_envs）
        minibatch_size = 64  # 小批量大小（应为 n_steps_per_env * n_envs 的因数）
        total_steps = 50000  # 总训练步数
        learning_rate = 30e-4  # 学习率
        folder_name = f'Fall_R{self.robot_type}'  # 日志文件夹名称
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
                return Fall(self.ip, self.server_p + i_env, self.monitor_p_1000 + i_env, self.robot_type, False)
            return thunk

        servers = Server(self.server_p, self.monitor_p_1000, n_envs + 1)  # 启动服务器（包括1个额外的服务器用于测试）
        env = SubprocVecEnv([init_env(i) for i in range(n_envs)])  # 创建多进程向量化环境
        eval_env = SubprocVecEnv([init_env(n_envs)])  # 创建测试环境

        try:
            if "model_file" in args:  # 重新训练现有模型
                model = PPO.load(args["model_file"], env=env, n_envs=n_envs, n_steps=n_steps_per_env, batch_size=minibatch_size, learning_rate=learning_rate)
            else:  # 训练新模型
                model = PPO("MlpPolicy", env=env, verbose=1, n_steps=n_steps_per_env, batch_size=minibatch_size, learning_rate=learning_rate)

            model_path = self.learn_model(  # 开始训练
                model, 
                total_steps, 
                model_path, 
                eval_env=eval_env, 
                eval_freq=n_steps_per_env * 10, 
                save_freq=n_steps_per_env * 20, 
                backup_env_file=__file__
            )
        except KeyboardInterrupt:  # 捕获 Ctrl+C 中断
            sleep(1)  # 等待子进程
            print("\nctrl+c pressed, aborting...\n")  # 打印中断信息
            servers.kill()  # 关闭服务器
            return

        env.close()  # 关闭环境
        eval_env.close()  # 关闭测试环境
        servers.kill()  # 关闭服务器

    def test(self, args):
        '''
        测试模型
        参数：
        - args: 测试参数
        '''
        server = Server(self.server_p - 1, self.monitor_p, 1)  # 启动测试服务器
        env = Fall(self.ip, self.server_p - 1, self.monitor_p, self.robot_type, True)  # 创建测试环境
        model = PPO.load(args["model_file"], env=env)  # 加载模型

        try:
            self.export_model(args["model_file"], args["model_file"] + ".pkl", False)  # 导出模型为 pkl 文件
            self.test_model(  # 测试模型
                model, 
                env, 
                log_path=args["folder_dir"], 
                model_path=args["folder_dir"]
            )
        except KeyboardInterrupt:  # 捕获 Ctrl+C 中断
            print()

        env.close()  # 关闭环境
        server.kill()  # 关闭服务器