from agent.Base_Agent import Base_Agent as Agent  # 导入基础智能体类
from pathlib import Path  # 导入路径处理模块
from scripts.commons.Server import Server  # 导入服务器管理类
from scripts.commons.Train_Base import Train_Base  # 导入基础训练类
from stable_baselines3 import PPO  # 导入 PPO 算法
from stable_baselines3.common.base_class import BaseAlgorithm  # 导入基础算法类
from stable_baselines3.common.vec_env import SubprocVecEnv  # 导入多进程向量化环境
from time import sleep  # 导入睡眠函数
from world.commons.Draw import Draw  # 导入绘图工具
import gym  # 导入 Gym 库
import numpy as np  # 导入 NumPy 库
import os  # 导入操作系统模块

'''
目标：
学习如何起身（4种变体，见第 157 行）
优化现有槽位行为的每个关键帧
----------
- 类 Get_Up：实现一个自定义的 OpenAI Gym 环境
- 类 Train：实现用于训练新模型或测试现有模型的算法
'''

class Get_Up(gym.Env):
    def __init__(self, ip, server_p, monitor_p, r_type, fall_direction, enable_draw) -> None:
        '''
        初始化自定义 Gym 环境
        参数：
        - ip: 服务器 IP 地址
        - server_p: 服务器端口
        - monitor_p: 监控端口
        - r_type: 机器人类型
        - fall_direction: 倒地方向（0：前方，1：左侧，2：右侧，3：后方）
        - enable_draw: 是否启用绘图
        '''
        self.robot_type = r_type  # 机器人类型
        self.fall_direction = fall_direction  # 倒地方向
        self.player = Agent(ip, server_p, monitor_p, 1, self.robot_type, "Gym", True, enable_draw, [])  # 初始化智能体
        self.get_up_names = {  # 起身行为的名称映射
            0: "Get_Up_Front", 
            1: "Get_Up_Side_Left", 
            2: "Get_Up_Side_Right", 
            3: "Get_Up_Back"
        }

        # 备份原始槽位
        self.original_slots = []
        for delta_ms, indices, angles in self.player.behavior.slot_engine.behaviors[self.get_up_names[self.fall_direction]]:
            self.original_slots.append((delta_ms, indices, np.array(angles)))

        # 观测空间：使用 one-hot 编码表示每个槽位
        self.obs = np.identity(len(self.original_slots))
        self.current_slot = 0  # 当前槽位索引

        MAX = np.finfo(np.float32).max  # 浮点数的最大值
        self.action_space = gym.spaces.Box(  # 动作空间：11 个动作（1 个时间步长调整 + 10 个关节角度调整）
            low=np.full(11, -MAX, np.float32), 
            high=np.full(11, MAX, np.float32), 
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(  # 观测空间：one-hot 编码
            low=np.zeros(len(self.obs), np.float32), 
            high=np.ones(len(self.obs), np.float32), 
            dtype=np.float32
        )

    def fall(self):
        '''
        使机器人倒下
        根据倒地方向设置关节目标位置
        '''
        r = self.player.world.robot  # 获取机器人对象
        joint_indices = [  # 关节索引
            r.J_LFOOT_PITCH, 
            r.J_RFOOT_PITCH, 
            r.J_LLEG_ROLL, 
            r.J_RLEG_ROLL
        ]

        if self.fall_direction == 0:  # 前方倒地
            r.set_joints_target_position_direct(joint_indices, np.array([50, 50, 0, 0]))
        elif self.fall_direction == 1:  # 左侧倒地
            r.set_joints_target_position_direct(joint_indices, np.array([0, 0, -20, 20]))
        elif self.fall_direction == 2:  # 右侧倒地
            r.set_joints_target_position_direct(joint_indices, np.array([0, 0, 20, -20]))
        elif self.fall_direction == 3:  # 后方倒地
            r.set_joints_target_position_direct(joint_indices, np.array([-20, -20, 0, 0]))
        else:
            raise ValueError("Invalid fall direction")

        self.player.scom.commit_and_send(r.get_command())  # 提交并发送命令
        self.player.scom.receive()  # 接收数据

    def get_up(self):
        '''
        执行起身行为
        返回行为是否完成
        '''
        r = self.player.world.robot  # 获取机器人对象
        finished = self.player.behavior.execute(self.get_up_names[self.fall_direction])  # 执行起身行为

        self.player.scom.commit_and_send(r.get_command())  # 提交并发送命令
        self.player.scom.receive()  # 接收数据
        return finished

    def other(self, behavior_name):
        '''
        执行其他行为
        参数：
        - behavior_name: 行为名称
        '''
        r = self.player.world.robot  # 获取机器人对象
        self.player.behavior.execute(behavior_name)  # 执行指定行为

        self.player.scom.commit_and_send(r.get_command())  # 提交并发送命令
        self.player.scom.receive()  # 接收数据

    def reset(self):
        '''
        重置环境
        使机器人倒下并初始化状态
        '''
        self.player.scom.commit_beam((-3, 0), 0)  # 将机器人传送至指定位置

        for _ in range(30):  # 重复执行倒地动作
            self.fall()
        while self.player.world.robot.cheat_abs_pos[2] > 0.32:  # 确保机器人倒在地上
            self.fall()

        import random
        t = random.randint(7, 17) if self.fall_direction == 0 else random.randint(10, 20)  # 随机等待时间
        for _ in range(t):  
            self.other("Zero")  # 执行零关节行为

        self.current_slot = 0  # 初始化当前槽位

        return self.obs[self.current_slot]  # 返回初始观测值
    
    def render(self, mode='human', close=False):
        '''
        渲染环境（此处为空实现）
        '''
        return
    
    def close(self):
        '''
        关闭环境
        清除绘图并关闭智能体连接
        '''
        Draw.clear_all()  # 清除所有绘图
        self.player.scom.close()  # 关闭智能体连接

    @staticmethod
    def scale_action(action: np.ndarray):
        '''
        缩放动作
        将动作扩展为对称动作
        '''
        new_action = np.zeros(len(action) * 2 - 1, action.dtype) 
        new_action[0] = action[0] * 10  # 时间步长调整
        new_action[1:] = np.repeat(action[1:] * 3, 2)  # 扩展对称动作

        return new_action

    @staticmethod
    def get_22_angles(angles, indices):
        '''
        获取 22 个关节角度
        参数：
        - angles: 角度数组
        - indices: 关节索引
        返回：
        - 包含 22 个关节角度的数组
        '''
        new_angles = np.zeros(22, np.float32)  # 初始化 22 个关节角度
        new_angles[indices] = angles  # 设置指定关节的角度

        return new_angles
    
    def step(self, action):
        '''
        执行一步动作
        参数：
        - action: 动作数组
        返回：
        - 观测值、奖励、是否终止、附加信息
        '''
        # 动作：1 个时间步长调整 + 10 个关节角度调整
        r = self.player.world.robot  # 获取机器人对象
        action = Get_Up.scale_action(action)  # 缩放动作

        delta, indices, angles = self.original_slots[self.current_slot]  # 获取当前槽位的原始参数
        angles = Get_Up.get_22_angles(angles, indices)  # 将原始角度扩展为22个关节角度

        angles[2:] += action[1:]  # 排除头部关节，将动作添加到关节角度
        new_delta = max((delta + action[0]) // 20 * 20, 20)  # 更新时间步长，确保为20的倍数且不小于20

        # 更新槽位行为
        self.player.behavior.slot_engine.behaviors[self.get_up_names[self.fall_direction]][self.current_slot] = (
            new_delta, slice(0, 22), angles
        )

        self.current_slot += 1  # 切换到下一个槽位
        terminal = bool(self.current_slot == len(self.obs))  # 判断是否完成所有槽位的调整
        reward = 0  # 初始化奖励

        if terminal:  # 如果完成所有槽位调整，执行起身行为并评估
            while not self.get_up():  # 执行起身行为直到完成
                reward -= 0.05  # 每次未完成起身给予负奖励

            for _ in range(50):  # 执行一些稳定行为
                self.other("Zero_Bent_Knees")
                reward += r.cheat_abs_pos[2] * 0.95 ** abs(r.gyro[1])  # 根据机器人高度和陀螺仪数据给予奖励

            print("rew:", reward)  # 打印奖励
            obs = self.obs[0]  # 返回一个虚拟观测值
        else:
            obs = self.obs[self.current_slot]  # 返回当前槽位的观测值

        return obs, reward, terminal, {}
    
class Train(Train_Base):
    def __init__(self, script) -> None:
        '''
        初始化训练类
        参数：
        - script: 脚本对象，包含训练配置
        '''
        super().__init__(script)  # 调用父类初始化
        self.fall_direction = 0  # 倒地方向（0：前方，1：左侧，2：右侧，3：后方）

    def train(self, args):
        '''
        训练模型
        参数：
        - args: 训练参数
        '''
        n_envs = min(15, os.cpu_count())  # 环境数量（最多15个，或根据CPU核心数）
        n_steps_per_env = 72  # 每个环境的步数
        minibatch_size = 72  # 小批量大小（应为n_steps_per_env * n_envs的因数）
        total_steps = 1000  # 总训练步数
        learning_rate = 2e-4  # 学习率
        folder_name = f'GetUp_R{self.robot_type}_Direction{self.fall_direction}'  # 日志文件夹名称
        model_path = f'./scripts/gyms/logs/{folder_name}/'  # 模型保存路径

        print("Model path:", model_path)  # 打印模型路径

        def init_env(i_env):
            '''
            初始化环境
            参数：
            - i_env: 环境索引
            '''
            def thunk():
                return Get_Up(self.ip, self.server_p + i_env, self.monitor_p_1000 + i_env, self.robot_type, self.fall_direction, False)
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
                backup_env_file=__file__
            )
        except KeyboardInterrupt:  # 捕获Ctrl+C中断
            sleep(1)  # 等待子进程
            print("\nctrl+c pressed, aborting...\n")
            servers.kill()  # 关闭服务器
            return

        # 生成起身行为的XML文件
        self.generate_get_up_behavior(model, model_path, eval_env.get_attr('original_slots')[0], "last_model.xml")

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
        env = Get_Up(self.ip, self.server_p - 1, self.monitor_p, self.robot_type, self.fall_direction, True)  # 创建测试环境
        model = PPO.load(args["model_file"], env=env)  # 加载模型

        # 生成起身行为的XML文件
        XML_name = Path(args["model_file"]).stem + ".xml"  # XML文件名
        if not os.path.isfile(os.path.join(args["folder_dir"], XML_name)):  # 如果文件不存在，则生成
            self.generate_get_up_behavior(model, args["folder_dir"], env.original_slots, XML_name)

        self.test_model(  # 测试模型
            model, 
            env, 
            log_path=args["folder_dir"], 
            model_path=args["folder_dir"]
        )

        env.close()  # 关闭环境
        server.kill()  # 关闭服务器

    def generate_get_up_behavior(self, model: BaseAlgorithm, folder_dir, original_slots, XML_name):
        '''
        生成起身行为的XML文件
        参数：
        - model: 训练好的模型
        - folder_dir: 保存路径
        - original_slots: 原始槽位数据
        - XML_name: XML文件名
        '''
        predictions = model.predict(np.identity(len(original_slots)), deterministic=True)[0]  # 预测动作
        slots = []

        for i in range(len(predictions)):
            pred = Get_Up.scale_action(predictions[i])  # 缩放动作
            delta = max((original_slots[i][0] + pred[0]) // 20 * 20, 20)  # 更新时间步长
            angles = Get_Up.get_22_angles(original_slots[i][2], original_slots[i][1])  # 获取22个关节角度
            angles[2:] += pred[1:]  # 更新关节角度
            slots.append((delta, range(22), angles))  # 保存调整后的槽位数据

        self.generate_slot_behavior(folder_dir, slots, False, XML_name)  # 生成XML文件

